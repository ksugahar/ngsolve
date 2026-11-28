# Build the netgen-mesher and ngsolve pybind11 wheels of the forks for one
# Python version on Windows (MKL single dynamic library from the mkl wheel;
# -Cuda adds ngscuda for sm_80+ with the CUDA runtime from exact-version pip
# packages). Nothing is uploaded: wheels go to <Out>\wheelhouse, with receipt.json.
#
#   pwsh -File build_wheels.ps1 -Netgen <netgen checkout> -NGSolve <ngsolve checkout> `
#        -Job <scratch dir> -Out <evidence dir> [-Python <python.exe>] [-Cores 8] [-Cuda [-Matlab <MATLAB root>]]
param(
  [Parameter(Mandatory)] [string] $Netgen,
  [Parameter(Mandatory)] [string] $NGSolve,
  [Parameter(Mandatory)] [string] $Job,
  [Parameter(Mandatory)] [string] $Out,
  [string] $Python = "C:\Program Files\Python312\python.exe",
  [int] $Cores = 8,
  [string] $MklVersion = "2026.2.1",
  [switch] $Cuda,
  [string] $Matlab = "C:\Program Files\MATLAB\R2026a",   # nvcc is MATLAB's bundled CUDA 12.8
  [string] $CudaRuntime = "12.8.57",
  [string] $CudaNvrtc = "12.8.61",
  [string] $Cudss = "0.8.0.10",           # headers only at build time; loaded at run time
  [string] $CudssWheelSha256 = "89bf57d4d05d25c7f6e1288acbc94c5eed519763c55d7ec1f6d73099ace8554e",
  [string] $Cublas = "12.8.3.14",         # cuDSS's runtime dependency
  [string] $VcVars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
  [string] $MsvcVersion = "14.44"
)
$ErrorActionPreference = "Stop"
# checkouts on a network share belong to another account: trust exactly these two
# worktrees for this process (git matches safe.directory against the worktree path)
$Netgen = (Resolve-Path $Netgen).Path; $NGSolve = (Resolve-Path $NGSolve).Path
$env:GIT_CONFIG_COUNT = "2"
$env:GIT_CONFIG_KEY_0 = "safe.directory"; $env:GIT_CONFIG_VALUE_0 = $Netgen.Replace('\','/')
$env:GIT_CONFIG_KEY_1 = "safe.directory"; $env:GIT_CONFIG_VALUE_1 = $NGSolve.Replace('\','/')
if ((Test-Path $Job) -and (Get-ChildItem -Force $Job | Where-Object Name -ne 'owner.json')) { throw "$Job is not empty; use a fresh job directory" }
$Wh = "$Out\wheelhouse"
if ((Test-Path $Wh) -and (Get-ChildItem -Force $Wh)) { throw "$Wh is not empty; use a fresh output directory" }
New-Item -ItemType Directory -Force -Path $Job, $Out | Out-Null
function G([string]$repo) { $o = & git.exe -C $repo @args; if ($LASTEXITCODE) { throw "git failed in ${repo}: $args" }; $o }
$receipt = [ordered]@{ lane = $(if ($Cuda) { "pybind11-cuda-mkl" } else { "pybind11-cpu-mkl" }); mkl = $MklVersion }
foreach ($r in @{netgen = $Netgen; ngsolve = $NGSolve}.GetEnumerator()) {
  $receipt[$r.Key] = [ordered]@{ head = (G $r.Value rev-parse HEAD); clean = -not (G $r.Value status --porcelain --ignore-submodules=none) }
  if (-not $receipt[$r.Key].clean) { throw "$($r.Key) checkout is not clean" } }
$gitlink = ((G $NGSolve ls-tree HEAD external_dependencies/netgen) -split '\s+')[2]
if ($receipt.netgen.head -ne $gitlink) { throw "Netgen HEAD $($receipt.netgen.head) is not the NGSolve gitlink $gitlink" }
Remove-Item "$Out\receipt.json" -ErrorAction SilentlyContinue

function Step([string]$n, [scriptblock]$b) {
  Write-Host "=== $n $(Get-Date -Format o)"; & $b
  if ($LASTEXITCODE -ne 0) { throw "step failed ($LASTEXITCODE): $n" }
}
cmd /c "`"$VcVars`" -vcvars_ver=$MsvcVersion >nul && set" | ForEach-Object {
  if ($_ -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2]) } }
$env:CL = "/utf-8"
$env:CMAKE_BUILD_PARALLEL_LEVEL = "$Cores"
$env:NETGEN_ARCH = "avx2"
$env:CMAKE_GENERATOR = "Ninja"
# a build tree registered in the user package registry must not leak into this build
$reg = 'HKCU:\Software\Kitware\CMake\Packages'
if ((Test-Path $reg) -and (Get-ChildItem $reg)) { throw "user CMake package registry is not empty: $((Get-ChildItem $reg).PSChildName -join ', ')" }

Step "venv" { & $Python -m venv "$Job\venv" }
$py = "$Job\venv\Scripts\python.exe"
$env:PATH = "$Job\venv\Scripts;$env:PATH"
# the build wheels (netgen-occt-devel, mkl-devel) install their CMake packages into the venv
$env:CMAKE_PREFIX_PATH = "$Job/venv".Replace('\','/')
Step "tools" { & $py -m pip install -q build delvewheel "scikit-build-core>=0.11" pybind11-stubgen cmake==3.31.10 ninja==1.13.2 numpy "mkl==$MklVersion" "mkl-devel==$MklVersion" }
$Dist = "$Job\dist"
New-Item -ItemType Directory -Force -Path $Dist, $Wh | Out-Null

# netgen-mesher: upstream configuration without the Tcl/Tk GUI
Push-Location $Netgen
try {
  Step "netgen requires" { & $py tests\utils.py --install-build-requires }
  Step "netgen wheel" { & $py -m build --wheel --no-isolation --outdir $Dist "-Cbuild-dir=$Job/b-netgen" "-Ccmake.define.USE_GUI=OFF" "-Ccmake.define.USE_CCACHE=OFF" . }
} finally { Pop-Location }
$ngWheel = @(Get-ChildItem "$Dist\netgen_mesher_radia-*.whl")
if ($ngWheel.Count -ne 1) { throw "expected one netgen wheel in $Dist, found $($ngWheel.Count)" }
Step "netgen repair" { & $py -m delvewheel repair --exclude "TK*.dll" --exclude "nglib.dll" --exclude "ngcore.dll" -w $Wh $ngWheel[0].FullName }
Step "netgen install" { & $py -m pip install --no-deps (Get-ChildItem "$Wh\netgen_mesher_radia-*.whl").FullName }

# ngsolve: MKL single dynamic library, loaded from the mkl wheel at import
$mklMajor = [int]$MklVersion.Split(".")[0]
$env:NGSOLVE_BLAS_REQUIREMENT = "mkl>=$mklMajor,<$($mklMajor + 1)"   # a range, so Radia's own MKL pin can coexist
$mklRoot = "$Job/venv/Library".Replace('\','/')
$cudaArgs = @("-Ccmake.define.USE_CUDA=OFF")
$cudaExclude = @()
$env:NGSOLVE_EXTRA_REQUIREMENTS = ""   # never inherited from the caller
if ($Cuda) {
  $cudaRoot = (Join-Path $Matlab "sys/cuda/win64/cuda").Replace('\','/')
  $cudaArgs = @("-Ccmake.define.USE_CUDA=ON", "-Ccmake.define.CUDAToolkit_ROOT=$cudaRoot", "-Ccmake.define.CMAKE_CUDA_COMPILER=$cudaRoot/bin/nvcc.exe", "-Ccmake.define.CMAKE_CUDA_ARCHITECTURES=80")
  $cudaExclude = @("--exclude", "cudart64_12.dll", "--exclude", "nvrtc64_120_0.dll", "--exclude", "nvcuda.dll",
                   "--exclude", "cudss64_0.dll", "--exclude", "cublas64_12.dll", "--exclude", "cublasLt64_12.dll")
  # cuDSS headers from the exact pinned wheel (hash-checked); nothing of it is linked or bundled
  $cd = "$Job\cudss"; New-Item -ItemType Directory -Force $cd | Out-Null
  Step "cudss wheel" { & $py -m pip download -q --no-deps --only-binary=:all: --platform win_amd64 "nvidia-cudss-cu12==$Cudss" -d $cd }
  $cw = @(Get-ChildItem "$cd\nvidia_cudss_cu12-$Cudss-*.whl")
  if ($cw.Count -ne 1) { throw "expected one cuDSS wheel, found $($cw.Count)" }
  if ((Get-FileHash $cw[0].FullName -Algorithm SHA256).Hash.ToLower() -ne $CudssWheelSha256) { throw "cuDSS wheel hash differs from the pin" }
  Expand-Archive $cw[0].FullName "$cd\x"
  $cudssInc = "$cd/x/nvidia/cu12/include".Replace('\','/')
  $cudaArgs += @("-Ccmake.define.USE_CUDSS=ON", "-Ccmake.define.CUDSS_INCLUDE_DIR=$cudssInc")
  $env:NGSOLVE_EXTRA_REQUIREMENTS = "nvidia-cuda-runtime-cu12==$CudaRuntime;;nvidia-cuda-nvrtc-cu12==$CudaNvrtc;;nvidia-cudss-cu12==$Cudss;;nvidia-cublas-cu12==$Cublas"
  $receipt.cuda = [ordered]@{ nvcc = $cudaRoot; arch = "sm_80"; runtime = $CudaRuntime; nvrtc = $CudaNvrtc; cudss = $Cudss; cudss_wheel_sha256 = $CudssWheelSha256; cublas = $Cublas }
}
Push-Location $NGSolve
try {
  Step "ngsolve wheel" { & $py -m build --wheel --no-isolation --outdir $Dist "-Cbuild-dir=$Job/b-ngsolve" "-Ccmake.define.USE_CCACHE=OFF" "-Ccmake.define.USE_MKL=ON" "-Ccmake.define.MKL_ROOT=$mklRoot" "-Ccmake.define.Python3_EXECUTABLE=$($py.Replace('\','/'))" @cudaArgs . }
} finally { Pop-Location }
$ngsWheel = @(Get-ChildItem "$Dist\ngsolve_radia-*.whl")
if ($ngsWheel.Count -ne 1) { throw "expected one ngsolve_radia wheel in $Dist, found $($ngsWheel.Count)" }
Step "ngsolve repair" { & $py -m delvewheel repair --ignore-existing --namespace-pkg "netgen;netgen.libs;netgen.include" --exclude "nglib.dll" --exclude "ngcore.dll" --exclude "TK*.dll" --exclude "mkl_*.dll" --exclude "libiomp5md.dll" @cudaExclude -w $Wh $ngsWheel.FullName }

# no NVIDIA library may be bundled into a repaired wheel
Add-Type -AssemblyName System.IO.Compression.FileSystem
foreach ($w in Get-ChildItem "$Wh\*.whl") {
  $z = [IO.Compression.ZipFile]::OpenRead($w.FullName)
  try { $bad = @($z.Entries | Where-Object { $_.Name -match '^(lib)?(cudss|cublas|cudart|nvrtc|nvcuda)[0-9A-Za-z_.-]*\.(dll|so)(\.[0-9.]+)?$' }) } finally { $z.Dispose() }
  if ($bad.Count) { throw "$($w.Name) bundles NVIDIA libraries: $($bad.Name -join ', ')" } }
$receipt.wheels = @(Get-ChildItem "$Wh\*.whl" | ForEach-Object { [ordered]@{ name = $_.Name; sha256 = (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower() } })
$receipt | ConvertTo-Json -Depth 4 | Set-Content "$Out\receipt.json"
Write-Host "=== WHEELS_DONE $(Get-Date -Format o)"
