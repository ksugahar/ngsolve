# Build the Python-free MEX SDK (CPU, optionally CUDA) of the netgen and
# ngsolve forks on Windows, then pack it as a relocatable zip.
#
# BLAS/LAPACK are MATLAB's ILP64 libmwblas/libmwlapack, given by name; the
# consumer supplies MATLAB's extern/lib directory. No Python is used.
#
#   pwsh -File build_sdk.ps1 -Netgen <netgen checkout> -NGSolve <ngsolve checkout> `
#        -Job <scratch dir> -Out <evidence dir> -Matlab <MATLAB root> [-Cuda] [-Cores 8]
param(
  [Parameter(Mandatory)] [string] $Netgen,
  [Parameter(Mandatory)] [string] $NGSolve,
  [Parameter(Mandatory)] [string] $Job,
  [Parameter(Mandatory)] [string] $Out,
  [Parameter(Mandatory)] [string] $Matlab,
  [switch] $Cuda,
  [int] $Cores = 8,
  [string] $Python = "python",
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
$MwLib = (Join-Path $Matlab "extern/lib/win64/microsoft").Replace('\','/')
$CudaRoot = (Join-Path $Matlab "sys/cuda/win64/cuda").Replace('\','/')
$Name = if ($Cuda) { "ngsolve-sdk-cuda-win64" } else { "ngsolve-sdk-cpu-win64" }
# a fresh job root: CMake install never removes stale files and the pack takes the whole prefix
if ((Test-Path $Job) -and (Get-ChildItem -Force $Job | Where-Object Name -ne 'owner.json')) { throw "$Job is not empty; use a fresh job directory" }
New-Item -ItemType Directory -Force -Path $Job, $Out | Out-Null
function G([string]$repo) { $o = & git.exe -C $repo @args; if ($LASTEXITCODE) { throw "git failed in ${repo}: $args" }; $o }
$receipt = [ordered]@{ cuda = [bool]$Cuda }
foreach ($r in @{netgen = $Netgen; ngsolve = $NGSolve}.GetEnumerator()) {
  $receipt[$r.Key] = [ordered]@{ head = (G $r.Value rev-parse HEAD); clean = -not (G $r.Value status --porcelain --ignore-submodules=none) } }
# the Netgen checkout must be the commit NGSolve's submodule records
$gitlink = ((G $NGSolve ls-tree HEAD external_dependencies/netgen) -split '\s+')[2]
if ($receipt.netgen.head -ne $gitlink) { throw "Netgen HEAD $($receipt.netgen.head) is not the NGSolve gitlink $gitlink" }
$receipt.netgen_gitlink = $gitlink
Remove-Item "$Out\receipt.json" -ErrorAction SilentlyContinue   # rewritten only after a successful pack

function Step([string]$n, [scriptblock]$b) {
  Write-Host "=== $n $(Get-Date -Format o)"; & $b
  if ($LASTEXITCODE -ne 0) { throw "step failed ($LASTEXITCODE): $n" }
}

cmd /c "`"$VcVars`" -vcvars_ver=$MsvcVersion >nul && set" | ForEach-Object {
  if ($_ -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2]) } }
$env:CL = "/utf-8"                       # sources are UTF-8 on any code page
$env:CMAKE_BUILD_PARALLEL_LEVEL = "$Cores"

Step "venv" { & $Python -m venv "$Job\venv" }
Step "tools" { & "$Job\venv\Scripts\python.exe" -m pip install -q cmake==3.31.10 ninja==1.13.2 }
$env:PATH = "$Job\venv\Scripts;$env:PATH"
$Inst = "$Job\sdk".Replace('\','/')
Step "describe" { git -C $Netgen describe --tags --long --dirty; git -C $NGSolve describe --tags --long --dirty }
$common = @("-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DUSE_CCACHE=OFF", "-DCMAKE_INSTALL_PREFIX=$Inst")
Step "netgen configure" { cmake -S $Netgen -B "$Job\b-netgen" @common -DUSE_SUPERBUILD=ON -DBUILD_ZLIB=ON -DUSE_PYTHON=OFF -DUSE_GUI=OFF -DUSE_OCC=OFF -DUSE_NATIVE_ARCH=OFF -DENABLE_UNIT_TESTS=OFF }
Step "netgen build" { cmake --build "$Job\b-netgen" --target install }
$cudaArgs = if ($Cuda) { @("-DUSE_CUDA=ON", "-DCUDAToolkit_ROOT=$CudaRoot") } else { @("-DUSE_CUDA=OFF") }
Step "ngsolve configure" { cmake -S $NGSolve -B "$Job\b-ngsolve" @common -DUSE_SUPERBUILD=OFF "-DNetgen_DIR=$Inst/cmake" -DUSE_MKL=OFF -DUSE_LAPACK=ON "-DLAPACK_LIBRARIES=libmwlapack;libmwblas" -DLAPACK_ILP64=ON "-DCMAKE_SHARED_LINKER_FLAGS=/LIBPATH:`"$MwLib`"" "-DCMAKE_EXE_LINKER_FLAGS=/LIBPATH:`"$MwLib`"" -DUSE_UMFPACK=OFF -DBUILD_STUB_FILES=OFF @cudaArgs }
Step "ngsolve build" { cmake --build "$Job\b-ngsolve" --target install -- -k 0 }
Copy-Item "$Job\b-netgen\CMakeCache.txt" "$Out\CMakeCache-netgen.txt"
Copy-Item "$Job\b-ngsolve\CMakeCache.txt" "$Out\CMakeCache-ngsolve.txt"
Step "pack" { Compress-Archive -Path "$Job\sdk\*" -DestinationPath "$Out\$Name.zip" -Force }
$receipt.zip = "$Name.zip"; $receipt.zip_sha256 = (Get-FileHash "$Out\$Name.zip" -Algorithm SHA256).Hash.ToLower()
$receipt | ConvertTo-Json -Depth 3 | Set-Content "$Out\receipt.json"
"{0}  {1}" -f (Get-FileHash "$Out\$Name.zip" -Algorithm SHA256).Hash.ToLower(), "$Name.zip" | Set-Content "$Out\SHA256SUMS"
Write-Host "=== BUILD_DONE $(Get-Date -Format o)"
