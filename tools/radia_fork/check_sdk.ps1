# Check a packed MEX SDK zip: relocation, path leaks, Python freedom, the
# find_package consumer (real and complex ILP64 LAPACK), and for a CUDA SDK
# the ngscudalib exports/imports plus a linked exe and mexw64.
#
# Nothing here creates a CUDA context: GPU execution belongs on a GPU host.
#
#   pwsh -File check_sdk.ps1 -Zip <sdk zip> -Job <scratch dir> -Out <evidence dir> -Matlab <MATLAB root> [-Cuda]
param(
  [Parameter(Mandatory)] [string] $Zip,
  [Parameter(Mandatory)] [string] $Job,
  [Parameter(Mandatory)] [string] $Out,
  [Parameter(Mandatory)] [string] $Matlab,
  [switch] $Cuda,
  [string] $VcVars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
  [string] $MsvcVersion = "14.44",
  [string[]] $LeakPatterns = @('C:/temp', 'MATLAB/R20', 'maintainer-checkouts')
)
$ErrorActionPreference = "Stop"
$MwLib = (Join-Path $Matlab "extern/lib/win64/microsoft").Replace('\','/')
$MwBin = Join-Path $Matlab "bin\win64"
$CudaRoot = (Join-Path $Matlab "sys/cuda/win64/cuda").Replace('\','/')
$Consumer = Join-Path $PSScriptRoot "consumer"
cmd /c "`"$VcVars`" -vcvars_ver=$MsvcVersion >nul && set" | ForEach-Object {
  if ($_ -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2]) } }
$env:CL = "/utf-8"
if (Test-Path "$Job\venv\Scripts") { $env:PATH = "$Job\venv\Scripts;$env:PATH" }
$dumpbin = (Get-Command dumpbin.exe).Source
function Dump([string]$mode, [string]$f) { $o = & $dumpbin $mode $f; if ($LASTEXITCODE) { throw "dumpbin $mode failed on $f" }; $o }
function Deps([string]$f) { @(Dump '-dependents' $f | Select-String '\.dll' | ForEach-Object { $_.Line.Trim() } | Where-Object { $_ -notmatch '^Dump of' }) }

$res = [ordered]@{ zip = (Split-Path $Zip -Leaf); zip_sha256 = (Get-FileHash $Zip -Algorithm SHA256).Hash.ToLower(); cuda = [bool]$Cuda; failures = @() }
$Rel = Join-Path $Job ("reloc-" + [guid]::NewGuid().ToString("N").Substring(0, 8))  # fresh directory owned by this run
Expand-Archive $Zip "$Rel\sdk"; $RelC = "$Rel\sdk".Replace('\','/')

# 1. installed config carries no build, source or MATLAB paths
$res.path_leaks = @(Select-String -Path "$Rel\sdk\cmake\*.cmake", "$Rel\sdk\bin\*.bat" -Pattern ($LeakPatterns -join '|') -ErrorAction SilentlyContinue | ForEach-Object { $_.ToString() })
if ($res.path_leaks.Count) { $res.failures += "path_leaks" }
# 2. no SDK DLL imports Python
$res.python_imports = @(Get-ChildItem "$Rel\sdk\bin\*.dll" | ForEach-Object { $f = $_; Deps $f.FullName | Where-Object { $_ -match '^python' } | ForEach-Object { "$($f.Name): $_" } })
if ($res.python_imports.Count) { $res.failures += "python_imports" }
# 3. superbuild zlib is not exported
$res.zlib_config = @(Select-String -Path "$Rel\sdk\cmake\NetgenConfig.cmake" -Pattern 'NETGEN_ZLIB_(INCLUDE_DIRS|LIBRARIES)' | ForEach-Object { $_.Line.Trim() })
foreach ($v in 'NETGEN_ZLIB_INCLUDE_DIRS', 'NETGEN_ZLIB_LIBRARIES') {   # exactly one declaration of each, and it is empty
  if (@($res.zlib_config | Where-Object { $_ -match "^set\($v\b" }).Count -ne 1 -or
      @($res.zlib_config | Where-Object { $_ -ceq "set($v `"`")" }).Count -ne 1) { $res.failures += "zlib_exported_$v" } }
# flavor: a CPU archive carries no ngscudalib, a CUDA archive must
if ((Test-Path "$Rel\sdk\bin\ngscudalib.dll") -ne [bool]$Cuda) { $res.failures += "flavor_mismatch" }

try {   # an aborted step is recorded as a failure in check.json
# 4. find_package consumer: build, then run the CPU LAPACK check
$cudaArgs = if ($Cuda) { @("-DCHECK_CUDA=ON", "-DCUDAToolkit_ROOT=$CudaRoot", "-DMatlab_ROOT_DIR=$($Matlab.Replace('\','/'))") } else { @("-DCHECK_CUDA=OFF") }
& cmake -S $Consumer -B "$Rel\b" -G Ninja -DCMAKE_BUILD_TYPE=Release "-DNGSolve_DIR=$RelC/cmake" "-DCMAKE_EXE_LINKER_FLAGS=/LIBPATH:`"$MwLib`"" "-DCMAKE_SHARED_LINKER_FLAGS=/LIBPATH:`"$MwLib`"" @cudaArgs 2>&1 | Set-Content "$Out\consumer-configure.log"
if ($LASTEXITCODE) { throw "consumer configure failed, see consumer-configure.log" }
& cmake --build "$Rel\b" -- -v 2>&1 | Set-Content "$Out\consumer-build.log"
if ($LASTEXITCODE) { throw "consumer build failed, see consumer-build.log" }
$res.lapack_link = @(Select-String -Path "$Out\consumer-build.log" -Pattern 'libmw(lapack|blas)\.lib' -AllMatches | ForEach-Object { $_.Matches.Value } | Sort-Object -Unique)
if ($res.lapack_link.Count -ne 2) { $res.failures += "lapack_not_from_target" }
$env:PATH = "$Rel\sdk\bin;$MwBin;$env:PATH"
$run = & "$Rel\b\lapack_check.exe"; if ($LASTEXITCODE) { throw "lapack_check failed" }
$res.cpu_lapack = @($run | ForEach-Object { $_ | ConvertFrom-Json })
$want = @('real/7', 'real/64', 'real/300', 'complex/7', 'complex/64', 'complex/300')
$got = @($res.cpu_lapack | ForEach-Object { "$($_.kind)/$($_.n)" })
$bad = @($res.cpu_lapack | Where-Object { $e = $_.max_abs_AinvA_minus_I; -not ($e -is [double] -or $e -is [decimal] -or $e -is [int]) -or -not [double]::IsFinite([double]$e) -or [double]$e -lt 0 -or [double]$e -gt 1e-12 })
if ($bad.Count -or (Compare-Object $want $got) -or $got.Count -ne 6) { $res.failures += "cpu_lapack" }

# 5. CUDA: exports and imports of ngscudalib, linked exe and mexw64 (not run)
if ($Cuda) {
  $lib = "$Rel\sdk\bin\ngscudalib.dll"
  $exports = (Dump '-exports' $lib | Out-String)
  $res.ngscuda_entry_points = @('InitCUDA', 'InitCudaDevice', 'InitCuLinalg', 'CreateDevMatrix', 'BufferDevPtr') | Where-Object { $exports -match $_ }
  if ($res.ngscuda_entry_points.Count -ne 5) { $res.failures += "ngscuda_exports" }
  $res.ngscudalib_imports = Deps $lib
  foreach ($need in 'cudart64_12.dll', 'nvrtc64_120_0.dll', 'nvcuda.dll', 'libngsolve.dll') {
    if ($res.ngscudalib_imports -notcontains $need) { $res.failures += "ngscudalib_missing_$need" } }
  $res.mex_imports = Deps "$Rel\b\ngs_cuda_mex.mexw64"
  if ($res.mex_imports -notcontains 'ngscudalib.dll') { $res.failures += "mex_not_linked_to_ngscudalib" }
  $res.gpu_run = "not run here"
}
} catch { $res.failures += "aborted: $($_.Exception.Message)" }
$res.pass = ($res.failures.Count -eq 0)
$res | ConvertTo-Json -Depth 5 | Set-Content "$Out\check.json"
Remove-Item -Recurse -Force $Rel
Write-Host "=== CHECK pass=$($res.pass) failures=$($res.failures -join ',')"
if (-not $res.pass) { exit 1 }
