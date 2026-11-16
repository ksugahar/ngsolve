# One maintenance round, source side: merge the latest upstream master into
# both forks' master, point ngsolve's netgen submodule at the merged netgen,
# and list the fork patches that upstream does not have yet.
#
# Stops at the first merge conflict; nothing is pushed or tagged here.
#
#   pwsh -File update.ps1 -Netgen <netgen fork checkout> -NGSolve <ngsolve fork checkout>
param(
  [Parameter(Mandatory)] [string] $Netgen,
  [Parameter(Mandatory)] [string] $NGSolve,
  [string] $NetgenUpstream = "https://github.com/NGSolve/netgen.git",
  [string] $NGSolveUpstream = "https://github.com/NGSolve/ngsolve.git",
  [string] $Branch = "master"
)
$ErrorActionPreference = "Stop"
function Git([string]$repo) { & git.exe -c safe.directory=* -C $repo @args; if ($LASTEXITCODE) { throw "git failed in ${repo}: $args" } }

function Merge-Upstream([string]$repo, [string]$url) {
  if (Git $repo status --porcelain) { throw "$repo has uncommitted or staged changes; commit or remove them first" }
  $remotes = @(Git $repo remote)
  if ($remotes -notcontains 'upstream') { Git $repo remote add upstream $url }
  elseif ((Git $repo remote get-url upstream) -ne $url) { throw "upstream remote of $repo is not $url" }
  if ((Git $repo rev-parse --is-shallow-repository) -eq 'true') { Git $repo fetch -q --unshallow upstream }
  Git $repo fetch -q upstream $Branch
  Git $repo fetch -q origin $Branch
  Git $repo checkout -q $Branch
  Git $repo merge -q --ff-only "origin/$Branch"
  $before = Git $repo rev-parse HEAD
  & git.exe -c safe.directory=* -C $repo merge --no-ff -q "upstream/$Branch" -m "Merge upstream $Branch"
  if ($LASTEXITCODE) { throw "merge conflict in $repo; resolve it, keep fork patches minimal, then rerun" }
  [ordered]@{ repo = $repo; before = $before; after = (Git $repo rev-parse HEAD)
              upstream = (Git $repo rev-parse "upstream/$Branch")
              fork_patches = @() }
}
function Patches([string]$repo) { @(Git $repo cherry -v "upstream/$Branch" HEAD | Where-Object { $_ -like '+*' }) }

# both checkouts must be clean before either merge
foreach ($r in $Netgen, $NGSolve) { if (Git $r status --porcelain) { throw "$r has uncommitted or staged changes" } }
foreach ($r in $Netgen, $NGSolve) { foreach ($k in 'user.name', 'user.email') {
  $v = & git.exe -C $r config --get $k; if (-not $v) { throw "$r has no git $k; set the committer identity first" } } }
$ng = Merge-Upstream $Netgen $NetgenUpstream
$ngs = Merge-Upstream $NGSolve $NGSolveUpstream

# ngsolve's submodule follows the merged netgen fork commit
$sub = 'external_dependencies/netgen'
$subUrl = Git $NGSolve config -f .gitmodules --get "submodule.$sub.url"
$forkUrl = Git $Netgen remote get-url origin
function Norm([string]$u) { ($u.TrimEnd("/") -replace "\.git$", "").ToLower() }
if ((Norm $subUrl) -ne (Norm $forkUrl)) { throw ".gitmodules URL $subUrl is not the netgen fork $forkUrl" }
$link = (Git $NGSolve ls-tree HEAD $sub) -split '\s+'
if ($link[0] -ne '160000') { throw "$sub is not a gitlink (mode $($link[0]))" }
if ($link[2] -ne $ng.after) {
  Git $NGSolve update-index --cacheinfo "160000,$($ng.after),$sub"
  $staged = @(Git $NGSolve diff --cached --name-only)
  if ($staged.Count -ne 1 -or $staged[0] -ne $sub) { throw "index holds more than the netgen gitlink: $staged" }
  Git $NGSolve commit -q -m "Point the netgen submodule at the merged netgen fork"
  $ngs.after = Git $NGSolve rev-parse HEAD
}
$ng.fork_patches = Patches $Netgen
$ngs.fork_patches = Patches $NGSolve
$out = [ordered]@{ netgen = $ng; ngsolve = $ngs; netgen_submodule = $ng.after }
$out | ConvertTo-Json -Depth 4
