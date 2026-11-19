# Following upstream

The forks carry a small series of Radia patches on top of upstream Netgen
and NGSolve. Each maintenance round moves the fork heads forward, lands any
pending patch series on the new heads, and requalifies the exact source
pair before anything is published. Netgen always goes first, because the
NGSolve tree pins Netgen through the `external_dependencies/netgen` gitlink.

## 1. Move the fork heads

```
pwsh -File tools/radia_fork/update.ps1 -Netgen <netgen checkout> -NGSolve <ngsolve checkout>
```

`update.ps1` merges upstream `master` into both forks, points the NGSolve
gitlink at the merged Netgen commit, and lists the fork patches upstream
does not have. It stops at the first conflict and pushes nothing.

## 2. Land pending series on the new heads

A series qualified on an older pair (for example a platform build route)
is replayed, not merged, so that each commit keeps its patch and message:

```
python tools/radia_fork/replay.py netgen  <netgen repo>  <old base> <old tip> <new base>
python tools/radia_fork/replay.py ngsolve <ngsolve repo> <old base> <old tip> <new base> \
    --map <old netgen sha>=<new netgen sha> ...
```

The `netgen` step prints the old -> new commit map; pass every entry that
an NGSolve commit references to `--map`. Each replayed NGSolve commit then
points at the Netgen image of its original gitlink. The tool only creates
commits; name them with `git branch` and set your committer identity first.
A conflict stops the replay with exit code 2: resolve it as a new, reviewed
commit rather than editing the series.

When fetching an older Netgen history into another clone, use
`git fetch --no-recurse-submodules`; intermediate commits may pin pybind11
revisions that are no longer fetchable, while the tips need only the
current one.

## 3. Check the replay independently

```
python tools/radia_fork/replay.py check \
    --netgen  <repo> <old base> <old tip> <new base> <new tip> \
    --ngsolve <repo> <old base> <old tip> <new base> <new tip>
```

`check` does not trust how the replay was made. For every commit it
compares order, parent chain, message, author and stable patch-id (the
Netgen gitlink excluded), and requires each NGSolve gitlink to be the image
of the original one and reachable from the new Netgen tip. A second person
or agent runs it on their own clone before review.

## 4. Requalify the exact pair

Build and validate from the two candidate SHAs, never from a moving branch:

- Windows CPU and CUDA wheels, and the LP64/ILP64 MATLAB SDKs with their
  consumer and acceptance checks (`build_wheels.ps1`, `build_sdk.ps1`,
  `check_sdk.ps1`).
- Flow2 Linux wheels, Type II (Grace/GB200) and Type I (x86_64), as in
  `tools/flow2/README.md`.

Earlier receipts stay as baselines; a replayed pair is qualified only by
its own runs.

## 5. Publish in source order

After review and owner approval: push Netgen, then NGSolve (fast-forward of
the fork heads), then binaries that name those exact SHAs. Downstream
packages move their pin only after the binaries for that pair exist.
