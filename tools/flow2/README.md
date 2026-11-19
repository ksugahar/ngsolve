Flow2 Linux wheel route
=======================

Flow2 is the primary Linux environment for the NGSolve and Netgen forks. Two
lanes build the `netgen-mesher` and `ngsolve` wheels from one exact commit pair
with `tools/radia_fork/build_wheels.sh`, install those SHA-256-verified wheels
into a clean environment and check them. The route follows the radia-optuna
Flow2 route; its distribution rules are repeated below.

- Type II (Grace + GB200, aarch64): CUDA `sm_100` with cuDSS, built on
  `t2-dev`, validated on a full GB200 on `t2-share`.
- Type I (x86_64): CPU wheels on `t1-single`, recorded reference differences and timings
  at 1, 16 and 48 threads.

Flow2 is outside GitHub Actions and has no runner. Wheels are qualification
artifacts of the job; only the distribution described below hands them to
Flow2 users. They are never uploaded to PyPI.

Fixed inputs
------------

| Item | Type II (aarch64) | Type I (x86_64) |
|---|---|---|
| Source | NGSolve commit (40 hex) and its Netgen gitlink; clean trees | same |
| Version | local tags `v6.2.2608+radia.<netgen 8>` and `v6.2.2608+radia.<ngsolve 9>.cuda` | `...<ngsolve 9>.cpu` |
| Python | system `python3.12` | system `python3.12` |
| Build tools | `requirements-build-aarch64.txt`, hash-pinned | `requirements-build-x86_64.txt` |
| Runtime for checks | `requirements-validate-aarch64.txt` | `requirements-validate-x86_64.txt` |
| OpenCASCADE | 8.0.1 built static by Netgen's own superbuild (`BUILD_OCC=ON`) and linked into `netgen-mesher` (no `netgen-occt` wheel exists for aarch64); the wheel declares no `netgen-occt` | `netgen-occt==8.0.1` wheel, as on Windows |
| BLAS/LAPACK | site NVPL (`NVPL_LIB_FLOW2`, sequential LP64), linked, neither bundled nor declared | `mkl` wheel (`mkl>=2026,<2027` declared) |
| CUDA | facility CUDA 12.8, `sm_100`; runtime from the pinned `nvidia-*-cu12` wheels | none |
| GPU check | one visible GPU, name contains `GB200`, compute capability `10.0` | none |

Both Linux wheels build NGSolve directly against the installed Netgen wheel.
They carry no UMFPACK (the NGSolve superbuild would fetch SuiteSparse on an
offline node); requesting it raises. Override paths only with
`CUDA_HOME_FLOW2`, `NVPL_LIB_FLOW2`, `CUDA_ARCH_FLOW2`, `EXPECT_GPU_NAME`
and `EXPECT_GPU_CC`. A missing tool or a different GPU fails the job.

Steps
-----

1. On the login node, stage a new job root per lane. The root must not exist.

   ```sh
   tools/flow2/stage.sh NGSOLVE_SRC NETGEN_SRC SHA "$HOME/<jobs>/<root2>" "<owner>" "<request>" 12 aarch64
   tools/flow2/stage.sh NGSOLVE_SRC NETGEN_SRC SHA "$HOME/<jobs>/<root1>" "<owner>" "<request>" 12 x86_64
   ```

   This clones both commits, tags the versions, fetches the zlib (and, on
   aarch64, OpenCASCADE) archives with their MD5 from Netgen's superbuild,
   downloads the hash-pinned wheels and writes `owner.json`,
   `source-heads.txt` and `inputs.sha256`. Compute jobs need no network.

2. Type II: chain `t2-dev` links on one build run, then validate on GB200.
   `t2-dev` gives 4 CPUs, one hour and one job per user; each link resumes the
   same build trees, and links after the seal exit at once. The static
   OpenCASCADE takes several links.

   ```sh
   t=$ROOT2/source/tools/flow2
   prev=$(qsub -q t2-dev -l gpu=1 -v JOB_ROOT=$ROOT2,RUN=build-1 "$t/build.pbs")
   for i in 2 3 4 5 6 7 8; do prev=$(qsub -q t2-dev -l gpu=1 -W depend=afterany:$prev -v JOB_ROOT=$ROOT2,RUN=build-1 "$t/build.pbs"); done
   qsub -q t2-share -l gpu=1 -W depend=afterany:$prev -v JOB_ROOT=$ROOT2,RUN=validate-1,BUILD=build-1 "$t/validate.pbs"
   ```

   `validate.pbs` refuses an unsealed build, re-hashes the wheels, records
   `validated-wheels.json`, installs the pinned runtime and the two wheels
   (`--no-deps`), runs `pip check`, checks that both packages import from the
   venv, builds an OpenCASCADE box, checks the GPU identity, runs a device SpMV
   against the CPU product, and requires exactly 18 cuDSS and 6 BFV pytest
   passes with no skip.

3. Type I: one whole-node job builds and validates.

   ```sh
   qsub -q t1-single -l node=1 -v JOB_ROOT=$ROOT1 "$ROOT1/source/tools/flow2/type1.pbs"
   ```

   `validate_cpu.sh` installs the verified wheels into a clean venv and runs
   `type1_bench.py`: the same P2 problem at 1, 16 and 48 threads pinned to
   physical cores of one NUMA node with memory bound to it, an analytic
   manufactured solution (convergence rate reported), SciPy SuperLU on the
   same system, and differences across thread counts. The SuperLU and
   thread differences are recorded under `gates.reference_agreement` but have
   no approved tolerance, so they are not gated and no agreement is claimed.
   Only the true residual
   limit 1e-6 gates; timings never gate.

4. Summarize, then recover:

   ```sh
   python3.12 $t/summarize.py report $ROOT --build build-1 --validate validate-1
   ```

   `result.json` lists every PBS job ID and fails when a run is not
   `complete` with exit code 0, the build is not sealed with exactly two
   wheels matching `wheel.sha256`, the receipt names other commits, the
   validated identity differs from the sealed build, or an evidence hash
   differs. Copy `result.json` and the run directories (without `venv/`,
   `src/`, `job/`) to private evidence storage, verify the SHA-256 values,
   then delete the job root and confirm it is gone. Do not edit a job root
   after a failure; stage a new one.

Distribution to Flow2 users
---------------------------

The lab distribution is the primary route for Flow2 users, for example
students running their own jobs under their own accounts. Users install from
the lab's shared distribution folder, never from a job root or a build tree.
The folder location is lab configuration and is not recorded in this
repository. Attaching the same verified files to a GitHub release is optional.

Items 1 to 5 follow the general Flow2 reference of radia-optuna. "Fork
settings" at the end lists what is specific to the Netgen and NGSolve wheels.

1. Eligibility. Only a release whose exact source commit is merged is
   distributed. Every qualification lane must be complete at that exact
   commit. Evidence from an earlier commit, including a pre-merge candidate,
   does not qualify a later merged commit, even with an equal tree; the lanes
   run again at the merged commit. Completion means:
   - every PBS job ID is recorded, and every run is `complete` with exit code 0;
   - every `result.json` summary passes;
   - the wheels, logs, result JSON and artifacts are recovered to private
     evidence storage, with SHA-256 matching at both ends.
   A missing or mismatched hash, a failed gate or an unrecorded job means no
   distribution. Placing files also needs the release owner's explicit approval.
2. Content. One folder per package pair and source commit, holding only:
   - each project wheel, as the exact file its validation job installed
     (`validated-wheels.json`); wheels are never rebuilt for distribution;
   - `wheelhouse/`: the hash-pinned runtime dependency wheels for every
     distributed architecture. These are the files the qualification jobs
     installed from their staged wheelhouses, checked against the pinned
     requirement files;
   - the pinned runtime requirement file of each architecture;
   - PBS examples for user jobs;
   - a short README with the source SHAs, PBS job IDs, Python version and the
     profile of each wheel;
   - `SHA256SUMS`, covering every other file in the folder (payload files,
     `wheelhouse/` included) and not itself.
   The distribution folder keeps only the latest verified release. A
   superseded release is removed after its replacement is verified.
3. Installation. A PBS example creates a job-local venv and installs offline
   from the folder only (`<machine>` is `aarch64` or `x86_64`):

   ```sh
   (cd "$DIST" && sha256sum -c SHA256SUMS)
   python3.12 -m venv venv
   venv/bin/python -m pip install --no-index --require-hashes      --find-links "$DIST/wheelhouse" -r "$DIST/requirements-validate-<machine>.txt"
   venv/bin/python -m pip install --no-index --no-deps      "$DIST/<netgen-mesher wheel for this node>" "$DIST/<ngsolve wheel for this node>"
   ```

   The runtime dependencies resolve first from the hash-pinned wheelhouse.
   The verified project wheels are then installed by their exact paths with
   `--no-deps`; their hashes were checked by `SHA256SUMS`.
4. Acceptance before announcement. From a separate user account, without
   Administrator paths, fetch the folder and run each PBS example to
   completion on each node type: manifest check, fresh venv, installation,
   and one run of the package. Record the job IDs and outputs with the release
   evidence. Until then the release is not announced.
5. Version control. Distribution files are not committed to a source tree.
   When the folder sits inside a checkout, it must be a path that Git ignores.

Fork settings (specific to the Netgen and NGSolve wheels):

- Lanes: Type II (aarch64, CUDA profile): one build run chained over `t2-dev`
  links, GB200 validation on `t2-share`. Type I (x86_64, CPU profile): one
  `t1-single` job (`-l node=1`) that builds, validates and times 1/16/48
  threads. Distributed wheels: the `netgen-mesher` and `ngsolve` pair of each
  lane, as recorded in that lane's `validated-wheels.json`.
- Runtime: CPython 3.12. `requirements-validate-aarch64.txt` and
  `requirements-validate-x86_64.txt` pin the runtime with one hash per
  platform wheel. `wheelhouse/` holds the union of the two staged
  `wheelhouse-validate` directories. The aarch64 PBS example also puts the
  site NVPL library directory on `LD_LIBRARY_PATH`; NVPL is neither bundled
  nor declared.
- Qualification: the gates in this README. The residual gate decides Type I;
  differences from SciPy SuperLU and across thread counts are reported as
  measured values, without an agreement claim, until a tolerance is approved.
  Timings are diagnostic and never gate.
- PyPI: Linux wheels are never uploaded.
