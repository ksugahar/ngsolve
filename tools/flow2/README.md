Flow2 Linux wheel route
=======================

Flow2 is the primary Linux environment for the NGSolve and Netgen forks. Two
lanes build the `netgen-mesher` and `ngsolve` wheels from one exact commit pair
with `tools/radia_fork/build_wheels.sh`, install those SHA-256-verified wheels
into a clean environment and check them. The route follows the radia-optuna
Flow2 route; its distribution rules are repeated below.

- Type II (Grace + GB200, aarch64): CUDA `sm_100` with cuDSS, built on
  `t2-dev`, validated on a full GB200 on `t2-share`.
- Type I (x86_64): CPU wheels on `t1-single`, reference agreement and timings
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
| OpenCASCADE | 8.0.1 built static into `netgen-mesher` (no `netgen-occt` wheel exists for aarch64); the wheel declares no `netgen-occt` | `netgen-occt==8.0.1` wheel, as on Windows |
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
   same system, and agreement across thread counts. Only the true residual
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

Flow2 users install wheels from the lab's shared distribution folder, never
from a job root or a build tree. The folder location is lab configuration and
is not recorded in this repository. The rules are those of radia-optuna.

1. Eligibility. Only a commit pair that is merged, with Type II and Type I
   both complete at that pair: every PBS job ID recorded, every run complete
   with exit code 0, both `result.json` summaries passing, and the wheels,
   logs, result JSON and artifacts recovered to private evidence storage with
   SHA-256 matching at both ends. Placing files needs the release owner's
   explicit approval.
2. Content. One folder per package pair and source commit, holding only the
   exact wheels each validation job installed (`validated-wheels.json`),
   `SHA256SUMS`, the matching `requirements-validate-*.txt`, PBS examples for
   user jobs (queue, resources, job-local venv, `pip install --no-index
   --require-hashes` then the wheels, and on aarch64 the NVPL library path),
   and a README with the source SHAs, PBS job IDs, Python 3.12 and each
   wheel's profile. Only the latest verified release is kept. Wheels are never
   rebuilt for distribution.
3. Acceptance. From a separate, non-administrator account, run
   `sha256sum -c SHA256SUMS`, install into a fresh venv on each node type and
   run each PBS example to completion before announcing the release.
4. Distribution files are not committed to a source tree.
