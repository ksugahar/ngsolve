#!/usr/bin/env bash
# Linux counterpart of build_wheels.ps1: build the netgen-mesher and ngsolve
# pybind11 wheels of the forks for one Python, CPU, without uploading.
# BLAS/LAPACK: MKL single dynamic library on x86_64 (mkl wheel), NVPL on
# aarch64 (nvpl-blas/nvpl-lapack wheels, LP64, OpenMP).
#
#   bash build_wheels.sh <netgen checkout> <ngsolve checkout> <job dir> <out dir> [python] [cores]
set -euo pipefail
NETGEN=$(realpath "$1"); NGSOLVE=$(realpath "$2"); JOB=$3; OUT=$4
PYTHON=${5:-python3.12}; CORES=${6:-8}
MKL_VERSION=${MKL_VERSION:-2026.2.1}
ARCH=$(uname -m)

if [ -d "$JOB" ] && [ -n "$(ls -A "$JOB" | grep -v '^owner.json$' || true)" ]; then
  echo "$JOB is not empty; use a fresh job directory" >&2; exit 1; fi
if [ -d "$OUT/wheelhouse" ] && [ -n "$(ls -A "$OUT/wheelhouse")" ]; then
  echo "$OUT/wheelhouse is not empty; use a fresh output directory" >&2; exit 1; fi
mkdir -p "$JOB" "$OUT"; JOB=$(realpath "$JOB"); OUT=$(realpath "$OUT")
for r in "$NETGEN" "$NGSOLVE"; do
  [ -z "$(git -C "$r" status --porcelain --ignore-submodules=none)" ] || { echo "$r is not clean" >&2; exit 1; }; done
NG_HEAD=$(git -C "$NETGEN" rev-parse HEAD); NGS_HEAD=$(git -C "$NGSOLVE" rev-parse HEAD)
GITLINK=$(git -C "$NGSOLVE" ls-tree HEAD external_dependencies/netgen | awk '{print $3}')
[ "$NG_HEAD" = "$GITLINK" ] || { echo "Netgen HEAD $NG_HEAD is not the NGSolve gitlink $GITLINK" >&2; exit 1; }
rm -f "$OUT/receipt.json"
step() { echo "=== $1 $(date -Iseconds)"; }

export NGSOLVE_EXTRA_REQUIREMENTS=""   # never inherited from the caller
export CMAKE_BUILD_PARALLEL_LEVEL=$CORES CMAKE_GENERATOR=Ninja NETGEN_ARCH=${NETGEN_ARCH:-}
step venv; "$PYTHON" -m venv "$JOB/venv"; PY=$JOB/venv/bin/python
export PATH="$JOB/venv/bin:$PATH" CMAKE_PREFIX_PATH="$JOB/venv"
step tools
"$PY" -m pip install -q build auditwheel patchelf "scikit-build-core>=0.11" pybind11-stubgen cmake==3.31.10 ninja==1.13.2 numpy
DIST=$JOB/dist; WH=$OUT/wheelhouse; mkdir -p "$DIST" "$WH"

step "netgen wheel"
(cd "$NETGEN" && "$PY" tests/utils.py --install-build-requires && \
 "$PY" -m build --wheel --no-isolation --outdir "$DIST" -Cbuild-dir="$JOB/b-netgen" \
   -Ccmake.define.USE_GUI=OFF -Ccmake.define.USE_CCACHE=OFF -Ccmake.define.USE_MPI=OFF .)
step "netgen repair"
"$PY" -m auditwheel repair --exclude 'libTK*' --exclude 'libngcore.so' --exclude 'libnglib.so' -w "$WH" "$DIST"/netgen_mesher-*.whl
"$PY" -m pip install --no-deps "$WH"/netgen_mesher-*.whl

case "$ARCH" in
  x86_64)
    "$PY" -m pip install -q "mkl==$MKL_VERSION" "mkl-devel==$MKL_VERSION"
    MKL_MAJOR=${MKL_VERSION%%.*}; export NGSOLVE_BLAS_REQUIREMENT="mkl>=$MKL_MAJOR,<$((MKL_MAJOR + 1))"
    BLAS_ARGS=(-Ccmake.define.USE_MKL=ON -Ccmake.define.MKL_ROOT="$JOB/venv")
    EXCLUDE=(--exclude 'libmkl_*' --exclude 'libiomp5.so') ;;
  aarch64)
    # the superbuild resets LAPACK_LIBRARIES, so NVPL needs a direct ngsolve build
    "$PY" -m pip install -q nvpl-blas nvpl-lapack
    NVPL_LIB=$("$PY" -c "import nvpl, os; print(os.path.join(os.path.dirname(nvpl.__path__[0]), 'nvpl', 'lib'))")
    # one PEP 508 requirement per variable: a ";" inside one would start an environment marker
    NVPL_BLAS=$("$PY" -c "from importlib.metadata import version; print(version('nvpl-blas'))")
    NVPL_LAPACK=$("$PY" -c "from importlib.metadata import version; print(version('nvpl-lapack'))")
    export NGSOLVE_BLAS_REQUIREMENT="nvpl-blas==$NVPL_BLAS" NGSOLVE_EXTRA_REQUIREMENTS="nvpl-lapack==$NVPL_LAPACK"
    NG_CMAKE=$("$PY" -c "import netgen, os; print(os.path.join(os.path.dirname(netgen.__file__), 'cmake'))")
    BLAS_ARGS=(-Ccmake.define.USE_SUPERBUILD=OFF -Ccmake.define.Netgen_DIR="$NG_CMAKE" -Ccmake.define.USE_MKL=OFF
      -Ccmake.define.USE_LAPACK=ON "-Ccmake.define.LAPACK_LIBRARIES=$NVPL_LIB/libnvpl_lapack_lp64_gomp.so.0;$NVPL_LIB/libnvpl_blas_lp64_gomp.so.0"
      -Ccmake.define.USE_UMFPACK=OFF)
    EXCLUDE=(--exclude 'libnvpl_*') ;;
  *) echo "unsupported architecture $ARCH" >&2; exit 1 ;;
esac

step "ngsolve wheel"
(cd "$NGSOLVE" && "$PY" -m build --wheel --no-isolation --outdir "$DIST" -Cbuild-dir="$JOB/b-ngsolve" \
   -Ccmake.define.USE_CCACHE=OFF -Ccmake.define.USE_CUDA=OFF -Ccmake.define.Python3_EXECUTABLE="$PY" "${BLAS_ARGS[@]}" .)
step "ngsolve repair"
"$PY" -m auditwheel repair --exclude 'libTK*' --exclude 'libngcore.so' --exclude 'libnglib.so' "${EXCLUDE[@]}" -w "$WH" "$DIST"/ngsolve-*.whl

"$PY" - "$OUT" "$NG_HEAD" "$NGS_HEAD" "$ARCH" <<'EOF'
import hashlib, json, pathlib, sys
out, ng, ngs, arch = sys.argv[1:]
wh = sorted(pathlib.Path(out, "wheelhouse").glob("*.whl"))
json.dump({"lane": f"pybind11-cpu-{arch}", "netgen": {"head": ng}, "ngsolve": {"head": ngs},
           "wheels": [{"name": w.name, "sha256": hashlib.sha256(w.read_bytes()).hexdigest()} for w in wh]},
          open(pathlib.Path(out, "receipt.json"), "w"), indent=2)
EOF
step WHEELS_DONE
