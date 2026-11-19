#!/usr/bin/env bash
# Linux counterpart of build_wheels.ps1: build the netgen-mesher and ngsolve
# pybind11 wheels of the forks for one Python, without uploading.
# BLAS/LAPACK: MKL single dynamic library on x86_64 (mkl wheel), NVPL on
# aarch64 (nvpl-blas/nvpl-lapack wheels, LP64, OpenMP).
#
#   bash build_wheels.sh <netgen checkout> <ngsolve checkout> <job dir> <out dir> [python] [cores]
#
# Optional environment (all fail loudly when inconsistent):
#   WHEELHOUSE, REQUIREMENTS  offline: install every build tool from the hash-pinned
#                             REQUIREMENTS file out of WHEELHOUSE (no network, no
#                             per-step pip installs); both or neither
#   OCC_SOURCE                OpenCASCADE source archive: build it statically into
#                             <job>/occ (resumable) and link it into netgen-mesher,
#                             which then declares no netgen-occt requirement
#   CUDA_ROOT, CUDA_ARCH      add ngscuda (nvcc from CUDA_ROOT, one architecture)
#   CUDSS_INCLUDE             cuDSS headers (from the pinned wheel); with CUDA only
#   CUDA_REQUIREMENTS         ";;"-separated runtime pins for the CUDA wheel
set -euo pipefail
NETGEN=$(realpath "$1"); NGSOLVE=$(realpath "$2"); JOB=$3; OUT=$4
PYTHON=${5:-python3.12}; CORES=${6:-8}
MKL_VERSION=${MKL_VERSION:-2026.2.1}
ARCH=$(uname -m)
if [ -n "${WHEELHOUSE:-}" ] && [ -z "${REQUIREMENTS:-}" ] || [ -z "${WHEELHOUSE:-}" ] && [ -n "${REQUIREMENTS:-}" ]; then
  echo "WHEELHOUSE and REQUIREMENTS go together" >&2; exit 1; fi
if [ -n "${CUDSS_INCLUDE:-}${CUDA_REQUIREMENTS:-}${CUDA_ARCH:-}" ] && [ -z "${CUDA_ROOT:-}" ]; then
  echo "CUDA_ARCH, CUDSS_INCLUDE and CUDA_REQUIREMENTS need CUDA_ROOT" >&2; exit 1; fi
if [ -n "${CUDA_ROOT:-}" ] && [ -z "${CUDA_ARCH:-}" ]; then echo "CUDA_ROOT needs CUDA_ARCH" >&2; exit 1; fi

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

export NGSOLVE_EXTRA_REQUIREMENTS="" NETGEN_BUNDLED_OCC=0   # never inherited from the caller
export CMAKE_BUILD_PARALLEL_LEVEL=$CORES CMAKE_GENERATOR=Ninja NETGEN_ARCH=${NETGEN_ARCH:-}
# the venv is reused by a resumed build (a chained job continues the same build dirs)
[ -x "$JOB/venv/bin/python" ] || { step venv; "$PYTHON" -m venv "$JOB/venv"; }
PY=$JOB/venv/bin/python
export PATH="$JOB/venv/bin:$PATH" CMAKE_PREFIX_PATH="$JOB/venv"
pipi() { "$PY" -m pip install -q --disable-pip-version-check "$@"; }
step tools
if [ -n "${WHEELHOUSE:-}" ]; then
  pipi --no-index --require-hashes --find-links "$WHEELHOUSE" -r "$REQUIREMENTS"
else
  pipi build auditwheel patchelf "scikit-build-core>=0.11" pybind11-stubgen cmake==3.31.10 ninja==1.13.2 numpy
fi
"$PY" -m pip freeze > "$OUT/build-freeze.txt"
DIST=$JOB/dist; WH=$OUT/wheelhouse; mkdir -p "$DIST" "$WH"; rm -f "$DIST"/*.whl

# pin the interpreter and its headers: a node may carry other Python versions' headers
PY_INC=$("$PY" -c "import sysconfig; print(sysconfig.get_path('include'))")
[ -f "$PY_INC/Python.h" ] || { echo "no Python.h in $PY_INC" >&2; exit 1; }
PY_ARGS=(-Ccmake.define.Python3_EXECUTABLE="$PY" -Ccmake.define.Python3_INCLUDE_DIR="$PY_INC"
  -Ccmake.define.Python3_FIND_UNVERSIONED_NAMES=NEVER)
NG_ARGS=(-Ccmake.define.USE_GUI=OFF -Ccmake.define.USE_CCACHE=OFF -Ccmake.define.USE_MPI=OFF "${PY_ARGS[@]}")
if [ -n "${OCC_SOURCE:-}" ]; then
  # static OpenCASCADE with the module set of Netgen's own BUILD_OCC superbuild
  step "occ"
  mkdir -p "$JOB/occ-src"
  [ -f "$JOB/occ-src/.unpacked" ] || { rm -rf "$JOB/occ-src"/*; "$PY" -m zipfile -e "$OCC_SOURCE" "$JOB/occ-src"; touch "$JOB/occ-src/.unpacked"; }
  OCC_SRC=$(find "$JOB/occ-src" -mindepth 1 -maxdepth 1 -type d | head -1)
  cmake -S "$OCC_SRC" -B "$JOB/b-occ" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$JOB/occ" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_LIBRARY_TYPE:STRING=Static \
    -DBUILD_MODULE_FoundationClasses:BOOL=ON -DBUILD_MODULE_ModelingData:BOOL=ON \
    -DBUILD_MODULE_ModelingAlgorithms:BOOL=ON -DBUILD_MODULE_DataExchange:BOOL=ON \
    -DBUILD_MODULE_Visualization:BOOL=OFF -DBUILD_MODULE_ApplicationFramework:BOOL=OFF \
    -DBUILD_MODULE_Draw:BOOL=OFF -DBUILD_MODULE_DETools:BOOL=OFF -DUSE_FREETYPE:BOOL=OFF \
    -DUSE_OPENGL:BOOL=OFF -DUSE_XLIB:BOOL=OFF -DBUILD_DOC_Overview:BOOL=OFF
  cmake --build "$JOB/b-occ" --parallel "$CORES"
  cmake --install "$JOB/b-occ" > /dev/null
  OCC_CMAKE=$(dirname "$(find "$JOB/occ" -name 'OpenCASCADEConfig.cmake' | head -1)")
  [ -d "$OCC_CMAKE" ] || { echo "static OpenCASCADE has no CMake config" >&2; exit 1; }
  ! find "$JOB/occ" -name 'libTK*.so*' | grep -q . || { echo "OpenCASCADE built shared libraries" >&2; exit 1; }
  export NETGEN_BUNDLED_OCC=1
  NG_ARGS+=(-Ccmake.define.BUILD_OCC=OFF -Ccmake.define.OpenCascade_DIR="$OCC_CMAKE")
fi

step "netgen wheel"
if [ -z "${WHEELHOUSE:-}" ] && [ "$NETGEN_BUNDLED_OCC" = 0 ]; then (cd "$NETGEN" && "$PY" tests/utils.py --install-build-requires); fi
(cd "$NETGEN" && "$PY" -m build --wheel --no-isolation --outdir "$DIST" -Cbuild-dir="$JOB/b-netgen" "${NG_ARGS[@]}" .)
step "netgen repair"
"$PY" -m auditwheel repair --exclude 'libTK*' --exclude 'libngcore.so' --exclude 'libnglib.so' -w "$WH" "$DIST"/netgen_mesher-*.whl
"$PY" -m pip install -q --no-deps "$WH"/netgen_mesher-*.whl

# direct ngsolve builds against the installed netgen-mesher: the ngsolve superbuild
# would fetch SuiteSparse at build time (compute nodes are offline), and it resets
# LAPACK_LIBRARIES. Linux wheels therefore carry no UMFPACK; requesting it raises.
NG_CMAKE=$("$PY" -c "import netgen, os; print(os.path.join(os.path.dirname(netgen.__file__), 'cmake'))")
DIRECT=(-Ccmake.define.USE_SUPERBUILD=OFF -Ccmake.define.Netgen_DIR="$NG_CMAKE" -Ccmake.define.USE_UMFPACK=OFF)
case "$ARCH" in
  x86_64)
    [ -n "${WHEELHOUSE:-}" ] || pipi "mkl==$MKL_VERSION" "mkl-devel==$MKL_VERSION"
    MKL_VERSION=$("$PY" -c "from importlib.metadata import version; print(version('mkl'))")
    MKL_MAJOR=${MKL_VERSION%%.*}; export NGSOLVE_BLAS_REQUIREMENT="mkl>=$MKL_MAJOR,<$((MKL_MAJOR + 1))"
    BLAS_ARGS=("${DIRECT[@]}" -Ccmake.define.USE_MKL=ON -Ccmake.define.MKL_ROOT="$JOB/venv")
    EXCLUDE=(--exclude 'libmkl_*' --exclude 'libiomp5.so') ;;
  aarch64)
    # NVPL has no PyPI wheel: the site NVPL (NVPL_LIB) is linked, not bundled and not
    # declared; the wheel needs it on LD_LIBRARY_PATH at run time. Sequential LP64,
    # as in the qualified source build.
    : "${NVPL_LIB:?aarch64 needs NVPL_LIB, the site NVPL lib directory}"
    for l in libnvpl_lapack_lp64_seq.so libnvpl_blas_lp64_seq.so; do
      [ -e "$NVPL_LIB/$l" ] || { echo "no $l in $NVPL_LIB" >&2; exit 1; }; done
    export NGSOLVE_BLAS_REQUIREMENT=""
    BLAS_ARGS=("${DIRECT[@]}" -Ccmake.define.USE_MKL=OFF -Ccmake.define.USE_LAPACK=ON
      "-Ccmake.define.LAPACK_LIBRARIES=$NVPL_LIB/libnvpl_lapack_lp64_seq.so;$NVPL_LIB/libnvpl_blas_lp64_seq.so")
    EXCLUDE=(--exclude 'libnvpl_*') ;;
  *) echo "unsupported architecture $ARCH" >&2; exit 1 ;;
esac
[ -z "${OCC_CMAKE:-}" ] || BLAS_ARGS+=(-Ccmake.define.OpenCascade_DIR="$OCC_CMAKE")

CUDA_ARGS=(-Ccmake.define.USE_CUDA=OFF)
if [ -n "${CUDA_ROOT:-}" ]; then
  CUDA_ARGS=(-Ccmake.define.USE_CUDA=ON -Ccmake.define.CUDAToolkit_ROOT="$CUDA_ROOT"
    -Ccmake.define.CMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" -Ccmake.define.CMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH")
  if [ -n "${CUDSS_INCLUDE:-}" ]; then
    [ -f "$CUDSS_INCLUDE/cudss.h" ] || { echo "no cudss.h in $CUDSS_INCLUDE" >&2; exit 1; }
    CUDA_ARGS+=(-Ccmake.define.USE_CUDSS=ON -Ccmake.define.CUDSS_INCLUDE_DIR="$CUDSS_INCLUDE"); fi
  [ -z "${CUDA_REQUIREMENTS:-}" ] || NGSOLVE_EXTRA_REQUIREMENTS="${NGSOLVE_EXTRA_REQUIREMENTS:+$NGSOLVE_EXTRA_REQUIREMENTS;;}$CUDA_REQUIREMENTS"
  EXCLUDE+=(--exclude 'libcudart.so*' --exclude 'libnvrtc.so*' --exclude 'libcuda.so*' --exclude 'libcudss.so*'
    --exclude 'libcublas.so*' --exclude 'libcublasLt.so*')
fi
export NGSOLVE_EXTRA_REQUIREMENTS

step "ngsolve wheel"
(cd "$NGSOLVE" && "$PY" -m build --wheel --no-isolation --outdir "$DIST" -Cbuild-dir="$JOB/b-ngsolve" \
   -Ccmake.define.USE_CCACHE=OFF "${PY_ARGS[@]}" "${BLAS_ARGS[@]}" "${CUDA_ARGS[@]}" .)
step "ngsolve repair"
"$PY" -m auditwheel repair --exclude 'libTK*' --exclude 'libngcore.so' --exclude 'libnglib.so' "${EXCLUDE[@]}" -w "$WH" "$DIST"/ngsolve-*.whl

step "contents"
"$PY" - "$WH" "$NETGEN_BUNDLED_OCC" <<'EOF'
import re, sys, zipfile, pathlib
wh, bundled = pathlib.Path(sys.argv[1]), sys.argv[2] == "1"
nvidia = re.compile(r"(^|/)lib(cudss|cublas|cublasLt|cudart|nvrtc|cuda)[^/]*\.so")
for w in sorted(wh.glob("*.whl")):
    names = zipfile.ZipFile(w).namelist()
    bad = [n for n in names if nvidia.search(n) or re.search(r"(^|/)libTK[^/]*\.so", n)]
    if bad:
        sys.exit(f"{w.name} bundles {bad}")
    meta = next(n for n in names if n.endswith(".dist-info/METADATA"))
    occ = [l for l in zipfile.ZipFile(w).read(meta).decode().splitlines() if l.startswith("Requires-Dist: netgen-occt")]
    if w.name.startswith("netgen_mesher") and bool(occ) == bundled:
        sys.exit(f"{w.name}: netgen-occt requirement {occ} contradicts NETGEN_BUNDLED_OCC={int(bundled)}")
EOF

"$PY" - "$OUT" "$NG_HEAD" "$NGS_HEAD" "$ARCH" "${CUDA_ARCH:-}" "$NETGEN_BUNDLED_OCC" <<'EOF'
import hashlib, json, pathlib, sys
out, ng, ngs, arch, cuda_arch, occ = sys.argv[1:]
wh = sorted(pathlib.Path(out, "wheelhouse").glob("*.whl"))
lane = f"pybind11-{'cuda' if cuda_arch else 'cpu'}-{arch}"
json.dump({"lane": lane, "cuda_arch": cuda_arch or None, "static_occ": occ == "1",
           "netgen": {"head": ng}, "ngsolve": {"head": ngs},
           "wheels": [{"name": w.name, "sha256": hashlib.sha256(w.read_bytes()).hexdigest()} for w in wh]},
          open(pathlib.Path(out, "receipt.json"), "w"), indent=2)
EOF
step WHEELS_DONE
