#!/bin/bash
# Type I step of type1.pbs: install the sealed build's verified wheels into a
# clean venv, then run type1_bench.py (analytic and SciPy references, agreement
# across 1/16/48 threads; only the residual <= 1e-6 gates, timings never gate).
set -Eeuo pipefail
umask 077
: "${JOB_ROOT:?}" "${RUN:?}" "${BUILD:?}"
run=$JOB_ROOT/$RUN
mkdir "$run"
cd "$run"
exec >run.log 2>&1
phase=allocation
trap 'rc=$?; printf "phase=%s\nexit_code=%s\n" "$phase" "$rc" > terminal.txt' EXIT
test "$(uname -m)" = x86_64
python3.12 -c 'import json,os,socket,sys; json.dump({"run":sys.argv[1],"pbs_jobid":os.environ["PBS_JOBID"],"node":socket.gethostname()},open("job.json","w"),indent=2)' "$RUN"
qstat -f "$PBS_JOBID" | sed '/^[[:space:]]*Variable_List = /,$d' > allocation.txt
lscpu > cpu.txt; numactl --hardware > numa.txt
b=$JOB_ROOT/$BUILD
test -e "$b/sealed.txt"
grep -qx 'phase=complete' "$b/terminal.txt"
(cd "$b/out/wheelhouse" && sha256sum -c ../../wheel.sha256) > wheel-verify.txt
python3.12 "$JOB_ROOT/source/tools/flow2/summarize.py" identity "$BUILD" "$b" > validated-wheels.json
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH LD_LIBRARY_PATH
phase=environment
python3.12 -m venv venv
export PATH="$run/venv/bin:$PATH"
python -m pip install -q --no-index --require-hashes --find-links "$JOB_ROOT/wheelhouse-validate" \
  -r "$JOB_ROOT/source/tools/flow2/requirements-validate-x86_64.txt"
python -m pip install -q --no-index --no-deps "$b"/out/wheelhouse/*.whl
python -m pip check > pip-check.txt
python -m pip freeze > freeze.txt
phase=import
python -I -c 'import json, ngsolve, netgen; print(json.dumps({"ngsolve": ngsolve.__file__, "netgen": netgen.__file__, "version": ngsolve.__version__}))' > import.json
python -I -c 'import netgen.occ as o; b = o.Box(o.Pnt(0,0,0), o.Pnt(1,2,3)); assert abs(b.mass - 6) < 1e-12, b.mass; print("occ box volume", b.mass)' > occ.txt
phase=bench
cp "$JOB_ROOT/source/tools/flow2/type1_bench.py" bench.py
python -I bench.py | tee perf.log
sha256sum validated-wheels.json wheel-verify.txt import.json occ.txt pip-check.txt freeze.txt result.json perf.log \
  bench.py cpu.txt numa.txt allocation.txt > evidence.sha256
phase=complete
