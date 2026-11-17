# CuDSSCholesky (fork patch P10): cuDSS sparse Cholesky of the free block on the GPU.
# Oracle: the CPU sparsecholesky on the same free block. Needs a CUDA build with
# USE_CUDSS and a GPU; skipped (visibly) otherwise.
import ctypes, os, sys, time
import numpy as np
import pytest
from ngsolve import *
import ngsolve.la as la
from ngsolve.meshes import MakeStructured3DMesh

SparseMatrixd = getattr(la, "SparseMatrixd", None) or getattr(la, "SparseMatrixdouble")  # name from typeid, MSVC: double
ngscuda = pytest.importorskip("ngsolve.ngscuda")
if not hasattr(ngscuda, "CuDSSCholesky"):
    pytest.skip("ngsolve built without USE_CUDSS", allow_module_level=True)
REPORT = ngscuda.LoadCuDSS()
S = ngscuda.CuDSSStorage
DIFF_TOL = 1e-10      # proposed, user approval pending (design rev 3, section 8)


def problem(n=6, order=2, symmetric=True, coef=1.0):
    mesh = MakeStructured3DMesh(hexes=False, nx=n, ny=n, nz=n)
    fes = H1(mesh, order=order, dirichlet="left|bottom")
    u, v = fes.TnT()
    a = BilinearForm(coef * grad(u) * grad(v) * dx + u * v * dx, symmetric=symmetric, symmetric_storage=symmetric).Assemble()
    f = LinearForm(sin(3 * x) * v * dx).Assemble()
    return fes, a, f


def coo(mat):
    r, c, v = mat.COO()
    return np.array(r), np.array(c), np.array(v)


def from_coo(r, c, v, n):
    return SparseMatrixd.CreateFromCOO(list(map(int, r)), list(map(int, c)), list(map(float, v)), n, n)


def full_lower_upper(a, n):
    r, c, v = coo(a.mat)                       # symmetric=True: lower triangle incl. diagonal
    lo = r >= c
    r, c, v = r[lo], c[lo], v[lo]
    off = r != c
    full = from_coo(np.r_[r, c[off]], np.r_[c, r[off]], np.r_[v, v[off]], n)
    lower = from_coo(r, c, v, n)
    upper = from_coo(c, r, v, n)
    return full, lower, upper


def cpu_solution(mat, free, f):
    inv = mat.Inverse(free, inverse="sparsecholesky")
    x = mat.CreateVector(); x.data = inv * f
    return np.array(x.FV())


def gpu_solve(chol, f, s=1.0, y0=None):
    x = chol.CreateColVector(); x.data = f                 # the test's own explicit upload
    y = chol.CreateColVector()
    if y0 is None:
        y[:] = 0
    else:
        y.data = y0
    chol.Solve(s, x, y)
    back = f.CreateVector(); back.data = y                  # explicit download
    return np.array(back.FV())


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


@pytest.mark.parametrize("order", [2, 3])
@pytest.mark.parametrize("storage", ["Full", "Lower", "Upper"])
def test_matches_cpu_sparsecholesky(order, storage):
    fes, a, f = problem(order=order)
    full, lower, upper = full_lower_upper(a, fes.ndof)
    mat = {"Full": full, "Lower": lower, "Upper": upper}[storage]
    ref = cpu_solution(full, fes.FreeDofs(), f.vec)
    chol = ngscuda.CuDSSCholesky(mat, fes.FreeDofs(), getattr(S, storage))
    got = gpu_solve(chol, f.vec)
    assert rel(got, ref) <= DIFF_TOL, (storage, order, rel(got, ref))
    assert '"last_rel_residual"' in chol.Report()


def test_constrained_entries_and_scaling():
    fes, a, f = problem()
    free = np.array(list(fes.FreeDofs()), bool)
    chol = ngscuda.CuDSSCholesky(a.mat, fes.FreeDofs(), S.Lower)
    ref = cpu_solution(a.mat, fes.FreeDofs(), f.vec)
    y0 = np.random.default_rng(1).random(fes.ndof)
    y0v = a.mat.CreateVector(); y0v.FV().NumPy()[:] = y0
    got = gpu_solve(chol, f.vec, s=-2.5, y0=y0v)
    assert np.array_equal(got[~free], y0[~free]), "constrained entries of y changed"
    assert rel(got[free] - y0[free], -2.5 * ref[free]) <= DIFF_TOL


def test_update_equals_fresh_factorization():
    fes, a, f = problem()
    chol = ngscuda.CuDSSCholesky(a.mat, fes.FreeDofs(), S.Lower)
    _, a2, _ = problem(coef=7.0)
    chol.Update(a2.mat)
    fresh = ngscuda.CuDSSCholesky(a2.mat, fes.FreeDofs(), S.Lower)
    assert rel(gpu_solve(chol, f.vec), gpu_solve(fresh, f.vec)) <= 1e-13


def test_update_refuses_changed_pattern_same_nnz():
    # all-free 4x4 SPD, lower storage: diag 4, one -1 moved (2,0) -> (2,1); same nnz and row offsets
    n = 4
    free = BitArray(n); free.Set()
    def lower4(col):
        return from_coo([0, 1, 2, 3, 2], [0, 1, 2, 3, col], [4.0, 4.0, 4.0, 4.0, -1.0], n)
    a, moved = lower4(0), lower4(1)
    ra, ca, _ = a.CSR(); rm, cm, _ = moved.CSR()
    assert a.nze == moved.nze and list(ra) == list(rm) and list(ca) != list(cm), (list(ra), list(ca), list(rm), list(cm))
    chol = ngscuda.CuDSSCholesky(a, free, S.Lower)
    f = a.CreateVector(); f.FV().NumPy()[:] = 1.0
    with pytest.raises(Exception, match="pattern changed"):
        chol.Update(moved)
    with pytest.raises(Exception, match="no valid factorization"):
        gpu_solve(chol, f)


def test_refusals():
    fes, a, f = problem()
    n = fes.ndof
    full, lower, upper = full_lower_upper(a, n)
    r, c, v = coo(full)
    k = np.flatnonzero((r < c) & np.array([fes.FreeDofs()[int(i)] and fes.FreeDofs()[int(j)] for i, j in zip(r, c)]))[0]
    v2 = v.copy(); v2[k] = np.nextafter(v2[k], np.inf)            # one ulp: not exactly symmetric
    with pytest.raises(Exception, match="not exactly symmetric"):
        ngscuda.CuDSSCholesky(from_coo(r, c, v2, n), fes.FreeDofs(), S.Full)
    v3 = v.copy(); v3[k] = np.nan
    with pytest.raises(Exception, match="non-finite"):
        ngscuda.CuDSSCholesky(from_coo(r, c, v3, n), fes.FreeDofs(), S.Full)
    with pytest.raises(Exception, match="not positive definite"):           # indefinite
        ngscuda.CuDSSCholesky(from_coo(r, c, -v, n), fes.FreeDofs(), S.Full)
    with pytest.raises(Exception, match="above the diagonal"):
        ngscuda.CuDSSCholesky(upper, fes.FreeDofs(), S.Lower)
    rl, cl_, vl = coo(lower)
    keep = ~((rl == cl_) & (rl == int(np.flatnonzero(np.array(list(fes.FreeDofs())))[0])))
    with pytest.raises(Exception, match="no stored diagonal"):
        ngscuda.CuDSSCholesky(from_coo(rl[keep], cl_[keep], vl[keep], n), fes.FreeDofs(), S.Lower)
    chol = ngscuda.CuDSSCholesky(lower, fes.FreeDofs(), S.Lower)
    with pytest.raises(Exception, match="DeviceVector"):                    # host vectors
        y = f.vec.CreateVector(); chol.Solve(1.0, f.vec, y)
    x = chol.CreateColVector(); x.data = f.vec
    with pytest.raises(Exception, match="alias"):
        chol.Solve(1.0, x, x)


def test_singular():
    mesh = MakeStructured3DMesh(hexes=False, nx=4, ny=4, nz=4)
    fes = H1(mesh, order=1)                                   # pure Neumann Laplacian: singular
    u, v = fes.TnT()
    a = BilinearForm(grad(u) * grad(v) * dx, symmetric=True).Assemble()
    with pytest.raises(Exception):
        chol = ngscuda.CuDSSCholesky(a.mat, fes.FreeDofs(), S.Lower)
        f = LinearForm(x * v * dx).Assemble()
        gpu_solve(chol, f.vec)                                # residual check must refuse


def test_empty_free_set_and_zero_rhs():
    fes, a, f = problem()
    none = BitArray(fes.ndof); none.Clear()
    chol = ngscuda.CuDSSCholesky(a.mat, none, S.Lower)
    y0 = a.mat.CreateVector(); y0.FV().NumPy()[:] = 3.0
    assert np.all(gpu_solve(chol, f.vec, y0=y0) == 3.0)
    chol = ngscuda.CuDSSCholesky(a.mat, fes.FreeDofs(), S.Lower)
    z = f.vec.CreateVector(); z[:] = 0
    assert np.all(gpu_solve(chol, z, y0=y0) == 3.0)


def test_conditioning_control():
    fes, a, f = problem(coef=1e6)                             # stiffness/mass contrast 1e6
    chol = ngscuda.CuDSSCholesky(a.mat, fes.FreeDofs(), S.Lower)
    ref = cpu_solution(a.mat, fes.FreeDofs(), f.vec)
    assert rel(gpu_solve(chol, f.vec), ref) <= DIFF_TOL


def test_analytic_1d_chain():
    # tridiag(-1, 2, -1) of size m with b = e_1: x_i = (m + 1 - i) / (m + 1), i = 1..m
    m = 200
    r = list(range(m)) + list(range(1, m)); c = list(range(m)) + list(range(m - 1))
    v = [2.0] * m + [-1.0] * (m - 1)
    mat = from_coo(r, c, v, m)
    allf = BitArray(m); allf.Set()
    chol = ngscuda.CuDSSCholesky(mat, allf, S.Lower)
    b = mat.CreateVector(); b[:] = 0; b[0] = 1
    got = gpu_solve(chol, b)
    exact = np.array([(m + 1 - i) / (m + 1) for i in range(1, m + 1)])
    assert rel(got, exact) <= 1e-12


# ---- caller-owned device CSR (constructor B) and the ready-event contract

class Cudart:
    def __init__(self):
        name = "cudart64_12.dll" if sys.platform.startswith("win") else "libcudart.so.12"
        self.l = ctypes.CDLL(name)
    def check(self, err, what):
        assert err == 0, f"{what}: cuda error {err}"
    def malloc(self, nbytes):
        p = ctypes.c_void_p(); self.check(self.l.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(nbytes)), "cudaMalloc"); return p.value
    def free(self, p): self.l.cudaFree(ctypes.c_void_p(p))
    def h2d(self, p, arr):
        arr = np.ascontiguousarray(arr)
        self.check(self.l.cudaMemcpy(ctypes.c_void_p(p), arr.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(arr.nbytes), 1), "h2d")


def device_csr(cu, mat, n):
    vals, cols, firsti = mat.CSR()
    rp = np.array(firsti, np.int32); cl = np.array(cols, np.int32); vv = np.array(vals, np.float64)
    ptrs = [cu.malloc(max(rp.nbytes, 4)), cu.malloc(max(cl.nbytes, 4)), cu.malloc(max(vv.nbytes, 8))]
    for p, arr in zip(ptrs, (rp, cl, vv)):
        cu.h2d(p, arr)
    return rp, cl, vv, ptrs


def test_device_csr_route_matches_host_route():
    cu = Cudart()
    fes, a, f = problem()
    rp, cl, vv, (prp, pcl, pv) = device_csr(cu, a.mat, fes.ndof)
    try:
        chol_b = ngscuda.CuDSSCholesky(fes.ndof, len(cl), prp, pcl, pv, S.Lower, fes.FreeDofs())
        chol_a = ngscuda.CuDSSCholesky(a.mat, fes.FreeDofs(), S.Lower)
        assert rel(gpu_solve(chol_b, f.vec), gpu_solve(chol_a, f.vec)) <= 1e-13
        assert '"host_to_device_bytes": ' in chol_b.Report()
    finally:
        for p in (prp, pcl, pv): cu.free(p)


def _gated_d2d_case():
    # Runs in a child process (bounded by the parent's timeout). 2*A is uploaded to a device
    # buffer first; then a gate in device memory is armed on a non-blocking producer stream
    # (cuStreamWaitValue32, GEQ 1), the D2D publication into the caller's value buffer and the
    # ready event are queued behind it, and a second stream opens the gate (cuStreamWriteValue32).
    import json
    cu = Cudart()
    drv = ctypes.CDLL("nvcuda.dll" if sys.platform.startswith("win") else "libcuda.so.1")
    wait = getattr(drv, "cuStreamWaitValue32_v2", None) or drv.cuStreamWaitValue32
    write = getattr(drv, "cuStreamWriteValue32_v2", None) or drv.cuStreamWriteValue32
    fes, a, f = problem()
    rp, cl, vv, (prp, pcl, pv) = device_csr(cu, a.mat, fes.ndof)
    host2 = np.ascontiguousarray(2 * vv)
    d2, flag = cu.malloc(host2.nbytes), cu.malloc(4)
    stream, opener, ev = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    cu.check(cu.l.cudaStreamCreateWithFlags(ctypes.byref(stream), 1), "stream")
    cu.check(cu.l.cudaStreamCreateWithFlags(ctypes.byref(opener), 1), "stream")
    cu.check(cu.l.cudaEventCreate(ctypes.byref(ev)), "event")
    out = {}
    try:
        chol = ngscuda.CuDSSCholesky(fes.ndof, len(cl), prp, pcl, pv, S.Lower, fes.FreeDofs())
        once = gpu_solve(chol, f.vec)                          # solution for A
        new = 0.5 * once                                       # solution for 2*A
        chol.Update(prp, pcl, pv)                              # control: nothing published yet
        old_sol = gpu_solve(chol, f.vec)
        out["control_vs_A"] = rel(old_sol, once)
        out["control_vs_2A"] = rel(old_sol, new)

        cu.h2d(d2, host2)                                      # pre-upload 2*A
        cu.check(cu.l.cudaMemset(ctypes.c_void_p(flag), 0, ctypes.c_size_t(4)), "gate reset")
        cu.check(cu.l.cudaDeviceSynchronize(), "sync")
        err = wait(stream, ctypes.c_uint64(flag), ctypes.c_uint32(1), ctypes.c_uint(0))   # CU_STREAM_WAIT_VALUE_GEQ
        assert err == 0, f"cuStreamWaitValue32: driver error {err}"
        cu.check(cu.l.cudaMemcpyAsync(ctypes.c_void_p(pv), ctypes.c_void_p(d2), ctypes.c_size_t(host2.nbytes),
                                      3, stream), "publish D2D")
        cu.check(cu.l.cudaEventRecord(ev, stream), "record")
        out["pending_before_open"] = cu.l.cudaEventQuery(ev) != 0  # the gate holds the publication
        err = write(opener, ctypes.c_uint64(flag), ctypes.c_uint32(1), ctypes.c_uint(0))
        assert err == 0, f"cuStreamWriteValue32: driver error {err}"
        chol.Update(prp, pcl, pv, ready_event=ev.value)
        out["event_vs_2A"] = rel(gpu_solve(chol, f.vec), new)
    finally:
        write(opener, ctypes.c_uint64(flag), ctypes.c_uint32(1), ctypes.c_uint(0))   # never leave it held
        cu.l.cudaStreamSynchronize(opener); cu.l.cudaStreamSynchronize(stream)
        cu.l.cudaStreamDestroy(stream); cu.l.cudaStreamDestroy(opener); cu.l.cudaEventDestroy(ev)
        for p in (prp, pcl, pv, d2, flag): cu.free(p)
    print(json.dumps(out))


GATED_CHILD = r"""
import importlib.util, sys
spec = importlib.util.spec_from_file_location("cudss_test_gated", sys.argv[1])
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
t._gated_d2d_case()
"""


def test_ready_event_orders_a_producer_on_another_stream(tmp_path):
    # Values only, production contract unchanged (an event-free Update synchronizes the device):
    #  - control: nothing published yet; event-free Update gives the A solution, distinguishable
    #    from the 2*A one;
    #  - event case: 2*A pre-uploaded, its D2D publication held behind a device-memory gate that a
    #    second stream opens; Update with the recorded event must give the 2*A solution.
    # Runs in a child process: a gate that never opens fails within the timeout instead of hanging.
    # That Update waits on exactly the passed event is checked by the probe test below.
    import json, subprocess
    (tmp_path / "child.py").write_text(GATED_CHILD)
    try:
        r = subprocess.run([sys.executable, str(tmp_path / "child.py"), os.path.abspath(__file__)],
                           capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        pytest.fail("gated producer did not complete within 180 s (gate never opened?)")
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["control_vs_A"] <= 1e-12, ("control did not give the A solution", out)
    assert out["control_vs_2A"] > 1e-3, ("A and 2*A solutions not distinguishable", out)
    assert out["pending_before_open"], ("the gate did not hold the publication", out)
    assert out["event_vs_2A"] <= 1e-12, ("update with the event did not see 2*A", out)


EVSYNC_SHIM = r"""
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
typedef int (*sync_fn)(void *);
int cudaEventSynchronize (void * e)
{
  static sync_fn real;
  if (!real) real = (sync_fn) dlsym (RTLD_NEXT, "cudaEventSynchronize");
  if (!real)                    /* cudart came in with a dlopen'd module: not in the global scope */
    {
      void * h = dlopen ("libcudart.so.12", RTLD_NOW | RTLD_NOLOAD);
      if (h) real = (sync_fn) dlsym (h, "cudaEventSynchronize");
    }
  if (!real) { fprintf (stderr, "evsync shim: cudaEventSynchronize not found\n"); abort (); }
  const char * d = getenv ("EVSYNC_DISABLE");           /* mutant: drop the wait for this one event only */
  if (d && strtoull (d, 0, 10) == (unsigned long long)(uintptr_t) e)
    {
      const char * h = getenv ("EVSYNC_SUPPRESSED");
      if (h) { FILE * f = fopen (h, "a"); if (f) { fprintf (f, "%llu\n", (unsigned long long)(uintptr_t) e); fclose (f); } }
      return 0;
    }
  const char * p = getenv ("EVSYNC_LOG");
  if (p) { FILE * f = fopen (p, "a"); if (f) { fprintf (f, "%llu\n", (unsigned long long)(uintptr_t) e); fclose (f); } }
  return real (e);
}
"""

EVSYNC_CHILD = r"""
import ctypes, importlib.util, json, os, sys
spec = importlib.util.spec_from_file_location("cudss_test_under_probe", sys.argv[1])
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
cu = t.Cudart()
fes, a, f = t.problem()
rp, cl, vv, (prp, pcl, pv) = t.device_csr(cu, a.mat, fes.ndof)
chol = t.ngscuda.CuDSSCholesky(fes.ndof, len(cl), prp, pcl, pv, t.S.Lower, fes.FreeDofs())
stream, ev, other = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
cu.check(cu.l.cudaStreamCreateWithFlags(ctypes.byref(stream), 1), "stream")
for e in (ev, other):
    cu.check(cu.l.cudaEventCreate(ctypes.byref(e)), "event")
    cu.check(cu.l.cudaEventRecord(e, stream), "record")
cu.check(cu.l.cudaStreamSynchronize(stream), "sync")
log = os.environ["EVSYNC_LOG"]
hits = os.environ["EVSYNC_SUPPRESSED"]
out = {"ev": ev.value, "other": other.value}
for phase, kw in (("passed", {"ready_event": ev.value}), ("withheld", {}), ("wrong", {"ready_event": other.value}),
                  ("wait_disabled", {"ready_event": ev.value})):
    open(log, "w").close(); open(hits, "w").close()
    if phase == "wait_disabled":
        os.environ["EVSYNC_DISABLE"] = str(ev.value)
    chol.Update(prp, pcl, pv, **kw)
    os.environ.pop("EVSYNC_DISABLE", None)
    out[phase] = [int(x) for x in open(log).read().split()]
    out[phase + "_suppressed"] = [int(x) for x in open(hits).read().split()]
print(json.dumps(out))
"""


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="probe uses LD_PRELOAD interposition (Linux)")
def test_ready_event_is_synchronized_probe(tmp_path):
    # Interpose cudaEventSynchronize: Update must synchronize exactly the event it was given.
    # The same check must fail when the event is withheld, a different event is passed, or the
    # right event is passed but the wait on that event alone is removed (isolated interposer mutant).
    import json, shutil, subprocess
    cc = shutil.which("cc") or shutil.which("gcc")
    assert cc, "a C compiler is required for the probe"
    shim = tmp_path / "evsync.so"
    (tmp_path / "evsync.c").write_text(EVSYNC_SHIM)
    subprocess.run([cc, "-shared", "-fPIC", "-O2", "-o", str(shim), str(tmp_path / "evsync.c"), "-ldl"], check=True)
    (tmp_path / "child.py").write_text(EVSYNC_CHILD)
    env = dict(os.environ, LD_PRELOAD=str(shim), EVSYNC_LOG=str(tmp_path / "evsync.log"),
               EVSYNC_SUPPRESSED=str(tmp_path / "evsync.suppressed"))
    r = subprocess.run([sys.executable, str(tmp_path / "child.py"), os.path.abspath(__file__)],
                       env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    waited = lambda phase, e: e in out[phase]
    assert waited("passed", out["ev"]), ("Update did not synchronize the given event", out)
    assert not waited("withheld", out["ev"]), ("the check passes without the event", out)
    assert not waited("wrong", out["ev"]) and waited("wrong", out["other"]), ("the check passes with a wrong event", out)
    assert out["wait_disabled_suppressed"] == [out["ev"]], ("the mutant did not suppress the target wait", out)
    assert not waited("wait_disabled", out["ev"]), ("the check passes with the wait removed", out)
    assert all(not out[p + "_suppressed"] for p in ("passed", "withheld", "wrong")), ("suppression outside the mutant", out)


def test_missing_library_is_an_error_at_load():
    # fresh process: ngscuda imports without cuDSS; loading a missing file raises at load time
    import subprocess
    missing = os.path.join(os.path.dirname(os.path.abspath(__file__)), "no_such_cudss.dll")
    code = "\n".join([
        "import ngsolve.ngscuda as g, ngsolve._ngscuda as m",
        "try:",
        "    m._LoadCuDSS(%r, 0, 8, 0)" % missing,
        "except Exception as e:",
        "    print('REFUSED', e)"])
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0 and "REFUSED cuDSS: cannot load" in out.stdout, (out.stdout, out.stderr)
