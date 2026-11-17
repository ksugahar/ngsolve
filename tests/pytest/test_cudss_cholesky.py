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


class StreamGate:
    """The producer stream waits (cuStreamWaitValue32, GEQ 1) on a host-mapped flag until the host opens it."""
    def __init__(self, cu, stream):
        self.cu, self.stream = cu, stream
        drv = ctypes.CDLL("nvcuda.dll" if sys.platform.startswith("win") else "libcuda.so.1")
        self.wait = getattr(drv, "cuStreamWaitValue32_v2", None) or drv.cuStreamWaitValue32
        self.write = getattr(drv, "cuStreamWriteValue32_v2", None) or drv.cuStreamWriteValue32
        self.host = ctypes.c_void_p()
        cu.check(cu.l.cudaHostAlloc(ctypes.byref(self.host), ctypes.c_size_t(4), 2), "cudaHostAlloc mapped")  # cudaHostAllocMapped
        self.flag = (ctypes.c_uint32 * 1).from_address(self.host.value)
        self.flag[0] = 0
        self.dev = ctypes.c_void_p()
        cu.check(cu.l.cudaHostGetDevicePointer(ctypes.byref(self.dev), self.host, 0), "cudaHostGetDevicePointer")

    def hold(self):
        assert self.flag[0] == 0
        err = self.wait(self.stream, ctypes.c_uint64(self.dev.value), ctypes.c_uint32(1), ctypes.c_uint(0))  # CU_STREAM_WAIT_VALUE_GEQ
        assert err == 0, f"cuStreamWaitValue32: driver error {err}"

    def open(self):
        self.flag[0] = 1

    def open_from(self, stream):
        # the device opens the gate from another stream: Update holds the GIL, a Python thread could not
        err = self.write(stream, ctypes.c_uint64(self.dev.value), ctypes.c_uint32(1), ctypes.c_uint(0))
        assert err == 0, f"cuStreamWriteValue32: driver error {err}"

    def free(self):
        self.cu.l.cudaFreeHost(self.host)


def test_ready_event_orders_a_producer_on_another_stream():
    # Values only, production contract unchanged (an event-free Update synchronizes the device):
    #  - control: no producer work submitted yet; event-free Update on the quiescent device gives
    #    the solution for A, distinguishable from the one for 2*A;
    #  - event case: the producer (2*A) is held on a non-blocking stream by an explicit gate that
    #    a second stream opens; Update with the recorded event must give the 2*A solution.
    # That Update waits on exactly the passed event is checked by the probe test below.
    cu = Cudart()
    fes, a, f = problem()
    rp, cl, vv, (prp, pcl, pv) = device_csr(cu, a.mat, fes.ndof)
    stream, opener, ev = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    cu.check(cu.l.cudaStreamCreateWithFlags(ctypes.byref(stream), 1), "stream")
    cu.check(cu.l.cudaStreamCreateWithFlags(ctypes.byref(opener), 1), "stream")
    cu.check(cu.l.cudaEventCreate(ctypes.byref(ev)), "event")
    gate = StreamGate(cu, stream)
    host2 = np.ascontiguousarray(2 * vv)
    try:
        chol = ngscuda.CuDSSCholesky(fes.ndof, len(cl), prp, pcl, pv, S.Lower, fes.FreeDofs())
        once = gpu_solve(chol, f.vec)                          # solution for A
        new = 0.5 * once                                       # solution for 2*A

        chol.Update(prp, pcl, pv)                              # control: publication withheld on the host
        old_sol = gpu_solve(chol, f.vec)
        assert rel(old_sol, once) <= 1e-12, ("control did not give the A solution", rel(old_sol, once))
        assert rel(old_sol, new) > 1e-3, ("A and 2*A solutions not distinguishable", rel(old_sol, new))

        gate.hold()                                            # now submit 2*A behind the gate
        cu.check(cu.l.cudaMemcpyAsync(ctypes.c_void_p(pv), host2.ctypes.data_as(ctypes.c_void_p),
                                      ctypes.c_size_t(host2.nbytes), 1, stream), "produce")
        cu.check(cu.l.cudaEventRecord(ev, stream), "record")
        gate.open_from(opener)
        chol.Update(prp, pcl, pv, ready_event=ev.value)
        assert rel(gpu_solve(chol, f.vec), new) <= 1e-12, "update with the event did not see 2*A"
    finally:
        gate.open()                                            # never leave the stream held
        cu.l.cudaStreamSynchronize(stream); cu.l.cudaStreamSynchronize(opener)
        cu.l.cudaStreamDestroy(stream); cu.l.cudaStreamDestroy(opener); cu.l.cudaEventDestroy(ev)
        gate.free()
        for p in (prp, pcl, pv): cu.free(p)


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
  if (getenv ("EVSYNC_DISABLE")) return 0;             /* mutant: the wait is removed, nothing recorded */
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
out = {"ev": ev.value, "other": other.value}
for phase, kw in (("passed", {"ready_event": ev.value}), ("withheld", {}), ("wrong", {"ready_event": other.value}),
                  ("wait_disabled", {"ready_event": ev.value})):
    open(log, "w").close()
    if phase == "wait_disabled":
        os.environ["EVSYNC_DISABLE"] = "1"
    chol.Update(prp, pcl, pv, **kw)
    os.environ.pop("EVSYNC_DISABLE", None)
    out[phase] = [int(x) for x in open(log).read().split()]
print(json.dumps(out))
"""


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="probe uses LD_PRELOAD interposition (Linux)")
def test_ready_event_is_synchronized_probe(tmp_path):
    # Interpose cudaEventSynchronize: Update must synchronize exactly the event it was given.
    # The same check must fail when the event is withheld, a different event is passed, or the
    # right event is passed but its wait is removed (isolated mutant in the interposer).
    import json, shutil, subprocess
    cc = shutil.which("cc") or shutil.which("gcc")
    assert cc, "a C compiler is required for the probe"
    shim = tmp_path / "evsync.so"
    (tmp_path / "evsync.c").write_text(EVSYNC_SHIM)
    subprocess.run([cc, "-shared", "-fPIC", "-O2", "-o", str(shim), str(tmp_path / "evsync.c"), "-ldl"], check=True)
    (tmp_path / "child.py").write_text(EVSYNC_CHILD)
    env = dict(os.environ, LD_PRELOAD=str(shim), EVSYNC_LOG=str(tmp_path / "evsync.log"))
    r = subprocess.run([sys.executable, str(tmp_path / "child.py"), os.path.abspath(__file__)],
                       env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    waited = lambda phase, e: e in out[phase]
    assert waited("passed", out["ev"]), ("Update did not synchronize the given event", out)
    assert not waited("withheld", out["ev"]), ("the check passes without the event", out)
    assert not waited("wrong", out["ev"]) and waited("wrong", out["other"]), ("the check passes with a wrong event", out)
    assert not waited("wait_disabled", out["ev"]), ("the check passes with the wait removed", out)


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
