# Flow2 Type I (t1-single) validation of the installed CPU wheels (MKL from the pinned mkl wheel).
# Performance: same mesh/RHS/formulation at 1, 16, 48 NGSolve threads, pinned to physical cores of one
# NUMA node with memory bound to that node. Reference checks: (a) analytic manufactured solution,
# (b) independent direct solver (SciPy SuperLU) on the same algebraic system, (c) solution agreement
# across thread counts. Only the policy residual limit 1e-6 gates; no other acceptance number is invented.
import hashlib, json, math, os, statistics, subprocess, sys, time
from pathlib import Path

LIMIT = 1e-6
N_PERF = 40
THREADS = (1, 16, 48)
RUN_ORDER = [48, 16, 1, 1, 16, 48]


def check_prefix():
    # the installed wheels in this venv, never a checkout
    import ngsolve, netgen
    venv = Path(sys.prefix).resolve()
    assert all(Path(m.__file__).resolve().is_relative_to(venv) for m in (ngsolve, netgen)), (ngsolve.__file__, netgen.__file__)


def perf_problem(n):
    from ngsolve import H1, BilinearForm, LinearForm, TaskManager, dx, grad, sin, x
    from ngsolve.meshes import MakeStructured3DMesh
    mesh = MakeStructured3DMesh(hexes=False, nx=n, ny=n, nz=n)
    fes = H1(mesh, order=2, dirichlet="left|bottom")
    u, v = fes.TnT()
    with TaskManager():
        a = BilinearForm(grad(u)*grad(v)*dx + u*v*dx, symmetric=True, symmetric_storage=True).Assemble()
        f = LinearForm(sin(3*x)*v*dx).Assemble()
    return mesh, fes, a, f


def free_residual(a, f, xv, free):
    import numpy as np
    from ngsolve import TaskManager
    r = f.vec.CreateVector()
    with TaskManager():
        r.data = f.vec - a.mat * xv
    m = np.array(list(free), bool)
    return float(np.linalg.norm(r.FV().NumPy()[m]) / np.linalg.norm(f.vec.FV().NumPy()[m]))


def sample(threads, tag):
    import numpy as np
    from ngsolve import SetNumThreads, TaskManager
    check_prefix()
    assert len(os.sched_getaffinity(0)) == threads
    SetNumThreads(threads)
    t0 = time.perf_counter()
    mesh, fes, a, f = perf_problem(N_PERF)
    t1 = time.perf_counter()
    free = fes.FreeDofs()
    xv = a.mat.CreateVector()
    t2 = time.perf_counter()
    with TaskManager():
        inv = a.mat.Inverse(free, inverse="sparsecholesky")
    t3 = time.perf_counter()
    solves = []
    for _ in range(3):
        t = time.perf_counter()
        with TaskManager():
            xv.data = inv * f.vec
        solves.append(time.perf_counter() - t)
    res = free_residual(a, f, xv, free)
    if not (math.isfinite(res) and res <= LIMIT):
        raise SystemExit(f"true residual {res} above {LIMIT}")
    np.save(f"x_{threads}_{tag}.npy", xv.FV().NumPy().copy())
    print(json.dumps(dict(threads=threads, tag=tag, cpu_ids=sorted(os.sched_getaffinity(0)), ndof=fes.ndof,
        nfree=sum(free), stored_entries=a.mat.nze, assembly_s=t1-t0, factorization_s=t3-t2,
        solve_s=statistics.median(solves), solve_samples_s=solves, total_s=t3-t0+solves[0],
        true_relative_residual=res)))


def analytic_reference():
    # -lap u + u = f on the unit cube, u = sin(pi x) sin(pi y) sin(pi z), u = 0 on every face
    from ngsolve import (H1, BilinearForm, LinearForm, GridFunction, TaskManager, Integrate, dx, grad, sin, x, y, z, SetNumThreads)
    from ngsolve.meshes import MakeStructured3DMesh
    import numpy as np
    SetNumThreads(len(os.sched_getaffinity(0)))
    exact = sin(math.pi*x)*sin(math.pi*y)*sin(math.pi*z)
    rows = []
    for n in (8, 16, 32):
        mesh = MakeStructured3DMesh(hexes=False, nx=n, ny=n, nz=n)
        fes = H1(mesh, order=2, dirichlet=".*")
        u, v = fes.TnT()
        with TaskManager():
            a = BilinearForm(grad(u)*grad(v)*dx + u*v*dx, symmetric=True, symmetric_storage=True).Assemble()
            f = LinearForm((3*math.pi**2 + 1)*exact*v*dx).Assemble()
            g = GridFunction(fes)
            g.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * f.vec
        res = free_residual(a, f, g.vec, fes.FreeDofs())
        err = math.sqrt(Integrate((g - exact)**2, mesh, order=8))
        rows.append(dict(n=n, h=1/n, ndof=fes.ndof, l2_error=err, true_relative_residual=res))
        if not (math.isfinite(res) and res <= LIMIT):
            raise SystemExit(f"analytic case residual {res}")
    for prev, cur in zip(rows, rows[1:]):
        cur["observed_rate"] = math.log(prev["l2_error"]/cur["l2_error"])/math.log(2)
    return dict(problem="-lap u + u = f, u = sin(pi x)sin(pi y)sin(pi z), homogeneous Dirichlet on all faces, H1 P2",
                expected_rate="p+1 = 3 (theory; reported, not gated)", rows=rows)


def scipy_reference(n=16):
    # Independent direct solver (SuperLU, general LU) on the same assembled free system.
    import numpy as np, scipy, scipy.sparse as sp, scipy.sparse.linalg as spla
    from ngsolve import TaskManager, SetNumThreads
    SetNumThreads(len(os.sched_getaffinity(0)))
    mesh, fes, a, f = perf_problem(n)
    free = np.array(list(fes.FreeDofs()), bool)
    vals, cols, rowptr = (np.asarray(t) for t in a.mat.CSR())
    vals, cols, rowptr = vals.astype(np.float64), cols.astype(np.int64), rowptr.astype(np.int64)
    L = sp.csr_matrix((vals, cols, rowptr), shape=(fes.ndof, fes.ndof))
    A = (L + L.T - sp.diags(L.diagonal())).tocsc()[free][:, free]
    b = f.vec.FV().NumPy()[free]
    xs = spla.spsolve(A, b)
    xv = a.mat.CreateVector()
    with TaskManager():
        xv.data = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * f.vec
    xn = xv.FV().NumPy()[free]
    return dict(n=n, ndof=fes.ndof, nfree=int(free.sum()), scipy=scipy.__version__, solver="scipy.sparse.linalg.spsolve (SuperLU)",
                rel_diff_ngsolve_vs_scipy=float(np.linalg.norm(xn - xs)/np.linalg.norm(xs)),
                scipy_true_relative_residual=float(np.linalg.norm(b - A @ xs)/np.linalg.norm(b)),
                ngsolve_true_relative_residual=float(np.linalg.norm(b - A @ xn)/np.linalg.norm(b)))


def run():
    import numpy as np
    topo = subprocess.check_output(["lscpu", "-p=CPU,CORE,SOCKET,NODE"], text=True)
    allowed = os.sched_getaffinity(0)
    nodes = {}
    for row in topo.splitlines():
        if row and not row.startswith("#"):
            cpu, core, socket, node = map(int, row.split(","))
            if cpu in allowed:
                nodes.setdefault(node, {}).setdefault((socket, core), cpu)
    node, mapping = next((nd, c) for nd, c in sorted(nodes.items()) if len(c) >= 48)
    selected = sorted(mapping.items(), key=lambda i: i[1])[:48]
    samples = {str(t): [] for t in THREADS}
    for i, t in enumerate(RUN_ORDER):
        cpus = ",".join(str(c) for _, c in selected[:t])
        cmd = ["numactl", f"--physcpubind={cpus}", f"--membind={node}", sys.executable, __file__, "sample", str(t), str(i)]
        r = subprocess.run(cmd, text=True, capture_output=True)
        if r.returncode:
            raise SystemExit(r.stdout + r.stderr)
        row = json.loads(r.stdout.splitlines()[-1]); print(json.dumps(row), flush=True)
        samples[str(t)].append(row)
        Path("partial.json").write_text(json.dumps(samples, indent=2))
    ident = {(r["ndof"], r["nfree"], r["stored_entries"]) for rs in samples.values() for r in rs}
    assert len(ident) == 1
    xs = {f"{t}_{i}": np.load(f"x_{t}_{i}.npy") for i, t in enumerate(RUN_ORDER)}
    ref = xs[f"1_{RUN_ORDER.index(1)}"]
    thread_agreement = {k: float(np.linalg.norm(v - ref)/np.linalg.norm(ref)) for k, v in xs.items()}
    keys = ("assembly_s", "factorization_s", "solve_s", "total_s")
    med = {t: {k: statistics.median(r[k] for r in rs) for k in keys} for t, rs in samples.items()}
    cpus48 = ",".join(str(c) for _, c in selected)
    refs = subprocess.run(["numactl", f"--physcpubind={cpus48}", f"--membind={node}", sys.executable, __file__, "reference"],
                          text=True, capture_output=True)
    if refs.returncode:
        raise SystemExit(refs.stdout + refs.stderr)
    reference = json.loads(refs.stdout.splitlines()[-1])
    # Gate (policy): every true relative residual <= LIMIT. No approved tolerance
    # exists for agreement with SciPy SuperLU or across thread counts, so those
    # differences are recorded, not gated, and the report must not claim agreement.
    ind = reference["independent_solver"]
    residuals = [r["true_relative_residual"] for rs in samples.values() for r in rs]
    residuals += [r["true_relative_residual"] for r in reference["analytic"]["rows"]]
    residuals += [ind["scipy_true_relative_residual"], ind["ngsolve_true_relative_residual"]]
    residual_pass = all(math.isfinite(v) and v <= LIMIT for v in residuals)
    gates = dict(
        residual=dict(limit=LIMIT, max=max(residuals), passed=residual_pass),
        reference_agreement=dict(status="recorded_not_gated", approved_tolerance=None, agreement_claimed=False,
                                 rel_diff_ngsolve_vs_scipy=ind["rel_diff_ngsolve_vs_scipy"],
                                 max_thread_rel_diff_vs_1thread=max(thread_agreement.values())))
    out = dict(schema="ngsolve-fork-type1-validation/2", kind="diagnostic timing; residual gate only; reference agreement recorded, not gated",
        job=os.environ.get("PBS_JOBID"), queue="t1-single", pbs_allowed_cpus=len(allowed), numa_node=node,
        memory_policy=f"numactl --membind={node}", selected_physical_cores=[dict(socket=s, core=c, cpu=cpu) for (s, c), cpu in selected],
        source=json.loads(Path("validated-wheels.json").read_text())["source"],
        wheels=json.loads(Path("validated-wheels.json").read_text())["wheels"],
        problem=f"3D H1 P2 Poisson+mass n={N_PERF}, dirichlet left|bottom, rhs sin(3x), sparsecholesky",
        timing="total = mesh + assembly + factorization + first solve; residual check excluded; MKL_NUM_THREADS=1",
        run_order=RUN_ORDER, samples=samples, medians_s=med,
        speedup_vs_1={t: {k: med["1"][k]/med[t][k] for k in keys} for t in med},
        thread_agreement_rel_diff_vs_1thread=thread_agreement, reference=reference, gates=gates,
        pass_=residual_pass)
    for p in Path(".").glob("x_*.npy"):
        p.unlink()
    Path("result.json").write_text(json.dumps(out, indent=2) + "\n")
    if not residual_pass:
        raise SystemExit(f"residual gate failed: max {gates['residual']['max']} > {LIMIT}")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "sample":
        sample(int(sys.argv[2]), sys.argv[3])
    elif len(sys.argv) == 2 and sys.argv[1] == "reference":
        check_prefix()
        print(json.dumps(dict(analytic=analytic_reference(), independent_solver=scipy_reference())))
    else:
        run()
