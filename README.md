# NGSolve (ksugahar fork)

Multi-purpose finite element library

Find the Open Source Community on https://ngsolve.org

Support & Services: https://cerbsim.com

## Fork Features

This fork includes additional features not yet in the upstream NGSolve:

### 1. SetGeomInfo API (PR #232)

Python API for high-order curving of externally imported meshes.
Enables accurate curving of meshes imported from external mesh generators (e.g., Coreform Cubit).

```python
from netgen.occ import *
from ngsolve import *

# Set UV parameters for curved surface elements
for el in mesh.Elements2D():
    for i, v in enumerate(el.vertices):
        el.SetGeomInfo(i, PointGeomInfo(u=..., v=...))

# Apply high-order curving
mesh.Curve(order=3)
```

### 2. Intel MKL Support

Build with Intel MKL for improved performance:
```bash
cmake .. -DUSE_MKL=ON
```

### 3. SparseSolv Integration

Integrated preconditioners and iterative solvers from [JP-MARs/SparseSolv](https://github.com/JP-MARs/SparseSolv).

#### Preconditioners

Use with NGSolve's built-in Krylov solvers:

- **ICPreconditioner**: Shifted Incomplete Cholesky
- **ILUPreconditioner**: Shifted Incomplete LU (for non-symmetric matrices)
- **SGSPreconditioner**: Symmetric Gauss-Seidel

```python
from ngsolve import *
from ngsolve.la import ICPreconditioner
from ngsolve.krylovspace import CGSolver

pre = ICPreconditioner(a.mat, freedofs=fes.FreeDofs(), shift=1.05)
pre.Update()
inv = CGSolver(a.mat, pre, printrates=True, tol=1e-10)
gfu.vec.data = inv * f.vec
```

#### Iterative Solvers (SparseSolvSolver)

Self-contained iterative solvers with **best-result tracking**: if the solver does
not converge, the best solution found during iteration is returned automatically.

Available methods:

| Method | Algorithm | Preconditioner |
|--------|-----------|----------------|
| `ICCG` | Conjugate Gradient | Incomplete Cholesky |
| `ICMRTR` | MRTR (residual minimization) | Incomplete Cholesky |
| `SGSMRTR` | MRTR with split formula | Built-in Symmetric Gauss-Seidel |
| `CG` | Conjugate Gradient | None |
| `MRTR` | MRTR | None |

**Usage as inverse operator** (drop-in replacement for NGSolve's CGSolver):

```python
from ngsolve import *
from ngsolve.la import SparseSolvSolver

fes = H1(mesh, order=2, dirichlet="left|right|top|bottom")
u, v = fes.TnT()
a = BilinearForm(fes)
a += grad(u)*grad(v)*dx
a.Assemble()

solver = SparseSolvSolver(a.mat, method="ICCG",
                          freedofs=fes.FreeDofs(), tol=1e-10)
gfu.vec.data = solver * f.vec
```

**Usage with detailed results:**

```python
solver = SparseSolvSolver(a.mat, method="ICCG",
                          freedofs=fes.FreeDofs(),
                          tol=1e-10, maxiter=1000,
                          save_residual_history=True)
result = solver.Solve(f.vec, gfu.vec)
print(f"Converged: {result.converged}")
print(f"Iterations: {result.iterations}")
print(f"Final residual: {result.final_residual}")
print(f"Residual history: {result.residual_history}")
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `method` | `"ICCG"` | Solver method |
| `freedofs` | `None` | BitArray for Dirichlet BCs |
| `tol` | `1e-10` | Relative convergence tolerance |
| `maxiter` | `1000` | Maximum iterations |
| `shift` | `1.05` | IC preconditioner shift parameter |
| `save_best_result` | `True` | Track best solution during iteration |
| `save_residual_history` | `False` | Record residual at each iteration |
| `printrates` | `False` | Print convergence info |

## Binary Releases

Pre-built binaries are available on GitHub Releases:
https://github.com/ksugahar/ngsolve/releases

| Version | Features |
|---------|----------|
| v6.2.2601-setgeominfo-mkl | SetGeomInfo API + Intel MKL + SparseSolv (Recommended) |
| v6.2.2601-setgeominfo | SetGeomInfo API + SparseSolv (OpenBLAS) |

## Building from Source

### Requirements

- CMake 3.16+
- C++17 compiler (MSVC, GCC, Clang)
- Python 3.x
- (Optional) Intel MKL

### Build Commands

```bash
git clone --recursive https://github.com/ksugahar/ngsolve.git
cd ngsolve
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

With Intel MKL:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_MKL=ON
```

## Related Repositories

- [ksugahar/netgen](https://github.com/ksugahar/netgen) - Netgen fork with SetGeomInfo API
- [JP-MARs/SparseSolv](https://github.com/JP-MARs/SparseSolv) - Original SparseSolv library

## Contributors

- NGSolve Team (upstream)
- Kengo Sugahara (fork maintainer)
- SparseSolv Contributors (Takahiro Sato, Shingo Hiruma, Tomonori Tsuburaya)

## License

See LICENSE file for details.
