# NGSolve Fork: SetGeomInfo API + MKL Windows Builds

This is a fork of [NGSolve](https://ngsolve.org/) that provides:

1. **SetGeomInfo API** -- `Element2D.SetGeomInfo(vertex_index, u, v)` for external mesh curving
2. **Pre-built Windows binaries** with Intel MKL

## Why This Fork?

NGSolve's `mesh.Curve(order)` requires UV parameters linking each surface element to the OCC geometry. Netgen sets these automatically for meshes it generates, but **external meshes** (from Cubit, Gmsh, etc.) lack this information.

The `SetGeomInfo` API (proposed in [NGSolve/netgen PR #232](https://github.com/NGSolve/netgen/pull/232)) allows setting UV parameters programmatically, enabling high-order curving for externally imported meshes.

### Without SetGeomInfo

```python
mesh = import_cubit_mesh("model.msh")
mesh.Curve(3)  # FAILS -- no UV parameters for surface elements
```

### With SetGeomInfo

```python
mesh = import_cubit_mesh("model.msh")
for el in mesh.Elements2D():
    for i, v in enumerate(el.vertices):
        u, v = compute_uv_on_occ_surface(v)
        el.SetGeomInfo(i, u, v)
mesh.Curve(3)  # Works correctly
```

## Accuracy

Tested with [Coreform Cubit Mesh Export](https://github.com/ksugahar/Coreform_Cubit_Mesh_Export):

| Geometry | Curve(2) Error | Curve(3) Error |
|----------|----------------|----------------|
| Complex (Boolean ops) | 0.0021% | 0.0004% |
| Cylinder | 0.0027% | 0.0006% |
| Sphere | 0.0027% | 0.0004% |
| Torus | 0.0010% | 0.0003% |

Achieves Netgen-native accuracy for externally imported hex/tet meshes.

## Downloads

Pre-built Windows binaries (Intel MKL):

- [v6.2.2601-setgeominfo-mkl](https://github.com/ksugahar/ngsolve/releases/tag/v6.2.2601-setgeominfo-mkl) -- NGSolve 6.2.2601 + SetGeomInfo + MKL

### Installation

```python
import sys
sys.path.insert(0, "path/to/extracted/Lib/site-packages")
import ngsolve
```

## Branch Structure

| Branch | Description |
|--------|-------------|
| `master` | Synced with [NGSolve/ngsolve](https://github.com/NGSolve/ngsolve) upstream |
| `feature/setgeominfo` | `master` + netgen submodule pointing to [ksugahar/netgen:add-setgeominfo-api](https://github.com/ksugahar/netgen) |

## Building from Source

Requirements: Visual Studio 2022, Intel oneAPI Base Toolkit (MKL), CMake

```powershell
git clone --recurse-submodules https://github.com/ksugahar/ngsolve.git
cd ngsolve
git checkout feature/setgeominfo
git submodule update --init --recursive
# Follow standard NGSolve build instructions with -DUSE_MKL=ON
```

## Related Projects

- [NGSolve](https://ngsolve.org/) -- Upstream project
- [Coreform Cubit Mesh Export](https://github.com/ksugahar/Coreform_Cubit_Mesh_Export) -- Cubit to NGSolve mesh pipeline using SetGeomInfo
- [Radia](https://github.com/ksugahar/Radia) -- Magnetostatic BEM solver with NGSolve integration

## License

Same as NGSolve (LGPL-2.1). See [LICENSE](LICENSE) for details.
