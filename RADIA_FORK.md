# Radia fork of NGSolve

This fork follows upstream NGSolve as closely as possible. It is not a
development line: it is upstream plus the bug fixes Radia needs plus the MATLAB (MEX
SDK) build. A fix is offered upstream only with text the maintainer of this
fork has approved, and it is dropped once an equivalent upstream change is
verified. Its companion is the netgen fork
(`ksugahar/netgen`), which `external_dependencies/netgen` points at.

## Branch and versions

- One long-lived branch: `master` = latest upstream `master` + fork patches.
- Each maintenance round tags netgen and ngsolve with the same tag and
  publishes wheels and SDKs on the GitHub Releases of both forks.
- Radia pins the exact published wheel versions (PEP 440, for example
  `6.2.2608.post100.dev0+radia.1`).

## Lanes

| Lane | CPU | GPU | Python | BLAS/LAPACK |
|---|---|---|---|---|
| pybind11 wheel | yes | yes | one wheel per Python version | MKL on x86, NVPL on aarch64 |
| MEX SDK | yes | yes | none | MATLAB `libmwblas`/`libmwlapack` (ILP64) |

The MEX GPU lane is a MEX + CUDA build: MSVC builds the MEX
(`matlab_add_mex`), which links `ngscudalib` and the CUDA runtime; `mexcuda`
is not used. A MEX registers the GPU backend itself with `InitCUDA`,
`InitCudaDevice` and `InitCuLinalg`.

## Fork patches

Source/build patches, listed by `git cherry -v upstream/master master`
(maintenance commits such as this file are not listed); each is a small
labeled commit.

| Patch | Purpose |
|---|---|
| Build: accept a PEP 440 local segment in the version tag | `+radia.N` versions |
| BoundaryFromVolumeCF: apply the integration-local deformation | bug fix |
| Build without Python: guard pybind uses and an unterminated comment token | MEX SDK |
| LAPACK: 64-bit integers on every platform for ILP64 libraries | MATLAB LAPACK |
| Review fix: fix the LAPACK integer width before the search | ILP64 search |
| Review fix: installed ngbla carries ILP64 LAPACK | complex LAPACK link |
| Windows: build ngscuda as a usable DLL for C++/MEX consumers | MEX + CUDA |
| Use the Radia netgen fork as the netgen submodule | fork pairing |

## Maintenance round

```
pwsh -File tools/radia_fork/update.ps1    -Netgen <netgen> -NGSolve <ngsolve>
pwsh -File tools/radia_fork/build_sdk.ps1 -Netgen <netgen> -NGSolve <ngsolve> -Job <scratch> -Out <evidence> -Matlab <MATLAB root> -Cuda
pwsh -File tools/radia_fork/check_sdk.ps1 -Zip <evidence>/ngsolve-sdk-cuda-win64.zip -Job <scratch> -Out <evidence> -Matlab <MATLAB root> -Cuda
```

`update.ps1` stops at a merge conflict and never pushes or tags.
`check_sdk.ps1` never creates a CUDA context; GPU execution is checked
separately on a GPU host. Every round is reviewed before it is pushed,
tagged or published.
