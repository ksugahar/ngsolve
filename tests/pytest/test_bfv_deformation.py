"""BoundaryFromVolumeCF under an integration-local deformation.

ds(deformation=gf) maps the boundary points and the measure; the volume field
read through BoundaryFromVolumeCF must be evaluated on the same deformed map,
as dx(deformation=gf) does for volume integrals.  A = x transported by
X = x + 0.1 x^2 has dA/dX = 1 / sqrt(1 + 0.4 X).
"""
import pytest
from netgen.occ import Box, OCCGeometry, Pnt, Rectangle
from ngsolve import (CF, H1, BoundaryFromVolumeCF, GridFunction, Integrate, Mesh,
                     VectorH1, ds, dx, grad, sqrt, x)


def _mesh(dim):
    if dim == 2:
        geo = OCCGeometry(Rectangle(1, 1).Face().Move((1, 0, 0)), dim=2)
        return Mesh(geo.GenerateMesh(maxh=0.4))
    return Mesh(OCCGeometry(Box(Pnt(1, 0, 0), Pnt(2, 1, 1))).GenerateMesh(maxh=0.5))


def _setup(dim, order, curve):
    mesh = _mesh(dim)
    if curve:
        mesh.Curve(2)
    gf = GridFunction(H1(mesh, order=order))
    gf.Set(x)
    disp = GridFunction(VectorH1(mesh, order=2))
    disp.Set(CF((0.1 * x * x,) + (0,) * (dim - 1)))
    return mesh, gf, disp


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("order", [1, 2])
def test_bfv_follows_integration_deformation(dim, order):
    # A = x is exact for H1 order >= 1 and the displacement is exact in
    # VectorH1(order=2), so the transported value and gradient are exact too
    mesh, gf, disp = _setup(dim, order, curve=(order == 2))
    dsd = ds(deformation=disp)
    norm = Integrate(CF(1) * dsd, mesh)
    x_ref = (sqrt(1 + 0.4 * x) - 1) / 0.2          # reference x of the deformed point X
    dadx = 1 / sqrt(1 + 0.4 * x)
    # the volume route, as the reference behaviour
    assert Integrate((grad(gf)[0] - dadx) ** 2 * dx(deformation=disp), mesh) < 1e-20 * norm
    value_err = Integrate((BoundaryFromVolumeCF(gf) - x_ref) ** 2 * dsd, mesh)
    grad_err = Integrate((BoundaryFromVolumeCF(grad(gf)[0]) - dadx) ** 2 * dsd, mesh)
    assert value_err < 1e-20 * norm, value_err
    assert grad_err < 1e-20 * norm, grad_err


@pytest.mark.parametrize("dim", [2, 3])
def test_bfv_undeformed_paths_unchanged(dim):
    mesh, gf, disp = _setup(dim, 2, curve=True)
    zero = GridFunction(VectorH1(mesh, order=2))
    for cf in (gf, grad(gf)[0]):
        b = BoundaryFromVolumeCF(cf)
        plain = Integrate(b * ds, mesh)
        assert Integrate(b * ds(deformation=zero), mesh) == plain
    # a mesh-level deformation is already in the volume map and is not applied twice
    mesh.SetDeformation(disp)
    try:
        bnd = Integrate((BoundaryFromVolumeCF(grad(gf)[0]) - 1 / sqrt(1 + 0.4 * x)) ** 2 * ds, mesh)
        vol = Integrate((grad(gf)[0] - 1 / sqrt(1 + 0.4 * x)) ** 2 * dx, mesh)
    finally:
        mesh.UnsetDeformation()
    assert bnd < 1e-20 and vol < 1e-20, (bnd, vol)
