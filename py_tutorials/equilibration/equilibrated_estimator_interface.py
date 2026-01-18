"""
Equilibrated Error Estimator with Interface Support

This example demonstrates the use of PatchwiseSolveWithInterface for
computing equilibrated error estimators in multi-material problems.

The example solves a Poisson problem with discontinuous coefficients
and computes the equilibrated flux error estimator that correctly
handles the interface jump conditions.

Theory:
-------
The equilibrated error estimator is based on the Prager-Synge theorem:
    ||grad(u - u_h)||^2 + ||grad(u) - sigma||^2 = ||grad(u_h) - sigma||^2

where sigma is the equilibrated flux satisfying div(sigma) + f = 0.

For multi-material problems, the interface jump condition [sigma * n] = 0
must be preserved during the patchwise solve.

Reference:
    Braess, D., & Schoeberl, J. (2008).
    Equilibrated residual error estimator for edge elements.
    Mathematics of Computation, 77(262), 651-672.

Author: K. Sugawara
Date: January 2026
"""

from ngsolve import *
from netgen.geom2d import SplineGeometry
import numpy as np


def create_two_material_mesh(maxh=0.1):
    """
    Create a mesh with two materials separated by an interface at x=0.5
    """
    geo = SplineGeometry()

    # Outer boundary
    geo.AddRectangle((0, 0), (1, 1), bcs=["bottom", "right", "top", "left"])

    # Interface line at x = 0.5
    geo.AddRectangle((0, 0), (0.5, 1), leftdomain=1, rightdomain=0)
    geo.AddRectangle((0.5, 0), (1, 1), leftdomain=2, rightdomain=0)

    # Set material names
    geo.SetMaterial(1, "mat1")
    geo.SetMaterial(2, "mat2")

    mesh = Mesh(geo.GenerateMesh(maxh=maxh))
    return mesh


def solve_poisson_two_materials(mesh, order=2):
    """
    Solve Poisson problem with discontinuous coefficient:
        -div(k * grad(u)) = f
    where k = k1 in mat1 and k = k2 in mat2
    """
    # Material coefficients
    k1, k2 = 1.0, 10.0
    k = mesh.MaterialCF({"mat1": k1, "mat2": k2})

    # Source term
    f = 1.0

    # H1 space with Dirichlet BC
    fes = H1(mesh, order=order, dirichlet="bottom|top|left|right")

    u, v = fes.TnT()

    # Bilinear and linear forms
    a = BilinearForm(k * grad(u) * grad(v) * dx)
    L = LinearForm(f * v * dx)

    a.Assemble()
    L.Assemble()

    # Solve
    gfu = GridFunction(fes)
    gfu.vec.data = a.mat.Inverse(fes.FreeDofs()) * L.vec

    return gfu, k, f


def compute_equilibrated_estimator_with_interface(mesh, gfu, k, f, order=2):
    """
    Compute equilibrated flux error estimator using PatchwiseSolveWithInterface.

    The equilibrated flux sigma is constructed to satisfy:
        div(sigma) + f = 0  (equilibration condition)

    For multi-material problems, we also enforce continuity of normal flux
    across interfaces: [sigma * n] = 0
    """
    # H(div) space for equilibrated flux
    fes_hdiv = HDiv(mesh, order=order)
    sigma, tau = fes_hdiv.TnT()

    # Residual: r = f + div(k * grad(u_h))
    # For patchwise solve, we need the weak form of the residual
    flux_h = k * grad(gfu)

    # Bilinear form for patchwise solve (H(div) inner product)
    bf = InnerProduct(sigma, tau) * dx

    # Linear form: residual weighted by test function
    # The residual is: f * v - k * grad(u_h) * grad(v)
    # In weak form for H(div): f * div(tau) + flux_h * tau
    lf = f * div(tau) * dx + flux_h * tau * dx

    # Interface term: enforce [sigma * n] continuity
    # This is handled automatically by H(div) space, but we can add
    # penalty terms for better conditioning if needed
    interface = mesh.Boundaries(".*")  # All boundaries including internal

    # Grid function for equilibrated flux
    gf_sigma = GridFunction(fes_hdiv)

    # Use PatchwiseSolveWithInterface to handle interface elements
    try:
        PatchwiseSolveWithInterface(bf, lf, gf_sigma)
        print("Using PatchwiseSolveWithInterface (with BBND support)")
    except Exception as e:
        print(f"Falling back to PatchwiseSolve: {e}")
        PatchwiseSolve(bf, lf, gf_sigma)

    # Compute error estimator: ||flux_h - sigma||
    error_cf = flux_h - gf_sigma
    error_estimator = Integrate(InnerProduct(error_cf, error_cf) * dx, mesh)

    return gf_sigma, sqrt(error_estimator)


def compute_element_errors(mesh, gfu, k, gf_sigma):
    """
    Compute element-wise error indicators for adaptive mesh refinement.
    """
    flux_h = k * grad(gfu)
    error_cf = flux_h - gf_sigma

    # Element-wise L2 norm squared
    element_errors = []
    for el in mesh.Elements():
        err_sq = Integrate(InnerProduct(error_cf, error_cf) * dx,
                           mesh, element_restriction=el)
        element_errors.append(sqrt(err_sq))

    return np.array(element_errors)


def main():
    """
    Main demonstration of equilibrated error estimator with interface.
    """
    print("=" * 60)
    print("Equilibrated Error Estimator with Interface Support")
    print("=" * 60)

    # Create mesh
    mesh = create_two_material_mesh(maxh=0.1)
    print(f"\nMesh: {mesh.ne} elements, {mesh.nv} vertices")
    print(f"Materials: {mesh.GetMaterials()}")
    print(f"Boundaries: {mesh.GetBoundaries()}")

    # Solve Poisson problem
    print("\nSolving Poisson problem with discontinuous coefficient...")
    gfu, k, f = solve_poisson_two_materials(mesh, order=2)

    # Compute equilibrated estimator
    print("\nComputing equilibrated error estimator...")
    gf_sigma, eta = compute_equilibrated_estimator_with_interface(
        mesh, gfu, k, f, order=2
    )

    print(f"\nEquilibrated error estimator: {eta:.6e}")

    # Compute element errors for potential AMR
    element_errors = compute_element_errors(mesh, gfu, k, gf_sigma)
    print(f"Max element error: {max(element_errors):.6e}")
    print(f"Min element error: {min(element_errors):.6e}")

    # Verify equilibration (div(sigma) + f should be small)
    div_sigma = div(gf_sigma)
    equilibration_residual = Integrate((div_sigma + f)**2 * dx, mesh)
    print(f"Equilibration residual ||div(sigma) + f||: {sqrt(equilibration_residual):.6e}")

    print("\n" + "=" * 60)
    print("Done!")

    return mesh, gfu, gf_sigma, eta


if __name__ == "__main__":
    mesh, gfu, gf_sigma, eta = main()

    # Optional: visualization with webgui
    try:
        from ngsolve.webgui import Draw
        Draw(gfu, mesh, "solution")
        Draw(gf_sigma, mesh, "equilibrated_flux")
    except ImportError:
        print("\nNote: Install ngsolve.webgui for visualization")
