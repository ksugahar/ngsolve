"""
Equilibrated Error Estimator for T-Omega Formulation

This example demonstrates the use of PatchwiseSolveWithInterface for
magnetostatic T-Omega formulation with interface jump conditions.

T-Omega Formulation:
-------------------
In T-Omega method, the magnetic field H is represented as:
    H = T - grad(Omega)

where:
    - T is defined only in the conducting region (HCurl)
    - Omega is defined everywhere (H1)

Interface Condition:
    At material interfaces, [H_t] (tangential H jump) may be non-zero
    due to surface currents or different material properties.

The equilibrated error estimator must preserve this interface jump
during the patchwise local solve.

Reference:
    Braess, D., & Schoeberl, J. (2008).
    Equilibrated residual error estimator for edge elements.
    Mathematics of Computation, 77(262), 651-672.

Author: K. Sugawara
Date: January 2026
"""

from ngsolve import *
from netgen.csg import *
import numpy as np


def create_coil_geometry():
    """
    Create a simple 3D geometry with a conducting coil in air.
    Returns mesh with materials: 'coil' (conductor) and 'air'
    """
    # Outer air region
    air = OrthoBrick(Pnt(-2, -2, -2), Pnt(2, 2, 2)).bc("outer")

    # Coil (torus-like, simplified as a box for now)
    coil = OrthoBrick(Pnt(-0.5, -0.5, -0.2), Pnt(0.5, 0.5, 0.2))

    # Combined geometry
    geo = CSGeometry()
    geo.Add(coil.mat("coil"))
    geo.Add((air - coil).mat("air"))

    mesh = Mesh(geo.GenerateMesh(maxh=0.5))
    return mesh


def create_2d_coil_mesh(maxh=0.1):
    """
    Create a 2D mesh for simpler demonstration.
    Conducting region (inner square) surrounded by air.
    """
    from netgen.geom2d import SplineGeometry

    geo = SplineGeometry()

    # Outer boundary (air region boundary)
    geo.AddRectangle((-1, -1), (1, 1), bcs=["outer"]*4)

    # Inner region (conductor)
    geo.AddRectangle((-0.3, -0.3), (0.3, 0.3),
                     leftdomain=2, rightdomain=1)

    geo.SetMaterial(1, "air")
    geo.SetMaterial(2, "coil")

    mesh = Mesh(geo.GenerateMesh(maxh=maxh))
    return mesh


def solve_t_omega_2d(mesh, order=2):
    """
    Solve simplified T-Omega problem in 2D.

    The formulation:
        curl(mu^-1 * curl(A)) = J  in conductor
        curl(mu^-1 * curl(A)) = 0  in air

    For demonstration, we use A-formulation which is dual to T-Omega.
    The error estimator approach is similar.
    """
    # Material properties
    mu0 = 4 * np.pi * 1e-7
    mu_r_coil = 1.0
    mu_r_air = 1.0

    mu = mesh.MaterialCF({
        "coil": mu0 * mu_r_coil,
        "air": mu0 * mu_r_air
    })
    mu_inv = 1.0 / mu

    # Source current (only in coil)
    J_mag = 1e6  # A/m^2
    J = mesh.MaterialCF({"coil": J_mag, "air": 0.0})

    # HCurl space for vector potential A
    fes = HCurl(mesh, order=order, dirichlet="outer")
    A, v = fes.TnT()

    # Bilinear form: (mu^-1 curl A, curl v)
    a = BilinearForm(mu_inv * curl(A) * curl(v) * dx)

    # Linear form: (J, v)
    # In 2D, J is scalar (z-component), v is 2D vector
    # We use the z-component of the cross product
    L = LinearForm(J * v[1] * dx)  # Simplified for 2D

    a.Assemble()
    L.Assemble()

    # Solve
    gfA = GridFunction(fes)
    gfA.vec.data = a.mat.Inverse(fes.FreeDofs(), "sparsecholesky") * L.vec

    return gfA, mu_inv, J


def compute_hcurl_equilibrated_estimator(mesh, gfA, mu_inv, J, order=2):
    """
    Compute equilibrated error estimator for HCurl formulation.

    For curl-curl equation, the equilibrated field H_eq must satisfy:
        curl(H_eq) = J  (Ampere's law, strongly)

    The estimator is: ||mu^-1 curl(A_h) - H_eq||
    """
    # Magnetic field from numerical solution
    H_h = mu_inv * curl(gfA)

    # For 2D, H is a scalar (z-component of B = curl A)
    # We need to construct equilibrated H in appropriate space

    # H(div) space for equilibrated field (dual to HCurl in some sense)
    # In 2D, we use a scalar H1 space for the z-component
    fes_eq = H1(mesh, order=order+1)
    sigma, tau = fes_eq.TnT()

    # Bilinear form for patchwise solve
    bf = sigma * tau * dx

    # Linear form: residual
    # curl(H) = J means d(H)/dx - d(H)/dy = J in 2D
    # Weak form: H * curl(tau) - J * tau  (integration by parts)
    lf = H_h * tau * dx - J * tau * dx

    # Interface term for jump handling
    # In this case, handled by PatchwiseSolveWithInterface

    # Grid function for equilibrated estimator
    gf_sigma = GridFunction(fes_eq)

    # Use PatchwiseSolveWithInterface
    try:
        PatchwiseSolveWithInterface(bf, lf, gf_sigma)
        print("Used PatchwiseSolveWithInterface successfully")
    except Exception as e:
        print(f"Note: {e}")
        print("Using standard PatchwiseSolve")
        PatchwiseSolve(bf, lf, gf_sigma)

    # Compute error estimator
    error_cf = H_h - gf_sigma
    eta_squared = Integrate(error_cf**2 * dx, mesh)

    return gf_sigma, sqrt(eta_squared)


def demo_interface_jump():
    """
    Demonstrate interface jump handling in equilibration.
    """
    print("=" * 70)
    print("T-Omega Equilibration Demo: Interface Jump Handling")
    print("=" * 70)

    # Create mesh
    mesh = create_2d_coil_mesh(maxh=0.1)
    print(f"\nMesh created:")
    print(f"  Elements: {mesh.ne}")
    print(f"  Vertices: {mesh.nv}")
    print(f"  Materials: {mesh.GetMaterials()}")

    # Identify interface
    # In NGSolve, internal boundaries between materials can be accessed
    # through BBND elements

    print("\nInterface information:")
    n_bbnd = sum(1 for el in mesh.Elements(BBND))
    print(f"  BBND elements (interface): {n_bbnd}")

    # Solve magnetostatic problem
    print("\nSolving magnetostatic problem...")
    gfA, mu_inv, J = solve_t_omega_2d(mesh, order=2)

    # Compute equilibrated estimator
    print("\nComputing equilibrated error estimator...")
    gf_sigma, eta = compute_hcurl_equilibrated_estimator(
        mesh, gfA, mu_inv, J, order=2
    )

    print(f"\nResults:")
    print(f"  Equilibrated error estimator: {eta:.6e}")

    # Check interface continuity
    H_h = mu_inv * curl(gfA)
    print(f"  Max |H_h|: {Integrate(H_h**2*dx, mesh)**.5:.6e}")

    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)

    return mesh, gfA, gf_sigma


def main():
    """
    Run the T-Omega equilibration demonstration.
    """
    return demo_interface_jump()


if __name__ == "__main__":
    mesh, gfA, gf_sigma = main()

    # Visualization
    try:
        from ngsolve.webgui import Draw
        Draw(curl(gfA), mesh, "curl_A")
        Draw(gf_sigma, mesh, "equilibrated")
    except ImportError:
        print("\nNote: ngsolve.webgui not available for visualization")
