import json
import sys
import math
import ngsolve
from ngsolve.ngstd import GetGPUDevice
from ngsolve.ngscuda import CreateDevMatrix
from netgen.occ import unit_square
import numpy as np

device = GetGPUDevice()
mesh = ngsolve.Mesh(unit_square.GenerateMesh(maxh=0.1))
space = ngsolve.H1(mesh, order=3)
u, v = space.TnT()
matrix = ngsolve.BilinearForm((ngsolve.grad(u)*ngsolve.grad(v) + u*v)*ngsolve.dx).Assemble().mat
x = matrix.CreateVector()
x.FV().NumPy()[:] = np.random.default_rng(3).random(space.ndof)
reference = matrix.CreateVector()
matrix.Mult(x, reference)
gpu_matrix = CreateDevMatrix(matrix)
gpu_x = gpu_matrix.CreateVector()
gpu_y = gpu_matrix.CreateVector()
gpu_x.data = x
gpu_matrix.Mult(gpu_x, gpu_y)
expected = reference.FV().NumPy()
error = float(np.linalg.norm(gpu_y.FV().NumPy() - expected) / np.linalg.norm(expected))
report = dict(device=device.name, ngsolve=ngsolve.__file__, ndof=space.ndof, spmv_rel_err=error)
# ngscuda prints a device query on stdout: write the record to the file argument
json.dump(report, open(sys.argv[1], "w"))
assert math.isfinite(error) and error < 1e-12
