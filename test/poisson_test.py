"""Test physics-informed GeoDeepONet using an analytical solution of the Poisson problem."""
import geodeeponet as gdn
from torch import sin, pi
import numpy as np

# Parameters
dim = 2
num_collocation_points = 2**dim
branch_width = 1
trunk_width = 128

# Domain
geom = gdn.geometry.UnitCube(dim)
collocation_points = geom.uniform_points(num_collocation_points)

# Transformation
phi = gdn.transformation.Affine(
    A=np.array([[3., 0.], [0., 1.]]),
    b=np.array([0., 0.])
)
# phi = gdn.transformation.PolarCoordinates(
#     r_min=0.5, r_max=1.0,
#     theta_min=0.0, theta_max=.5*pi,
# )

# Exact solution
def exact_solution(x):
    return sin(2 * pi * x[..., 0]) * sin(2 * pi * x[..., 1]) / (8 * pi**2)

# Source
def source(x):
    return sin(2 * pi * x[..., 0]) * sin(2 * pi * x[..., 1])

# Boundary condition
bc = gdn.bc.UnitCubeDirichletBC({
    w: (lambda x: exact_solution(phi.inv(x))) for w in ["left", "right", "top", "bottom"]
})

# Define PDE
pde = gdn.pde.Poisson(bc, lambda x: source(phi.inv(x)))

# Setup DeepONet
model = gdn.deeponet.GeoDeepONet(
    branch_width=branch_width,
    trunk_width=trunk_width,
    num_collocation_points=len(collocation_points),
    dimension=geom.dim,
    outputs=pde.outputs,
)

# Train model
gdn.train.train_model(
    geom, model, collocation_points, [phi], pde
)

# Plot solution for one sample transformation
gdn.plot.plot_solution(geom, model, collocation_points, phi, writer="show")

# Compute error
l2 = gdn.error.l2(geom, model, collocation_points, phi, exact_solution)
print(f"L2-Error: {l2:.3e}")
assert l2 < 1e-3
