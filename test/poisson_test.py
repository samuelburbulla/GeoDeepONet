"""Test physics-informed GeoDeepONet using an analytical solution of the Poisson problem."""
import geodeeponet as gdn
from torch import sin, pi

# Hyperparameters
dim = 2
num_collocation_points = 2**dim
branch_width = 1
trunk_width = 128
num_loss_points = 3**dim

# Domain
geom = gdn.geometry.UnitCube(dim)
collocation_points = geom.uniform_points(num_collocation_points)

# Transformation
phi = gdn.transformation.Identity()

# Boundary condition
bc = gdn.bc.UnitCubeDirichletBC({"left": 0, "right": 0, "top": 0, "bottom": 0})

# Source
def source(x):
    return sin(2 * pi * x[..., 0]) * sin(2 * pi * x[..., 1])

# Exact
def exact_solution(x):
    return source(x) / (8 * pi**2)

# Define PDE
pde = gdn.pde.Poisson(bc, source)

# Setup DeepONet
model = gdn.deeponet.GeoDeepONet(
    branch_width=branch_width,
    trunk_width=trunk_width,
    num_collocation_points=len(collocation_points),
    dimension=geom.dim,
    outputs=pde.outputs,
)

# Setup loss points and bc
loss_points = geom.uniform_points(num_loss_points)
pde.setup_bc(loss_points)

# Train model
gdn.train.train_model(
    geom, model, collocation_points, [phi], pde, loss_points
)

# Plot solution for one sample transformation
gdn.plot.plot_solution(geom, model, collocation_points, phi, writer="show")

# Compute error
l2 = gdn.error.l2(geom, model, collocation_points, phi, exact_solution)
print(f"L2-Error: {l2:.3e}")

assert l2 < 1e-6
