"""Physics-informed GeoDeepONet."""
import geodeeponet as gdn
import numpy as np

# Hyperparameters
dim = 2
num_collocation_points = 2**dim

# Reference domain
geom = gdn.geometry.UnitCube(dim)
collocation_points = geom.uniform_points(num_collocation_points)

# Transformations
phis = [gdn.transformation.PolarCoordinates() for _ in range(1)]

# Define PDE
bc = gdn.bc.UnitCubeZeroDirichletBC()
pde = gdn.pde.Poisson(bc, 1)

# Setup DeepONet
model = gdn.deeponet.GeoDeepONet(
    branch_width=1,
    trunk_width=32,
    num_collocation_points=len(collocation_points),
    dimension=geom.dim,
    outputs=pde.outputs,
)

# Train model
gdn.train.train_model(geom, model, collocation_points, phis, pde)

# Plot solution
phi = phis[0]
gdn.plot.plot_solution(geom, model, collocation_points, phi, writer="show")
