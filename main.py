"""Physics-informed GeoDeepONet for the linear elasticity."""
import geodeeponet as gdn
import torch

# Hyperparameters
dim = 2
num_collocation_points = 2**dim
branch_width = 1
trunk_width = 512

# Domain
geom = gdn.geometry.UnitCube(dim)
collocation_points = geom.uniform_points(num_collocation_points)

# Transformations
import numpy as np
phi = gdn.transformation.Affine(
    A=np.array([[1., 0.], [0., 1.]]),
    b=np.zeros(dim)
)

# Boundary condition
bc = gdn.bc.UnitCubeDirichletBC({"left": [0.]*dim, "right": [0.1]*dim})

# Define PDE
pde = gdn.pde.Elasticity(bc, dim, gravity=lambda x: torch.zeros_like(x))

# Setup DeepONet
model = gdn.deeponet.GeoDeepONet(
    branch_width=branch_width,
    trunk_width=trunk_width,
    num_collocation_points=len(collocation_points),
    dimension=geom.dim,
    outputs=pde.outputs,
)

# Train model
gdn.train.train_model(geom, model, collocation_points, [phi], pde)

# Plot solution
gdn.plot.plot_solution(geom, model, collocation_points, phi, writer="show")
