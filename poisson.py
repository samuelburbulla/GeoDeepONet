"""Physics-informed GeoDeepONet for the Poisson problem (2D)."""
import torch
import numpy as np
import geodeeponet as gdn

# Hyperparameters
num_collocation_points = 2**2
branch_width = 1
trunk_width = 64
num_loss_points = 10**2

# Domain
geom = gdn.geometry.UnitSquare()
collocation_points = geom.uniform_points(num_collocation_points)

# Transformations
phis = [
    gdn.transformation.Affine(A=np.eye(2), b=np.ones(2)),
]

# Boundary condition
bc = gdn.bc.UnitCubeDirichletBC({"left": 0, "right": 0, "top": 0, "bottom": 0})

# Define PDE
pde = gdn.pde.Poisson(bc, source=1)

# Setup DeepONet
model = gdn.deeponet.GeoDeepONet(
    branch_width=branch_width,
    trunk_width=trunk_width,
    num_collocation_points=len(collocation_points),
    dimension=geom.dimension
)

# Setup loss points and bc
loss_points = geom.uniform_points(num_loss_points)
pde.setup_bc(loss_points)

# Train model
gdn.train.train_model(
    geom, model, collocation_points, phis, pde, loss_points
)

# Plot solution for a sample transformation
phi = gdn.transformation.Affine(A=np.eye(2), b=np.zeros(2))
loss_points = torch.stack([loss_points])
outputs = model((phi.inv(collocation_points), loss_points))
loss, bc = pde(outputs, loss_points)
print(f"Sample transformation   Loss: {loss:.3e}  BC: {bc:.3e}")
gdn.plot.plot_solution(geom, model, collocation_points, phi, writer="show")
