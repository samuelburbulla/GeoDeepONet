"""Physics-informed GeoDeepONet for the Poisson problem (2D)."""
import torch
import numpy as np
import geodeeponet as gdn

# Hyperparameters
num_collocation_points = 2**2
branch_width = 64
trunk_width = 128
num_loss_points = 10**2

# Domain
geom = gdn.geometry.UnitSquare()
collocation_points = geom.uniform_points(num_collocation_points)

# Transformations
phis = []

num_affines = 10
for _ in range(num_affines):
    phis += [gdn.transformation.Affine(dim=geom.dimension)]

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
phi = gdn.transformation.Affine(dim=geom.dimension)
phi_loss_points = torch.stack([phi(loss_points)])
phi_collocation_points = phi(collocation_points)
outputs = model((phi_collocation_points, phi_loss_points))
loss, bc = pde(outputs, phi_loss_points)
print(f"Sample transformation   Loss: {loss:.3e}  BC: {bc:.3e}")

gdn.plot.plot_solution(geom, model, collocation_points, phi)
