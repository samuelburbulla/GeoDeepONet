# # Physics-informed GeoDeepONet
#
# Consider the Poisson problem
#
# \begin{align}
# -\Delta u &= 1, \quad \text{in } \Omega_\phi, \\
# u &= 0, \quad \text{on } \partial \Omega_\phi,
# \end{align}
#
# where domain $\Omega_\phi = \phi(\Omega)$ is parameterised by
# $\phi: \Omega \to \mathbb{R}^d$.

import numpy as np
import geodeeponet as gdn

# Hyperparameters
num_collocation_points = 4
branch_width = 1
trunk_width = 128
num_loss_points = 100**2

# Domain
geom = gdn.geometry.UnitSquare()
collocation_points = geom.uniform_points(num_collocation_points)

# Transformation
phi = gdn.transformation.PolarCoordinates(
    r_min=0.5, r_max=1.0,
    theta_min=0.0, theta_max=np.pi / 2
)

# Boundary condition
bc = gdn.bc.UnitCubeDirichletBC({"left": 0, "right": 0})

# Define PDE
pde = gdn.pde.Poisson(bc, source=1)

# Setup DeepONet
model = gdn.deeponet.GeoDeepONet(
    branch_width=branch_width,
    trunk_width=trunk_width,
    num_collocation_points=len(collocation_points),
    d=geom.dimension
)

# Setup loss points and bc
loss_points = geom.uniform_points(num_loss_points)
pde.setup_bc(loss_points)

# Train model
gdn.train.train_model(
    geom, model, collocation_points, phi, pde, loss_points
)
