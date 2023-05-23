# Physics-informed DeepONet with parametrised geometry
#
# We want to solve the Poisson problem
#
# \begin{align}
# -\Delta u &= 1, \quad \text{in } \Omega_\phi, \\
# u &= 0, \quad \text{on } \partial \Omega_\phi,
# \end{align}
#
# where domain $\Omega_\phi = \phi(\Omega)$ is parameterised by
# $\phi: \Omega \to \mathbb{R}^d$.

import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from geopideeponet.deeponet import BranchNetwork, TrunkNetwork, DeepONet
from geopideeponet.geometry import unit_square

# Initialize random seeds
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)


# Domain is unit square deformed
d = 2
n = 5
xy = unit_square(n)


# Transformation
def phi(xs):
    ys = torch.zeros_like(xs)
    for i, x in enumerate(xs):
        r, theta = 0.5 + 0.5 * x[0], x[1] * np.pi / 4
        ys[i][0] = r * torch.cos(theta)
        ys[i][1] = r * torch.sin(theta)
    return ys


# Boundary condition (on reference domain!)
def bc(x):
    left, right = x[0] < 1e-6, x[0] > 1 - 1e-6
    top, bottom = x[1] > 1 - 1e-6, x[1] < 1e-6
    on_boundary = left or right  # or top or bottom
    dirichlet_value = 0
    return on_boundary, dirichlet_value


# Define PDE
def pde(u, points, phi_points):
    loss = 0
    kw = lambda w: {"grad_outputs": torch.ones_like(w), "create_graph": True}
    du = torch.autograd.grad(u, phi_points, **kw(u))[0]
    u_xx = torch.autograd.grad(du[:, 0], phi_points, **kw(du[:, 0]))[0][:, 0]
    u_yy = torch.autograd.grad(du[:, 1], phi_points, **kw(du[:, 1]))[0][:, 1]
    q = torch.ones_like(u_xx)
    laplace_u = u_xx + u_yy
    loss += (- laplace_u - q).norm()**2

    for i, x in enumerate(points):
        on_boundary, dirichlet_value = bc(x)
        if on_boundary:
            loss += (u[:, i][0] - dirichlet_value)**2

    return loss


# Setup DeepONet
collocation_points = n**2
trunk_width = 32
branch = BranchNetwork(
    input_size=collocation_points * d,
    layer_width=16,
    output_size=trunk_width,
)
trunk = TrunkNetwork(
    input_size=d,
    output_size=trunk_width,
)
model = DeepONet(branch, trunk)


# Train model
loss_points = unit_square(n=10)
optimizer = torch.optim.LBFGS(model.parameters())
for i in range(100):

    def closure():
        optimizer.zero_grad()
        phi_points = phi(loss_points)
        outputs = model((phi(xy), phi_points))
        pde_loss = pde(outputs, loss_points, phi_points)
        pde_loss.backward()
        return pde_loss

    optimizer.step(closure)

    loss = closure()
    print(f"\rStep {i+1}: loss = {loss:.4g}", end="")

    if loss < 1e-6:
        break

print("")

# Evaluate operator
xs = unit_square(100)
u = model((phi(xy), phi(xs)))
phix = phi(xs)

# Detach tensors
phix = phix.detach().numpy()
u = u.detach().numpy()

# Scatter plot
plt.scatter(phix[:, 0], phix[:, 1], c=u, s=10)
plt.axis("equal")
plt.colorbar()
plt.show()
