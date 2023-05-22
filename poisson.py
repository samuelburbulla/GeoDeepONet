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
n = 3
xy = unit_square(n)


# Transformation
def phi(x):
    return 2 * x


# Boundary condition (on reference domain)
def bc(x):
    left, right = x[0] < 1e-6, x[0] > 1 - 1e-6
    top, bottom = x[1] > 1 - 1e-6, x[1] < 1e-6
    on_boundary = left or right or top or bottom
    dirichlet_value = 0
    return on_boundary, dirichlet_value


# Define PDE
def pde(u, points, phi):
    loss = 0

    kw = lambda w: {"grad_outputs": torch.ones_like(w), "create_graph": True}
    phi_jac = torch.autograd.grad(phi(points), points, **kw(points))[0]
    du = phi_jac.T @ torch.autograd.grad(u, points, **kw(u))[0]
    u_xx = torch.autograd.grad(du[:, 0], points, **kw(du[:, 0]))[0][:, 0] # TODO: phi_jac?
    u_yy = torch.autograd.grad(du[:, 1], points, **kw(du[:, 1]))[0][:, 1] # TODO: phi_jac?
    q = torch.ones_like(u_xx)
    loss += (- u_xx - u_yy - q).norm()**2

    for i, x in enumerate(points):
        on_boundary, dirichlet_value = bc(x)
        if on_boundary:
            loss += (u[:, i][0] - dirichlet_value)**2

    return loss


# Setup DeepONet
collocation_points = n**2
trunk_width = 256
branch = BranchNetwork(
    input_size=collocation_points * d,
    layer_width=128,
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
        outputs = model((phi(xy), loss_points))
        pde_loss = pde(outputs, loss_points, phi)
        pde_loss.backward()
        return pde_loss

    optimizer.step(closure)

    loss = closure()
    print(f"\rStep {i+1}: loss = {loss:.4g}", end="")

    if loss < 1e-5:
        break

print("")

# Evaluate operator
xs = unit_square(100)
phix = phi(xs)
y = model((phi(xy), xs))

# Detach tensors
phix = phix.detach().numpy()
y = y.detach().numpy()

# Scatter plot
plt.scatter(phix[:, 0], phix[:, 1], c=y, s=10)
plt.axis("equal")
plt.colorbar()
plt.show()
