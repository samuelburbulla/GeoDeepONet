"""Physics-informed GeoDeepONet for the Poisson problem (2D)."""
import geodeeponet as gdn

# Hyperparameters
dim = 3
num_collocation_points = 2**dim
branch_width = 8
trunk_width = 64
num_loss_points = 10**dim
num_train = 10
num_test = 3

# Domain
geom = gdn.geometry.UnitCube(dim)
collocation_points = geom.uniform_points(num_collocation_points)

# Transformations
phis = [
    gdn.transformation.PolarCoordinates() for _ in range(num_train)
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
    dimension=geom.dim
)

# Setup loss points and bc
loss_points = geom.uniform_points(num_loss_points)
pde.setup_bc(loss_points)

# Train model
gdn.train.train_model(
    geom, model, collocation_points, phis, pde, loss_points, plot_phis=True
)

# Validate model
phis = [
    gdn.transformation.PolarCoordinates() for _ in range(num_test)
]
global_collocation_points = [
    phi.inv(collocation_points) for phi in phis
]

loss, bc = gdn.train.compute_losses(model, pde, global_collocation_points, loss_points)
print(f"Validation   Loss: {loss.mean():.3e}  BC: {bc.mean():.3e}")

# Plot solution for one sample transformation
gdn.plot.plot_solution(geom, model, collocation_points, phis[0], writer="show")
