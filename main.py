"""Physics-informed GeoDeepONet for linear elasticity."""
import geodeeponet as gdn
import numpy as np

# Hyperparameters
dim = 2
num_collocation_points = 2**dim
branch_width = 1
trunk_width = 64

# Domain
geom = gdn.geometry.UnitCube(dim)
collocation_points = geom.uniform_points(num_collocation_points)

# Transformations
phi = gdn.transformation.Affine(
    A=np.array([[1., 0.], [0., 1.]]),
    b=np.zeros(dim)
)

# Boundary condition
bc = gdn.bc.UnitCubeDirichletBC({
    w: lambda x: [0.]*dim for w in ["left"]
})

# Define PDE
pde = gdn.pde.Elasticity(bc, dim)

# Setup DeepONet
model = gdn.deeponet.GeoDeepONet(
    branch_width=branch_width,
    trunk_width=trunk_width,
    num_collocation_points=len(collocation_points),
    dimension=geom.dim,
    outputs=pde.outputs,
)

# Train model
gdn.train.train_model(geom, model, collocation_points, [phi], pde, tolerance=2e-5)

# Plot solution
gdn.plot.plot_solution(geom, model, collocation_points, phi, writer="show")
