import deepxde as dde
import numpy as np
sin = dde.backend.pytorch.sin

# General parameters
n = 2
hard_constraint = False
iterations = 5000
parameters = [1e-3, 3, 32, "tanh"]
learning_rate, num_dense_layers, num_dense_nodes, activation = parameters


def pde(x, y):
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_yy = dde.grad.hessian(y, x, i=1, j=1)
    f = sin(np.pi * x[:, 0:1]) * sin(np.pi * x[:, 1:2])
    return -dy_xx - dy_yy - f


def transform(x, y):
    res = x[:, 0:1] * (1 - x[:, 0:1]) * x[:, 1:2] * (1 - x[:, 1:2])
    return res * y


def boundary(_, on_boundary):
    return on_boundary


geom = dde.geometry.Rectangle([0, 0], [1, 1])

if hard_constraint == True:
    bc = []
else:
    bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary)

data = dde.data.PDE(
    geom,
    pde,
    bc,
    num_domain=128,
    num_boundary=32,
    num_test=128,
)

net = dde.nn.FNN(
    [2] + [num_dense_nodes] * num_dense_layers + [1], activation, "Glorot uniform"
)

if hard_constraint == True:
    net.apply_output_transform(transform)

model = dde.Model(data, net)
model.compile("adam", lr=learning_rate)

losshistory, train_state = model.train(iterations=iterations)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
