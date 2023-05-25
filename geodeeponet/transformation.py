import numpy as np
import torch


""" Identity transformation
"""
class Identity:
    def __call__(self, xs):
        return xs
    

""" Affine transformation

    phi(x) = A x + b

    Args:
        A: (dim, dim) array_like
        b: (dim) array_like
"""
class Affine():
    def __init__(self, A=None, b=None, dim=None):
        if A is None:
            assert dim is not None
            self.A = np.eye(dim) + np.random.rand(dim, dim)

        if b is None:
            assert dim is not None
            self.b = np.random.rand(dim)

        self.A = torch.tensor(self.A, dtype=torch.float64)
        self.b = torch.tensor(self.b, dtype=torch.float64)

    def __call__(self, xs):
        ys = torch.zeros_like(xs)
        for i, x in enumerate(xs):
            ys[i] = self.A @ xs[i] + self.b
        return ys


# Polar coordinate transformation
#
# phi: [0, 1]^2 -> R^2
# phi(x) = (r(x) cos(theta(x)), r(x) sin(theta(x)))
# where
#   r(x) = r_min + (r_max - r_min) x[0]
#   theta(x) = theta_min + (theta_max - theta_min) x[1]
class PolarCoordinates:
    def __init__(self, r_min=None, r_max=None, theta_min=None, theta_max=None):
        if r_min is None:
            self.r_min = np.random.rand()
        if r_max is None:
            self.r_max = self.r_min + np.random.rand()
        if theta_min is None:
            self.theta_min = 2 * np.pi * np.random.rand()
        if theta_max is None:
            self.theta_max = self.theta_min + 2 * np.pi * np.random.rand()

    def __call__(self, xs):
        ys = torch.zeros_like(xs)
        for i, x in enumerate(xs):
            r = self.r_min + (self.r_max - self.r_min) * x[0]
            theta = self.theta_min + (self.theta_max - self.theta_min) * x[1]
            ys[i][0] = r * torch.cos(theta)
            ys[i][1] = r * torch.sin(theta)
        return ys
