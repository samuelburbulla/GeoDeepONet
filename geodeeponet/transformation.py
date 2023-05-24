import numpy as np
import torch


# Polar coordinate transformation
#
# phi: [0, 1]^2 -> R^2
# phi(x) = (r(x) cos(theta(x)), r(x) sin(theta(x)))
# where
#   r(x) = r_min + (r_max - r_min) x[0]
#   theta(x) = theta_min + (theta_max - theta_min) x[1]
class PolarCoordinates:
    def __init__(self, r_min=0.5, r_max=1.0,
                 theta_min=0.0, theta_max=np.pi / 2):

        self.r_min = r_min
        self.r_max = r_max
        self.theta_min = theta_min
        self.theta_max = theta_max

    def __call__(self, xs):
        ys = torch.zeros_like(xs)
        for i, x in enumerate(xs):
            r = self.r_min + (self.r_max - self.r_min) * x[0]
            theta = self.theta_min + (self.theta_max - self.theta_min) * x[1]
            ys[i][0] = r * torch.cos(theta)
            ys[i][1] = r * torch.sin(theta)
        return ys
