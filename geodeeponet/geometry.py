import numpy as np
import torch


# Unit square with uniform sampling
class UnitSquare:

    def __init__(self):
        self.dimension = 2

    # Uniform points
    @staticmethod
    def uniform_points(num_points):
        n = int(np.ceil(np.sqrt(num_points)))
        if n**2 != num_points:
            print(f"Warning: {n**2} instead of {num_points} "
                  f"collocation points used.")

        h = 1 / (n - 1)
        xy = np.mgrid[0:1+1e-8:h, 0:1+1e-8:h].reshape(2, -1).T
        xy = torch.tensor(xy, requires_grad=True)
        return xy
