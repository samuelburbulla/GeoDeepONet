import numpy as np
import torch


class UnitSquare:
    """A class representing a unit square with uniform sampling.

    Attributes:
        dimension (int): The dimension of the unit square.

    """

    def __init__(self):
        self.dimension = 2

    @staticmethod
    def uniform_points(num_points):
        """Generate uniform collocation points within the unit square.

        Args:
            num_points (int): The number of collocation points to generate.

        Returns:
            torch.Tensor: A tensor of shape (num_points, 2) representing the
                generated collocation points.

        """
        n = int(np.ceil(np.sqrt(num_points)))
        if n**2 != num_points:
            print(f"Warning: {n**2} instead of {num_points} "
                  f"collocation points used.")

        h = 1 / (n - 1)
        xy = np.mgrid[0:1+1e-8:h, 0:1+1e-8:h].reshape(2, -1).T
        xy = torch.tensor(xy, requires_grad=True)
        return xy