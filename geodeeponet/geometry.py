import numpy as np
import torch


class UnitCube:
    """A class representing a unit cube with uniform sampling.

    Attributes:
        dim (int): The dimension of the unit cube.

    """

    def __init__(self, dim):
        self.dim = dim

    def uniform_points(self, num_points):
        """Generate uniform collocation points within the unit cube.

        Args:
            num_points (int): The number of collocation points to generate.

        Returns:
            torch.Tensor: A tensor of shape (num_points, dim) representing the
                generated collocation points.

        """
        d = self.dim
        n = int(np.ceil(num_points**(1/d)))
        if n**d != num_points:
            print(f"Warning: {n**d} instead of {num_points} "
                  f"collocation points used.")

        grids = []
        for _ in range(d):
            grids.append(np.linspace(0, 1, n))
        mesh = np.meshgrid(*grids, indexing='ij')
        xy = np.array(mesh).reshape(d, -1).T
        xy = torch.tensor(xy, requires_grad=True)
        return xy
    
    def random_points(self, num_points):
        """Generate random collocation points within the unit cube.

        Args:
            num_points (int): The number of collocation points to generate.

        Returns:
            torch.Tensor: A tensor of shape (num_points, dim) representing the
                generated collocation points.

        """
        return torch.rand(num_points, self.dim, requires_grad=True, dtype=torch.float64)
    
    def random_boundary_points(self, num_points):
        """Generate random points on the boundary of the unit cube.

        Args:
            num_points (int): The number of boundary points to generate.

        Returns:
            torch.Tensor: A tensor of shape (num_points, dim) representing the
                generated boundary points.

        """
        xs = torch.rand(num_points, self.dim)
        for i in range(num_points):
            xs[i, np.random.randint(self.dim)] = np.random.choice([0, 1])
        return xs