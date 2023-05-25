import numpy as np
import torch


class Identity:
    """An identity transformation.

    Methods:
        __call__(xs):
            Returns the input.

    """

    def __call__(self, xs):
        """Returns the input.

        Args:
            xs (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The input tensor.

        """
        return xs
    

class Affine:
    """An affine transformation that maps global to local coordinates.

    Attributes:
        A (torch.Tensor): The linear transformation matrix.
        Ainv (torch.Tensor): The inverse of the linear transformation matrix.
        b (torch.Tensor): The translation vector.

    Methods:
        __init__(A=None, b=None, dim=None):
            Initializes the Affine class.
        inv(xs):
            Applies the inverse affine transformation (local to global).
        __call__(xs):
            Applies the affine transformation (global to local).

    """

    def __init__(self, A=None, b=None, dim=None):
        """Initializes the Affine class.

        Args:
            A (array_like, optional): The linear transformation matrix. If None, identity matrix plus a random matrix is used. Defaults to None.
            b (array_like, optional): The translation vector. If None, a random vector is generated. Defaults to None.
            dim (int, optional): The dimension of the input and output spaces. Required if A or b is None. Defaults to None.

        """
        if A is None:
            assert dim is not None
            A = np.eye(dim) + np.random.rand(dim, dim)

        if b is None:
            assert dim is not None
            self.b = np.random.rand(dim)
        else:
            self.b = b

        self.A = torch.tensor(A, dtype=torch.float64)
        self.Ainv = torch.tensor(np.linalg.inv(A), dtype=torch.float64)
        self.b = torch.tensor(self.b, dtype=torch.float64)


    def inv(self, xs):
        """Applies the affine transformation from local to global coordinates.

        Args:
            xs (torch.Tensor): The local input tensor.

        Returns:
            torch.Tensor: The global output tensor.

        """
        ys = torch.zeros_like(xs)
        for i, x in enumerate(xs):
            ys[i] = self.A @ x + self.b
        return ys

    def __call__(self, ys):
        """Applies the affine transformation from global to local coordinates.

        Args:
            ys (torch.Tensor): The global input tensor.

        Returns:
            torch.Tensor: The local output tensor.

        """
        xs = torch.zeros_like(ys)
        for i, y in enumerate(ys):
            xs[i] = self.Ainv @ (y - self.b)
        return xs


class PolarCoordinates:
    """A polar coordinate transformation.

    Attributes:
        r_min (float): The minimum radius.
        r_max (float): The maximum radius.
        theta_min (float): The minimum angle.
        theta_max (float): The maximum angle.

    Methods:
        __init__(r_min=None, r_max=None, theta_min=None, theta_max=None):
            Initializes the PolarCoordinates class.
        __call__(xs):
            Applies the polar coordinate transformation.

    """

    def __init__(self, r_min=None, r_max=None, theta_min=None, theta_max=None):
        """Initializes the PolarCoordinates class.

        Args:
            r_min (float, optional): The minimum radius. If None, a random value is generated. Defaults to None.
            r_max (float, optional): The maximum radius. If None, a random value plus r_min is generated. Defaults to None.
            theta_min (float, optional): The minimum angle. If None, a random value is generated. Defaults to None.
            theta_max (float, optional): The maximum angle. If None, a random value plus theta_min is generated. Defaults to None.

        """
        if r_min is None:
            self.r_min = np.random.rand()
        if r_max is None:
            self.r_max = self.r_min + np.random.rand()
        if theta_min is None:
            self.theta_min = 2 * np.pi * np.random.rand()
        if theta_max is None:
            self.theta_max = self.theta_min + 2 * np.pi * np.random.rand()

    def __call__(self, xs):
        """Applies the polar coordinate transformation.

        Args:
            xs (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        ys = torch.zeros_like(xs)
        for i, x in enumerate(xs):
            r = self.r_min + (self.r_max - self.r_min) * x[0]
            theta = self.theta_min + (self.theta_max - self.theta_min) * x[1]
            ys[i][0] = r * torch.cos(theta)
            ys[i][1] = r * torch.sin(theta)
        return ys