import numpy as np
import torch


class Identity:
    """An identity transformation.

    Methods:
        __call__(xs):
            Returns the input.

    """

    def inv(self, xs):
        """Applies the inverse identity transformation from local to global coordinates.

        Args:
            xs (torch.Tensor): The local input tensor.

        Returns:
            torch.Tensor: The global output tensor.

        """
        return xs

    def __call__(self, ys):
        """Applies the identity transformation from global to local coordinates.

        Args:
            ys (torch.Tensor): The global input tensor.

        Returns:
            torch.Tensor: The local output tensor.

        """
        return ys
    

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

    def __init__(self, A=None, b=None, dim=2):
        """Initializes the Affine class.

        Args:
            A (array_like, optional): The linear transformation matrix. If None, identity matrix plus a random matrix is used. Defaults to None.
            b (array_like, optional): The translation vector. If None, a random vector is generated. Defaults to None.
            dim (int, optional): The dimension of the input and output spaces. Defaults to 2.

        """
        if A is None:
            A = np.eye(dim) + np.random.rand(dim, dim)

        if b is None:
            b = np.random.rand(dim)

        self.A = torch.tensor(A.T, requires_grad=True)
        self.b = torch.tensor(b, requires_grad=True)


    def inv(self, xs):
        """Applies the affine transformation from local to global coordinates.

        Args:
            xs (torch.Tensor): The local input tensor.

        Returns:
            torch.Tensor: The global output tensor.

        """
        return torch.matmul(xs, self.A) + self.b

    def __call__(self, ys):
        """Applies the affine transformation from global to local coordinates.

        Args:
            ys (torch.Tensor): The global input tensor.

        Returns:
            torch.Tensor: The local output tensor.

        """
        return torch.matmul(ys - self.b, torch.inverse(self.A))


class PolarCoordinates:
    """A polar coordinate transformation.

    Attributes:
        r_min (float): The minimum radius.
        r_max (float): The maximum radius.
        theta_min (float): The minimum angle.
        theta_max (float): The maximum angle.
        phi_min (float): The second minimum angle (3D).
        phi_max (float): The second maximum angle (3D).

    Methods:
        __init__(r_min=None, r_max=None, theta_min=None, theta_max=None):
            Initializes the PolarCoordinates class.
        __call__(xs):
            Applies the polar coordinate transformation.

    """

    def __init__(self, r_min=None, r_max=None, theta_min=None, theta_max=None, phi_min=None, phi_max=None):
        """Initializes the PolarCoordinates class.

        Args:
            r_min (float, optional): The minimum radius. If None, a random value is generated. Defaults to None.
            r_max (float, optional): The maximum radius. If None, a random value plus r_min is generated. Defaults to None.
            theta_min (float, optional): The minimum angle. If None, a random value is generated. Defaults to None.
            theta_max (float, optional): The maximum angle. If None, a random value plus theta_min is generated. Defaults to None.
            phi_min (float, optional): The second minimum angle (3D). If None, a random value is generated. Defaults to None.
            phi_max (float, optional): The second maximum angle (3D). If None, a random value plus phi_min is generated. Defaults to None.

        """
        if r_min is None:
            r_min = np.random.rand()
        if r_max is None:
            r_max = r_min + np.random.rand()
        if theta_min is None:
            theta_min = 2 * np.pi * np.random.rand()
        if theta_max is None:
            theta_max = theta_min + 2 * np.pi * np.random.rand()
            theta_max = theta_max % (2*np.pi)
        if phi_min is None:
            phi_min = np.pi * np.random.rand()
        if phi_max is None:
            phi_max = phi_min + np.pi * np.random.rand()
            phi_max = phi_max % np.pi

        self.r_min = r_min
        self.r_max = r_max
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.phi_min = phi_min
        self.phi_max = phi_max

    def inv(self, xs):
        """Applies the transformation from local to global coordinates.

        Args:
            xs (torch.Tensor): The local input tensor.

        Returns:
            torch.Tensor: The global output tensor.

        """
        if len(xs.shape) == 1:
            xs = xs.unsqueeze(0)

        dim = xs.shape[1]
        ys = torch.zeros_like(xs)
        r = self.r_min + (self.r_max - self.r_min) * xs[:, 0]
        theta = self.theta_min + (self.theta_max - self.theta_min) * xs[:, 1]

        if dim == 2:
            ys[:, 0] = r * torch.cos(theta)
            ys[:, 1] = r * torch.sin(theta)

        if dim == 3:
            phi = self.phi_min + (self.phi_max - self.phi_min) * xs[:, 2]
            ys[:, 0] = r * torch.cos(theta) * torch.sin(phi)
            ys[:, 1] = r * torch.sin(theta) * torch.sin(phi)
            ys[:, 2] = r * torch.cos(phi)
        return ys
    

    def __call__(self, ys):
        """Applies the transformation from global to local coordinates.

        Args:
            ys (torch.Tensor): The global input tensor.

        Returns:
            torch.Tensor: The local output tensor.

        """
        xs = torch.zeros_like(ys)
        for i, y in enumerate(ys):
            r = torch.norm(y)
            theta = torch.atan2(y[1], y[0])

            xs[i][0] = (r - self.r_min) / (self.r_max - self.r_min)
            xs[i][1] = (theta - self.theta_min) / (self.theta_max - self.theta_min)

            if len(y) == 3:
                phi = torch.arccos(y[2] / r)
                xs[i][2] = (phi - self.phi_min) / (self.phi_max - self.phi_min)
        return xs
