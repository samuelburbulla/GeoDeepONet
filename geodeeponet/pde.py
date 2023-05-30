import abc
import torch
import torch.autograd
from geodeeponet.vectorcalculus import VectorCalculus
from torch.autograd import grad

class PDE(abc.ABC):
    """Abstract base class for partial differential equations."""

    @abc.abstractmethod
    def setup_bc(self, points):
        """Sets up the boundary condition.

        Args:
            points (torch.Tensor): The points in the domain.

        """

    @abc.abstractmethod
    def __call__(self, u, phi_points):
        """Computes the loss.

        Args:
            u (torch.Tensor): The solution tensor.
            phi_points (torch.Tensor): The points in the domain.

        Returns:
            torch.Tensor: The loss tensor.

        """


class Poisson(PDE):
    """A class representing Poisson's equation with Dirichlet boundary conditions.
    
    \begin{align}
      -\Delta u &= q, \quad \text{in } \Omega_\phi, \\
      u &= uD, \quad \text{on } \partial \Omega_\phi,
    \end{align}
    where domain $\Omega_\phi = \phi(\Omega)$ is parameterised by $\phi: \Omega \to \mathbb{R}^d$.

    Methods:
        setup_bc(points):
            Sets up the boundary condition.
        __call__(u, phi_points):
            Computes the loss.

    """

    def __init__(self, bc, source=0):
        """Initializes the Poisson class.

        Args:
            bc: The boundary conditions.
            source (float or callable, optional): The source term (evaluated in local coordinates). Defaults to 0.

        """
        self.outputs = 1
        self.bc = bc
        self.source = source if callable(source) else lambda x: source
        self.dirichlet_indices = []
        self.dirichlet_values = torch.tensor([])

    def setup_bc(self, points):
        """Sets up the boundary condition.

        Args:
            points (torch.Tensor): The points sampling the domain.

        """
        for i, x in enumerate(points):
            if self.bc.is_dirichlet(x):
                self.dirichlet_indices += [i]
                self.dirichlet_values = torch.cat((self.dirichlet_values, torch.tensor([self.bc.value(x)])))

    def __call__(self, u, points):
        """Computes the loss.

        Args:
            u (torch.Tensor): The solution tensor.
            points (torch.Tensor): The points in the (global) domain.

        Returns:
            torch.Tensor: The loss tensor.

        """
        # Compute derivatives
        vc = VectorCalculus(points)
        laplace_u = vc.div(vc.grad(u))

        # Evaluate source term
        q = torch.tensor(self.source(points))

        # Compute inner loss
        inner_loss = ((- laplace_u - q)**2).mean()

        # Compute boundary loss
        u_boundary = u[:, :, self.dirichlet_indices]
        u_dirichlet = self.dirichlet_values.unsqueeze(0)
        boundary_loss = ((u_boundary - u_dirichlet)**2).mean()
        
        return inner_loss, boundary_loss



class Elasticity(PDE):
    """A class implementing linear elasticity with Dirichlet boundary conditions.

    Methods:
        setup_bc(points):
            Sets up the boundary condition.
        __call__(u, phi_points):
            Computes the loss.

    """

    def __init__(self, bc, dim, E=100, nu=0.3, rho=1, gravity=None):
        """Initializes the Elasticity class.

        Args:
            bc: The boundary conditions.
            dim (int): The dimension of the problem.
            E (float, optional): Young's modulus $E$. Defaults to 200e9 (steel).
            nu (float, optional): Poisson's ratio $\nu$. Defaults to 0.3 (steel).
            rho (float, optional): The density $\rho$. Defaults to 8e3 (steel).
            gravity (torch.Tensor): The gravity vector. Defaults to [0, 0, -9.81].

        """
        self.outputs = dim
        self.bc = bc
        self.dim = dim

        # Lame parameters
        self.lamb = E * nu / ((1 + nu) * (1 - 2 * nu))
        self.mu = E / (2 * (1 + nu))

        if gravity is None:
            gravity = torch.zeros(dim)
            gravity[-1] = -9.81
        self.gravity = rho * gravity

        self.dirichlet_indices = []
        self.dirichlet_values = torch.tensor([])

    def setup_bc(self, points):
        """Sets up the boundary condition.

        Args:
            points (torch.Tensor): The points sampling the domain.

        """
        for i, x in enumerate(points):
            if self.bc.is_dirichlet(x):
                self.dirichlet_indices += [i]
                self.dirichlet_values = torch.cat(
                    (self.dirichlet_values, torch.tensor([self.bc.value(x)]))
                )
        self.dirichlet_values = self.dirichlet_values.transpose(0, 1)

    def __call__(self, u, points):
        """Computes the loss.

        Args:
            u (torch.Tensor): The solution tensor (displacement vector) of shape [batch_size, output_size, num_points].
            points (torch.Tensor): The points in the (global) domain of shape [batch_size, num_points, dim].

        Returns:
            (torch.Tensor, torch.Tensor): Tuple of inner and boundary loss tensor.

        """
        batch_size = u.shape[0]
        num_points = points.shape[1]
        vc = VectorCalculus(points)

        # Create gravity tensor
        g = self.gravity.unsqueeze(-1).repeat(1, 1, num_points)

        # Assemble: sigma = lamb * div(u) * I
        divu = vc.div(u)
        divuI = (divu.view(1, num_points, 1, 1) * torch.eye(self.dim)).transpose(1, 2)
        sigma = self.lamb * divuI

        # Assemble: sigma += mu * (grad(u) + grad(u)^T)
        gradu = vc.grad(u)
        graduT = torch.transpose(gradu, 1, 3)
        sigma = self.mu * (gradu + graduT)

        # Compute div(sigma)
        divsigma = vc.div(sigma)

        # Compute: - div(sigma) = g
        inner_loss = ((- divsigma - g)**2).mean()

        # Compute boundary loss
        boundary_loss = torch.zeros((batch_size))
        u_boundary = u[:, :, self.dirichlet_indices]
        u_dirichlet = self.dirichlet_values
        boundary_loss = ((u_boundary - u_dirichlet)**2).mean()

        return inner_loss, boundary_loss
