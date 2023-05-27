import abc
import torch
import torch.autograd
from geodeeponet.grad import gradient, div, jacobian


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
            phi_points (torch.Tensor): The points in the (global) domain.

        Returns:
            torch.Tensor: The loss tensor.

        """
        batch_size = u.shape[0]
       
        # Compute derivatives
        inner_loss = torch.zeros((batch_size))
        for i in range(batch_size):
            gradu = gradient(u[i, :, :], points)
            laplace_u = div(gradu, points)

            # Evaluate source term
            q = torch.tensor(self.source(points))

            # Compute inner loss
            inner_loss[i] = ((- laplace_u - q)**2).mean()

        # Compute boundary loss
        boundary_loss = torch.zeros((batch_size))
        for i in range(batch_size):
            u_boundary = u[i, :, self.dirichlet_indices]
            u_dirichlet = self.dirichlet_values.unsqueeze(0)
            boundary_loss[i] = ((u_boundary - u_dirichlet)**2).mean()
        
        return inner_loss, boundary_loss



class Elasticity(PDE):
    """A class implementing linear elasticity with Dirichlet boundary conditions.

    Methods:
        setup_bc(points):
            Sets up the boundary condition.
        __call__(u, phi_points):
            Computes the loss.

    """

    def __init__(self, bc, dim, lamb=1.0, mu=1.0, gravity=None):
        """Initializes the Elasticity class.

        Args:
            bc: The boundary conditions.
            dim (int): The dimension of the problem.
            lamb (float, optional): The Lamé parameter $\lambda$. Defaults to 1.0.
            mu (float, optional): The Lamé parameter $\mu$. Defaults to 1.0.
            gravity (torch.Tensor): The gravity vector. Defaults to [-1, 0, 0].

        """
        self.outputs = dim
        self.bc = bc
        self.dim = dim
        self.lamb = lamb
        self.mu = mu

        if gravity is None:
            gravity = torch.zeros(dim)
            gravity[-1] = -1
        self.gravity = gravity

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
        self.dirichlet_values = self.dirichlet_values.T

    def __call__(self, u, points):
        """Computes the loss.

        Args:
            u (torch.Tensor): The solution tensor (displacement vector).
            points (torch.Tensor): The points in the (global) domain.

        Returns:
            (torch.Tensor, torch.Tensor): Tuple of inner and boundary loss tensor.

        """
        batch_size = u.shape[0]
        num_points = points.shape[1]

        # Create gravity tensor
        g = self.gravity.repeat(num_points, 1)

        # Compute derivatives
        inner_loss = torch.zeros((batch_size))
        for i in range(batch_size):
            # Assemble: sigma = lamb * div(u) * I
            us = u[i, :, :].transpose(0, 1).unsqueeze(0)
            divu = div(us, points)
            divuI = divu.view(1, num_points, 1, 1) * torch.eye(self.dim)
            sigma = self.lamb * divuI[0]

            # Assemble: sigma += mu * (grad(u) + grad(u)^T)
            gradu = jacobian(u[i], points)
            graduT = torch.transpose(gradu, 1, 2)
            sigma += self.mu * (gradu + graduT)

            # Compute div(sigma)
            divsigma = []
            for d in range(self.dim):
                divsigma += [div(sigma[:, :, d].unsqueeze(0), points)]
            divsigma = torch.stack(divsigma, dim=2)[0]

            # Compute: - div(sigma) = g
            inner_loss[i] = ((- divsigma - g)**2).mean()

        # Compute boundary loss
        boundary_loss = torch.zeros((batch_size))
        for i in range(batch_size):
            u_boundary = u[i, :, self.dirichlet_indices]
            u_dirichlet = self.dirichlet_values
            boundary_loss[i] = ((u_boundary - u_dirichlet)**2).mean()

        return inner_loss, boundary_loss
