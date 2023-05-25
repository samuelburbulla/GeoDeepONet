import abc
import torch
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
    """A class representing Poisson's equation (2D) with Dirichlet boundary conditions.
    
    \begin{align}
      -\Delta u &= q, \quad \text{in } \Omega_\phi, \\
      u &= uD, \quad \text{on } \partial \Omega_\phi,
    \end{align}
    where domain $\Omega_\phi = \phi(\Omega)$ is parameterised by $\phi: \Omega \to \mathbb{R}^d$.

    Attributes:
        bc: The boundary conditions.
        source (float or callable): The source term.

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
        def kw(w):
            return {
                "grad_outputs": torch.ones_like(w),
                "create_graph": True,
            }

        # Compute derivatives
        du = grad(u, points, **kw(u))[0]
        u_xx = grad(du[:, :, 0], points, **kw(du[:, :, 0]))[0][:, :, 0]
        u_yy = grad(du[:, :, 1], points, **kw(du[:, :, 1]))[0][:, :, 1]
        laplace_u = u_xx + u_yy

        # Evaluate source term
        q = torch.tensor(self.source(points))

        # Compute inner loss
        inner_loss = ((- laplace_u - q)**2).mean()

        # Compute boundary loss
        u_boundary = u[:, :, self.dirichlet_indices]
        u_dirichlet = self.dirichlet_values.repeat(u.shape[0], 1, 1)
        boundary_loss = ((u_boundary - u_dirichlet)**2).mean()

        return inner_loss, boundary_loss
