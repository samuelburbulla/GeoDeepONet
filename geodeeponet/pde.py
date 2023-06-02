import abc
import torch
import torch.autograd
from geodeeponet.deriv import grad, div
from geodeeponet.bc import BoundaryCondition

class PDE(abc.ABC):
    """Abstract base class for partial differential equations."""
    bc: BoundaryCondition
    
    def setup_bc(self, points):
        """Sets up the boundary condition.

        Args:
            points (torch.Tensor): The points sampling the domain.

        """
        self.dirichlet_indices = []
        self.neumann_indices = []
        dirichlet_values = []
        neumann_normals = []

        for i, x in enumerate(points):
            if self.bc.is_dirichlet(x):
                self.dirichlet_indices += [i]
                dirichlet_values.append(self.bc.value(x))

            if self.bc.is_neumann(x):
                self.neumann_indices += [i]
                neumann_normals.append(self.bc.normal(x))

        self.dirichlet_values = torch.tensor(dirichlet_values)
        if len(neumann_normals) > 0:
            self.neumann_normals = torch.stack(neumann_normals)


    def has_dirichlet(self):
        """Returns if current points contain Dirichlet boundary conditions."""
        return len(self.dirichlet_indices) > 0
    

    def has_neumann(self):
        """Returns if current points contain Neumann boundary conditions."""
        return len(self.neumann_indices) > 0


    @abc.abstractmethod
    def __call__(self, u, points, jacobians):
        """Computes the loss.

        Args:
            u (torch.Tensor): The solution tensor.
            points (torch.Tensor): The points in the domain.
            jacobians (torch.Tensor): The Jacobians of the transformation functions.

        Returns:
            tuple: The inner loss and boundary loss.

        """


class Poisson(PDE):
    """A class representing Poisson's equation with Dirichlet boundary conditions.
    
      - Delta u = q, in domain,
      u = uD, on Dirichlet boundary,
      grad u * n = 0, on Neumann boundary.
    
    Methods:
        __call__(u, phi_points):
            Computes the loss.

    """

    def __init__(self, bc, source=lambda x: torch.tensor(0.)):
        """Initializes the Poisson class.

        Args:
            bc: The boundary conditions.
            source (float or callable, optional): The source term (evaluated in local coordinates). Defaults to 0.

        """
        self.outputs = 1
        self.bc = bc
        self.source = source if callable(source) else lambda x: source


    def __call__(self, u, points, jacobians):
        """Computes the loss.

        Args:
            u (torch.Tensor): The solution tensor (shape: (batch_size, outputs, num_points)).
            points (torch.Tensor): The points in the domain where loss is evaluated (shape: (num_points, dim)).
            jacobians (torch.Tensor): The Jacobians of the transformation functions (shape: (batch_size, num_points, dim, dim)).

        Returns:
            tuple: The inner loss and boundary loss.

        """
        # Compute derivatives
        gradu = grad(u, points, jacobians)
        laplace_u = div(gradu, points, jacobians)

        # Evaluate source term
        q = self.source(points)

        # Compute inner loss
        inner_loss = ((- laplace_u - q)**2).mean()

        # Compute Dirichlet boundary loss
        boundary_loss = 0
        if self.has_dirichlet():
            u_boundary = u[:, :, self.dirichlet_indices]
            boundary_loss += ((u_boundary - self.dirichlet_values)**2).mean()

        # Compute Neumann boundary loss
        if self.has_neumann():
            # Compute the normals of transformed domain
            jacs = jacobians[:, self.neumann_indices]
            n = self.neumann_normals.unsqueeze(-1)
            phi_n = torch.matmul(jacs, n)

            # Normalize the normals
            phi_n = torch.nn.functional.normalize(phi_n, dim=-2)

            # Compute the normal gradient of u
            grads = gradu[:, :, self.neumann_indices]
            grads = grads.unsqueeze(-1).transpose(-1, -2)
            gradu_n = torch.matmul(grads, phi_n)
            
            boundary_loss += ((gradu_n)**2).mean()
        
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
        self.lamb = 1.25  # E * nu / ((1 + nu) * (1 - 2 * nu))
        self.mu = 1  # E / (2 * (1 + nu))

        if gravity is None:
            gravity = torch.zeros(dim)
            gravity[-1] = 1 #-0.016
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

    def __call__(self, u, points, jacobians):
        """Computes the loss.

        Args:
            u (torch.Tensor): The solution tensor (displacement vector) of shape [batch_size, output_size, num_points].
            points (torch.Tensor): The points in the (global) domain of shape [batch_size, num_points, dim].
            jacobians (torch.Tensor): The Jacobians of the transformation functions.
            
        Returns:
            tuple: The inner loss and boundary loss.

        """
        num_points = points.shape[1]

        # Create gravity tensor
        g = self.gravity.unsqueeze(-1).repeat(1, 1, num_points)

        # Assemble: sigma = lamb * div(u) * I
        divu = div(u, points, jacobians)
        divuI = (divu.view(1, num_points, 1, 1) * torch.eye(self.dim)).transpose(1, 2)
        sigma = self.lamb * divuI

        # Assemble: sigma += mu * (grad(u) + grad(u)^T)
        gradu = grad(u, points, jacobians)
        graduT = torch.transpose(gradu, 1, 3)
        sigma += self.mu * (gradu + graduT)
        
        # Compute divergence of sigma
        divsigma = div(sigma, points, jacobians)

        # Compute loss
        inner_loss = ((- divsigma - g)**2).mean()

        # Compute Dirichlet boundary loss
        u_boundary = u[:, :, self.dirichlet_indices]
        u_dirichlet = self.dirichlet_values.unsqueeze(0)
        boundary_loss = ((u_boundary - u_dirichlet)**2).mean()

        # Compute Neumann boundary loss (sigma * n = 0)
        neumann_loss = 0
        num_neumann = 0
        for i in range(num_points):
            x = points[:, i, :][0]
            if self.bc.is_neumann(x):
                sigma_n = (sigma[:, :, i, :] @ self.bc.normal(x))
                neumann_loss += (sigma_n**2).sum()
                num_neumann += 1
        inner_loss += (neumann_loss / num_neumann)

        return inner_loss, boundary_loss
