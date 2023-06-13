import abc
import torch
import torch.autograd
from geodeeponet.grad import grad, div
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
        neumann_values = []

        for i, x in enumerate(points):
            if self.bc.is_dirichlet(x):
                self.dirichlet_indices += [i]
                v = self.bc.value(x)
                if not isinstance(v, torch.Tensor):
                    v = torch.tensor(v)
                dirichlet_values.append(v)

            if self.bc.is_neumann(x):
                self.neumann_indices += [i]
                neumann_normals.append(self.bc.normal(x))
                neumann_values.append(self.bc.neumann(x))

        if len(dirichlet_values) > 0:
            self.dirichlet_values = torch.stack(dirichlet_values)
        if len(neumann_normals) > 0:
            self.neumann_normals = torch.stack(neumann_normals)
            self.neumann_values = torch.stack(neumann_values)


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
    """A PDE class implementing Poisson's equation.
    
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
            u (torch.Tensor): The solution tensor of shape [batch_size, outputs, num_points].
            points (torch.Tensor): The points in the domain where loss is evaluated of shape [num_points, dim].
            jacobians (torch.Tensor): The Jacobians of the transformation functions of shape [batch_size, num_points, dim, dim].

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
    """A PDE class implementing linear elasticity.

    -div(sigma(u)) = g, in domain,
    sigma(u) = lambda * div(u) * I + mu * (grad(u) + grad(u)^T)
    u = uD, on Dirichlet boundary,
    sigma(u) * n = 0, on Neumann boundary.

    Methods:
        __call__(u, phi_points):
            Computes the loss.

    """

    def __init__(self, bc, dim, lamb=1, mu=1, rho=1, gravity=None):
        """Initializes the Elasticity class.

        Args:
            bc: The boundary conditions.
            dim (int): The dimension of the problem.
            lamb (float, optional): Lame parameter lambda. Defaults to 1.
            mu (float, optional): Lame parameter mu. Defaults to 1.
            rho (float, optional): The density rho. Defaults to 1.
            gravity (callable, optional): The gravity vector. Defaults to lambda x: [0, -0.1].

        """
        self.outputs = dim
        self.bc = bc
        self.dim = dim

        # Lame parameters
        self.lamb = lamb  # E * nu / ((1 + nu) * (1 - 2 * nu))
        self.mu = mu      # E / (2 * (1 + nu))
        self.rho = rho

        if gravity is None:
            def gravity(x):
                g = torch.zeros_like(x)
                g[:, -1] = -0.1
                return g.T
            self.gravity = gravity

        elif not callable(gravity):
            self.gravity = lambda x: torch.tensor(gravity).repeat(x.shape[0], 1).T

        else:
            self.gravity = gravity


    def setup_bc(self, points):
        """Sets up the gravity vector."""
        super().setup_bc(points)
        self.g = self.rho * self.gravity(points)

    def __call__(self, u, points, jacobians):
        """Computes the loss.

        Args:
            u (torch.Tensor): The solution tensor (displacement vector) of shape [batch_size, output_size, num_points].
            points (torch.Tensor): The points in the (global) domain of shape [num_points, dim].
            jacobians (torch.Tensor): The Jacobians of the transformation functions of shape [batch_size, num_points, dim, dim].
            
        Returns:
            tuple: The inner loss and boundary loss.

        """
        # Assemble: sigma = lamb * div(u) * I
        divu = div(u, points, jacobians).transpose(-1, -2).unsqueeze(-1)
        divuI = divu * torch.eye(self.dim)
        divuI = divuI.permute(0, 2, 1, 3)
        sigma = self.lamb * divuI

        # Assemble: sigma += mu * (grad(u) + grad(u)^T)
        gradu = grad(u, points, jacobians)
        graduT = torch.transpose(gradu, 1, 3)
        sigma += self.mu * (gradu + graduT)
        
        # Compute divergence of sigma
        divsigma = div(sigma, points, jacobians)

        # Compute loss
        inner_loss = ((- divsigma - self.g)**2).mean()

        # Compute Dirichlet boundary loss
        boundary_loss = 0
        if self.has_dirichlet():
            u_boundary = u[:, :, self.dirichlet_indices].transpose(1, 2)
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
            sigmas = sigma[:, :, self.neumann_indices]
            sigmas = sigmas.transpose(-2, -3)
            sigmau_n = torch.matmul(sigmas, phi_n).squeeze(-1)
            boundary_loss += ((sigmau_n - self.neumann_values)**2).mean()

        return inner_loss, boundary_loss
