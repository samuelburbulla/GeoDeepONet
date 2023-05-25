import torch
from torch.autograd import grad


# Poisson's equation (2D) with Dirichlet boundary conditions
class Poisson:
    def __init__(self, bc, source=0):
        self.bc = bc
        self.source = source if callable(source) else lambda x: source


    # Setup boundary condition
    def setup_bc(self, points):
        self.dirichlet_indices = []
        self.dirichlet_values = []
        for i, x in enumerate(points):
            if self.bc.is_dirichlet(x):
                self.dirichlet_indices += [i]
                self.dirichlet_values += [self.bc.value(x)]
        self.dirichlet_values = torch.tensor([self.dirichlet_values])


    # Compute loss
    def __call__(self, u, phi_points):

        def kw(w):
            return {
                "grad_outputs": torch.ones_like(w),
                "create_graph": True,
            }

        # Compute derivatives
        du = grad(u, phi_points, **kw(u))[0]
        u_xx = grad(du[:, :, 0], phi_points, **kw(du[:, :, 0]))[0][:, :, 0]
        u_yy = grad(du[:, :, 1], phi_points, **kw(du[:, :, 1]))[0][:, :, 1]
        laplace_u = u_xx + u_yy

        # Evaluate source term
        q = torch.tensor(self.source(phi_points))

        # Compute inner loss
        inner_loss = ((- laplace_u - q)**2).mean()

        # Compute boundary loss
        u_boundary = u[:, :, self.dirichlet_indices]
        u_dirichlet = self.dirichlet_values.repeat(u.shape[0], 1, 1)
        boundary_loss = ((u_boundary - u_dirichlet)**2).mean()

        return inner_loss, boundary_loss
