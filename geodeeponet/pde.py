import torch
from torch.autograd import grad


# Poisson's equation (2D) with Dirichlet boundary conditions
class Poisson:
    def __init__(self, bc, source=0):
        self.bc = bc
        self.source = source if callable(source) else lambda x: source

    def __call__(self, u, points, phi_points):
        inner_loss = 0

        def kw(w):
            return {
                "grad_outputs": torch.ones_like(w),
                "create_graph": True,
            }

        # Compute derivatives
        du = grad(u, phi_points, **kw(u))[0]
        u_xx = grad(du[:, 0], phi_points, **kw(du[:, 0]))[0][:, 0]
        u_yy = grad(du[:, 1], phi_points, **kw(du[:, 1]))[0][:, 1]
        laplace_u = u_xx + u_yy

        # Evaluate source term
        q = torch.tensor(self.source(points))

        # Compute inner loss
        inner_loss += (- laplace_u - q).norm()**2

        # Compute boundary loss
        boundary_loss = 0
        for i, x in enumerate(points):
            if self.bc.is_dirichlet(x):
                u_d = self.bc.value(x)
                boundary_loss += (u[:, i][0] - u_d)**2

        return inner_loss, boundary_loss
