import torch
import torch.autograd


def gradient(u, x):
    """Helper function to compute gradients."""
    def kw(w):
        return {
            "grad_outputs": torch.ones_like(w),
            "create_graph": True,
        }
    return torch.autograd.grad(u, x, **kw(u))[0]


def div(u, x):
    """Helper function to compute the divergence."""
    _, num_points, dim = x.shape
    div = torch.zeros((1, num_points))
    for i in range(dim):
        div += gradient(u[:, :, i], x)[:, :, i]
    return div


def jacobian(u, x):
    """Helper function to compute jacobian."""
    dim = u.shape[0]
    jac = []
    for i in range(dim):
        jac += [gradient(u[i], x)[0]]
    return torch.stack(jac, dim=2)

