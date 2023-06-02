"""Vector derivatives for PyTorch with coordinate transformation.
This module encapsulates the computation of the gradient and divergence 
operations for batched inputs and applies chain rule for coordinate transformation.
"""
import torch


def grad(u, x, jacs):
    """Compute the gradient of u.
    
    Args:
        u (torch.Tensor): A tensor of shape [batch_size, output_size, num_points].
        x (torch.Tensor): A tensor of shape [batch_size, num_points, dim].
        jac (torch.Tensor): A tensor of shape [batch_size, num_points, dim, dim].
    
    Returns:
        torch.Tensor: The gradient tensor of shape [batch_size, output_size, num_points, dim].
    """
    batch_size, output_size, num_points = u.shape
    dim = x.shape[-1]
    grad_output = torch.zeros((batch_size, output_size, num_points, dim), dtype=torch.float64)

    def compute_grad(u, x, jacs):
        g = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        # Apply jacobian of transformation
        return torch.bmm(jacs[0], g.unsqueeze(-1)).squeeze(-1)

    for i in range(batch_size):
        for j in range(output_size):
            grad_output[i, j] = compute_grad(u[i, j], x, jacs)

    return grad_output


def div(u, x, jacs):
    """Compute the divergence of u.
    
    Args:
        u (torch.Tensor): A tensor of shape [batch_size, dim, num_points].
        x (torch.Tensor): A tensor of shape [batch_size, num_points, dim].
        jac (torch.Tensor): A tensor of shape [batch_size, num_points, dim, dim].
    
    Returns:
        torch.Tensor: The divergence tensor of shape [batch_size, output_size, num_points].
    """
    if len(u.shape) == 3:
        # Reshape to match gradient output
        batch_size, output_size, num_points = u.shape
        u = u.view(batch_size, 1, num_points, output_size)

    batch_size, output_size, num_points, u_dim = u.shape
    assert u_dim == x.shape[-1], "Dimensions mismatch for divergence computation"
    div_output = torch.zeros((batch_size, output_size, num_points), dtype=torch.float)

    for d in range(u_dim):
        div_output += grad(u[:, :, :, d], x, jacs)[:, :, :, d]

    return div_output

