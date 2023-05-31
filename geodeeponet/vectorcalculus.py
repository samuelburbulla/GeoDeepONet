import torch
from torch.autograd import grad

class VectorCalculus:
    """Vector calculus for PyTorch with coordinate transformation.
    
    This class encapsulates the computation of the gradient and divergence 
    operations for batched inputs and applies chain rule for coordinate transformation.

    Attributes:
        coordinates (torch.Tensor): A tensor of shape [dim, num_points] representing 
                                    the coordinates over which operations are computed.
    """
    def __init__(self, coordinates, jacobians):
        """Initialize the UFL class with given coordinates.
        
        Args:
            coordinates (torch.Tensor): A tensor of shape [dim, num_points].
        """
        self.coordinates = coordinates
        self.jacobians = jacobians

    def _compute_grad(self, u, coordinate):
        """Compute gradient of u with respect to the given coordinate.
        
        Args:
            u (torch.Tensor): The tensor whose gradient is to be calculated.
            coordinate (torch.Tensor): The coordinate with respect to which the gradient is calculated.
        
        Returns:
            torch.Tensor: The gradient tensor.
        """
        g = grad(u, coordinate, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        # Apply jacobian of transformation
        mm = torch.bmm(self.jacobians[0], g.unsqueeze(-1))

        return mm.squeeze(-1)

    def grad(self, u):
        """Compute the gradient of u with respect to the coordinates.
        
        Args:
            u (torch.Tensor): A tensor of shape [batch_size, output_size, num_points].
        
        Returns:
            torch.Tensor: The gradient tensor of shape [batch_size, output_size, num_points, dim].
        """
        batch_size, output_size, num_points = u.shape
        dim = self.coordinates.shape[-1]
        grad_output = torch.zeros((batch_size, output_size, num_points, dim), dtype=torch.float)

        for i in range(batch_size):
            for j in range(output_size):
                grad_output[i, j] = self._compute_grad(u[i, j], self.coordinates)

        return grad_output

    def div(self, u):
        """Compute the divergence of u.
        
        Args:
            u (torch.Tensor): A tensor of shape [batch_size, dim, num_points].
        
        Returns:
            torch.Tensor: The divergence tensor of shape [batch_size, output_size, num_points].
        """
        if len(u.shape) == 3:
            # Reshape to match gradient output
            batch_size, output_size, num_points = u.shape
            u = u.view(batch_size, 1, num_points, output_size)

        batch_size, output_size, num_points, u_dim = u.shape
        assert u_dim == self.coordinates.shape[-1], "Dimensions mismatch for divergence computation"
        div_output = torch.zeros((batch_size, output_size, num_points), dtype=torch.float)

        for d in range(u_dim):
            div_output += self.grad(u[:, :, :, d])[:, :, :, d]

        return div_output

