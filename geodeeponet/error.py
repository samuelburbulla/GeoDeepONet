import torch

def l2(geom, model, collocation_points, phi, exact_solution, num_points=10**6):
    """Computes the L2 error of a PDE solution.

    Args:
        geom (Geometry): The geometry of the domain.
        model (nn.Module): The neural network model.
        collocation_points (torch.Tensor): The collocation points.
        phi (nn.Module): The parameterization function.
        exact_solution (callable): The exact solution.
        num_points (int, optional): The number of points to evaluate the solution at. Defaults to 10**6.
    """
    # Evaluate operator
    xs = geom.uniform_points(num_points)
    u = model((phi.inv(collocation_points), xs))[0][0]

    # Evaluate exact solution
    phix = phi.inv(xs)
    u_exact = exact_solution(phix)

    # Approximate error integral
    intl2 = (((u - u_exact)**2).sum() / num_points)**(1/2)

    # Return error
    return intl2
