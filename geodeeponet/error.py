import torch

def l2(geom, model, collocation_points, phi, exact_solution, num_points=1024, tol=1e-8):
    """Computes the L2 error of a PDE solution.

    Args:
        geom (Geometry): The geometry of the domain.
        model (nn.Module): The neural network model.
        collocation_points (torch.Tensor): The collocation points.
        phi (nn.Module): The parameterization function.
        exact_solution (callable): The exact solution.
        num_points (int, optional): The number of points to evaluate the solution at randomly at once.
    """

    # Initialize error
    l2 = 0.
    last = -1.

    # Initialize sum and number of evaluations
    sum = 0.
    n = 0

    # Iterate until convergence
    while abs(last - l2) > tol:
        # Update last
        last = l2

        # Evaluate operator
        xs = geom.random_points(num_points)
        u = model((phi.inv(collocation_points), xs))[0][0]

        # Evaluate exact solution
        phix = phi.inv(xs)
        u_exact = exact_solution(phix)

        # Compute sum of squared errors and number of evaluations
        sum += ((u - u_exact)**2).sum()
        n += num_points

        # Compute error
        l2 = (sum / n)**(1/2)

    # Return error
    return l2
