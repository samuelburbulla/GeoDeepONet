import torch
import time
from torch.utils.tensorboard.writer import SummaryWriter
from geodeeponet.plot import plot_solution


def compute_losses(model, pde, global_collocation_points, loss_points, jacobians):
    """Computes the inner and boundary loss.

    Args:
        model (GeoDeepONet): The GeoDeepONet model.
        pde (PDE): The partial differential equation.
        global_collocation_points (torch.Tensor): The collocation points transformed to global coordinates.
        loss_points (torch.Tensor): The points to evaluate the loss at.
        jacobians (torch.Tensor): The Jacobians of the transformation functions.

    Returns:
        tuple: Inner loss and boundary loss.

    """
    outputs = model((global_collocation_points, loss_points))
    loss, bc = pde(outputs, loss_points, jacobians)
    return loss, bc


def compute_jacobians(phis, loss_points):
    """Computes the Jacobians of the transformation functions.

    Args:
        phis (list of nn.Module): The transformation functions.
        loss_points (torch.Tensor): The points to evaluate the Jacobians at.

    Returns:
        torch.Tensor: The jacobians.
    
    """
    
    return torch.stack([
        torch.autograd.functional.jacobian(phi, phi.inv(loss_points))
            .sum(axis=2).transpose(-1, -2) # type: ignore
        for phi in phis
    ])


def train_model(geom, model, collocation_points, phis, pde, 
                num_inner_points=128, num_boundary_points=128,
                tolerance=1e-5, steps=1000_000, print_every=1000,
                sample_every=100, plot_phis=False):
    """Trains a physics-informed GeoDeepONet model.

    Args:
        geom (Geometry): The geometry of the domain.
        model (GeoDeepONet): The GeoDeepONet model.
        collocation_points (torch.Tensor): The collocation points.
        phis (list of nn.Module): The transformation function.
        pde (PDE): The partial differential equation.
        num_inner_points (int, optional): The number of inner points to sample. Defaults to 512.
        num_boundary_points (int, optional): The number of boundary points to sample. Defaults to 128.
        tolerance (float, optional): The tolerance for the LBFGS optimizer. Defaults to 1e-5.
        steps (int, optional): The number of optimization steps. Defaults to 1000.
        print_every (int, optional): The frequency of printing the loss. Defaults to 100.
        sample_every (int, optional): The frequency of sampling new loss points. Defaults to 100.
        plot_phis (bool, optional): Whether to plot the transformation functions. Defaults to False.

    """
    start_time = time.time()
    writer = SummaryWriter()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Compute global collocation points
    global_collocation_points = torch.stack([
        phi.inv(collocation_points) for phi in phis
    ])

    loss_points, jacobians = None, None
    step = 0
    while step < steps:
        step += 1

        # Sample loss points
        if step % sample_every == 1:
            # Sample loss points
            inner_points = geom.random_points(num_inner_points)
            boundary_points = geom.random_boundary_points(num_boundary_points)
            loss_points = torch.cat([inner_points, boundary_points])

            # Setup boundary condition and compute test jacobians
            pde.setup_bc(loss_points)
            jacobians = compute_jacobians(phis, loss_points)

        # Define closure
        def closure():
            optimizer.zero_grad()
            inner_loss, boundary_loss = compute_losses(model, pde, global_collocation_points, loss_points, jacobians)
            pde_loss = inner_loss + boundary_loss
            pde_loss.backward(retain_graph=True)
            return pde_loss

        # Optimize
        optimizer.step(closure)

        # Print losses
        if step % print_every == 0:
            train_loss, train_boundary = compute_losses(model, pde, global_collocation_points, loss_points, jacobians)

            # Sample test points
            inner_test_points = geom.random_points(num_inner_points)
            boundary_test_points = geom.random_boundary_points(num_boundary_points)
            test_points = torch.cat([inner_test_points, boundary_test_points])

            # Setup boundary condition and compute test jacobians
            pde.setup_bc(test_points)
            jacobians_test = compute_jacobians(phis, test_points)

            # Compute test losses
            test_loss, test_boundary = compute_losses(model, pde, global_collocation_points, test_points, jacobians_test)

            # Add train loss to tensorboard
            writer.add_scalar("loss/train", train_loss, step)
            writer.add_scalar("boundary/train", train_boundary, step)
            writer.add_scalar("loss/test", test_loss, step)
            writer.add_scalar("boundary/test", test_boundary, step)
            writer.flush()

            # Print to console
            steps_per_sec = step / (time.time() - start_time)
            print(f"\rStep {step}  Train: {train_loss:.3e}  BC {train_boundary:.3e}   " \
                  f"Test: {test_loss:.3e}  BC {test_boundary:.3e}   " \
                  f"({steps_per_sec:.2f} steps/sec)", end="")

            if test_loss < tolerance and test_boundary < tolerance:
                break
    
    print(" - not converged!" if step == steps else " - done")

    # Plot solutions on tensorboard
    if plot_phis:
        print("Plotting...")
        for i, phi in enumerate(phis):
            plot_solution(geom, model, collocation_points, phi, writer=writer, step=i) # type: ignore

    writer.close()
