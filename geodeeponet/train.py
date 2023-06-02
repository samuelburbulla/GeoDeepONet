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


def train_model(geom, model, collocation_points, phis, pde, 
                num_inner_points=128, num_boundary_points=128,
                tolerance=1e-5, steps=200, print_every=1, plot_phis=False):
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
        print_every (int, optional): The frequency of printing the loss. Defaults to 1.

    """
    start_time = time.time()
    writer = SummaryWriter()

    # LBFGS optimizer
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        tolerance_grad=0,
        tolerance_change=0,
        line_search_fn='strong_wolfe',
    )

    # Get global collocation
    global_collocation_points = torch.stack([
        phi.inv(collocation_points) for phi in phis
    ])

    i = 0
    for i in range(steps):

        # Sample loss points
        inner_points = geom.random_points(num_inner_points)
        boundary_points = geom.random_boundary_points(num_boundary_points)
        loss_points = torch.cat([inner_points, boundary_points])

        # Setup boundary condition
        pde.setup_bc(loss_points)

        # Compute Jacobians
        jacobians = torch.stack([
            torch.autograd.functional.jacobian(phi, phi.inv(loss_points))
              .sum(axis=2).transpose(-1, -2) # type: ignore
            for phi in phis
        ])

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
        if i % print_every == print_every-1:
            step = i+1
            train_loss, train_boundary = compute_losses(model, pde, global_collocation_points, loss_points, jacobians)

            # Add train loss to tensorboard
            writer.add_scalar("loss/train", train_loss, step)
            writer.add_scalar("boundary/train", train_boundary, step)
            writer.flush()

            # Print to console
            steps_per_sec = step / (time.time() - start_time)
            print(f"\rStep {step}  Loss: {train_loss:.3e}  BC: {train_boundary:.3e} ({steps_per_sec:.2f} steps/sec)", end="")

            if train_loss < tolerance and train_boundary < tolerance:
                break
    
    print(" - not converged!" if i+1 == steps else " - done")

    # Plot solutions on tensorboard
    if plot_phis:
        print("Plotting...")
        for i, phi in enumerate(phis):
            plot_solution(geom, model, collocation_points, phi, writer=writer, step=i) # type: ignore

    writer.close()
