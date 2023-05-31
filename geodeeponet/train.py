import torch
import time
import numpy as np
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


def train_model(geom, model, collocation_points, phis, pde, loss_points,
                tolerance=1e-5, steps=200, print_every=1, plot_phis=False):
    """Trains a physics-informed GeoDeepONet model.

    Args:
        geom (Geometry): The geometry of the domain.
        model (GeoDeepONet): The GeoDeepONet model.
        collocation_points (torch.Tensor): The collocation points.
        phis (list of nn.Module): The transformation function.
        pde (PDE): The partial differential equation.
        loss_points (torch.Tensor): The points to evaluate the loss at.
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

    # Get global collocation and loss points
    global_collocation_points = []
    jacobians = []
    for phi in phis:
        global_collocation_points += [phi.inv(collocation_points)]
        global_loss_points = phi.inv(loss_points)
        jac = torch.autograd.functional.jacobian(phi, global_loss_points)
        jacobians += [jac.sum(axis=2)] # type: ignore

    global_collocation_points = torch.stack(global_collocation_points)
    jacobians = torch.stack(jacobians)

    # Define closure
    def closure():
        optimizer.zero_grad()
        inner_loss, boundary_loss = compute_losses(model, pde, global_collocation_points, loss_points, jacobians)
        pde_loss = inner_loss + boundary_loss
        pde_loss.backward(retain_graph=True)
        return pde_loss

    for i in range(steps):
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
    print("")

    # Plot solutions on tensorboard
    if plot_phis:
        print("Plotting...")
        for i, phi in enumerate(phis):
            plot_solution(geom, model, collocation_points, phi, writer=writer, step=i)

    writer.close()
