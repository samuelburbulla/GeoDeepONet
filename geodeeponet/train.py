import torch
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from geodeeponet.plot import plot_solution


# Train physics-informed GeoDeepONet
def train_model(geom, model, collocation_points, phis, pde, loss_points,
                tolerance=1e-5, steps=1000, print_every=1):
    start_time = time.time()
    writer = SummaryWriter()

    # LBFGS optimizer
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        tolerance_grad=0,
        tolerance_change=0,
        line_search_fn='strong_wolfe'
    )

    # Helper function to compute losses
    phi_loss_points, phi_collocation_points = [], []
    for phi in phis:
        phi_loss_points += [phi(loss_points)]
        phi_collocation_points += [phi(collocation_points)]
    phi_loss_points = torch.stack(phi_loss_points)
    phi_collocation_points = torch.stack(phi_collocation_points)

    def compute_losses():
        outputs = model((phi_collocation_points, phi_loss_points))
        loss, bc = pde(outputs, phi_loss_points)
        return loss.mean(), bc.mean() 

    # Define closure
    def closure():
        optimizer.zero_grad()
        inner_loss, boundary_loss = compute_losses()
        pde_loss = inner_loss + boundary_loss
        pde_loss.backward(retain_graph=True)
        return pde_loss

    for i in range(steps):
        optimizer.step(closure)

        # Print losses
        if i % print_every == print_every-1:
            step = i+1
            train_loss, train_boundary = compute_losses()

            # Add train loss to tensorboard
            writer.add_scalar("loss/train", train_loss, step)
            writer.add_scalar("boundary/train", train_boundary, step)
            writer.flush()

            # Print to console
            steps_per_sec = step / (time.time() - start_time)
            print(
                f"\rStep {step}  Loss: {train_loss:.3e}  BC: {train_boundary:.3e}"
                f" ({steps_per_sec:.2f} steps/sec)", end=""
            )

            if train_loss < tolerance and train_boundary < tolerance:
                break

    # Plot solutions on tensorboard
    for i, phi in enumerate(phis):
        plot_solution(geom, model, collocation_points, phi, writer=writer, step=i)

    writer.close()
    print("")
