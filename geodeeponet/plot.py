import torch
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def plot_solution(geom, model, collocation_points, phi, num_points=10000, writer=None, step=0):
    """Plots the solution of a PDE on a geometric domain.

    Args:
        geom (Geometry): The geometry of the domain.
        model (nn.Module): The neural network model.
        collocation_points (torch.Tensor): The collocation points.
        phi (nn.Module): The parameterization function.
        num_points (int, optional): The number of points to evaluate the solution at. Defaults to 10000.
        writer (str, optional): The Tensorboard writer or "show". Defaults to None.
        step (int, optional): The current step. Defaults to 0.

    """
    if writer is None:
        return
    
    # Evaluate operator
    xs = geom.uniform_points(num_points=num_points)
    u = model((phi(collocation_points), phi(xs)))
    phix = phi(xs)

    # Detach tensors
    phix = phix.detach().numpy()
    u = u.detach().numpy()

    # Scatter plot
    fig = plt.figure()
    plt.scatter(phix[:, 0], phix[:, 1], c=u, s=10)
    plt.axis("equal")
    plt.colorbar()

    if writer == "show":
        plt.savefig("solution.png", dpi=300)
    else:
        # Add figure to tensorboard
        canvas = fig.canvas
        canvas.draw()
        img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8) # type: ignore
        img = img.reshape(canvas.get_width_height()[::-1] + (3,))
        writer.add_image('plot_solution', img, step, dataformats='HWC')

    plt.close(fig)
