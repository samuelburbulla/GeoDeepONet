import torch
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# Plot solution
def plot_solution(geom, model, collocation_points, phi, num_points=10000, writer="show", step=0):
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
        canvas = plt.gcf().canvas
        canvas.draw()
        img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(canvas.get_width_height()[::-1] + (3,))
        writer.add_image('plot_solution', img, step, dataformats='HWC')

    plt.close(fig)

