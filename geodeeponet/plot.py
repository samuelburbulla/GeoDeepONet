import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.tri as tri


def plot_solution(geom, model, collocation_points, phi, num_points=5**6, writer=None, step=0):
    """Plots the solution of a PDE on a geometric domain.
       For now, we assume a grid-like structure of the uniform points in a unit square.

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
    u = model((phi.inv(collocation_points), xs))
    phix = phi.inv(xs)

    # Detach tensors
    phix = phix.detach().numpy()
    u = u.detach().numpy()[0][0]

    # Plot
    fig = plt.figure()

    if geom.dim == 1:
        raise NotImplementedError("1D plotting not implemented yet.")

    if geom.dim == 2:
        # Generate triangles (assumes grid-like structure)
        n = m = int(np.sqrt(num_points))
        triangles = []
        for i in range(n-1):
            for j in range(m-1):
                triangles.append([i*m+j, i*m+j+1, (i+1)*m+j])
                triangles.append([i*m+j+1, (i+1)*m+j, (i+1)*m+j+1])
        triangles = np.array(triangles)
        x, y = phix[:, 0], phix[:, 1]
        triang = tri.Triangulation(x, y, triangles)
        plt.tricontourf(triang, u)
        plt.colorbar()

    if geom.dim == 3:
        x, y, z = phix[:, 0], phix[:, 1], phix[:, 2]
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(x, y, z, c=u, alpha=0.5, lw=0, s=30)
        plt.colorbar(scatter)

    plt.axis("equal")

    # Save figure
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
