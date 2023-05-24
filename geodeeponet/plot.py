import matplotlib.pyplot as plt


# Plot solution
def plot_solution(geom, model, collocation_points, phi, num_points=10000):
    # Evaluate operator
    xs = geom.uniform_points(num_points=num_points)
    u = model((phi(collocation_points), phi(xs)))
    phix = phi(xs)

    # Detach tensors
    phix = phix.detach().numpy()
    u = u.detach().numpy()

    # Scatter plot
    plt.scatter(phix[:, 0], phix[:, 1], c=u, s=10)
    plt.axis("equal")
    plt.colorbar()
    plt.show()
