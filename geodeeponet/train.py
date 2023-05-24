import torch


# Train physics-informed GeoDeepONet
def train_model(geom, model, collocation_points, phi, pde,
                integration_points, test_points,
                tolerance=1e-4, steps=1000, print_every=1):

    loss_points = geom.uniform_points(num_points=integration_points)
    test_points = geom.uniform_points(num_points=test_points)
    optimizer = torch.optim.LBFGS(model.parameters())

    for i in range(steps):
        def compute_losses(collocation, evaluation):
            phi_points = phi(evaluation)
            outputs = model((phi(collocation), phi_points))
            return pde(outputs, evaluation, phi_points)

        def closure():
            optimizer.zero_grad()
            inner_loss, boundary_loss = compute_losses(
                collocation_points, loss_points
            )
            pde_loss = inner_loss + boundary_loss
            pde_loss.backward()
            return pde_loss

        optimizer.step(closure)

        # Print losses
        if i % print_every == print_every-1:
            train_loss, train_boundary = compute_losses(collocation_points,
                                                        loss_points)
            test_loss, test_boundary = compute_losses(collocation_points,
                                                      test_points)
            print(
                f"\rStep {i + 1}  Train: {train_loss:.3e} (BC {train_boundary:.3e})"
                f"  Test: {test_loss:.3e} (BC {test_boundary:.3e})",
                end=""
            )

            if test_loss < tolerance and test_boundary < tolerance:
                break

    print("")
