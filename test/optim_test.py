"""Test optimizers and approximation property of DeepONet."""
import unittest
import torch
import geodeeponet as gdn
import deepxde as dde


class TestOptim(unittest.TestCase):

    # Test optimizer
    def test_optimizer(self):
        for use_lbfgs in [True, False]:
          for width in [1, 2, 4, 8, 16, 32]:

            num_collocation_points = 1
            c = torch.ones(num_collocation_points, 2)
            model = gdn.deeponet.GeoDeepONet(
                branch_width=1,
                trunk_width=width,
                num_collocation_points=num_collocation_points,
                dimension=2,
            )

            def f(x):
                return torch.sin(2 * torch.pi * x[:, 0]) * torch.sin(2 * torch.pi * x[:, 1])

            if use_lbfgs:
                optimizer = torch.optim.LBFGS(
                    model.parameters(),
                    line_search_fn="strong_wolfe",
                )
                max_steps = 1000
            else:
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=1e-3,
                )
                max_steps = 30_000

            
            # Optimize
            num_loss_points = 128
            x = torch.rand(num_loss_points, 2)
            fx = f(x).reshape(num_loss_points)
            i = 0
            for i in range(max_steps):

                # Define closure
                def closure():
                    optimizer.zero_grad()
                    y = model((c, x)).reshape(num_loss_points)
                    loss = ((y - fx)**2).mean()
                    loss.backward()
                    return loss

                # Print (test) loss
                loss = closure().item()
                print(f"\rWidth {width:3d} "
                      f"  Step {i+1:3d}/{max_steps:3d} "
                      f"  Loss {loss:.3e}", end="")
                
                # Check convergence
                if loss < 1e-5:
                    break
                
                # Otherwise, optimize
                optimizer.step(closure)

            if i == max_steps-1:
                print(" - not converged!")
                assert width < 8 or use_lbfgs, "width >= 8 is expected to converge"
            else:
                print(" - success")


if __name__ == '__main__':
    unittest.main()
