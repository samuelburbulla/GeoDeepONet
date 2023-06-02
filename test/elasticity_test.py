"""Test PDE loss of linear elasticity using an analytical solution."""
import unittest
import geodeeponet as gdn
from torch import sin, pi
import torch

class TestElasticityProblem(unittest.TestCase):

    def test_pde_loss(self):
        geom = gdn.geometry.UnitCube(2)
        loss_points = geom.uniform_points(100)
        # loss_points = torch.tensor([[.1, .1], [.1, .1], [.1, .1]], requires_grad=True)

        # Define exact solution with corresponding source term
        def exact(points):
            x, y = points[..., 0], points[..., 1]
            u0 = x**2 * y**2
            u1 = torch.zeros_like(u0)
            return torch.stack([u0, u1])

        def g(points):
            x, y = points[..., 0], points[..., 1]
            # (gradu + gradu^T) / 2
            g0 = 2 * y**2 + x**2
            g1 = 2 * x * y
            # divu I
            g0 += 2 * y**2
            g1 += 4 * x * y
            return -torch.stack([g0, g1])

        # Define boundary condition
        bc = gdn.bc.UnitCubeDirichletBC({
            w: lambda x: exact(x) for w in ["left", "right", "top", "bottom"]
        })

        # Setup PDE
        pde = gdn.pde.Elasticity(bc, 2, gravity=g(loss_points))
        pde.setup_bc(loss_points)

        # Get Jacobians
        phi = gdn.transformation.Identity()
        jacobians = gdn.train.compute_jacobians([phi], loss_points)

        # Compute loss of exact solution
        u = exact(loss_points).unsqueeze(0)
        inner_loss, boundary_loss = pde(u, loss_points, jacobians)

        print(f"Inner loss: {inner_loss:.3e}")
        print(f"Boundary loss: {boundary_loss:.3e}")
        assert inner_loss < 1e-10



if __name__ == '__main__':
    # unittest.main()
    TestElasticityProblem().test_pde_loss()