"""Test physics-informed GeoDeepONet using an analytical solution of the Poisson problem."""
import unittest
import geodeeponet as gdn
from torch import sin, pi, ones_like
import numpy as np


class TestPoissonProblem(unittest.TestCase):

    # Run test for given transformation, exact solution, source and boundary condition
    def _run(self, phi, exact_solution, source, bc):
        # Parameters
        dim = 2
        num_collocation_points = 2**dim
        branch_width = 1
        trunk_width = 256

        # Domain
        geom = gdn.geometry.UnitCube(dim)
        collocation_points = geom.uniform_points(num_collocation_points)

        # Define PDE
        pde = gdn.pde.Poisson(bc, lambda x: source(phi.inv(x)))

        # Setup DeepONet
        model = gdn.deeponet.GeoDeepONet(
            branch_width=branch_width,
            trunk_width=trunk_width,
            num_collocation_points=len(collocation_points),
            dimension=geom.dim,
            outputs=pde.outputs,
        )

        # Train model
        gdn.train.train_model(geom, model, collocation_points, [phi], pde)

        # Plot solution for one sample transformation
        gdn.plot.plot_solution(geom, model, collocation_points, phi, writer="show")

        # Compute error
        l2 = gdn.error.l2(geom, model, collocation_points, phi, exact_solution)
        print(f"L2-Error: {l2:.3e}")
        assert l2 < 1e-3


    # Analytical solution with Dirichlet boundary conditions
    def test_poisson(self):
        print("Test Poisson problem with Dirichlet boundary conditions on deformed domain")

        # Transformation
        phi = gdn.transformation.Affine(
            A=np.array([[1., 0.], [1., 1.]]),
            b=np.array([0., 0.]),
        )

        # Exact solution
        def exact_solution(x):
            return sin(2 * pi * x[..., 0]) * sin(2 * pi * x[..., 1]) / (8 * pi**2)

        # Source
        def source(x):
            return sin(2 * pi * x[..., 0]) * sin(2 * pi * x[..., 1])
        
        # Boundary condition
        bc = gdn.bc.UnitCubeDirichletBC({
            w: (lambda x: exact_solution(phi.inv(x)))
            for w in ["left", "right", "bottom", "top"]
        })

        self._run(phi, exact_solution, source, bc)


    # Analytical solution with Neumann boundary conditions
    def test_poisson_neumann(self):
        print("Test Poisson problem with Neumann boundary conditions")

        # Simple transformation
        phi = gdn.transformation.Affine(alpha=30)

        # Exact solution
        def exact_solution(y):
            x0 = phi(y)[..., 0]
            return (1 - x0) * x0 / 2

        # Source
        def source(x):
            return ones_like(x[..., 0])

        # Boundary condition
        local = lambda x: exact_solution(phi.inv(x))
        bc = gdn.bc.UnitCubeDirichletBC({
            "left": 0, "right": 0,
        })

        self._run(phi, exact_solution, source, bc)


if __name__ == '__main__':
    unittest.main()