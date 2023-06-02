"""Test grad and div with transformation jacobians."""
import unittest
import torch
from geodeeponet.grad import grad, div

class TestGrad(unittest.TestCase):

    # Test grad function
    def test_grad(self):

        x = torch.tensor([
            [0.5, 1.0],
            [1.0, 0.25]
        ], requires_grad=True)
        num_points = x.shape[0]

        u = torch.stack([
            x[:, 0]**2 * x[:, 1],
            2 * x[:, 1]**2]
        ).unsqueeze(0)

        jx = 1.0
        jacs = torch.tensor([[
                [[jx, 0.0], [0.0, 1.0]]
            ] * num_points
        ])

        # grad u = [[ 2 * x * y, x**2 ], [ 0, 4 * y ]]
        expected_grad_output = torch.tensor([
            # First point      # Second point
            [[jx * 1.0, 0.25], [jx * 0.5, 1.0]],
            [[0.0, 4.0],       [0.0, 1.0]],
        ])

        # Test grad function
        grad_output = grad(u, x, jacs)
        assert torch.allclose(grad_output, expected_grad_output), "grad function test failed"

        # Test div function

        # div grad u = [ 2 * y, 4 ]
        expected_div_output = torch.tensor([[
            # First point   # Second point
            [ 2.0,          0.5 ],
            [ 4.0,          4.0 ],
        ]])

        div_output = div(grad_output, x, jacs)
        assert torch.allclose(div_output, expected_div_output), "div function test failed"

if __name__ == '__main__':
    unittest.main()