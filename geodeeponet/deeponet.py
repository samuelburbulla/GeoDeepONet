import torch
import random
import numpy as np

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


class BranchNetwork(torch.nn.Module):
    """A class representing a branch network in a DeepONet model.

    Attributes:
        input_size (int): The size of the input layer.
        layer_width (int): The width of the hidden layer.
        output_size (int): The size of the output layer.

    Methods:
        forward(u):
            Computes the forward pass of the branch network.

    """

    def __init__(self, input_size, layer_width, output_size):
        """Initializes the BranchNetwork class.

        Args:
            input_size (int): The size of the input layer.
            layer_width (int): The width of the hidden layer.
            output_size (int): The size of the output layer.

        """
        super(BranchNetwork, self).__init__()
        self.input_size = input_size
        self.layer_width = layer_width
        self.output_size = output_size
        self.fc1 = torch.nn.Linear(
            self.input_size,
            self.layer_width,
            bias=True,
            dtype=torch.float64,
        )
        self.fc2 = torch.nn.Linear(
            self.layer_width,
            self.output_size,
            bias=False,
            dtype=torch.float64,
        )

    def forward(self, u):
        """Computes the forward pass of the branch network.

        Args:
            u (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        u = self.fc1(u)
        u = torch.nn.functional.tanh(u)
        u = self.fc2(u)
        return u


class TrunkNetwork(torch.nn.Module):
    """A class representing a trunk network in a DeepONet model.

    Attributes:
        input_size (int): The size of the input layer.
        output_size (int): The size of the output layer.

    Methods:
        forward(x):
            Computes the forward pass of the trunk network.

    """

    def __init__(self, input_size, output_size):
        """Initializes the TrunkNetwork class.

        Args:
            input_size (int): The size of the input layer.
            output_size (int): The size of the output layer.

        """
        super(TrunkNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc = torch.nn.Linear(
            self.input_size,
            self.output_size,
            bias=True,
            dtype=torch.float64,
        )

    def forward(self, y):
        """Computes the forward pass of the trunk network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        y = self.fc(y)
        y = torch.nn.functional.tanh(y)
        return y


class GeoDeepONet(torch.nn.Module):
    """A class representing a DeepONet model for solving PDEs on geometrically parameterised domains.

    Attributes:
        branch (BranchNetwork): The branch network of the DeepONet model.
        trunk (TrunkNetwork): The trunk network of the DeepONet model.

    Methods:
        forward(x):
            Computes the forward pass of the DeepONet model.

    """

    def __init__(self, branch_width, trunk_width, num_collocation_points, dimension):
        """Initializes the GeoDeepONet class.

        Args:
            branch_width (int): The width of the hidden layer in the branch network.
            trunk_width (int): The width of the output layer of the trunk network.
            num_collocation_points (int): The number of collocation points.
            dimension (int): The dimension of the domain.

        """
        super(GeoDeepONet, self).__init__()
        self.branch = BranchNetwork(
            input_size=num_collocation_points * dimension,
            layer_width=branch_width,
            output_size=trunk_width,
        )
        self.trunk = TrunkNetwork(
            input_size=dimension,
            output_size=trunk_width,
        )

    def forward(self, x):
        """Computes the forward pass of the DeepONet model.

        Args:
            x (tuple): A tuple containing the parameter tensor and the point tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        param, point = x

        # Convert param and point to batches, if necessary
        if len(param.shape) == 2:
            param = torch.stack([param])
        if len(point.shape) == 2:
            point = torch.stack([point])

        # Change view of params
        param = param.view(param.shape[0], 1, param.shape[1] * param.shape[2])

        # Subtract mean
        mean = torch.mean(param, dim=2, keepdim=True)
        param = param - mean

        # Apply branch and trunk
        b = self.branch(param)
        t = self.trunk(point)

        # Batch-wise dot product
        return torch.einsum("bki,bni->bkn", b, t)