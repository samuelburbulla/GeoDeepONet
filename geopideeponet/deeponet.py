import torch


# Branch network
class BranchNetwork(torch.nn.Module):
    def __init__(self, input_size, layer_width, output_size):
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
        u = self.fc1(u)
        u = torch.nn.functional.tanh(u)
        u = self.fc2(u)
        return u


# Trunk network
class TrunkNetwork(torch.nn.Module):
    def __init__(self, input_size, output_size):
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
        y = self.fc(y)
        y = torch.nn.functional.tanh(y)
        return y


# DeepONet
class DeepONet(torch.nn.Module):
    def __init__(self, branch, trunk):
        super(DeepONet, self).__init__()
        assert branch.output_size == trunk.output_size
        self.branch = branch
        self.trunk = trunk

    def forward(self, x):
        param, point = x

        # Reshape params
        param = param.reshape(1, self.branch.input_size)

        # Apply branch and trunk
        b = self.branch(param)
        t = self.trunk(point)

        # Batch-wise dot product
        return torch.einsum("bi,ni->bn", b, t)
