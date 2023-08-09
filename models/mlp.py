import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List

ACTIVATIONS = {"relu": F.relu}


class MultiLayerPerceptron(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        dims: List = [768, 512],
        activation: str = "relu",
        num_classes: int = 2,
    ):
        assert num_classes > 1, "Please provide more than a single class."
        assert (
            activation.lower() in ACTIVATIONS.keys()
        ), f"Expected one of {ACTIVATIONS.keys()}, got {activation}"

        super(MultiLayerPerceptron, self).__init__()
        self.modules = []
        # first layer manually
        self.modules.append(nn.Linear(input_dim, dims[0]))

        # intermediate layers
        for i, dim in enumerate(dims[:-1]):
            self.modules.append(nn.Linear(dims[i], dims[i+1]))

        # final prediction layer
        self.modules.append(nn.Linear(dims[-1], num_classes))

        self.activation = ACTIVATIONS[activation]

    def forward(self, x: torch.Tensor):
        for i, module in enumerate(self.modules):
            x = module(x)
            if(i == len(self.modules)-1):
                x = F.sigmoid(x)
            else:
                x = self.activation(x)

        return x
