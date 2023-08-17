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
        assert (
            activation.lower() in ACTIVATIONS.keys()
        ), f"Expected one of {ACTIVATIONS.keys()}, got {activation}"

        super(MultiLayerPerceptron, self).__init__()
        modules = []
        # first layer manually
        modules.append(nn.Linear(input_dim, dims[0]))
        if(activation == "relu"):
                modules.append(nn.ReLU())

        # intermediate layers
        for i, dim in enumerate(dims[:-1]):
            modules.append(nn.Linear(dims[i], dims[i+1]))
            if(activation == "relu"):
                modules.append(nn.ReLU())

        # final prediction layer
        modules.append(nn.Linear(dims[-1], num_classes))
        modules.append(nn.Sigmoid())

        self.model = nn.Sequential(*modules)


    def forward(self, x: torch.Tensor):
        out = self.model(x)
        return out
