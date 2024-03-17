import torch
from torch import nn


class ChannelIn(nn.Module):
    def __init__(self, conv_dimension):
        super().__init__()
        c_in, c_out = conv_dimension
        self.params_in = nn.Parameter(torch.zeros(1, 1, c_in, 1, 1))

    def forward(self):
        return {
            "in": self.params_in
        }


class Parametrization(nn.Module):
    def __init__(self, conv_dimensions):
        super().__init__()
        self.heads = nn.ModuleDict({
            f"conv_{idx}": ChannelIn(conv_dimension) for
            idx, conv_dimension in enumerate(conv_dimensions)
        })

    def forward(self):
        return {key: model() for key, model in self.heads.items()}
