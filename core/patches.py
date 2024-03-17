import torch
from torch import nn


class Patch(nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device

    def to(self, device):
        super().to(device)


class BaseModulationPatch(Patch):
    def __init__(self, conv_weight: torch.Tensor):
        super().__init__()
        self.shape = conv_weight.shape
        self.register_buffer('ones', torch.ones(self.shape))
        _, self.c_out, self.c_in, k_x, k_y = self.shape

    def forward(self, weight, offsets):
        raise NotImplementedError()


class DomainModulationPatch(BaseModulationPatch):
    def forward(self, weight, params):
        mult = self.ones + params
        return weight * mult
