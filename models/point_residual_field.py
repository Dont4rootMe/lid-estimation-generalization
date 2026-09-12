"""Shared residual field on point coordinates, with separate time conditioning."""
import math

import torch
from torch import nn

from models.spectral_residual_field import ConditionedResidualBlock


class PointResidualCore(nn.Module):
    """The spatial input is exactly x; no Fourier features or noise filtering."""

    def __init__(self, dimension, width=512, blocks=4, condition_dim=128):
        super().__init__()
        self.register_buffer('condition_frequencies', torch.exp(
            -math.log(10000)*torch.arange(condition_dim//2)/(condition_dim//2-1)))
        self.condition = nn.Sequential(nn.Linear(condition_dim, condition_dim),
                                       nn.SiLU(), nn.Linear(condition_dim, condition_dim))
        self.input = nn.Linear(dimension, width)
        self.blocks = nn.ModuleList(ConditionedResidualBlock(width, condition_dim)
                                    for _ in range(blocks))
        self.output = nn.Linear(width, dimension)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, condition):
        time = condition[:, None]*self.condition_frequencies
        embedded = self.condition(torch.cat((time.sin(), time.cos()), 1))
        h = self.input(x)
        for block in self.blocks:
            h = block(h, embedded)
        return self.output(torch.nn.functional.silu(h))
