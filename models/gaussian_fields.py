"""Gaussian residual parameterization for native VE and rectified flow.

For Y=a X+b Z and s^2=a^2+b^2, learn F(Y/s,log(b/a)) with
target (b X-a Z)/s. The analytic Gaussian part carries every ambient
direction; the learned residual has no unconditioned raw-input output skip.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from models.vp_baseline import ConditionedBottleneckMLP


class NativeGaussianBottleneck(ConditionedBottleneckMLP):
    def __init__(self, *args, native_family: str, **kwargs):
        super().__init__(*args, **kwargs)
        if native_family not in {"gaussian_diffusion", "rectified_flow"}:
            raise ValueError("native Gaussian field supports VE and rectified flow")
        expected = "log" if native_family == "gaussian_diffusion" else "linear"
        if self.config.condition_transform != expected:
            raise ValueError("native condition transform mismatch")
        self.native_family = native_family
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def _final_skip(self, skip: Tensor) -> Tensor:
        return torch.cat((torch.zeros_like(skip[:, :self.ambient_dim]),
                          skip[:, self.ambient_dim:]), dim=1)

    def coefficients(self, inputs: Tensor, condition: Tensor):
        condition = torch.as_tensor(condition, device=inputs.device, dtype=inputs.dtype)
        if condition.ndim == 0:
            condition = condition.expand(len(inputs))
        if condition.shape != (len(inputs),):
            raise ValueError("native condition must be scalar or (batch,)")
        if self.native_family == "gaussian_diffusion":
            a, b = torch.ones_like(condition), condition
        else:
            a, b = condition, 1 - condition
        if not torch.all(torch.isfinite(a) & torch.isfinite(b) & (a > 0) & (b > 0)):
            raise ValueError("native condition must give finite positive a and b")
        a, b = a[:, None], b[:, None]
        inverse = (a.square() + b.square()).rsqrt()
        return a, b, inverse

    def residual(self, inputs: Tensor, noise_ratio: Tensor) -> Tensor:
        # The legacy VE backbone applies log internally; RF does not.
        condition = noise_ratio if self.native_family == "gaussian_diffusion" else noise_ratio.log()
        return super().forward(inputs, condition)

    def forward(self, inputs: Tensor, condition: Tensor) -> Tensor:
        a, b, inverse = self.coefficients(inputs, condition)
        residual = self.residual(inputs * inverse, (b / a).flatten())
        if self.native_family == "gaussian_diffusion":
            return a * inverse.square() * inputs + b * inverse * residual
        return (a - b) * inverse.square() * inputs + inverse * residual

    def residual_loss(self, clean: Tensor, condition: Tensor, noise: Tensor) -> Tensor:
        a, b, inverse = self.coefficients(clean, condition)
        predicted = self.residual((a * clean + b * noise) * inverse, (b / a).flatten())
        target = (b * clean - a * noise) * inverse
        return ((predicted - target).square() * inverse.square()).mean()
