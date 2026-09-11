"""Unrestricted vector posterior with scalar schedule-only neural scaling.

Every supplied coordinate is a learned input and output. There is no fitted
basis, covariance, rank estimate, normal complement, or support constraint.
The unit-isotropic coefficients initialize a network; they do not supply any
coordinate's final posterior. All native families share the same core.
"""
import torch
from torch import nn

from models.preconditioned_field import PreconditionedField
from models.spectral_residual_field import SpectralResidualCore


class AmbientPosteriorField(PreconditionedField):
    def __init__(self, architecture, family, training_config):
        nn.Module.__init__(self)
        self.config = architecture
        self.ambient_dim = architecture.ambient_dim
        self.family = family
        self.training_config = training_config
        if training_config.field_projection_rank is not None:
            raise ValueError('ambient fields must not declare a projection rank')
        self.core = SpectralResidualCore(self.ambient_dim,
            width=training_config.field_residual_width)

    def canonical_posterior(self, canonical, noise_ratio):
        scale = torch.as_tensor(noise_ratio, dtype=canonical.dtype, device=canonical.device)
        if scale.ndim == 0: scale = scale.expand(len(canonical))
        inverse = torch.rsqrt(1 + scale.square())[:, None]
        residual = self.core(canonical * inverse, scale.log(),
            (scale[:, None] * inverse).expand_as(canonical))
        mode = self.training_config.field_residual_scaling
        if mode == 'noise_v1': factor = scale[:, None] * inverse
        elif mode == 'gaussian_tail_v1': factor = scale[:, None] * inverse.square()
        elif mode == 'data_v1': factor = inverse.square()
        else: raise ValueError('unknown ambient residual scaling')
        return canonical * inverse.square() + factor * residual

    def posterior(self, inputs, condition):
        condition = torch.as_tensor(condition, dtype=inputs.dtype, device=inputs.device)
        if condition.ndim == 0: condition = condition.expand(len(inputs))
        scale, alpha = self.channel(condition)
        # Call the implementation directly to avoid re-entering native adapters.
        return AmbientPosteriorField.canonical_posterior(self, inputs / alpha[:, None], scale)
