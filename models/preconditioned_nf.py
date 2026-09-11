"""Exactly normalized conditional RealNVP with a train-only covariance map.

The active affine span uses a conditional RealNVP. Orthogonal coordinates have
the exact Gaussian smoothing density. This is an invertible D-dimensional map
at every positive scale, with its complete change-of-variables determinant.
"""
from dataclasses import replace
import torch
from torch import nn
from models.normalizing_flow import ScaleConditionedRealNVP


class CovariancePreconditionedNF(ScaleConditionedRealNVP):
    def __init__(self, architecture, training_config):
        nn.Module.__init__(self)
        self.config = architecture
        rank = training_config.field_projection_rank
        if rank is None or not 2 <= rank <= architecture.ambient_dim:
            raise ValueError('NF covariance rank must lie between 2 and D')
        self.core = ScaleConditionedRealNVP(replace(architecture, ambient_dim=rank))
        self.register_buffer('basis', torch.zeros(architecture.ambient_dim, rank))
        self.register_buffer('normal_basis', torch.zeros(architecture.ambient_dim, architecture.ambient_dim-rank))
        self.register_buffer('variances', torch.ones(rank))
        self.register_buffer('omitted_variance_fraction', torch.zeros(1))

    def _scales(self, epsilon, reference):
        value = torch.as_tensor(epsilon, dtype=reference.dtype, device=reference.device)
        if value.ndim == 0: value = value.expand(len(reference))
        if value.ndim == 2 and value.shape[1] == 1: value = value[:, 0]
        if value.shape != (len(reference),) or not torch.isfinite(value).all() or torch.any(value <= 0):
            raise ValueError('positive finite scalar or per-example epsilon required')
        return value[:, None]

    def encode(self, observations, epsilon):
        x = self._flatten(observations)
        scale = self._scales(epsilon, x)
        standard = torch.sqrt(self.variances + scale.square())
        active, determinant = self.core.encode((x @ self.basis) / standard, scale[:, 0])
        normal = (x @ self.normal_basis) / scale
        determinant = determinant - standard.log().sum(1) - self.normal_basis.shape[1] * scale[:, 0].log()
        return torch.cat((active, normal), 1), determinant

    def decode(self, latent, epsilon):
        z = self._flatten(latent)
        scale = self._scales(epsilon, z)
        standard = torch.sqrt(self.variances + scale.square())
        rank = self.basis.shape[1]
        active, determinant = self.core.decode(z[:, :rank], scale[:, 0])
        x = (active * standard) @ self.basis.T + (z[:, rank:] * scale) @ self.normal_basis.T
        determinant = determinant + standard.log().sum(1) + self.normal_basis.shape[1] * scale[:, 0].log()
        return x, determinant


class AmbientIsotropicNF(ScaleConditionedRealNVP):
    """Full-dimensional invertible flow; only scalar schedule normalization."""
    def _standard(self, epsilon, reference):
        scale = torch.as_tensor(epsilon, dtype=reference.dtype, device=reference.device)
        if scale.ndim == 0: scale = scale.expand(len(reference))
        if scale.ndim == 2 and scale.shape[1] == 1: scale = scale[:, 0]
        if scale.shape != (len(reference),) or not torch.isfinite(scale).all() or torch.any(scale <= 0):
            raise ValueError('positive finite scalar or per-example epsilon required')
        return torch.sqrt(1 + scale[:, None].square())

    def encode(self, observations, epsilon):
        x = self._flatten(observations)
        standard = self._standard(epsilon, x)
        latent, determinant = super().encode(x / standard, epsilon)
        return latent, determinant - self.config.ambient_dim * standard[:, 0].log()

    def decode(self, latent, epsilon):
        z = self._flatten(latent)
        standard = self._standard(epsilon, z)
        x, determinant = super().decode(z, epsilon)
        return x * standard, determinant + self.config.ambient_dim * standard[:, 0].log()


def build_nf(architecture, config):
    if config.field_preconditioning == 'image_gaussian_v1':
        from models.image_normalizing_flow import ImageConditionedRealNVP
        return ImageConditionedRealNVP(architecture,config)
    if config.field_preconditioning == 'covariance_span_v1':
        return CovariancePreconditionedNF(architecture, config)
    if config.field_preconditioning == 'ambient_isotropic_v1':
        return AmbientIsotropicNF(architecture)
    if config.field_preconditioning is not None:
        raise ValueError('NF supports only covariance_span_v1')
    return ScaleConditionedRealNVP(architecture)
