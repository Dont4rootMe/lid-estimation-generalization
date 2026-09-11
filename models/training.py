"""Deterministic training and inference for all trainable pilot families.

Five objectives are implemented:

* variance-exploding diffusion with an ``x0`` denoiser parameterization;
* linear rectified-flow matching from Gaussian noise to the data.
* Hydra-declared independent affine flow matching with either a direct
  velocity or posterior-mean target;
* an explicit generalized Brownian bridge with a terminal denoiser;
* exact-likelihood scale-conditioned RealNVP on Gaussian-smoothed data.

The diffusion denoiser is converted to a score only at inference time.  This
keeps training and divergence estimation numerically stable while preserving
the denoising-score-matching identity
``score(y, sigma) = (x0_hat(y, sigma) - y) / sigma**2``.
"""

from __future__ import annotations

import hashlib
import copy
import json
import math
import os
import random
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn

from models.affine_flow import (
    AffineFlowSpec,
    affine_flow_contract,
    affine_interpolant_and_target,
    affine_schedule_state,
    canonical_parameterization,
    flow_matching_loss_weights,
    posterior_divergence_to_channel_score_divergence,
    posterior_divergence_to_marginal_score_divergence,
    posterior_divergence_to_velocity_divergence,
    posterior_to_channel_score,
    posterior_to_marginal_score,
    posterior_to_velocity,
    sample_noise_ratio,
    schedule_condition,
    velocity_to_posterior,
)
from models.neural_fields import (
    NeuralFieldConfig,
    ScaleConditionedNeuralField,
    exact_divergence,
    hutchinson_divergence,
    rademacher_probes_like,
)
from models.gaussian_fields import NativeGaussianBottleneck
from models.image_gaussian_fields import ImageFieldConfig, NativeGaussianImageField
from models.normalizing_flow import (
    NF_DENSITY_CONTRACT,
    ConditionalFlowConfig,
    ScaleConditionedRealNVP,
    conditional_smoothed_nll,
    fixed_point_lid,
    fixed_point_lid_local_ols,
    fixed_point_likelihood_readouts,
    fixed_point_log_likelihood_curve,
)
from models.readouts import (
    diffusion_flipd,
    rectified_flow_full,
    rectified_flow_response,
    sb_forward_full,
    sb_forward_response,
)
from models.schrodinger_bridge import (
    BrownianBridgeSpec,
    brownian_bridge_contract,
    brownian_sb_terminal_denoising_loss,
    denoiser_to_forward_drift,
    validate_time_to_go_bounds,
)
from models.vp_baseline import (
    ConditionedBottleneckMLP,
    PreconditionedLogNoiseMLP,
    PreconditionedLogNoiseNoInputSkipMLP,
    VPBottleneckConfig,
    VPBottleneckMLP,
    VPSchedule,
    bottleneck_parameter_count,
    vp_loss,
    vp_primitives,
)

Family = Literal[
    "gaussian_diffusion",
    "vp_diffusion",
    "rectified_flow",
    "independent_affine_flow",
    "brownian_schrodinger_bridge",
    "scale_conditioned_normalizing_flow",
]
LogCallback = Callable[[Mapping[str, float | int | bool | str]], None]
CHECKPOINT_SCHEMA_VERSION = 2
LEGACY_CHECKPOINT_SCHEMA_VERSION = 1
FIXED_STEP_CHECKPOINT_SCHEMA_VERSION = 3
TRAINING_PROGRESS_SCHEMA_VERSION = 1
FIXED_STEP_PROGRESS_SCHEMA_VERSION = 2
IMAGE_PRECONDITIONING_MODES = frozenset({
    "unit_rms_gaussian_image_v1", "unit_rms_pixel_mixture_image_v1",
    "unit_rms_gaussian_tail_image_v1",
})
TRAINING_PROGRESS_INTERVAL_EPOCHS = 20
TrainableModel = (
    ScaleConditionedNeuralField | ScaleConditionedRealNVP | ConditionedBottleneckMLP
    | NativeGaussianImageField
)


@dataclass(frozen=True)
class TrainingConfig:
    """Hydra-friendly scalar settings for one model/dataset training run.

    ``num_workers`` is retained in the serialized experiment schema for
    launcher compatibility.  Training receives already materialized arrays
    and moves them to one contiguous tensor, so it intentionally creates no
    data-loader worker processes.
    """

    seed: int = 0
    device: str = "auto"
    epochs: int = 100
    training_mode: str = "epochs_v1"
    steps: int | None = None
    warmup_steps: int | None = None
    validation_interval_steps: int | None = None
    batch_size: int = 256
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    hidden_dim: int = 256
    depth: int | None = None
    time_embedding_dim: int = 64
    validation_interval: int = 1
    early_stopping_patience: int | None = 20
    gradient_clip_norm: float | None = 1.0
    num_workers: int = 0
    deterministic: bool = True
    sigma_min: float | None = None
    sigma_max: float | None = None
    time_min: float | None = None
    time_max: float | None = None
    fourier_features: int = 32
    max_condition_frequency: float = 100.0
    dropout: float = 0.0
    normalize: bool = True
    normalization_epsilon: float = 1.0e-8
    num_coupling_layers: int | None = None
    conditioner_depth: int | None = None
    log_scale_limit: float | None = None
    epsilon_min: float | None = None
    epsilon_max: float | None = None
    bridge_construction: str | None = None
    bridge_reference_process: str | None = None
    bridge_initial_marginal: str | None = None
    bridge_terminal_marginal: str | None = None
    bridge_factor_f: str | None = None
    bridge_factor_g: str | None = None
    bridge_conditioning: str | None = None
    bridge_diffusivity: float | None = None
    bridge_terminal_time: float | None = None
    bridge_tau_min: float | None = None
    bridge_tau_max: float | None = None
    flow_variant_id: str | None = None
    flow_schedule: str | None = None
    flow_parameterization: str | None = None
    flow_conditioning: str | None = None
    flow_scale_sampling: str | None = None
    flow_loss_weighting: str | None = None
    flow_noise_ratio_min: float | None = None
    flow_noise_ratio_max: float | None = None
    field_hidden_sizes: tuple[int, ...] | None = None
    field_preconditioning: str | None = None
    field_projection_rank: int | None = None
    field_backbone: str = 'bottleneck_v1'
    field_residual_scaling: str = 'noise_v1'
    field_residual_width: int = 512
    noise_pairing: str = 'iid'
    training_target: str = 'sample_v1'
    training_target_start_step: int = 0
    terminal_decay_steps: int = 0
    terminal_learning_rate_ratio: float = 0.01
    ema_decay: float | None = None
    ema_start_step: int = 0
    optimizer_schedule_policy: str = 'native_v1'
    posterior_preconditioning: str | None = None
    native_preconditioning: str | None = None
    image_shape: tuple[int, int, int] | None = None
    image_width: int = 32
    image_training_bf16: bool = False
    native_noise_pairing: str = "independent_v1"
    image_layout: str = 'nhwc'
    vp_hidden_sizes: tuple[int, ...] | None = None
    vp_beta_min: float | None = None
    vp_beta_max: float | None = None

    def __post_init__(self) -> None:
        if self.optimizer_schedule_policy not in {'native_v1', 'shared_terminal_v1'}:
            raise ValueError('unknown optimizer schedule policy')
        if self.optimizer_schedule_policy != 'native_v1' and (
            self.training_mode != 'fixed_steps_v1' or not self.terminal_decay_steps or self.warmup_steps != 0
        ):
            raise ValueError('shared optimizer policy requires explicit terminal decay and zero warmup')
        if self.field_preconditioning is not None and (self.native_preconditioning is not None or self.posterior_preconditioning is not None):
            raise ValueError('choose one field preconditioning interface')
        if self.training_target not in {'sample_v1','empirical_posterior_v1'}:
            raise ValueError('unknown training target estimator')
        if self.training_target != 'sample_v1' and (self.training_mode!='fixed_steps_v1' or self.num_coupling_layers is not None):
            raise ValueError('empirical posterior targets require fixed-step vector-field training')
        if isinstance(self.training_target_start_step,bool) or not isinstance(self.training_target_start_step,int) or self.training_target_start_step<0:
            raise ValueError('training target start must be a nonnegative integer')
        if self.training_target != 'sample_v1' and not self.training_target_start_step<self.steps:
            raise ValueError('training target must start before the budget ends')
        if isinstance(self.field_residual_width, bool) or not isinstance(self.field_residual_width, int) or self.field_residual_width <= 0:
            raise ValueError('field_residual_width must be a positive integer')
        if self.field_residual_width != 512 and self.field_backbone != 'spectral_residual_v1':
            raise ValueError('residual width applies only to the spectral residual backbone')
        if self.field_residual_scaling not in {'noise_v1', 'data_v1', 'gaussian_tail_v1'}:
            raise ValueError('unknown residual scaling')
        if self.field_residual_scaling != 'noise_v1' and self.field_preconditioning not in {'covariance_span_v1','image_gaussian_v1'}:
            raise ValueError('data residual scaling requires covariance-span fields')
        if self.field_residual_scaling != 'noise_v1' and self.num_coupling_layers is not None:
            raise ValueError('NF has no field residual scaling')
        if self.terminal_decay_steps or self.ema_decay is not None:
            if self.training_mode != 'fixed_steps_v1' or self.steps is None:
                raise ValueError('terminal decay and EMA require fixed-step training')
        if isinstance(self.terminal_decay_steps, bool) or not isinstance(self.terminal_decay_steps, int) or self.terminal_decay_steps < 0:
            raise ValueError('terminal_decay_steps must be a nonnegative integer')
        if self.terminal_decay_steps and self.terminal_decay_steps > self.steps:
            raise ValueError('terminal decay exceeds training budget')
        if not 0 < self.terminal_learning_rate_ratio <= 1:
            raise ValueError('terminal_learning_rate_ratio must lie in (0, 1]')
        if self.ema_decay is not None and not 0 <= self.ema_decay < 1:
            raise ValueError('ema_decay must lie in [0, 1)')
        if isinstance(self.ema_start_step, bool) or not isinstance(self.ema_start_step, int) or self.ema_start_step < 0:
            raise ValueError('ema_start_step must be a nonnegative integer')
        if self.ema_decay is not None and self.ema_start_step >= self.steps:
            raise ValueError('EMA must start before training ends')
        if self.noise_pairing not in {'iid','antithetic_v1'}:
            raise ValueError('unknown noise pairing')
        if self.noise_pairing == 'antithetic_v1' and self.batch_size % 2:
            raise ValueError('antithetic training requires even batch_size')
        if self.field_backbone not in {'bottleneck_v1','spectral_residual_v1','image_unet_v1'}:
            raise ValueError('unknown field backbone')
        if self.field_backbone == 'spectral_residual_v1' and self.field_preconditioning != 'covariance_span_v1':
            raise ValueError('spectral residual backbone requires covariance_span_v1')
        if self.field_backbone == 'image_unet_v1' and self.field_preconditioning != 'image_gaussian_v1':
            raise ValueError('image backbone requires image_gaussian_v1')
        if self.field_projection_rank is not None:
            if self.field_preconditioning != 'covariance_span_v1' or isinstance(self.field_projection_rank, bool) or not isinstance(self.field_projection_rank, int) or self.field_projection_rank <= 0:
                raise ValueError('projection rank requires covariance_span_v1 and a positive integer')
        if self.field_preconditioning is not None:
            if self.field_preconditioning not in {'gaussian_v1', 'gaussian_no_input_skip_v1', 'covariance_span_v1','image_gaussian_v1'}:
                raise ValueError('unknown field_preconditioning')
            is_covariance_nf = self.field_preconditioning in {'covariance_span_v1','image_gaussian_v1'} and self.num_coupling_layers is not None
            if not self.normalize or (self.field_hidden_sizes is None and self.vp_hidden_sizes is None and not is_covariance_nf):
                raise ValueError('field preconditioning requires a normalized bottleneck field')
        if self.native_noise_pairing not in {"independent_v1", "antithetic_v1"}:
            raise ValueError("unknown native_noise_pairing")
        if self.native_noise_pairing == "antithetic_v1":
            if self.native_preconditioning is None or self.batch_size % 2:
                raise ValueError("antithetic native noise requires preconditioning and an even batch")
        if self.native_preconditioning is not None:
            if self.native_preconditioning not in ({"unit_rms_gaussian_no_input_skip_v1"} | IMAGE_PRECONDITIONING_MODES):
                raise ValueError("unknown native_preconditioning")
            if self.posterior_preconditioning is not None or not self.normalize:
                raise ValueError("native preconditioning requires normalization and no posterior override")
            if self.native_preconditioning == "unit_rms_gaussian_no_input_skip_v1" and self.field_hidden_sizes is None:
                raise ValueError("native bottleneck preconditioning requires field_hidden_sizes")
        if self.image_layout not in {'nhwc','nchw'}:
            raise ValueError('unknown image layout')
        if self.field_preconditioning == 'image_gaussian_v1':
            if self.image_shape is None or self.field_backbone != 'image_unet_v1' or self.field_projection_rank is not None:
                raise ValueError('shared image fields require shape, image backbone, and no covariance projection')
            ImageFieldConfig(math.prod(self.image_shape),tuple(self.image_shape),self.image_width,self.image_training_bf16)
        elif self.native_preconditioning in IMAGE_PRECONDITIONING_MODES:
            if self.field_hidden_sizes is not None or self.image_shape is None:
                raise ValueError("native image field requires image_shape and no field_hidden_sizes")
            ImageFieldConfig(math.prod(self.image_shape),tuple(self.image_shape),self.image_width,self.image_training_bf16)
        elif self.image_shape is not None or self.image_width != 32 or self.image_training_bf16:
            raise ValueError("inactive image settings require the native image field")
        if self.posterior_preconditioning is not None:
            if self.posterior_preconditioning not in {
                "unit_rms_gaussian_v1", "unit_rms_gaussian_no_input_skip_v1"
            }:
                raise ValueError("unknown posterior_preconditioning")
            if not (
                self.flow_schedule == "log_noise"
                and self.flow_parameterization == "posterior_mean"
                and self.flow_conditioning == "log_noise_ratio"
                and self.flow_loss_weighting == "posterior_bias_equivalent"
                and self.field_hidden_sizes is not None
                and self.normalize
            ):
                raise ValueError("posterior preconditioning requires normalized log-noise posterior bottleneck")
        integer_positive = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "hidden_dim": self.hidden_dim,
            "time_embedding_dim": self.time_embedding_dim,
            "validation_interval": self.validation_interval,
            "fourier_features": self.fourier_features,
        }
        for name, value in integer_positive.items():
            if isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.training_mode not in {"epochs_v1", "fixed_steps_v1"}:
            raise ValueError("training_mode must be epochs_v1 or fixed_steps_v1")
        fixed_step_values = (
            self.steps,
            self.warmup_steps,
            self.validation_interval_steps,
        )
        if self.training_mode == "epochs_v1":
            if any(value is not None for value in fixed_step_values):
                raise ValueError(
                    "fixed-step settings require training_mode=fixed_steps_v1"
                )
        else:
            if any(value is None for value in fixed_step_values):
                raise ValueError(
                    "fixed_steps_v1 requires steps, warmup_steps, and "
                    "validation_interval_steps"
                )
            assert self.steps is not None
            assert self.warmup_steps is not None
            assert self.validation_interval_steps is not None
            if (
                isinstance(self.steps, bool)
                or not isinstance(self.steps, int)
                or self.steps <= 0
            ):
                raise ValueError("steps must be a positive integer")
            if (
                isinstance(self.warmup_steps, bool)
                or not isinstance(self.warmup_steps, int)
                or not 0 <= self.warmup_steps < self.steps
            ):
                raise ValueError("warmup_steps must be an integer in [0, steps)")
            if (
                isinstance(self.validation_interval_steps, bool)
                or not isinstance(self.validation_interval_steps, int)
                or self.validation_interval_steps <= 0
            ):
                raise ValueError("validation_interval_steps must be positive")
            if self.early_stopping_patience is not None:
                raise ValueError("fixed_steps_v1 forbids early stopping")
        if self.depth is not None and (
            isinstance(self.depth, bool)
            or not isinstance(self.depth, int)
            or self.depth <= 0
        ):
            raise ValueError("depth must be null or a positive integer")
        if isinstance(self.seed, bool) or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a non-empty string")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if self.early_stopping_patience is not None and (
            isinstance(self.early_stopping_patience, bool)
            or self.early_stopping_patience <= 0
        ):
            raise ValueError("early_stopping_patience must be null or positive")
        if self.gradient_clip_norm is not None and (
            not math.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0
        ):
            raise ValueError("gradient_clip_norm must be null or positive")
        if isinstance(self.num_workers, bool) or self.num_workers < 0:
            raise ValueError("num_workers must be a non-negative integer")
        if not isinstance(self.deterministic, bool):
            raise TypeError("deterministic must be boolean")
        sigma_values = (self.sigma_min, self.sigma_max)
        if any(value is not None for value in sigma_values):
            if any(value is None for value in sigma_values):
                raise ValueError("sigma_min and sigma_max must be provided together")
            assert self.sigma_min is not None and self.sigma_max is not None
            if not 0 < self.sigma_min < self.sigma_max:
                raise ValueError("sigma_min and sigma_max must satisfy 0 < min < max")
        time_values = (self.time_min, self.time_max)
        if any(value is not None for value in time_values):
            if any(value is None for value in time_values):
                raise ValueError("time_min and time_max must be provided together")
            assert self.time_min is not None and self.time_max is not None
            if not 0 <= self.time_min < self.time_max <= 1:
                raise ValueError("time bounds must satisfy 0 <= min < max <= 1")
        if (
            not math.isfinite(self.max_condition_frequency)
            or self.max_condition_frequency < 1
        ):
            raise ValueError("max_condition_frequency must be finite and >= 1")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")
        if not isinstance(self.normalize, bool):
            raise TypeError("normalize must be boolean")
        if (
            not math.isfinite(self.normalization_epsilon)
            or self.normalization_epsilon <= 0
        ):
            raise ValueError("normalization_epsilon must be finite and positive")
        nf_values = (
            self.num_coupling_layers,
            self.conditioner_depth,
            self.log_scale_limit,
            self.epsilon_min,
            self.epsilon_max,
        )
        if any(value is not None for value in nf_values):
            if any(value is None for value in nf_values):
                raise ValueError(
                    "scale-conditioned NF settings must be provided as one "
                    "complete block"
                )
            _nf_architecture_from_training_config(self, ambient_dim=2)
        bridge_values = (
            self.bridge_construction,
            self.bridge_reference_process,
            self.bridge_initial_marginal,
            self.bridge_terminal_marginal,
            self.bridge_factor_f,
            self.bridge_factor_g,
            self.bridge_conditioning,
            self.bridge_diffusivity,
            self.bridge_terminal_time,
            self.bridge_tau_min,
            self.bridge_tau_max,
        )
        if any(value is not None for value in bridge_values):
            if any(value is None for value in bridge_values):
                raise ValueError(
                    "Schrodinger-bridge settings must be provided as one complete block"
                )
            spec = _bridge_spec_from_training_config(self)
            assert self.bridge_tau_min is not None
            assert self.bridge_tau_max is not None
            validate_time_to_go_bounds(
                minimum=float(self.bridge_tau_min),
                maximum=float(self.bridge_tau_max),
                spec=spec,
            )
        affine_values = (
            self.flow_variant_id,
            self.flow_schedule,
            self.flow_parameterization,
            self.flow_conditioning,
            self.flow_scale_sampling,
            self.flow_loss_weighting,
            self.flow_noise_ratio_min,
            self.flow_noise_ratio_max,
        )
        if any(value is not None for value in affine_values):
            if any(value is None for value in affine_values):
                raise ValueError(
                    "independent affine-flow settings must be provided as one "
                    "complete block"
                )
            _affine_spec_from_training_config(self)
        vp_values = (self.vp_hidden_sizes, self.vp_beta_min, self.vp_beta_max)
        if any(value is not None for value in vp_values):
            if any(value is None for value in vp_values):
                raise ValueError("VP settings must be provided as one complete block")
            assert self.vp_hidden_sizes is not None
            object.__setattr__(self, "vp_hidden_sizes", tuple(self.vp_hidden_sizes))
            _vp_architecture_from_training_config(self, ambient_dim=2)
            _vp_schedule_from_training_config(self)
        if self.field_hidden_sizes is not None:
            object.__setattr__(
                self, "field_hidden_sizes", tuple(self.field_hidden_sizes)
            )
            _bottleneck_architecture_from_training_config(
                self, family="rectified_flow", ambient_dim=2
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TrainingConfig:
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown training settings: {sorted(unknown)}")
        fields = dict(value)
        if isinstance(fields.get("vp_hidden_sizes"), list):
            fields["vp_hidden_sizes"] = tuple(fields["vp_hidden_sizes"])
        if isinstance(fields.get("field_hidden_sizes"), list):
            fields["field_hidden_sizes"] = tuple(fields["field_hidden_sizes"])
        if isinstance(fields.get("image_shape"), list):
            fields["image_shape"] = tuple(fields["image_shape"])
        return cls(**fields)


def _bridge_spec_from_training_config(config: TrainingConfig) -> BrownianBridgeSpec:
    """Require and validate the complete Hydra-declared bridge identity."""

    fields = {
        "construction": config.bridge_construction,
        "reference_process": config.bridge_reference_process,
        "initial_marginal": config.bridge_initial_marginal,
        "terminal_marginal": config.bridge_terminal_marginal,
        "factor_f": config.bridge_factor_f,
        "factor_g": config.bridge_factor_g,
        "conditioning": config.bridge_conditioning,
        "diffusivity": config.bridge_diffusivity,
        "terminal_time": config.bridge_terminal_time,
    }
    missing = [name for name, value in fields.items() if value is None]
    if missing:
        raise ValueError(
            "brownian_schrodinger_bridge requires explicit Hydra settings: "
            f"{sorted(missing)}"
        )
    return BrownianBridgeSpec(
        construction=str(fields["construction"]),
        reference_process=str(fields["reference_process"]),
        initial_marginal=str(fields["initial_marginal"]),
        terminal_marginal=str(fields["terminal_marginal"]),
        factor_f=str(fields["factor_f"]),
        factor_g=str(fields["factor_g"]),
        conditioning=str(fields["conditioning"]),
        diffusivity=float(fields["diffusivity"]),
        terminal_time=float(fields["terminal_time"]),
    )


def _affine_spec_from_training_config(config: TrainingConfig) -> AffineFlowSpec:
    """Require the complete immutable Hydra-declared affine-FM identity."""

    fields = {
        "variant_id": config.flow_variant_id,
        "schedule": config.flow_schedule,
        "parameterization": config.flow_parameterization,
        "conditioning": config.flow_conditioning,
        "scale_sampling": config.flow_scale_sampling,
        "loss_weighting": config.flow_loss_weighting,
        "noise_ratio_min": config.flow_noise_ratio_min,
        "noise_ratio_max": config.flow_noise_ratio_max,
    }
    missing = [name for name, value in fields.items() if value is None]
    if missing:
        raise ValueError(
            "independent_affine_flow requires explicit Hydra settings: "
            f"{sorted(missing)}"
        )
    inactive = {
        "sigma_min": config.sigma_min,
        "sigma_max": config.sigma_max,
        "time_min": config.time_min,
        "time_max": config.time_max,
        "num_coupling_layers": config.num_coupling_layers,
        "conditioner_depth": config.conditioner_depth,
        "log_scale_limit": config.log_scale_limit,
        "epsilon_min": config.epsilon_min,
        "epsilon_max": config.epsilon_max,
        "bridge_construction": config.bridge_construction,
        "bridge_reference_process": config.bridge_reference_process,
        "bridge_initial_marginal": config.bridge_initial_marginal,
        "bridge_terminal_marginal": config.bridge_terminal_marginal,
        "bridge_factor_f": config.bridge_factor_f,
        "bridge_factor_g": config.bridge_factor_g,
        "bridge_conditioning": config.bridge_conditioning,
        "bridge_diffusivity": config.bridge_diffusivity,
        "bridge_terminal_time": config.bridge_terminal_time,
        "bridge_tau_min": config.bridge_tau_min,
        "bridge_tau_max": config.bridge_tau_max,
        "vp_hidden_sizes": config.vp_hidden_sizes,
        "vp_beta_min": config.vp_beta_min,
        "vp_beta_max": config.vp_beta_max,
    }
    present_inactive = [name for name, value in inactive.items() if value is not None]
    if present_inactive:
        raise ValueError(
            "independent_affine_flow forbids inactive family settings: "
            f"{sorted(present_inactive)}"
        )
    return AffineFlowSpec.from_mapping(fields)


def _has_affine_settings(config: TrainingConfig) -> bool:
    return any(
        value is not None
        for value in (
            config.flow_variant_id,
            config.flow_schedule,
            config.flow_parameterization,
            config.flow_conditioning,
            config.flow_scale_sampling,
            config.flow_loss_weighting,
            config.flow_noise_ratio_min,
            config.flow_noise_ratio_max,
        )
    )


def _has_vp_settings(config: TrainingConfig) -> bool:
    return any(
        value is not None
        for value in (
            config.vp_hidden_sizes,
            config.vp_beta_min,
            config.vp_beta_max,
        )
    )


def _vp_architecture_from_training_config(
    config: TrainingConfig, *, ambient_dim: int
) -> VPBottleneckConfig:
    if config.vp_hidden_sizes is None:
        raise ValueError("vp_diffusion requires explicit vp_hidden_sizes")
    return VPBottleneckConfig(
        ambient_dim=ambient_dim,
        hidden_sizes=tuple(config.vp_hidden_sizes),
        time_dim=config.time_embedding_dim,
    )


def _vp_schedule_from_training_config(config: TrainingConfig) -> VPSchedule:
    if config.vp_beta_min is None or config.vp_beta_max is None:
        raise ValueError("vp_diffusion requires explicit VP beta endpoints")
    inactive = {
        "sigma_min": config.sigma_min,
        "sigma_max": config.sigma_max,
        "time_min": config.time_min,
        "time_max": config.time_max,
        "num_coupling_layers": config.num_coupling_layers,
        "conditioner_depth": config.conditioner_depth,
        "log_scale_limit": config.log_scale_limit,
        "epsilon_min": config.epsilon_min,
        "epsilon_max": config.epsilon_max,
        "bridge_construction": config.bridge_construction,
        "bridge_reference_process": config.bridge_reference_process,
        "bridge_initial_marginal": config.bridge_initial_marginal,
        "bridge_terminal_marginal": config.bridge_terminal_marginal,
        "bridge_factor_f": config.bridge_factor_f,
        "bridge_factor_g": config.bridge_factor_g,
        "bridge_conditioning": config.bridge_conditioning,
        "bridge_diffusivity": config.bridge_diffusivity,
        "bridge_terminal_time": config.bridge_terminal_time,
        "bridge_tau_min": config.bridge_tau_min,
        "bridge_tau_max": config.bridge_tau_max,
        "flow_variant_id": config.flow_variant_id,
        "flow_schedule": config.flow_schedule,
        "flow_parameterization": config.flow_parameterization,
        "flow_conditioning": config.flow_conditioning,
        "flow_scale_sampling": config.flow_scale_sampling,
        "flow_loss_weighting": config.flow_loss_weighting,
        "flow_noise_ratio_min": config.flow_noise_ratio_min,
        "flow_noise_ratio_max": config.flow_noise_ratio_max,
        "field_hidden_sizes": config.field_hidden_sizes,
    }
    present = [name for name, value in inactive.items() if value is not None]
    if present:
        raise ValueError(f"vp_diffusion forbids inactive family settings: {present}")
    return VPSchedule(
        beta_min=float(config.vp_beta_min), beta_max=float(config.vp_beta_max)
    )


def _bottleneck_architecture_from_training_config(
    config: TrainingConfig, *, family: Family, ambient_dim: int
) -> VPBottleneckConfig:
    if config.field_hidden_sizes is None:
        raise ValueError("field_hidden_sizes are required for the bottleneck backbone")
    condition_transform: Literal["linear", "log"] = (
        "log"
        if family in {"gaussian_diffusion", "brownian_schrodinger_bridge"}
        else "linear"
    )
    return VPBottleneckConfig(
        ambient_dim=ambient_dim,
        hidden_sizes=tuple(config.field_hidden_sizes),
        time_dim=config.time_embedding_dim,
        condition_transform=condition_transform,
    )


def _image_architecture(config: TrainingConfig,ambient_dim: int) -> ImageFieldConfig:
    prior = ('pixel_spike_gaussian_v1' if config.native_preconditioning == 'unit_rms_pixel_mixture_image_v1'
             else 'unit_gaussian')
    return ImageFieldConfig(ambient_dim,tuple(config.image_shape),config.image_width,
                            config.image_training_bf16,prior,
                            2 if config.native_preconditioning=='unit_rms_gaussian_tail_image_v1' else 1)


def field_parameter_count(
    family: str,
    config: TrainingConfig | Mapping[str, Any],
    ambient_dim: int,
) -> int:
    """Return the exact trainable vector-field capacity for one cell."""

    canonical = _canonical_family(family)
    resolved = _coerce_config(config)
    if canonical == "scale_conditioned_normalizing_flow":
        raise ValueError("normalizing flows are not vector fields")
    if resolved.field_preconditioning is not None:
        from models.preconditioned_field import build_bottleneck
        if resolved.field_preconditioning == 'covariance_span_v1' and resolved.field_projection_rank is None:
            raise ValueError('fit the training covariance rank before declaring field capacity')
        architecture = (_vp_architecture_from_training_config(resolved, ambient_dim=ambient_dim)
                        if canonical == 'vp_diffusion' else
                        _bottleneck_architecture_from_training_config(resolved, family=canonical, ambient_dim=ambient_dim))
        with torch.device('meta'):
            model = build_bottleneck(architecture, canonical, resolved)
        return sum(parameter.numel() for parameter in model.parameters())
    if resolved.native_preconditioning in IMAGE_PRECONDITIONING_MODES:
        _model_contract(canonical,resolved)
        architecture = _image_architecture(resolved,ambient_dim)
        with torch.device("meta"):
            image_model = NativeGaussianImageField(architecture,native_family=canonical)
        return sum(parameter.numel() for parameter in image_model.parameters())
    if canonical == "vp_diffusion":
        architecture = _vp_architecture_from_training_config(
            resolved, ambient_dim=ambient_dim
        )
        return bottleneck_parameter_count(
            architecture.ambient_dim,
            architecture.hidden_sizes,
            architecture.time_dim,
        )
    if resolved.field_hidden_sizes is not None:
        architecture = _bottleneck_architecture_from_training_config(
            resolved, family=canonical, ambient_dim=ambient_dim
        )
        return bottleneck_parameter_count(
            architecture.ambient_dim,
            architecture.hidden_sizes,
            architecture.time_dim,
            condition_transform=architecture.condition_transform,
        )
    if canonical == "independent_affine_flow":
        legacy = _affine_field_architecture_from_training_config(
            resolved, ambient_dim=ambient_dim
        )
    elif resolved.depth is None:
        raise ValueError(f"{canonical} requires explicit neural-field depth")
    else:
        legacy = NeuralFieldConfig(
            ambient_dim=ambient_dim,
            hidden_dim=resolved.hidden_dim,
            depth=resolved.depth,
            condition_dim=resolved.time_embedding_dim,
            fourier_features=resolved.fourier_features,
            max_condition_frequency=resolved.max_condition_frequency,
            dropout=resolved.dropout,
            condition_transform=(
                "log"
                if canonical in {"gaussian_diffusion", "brownian_schrodinger_bridge"}
                else "linear"
            ),
        )
    with torch.device("meta"):
        model = ScaleConditionedNeuralField(legacy)
    return sum(parameter.numel() for parameter in model.parameters())


def _affine_field_architecture_from_training_config(
    config: TrainingConfig, *, ambient_dim: int
) -> NeuralFieldConfig:
    """Recompute the exact affine neural-field architecture from Hydra."""

    _affine_spec_from_training_config(config)
    if config.depth is None:
        raise ValueError("independent_affine_flow requires explicit neural-field depth")
    return NeuralFieldConfig(
        ambient_dim=ambient_dim,
        hidden_dim=config.hidden_dim,
        depth=config.depth,
        condition_dim=config.time_embedding_dim,
        fourier_features=config.fourier_features,
        max_condition_frequency=config.max_condition_frequency,
        dropout=config.dropout,
        condition_transform="linear",
    )


def _nf_architecture_from_training_config(
    config: TrainingConfig, *, ambient_dim: int
) -> ConditionalFlowConfig:
    """Require the complete Hydra-declared conditional-flow architecture."""

    fields = {
        "num_coupling_layers": config.num_coupling_layers,
        "conditioner_depth": config.conditioner_depth,
        "log_scale_limit": config.log_scale_limit,
        "epsilon_min": config.epsilon_min,
        "epsilon_max": config.epsilon_max,
    }
    missing = [name for name, value in fields.items() if value is None]
    if missing:
        raise ValueError(
            "scale_conditioned_normalizing_flow requires explicit Hydra settings: "
            f"{sorted(missing)}"
        )
    epsilon_min = float(fields["epsilon_min"])
    epsilon_max = float(fields["epsilon_max"])
    for name in ("num_coupling_layers", "conditioner_depth"):
        value = fields[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(fields["log_scale_limit"], bool) or not isinstance(
        fields["log_scale_limit"], (int, float)
    ):
        raise TypeError("log_scale_limit must be numeric")
    for name in ("epsilon_min", "epsilon_max"):
        if isinstance(fields[name], bool) or not isinstance(fields[name], (int, float)):
            raise TypeError(f"{name} must be numeric")
    if not (
        math.isfinite(epsilon_min)
        and math.isfinite(epsilon_max)
        and 0.0 < epsilon_min <= epsilon_max
    ):
        raise ValueError("epsilon bounds must satisfy 0 < epsilon_min <= epsilon_max")
    if config.dropout != 0.0:
        raise ValueError(
            "scale-conditioned exact likelihood requires dropout to be exactly 0"
        )
    return ConditionalFlowConfig(
        ambient_dim=ambient_dim,
        hidden_dim=config.hidden_dim,
        num_coupling_layers=fields["num_coupling_layers"],
        conditioner_depth=fields["conditioner_depth"],
        condition_dim=config.time_embedding_dim,
        fourier_features=config.fourier_features,
        max_condition_frequency=config.max_condition_frequency,
        dropout=config.dropout,
        log_scale_limit=float(fields["log_scale_limit"]),
    )


def _model_contract(family: Family, config: TrainingConfig) -> dict[str, Any]:
    contract = _native_model_contract(family, config)
    if config.field_preconditioning is not None:
        contract = {**contract, 'field_preconditioning': config.field_preconditioning}
    if config.field_backbone != 'bottleneck_v1':
        contract = {**contract, 'field_backbone': config.field_backbone}
    if config.field_residual_scaling != 'noise_v1':
        contract = {**contract, 'field_residual_scaling': config.field_residual_scaling}
    if config.field_residual_width != 512:
        contract = {**contract, 'field_residual_width': config.field_residual_width}
    if config.noise_pairing != 'iid':
        contract = {**contract, 'noise_pairing': config.noise_pairing}
    if config.optimizer_schedule_policy != 'native_v1':
        contract = {**contract, 'optimizer_schedule_policy': config.optimizer_schedule_policy}
    if config.training_target != 'sample_v1':
        contract = {**contract,'training_target':config.training_target,
                    'training_target_start_step':config.training_target_start_step}
    return contract


def _native_model_contract(family: Family, config: TrainingConfig) -> dict[str, Any]:
    """Return the checkpointed scientific identity for one canonical family."""

    if config.native_preconditioning is not None:
        if family not in {"gaussian_diffusion", "rectified_flow"}:
            raise ValueError("native preconditioning supports VE and rectified flow")
        return {"schema_version": 1, "family": family,
                "native_preconditioning": config.native_preconditioning}
    if family == "vp_diffusion":
        schedule = _vp_schedule_from_training_config(config)
        return {
            "schema_version": 1,
            "family": "vp_diffusion",
            "objective": schedule.contract(),
        }
    if family == "scale_conditioned_normalizing_flow":
        _nf_architecture_from_training_config(config, ambient_dim=2)
        assert config.epsilon_min is not None
        assert config.epsilon_max is not None
        return {
            **dict(NF_DENSITY_CONTRACT),
            "epsilon_min": float(config.epsilon_min),
            "epsilon_max": float(config.epsilon_max),
        }
    if family == "brownian_schrodinger_bridge":
        spec = _bridge_spec_from_training_config(config)
        if config.bridge_tau_min is None or config.bridge_tau_max is None:
            raise ValueError("bridge model contract requires explicit tau bounds")
        return brownian_bridge_contract(
            spec,
            tau_min=float(config.bridge_tau_min),
            tau_max=float(config.bridge_tau_max),
        )
    if family == "independent_affine_flow":
        contract = affine_flow_contract(_affine_spec_from_training_config(config))
        if config.posterior_preconditioning is not None:
            contract = {**contract, "posterior_preconditioning": config.posterior_preconditioning}
        return contract
    return {"schema_version": 1, "family": family}


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    validation_loss: float
    learning_rate: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class StepMetrics:
    step: int
    examples_seen: int
    train_loss: float
    validation_loss: float
    learning_rate: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class FieldPrediction:
    """Primitive field values in model coordinates for one readout scale."""

    field: npt.NDArray[np.float64]
    divergence: npt.NDArray[np.float64]
    evaluation_point: npt.NDArray[np.float64]
    condition: float


@dataclass(frozen=True)
class NFLikelihoodReadoutPrediction:
    """All fixed-point NF likelihood readouts at one declared epsilon."""

    epsilon: float
    finite_difference_log_step: float
    ols_log_step: float
    finite_difference_epsilons: npt.NDArray[np.float64]
    finite_difference_log_likelihood: npt.NDArray[np.float64]
    ols_epsilons: npt.NDArray[np.float64]
    ols_log_likelihood: npt.NDArray[np.float64]
    lid_autograd: npt.NDArray[np.float64]
    lid_symmetric_fd: npt.NDArray[np.float64]
    lid_ols3: npt.NDArray[np.float64]
    lid_ols5: npt.NDArray[np.float64]
    lid_ols9: npt.NDArray[np.float64]

    @property
    def lid_by_readout(self) -> Mapping[str, npt.NDArray[np.float64]]:
        """Return stable public readout identifiers for artifact generation."""

        return {
            "autograd": self.lid_autograd,
            "symmetric_fd": self.lid_symmetric_fd,
            "ols3": self.lid_ols3,
            "ols5": self.lid_ols5,
            "ols9": self.lid_ols9,
        }


@dataclass(frozen=True)
class AffineFieldPrediction:
    """All equivalent Gaussian affine-FM primitives at one noise ratio.

    Divergence is taken with respect to the model input ``y``.  The
    ``channel_score`` instead lives in normalized channel coordinates
    ``r=y/alpha``; the distinction is explicit to prevent an ``alpha`` factor
    from being lost in FM-to-score diagnostics.
    """

    model_output: npt.NDArray[np.float64]
    model_output_divergence: npt.NDArray[np.float64]
    velocity: npt.NDArray[np.float64]
    velocity_divergence: npt.NDArray[np.float64]
    velocity_divergence_from_posterior: npt.NDArray[np.float64]
    posterior_mean: npt.NDArray[np.float64]
    posterior_divergence: npt.NDArray[np.float64]
    marginal_score: npt.NDArray[np.float64]
    marginal_score_divergence: npt.NDArray[np.float64]
    channel_score: npt.NDArray[np.float64]
    channel_score_divergence: npt.NDArray[np.float64]
    evaluation_point: npt.NDArray[np.float64]
    channel_point: npt.NDArray[np.float64]
    variant_id: str
    schedule: str
    parameterization: str
    noise_ratio: float
    native_time: float
    alpha: float
    beta: float
    alpha_derivative: float
    beta_derivative: float
    alpha_log_derivative: float
    log_noise_ratio_derivative: float
    model_condition: float
    divergence_backend: str
    trace_probe_kind: str
    trace_seed: int | None
    trace_probes: int
    shared_posterior_velocity_probes: bool
    primary_trace_field: str
    velocity_divergence_source: str


@dataclass(frozen=True)
class TrainingResult:
    family: Family
    model: TrainableModel
    config: TrainingConfig
    history: tuple[EpochMetrics | StepMetrics, ...]
    best_epoch: int
    best_validation_loss: float
    checkpoint_path: Path
    checkpoint_sha256: str
    normalization_mean: Tensor
    normalization_scale: float
    preprocessing: Mapping[str, str | float | int]
    preprocessing_sha256: str
    weights_metadata: Mapping[str, Any] = field(default_factory=dict)
    final_model_state: Mapping[str, Tensor] | None = None

    @property
    def metrics(self) -> dict[str, float | int]:
        final = self.history[-1]
        if isinstance(final, StepMetrics):
            return {
                "steps_completed": final.step,
                "examples_seen_total": final.examples_seen,
                "best_step": self.best_epoch,
                "examples_seen_at_selected_checkpoint": self.best_epoch
                * self.config.batch_size,
                "best_validation_loss": self.best_validation_loss,
                "final_train_loss": final.train_loss,
                "final_validation_loss": final.validation_loss,
            }
        return {
            "epochs_completed": final.epoch,
            "best_epoch": self.best_epoch,
            "best_validation_loss": self.best_validation_loss,
            "final_train_loss": final.train_loss,
            "final_validation_loss": final.validation_loss,
        }

    @property
    def model_contract(self) -> Mapping[str, Any]:
        """Scientific identity of the trained likelihood/readout interface."""

        return _model_contract(self.family, self.config)


def _canonical_family(family: str) -> Family:
    aliases: dict[str, Family] = {
        "diffusion": "gaussian_diffusion",
        "gaussian_diffusion": "gaussian_diffusion",
        "vp": "vp_diffusion",
        "vp_diffusion": "vp_diffusion",
        "rectified_flow": "rectified_flow",
        "independent_affine_flow": "independent_affine_flow",
        "schrodinger_bridge": "brownian_schrodinger_bridge",
        "brownian_schrodinger_bridge": "brownian_schrodinger_bridge",
        "scale_conditioned_nf": "scale_conditioned_normalizing_flow",
        "scale_conditioned_normalizing_flow": "scale_conditioned_normalizing_flow",
    }
    try:
        return aliases[str(family)]
    except KeyError as exc:
        raise ValueError(
            "family must be diffusion, gaussian_diffusion, vp, vp_diffusion, "
            "rectified_flow, "
            "independent_affine_flow, schrodinger_bridge, "
            "brownian_schrodinger_bridge, "
            "scale_conditioned_nf, or scale_conditioned_normalizing_flow"
        ) from exc


def _coerce_config(config: TrainingConfig | Mapping[str, Any]) -> TrainingConfig:
    if isinstance(config, TrainingConfig):
        return config
    if not isinstance(config, Mapping):
        raise TypeError("config must be TrainingConfig or a mapping")
    return TrainingConfig.from_mapping(config)


def _resolve_device(configured: str) -> torch.device:
    if configured == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(configured)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return device


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch before model initialization."""

    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _flat_finite_data(value: Any, *, name: str) -> Tensor:
    tensor = torch.as_tensor(value)
    if tensor.ndim < 2 or tensor.shape[0] <= 0:
        raise ValueError(f"{name} must have shape (nonempty batch, ...)")
    tensor = (
        tensor.detach()
        .to(device="cpu", dtype=torch.float32)
        .reshape(tensor.shape[0], -1)
    )
    if tensor.shape[1] <= 0:
        raise ValueError(f"{name} has no feature dimensions")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values")
    return tensor.contiguous()


def _mean_sha256(mean: Tensor) -> str:
    array = np.ascontiguousarray(mean.detach().cpu().numpy().astype("<f4", copy=False))
    digest = hashlib.sha256()
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _preprocessing_identity(
    mean: Tensor, scale: float, *, normalized: bool
) -> tuple[dict[str, str | float | int], str]:
    preprocessing: dict[str, str | float | int] = {
        "schema_version": 1,
        "kind": "train_mean_global_rms_v1" if normalized else "identity_v1",
        "ambient_dim": mean.numel(),
        "mean_sha256": _mean_sha256(mean),
        "scalar_scale": float(scale),
    }
    canonical = json.dumps(
        preprocessing,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return preprocessing, hashlib.sha256(canonical).hexdigest()


def _normalization(
    train: Tensor, *, enabled: bool, epsilon: float
) -> tuple[Tensor, float, dict[str, str | float | int], str]:
    if enabled:
        mean = train.mean(dim=0)
        rms = torch.sqrt(torch.mean((train - mean).square()))
        scale = float(rms.item())
        if not math.isfinite(scale) or scale < epsilon:
            scale = 1.0
    else:
        mean = torch.zeros(train.shape[1], dtype=train.dtype)
        scale = 1.0
    preprocessing, preprocessing_sha256 = _preprocessing_identity(
        mean, scale, normalized=enabled
    )
    return mean.contiguous(), scale, preprocessing, preprocessing_sha256


def _sample_log_uniform(
    batch_size: int,
    *,
    minimum: float,
    maximum: float,
    data: Tensor,
    generator: torch.Generator,
) -> Tensor:
    uniform = torch.rand(
        batch_size,
        device=data.device,
        dtype=data.dtype,
        generator=generator,
    )
    return torch.exp(
        math.log(minimum) + uniform * (math.log(maximum) - math.log(minimum))
    )


def diffusion_ve_dsm_loss(
    model: nn.Module,
    clean: Tensor,
    *,
    sigma_min: float,
    sigma_max: float,
    generator: torch.Generator,
) -> Tensor:
    """VE denoising-score-matching loss in a stable ``x0`` parameterization."""

    sigma = _sample_log_uniform(
        clean.shape[0],
        minimum=sigma_min,
        maximum=sigma_max,
        data=clean,
        generator=generator,
    )
    noise = torch.randn(
        clean.shape, device=clean.device, dtype=clean.dtype, generator=generator
    )
    from models.noise_pairing import paired_corruption
    clean,sigma,noise=paired_corruption(model,clean,sigma,noise)
    if isinstance(model, (NativeGaussianBottleneck, NativeGaussianImageField)):
        if model.native_family != "gaussian_diffusion":
            raise ValueError("VE loss requires a VE field")
        return model.residual_loss(clean, sigma, noise)
    sigma_broadcast = sigma.reshape(-1, *([1] * (clean.ndim - 1)))
    perturbed = clean + sigma_broadcast * noise
    denoised = model(perturbed, sigma)
    from models.empirical_target import denoising_target
    clean=denoising_target(model,perturbed,sigma,clean)
    # Equivalent to sigma^2 * ||score + noise/sigma||^2 after substituting
    # score=(denoised-perturbed)/sigma^2.
    return ((denoised - clean) / sigma_broadcast).square().flatten(1).mean()


def rectified_flow_matching_loss(
    model: nn.Module,
    data: Tensor,
    *,
    time_min: float,
    time_max: float,
    generator: torch.Generator,
) -> Tensor:
    """Linear flow-matching objective for ``z -> data`` transport."""

    time = time_min + (time_max - time_min) * torch.rand(
        data.shape[0],
        device=data.device,
        dtype=data.dtype,
        generator=generator,
    )
    noise = torch.randn(
        data.shape, device=data.device, dtype=data.dtype, generator=generator
    )
    from models.noise_pairing import paired_corruption
    data,time,noise=paired_corruption(model,data,time,noise)
    if isinstance(model, (NativeGaussianBottleneck, NativeGaussianImageField)):
        if model.native_family != "rectified_flow":
            raise ValueError("rectified loss requires a rectified field")
        return model.residual_loss(data, time, noise)
    time_broadcast = time.reshape(-1, *([1] * (data.ndim - 1)))
    interpolated = time_broadcast * data + (1.0 - time_broadcast) * noise
    target_velocity = data - noise
    from models.empirical_target import empirical_target_enabled,denoising_target
    if empirical_target_enabled(model):
        posterior=denoising_target(model,interpolated/time_broadcast,(1-time)/time,data)
        target_velocity=(posterior-interpolated)/(1-time_broadcast)
    velocity = model(interpolated, time)
    return (velocity - target_velocity).square().flatten(1).mean()


def independent_affine_flow_matching_loss(
    model: nn.Module,
    data: Tensor,
    *,
    spec: AffineFlowSpec,
    generator: torch.Generator,
) -> Tensor:
    """Independent Gaussian affine-FM objective under an explicit contract.

    The sampled physical scale is always ``lambda=beta/alpha``.  The schedule
    controls the interpolant and (for ``direct_velocity``) the derivative
    target; the parameterization controls whether the network regresses that
    target or the stable posterior mean ``E[X|Y]``.
    """

    noise_ratio = sample_noise_ratio(
        data.shape[0], spec=spec, data=data, generator=generator
    )
    state = affine_schedule_state(noise_ratio, spec.schedule)
    noise = torch.randn(
        data.shape, device=data.device, dtype=data.dtype, generator=generator
    )
    from models.noise_pairing import paired_corruption
    original_noise=noise
    data,noise_ratio,noise=paired_corruption(model,data,noise_ratio,noise)
    if noise is not original_noise:
        # Antithetic pairing repeats the first half. Reuse its already checked
        # schedule tensors exactly, rather than synchronizing the GPU again.
        half=len(noise_ratio)//2
        state=replace(state,**{name:torch.cat((getattr(state,name)[:half],)*2)
                              for name in state.__dataclass_fields__})
    if isinstance(model, PreconditionedLogNoiseMLP):
        if not (
            spec.schedule == "log_noise"
            and spec.parameterization == "posterior_mean"
            and spec.conditioning == "log_noise_ratio"
            and spec.loss_weighting == "posterior_bias_equivalent"
        ):
            raise ValueError("preconditioned MLP received an incompatible affine spec")
        return model.residual_loss(data, noise_ratio, noise)
    interpolated, target = affine_interpolant_and_target(
        data,
        noise,
        state,
        parameterization=spec.parameterization,
    )
    condition = schedule_condition(state, spec.conditioning)
    from models.empirical_target import empirical_target_enabled,denoising_target
    if empirical_target_enabled(model):
        posterior=denoising_target(model,interpolated/state.alpha[:,None],noise_ratio,data)
        target=posterior if spec.parameterization=='posterior_mean' else posterior_to_velocity(posterior,interpolated,state)
    prediction = model(interpolated, condition)
    per_example = (prediction - target).square().flatten(1).mean(dim=1)
    weights = flow_matching_loss_weights(state, spec)
    return torch.mean(weights * per_example)


def _objective(
    family: Family,
    model: nn.Module,
    batch: Tensor,
    config: TrainingConfig,
    generator: torch.Generator,
) -> Tensor:
    if config.native_noise_pairing == "antithetic_v1" and model.training:
        # Same marginal native objective and the same number of network
        # examples: half as many independent clean points, each with Z and -Z.
        # Validation keeps its original independent samples and RNG sequence.
        if family not in {"gaussian_diffusion", "rectified_flow"}:
            raise ValueError("antithetic native noise supports VE and rectified flow")
        if len(batch) % 2:
            raise ValueError("antithetic training requires an even batch")
        clean = batch[:len(batch) // 2]
        if family == "gaussian_diffusion":
            condition = _sample_log_uniform(len(clean), minimum=config.sigma_min,
                maximum=config.sigma_max, data=clean, generator=generator)
        else:
            condition = config.time_min + (config.time_max-config.time_min)*torch.rand(
                len(clean), device=clean.device, dtype=clean.dtype, generator=generator)
        noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
        return model.residual_loss(torch.cat((clean,clean)),torch.cat((condition,condition)),
                                   torch.cat((noise,-noise)))
    if family == "vp_diffusion":
        schedule = _vp_schedule_from_training_config(config)
        time = torch.rand(
            batch.shape[0],
            device=batch.device,
            dtype=batch.dtype,
            generator=generator,
        )
        noise = torch.randn(
            batch.shape,
            device=batch.device,
            dtype=batch.dtype,
            generator=generator,
        )
        return vp_loss(model, batch, time, noise, schedule)
    if family == "gaussian_diffusion":
        if config.sigma_min is None or config.sigma_max is None:
            raise ValueError(
                "gaussian_diffusion requires explicit sigma_min and sigma_max"
            )
        return diffusion_ve_dsm_loss(
            model,
            batch,
            sigma_min=config.sigma_min,
            sigma_max=config.sigma_max,
            generator=generator,
        )
    if family == "rectified_flow":
        if config.time_min is None or config.time_max is None:
            raise ValueError("rectified_flow requires explicit time_min and time_max")
        return rectified_flow_matching_loss(
            model,
            batch,
            time_min=config.time_min,
            time_max=config.time_max,
            generator=generator,
        )
    if family == "independent_affine_flow":
        return independent_affine_flow_matching_loss(
            model,
            batch,
            spec=_affine_spec_from_training_config(config),
            generator=generator,
        )
    if family == "scale_conditioned_normalizing_flow":
        if not isinstance(model, ScaleConditionedRealNVP):
            raise TypeError("scale-conditioned NF objective requires RealNVP")
        _nf_architecture_from_training_config(
            config, ambient_dim=model.config.ambient_dim
        )
        assert config.epsilon_min is not None
        assert config.epsilon_max is not None
        return conditional_smoothed_nll(
            model,
            batch,
            epsilon_min=float(config.epsilon_min),
            epsilon_max=float(config.epsilon_max),
            generator=generator,
        )
    if family == "brownian_schrodinger_bridge":
        spec = _bridge_spec_from_training_config(config)
        if config.bridge_tau_min is None or config.bridge_tau_max is None:
            raise ValueError(
                "brownian_schrodinger_bridge requires explicit bridge_tau bounds"
            )
        return brownian_sb_terminal_denoising_loss(
            model,
            batch,
            tau_min=float(config.bridge_tau_min),
            tau_max=float(config.bridge_tau_max),
            spec=spec,
            generator=generator,
        )
    raise AssertionError(f"unhandled canonical family: {family}")


def _validation_loss(
    family: Family,
    model: nn.Module,
    validation: Tensor,
    config: TrainingConfig,
    *,
    seed: int,
) -> float:
    model.eval()
    generator = torch.Generator(device=validation.device)
    generator.manual_seed(seed)
    weighted_loss = 0.0
    with torch.no_grad():
        for start in range(0, validation.shape[0], config.batch_size):
            batch = validation[start : start + config.batch_size]
            loss = _objective(family, model, batch, config, generator)
            weighted_loss += float(loss.item()) * batch.shape[0]
    return weighted_loss / validation.shape[0]


def evaluate_native_loss(
    result: TrainingResult,
    data: Any,
    *,
    seed: int,
) -> float:
    """Evaluate the checkpoint's target-free native loss on declared data."""

    if not isinstance(result, TrainingResult):
        raise TypeError("evaluate_native_loss requires a TrainingResult")
    data_cpu = _flat_finite_data(data, name="validation")
    if data_cpu.shape[1] != result.model.config.ambient_dim:
        raise ValueError("validation ambient dimension does not match model")
    normalized = (
        data_cpu - result.normalization_mean.reshape(1, -1)
    ) / result.normalization_scale
    parameter = next(result.model.parameters())
    validation = normalized.to(device=parameter.device, dtype=parameter.dtype)
    return _validation_loss(
        result.family,
        result.model,
        validation,
        result.config,
        seed=seed,
    )


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cpu_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone().contiguous()
        for name, value in model.state_dict().items()
    }


def _state_dict_sha256(state: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = torch.as_tensor(state[name]).detach().cpu().contiguous()
        header = json.dumps(
            {
                "name": name,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class _FixedStepScheduler:
    """Minimal versioned step scheduler with a strict resumable state."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        base_learning_rate: float,
        total_steps: int,
        warmup_steps: int,
        schedule: Literal["constant_v1", "warmup_cosine_v1", "terminal_cosine_v1"],
        terminal_decay_steps: int = 0,
        terminal_learning_rate_ratio: float = 0.01,
    ) -> None:
        self.optimizer = optimizer
        self.base_learning_rate = base_learning_rate
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.schedule = schedule
        self.terminal_decay_steps = terminal_decay_steps
        self.terminal_learning_rate_ratio = terminal_learning_rate_ratio
        self.last_completed_step = 0

    def _multiplier(self, step: int) -> float:
        if self.schedule == 'terminal_cosine_v1':
            start = self.total_steps - self.terminal_decay_steps
            progress = max(0.0, (step - start) / self.terminal_decay_steps)
            return self.terminal_learning_rate_ratio + (1 - self.terminal_learning_rate_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        if self.schedule == "constant_v1":
            return 1.0
        if self.warmup_steps and step <= self.warmup_steps:
            return step / self.warmup_steps
        if self.warmup_steps:
            progress = (step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
        elif self.total_steps == 1:
            progress = 0.0
        else:
            progress = (step - 1) / (self.total_steps - 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def prepare_step(self, step: int) -> float:
        if step != self.last_completed_step + 1:
            raise ValueError("scheduler step is inconsistent with resume state")
        learning_rate = self.base_learning_rate * self._multiplier(step)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        return learning_rate

    def complete_step(self, step: int) -> None:
        if step != self.last_completed_step + 1:
            raise ValueError("scheduler completion is inconsistent with resume state")
        self.last_completed_step = step

    def state_dict(self) -> dict[str, Any]:
        state = {
            "schema_version": 1,
            "schedule": self.schedule,
            "base_learning_rate": self.base_learning_rate,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "last_completed_step": self.last_completed_step,
        }
        if self.schedule == 'terminal_cosine_v1':
            state.update(terminal_decay_steps=self.terminal_decay_steps,
                         terminal_learning_rate_ratio=self.terminal_learning_rate_ratio)
        return state

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        expected = {
            "schema_version": 1,
            "schedule": self.schedule,
            "base_learning_rate": self.base_learning_rate,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
        }
        if self.schedule == 'terminal_cosine_v1':
            expected.update(terminal_decay_steps=self.terminal_decay_steps,
                            terminal_learning_rate_ratio=self.terminal_learning_rate_ratio)
        if (
            not isinstance(value, Mapping)
            or {key: value.get(key) for key in expected} != expected
        ):
            raise ValueError("fixed-step scheduler state mismatch")
        last_step = value.get("last_completed_step")
        if (
            isinstance(last_step, bool)
            or not isinstance(last_step, int)
            or not 0 <= last_step <= self.total_steps
        ):
            raise ValueError("fixed-step scheduler step is invalid")
        self.last_completed_step = last_step
        if last_step:
            learning_rate = self.base_learning_rate * self._multiplier(last_step)
            for group in self.optimizer.param_groups:
                group["lr"] = learning_rate


def _tensor_identity_sha256(value: Tensor) -> str:
    """Hash the exact finite float32 tensor consumed by the trainer."""

    array = np.ascontiguousarray(value.detach().cpu().numpy().astype("<f4", copy=False))
    digest = hashlib.sha256()
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _save_checkpoint(
    path: Path,
    *,
    family: Family,
    model: TrainableModel,
    config: TrainingConfig,
    history: tuple[EpochMetrics, ...],
    best_epoch: int,
    best_validation_loss: float,
    normalization_mean: Tensor,
    normalization_scale: float,
    preprocessing: Mapping[str, str | float | int],
    preprocessing_sha256: str,
) -> str:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "family": family,
        "model_contract": _model_contract(family, config),
        "architecture": model.config.to_dict(),
        "training_config": config.to_dict(),
        "model_state": _cpu_state_dict(model),
        "history": [metric.to_dict() for metric in history],
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "normalization": {
            "mean": normalization_mean.detach().cpu().contiguous(),
            "scale": normalization_scale,
            "preprocessing": dict(preprocessing),
            "sha256": preprocessing_sha256,
        },
    }
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as stream:
            temporary_path = Path(stream.name)
        torch.save(payload, temporary_path)
        with temporary_path.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return _checkpoint_sha256(path)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably replace one trusted local torch payload.

    Global campaigns can run for several days.  A scheduler interruption must
    therefore leave either the previous complete epoch snapshot or the new
    one, never a partially written file.
    """

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as stream:
            temporary_path = Path(stream.name)
        torch.save(dict(payload), temporary_path)
        with temporary_path.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _fixed_weights_metadata(
    *,
    best_state: Mapping[str, Tensor],
    final_state: Mapping[str, Tensor],
    best_step: int,
    final_step: int,
    batch_size: int,
    initial_validation_loss: float,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "selection": "minimum_train_selection_native_loss_v1",
        "initial": {
            "step": 0,
            "examples_seen": 0,
            "validation_loss": initial_validation_loss,
        },
        "selected": {
            "kind": "validation_best",
            "step": best_step,
            "examples_seen": best_step * batch_size,
            "state_sha256": _state_dict_sha256(best_state),
        },
        "final": {
            "kind": "final",
            "step": final_step,
            "examples_seen": final_step * batch_size,
            "state_sha256": _state_dict_sha256(final_state),
        },
    }


def _save_fixed_step_checkpoint(
    path: Path,
    *,
    family: Family,
    model: TrainableModel,
    config: TrainingConfig,
    history: tuple[StepMetrics, ...],
    best_step: int,
    best_validation_loss: float,
    best_state: Mapping[str, Tensor],
    final_state: Mapping[str, Tensor],
    initial_validation_loss: float,
    normalization_mean: Tensor,
    normalization_scale: float,
    preprocessing: Mapping[str, str | float | int],
    preprocessing_sha256: str,
) -> tuple[str, dict[str, Any]]:
    assert config.steps is not None
    weights_metadata = _fixed_weights_metadata(
        best_state=best_state,
        final_state=final_state,
        best_step=best_step,
        final_step=config.steps,
        batch_size=config.batch_size,
        initial_validation_loss=initial_validation_loss,
    )
    payload: dict[str, Any] = {
        "schema_version": FIXED_STEP_CHECKPOINT_SCHEMA_VERSION,
        "family": family,
        "model_contract": _model_contract(family, config),
        "architecture": model.config.to_dict(),
        "training_config": config.to_dict(),
        "model_state": {
            name: value.detach().cpu().clone().contiguous()
            for name, value in best_state.items()
        },
        "final_model_state": {
            name: value.detach().cpu().clone().contiguous()
            for name, value in final_state.items()
        },
        "history": [metric.to_dict() for metric in history],
        "best_step": best_step,
        "best_validation_loss": best_validation_loss,
        "global_step": config.steps,
        "examples_seen_total": config.steps * config.batch_size,
        "examples_seen_at_selected_checkpoint": best_step * config.batch_size,
        "weights_metadata": weights_metadata,
        "normalization": {
            "mean": normalization_mean.detach().cpu().contiguous(),
            "scale": normalization_scale,
            "preprocessing": dict(preprocessing),
            "sha256": preprocessing_sha256,
        },
    }
    _atomic_torch_save(path, payload)
    return _checkpoint_sha256(path), weights_metadata


def _training_progress_payload(
    *,
    family: Family,
    model: TrainableModel,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    epoch: int,
    history: tuple[EpochMetrics, ...],
    best_epoch: int,
    best_validation_loss: float,
    best_state: Mapping[str, Tensor],
    stale_validations: int,
    normalization_mean: Tensor,
    normalization_scale: float,
    preprocessing: Mapping[str, str | float | int],
    preprocessing_sha256: str,
    data_identity: Mapping[str, Any],
    shuffle_generator: torch.Generator,
    objective_generator: torch.Generator,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "schema_version": TRAINING_PROGRESS_SCHEMA_VERSION,
        "family": family,
        "model_contract": _model_contract(family, config),
        "architecture": model.config.to_dict(),
        "training_config": config.to_dict(),
        "epoch": epoch,
        "model_state": _cpu_state_dict(model),
        "optimizer_state": optimizer.state_dict(),
        "history": [metric.to_dict() for metric in history],
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "best_state": {
            name: value.detach().cpu().clone().contiguous()
            for name, value in best_state.items()
        },
        "stale_validations": stale_validations,
        "normalization": {
            "mean": normalization_mean.detach().cpu().contiguous(),
            "scale": normalization_scale,
            "preprocessing": dict(preprocessing),
            "sha256": preprocessing_sha256,
        },
        "data_identity": dict(data_identity),
        "rng": {
            "shuffle_generator": shuffle_generator.get_state().cpu(),
            "objective_generator": objective_generator.get_state().cpu(),
            "torch": torch.get_rng_state().cpu(),
            "cuda": [state.cpu() for state in torch.cuda.get_rng_state_all()]
            if device.type == "cuda"
            else [],
            "device_type": device.type,
        },
    }


def _load_training_progress(
    path: Path,
    *,
    family: Family,
    model: TrainableModel,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    normalization_mean: Tensor,
    normalization_scale: float,
    preprocessing: Mapping[str, str | float | int],
    preprocessing_sha256: str,
    data_identity: Mapping[str, Any],
    shuffle_generator: torch.Generator,
    objective_generator: torch.Generator,
    device: torch.device,
) -> tuple[
    int,
    list[EpochMetrics],
    int,
    float,
    dict[str, Tensor],
    int,
]:
    """Strictly restore the last validated epoch of an interrupted cell."""

    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "schema_version",
        "family",
        "model_contract",
        "architecture",
        "training_config",
        "epoch",
        "model_state",
        "optimizer_state",
        "history",
        "best_epoch",
        "best_validation_loss",
        "best_state",
        "stale_validations",
        "normalization",
        "data_identity",
        "rng",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("training progress schema mismatch")
    if payload["schema_version"] != TRAINING_PROGRESS_SCHEMA_VERSION:
        raise ValueError("unsupported training progress schema_version")
    if _canonical_family(payload["family"]) != family:
        raise ValueError("training progress family mismatch")
    try:
        stored_config = TrainingConfig.from_mapping(payload["training_config"])
    except (TypeError, ValueError) as exc:
        raise ValueError("training progress config mismatch") from exc
    if stored_config != config:
        raise ValueError("training progress config mismatch")
    if payload["model_contract"] != _model_contract(family, config):
        raise ValueError("training progress model contract mismatch")
    if payload["architecture"] != model.config.to_dict():
        raise ValueError("training progress architecture mismatch")
    if payload["data_identity"] != dict(data_identity):
        raise ValueError("training progress data identity mismatch")

    normalization = payload["normalization"]
    if not isinstance(normalization, dict) or set(normalization) != {
        "mean",
        "scale",
        "preprocessing",
        "sha256",
    }:
        raise ValueError("training progress normalization schema mismatch")
    stored_mean = torch.as_tensor(normalization["mean"]).detach().cpu().float()
    if not torch.equal(stored_mean, normalization_mean.detach().cpu().float()):
        raise ValueError("training progress normalization mean mismatch")
    if float(normalization["scale"]) != normalization_scale:
        raise ValueError("training progress normalization scale mismatch")
    if (
        normalization["preprocessing"] != dict(preprocessing)
        or normalization["sha256"] != preprocessing_sha256
    ):
        raise ValueError("training progress preprocessing identity mismatch")

    epoch = payload["epoch"]
    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or not 1 <= epoch <= config.epochs
    ):
        raise ValueError("training progress epoch is invalid")
    history_raw = payload["history"]
    if not isinstance(history_raw, list) or not history_raw:
        raise ValueError("training progress history is invalid")
    history: list[EpochMetrics] = []
    history_fields = {"epoch", "train_loss", "validation_loss", "learning_rate"}
    for item in history_raw:
        if not isinstance(item, dict) or set(item) != history_fields:
            raise ValueError("training progress history entry schema mismatch")
        metric = EpochMetrics(**item)
        if not all(
            math.isfinite(float(value))
            for value in (
                metric.train_loss,
                metric.validation_loss,
                metric.learning_rate,
            )
        ):
            raise ValueError("training progress history contains non-finite values")
        history.append(metric)
    if history[-1].epoch != epoch or any(
        left.epoch >= right.epoch for left, right in pairwise(history)
    ):
        raise ValueError("training progress history epochs are inconsistent")

    best_epoch = payload["best_epoch"]
    best_validation_loss = float(payload["best_validation_loss"])
    stale_validations = payload["stale_validations"]
    if (
        isinstance(best_epoch, bool)
        or not isinstance(best_epoch, int)
        or best_epoch not in {metric.epoch for metric in history}
        or not math.isfinite(best_validation_loss)
        or isinstance(stale_validations, bool)
        or not isinstance(stale_validations, int)
        or stale_validations < 0
    ):
        raise ValueError("training progress best-state metadata is invalid")
    best_state = payload["best_state"]
    if not isinstance(best_state, dict) or not best_state:
        raise ValueError("training progress best_state is invalid")

    rng = payload["rng"]
    if not isinstance(rng, dict) or set(rng) != {
        "shuffle_generator",
        "objective_generator",
        "torch",
        "cuda",
        "device_type",
    }:
        raise ValueError("training progress RNG schema mismatch")
    if rng["device_type"] != device.type:
        raise ValueError("training progress device type mismatch")
    cuda_states = rng["cuda"]
    if not isinstance(cuda_states, list):
        raise TypeError("training progress CUDA RNG states are invalid")
    if device.type == "cuda" and len(cuda_states) != torch.cuda.device_count():
        raise ValueError("training progress CUDA device count mismatch")
    if device.type != "cuda" and cuda_states:
        raise ValueError("CPU training progress cannot carry CUDA RNG states")

    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    shuffle_generator.set_state(torch.as_tensor(rng["shuffle_generator"]).cpu())
    objective_generator.set_state(torch.as_tensor(rng["objective_generator"]).cpu())
    torch.set_rng_state(torch.as_tensor(rng["torch"]).cpu())
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(
            [torch.as_tensor(state).cpu() for state in cuda_states]
        )
    return (
        epoch + 1,
        history,
        int(best_epoch),
        best_validation_loss,
        {
            str(name): torch.as_tensor(value).detach().cpu().clone().contiguous()
            for name, value in best_state.items()
        },
        int(stale_validations),
    )


def _fixed_training_progress_payload(
    *,
    family: Family,
    model: TrainableModel,
    optimizer: torch.optim.Optimizer,
    scheduler: _FixedStepScheduler,
    config: TrainingConfig,
    global_step: int,
    history: tuple[StepMetrics, ...],
    best_step: int,
    best_validation_loss: float,
    initial_validation_loss: float,
    best_state: Mapping[str, Tensor],
    normalization_mean: Tensor,
    normalization_scale: float,
    preprocessing: Mapping[str, str | float | int],
    preprocessing_sha256: str,
    data_identity: Mapping[str, Any],
    sampler_generator: torch.Generator,
    objective_generator: torch.Generator,
    device: torch.device,
    ema_model: TrainableModel | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": FIXED_STEP_PROGRESS_SCHEMA_VERSION,
        "family": family,
        "model_contract": _model_contract(family, config),
        "architecture": model.config.to_dict(),
        "training_config": config.to_dict(),
        "global_step": global_step,
        "model_state": _cpu_state_dict(model),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "history": [metric.to_dict() for metric in history],
        "best_step": best_step,
        "best_validation_loss": best_validation_loss,
        "initial_validation_loss": initial_validation_loss,
        "best_state": {
            name: value.detach().cpu().clone().contiguous()
            for name, value in best_state.items()
        },
        "normalization": {
            "mean": normalization_mean.detach().cpu().contiguous(),
            "scale": normalization_scale,
            "preprocessing": dict(preprocessing),
            "sha256": preprocessing_sha256,
        },
        "data_identity": dict(data_identity),
        "sampler": {
            "kind": "uniform_with_replacement_v1",
            "generator_state": sampler_generator.get_state().cpu(),
        },
        "counters": {
            "examples_seen_total": global_step * config.batch_size,
        },
        "rng": {
            "objective_generator": objective_generator.get_state().cpu(),
            "torch": torch.get_rng_state().cpu(),
            "cuda": [state.cpu() for state in torch.cuda.get_rng_state_all()]
            if device.type == "cuda"
            else [],
            "device_type": device.type,
        },
    }
    if config.ema_decay is not None:
        payload['ema_state'] = None if ema_model is None else _cpu_state_dict(ema_model)
    return payload


def _load_fixed_training_progress(
    path: Path,
    *,
    family: Family,
    model: TrainableModel,
    optimizer: torch.optim.Optimizer,
    scheduler: _FixedStepScheduler,
    config: TrainingConfig,
    normalization_mean: Tensor,
    normalization_scale: float,
    preprocessing: Mapping[str, str | float | int],
    preprocessing_sha256: str,
    data_identity: Mapping[str, Any],
    sampler_generator: torch.Generator,
    objective_generator: torch.Generator,
    device: torch.device,
    expected_initial_validation_loss: float,
) -> tuple[int, list[StepMetrics], int, float, float, dict[str, Tensor]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "schema_version",
        "family",
        "model_contract",
        "architecture",
        "training_config",
        "global_step",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "history",
        "best_step",
        "best_validation_loss",
        "initial_validation_loss",
        "best_state",
        "normalization",
        "data_identity",
        "sampler",
        "counters",
        "rng",
    }
    if config.ema_decay is not None:
        required.add('ema_state')
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("fixed-step training progress schema mismatch")
    if payload["schema_version"] != FIXED_STEP_PROGRESS_SCHEMA_VERSION:
        raise ValueError("unsupported fixed-step progress schema_version")
    if _canonical_family(payload["family"]) != family:
        raise ValueError("fixed-step training progress family mismatch")
    try:
        stored_config = TrainingConfig.from_mapping(payload["training_config"])
    except (TypeError, ValueError) as exc:
        raise ValueError("fixed-step training progress config mismatch") from exc
    if stored_config != config:
        raise ValueError("fixed-step training progress config mismatch")
    if payload["model_contract"] != _model_contract(family, config):
        raise ValueError("fixed-step training progress model contract mismatch")
    stored_architecture = payload["architecture"]
    if isinstance(model, NativeGaussianImageField):
        stored_architecture = ImageFieldConfig.from_mapping(stored_architecture).to_dict()
    if stored_architecture != model.config.to_dict():
        raise ValueError("fixed-step training progress architecture mismatch")
    if payload["data_identity"] != dict(data_identity):
        raise ValueError("fixed-step training progress data identity mismatch")

    normalization = payload["normalization"]
    if not isinstance(normalization, dict) or set(normalization) != {
        "mean",
        "scale",
        "preprocessing",
        "sha256",
    }:
        raise ValueError("fixed-step progress normalization schema mismatch")
    stored_mean = torch.as_tensor(normalization["mean"]).detach().cpu().float()
    if not torch.equal(stored_mean, normalization_mean.detach().cpu().float()):
        raise ValueError("fixed-step progress normalization mean mismatch")
    if float(normalization["scale"]) != normalization_scale:
        raise ValueError("fixed-step progress normalization scale mismatch")
    if (
        normalization["preprocessing"] != dict(preprocessing)
        or normalization["sha256"] != preprocessing_sha256
    ):
        raise ValueError("fixed-step progress preprocessing identity mismatch")

    assert config.steps is not None
    global_step = payload["global_step"]
    if (
        isinstance(global_step, bool)
        or not isinstance(global_step, int)
        or not 1 <= global_step <= config.steps
    ):
        raise ValueError("fixed-step progress global_step is invalid")
    history_raw = payload["history"]
    if not isinstance(history_raw, list) or not history_raw:
        raise ValueError("fixed-step progress history is invalid")
    history: list[StepMetrics] = []
    history_fields = {
        "step",
        "examples_seen",
        "train_loss",
        "validation_loss",
        "learning_rate",
    }
    for item in history_raw:
        if not isinstance(item, dict) or set(item) != history_fields:
            raise ValueError("fixed-step progress history entry schema mismatch")
        metric = StepMetrics(**item)
        if metric.examples_seen != metric.step * config.batch_size or not all(
            math.isfinite(float(value))
            for value in (
                metric.train_loss,
                metric.validation_loss,
                metric.learning_rate,
            )
        ):
            raise ValueError("fixed-step progress history entry is invalid")
        history.append(metric)
    if history[-1].step != global_step or any(
        left.step >= right.step for left, right in pairwise(history)
    ):
        raise ValueError("fixed-step progress history steps are inconsistent")

    best_step = payload["best_step"]
    best_validation_loss = float(payload["best_validation_loss"])
    initial_validation_loss = float(payload["initial_validation_loss"])
    if (
        isinstance(best_step, bool)
        or not isinstance(best_step, int)
        or best_step not in {metric.step for metric in history}
        or not math.isfinite(best_validation_loss)
        or not math.isfinite(initial_validation_loss)
    ):
        raise ValueError("fixed-step progress best-state metadata is invalid")
    if initial_validation_loss != expected_initial_validation_loss:
        raise ValueError("fixed-step progress initial validation loss mismatch")
    expected_best = min(
        history, key=lambda metric: (metric.validation_loss, metric.step)
    )
    if (
        best_step != expected_best.step
        or best_validation_loss != expected_best.validation_loss
    ):
        raise ValueError("fixed-step progress best selection is invalid")
    best_state = payload["best_state"]
    if not isinstance(best_state, dict) or not best_state:
        raise ValueError("fixed-step progress best_state is invalid")
    sampler = payload["sampler"]
    if not isinstance(sampler, dict) or set(sampler) != {
        "kind",
        "generator_state",
    }:
        raise ValueError("fixed-step progress sampler schema mismatch")
    if sampler["kind"] != "uniform_with_replacement_v1":
        raise ValueError("fixed-step progress sampler kind mismatch")
    counters = payload["counters"]
    if counters != {"examples_seen_total": global_step * config.batch_size}:
        raise ValueError("fixed-step progress counters mismatch")
    rng = payload["rng"]
    if not isinstance(rng, dict) or set(rng) != {
        "objective_generator",
        "torch",
        "cuda",
        "device_type",
    }:
        raise ValueError("fixed-step progress RNG schema mismatch")
    if rng["device_type"] != device.type:
        raise ValueError("fixed-step progress device type mismatch")
    cuda_states = rng["cuda"]
    if not isinstance(cuda_states, list):
        raise TypeError("fixed-step progress CUDA RNG states are invalid")
    if device.type == "cuda" and len(cuda_states) != torch.cuda.device_count():
        raise ValueError("fixed-step progress CUDA device count mismatch")
    if device.type != "cuda" and cuda_states:
        raise ValueError("CPU fixed-step progress cannot carry CUDA RNG states")

    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    if scheduler.last_completed_step != global_step:
        raise ValueError("fixed-step progress scheduler/global_step mismatch")
    sampler_generator.set_state(
        torch.as_tensor(sampler["generator_state"]).detach().cpu()
    )
    objective_generator.set_state(
        torch.as_tensor(rng["objective_generator"]).detach().cpu()
    )
    torch.set_rng_state(torch.as_tensor(rng["torch"]).detach().cpu())
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(
            [torch.as_tensor(state).detach().cpu() for state in cuda_states]
        )
    return (
        global_step + 1,
        history,
        best_step,
        best_validation_loss,
        initial_validation_loss,
        {
            str(name): torch.as_tensor(value).detach().cpu().clone().contiguous()
            for name, value in best_state.items()
        },
    )


def train_model(
    family: str,
    train: Any,
    validation: Any,
    config: TrainingConfig | Mapping[str, Any],
    checkpoint_path: str | Path,
    log_callback: LogCallback | None = None,
    *,
    progress_checkpoint_path: str | Path | None = None,
) -> TrainingResult:
    """Train one family and atomically write its best checkpoint.

    When ``progress_checkpoint_path`` is supplied, validated epochs are
    snapshotted at the declared progress interval, at the final epoch, and
    when early stopping fires.  Each snapshot includes optimizer and generator
    state.  A subsequent invocation resumes from that exact epoch and removes
    the progress file only after the final portable best-model checkpoint is
    safely installed.
    """

    canonical_family = _canonical_family(family)
    resolved_config = _coerce_config(config)
    if canonical_family == "independent_affine_flow":
        _affine_spec_from_training_config(resolved_config)
    elif _has_affine_settings(resolved_config):
        raise ValueError(
            f"{canonical_family} cannot carry inactive independent affine-flow settings"
        )
    if canonical_family == "vp_diffusion":
        _vp_architecture_from_training_config(resolved_config, ambient_dim=2)
        _vp_schedule_from_training_config(resolved_config)
        if resolved_config.training_mode != "fixed_steps_v1":
            raise ValueError("vp_diffusion requires training_mode=fixed_steps_v1")
        if resolved_config.gradient_clip_norm is not None and resolved_config.optimizer_schedule_policy == 'native_v1':
            raise ValueError("vp_diffusion requires gradient_clip_norm=null")
        if resolved_config.field_hidden_sizes is not None:
            raise ValueError(
                "vp_diffusion uses vp_hidden_sizes, not field_hidden_sizes"
            )
    elif _has_vp_settings(resolved_config):
        raise ValueError(f"{canonical_family} cannot carry inactive VP settings")
    if (
        canonical_family == "scale_conditioned_normalizing_flow"
        and resolved_config.field_hidden_sizes is not None
    ):
        raise ValueError("scale-conditioned NF cannot carry field_hidden_sizes")
    if (
        resolved_config.training_mode == "fixed_steps_v1"
        and canonical_family != "vp_diffusion"
        and resolved_config.warmup_steps != 0
    ):
        raise ValueError("non-VP fixed-step training requires warmup_steps=0")
    if canonical_family == "brownian_schrodinger_bridge":
        bridge_spec = _bridge_spec_from_training_config(resolved_config)
        if (
            resolved_config.bridge_tau_min is None
            or resolved_config.bridge_tau_max is None
        ):
            raise ValueError(
                "brownian_schrodinger_bridge requires explicit bridge_tau bounds"
            )
        validate_time_to_go_bounds(
            minimum=float(resolved_config.bridge_tau_min),
            maximum=float(resolved_config.bridge_tau_max),
            spec=bridge_spec,
        )
    train_cpu = _flat_finite_data(train, name="train")
    validation_cpu = _flat_finite_data(validation, name="validation")
    if train_cpu.shape[1] != validation_cpu.shape[1]:
        raise ValueError("train and validation ambient dimensions do not match")
    data_identity = {
        "schema_version": 1,
        "train_sha256": _tensor_identity_sha256(train_cpu),
        "validation_sha256": _tensor_identity_sha256(validation_cpu),
    }
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    progress_path = (
        None
        if progress_checkpoint_path is None
        else Path(progress_checkpoint_path).expanduser().resolve()
    )
    if progress_path is not None:
        paths_match = checkpoint == progress_path
        if checkpoint.exists() and progress_path.exists():
            paths_match = paths_match or os.path.samefile(checkpoint, progress_path)
        if paths_match:
            raise ValueError(
                "training progress path must differ from final checkpoint path"
            )
    mean, scale, preprocessing, preprocessing_sha256 = _normalization(
        train_cpu,
        enabled=resolved_config.normalize,
        epsilon=resolved_config.normalization_epsilon,
    )
    train_cpu = (train_cpu - mean) / scale
    validation_cpu = (validation_cpu - mean) / scale
    device = _resolve_device(resolved_config.device)
    covariance_span = None
    if resolved_config.field_preconditioning == 'covariance_span_v1':
        from models.preconditioned_field import training_covariance_span
        covariance_span = training_covariance_span(train_cpu, device,
            minimum_rank=2 if canonical_family == 'scale_conditioned_normalizing_flow' else 1)
        if resolved_config.optimizer_schedule_policy == 'shared_terminal_v1' and resolved_config.field_projection_rank is not None and resolved_config.field_projection_rank != covariance_span[0].shape[1]:
            raise ValueError('fitted covariance rank changed after capacity resolution')
        resolved_config = replace(resolved_config, field_projection_rank=covariance_span[0].shape[1])
    seed_everything(resolved_config.seed)

    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_cudnn_benchmark = torch.backends.cudnn.benchmark
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    previous_cuda_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    if resolved_config.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if canonical_family == "vp_diffusion":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    try:
        if canonical_family == "scale_conditioned_normalizing_flow":
            nf_architecture = _nf_architecture_from_training_config(
                resolved_config, ambient_dim=train_cpu.shape[1]
            )
            from models.preconditioned_nf import build_nf
            model: TrainableModel = build_nf(nf_architecture, resolved_config).to(device)
        elif canonical_family == "vp_diffusion":
            vp_architecture = _vp_architecture_from_training_config(
                resolved_config, ambient_dim=train_cpu.shape[1]
            )
            from models.preconditioned_field import build_bottleneck
            model = build_bottleneck(vp_architecture, canonical_family, resolved_config).to(device)
        elif resolved_config.native_preconditioning in IMAGE_PRECONDITIONING_MODES:
            image_architecture = _image_architecture(resolved_config,train_cpu.shape[1])
            model = NativeGaussianImageField(image_architecture,native_family=canonical_family).to(device)
            if image_architecture.prior == 'pixel_spike_gaussian_v1':
                model.pixel_prior.fit(train_cpu,-mean/scale)
        elif resolved_config.field_hidden_sizes is not None:
            bottleneck_architecture = _bottleneck_architecture_from_training_config(
                resolved_config,
                family=canonical_family,
                ambient_dim=train_cpu.shape[1],
            )
            from models.preconditioned_field import build_bottleneck
            model = build_bottleneck(bottleneck_architecture, canonical_family, resolved_config).to(device)
        else:
            if canonical_family == "independent_affine_flow":
                field_architecture = _affine_field_architecture_from_training_config(
                    resolved_config, ambient_dim=train_cpu.shape[1]
                )
            elif resolved_config.depth is None:
                raise ValueError(
                    f"{canonical_family} requires explicit neural-field depth"
                )
            else:
                field_architecture = NeuralFieldConfig(
                    ambient_dim=train_cpu.shape[1],
                    hidden_dim=resolved_config.hidden_dim,
                    depth=resolved_config.depth,
                    condition_dim=resolved_config.time_embedding_dim,
                    fourier_features=resolved_config.fourier_features,
                    max_condition_frequency=resolved_config.max_condition_frequency,
                    dropout=resolved_config.dropout,
                    condition_transform=(
                        "log"
                        if canonical_family
                        in {"gaussian_diffusion", "brownian_schrodinger_bridge"}
                        else "linear"
                    ),
                )
            model = ScaleConditionedNeuralField(field_architecture).to(device)
        if covariance_span is not None:
            with torch.no_grad():
                model.basis.copy_(covariance_span[0])
                model.variances.copy_(covariance_span[1])
                model.omitted_variance_fraction.copy_(covariance_span[2])
                if canonical_family == 'scale_conditioned_normalizing_flow':
                    complete_basis = torch.linalg.qr(model.basis.double(), mode='complete').Q
                    model.normal_basis.copy_(complete_basis[:, model.basis.shape[1]:])
        model._lid_family = canonical_family
        model._lid_noise_pairing = resolved_config.noise_pairing
        if canonical_family == "vp_diffusion":
            model._lid_vp_schedule = _vp_schedule_from_training_config(resolved_config)
        if canonical_family == "independent_affine_flow":
            model._lid_affine_spec = _affine_spec_from_training_config(resolved_config)
        train_tensor = train_cpu.to(device)
        validation_tensor = validation_cpu.to(device)
        if resolved_config.training_target == 'empirical_posterior_v1':
            from models.empirical_target import EmpiricalPosteriorTarget
            model._lid_empirical_teacher=EmpiricalPosteriorTarget(train_tensor)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=resolved_config.learning_rate,
            weight_decay=resolved_config.weight_decay,
        )
        shuffle_generator = torch.Generator(device="cpu")
        shuffle_generator.manual_seed(resolved_config.seed + 17)
        objective_generator = torch.Generator(device=device)
        objective_generator.manual_seed(resolved_config.seed + 31)

        if resolved_config.training_mode == "fixed_steps_v1":
            assert resolved_config.steps is not None
            assert resolved_config.warmup_steps is not None
            assert resolved_config.validation_interval_steps is not None
            if resolved_config.terminal_decay_steps and canonical_family == 'vp_diffusion' and resolved_config.optimizer_schedule_policy == 'native_v1':
                raise ValueError('native VP already uses cosine decay; terminal override is unsupported')
            scheduler = _FixedStepScheduler(
                optimizer,
                base_learning_rate=resolved_config.learning_rate,
                total_steps=resolved_config.steps,
                warmup_steps=resolved_config.warmup_steps,
                schedule=(
                    'terminal_cosine_v1' if resolved_config.terminal_decay_steps else
                    "warmup_cosine_v1"
                    if canonical_family == "vp_diffusion"
                    else "constant_v1"
                ),
                terminal_decay_steps=resolved_config.terminal_decay_steps,
                terminal_learning_rate_ratio=resolved_config.terminal_learning_rate_ratio,
            )
            step_history: list[StepMetrics] = []
            best_validation_loss = math.inf
            initial_validation_loss = _validation_loss(
                canonical_family,
                model,
                validation_tensor,
                resolved_config,
                seed=resolved_config.seed + 1_000_003,
            )
            if not math.isfinite(initial_validation_loss):
                raise FloatingPointError("non-finite initial validation loss")
            best_step = 0
            best_state: dict[str, Tensor] | None = None
            start_step = 1
            ema_model = None
            if progress_path is not None and progress_path.exists():
                (
                    start_step,
                    step_history,
                    best_step,
                    best_validation_loss,
                    initial_validation_loss,
                    best_state,
                ) = _load_fixed_training_progress(
                    progress_path,
                    family=canonical_family,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=resolved_config,
                    normalization_mean=mean,
                    normalization_scale=scale,
                    preprocessing=preprocessing,
                    preprocessing_sha256=preprocessing_sha256,
                    data_identity=data_identity,
                    sampler_generator=shuffle_generator,
                    objective_generator=objective_generator,
                    device=device,
                    expected_initial_validation_loss=initial_validation_loss,
                )
                if resolved_config.ema_decay is not None:
                    ema_state = torch.load(progress_path, map_location='cpu', weights_only=True)['ema_state']
                    if ema_state is not None:
                        ema_model = copy.deepcopy(model).requires_grad_(False)
                        ema_model.load_state_dict(ema_state, strict=True)
                    elif start_step > resolved_config.ema_start_step + 1:
                        raise ValueError('resumed EMA state is missing after its start')
            interval_loss_sum = 0.0
            interval_examples = 0
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            for step in range(start_step, resolved_config.steps + 1):
                model._lid_use_empirical_target=(resolved_config.training_target=='empirical_posterior_v1'
                                                and step>resolved_config.training_target_start_step)
                if resolved_config.ema_decay is not None and ema_model is None and step > resolved_config.ema_start_step:
                    ema_model = copy.deepcopy(model).requires_grad_(False)
                model.train()
                indices = torch.randint(
                    train_tensor.shape[0],
                    (resolved_config.batch_size,),
                    generator=shuffle_generator,
                    device="cpu",
                ).to(device)
                batch = train_tensor[indices]
                learning_rate = scheduler.prepare_step(step)
                optimizer.zero_grad(set_to_none=True)
                loss = _objective(
                    canonical_family,
                    model,
                    batch,
                    resolved_config,
                    objective_generator,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite training loss at step {step}")
                loss.backward()
                if resolved_config.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), resolved_config.gradient_clip_norm
                    )
                optimizer.step()
                if ema_model is not None:
                    with torch.no_grad():
                        torch._foreach_lerp_(list(ema_model.parameters()), list(model.parameters()), 1 - resolved_config.ema_decay)
                        # Running statistics must track their source; these fields
                        # currently have static covariance/Fourier buffers only.
                        for target, source in zip(ema_model.buffers(), model.buffers()):
                            target.copy_(source)
                scheduler.complete_step(step)
                interval_loss_sum += float(loss.item()) * batch.shape[0]
                interval_examples += batch.shape[0]

                should_validate = (
                    step % resolved_config.validation_interval_steps == 0
                    or step == resolved_config.steps
                )
                if not should_validate:
                    continue
                train_loss = interval_loss_sum / interval_examples
                validation_loss = _validation_loss(
                    canonical_family,
                    model,
                    validation_tensor,
                    resolved_config,
                    seed=resolved_config.seed + 1_000_003,
                )
                if not math.isfinite(validation_loss):
                    raise FloatingPointError(
                        f"non-finite validation loss at step {step}"
                    )
                raw_validation_loss = validation_loss
                ema_validation_loss = None
                candidate_model = model
                if ema_model is not None:
                    ema_validation_loss = _validation_loss(
                        canonical_family, ema_model, validation_tensor, resolved_config,
                        seed=resolved_config.seed + 1_000_003,
                    )
                    if not math.isfinite(ema_validation_loss):
                        raise FloatingPointError('non-finite EMA validation loss')
                    if ema_validation_loss < validation_loss:
                        validation_loss = ema_validation_loss
                        candidate_model = ema_model
                metric = StepMetrics(
                    step=step,
                    examples_seen=step * resolved_config.batch_size,
                    train_loss=train_loss,
                    validation_loss=validation_loss,
                    learning_rate=learning_rate,
                )
                step_history.append(metric)
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_step = step
                    best_state = _cpu_state_dict(candidate_model)
                if progress_path is not None:
                    assert best_state is not None
                    _atomic_torch_save(
                        progress_path,
                        _fixed_training_progress_payload(
                            family=canonical_family,
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            config=resolved_config,
                            global_step=step,
                            history=tuple(step_history),
                            best_step=best_step,
                            best_validation_loss=best_validation_loss,
                            initial_validation_loss=initial_validation_loss,
                            best_state=best_state,
                            normalization_mean=mean,
                            normalization_scale=scale,
                            preprocessing=preprocessing,
                            preprocessing_sha256=preprocessing_sha256,
                            data_identity=data_identity,
                            sampler_generator=shuffle_generator,
                            objective_generator=objective_generator,
                            device=device,
                            ema_model=ema_model,
                        ),
                    )
                if log_callback is not None:
                    log_callback(
                        {
                            **metric.to_dict(),
                            "best_step": best_step,
                            "best_validation_loss": best_validation_loss,
                            "parameter_count": parameter_count,
                            "validated": True,
                            **({'raw_validation_loss': raw_validation_loss,
                                'ema_validation_loss': ema_validation_loss,
                                'validation_candidate': 'ema' if candidate_model is ema_model else 'raw'}
                               if resolved_config.ema_decay is not None else {}),
                        }
                    )
                interval_loss_sum = 0.0
                interval_examples = 0
            if best_state is None or not step_history:
                raise RuntimeError("training ended without a validation measurement")
            final_state = _cpu_state_dict(model)
            model.load_state_dict(best_state, strict=True)
            model.eval()
            history = tuple(step_history)
            checkpoint_sha256, weights_metadata = _save_fixed_step_checkpoint(
                checkpoint,
                family=canonical_family,
                model=model,
                config=resolved_config,
                history=history,
                best_step=best_step,
                best_validation_loss=best_validation_loss,
                best_state=best_state,
                final_state=final_state,
                initial_validation_loss=initial_validation_loss,
                normalization_mean=mean,
                normalization_scale=scale,
                preprocessing=preprocessing,
                preprocessing_sha256=preprocessing_sha256,
            )
            if progress_path is not None:
                progress_path.unlink(missing_ok=True)
            return TrainingResult(
                family=canonical_family,
                model=model,
                config=resolved_config,
                history=history,
                best_epoch=best_step,
                best_validation_loss=best_validation_loss,
                checkpoint_path=checkpoint,
                checkpoint_sha256=checkpoint_sha256,
                normalization_mean=mean.detach().cpu().contiguous(),
                normalization_scale=scale,
                preprocessing=preprocessing,
                preprocessing_sha256=preprocessing_sha256,
                weights_metadata=weights_metadata,
                final_model_state=final_state,
            )

        history_list: list[EpochMetrics] = []
        best_validation_loss = math.inf
        best_epoch = 0
        best_state: dict[str, Tensor] | None = None
        stale_validations = 0
        start_epoch = 1
        if progress_path is not None and progress_path.exists():
            (
                start_epoch,
                history_list,
                best_epoch,
                best_validation_loss,
                best_state,
                stale_validations,
            ) = _load_training_progress(
                progress_path,
                family=canonical_family,
                model=model,
                optimizer=optimizer,
                config=resolved_config,
                normalization_mean=mean,
                normalization_scale=scale,
                preprocessing=preprocessing,
                preprocessing_sha256=preprocessing_sha256,
                data_identity=data_identity,
                shuffle_generator=shuffle_generator,
                objective_generator=objective_generator,
                device=device,
            )
            if (
                resolved_config.early_stopping_patience is not None
                and stale_validations >= resolved_config.early_stopping_patience
            ):
                start_epoch = resolved_config.epochs + 1
        for epoch in range(start_epoch, resolved_config.epochs + 1):
            model.train()
            permutation = torch.randperm(
                train_tensor.shape[0], generator=shuffle_generator
            )
            accumulated = 0.0
            for start in range(0, train_tensor.shape[0], resolved_config.batch_size):
                indices = permutation[start : start + resolved_config.batch_size].to(
                    device
                )
                batch = train_tensor[indices]
                optimizer.zero_grad(set_to_none=True)
                loss = _objective(
                    canonical_family,
                    model,
                    batch,
                    resolved_config,
                    objective_generator,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite training loss at epoch {epoch}"
                    )
                loss.backward()
                if resolved_config.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), resolved_config.gradient_clip_norm
                    )
                optimizer.step()
                accumulated += float(loss.item()) * batch.shape[0]
            train_loss = accumulated / train_tensor.shape[0]

            should_validate = (
                epoch % resolved_config.validation_interval == 0
                or epoch == resolved_config.epochs
            )
            if not should_validate:
                if log_callback is not None:
                    log_callback(
                        {
                            "epoch": epoch,
                            "train_loss": train_loss,
                            "validated": False,
                        }
                    )
                continue
            validation_loss = _validation_loss(
                canonical_family,
                model,
                validation_tensor,
                resolved_config,
                seed=resolved_config.seed + 1_000_003,
            )
            if not math.isfinite(validation_loss):
                raise FloatingPointError(f"non-finite validation loss at epoch {epoch}")
            metric = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                validation_loss=validation_loss,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
            )
            history_list.append(metric)
            improved = validation_loss < best_validation_loss
            if improved:
                best_validation_loss = validation_loss
                best_epoch = epoch
                best_state = _cpu_state_dict(model)
                stale_validations = 0
            else:
                stale_validations += 1
            if progress_path is not None:
                patience_reached = (
                    resolved_config.early_stopping_patience is not None
                    and stale_validations >= resolved_config.early_stopping_patience
                )
                if (
                    epoch % TRAINING_PROGRESS_INTERVAL_EPOCHS == 0
                    or epoch == resolved_config.epochs
                    or patience_reached
                ):
                    assert best_state is not None
                    progress_payload = _training_progress_payload(
                        family=canonical_family,
                        model=model,
                        optimizer=optimizer,
                        config=resolved_config,
                        epoch=epoch,
                        history=tuple(history_list),
                        best_epoch=best_epoch,
                        best_validation_loss=best_validation_loss,
                        best_state=best_state,
                        stale_validations=stale_validations,
                        normalization_mean=mean,
                        normalization_scale=scale,
                        preprocessing=preprocessing,
                        preprocessing_sha256=preprocessing_sha256,
                        data_identity=data_identity,
                        shuffle_generator=shuffle_generator,
                        objective_generator=objective_generator,
                        device=device,
                    )
                    _atomic_torch_save(progress_path, progress_payload)
            if log_callback is not None:
                log_callback(
                    {
                        **metric.to_dict(),
                        "best_validation_loss": best_validation_loss,
                        "validated": True,
                    }
                )
            if (
                resolved_config.early_stopping_patience is not None
                and stale_validations >= resolved_config.early_stopping_patience
            ):
                break
        if best_state is None or not history_list:
            raise RuntimeError("training ended without a validation measurement")
        model.load_state_dict(best_state, strict=True)
        model.eval()
        history = tuple(history_list)
        checkpoint_sha256 = _save_checkpoint(
            checkpoint,
            family=canonical_family,
            model=model,
            config=resolved_config,
            history=history,
            best_epoch=best_epoch,
            best_validation_loss=best_validation_loss,
            normalization_mean=mean,
            normalization_scale=scale,
            preprocessing=preprocessing,
            preprocessing_sha256=preprocessing_sha256,
        )
        if progress_path is not None:
            progress_path.unlink(missing_ok=True)
        return TrainingResult(
            family=canonical_family,
            model=model,
            config=resolved_config,
            history=history,
            best_epoch=best_epoch,
            best_validation_loss=best_validation_loss,
            checkpoint_path=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            normalization_mean=mean.detach().cpu().contiguous(),
            normalization_scale=scale,
            preprocessing=preprocessing,
            preprocessing_sha256=preprocessing_sha256,
        )
    finally:
        if resolved_config.deterministic:
            torch.use_deterministic_algorithms(previous_deterministic)
            torch.backends.cudnn.benchmark = previous_cudnn_benchmark
            torch.backends.cudnn.deterministic = previous_cudnn_deterministic
        if canonical_family == "vp_diffusion":
            torch.backends.cuda.matmul.allow_tf32 = previous_cuda_matmul_tf32
            torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32


def load_checkpoint(
    checkpoint_path: str | Path, *, device: str = "cpu"
) -> TrainingResult:
    """Load a portable weights-only checkpoint and validate its preprocessing."""

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    resolved_device = _resolve_device(device)
    payload = torch.load(path, map_location=resolved_device, weights_only=True)
    legacy_required = {
        "schema_version",
        "family",
        "architecture",
        "training_config",
        "model_state",
        "history",
        "best_epoch",
        "best_validation_loss",
        "normalization",
    }
    fixed_required = {
        "schema_version",
        "family",
        "model_contract",
        "architecture",
        "training_config",
        "model_state",
        "final_model_state",
        "history",
        "best_step",
        "best_validation_loss",
        "global_step",
        "examples_seen_total",
        "examples_seen_at_selected_checkpoint",
        "weights_metadata",
        "normalization",
    }
    if not isinstance(payload, dict):
        raise TypeError("checkpoint payload must be a mapping")
    schema_version = payload.get("schema_version")
    if schema_version == LEGACY_CHECKPOINT_SCHEMA_VERSION:
        required = legacy_required
    elif schema_version == CHECKPOINT_SCHEMA_VERSION:
        required = legacy_required | {"model_contract"}
    elif schema_version == FIXED_STEP_CHECKPOINT_SCHEMA_VERSION:
        required = fixed_required
    else:
        raise ValueError("unsupported checkpoint schema_version")
    if set(payload) != required:
        raise ValueError("checkpoint schema mismatch")
    family = _canonical_family(payload["family"])
    if schema_version == LEGACY_CHECKPOINT_SCHEMA_VERSION and family in {
        "scale_conditioned_normalizing_flow",
        "independent_affine_flow",
    }:
        raise ValueError(f"{family} requires checkpoint schema_version 2")
    if schema_version == LEGACY_CHECKPOINT_SCHEMA_VERSION and family == "vp_diffusion":
        raise ValueError("vp_diffusion requires fixed-step checkpoint schema")
    config = TrainingConfig.from_mapping(payload["training_config"])
    if (
        schema_version == FIXED_STEP_CHECKPOINT_SCHEMA_VERSION
        and config.training_mode != "fixed_steps_v1"
    ):
        raise ValueError("fixed-step checkpoint training mode mismatch")
    if (
        schema_version != FIXED_STEP_CHECKPOINT_SCHEMA_VERSION
        and config.training_mode != "epochs_v1"
    ):
        raise ValueError("epoch checkpoint training mode mismatch")
    if family != "independent_affine_flow" and _has_affine_settings(config):
        raise ValueError(
            "checkpoint carries inactive independent affine-flow settings for "
            f"family {family}"
        )
    if family == "vp_diffusion":
        _vp_architecture_from_training_config(config, ambient_dim=2)
        _vp_schedule_from_training_config(config)
        if schema_version != FIXED_STEP_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("vp_diffusion requires fixed-step checkpoint schema")
    elif _has_vp_settings(config):
        raise ValueError(f"checkpoint carries inactive VP settings for family {family}")
    expected_contract = _model_contract(family, config)
    if (
        schema_version
        in {CHECKPOINT_SCHEMA_VERSION, FIXED_STEP_CHECKPOINT_SCHEMA_VERSION}
        and payload["model_contract"] != expected_contract
    ):
        raise ValueError("checkpoint model_contract mismatch")
    if family == "scale_conditioned_normalizing_flow":
        flow_architecture = ConditionalFlowConfig.from_mapping(payload["architecture"])
        declared_architecture = _nf_architecture_from_training_config(
            config, ambient_dim=flow_architecture.ambient_dim
        )
        if declared_architecture != flow_architecture:
            raise ValueError(
                "checkpoint NF architecture does not match training_config"
            )
        from models.preconditioned_nf import build_nf
        model: TrainableModel = build_nf(flow_architecture, config).to(resolved_device)
        ambient_dim = flow_architecture.ambient_dim
    elif family == "vp_diffusion":
        vp_architecture = VPBottleneckConfig.from_mapping(payload["architecture"])
        declared_architecture = _vp_architecture_from_training_config(
            config, ambient_dim=vp_architecture.ambient_dim
        )
        if declared_architecture != vp_architecture:
            raise ValueError(
                "checkpoint VP architecture does not match training_config"
            )
        from models.preconditioned_field import build_bottleneck
        model = build_bottleneck(vp_architecture, family, config).to(resolved_device)
        ambient_dim = vp_architecture.ambient_dim
    elif config.native_preconditioning in IMAGE_PRECONDITIONING_MODES:
        image_architecture = ImageFieldConfig.from_mapping(payload["architecture"])
        declared_architecture = _image_architecture(config,image_architecture.ambient_dim)
        if image_architecture != declared_architecture:
            raise ValueError("checkpoint image architecture does not match training_config")
        model = NativeGaussianImageField(image_architecture,native_family=family).to(resolved_device)
        ambient_dim = image_architecture.ambient_dim
    elif config.field_hidden_sizes is not None:
        bottleneck_architecture = VPBottleneckConfig.from_mapping(
            payload["architecture"]
        )
        declared_architecture = _bottleneck_architecture_from_training_config(
            config,
            family=family,
            ambient_dim=bottleneck_architecture.ambient_dim,
        )
        if declared_architecture != bottleneck_architecture:
            raise ValueError(
                "checkpoint bottleneck architecture does not match training_config"
            )
        from models.preconditioned_field import build_bottleneck
        model = build_bottleneck(bottleneck_architecture, family, config).to(resolved_device)
        ambient_dim = bottleneck_architecture.ambient_dim
    else:
        field_architecture = NeuralFieldConfig.from_mapping(payload["architecture"])
        if family == "independent_affine_flow":
            expected_architecture = _affine_field_architecture_from_training_config(
                config, ambient_dim=field_architecture.ambient_dim
            )
            if field_architecture != expected_architecture:
                raise ValueError(
                    "checkpoint affine neural-field architecture does not match "
                    "training_config"
                )
        model = ScaleConditionedNeuralField(field_architecture).to(resolved_device)
        ambient_dim = field_architecture.ambient_dim
    model.load_state_dict(payload["model_state"], strict=True)
    model._lid_family = family
    model._lid_noise_pairing = config.noise_pairing
    if family == "vp_diffusion":
        model._lid_vp_schedule = _vp_schedule_from_training_config(config)
    if family == "independent_affine_flow":
        model._lid_affine_spec = _affine_spec_from_training_config(config)
    model.eval()

    history_raw = payload["history"]
    if not isinstance(history_raw, list) or not history_raw:
        raise ValueError("checkpoint history must be a non-empty list")
    history: list[EpochMetrics | StepMetrics] = []
    weights_metadata: Mapping[str, Any] = {}
    final_model_state: Mapping[str, Tensor] | None = None
    if schema_version == FIXED_STEP_CHECKPOINT_SCHEMA_VERSION:
        step_fields = {
            "step",
            "examples_seen",
            "train_loss",
            "validation_loss",
            "learning_rate",
        }
        for item in history_raw:
            if not isinstance(item, dict) or set(item) != step_fields:
                raise ValueError("fixed-step checkpoint history entry schema mismatch")
            metric = StepMetrics(**item)
            if metric.examples_seen != metric.step * config.batch_size or not all(
                math.isfinite(float(value))
                for value in (
                    metric.train_loss,
                    metric.validation_loss,
                    metric.learning_rate,
                )
            ):
                raise ValueError("fixed-step checkpoint history entry is invalid")
            history.append(metric)
        typed_step_history = [
            metric for metric in history if isinstance(metric, StepMetrics)
        ]
        if any(left.step >= right.step for left, right in pairwise(typed_step_history)):
            raise ValueError("fixed-step checkpoint history is not ordered")
        assert config.steps is not None
        global_step = payload["global_step"]
        if global_step != config.steps or typed_step_history[-1].step != global_step:
            raise ValueError("fixed-step checkpoint global_step mismatch")
        if payload["examples_seen_total"] != global_step * config.batch_size:
            raise ValueError("fixed-step checkpoint total exposure mismatch")
        best_epoch = payload["best_step"]
        if (
            isinstance(best_epoch, bool)
            or not isinstance(best_epoch, int)
            or best_epoch not in {metric.step for metric in typed_step_history}
        ):
            raise ValueError("fixed-step checkpoint best_step is invalid")
        if (
            payload["examples_seen_at_selected_checkpoint"]
            != best_epoch * config.batch_size
        ):
            raise ValueError("fixed-step checkpoint selected exposure mismatch")
        best_validation_loss = float(payload["best_validation_loss"])
        selected_metric = next(
            metric for metric in typed_step_history if metric.step == best_epoch
        )
        if (
            not math.isfinite(best_validation_loss)
            or best_validation_loss != selected_metric.validation_loss
            or best_epoch
            != min(
                typed_step_history,
                key=lambda metric: (metric.validation_loss, metric.step),
            ).step
        ):
            raise ValueError("fixed-step checkpoint best selection is invalid")
        raw_final_state = payload["final_model_state"]
        if not isinstance(raw_final_state, dict) or not raw_final_state:
            raise ValueError("fixed-step checkpoint final_model_state is invalid")
        model.load_state_dict(raw_final_state, strict=True)
        model.load_state_dict(payload["model_state"], strict=True)
        final_model_state = {
            str(name): torch.as_tensor(value).detach().cpu().clone().contiguous()
            for name, value in raw_final_state.items()
        }
        raw_weights_metadata = payload["weights_metadata"]
        raw_initial = (
            raw_weights_metadata.get("initial")
            if isinstance(raw_weights_metadata, Mapping)
            else None
        )
        if not isinstance(raw_initial, Mapping):
            raise ValueError("fixed-step checkpoint initial metadata is invalid")
        initial_validation_loss = float(raw_initial.get("validation_loss", math.nan))
        if not math.isfinite(initial_validation_loss):
            raise ValueError("fixed-step checkpoint initial loss is invalid")
        expected_weights_metadata = _fixed_weights_metadata(
            best_state=payload["model_state"],
            final_state=final_model_state,
            best_step=best_epoch,
            final_step=global_step,
            batch_size=config.batch_size,
            initial_validation_loss=initial_validation_loss,
        )
        if raw_weights_metadata != expected_weights_metadata:
            raise ValueError("fixed-step checkpoint weights metadata mismatch")
        weights_metadata = expected_weights_metadata
    else:
        epoch_fields = {
            "epoch",
            "train_loss",
            "validation_loss",
            "learning_rate",
        }
        for item in history_raw:
            if not isinstance(item, dict) or set(item) != epoch_fields:
                raise ValueError("checkpoint history entry schema mismatch")
            history.append(EpochMetrics(**item))
    normalization = payload["normalization"]
    if not isinstance(normalization, dict) or set(normalization) != {
        "mean",
        "scale",
        "preprocessing",
        "sha256",
    }:
        raise ValueError("checkpoint normalization schema mismatch")
    mean = torch.as_tensor(normalization["mean"]).detach().cpu().float().contiguous()
    if mean.shape != (ambient_dim,) or not torch.isfinite(mean).all():
        raise ValueError("checkpoint normalization mean is invalid")
    scale = float(normalization["scale"])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("checkpoint normalization scale is invalid")
    preprocessing, preprocessing_sha256 = _preprocessing_identity(
        mean, scale, normalized=config.normalize
    )
    if (
        normalization["preprocessing"] != preprocessing
        or normalization["sha256"] != preprocessing_sha256
    ):
        raise ValueError("checkpoint preprocessing identity mismatch")
    if schema_version != FIXED_STEP_CHECKPOINT_SCHEMA_VERSION:
        best_epoch = int(payload["best_epoch"])
        best_validation_loss = float(payload["best_validation_loss"])
        epoch_history = [
            metric for metric in history if isinstance(metric, EpochMetrics)
        ]
        if best_epoch not in {metric.epoch for metric in epoch_history}:
            raise ValueError("checkpoint best_epoch is absent from history")
        if not math.isfinite(best_validation_loss):
            raise ValueError("checkpoint best_validation_loss must be finite")
    return TrainingResult(
        family=family,
        model=model,
        config=config,
        history=tuple(history),
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        checkpoint_path=path,
        checkpoint_sha256=_checkpoint_sha256(path),
        normalization_mean=mean,
        normalization_scale=scale,
        preprocessing=preprocessing,
        preprocessing_sha256=preprocessing_sha256,
        weights_metadata=weights_metadata,
        final_model_state=final_model_state,
    )


def _model_and_context(
    model_or_result: nn.Module | TrainingResult,
    family: str | None,
) -> tuple[nn.Module, Family, Tensor, float]:
    if isinstance(model_or_result, TrainingResult):
        canonical = model_or_result.family
        if family is not None and _canonical_family(family) != canonical:
            raise ValueError("family does not match TrainingResult")
        return (
            model_or_result.model,
            canonical,
            model_or_result.normalization_mean,
            model_or_result.normalization_scale,
        )
    model = model_or_result
    inferred = family or getattr(model, "_lid_family", None)
    if inferred is None:
        raise ValueError("family is required when passing a bare model")
    canonical = _canonical_family(inferred)
    if canonical == "scale_conditioned_normalizing_flow" and not isinstance(
        model, ScaleConditionedRealNVP
    ):
        raise TypeError("scale-conditioned NF family requires ScaleConditionedRealNVP")
    if canonical == "vp_diffusion" and not isinstance(model, VPBottleneckMLP):
        raise TypeError("vp_diffusion family requires VPBottleneckMLP")
    mean = torch.zeros(model.config.ambient_dim, dtype=torch.float32)
    return model, canonical, mean, 1.0


def _vp_schedule_for_model(
    model_or_result: VPBottleneckMLP | TrainingResult,
) -> VPSchedule:
    if isinstance(model_or_result, TrainingResult):
        return _vp_schedule_from_training_config(model_or_result.config)
    schedule = getattr(model_or_result, "_lid_vp_schedule", None)
    if isinstance(schedule, VPSchedule):
        return schedule
    raise ValueError(
        "bare VP models require their checkpointed _lid_vp_schedule contract"
    )


def _bridge_spec_for_model(
    model_or_result: ScaleConditionedNeuralField | TrainingResult,
) -> BrownianBridgeSpec:
    """Recover the checkpointed bridge identity; bare bridge models are unsafe."""

    if isinstance(model_or_result, TrainingResult):
        return _bridge_spec_from_training_config(model_or_result.config)
    raise ValueError(
        "bare Schrodinger-bridge models are forbidden because the explicit "
        "Hydra bridge contract is unavailable"
    )


def _affine_spec_for_model(
    model_or_result: ScaleConditionedNeuralField | TrainingResult,
) -> AffineFlowSpec:
    """Recover the immutable affine-FM contract used for training."""

    if isinstance(model_or_result, TrainingResult):
        return _affine_spec_from_training_config(model_or_result.config)
    spec = getattr(model_or_result, "_lid_affine_spec", None)
    if isinstance(spec, AffineFlowSpec):
        return spec
    raise ValueError(
        "bare independent-affine models require their checkpointed "
        "_lid_affine_spec contract"
    )


class _PosteriorFromAffineVelocity(nn.Module):
    """View a direct affine velocity network as its posterior-mean field."""

    def __init__(
        self,
        velocity_model: nn.Module,
        state: Any,
    ) -> None:
        super().__init__()
        self.velocity_model = velocity_model
        self.state = state

    def forward(self, inputs: Tensor, condition: Tensor | float) -> Tensor:
        velocity = self.velocity_model(inputs, condition)
        return velocity_to_posterior(velocity, inputs, self.state)


def predict_affine_primitives(
    model_or_result: nn.Module | TrainingResult,
    query: Any,
    noise_ratio: float,
    *,
    family: str | None = None,
    divergence_backend: Literal["exact", "hutchinson"] = "hutchinson",
    trace_probes: int = 16,
    trace_seed: int = 0,
    batch_size: int = 256,
) -> AffineFieldPrediction:
    """Evaluate all equivalent affine-FM fields at one physical ``lambda``.

    The primary trace is always taken through the posterior field.  For the
    posterior rectified variant this differentiates the raw network output,
    never its singular reconstructed velocity.  Direct-velocity variants are
    wrapped by their exact affine posterior identity before differentiation.
    """

    model, canonical_family, mean, normalization_scale = _model_and_context(
        model_or_result, family
    )
    if canonical_family != "independent_affine_flow":
        raise ValueError("predict_affine_primitives requires independent_affine_flow")
    spec = _affine_spec_for_model(model_or_result)
    if not math.isfinite(noise_ratio) or noise_ratio <= 0.0:
        raise ValueError("affine-FM noise ratio must be finite and positive")
    if not spec.noise_ratio_min <= noise_ratio <= spec.noise_ratio_max:
        raise ValueError(
            "affine-FM noise ratio lies outside the checkpointed training interval"
        )
    if divergence_backend not in {"exact", "hutchinson"}:
        raise ValueError("divergence_backend must be exact or hutchinson")
    if isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if divergence_backend == "hutchinson" and (
        isinstance(trace_probes, bool) or trace_probes <= 0
    ):
        raise ValueError("trace_probes must be positive for Hutchinson divergence")
    if isinstance(trace_seed, bool) or trace_seed < 0:
        raise ValueError("trace_seed must be a non-negative integer")

    query_cpu = _flat_finite_data(query, name="query")
    if query_cpu.shape[1] != model.config.ambient_dim:
        raise ValueError("query ambient dimension does not match model")
    normalized = (query_cpu - mean.reshape(1, -1)) / normalization_scale
    parameter = next(model.parameters())
    device = parameter.device
    dtype = parameter.dtype
    model.eval()
    generator: torch.Generator | None = None
    if divergence_backend == "hutchinson":
        generator = torch.Generator(device=device)
        generator.manual_seed(trace_seed)

    model_outputs: list[Tensor] = []
    model_output_divergences: list[Tensor] = []
    velocities: list[Tensor] = []
    velocity_divergences: list[Tensor] = []
    velocity_divergences_from_posterior: list[Tensor] = []
    posterior_means: list[Tensor] = []
    posterior_divergences: list[Tensor] = []
    marginal_scores: list[Tensor] = []
    marginal_score_divergences: list[Tensor] = []
    channel_scores: list[Tensor] = []
    channel_score_divergences: list[Tensor] = []
    evaluation_points: list[Tensor] = []
    channel_points: list[Tensor] = []
    model_conditions: list[Tensor] = []

    parameterization = canonical_parameterization(spec.parameterization)
    for start in range(0, normalized.shape[0], batch_size):
        channel_point = normalized[start : start + batch_size].to(
            device=device, dtype=dtype
        )
        batch_noise_ratio = torch.full(
            (channel_point.shape[0],),
            float(noise_ratio),
            device=device,
            dtype=dtype,
        )
        state = affine_schedule_state(batch_noise_ratio, spec.schedule)
        alpha = state.alpha.reshape(-1, 1)
        evaluation_point = alpha * channel_point
        condition = schedule_condition(state, spec.conditioning)

        if parameterization == "posterior_mean":
            posterior_field: nn.Module = model
        else:
            posterior_field = _PosteriorFromAffineVelocity(model, state)
        if divergence_backend == "exact":
            posterior_divergence = exact_divergence(
                posterior_field,
                evaluation_point,
                condition,
                create_graph=False,
            )
            raw_model_divergence = (
                posterior_divergence
                if parameterization == "posterior_mean"
                else exact_divergence(
                    model,
                    evaluation_point,
                    condition,
                    create_graph=False,
                )
            )
        else:
            assert generator is not None
            probes = rademacher_probes_like(
                evaluation_point,
                num_probes=trace_probes,
                seed=None,
                generator=generator,
            )
            posterior_divergence = hutchinson_divergence(
                posterior_field,
                evaluation_point,
                condition,
                num_probes=trace_probes,
                seed=None,
                probes=probes,
                create_graph=False,
            )
            raw_model_divergence = (
                posterior_divergence
                if parameterization == "posterior_mean"
                else hutchinson_divergence(
                    model,
                    evaluation_point,
                    condition,
                    num_probes=trace_probes,
                    seed=None,
                    probes=probes,
                    create_graph=False,
                )
            )
        with torch.no_grad():
            model_output = model(evaluation_point, condition)
            identity_state = affine_schedule_state(
                torch.full(
                    (channel_point.shape[0],),
                    float(noise_ratio),
                    device=device,
                    dtype=torch.float64,
                ),
                spec.schedule,
            )
            velocity_divergence_from_posterior = (
                posterior_divergence_to_velocity_divergence(
                    posterior_divergence.to(torch.float64),
                    identity_state,
                    ambient_dim=model.config.ambient_dim,
                )
            )
            if parameterization == "posterior_mean":
                posterior_mean = model_output
                velocity = posterior_to_velocity(
                    posterior_mean, evaluation_point, state
                )
                velocity_divergence = velocity_divergence_from_posterior
                model_output_divergence = posterior_divergence
            else:
                velocity = model_output
                posterior_mean = velocity_to_posterior(
                    velocity, evaluation_point, state
                )
                velocity_divergence = raw_model_divergence
                model_output_divergence = raw_model_divergence
            marginal_score = posterior_to_marginal_score(
                posterior_mean, evaluation_point, state
            )
            marginal_score_divergence = (
                posterior_divergence_to_marginal_score_divergence(
                    posterior_divergence,
                    state,
                    ambient_dim=model.config.ambient_dim,
                )
            )
            channel_score = posterior_to_channel_score(
                posterior_mean, channel_point, state
            )
            channel_score_divergence = posterior_divergence_to_channel_score_divergence(
                posterior_divergence,
                state,
                ambient_dim=model.config.ambient_dim,
            )
        model_outputs.append(model_output.detach().cpu())
        model_output_divergences.append(model_output_divergence.detach().cpu())
        velocities.append(velocity.detach().cpu())
        velocity_divergences.append(velocity_divergence.detach().cpu())
        velocity_divergences_from_posterior.append(
            velocity_divergence_from_posterior.detach().cpu()
        )
        posterior_means.append(posterior_mean.detach().cpu())
        posterior_divergences.append(posterior_divergence.detach().cpu())
        marginal_scores.append(marginal_score.detach().cpu())
        marginal_score_divergences.append(marginal_score_divergence.detach().cpu())
        channel_scores.append(channel_score.detach().cpu())
        channel_score_divergences.append(channel_score_divergence.detach().cpu())
        evaluation_points.append(evaluation_point.detach().cpu())
        channel_points.append(channel_point.detach().cpu())
        model_conditions.append(condition.detach().cpu())

    scalar_state = affine_schedule_state(
        torch.tensor(float(noise_ratio), dtype=torch.float64), spec.schedule
    )

    def array(values: list[Tensor]) -> npt.NDArray[np.float64]:
        return np.asarray(torch.cat(values).numpy(), dtype=np.float64)

    return AffineFieldPrediction(
        model_output=array(model_outputs),
        model_output_divergence=array(model_output_divergences),
        velocity=array(velocities),
        velocity_divergence=array(velocity_divergences),
        velocity_divergence_from_posterior=array(velocity_divergences_from_posterior),
        posterior_mean=array(posterior_means),
        posterior_divergence=array(posterior_divergences),
        marginal_score=array(marginal_scores),
        marginal_score_divergence=array(marginal_score_divergences),
        channel_score=array(channel_scores),
        channel_score_divergence=array(channel_score_divergences),
        evaluation_point=array(evaluation_points),
        channel_point=array(channel_points),
        variant_id=spec.variant_id,
        schedule=spec.schedule,
        parameterization=parameterization,
        noise_ratio=float(noise_ratio),
        native_time=float(scalar_state.native_time.item()),
        alpha=float(scalar_state.alpha.item()),
        beta=float(scalar_state.beta.item()),
        alpha_derivative=float(scalar_state.alpha_derivative.item()),
        beta_derivative=float(scalar_state.beta_derivative.item()),
        alpha_log_derivative=float(scalar_state.alpha_log_derivative.item()),
        log_noise_ratio_derivative=float(
            scalar_state.log_noise_ratio_derivative.item()
        ),
        model_condition=float(torch.cat(model_conditions)[0].item()),
        divergence_backend=divergence_backend,
        trace_probe_kind=(
            "rademacher" if divergence_backend == "hutchinson" else "exact"
        ),
        trace_seed=trace_seed if divergence_backend == "hutchinson" else None,
        trace_probes=trace_probes if divergence_backend == "hutchinson" else 0,
        shared_posterior_velocity_probes=(
            divergence_backend == "hutchinson" and parameterization == "direct_velocity"
        ),
        primary_trace_field="posterior_mean",
        velocity_divergence_source=(
            "raw_model_trace"
            if parameterization == "direct_velocity"
            else "derived_from_posterior_trace"
        ),
    )


def predict_primitives(
    model_or_result: nn.Module | TrainingResult,
    query: Any,
    scale: float,
    *,
    family: str | None = None,
    divergence_backend: Literal["exact", "hutchinson"] = "hutchinson",
    trace_probes: int = 16,
    trace_seed: int = 0,
    batch_size: int = 256,
) -> FieldPrediction:
    """Evaluate score/velocity and divergence at one model-space scale.

    Scale grids are defined after the checkpointed preprocessing transform,
    exactly like ``sigma_min``/``sigma_max`` used during training.
    """

    model, canonical_family, mean, normalization_scale = _model_and_context(
        model_or_result, family
    )
    if canonical_family == "independent_affine_flow":
        affine = predict_affine_primitives(
            model_or_result,
            query,
            scale,
            family=canonical_family,
            divergence_backend=divergence_backend,
            trace_probes=trace_probes,
            trace_seed=trace_seed,
            batch_size=batch_size,
        )
        return FieldPrediction(
            field=affine.velocity,
            divergence=affine.velocity_divergence,
            evaluation_point=affine.evaluation_point,
            condition=affine.noise_ratio,
        )
    if canonical_family == "scale_conditioned_normalizing_flow":
        raise ValueError(
            "scale-conditioned NF exposes a fixed-likelihood readout, not "
            "vector-field primitives"
        )
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    if canonical_family == "rectified_flow" and not 0 < scale < 1:
        raise ValueError("rectified-flow scale t must lie strictly between 0 and 1")
    if canonical_family == "brownian_schrodinger_bridge":
        bridge_spec = _bridge_spec_for_model(model_or_result)
        if not 0 < scale <= bridge_spec.terminal_time:
            raise ValueError(
                "Schrodinger-bridge time-to-go tau must lie in (0, terminal_time]"
            )
    if divergence_backend not in {"exact", "hutchinson"}:
        raise ValueError("divergence_backend must be exact or hutchinson")
    if isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if divergence_backend == "hutchinson" and (
        isinstance(trace_probes, bool) or trace_probes <= 0
    ):
        raise ValueError("trace_probes must be positive for Hutchinson divergence")
    if isinstance(trace_seed, bool) or trace_seed < 0:
        raise ValueError("trace_seed must be a non-negative integer")

    query_cpu = _flat_finite_data(query, name="query")
    if query_cpu.shape[1] != model.config.ambient_dim:
        raise ValueError("query ambient dimension does not match model")
    normalized = (query_cpu - mean.reshape(1, -1)) / normalization_scale
    parameter = next(model.parameters())
    device = parameter.device
    dtype = parameter.dtype
    model.eval()
    generator: torch.Generator | None = None
    if divergence_backend == "hutchinson":
        generator = torch.Generator(device=device)
        generator.manual_seed(trace_seed)

    if canonical_family == "vp_diffusion":
        schedule = _vp_schedule_for_model(model_or_result)
        schedule.time_for_lambda(scale)
        vp_fields: list[Tensor] = []
        vp_divergences: list[Tensor] = []
        vp_points: list[Tensor] = []
        native_sigma: float | None = None
        for start in range(0, normalized.shape[0], batch_size):
            data_batch = normalized[start : start + batch_size].to(
                device=device, dtype=dtype
            )
            primitives = vp_primitives(
                model,
                data_batch,
                scale,
                schedule=schedule,
                probes=trace_probes if divergence_backend == "hutchinson" else 0,
                seed=trace_seed,
                generator=generator,
            )
            vp_fields.append(primitives.score.detach().cpu())
            vp_divergences.append(primitives.score_divergence.detach().cpu())
            vp_points.append(primitives.native_query.detach().cpu())
            native_sigma = primitives.sigma
        assert native_sigma is not None
        return FieldPrediction(
            field=np.asarray(torch.cat(vp_fields).numpy(), dtype=np.float64),
            divergence=np.asarray(torch.cat(vp_divergences).numpy(), dtype=np.float64),
            evaluation_point=np.asarray(torch.cat(vp_points).numpy(), dtype=np.float64),
            condition=native_sigma,
        )

    fields: list[Tensor] = []
    divergences: list[Tensor] = []
    points: list[Tensor] = []
    model_scale = scale
    if model_scale <= 0:
        raise ValueError("normalized diffusion scale must be positive")
    for start in range(0, normalized.shape[0], batch_size):
        data_batch = normalized[start : start + batch_size].to(
            device=device, dtype=dtype
        )
        evaluation_point = (
            model_scale * data_batch
            if canonical_family == "rectified_flow"
            else data_batch
        )
        condition = torch.full(
            (data_batch.shape[0],),
            model_scale,
            device=device,
            dtype=dtype,
        )
        if divergence_backend == "exact":
            raw_divergence = exact_divergence(
                model, evaluation_point, condition, create_graph=False
            )
        else:
            assert generator is not None
            raw_divergence = hutchinson_divergence(
                model,
                evaluation_point,
                condition,
                num_probes=trace_probes,
                seed=None,
                generator=generator,
                create_graph=False,
            )
        with torch.no_grad():
            raw_field = model(evaluation_point, condition)
        if canonical_family == "gaussian_diffusion":
            variance = model_scale**2
            field = (raw_field - evaluation_point) / variance
            divergence = (raw_divergence - model.config.ambient_dim) / variance
        elif canonical_family == "rectified_flow":
            field = raw_field
            divergence = raw_divergence
        else:
            field, divergence = denoiser_to_forward_drift(
                raw_field,
                raw_divergence,
                evaluation_point,
                tau=model_scale,
                ambient_dim=model.config.ambient_dim,
            )
        fields.append(field.detach().cpu())
        divergences.append(divergence.detach().cpu())
        points.append(evaluation_point.detach().cpu())
    return FieldPrediction(
        field=np.asarray(torch.cat(fields).numpy(), dtype=np.float64),
        divergence=np.asarray(torch.cat(divergences).numpy(), dtype=np.float64),
        evaluation_point=np.asarray(torch.cat(points).numpy(), dtype=np.float64),
        condition=float(model_scale),
    )


def predict_nf_log_likelihood(
    model_or_result: nn.Module | TrainingResult,
    query: Any,
    epsilon: float,
    *,
    family: str | None = None,
    batch_size: int = 256,
) -> npt.NDArray[np.float64]:
    """Evaluate the exact normalized NF log density at one declared epsilon."""

    model, canonical_family, mean, normalization_scale = _model_and_context(
        model_or_result, family
    )
    if canonical_family != "scale_conditioned_normalizing_flow" or not isinstance(
        model, ScaleConditionedRealNVP
    ):
        raise TypeError("predict_nf_log_likelihood requires ScaleConditionedRealNVP")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise TypeError("NF epsilon must be numeric")
    scale = float(epsilon)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("NF epsilon must be finite and positive")
    if isinstance(model_or_result, TrainingResult):
        epsilon_min = model_or_result.config.epsilon_min
        epsilon_max = model_or_result.config.epsilon_max
        if epsilon_min is None or epsilon_max is None:
            raise ValueError("checkpoint lacks the NF epsilon training interval")
        if not float(epsilon_min) <= scale <= float(epsilon_max):
            raise ValueError(
                "NF epsilon lies outside the checkpointed training interval"
            )
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer")
    query_cpu = _flat_finite_data(query, name="query")
    if query_cpu.shape[1] != model.config.ambient_dim:
        raise ValueError("query ambient dimension does not match model")
    normalized = (query_cpu - mean.reshape(1, -1)) / normalization_scale
    parameter = next(model.parameters())
    model.eval()
    likelihoods: list[Tensor] = []
    for start in range(0, normalized.shape[0], batch_size):
        batch = normalized[start : start + batch_size].to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        curve = fixed_point_log_likelihood_curve(model, batch, (scale,))
        likelihoods.append(curve[:, 0].detach().cpu())
    result = np.ascontiguousarray(
        torch.cat(likelihoods, dim=0).numpy(),
        dtype=np.float64,
    )
    if result.shape != (query_cpu.shape[0],) or not np.isfinite(result).all():
        raise FloatingPointError("NF log-likelihood inference produced invalid output")
    return result


def predict_nf_lid_ols5(
    model_or_result: nn.Module | TrainingResult,
    query: Any,
    epsilon: float,
    *,
    family: str | None = None,
    ols_log_step: float = 0.05,
    batch_size: int = 256,
) -> npt.NDArray[np.float64]:
    """Evaluate only the five-point local-OLS NF LID readout."""

    model, canonical_family, mean, normalization_scale = _model_and_context(
        model_or_result, family
    )
    if canonical_family != "scale_conditioned_normalizing_flow" or not isinstance(
        model, ScaleConditionedRealNVP
    ):
        raise TypeError("predict_nf_lid_ols5 requires ScaleConditionedRealNVP")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise TypeError("NF epsilon must be numeric")
    center = float(epsilon)
    if not math.isfinite(center) or center <= 0.0:
        raise ValueError("NF epsilon must be finite and positive")
    if isinstance(ols_log_step, bool) or not isinstance(ols_log_step, (int, float)):
        raise TypeError("ols_log_step must be numeric")
    step = float(ols_log_step)
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("ols_log_step must be finite and positive")
    if isinstance(model_or_result, TrainingResult):
        epsilon_min = model_or_result.config.epsilon_min
        epsilon_max = model_or_result.config.epsilon_max
        if epsilon_min is None or epsilon_max is None:
            raise ValueError("checkpoint lacks the NF epsilon training interval")
        lower = center * math.exp(-2.0 * step)
        upper = center * math.exp(2.0 * step)
        tolerance = 32.0 * torch.finfo(torch.float64).eps * max(1.0, upper)
        if (
            lower < float(epsilon_min) - tolerance
            or upper > float(epsilon_max) + tolerance
        ):
            raise ValueError(
                "NF OLS5 window lies outside the checkpointed training interval"
            )
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer")
    query_cpu = _flat_finite_data(query, name="query")
    if query_cpu.shape[1] != model.config.ambient_dim:
        raise ValueError("query ambient dimension does not match model")
    normalized = (query_cpu - mean.reshape(1, -1)) / normalization_scale
    parameter = next(model.parameters())
    model.eval()
    predictions: list[Tensor] = []
    for start in range(0, normalized.shape[0], batch_size):
        batch = normalized[start : start + batch_size].to(
            device=parameter.device, dtype=parameter.dtype
        )
        predictions.append(
            fixed_point_lid_local_ols(
                model, batch, center, log_step=step, window_size=5
            )
            .detach()
            .cpu()
        )
    result = np.ascontiguousarray(
        torch.cat(predictions, dim=0).numpy(), dtype=np.float64
    )
    if result.shape != (query_cpu.shape[0],) or not np.isfinite(result).all():
        raise FloatingPointError("NF OLS5 inference produced invalid output")
    return result


def predict_nf_readouts(
    model_or_result: nn.Module | TrainingResult,
    query: Any,
    epsilon: float,
    *,
    family: str | None = None,
    finite_difference_log_step: float = 0.01,
    ols_log_step: float = 0.05,
    batch_size: int = 256,
) -> NFLikelihoodReadoutPrediction:
    """Evaluate every NF likelihood readout and its likelihood-path evidence.

    Symmetric windows are required to stay inside the checkpointed training
    interval.  The function never clips or substitutes one-sided windows,
    because doing so would silently change the estimator near a boundary.
    """

    model, canonical_family, mean, normalization_scale = _model_and_context(
        model_or_result, family
    )
    if canonical_family != "scale_conditioned_normalizing_flow" or not isinstance(
        model, ScaleConditionedRealNVP
    ):
        raise TypeError("predict_nf_readouts requires ScaleConditionedRealNVP")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise TypeError("NF epsilon must be numeric")
    center = float(epsilon)
    if not math.isfinite(center) or center <= 0.0:
        raise ValueError("NF epsilon must be finite and positive")
    for name, value in (
        ("finite_difference_log_step", finite_difference_log_step),
        ("ols_log_step", ols_log_step),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    fd_step = float(finite_difference_log_step)
    regression_step = float(ols_log_step)
    if isinstance(model_or_result, TrainingResult):
        epsilon_min = model_or_result.config.epsilon_min
        epsilon_max = model_or_result.config.epsilon_max
        if epsilon_min is None or epsilon_max is None:
            raise ValueError("checkpoint lacks the NF epsilon training interval")
        largest_log_radius = max(fd_step, 4.0 * regression_step)
        lower_log_epsilon = math.log(center) - largest_log_radius
        upper_log_epsilon = math.log(center) + largest_log_radius
        if lower_log_epsilon < math.log(
            float(epsilon_min)
        ) or upper_log_epsilon > math.log(float(epsilon_max)):
            raise ValueError(
                "NF readout window lies outside the checkpointed training interval"
            )
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer")
    query_cpu = _flat_finite_data(query, name="query")
    if query_cpu.shape[1] != model.config.ambient_dim:
        raise ValueError("query ambient dimension does not match model")
    normalized = (query_cpu - mean.reshape(1, -1)) / normalization_scale
    parameter = next(model.parameters())
    model.eval()

    finite_difference_epsilons: Tensor | None = None
    ols_epsilons: Tensor | None = None
    finite_difference_curves: list[Tensor] = []
    ols_curves: list[Tensor] = []
    readout_batches: dict[str, list[Tensor]] = {
        "autograd": [],
        "symmetric_fd": [],
        "ols3": [],
        "ols5": [],
        "ols9": [],
    }
    for start in range(0, normalized.shape[0], batch_size):
        batch = normalized[start : start + batch_size].to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        prediction = fixed_point_likelihood_readouts(
            model,
            batch,
            center,
            finite_difference_log_step=fd_step,
            ols_log_step=regression_step,
        )
        batch_fd_epsilons = prediction.finite_difference_epsilons.detach().cpu()
        batch_ols_epsilons = prediction.ols_epsilons.detach().cpu()
        if finite_difference_epsilons is None:
            finite_difference_epsilons = batch_fd_epsilons
            ols_epsilons = batch_ols_epsilons
        elif not torch.equal(
            finite_difference_epsilons, batch_fd_epsilons
        ) or not torch.equal(ols_epsilons, batch_ols_epsilons):
            raise RuntimeError("NF likelihood grids changed between inference batches")
        finite_difference_curves.append(
            prediction.finite_difference_log_likelihood.detach().cpu()
        )
        ols_curves.append(prediction.ols_log_likelihood.detach().cpu())
        readout_batches["autograd"].append(prediction.lid_autograd.detach().cpu())
        readout_batches["symmetric_fd"].append(
            prediction.lid_symmetric_fd.detach().cpu()
        )
        readout_batches["ols3"].append(prediction.lid_ols3.detach().cpu())
        readout_batches["ols5"].append(prediction.lid_ols5.detach().cpu())
        readout_batches["ols9"].append(prediction.lid_ols9.detach().cpu())

    assert finite_difference_epsilons is not None and ols_epsilons is not None

    def array(value: Tensor) -> npt.NDArray[np.float64]:
        result = np.ascontiguousarray(value.numpy(), dtype=np.float64)
        if not np.isfinite(result).all():
            raise FloatingPointError("NF readout inference produced non-finite values")
        return result

    return NFLikelihoodReadoutPrediction(
        epsilon=center,
        finite_difference_log_step=fd_step,
        ols_log_step=regression_step,
        finite_difference_epsilons=array(finite_difference_epsilons),
        finite_difference_log_likelihood=array(
            torch.cat(finite_difference_curves, dim=0)
        ),
        ols_epsilons=array(ols_epsilons),
        ols_log_likelihood=array(torch.cat(ols_curves, dim=0)),
        lid_autograd=array(torch.cat(readout_batches["autograd"], dim=0)),
        lid_symmetric_fd=array(torch.cat(readout_batches["symmetric_fd"], dim=0)),
        lid_ols3=array(torch.cat(readout_batches["ols3"], dim=0)),
        lid_ols5=array(torch.cat(readout_batches["ols5"], dim=0)),
        lid_ols9=array(torch.cat(readout_batches["ols9"], dim=0)),
    )


def predict_lid(
    model_or_result: nn.Module | TrainingResult,
    query: Any,
    scale: float,
    *,
    family: str | None = None,
    readout: Literal["full", "response", "fm_to_score", "fixed_likelihood"] = "full",
    divergence_backend: Literal["exact", "hutchinson", "active_exact"] = "hutchinson",
    trace_probes: int = 16,
    trace_seed: int = 0,
    batch_size: int = 256,
) -> npt.NDArray[np.float64]:
    """Return one LID estimate per query using the paper's native readout."""

    model, canonical_family, mean, normalization_scale = _model_and_context(
        model_or_result, family
    )
    if divergence_backend == 'active_exact':
        from models.preconditioned_field import covariance_span_value_and_trace,CovariancePreconditionedField
        if not isinstance(model,CovariancePreconditionedField):
            raise TypeError('active_exact requires the checkpointed covariance-span field')
        allowed={'full','response','fm_to_score'} if canonical_family=='independent_affine_flow' else {'full','response'}
        if canonical_family in {'vp_diffusion','gaussian_diffusion'}:allowed={'full'}
        if readout not in allowed:raise ValueError('unsupported native-family readout')
        if not math.isfinite(scale) or scale<=0 or batch_size<=0:raise ValueError('positive scale and batch_size required')
        noise_ratio=float(scale)
        if canonical_family=='rectified_flow':
            if scale>=1:raise ValueError('rectified-flow native time must be in (0,1)')
            noise_ratio=(1-scale)/scale
        elif canonical_family=='brownian_schrodinger_bridge':
            noise_ratio=math.sqrt(scale*model.training_config.bridge_diffusivity)
        query_cpu=_flat_finite_data(query,name='query')
        if query_cpu.shape[1]!=model.config.ambient_dim:raise ValueError('query ambient dimension differs')
        normalized=(query_cpu-mean.reshape(1,-1))/normalization_scale
        parameter=next(model.parameters());model.eval();predictions=[]
        for start in range(0,len(normalized),batch_size):
            batch=normalized[start:start+batch_size].to(device=parameter.device,dtype=parameter.dtype)
            value,response=covariance_span_value_and_trace(model,batch,noise_ratio)
            value,response=value.double(),response.double()
            full=response+((value-batch.double())/noise_ratio).square().sum(1)
            predictions.append((response if readout=='response' else full).cpu())
        return np.asarray(torch.cat(predictions).numpy(),dtype=np.float64)
    if canonical_family == "scale_conditioned_normalizing_flow":
        if not isinstance(model, ScaleConditionedRealNVP):
            raise TypeError("fixed-likelihood readout requires ScaleConditionedRealNVP")
        if readout != "fixed_likelihood":
            raise ValueError(
                "scale-conditioned NF supports only the fixed_likelihood readout"
            )
        if divergence_backend != "exact":
            raise ValueError(
                "scale-conditioned NF requires exact autograd scale differentiation"
            )
        if trace_probes not in {0, None}:
            raise ValueError("exact NF scale differentiation requires trace_probes: 0")
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("NF epsilon must be finite and positive")
        if isinstance(model_or_result, TrainingResult):
            epsilon_min = model_or_result.config.epsilon_min
            epsilon_max = model_or_result.config.epsilon_max
            if epsilon_min is None or epsilon_max is None:
                raise ValueError("checkpoint lacks the NF epsilon training interval")
            if not float(epsilon_min) <= scale <= float(epsilon_max):
                raise ValueError(
                    "NF epsilon lies outside the checkpointed training interval"
                )
        if isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if isinstance(trace_seed, bool) or trace_seed < 0:
            raise ValueError("trace_seed must be a non-negative integer")
        query_cpu = _flat_finite_data(query, name="query")
        if query_cpu.shape[1] != model.config.ambient_dim:
            raise ValueError("query ambient dimension does not match model")
        normalized = (query_cpu - mean.reshape(1, -1)) / normalization_scale
        parameter = next(model.parameters())
        model.eval()
        predictions: list[Tensor] = []
        for start in range(0, normalized.shape[0], batch_size):
            batch = normalized[start : start + batch_size].to(
                device=parameter.device, dtype=parameter.dtype
            )
            predictions.append(fixed_point_lid(model, batch, scale).detach().cpu())
        return np.asarray(torch.cat(predictions).numpy(), dtype=np.float64)
    if canonical_family == "independent_affine_flow":
        prediction = predict_affine_primitives(
            model_or_result,
            query,
            scale,
            family=canonical_family,
            divergence_backend=divergence_backend,
            trace_probes=trace_probes,
            trace_seed=trace_seed,
            batch_size=batch_size,
        )
        response = prediction.alpha * prediction.posterior_divergence
        if readout == "response":
            return np.asarray(response, dtype=np.float64)
        if readout == "full":
            scaled_bias = (
                prediction.posterior_mean - prediction.channel_point
            ) / prediction.noise_ratio
            return np.asarray(
                response + np.einsum("ij,ij->i", scaled_bias, scaled_bias),
                dtype=np.float64,
            )
        if readout == "fm_to_score":
            return np.asarray(
                model.config.ambient_dim
                + prediction.beta**2
                * (
                    prediction.marginal_score_divergence
                    + np.einsum(
                        "ij,ij->i",
                        prediction.marginal_score,
                        prediction.marginal_score,
                    )
                ),
                dtype=np.float64,
            )
        raise ValueError(
            "independent affine flow readout must be response, full, or fm_to_score"
        )
    prediction = predict_primitives(
        model_or_result,
        query,
        scale,
        family=canonical_family,
        divergence_backend=divergence_backend,
        trace_probes=trace_probes,
        trace_seed=trace_seed,
        batch_size=batch_size,
    )
    ambient_dim = model.config.ambient_dim
    if canonical_family in {"gaussian_diffusion", "vp_diffusion"}:
        if readout != "full":
            raise ValueError("diffusion supports only the full readout")
        return np.asarray(
            diffusion_flipd(
                prediction.field,
                prediction.divergence,
                sigma=prediction.condition,
                ambient_dim=ambient_dim,
            ),
            dtype=np.float64,
        )
    if canonical_family == "brownian_schrodinger_bridge":
        bridge_spec = _bridge_spec_for_model(model_or_result)
        if readout == "response":
            result = sb_forward_response(
                prediction.divergence,
                time_to_go=scale,
                ambient_dim=ambient_dim,
            )
        elif readout == "full":
            result = sb_forward_full(
                prediction.field,
                prediction.divergence,
                time_to_go=scale,
                diffusivity=bridge_spec.diffusivity,
                ambient_dim=ambient_dim,
            )
        else:
            raise ValueError("Schrodinger bridge readout must be full or response")
        return np.asarray(result, dtype=np.float64)
    if readout == "response":
        return np.asarray(
            rectified_flow_response(
                prediction.divergence, t=scale, ambient_dim=ambient_dim
            ),
            dtype=np.float64,
        )
    if readout != "full":
        raise ValueError("readout must be full or response")
    normalized_query = _flat_finite_data(query, name="query")
    if isinstance(model_or_result, TrainingResult):
        normalized_query = (
            normalized_query - model_or_result.normalization_mean.reshape(1, -1)
        ) / model_or_result.normalization_scale
    return np.asarray(
        rectified_flow_full(
            prediction.field,
            prediction.divergence,
            normalized_query.numpy(),
            t=scale,
            ambient_dim=ambient_dim,
        ),
        dtype=np.float64,
    )
