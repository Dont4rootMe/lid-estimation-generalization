"""Independent affine flow-matching schedules and stable field identities.

Every schedule in this module realizes an interpolant

``Y = alpha * X + beta * Z``, ``Z ~ Normal(0, I)``, ``X independent of Z``.

The public scale is always the dimensionless noise ratio
``lambda = beta / alpha``.  Keeping that convention independent of the
schedule makes endpoint grids comparable and prevents a time coordinate from
being silently interpreted as a noise level.  The schedule only determines
the native derivative used by a direct velocity target.

For a posterior-mean model ``q(y, lambda) = E[X | Y=y]``, the population
velocity and Gaussian channel score are recovered algebraically.  In
particular, at ``y=alpha*x`` the native response is the numerically stable
quantity ``alpha * div_y q``; it does not require subtracting two
ambient-dimensional terms near the data endpoint.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
from torch import Tensor

AffineSchedule = Literal["rectified_linear", "log_noise", "vp_trigonometric"]
AffineParameterization = Literal["direct_velocity", "posterior_mean"]
AffineConditioning = Literal["native_time", "noise_ratio", "log_noise_ratio"]
AffineScaleSampling = Literal["log_uniform_noise_ratio", "uniform_noise_ratio"]
AffineLossWeighting = Literal["posterior_bias_equivalent"]

AFFINE_FLOW_CONTRACT_SCHEMA_VERSION = 1

_VARIANT_IDENTITIES: dict[str, tuple[AffineSchedule, AffineParameterization]] = {
    "direct_rectified_flow": ("rectified_linear", "direct_velocity"),
    "posterior_rectified_flow": ("rectified_linear", "posterior_mean"),
    "direct_log_noise_affine_flow": ("log_noise", "direct_velocity"),
    "posterior_log_noise_affine_flow": ("log_noise", "posterior_mean"),
    "direct_vp_trigonometric_flow": (
        "vp_trigonometric",
        "direct_velocity",
    ),
    "posterior_vp_trigonometric_flow": (
        "vp_trigonometric",
        "posterior_mean",
    ),
}


def canonical_schedule(value: str) -> AffineSchedule:
    aliases: dict[str, AffineSchedule] = {
        "rectified": "rectified_linear",
        "rectified_linear": "rectified_linear",
        "log_noise": "log_noise",
        "log_noise_affine": "log_noise",
        "trigonometric": "vp_trigonometric",
        "vp": "vp_trigonometric",
        "vp_trigonometric": "vp_trigonometric",
    }
    try:
        return aliases[str(value)]
    except KeyError as exc:
        raise ValueError(
            "flow_schedule must be rectified_linear, log_noise, or vp_trigonometric"
        ) from exc


def canonical_parameterization(value: str) -> AffineParameterization:
    aliases: dict[str, AffineParameterization] = {
        "velocity": "direct_velocity",
        "direct_velocity": "direct_velocity",
        "x0": "posterior_mean",
        "posterior": "posterior_mean",
        "posterior_mean": "posterior_mean",
    }
    try:
        return aliases[str(value)]
    except KeyError as exc:
        raise ValueError(
            "flow_parameterization must be direct_velocity or posterior_mean"
        ) from exc


@dataclass(frozen=True)
class AffineFlowSpec:
    """Hydra-serializable scientific contract for one affine-FM model."""

    variant_id: str
    schedule: AffineSchedule
    parameterization: AffineParameterization
    conditioning: AffineConditioning
    scale_sampling: AffineScaleSampling
    loss_weighting: AffineLossWeighting
    noise_ratio_min: float
    noise_ratio_max: float

    def __post_init__(self) -> None:
        schedule = canonical_schedule(self.schedule)
        parameterization = canonical_parameterization(self.parameterization)
        try:
            expected = _VARIANT_IDENTITIES[self.variant_id]
        except KeyError as exc:
            raise ValueError(
                "flow_variant_id must be one of: "
                f"{', '.join(sorted(_VARIANT_IDENTITIES))}"
            ) from exc
        if (schedule, parameterization) != expected:
            raise ValueError(
                f"flow_variant_id {self.variant_id!r} requires schedule/"
                f"parameterization {expected!r}; got "
                f"{(schedule, parameterization)!r}"
            )
        if self.conditioning not in {
            "native_time",
            "noise_ratio",
            "log_noise_ratio",
        }:
            raise ValueError(
                "flow_conditioning must be native_time, noise_ratio, or log_noise_ratio"
            )
        if self.conditioning != "log_noise_ratio":
            raise ValueError(
                "campaign affine-flow variants require log_noise_ratio conditioning"
            )
        if self.scale_sampling not in {
            "log_uniform_noise_ratio",
            "uniform_noise_ratio",
        }:
            raise ValueError(
                "flow_scale_sampling must be log_uniform_noise_ratio or "
                "uniform_noise_ratio"
            )
        if self.scale_sampling != "log_uniform_noise_ratio":
            raise ValueError(
                "campaign affine-flow variants require log_uniform_noise_ratio sampling"
            )
        if self.loss_weighting != "posterior_bias_equivalent":
            raise ValueError(
                "campaign affine-flow variants require "
                "flow_loss_weighting=posterior_bias_equivalent"
            )
        bounds = (self.noise_ratio_min, self.noise_ratio_max)
        if any(isinstance(value, bool) for value in bounds):
            raise TypeError("flow noise-ratio bounds must be numeric")
        if not all(math.isfinite(float(value)) for value in bounds):
            raise ValueError("flow noise-ratio bounds must be finite")
        if not 0.0 < float(self.noise_ratio_min) < float(self.noise_ratio_max):
            raise ValueError(
                "flow noise-ratio bounds must satisfy 0 < minimum < maximum"
            )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["schedule"] = canonical_schedule(self.schedule)
        value["parameterization"] = canonical_parameterization(self.parameterization)
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> AffineFlowSpec:
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown affine-flow settings: {sorted(unknown)}")
        payload = dict(value)
        if "schedule" in payload:
            payload["schedule"] = canonical_schedule(payload["schedule"])
        if "parameterization" in payload:
            payload["parameterization"] = canonical_parameterization(
                payload["parameterization"]
            )
        return cls(**payload)


@dataclass(frozen=True)
class AffineScheduleState:
    """Batched schedule values at one or more positive noise ratios."""

    noise_ratio: Tensor
    native_time: Tensor
    alpha: Tensor
    beta: Tensor
    alpha_derivative: Tensor
    beta_derivative: Tensor
    alpha_log_derivative: Tensor
    log_noise_ratio_derivative: Tensor


def _positive_noise_ratio(value: Tensor | float) -> Tensor:
    noise_ratio = torch.as_tensor(value)
    if not torch.is_floating_point(noise_ratio):
        noise_ratio = noise_ratio.float()
    if noise_ratio.ndim > 1:
        raise ValueError("noise_ratio must be scalar or one-dimensional")
    if not torch.isfinite(noise_ratio).all() or torch.any(noise_ratio <= 0):
        raise ValueError("noise_ratio must be finite and strictly positive")
    return noise_ratio


def affine_schedule_state(
    noise_ratio: Tensor | float, schedule: str
) -> AffineScheduleState:
    """Evaluate an affine schedule using ``lambda=beta/alpha`` as input.

    ``native_time`` is ``t`` for the rectified and trigonometric schedules and
    ``u=log(lambda)`` for the log-noise schedule.  All derivatives in the
    returned state are with respect to that native coordinate.
    """

    lam = _positive_noise_ratio(noise_ratio)
    canonical = canonical_schedule(schedule)
    if canonical == "rectified_linear":
        # alpha=t, beta=1-t, lambda=(1-t)/t.
        alpha = torch.reciprocal(1.0 + lam)
        beta = lam * alpha
        native_time = alpha
        alpha_derivative = torch.ones_like(lam)
        beta_derivative = -torch.ones_like(lam)
    elif canonical == "log_noise":
        # R_u=X+exp(u)Z, u=log(lambda).
        alpha = torch.ones_like(lam)
        beta = lam
        native_time = torch.log(lam)
        alpha_derivative = torch.zeros_like(lam)
        beta_derivative = beta
    else:
        # Source-to-data VP path: alpha=sin(pi*t/2), beta=cos(pi*t/2).
        # Therefore lambda=cot(pi*t/2) and t -> 1 is the data endpoint.
        inverse_norm = torch.rsqrt(1.0 + lam.square())
        alpha = inverse_norm
        beta = lam * inverse_norm
        native_time = (2.0 / math.pi) * torch.atan(torch.reciprocal(lam))
        angular_rate = math.pi / 2.0
        alpha_derivative = angular_rate * beta
        beta_derivative = -angular_rate * alpha

    alpha_log_derivative = alpha_derivative / alpha
    log_noise_ratio_derivative = beta_derivative / beta - alpha_log_derivative
    if torch.any(log_noise_ratio_derivative == 0):
        raise FloatingPointError("affine schedule has zero log-noise derivative")
    return AffineScheduleState(
        noise_ratio=lam,
        native_time=native_time,
        alpha=alpha,
        beta=beta,
        alpha_derivative=alpha_derivative,
        beta_derivative=beta_derivative,
        alpha_log_derivative=alpha_log_derivative,
        log_noise_ratio_derivative=log_noise_ratio_derivative,
    )


def schedule_condition(
    state: AffineScheduleState, conditioning: AffineConditioning
) -> Tensor:
    """Return the explicitly selected network-conditioning coordinate."""

    if conditioning == "native_time":
        return state.native_time
    if conditioning == "noise_ratio":
        return state.noise_ratio
    if conditioning == "log_noise_ratio":
        return torch.log(state.noise_ratio)
    raise ValueError(
        "flow_conditioning must be native_time, noise_ratio, or log_noise_ratio"
    )


def sample_noise_ratio(
    batch_size: int,
    *,
    spec: AffineFlowSpec,
    data: Tensor,
    generator: torch.Generator,
) -> Tensor:
    """Sample the Hydra-declared physical noise-ratio distribution."""

    if isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    uniform = torch.rand(
        batch_size,
        device=data.device,
        dtype=data.dtype,
        generator=generator,
    )
    minimum = float(spec.noise_ratio_min)
    maximum = float(spec.noise_ratio_max)
    if spec.scale_sampling == "log_uniform_noise_ratio":
        return torch.exp(
            math.log(minimum) + uniform * (math.log(maximum) - math.log(minimum))
        )
    return minimum + uniform * (maximum - minimum)


def flow_matching_loss_weights(
    state: AffineScheduleState, spec: AffineFlowSpec
) -> Tensor:
    """Return weights for squared error in normalized posterior-bias units.

    For a posterior target, or the log-noise velocity ``w=Y-X``, this is
    ``lambda**-2``.  For a direct velocity in a general schedule,
    ``q-q* = -(v-v*)/(alpha*kappa)`` and therefore the exact weight is
    ``1/(alpha*kappa*lambda)**2``.  In the VP trigonometric schedule this
    simplifies to ``alpha**2/(pi/2)**2``.
    """

    if canonical_parameterization(spec.parameterization) == "posterior_mean":
        weight = torch.reciprocal(state.noise_ratio.square())
    elif spec.loss_weighting == "posterior_bias_equivalent":
        denominator = (
            state.alpha * state.log_noise_ratio_derivative * state.noise_ratio
        ).square()
        weight = torch.reciprocal(denominator)
    else:
        raise AssertionError(spec.loss_weighting)
    if not torch.isfinite(weight).all() or torch.any(weight <= 0):
        raise FloatingPointError("affine-FM loss weights must be finite and positive")
    return weight


def _batch_scalar(value: Tensor, reference: Tensor, *, name: str) -> Tensor:
    if value.ndim == 0:
        value = value.expand(reference.shape[0])
    if value.shape != (reference.shape[0],):
        raise ValueError(f"{name} must be scalar or have shape (batch,)")
    return value.reshape(-1, *([1] * (reference.ndim - 1)))


def affine_interpolant_and_target(
    data: Tensor,
    noise: Tensor,
    state: AffineScheduleState,
    *,
    parameterization: AffineParameterization,
) -> tuple[Tensor, Tensor]:
    """Construct the noisy input and the declared FM regression target."""

    if data.shape != noise.shape or data.ndim < 2:
        raise ValueError("data and noise must have the same batched shape")
    alpha = _batch_scalar(state.alpha, data, name="alpha")
    beta = _batch_scalar(state.beta, data, name="beta")
    interpolated = alpha * data + beta * noise
    canonical = canonical_parameterization(parameterization)
    if canonical == "posterior_mean":
        return interpolated, data
    alpha_derivative = _batch_scalar(
        state.alpha_derivative, data, name="alpha_derivative"
    )
    beta_derivative = _batch_scalar(state.beta_derivative, data, name="beta_derivative")
    return interpolated, alpha_derivative * data + beta_derivative * noise


def posterior_to_velocity(
    posterior_mean: Tensor,
    evaluation_point: Tensor,
    state: AffineScheduleState,
) -> Tensor:
    """Convert ``E[X|Y]`` to velocity for the schedule's native coordinate."""

    if posterior_mean.shape != evaluation_point.shape or posterior_mean.ndim < 2:
        raise ValueError(
            "posterior_mean and evaluation_point must have the same batched shape"
        )
    a_plus_kappa = _batch_scalar(
        state.beta_derivative / state.beta,
        evaluation_point,
        name="beta_log_derivative",
    )
    alpha_kappa = _batch_scalar(
        state.alpha * state.log_noise_ratio_derivative,
        evaluation_point,
        name="alpha_log_noise_derivative",
    )
    return a_plus_kappa * evaluation_point - alpha_kappa * posterior_mean


def posterior_divergence_to_velocity_divergence(
    posterior_divergence: Tensor,
    state: AffineScheduleState,
    *,
    ambient_dim: int,
) -> Tensor:
    """Convert ``div_y E[X|Y]`` to native velocity divergence."""

    if posterior_divergence.ndim == 0:
        posterior_divergence = posterior_divergence.reshape(1)
    if posterior_divergence.ndim != 1:
        raise ValueError("posterior_divergence must be one-dimensional")
    if isinstance(ambient_dim, bool) or ambient_dim <= 0:
        raise ValueError("ambient_dim must be a positive integer")
    beta_log_derivative = state.beta_derivative / state.beta
    alpha_kappa = state.alpha * state.log_noise_ratio_derivative
    return float(ambient_dim) * beta_log_derivative - alpha_kappa * posterior_divergence


def velocity_to_posterior(
    velocity: Tensor,
    evaluation_point: Tensor,
    state: AffineScheduleState,
) -> Tensor:
    """Recover the Gaussian posterior mean from a direct velocity field."""

    if velocity.shape != evaluation_point.shape or velocity.ndim < 2:
        raise ValueError(
            "velocity and evaluation_point must have the same batched shape"
        )
    beta_log_derivative = _batch_scalar(
        state.beta_derivative / state.beta,
        evaluation_point,
        name="beta_log_derivative",
    )
    alpha_kappa = _batch_scalar(
        state.alpha * state.log_noise_ratio_derivative,
        evaluation_point,
        name="alpha_log_noise_derivative",
    )
    return (beta_log_derivative * evaluation_point - velocity) / alpha_kappa


def velocity_divergence_to_posterior_divergence(
    velocity_divergence: Tensor,
    state: AffineScheduleState,
    *,
    ambient_dim: int,
) -> Tensor:
    """Recover ``div_y E[X|Y]`` from direct velocity divergence."""

    if velocity_divergence.ndim == 0:
        velocity_divergence = velocity_divergence.reshape(1)
    if velocity_divergence.ndim != 1:
        raise ValueError("velocity_divergence must be one-dimensional")
    beta_log_derivative = state.beta_derivative / state.beta
    alpha_kappa = state.alpha * state.log_noise_ratio_derivative
    return (
        float(ambient_dim) * beta_log_derivative - velocity_divergence
    ) / alpha_kappa


def posterior_to_channel_score(
    posterior_mean: Tensor,
    channel_point: Tensor,
    state: AffineScheduleState,
) -> Tensor:
    """Return the score of ``X + lambda Z`` at ``channel_point=Y/alpha``."""

    if posterior_mean.shape != channel_point.shape or posterior_mean.ndim < 2:
        raise ValueError(
            "posterior_mean and channel_point must have the same batched shape"
        )
    variance = _batch_scalar(
        state.noise_ratio.square(), channel_point, name="noise_ratio_squared"
    )
    return (posterior_mean - channel_point) / variance


def posterior_to_marginal_score(
    posterior_mean: Tensor,
    evaluation_point: Tensor,
    state: AffineScheduleState,
) -> Tensor:
    """Return ``grad_y log p_Y(y)`` for the Gaussian affine marginal."""

    if posterior_mean.shape != evaluation_point.shape or posterior_mean.ndim < 2:
        raise ValueError(
            "posterior_mean and evaluation_point must have the same batched shape"
        )
    alpha = _batch_scalar(state.alpha, evaluation_point, name="alpha")
    variance = _batch_scalar(state.beta.square(), evaluation_point, name="beta_squared")
    return (alpha * posterior_mean - evaluation_point) / variance


def posterior_divergence_to_channel_score_divergence(
    posterior_divergence: Tensor,
    state: AffineScheduleState,
    *,
    ambient_dim: int,
) -> Tensor:
    """Return divergence of the normalized Gaussian-channel score."""

    if posterior_divergence.ndim == 0:
        posterior_divergence = posterior_divergence.reshape(1)
    if posterior_divergence.ndim != 1:
        raise ValueError("posterior_divergence must be one-dimensional")
    return (
        state.alpha * posterior_divergence - float(ambient_dim)
    ) / state.noise_ratio.square()


def posterior_divergence_to_marginal_score_divergence(
    posterior_divergence: Tensor,
    state: AffineScheduleState,
    *,
    ambient_dim: int,
) -> Tensor:
    """Return ``div_y grad_y log p_Y`` for the Gaussian affine marginal."""

    if posterior_divergence.ndim == 0:
        posterior_divergence = posterior_divergence.reshape(1)
    if posterior_divergence.ndim != 1:
        raise ValueError("posterior_divergence must be one-dimensional")
    return (
        state.alpha * posterior_divergence - float(ambient_dim)
    ) / state.beta.square()


def affine_response_from_velocity_divergence(
    velocity_divergence: Tensor,
    state: AffineScheduleState,
    *,
    ambient_dim: int,
) -> Tensor:
    """Native affine-FM response from a direct velocity divergence."""

    if velocity_divergence.ndim == 0:
        velocity_divergence = velocity_divergence.reshape(1)
    if velocity_divergence.ndim != 1:
        raise ValueError("velocity_divergence must be one-dimensional")
    return (
        float(ambient_dim)
        + (float(ambient_dim) * state.alpha_log_derivative - velocity_divergence)
        / state.log_noise_ratio_derivative
    )


def affine_response_from_posterior_divergence(
    posterior_divergence: Tensor,
    state: AffineScheduleState,
) -> Tensor:
    """Stable native response ``alpha * div_y E[X|Y]``."""

    if posterior_divergence.ndim == 0:
        posterior_divergence = posterior_divergence.reshape(1)
    if posterior_divergence.ndim != 1:
        raise ValueError("posterior_divergence must be one-dimensional")
    return state.alpha * posterior_divergence


def affine_full_from_posterior(
    posterior_mean: Tensor,
    posterior_divergence: Tensor,
    channel_point: Tensor,
    state: AffineScheduleState,
) -> Tensor:
    """Stable Gaussian full-density readout at ``Y=alpha*channel_point``."""

    if posterior_mean.shape != channel_point.shape or posterior_mean.ndim < 2:
        raise ValueError(
            "posterior_mean and channel_point must have the same batched shape"
        )
    response = affine_response_from_posterior_divergence(posterior_divergence, state)
    if response.shape != (channel_point.shape[0],):
        raise ValueError("posterior_divergence batch size does not match fields")
    scaled_bias = (posterior_mean - channel_point) / _batch_scalar(
        state.noise_ratio, channel_point, name="noise_ratio"
    )
    return response + scaled_bias.square().flatten(1).sum(dim=1)


def affine_fm_to_score_full(
    marginal_score: Tensor,
    marginal_score_divergence: Tensor,
    state: AffineScheduleState,
    *,
    ambient_dim: int,
) -> Tensor:
    """Gaussian FM-to-score/FLIPD readout in the affine marginal coordinates."""

    if marginal_score.ndim < 2:
        raise ValueError("marginal_score must have a batch and feature dimensions")
    if marginal_score.reshape(marginal_score.shape[0], -1).shape[1] != ambient_dim:
        raise ValueError("marginal_score ambient dimension does not match ambient_dim")
    if marginal_score_divergence.ndim == 0:
        marginal_score_divergence = marginal_score_divergence.reshape(1)
    if marginal_score_divergence.shape != (marginal_score.shape[0],):
        raise ValueError(
            "marginal_score_divergence batch size does not match marginal_score"
        )
    squared_norm = marginal_score.square().flatten(1).sum(dim=1)
    return float(ambient_dim) + state.beta.square() * (
        marginal_score_divergence + squared_norm
    )


def affine_flow_contract(spec: AffineFlowSpec) -> dict[str, Any]:
    """Return the checkpointed scientific identity of an affine-FM model."""

    schedule = canonical_schedule(spec.schedule)
    if schedule == "log_noise":
        native_coordinate = "u=log(lambda)"
        native_orientation = "data_to_noise_as_u_increases"
        data_endpoint = "u->-infinity;lambda->0"
        schedule_definition = (
            "alpha=1;beta=lambda;native_time=log(lambda);d_alpha/du=0;d_beta/du=beta"
        )
    else:
        native_coordinate = "t"
        native_orientation = "source_to_data_as_t_increases"
        data_endpoint = "t->1;lambda->0"
        schedule_definition = (
            "alpha=t=1/(1+lambda);beta=1-t;d_alpha/dt=1;d_beta/dt=-1"
            if schedule == "rectified_linear"
            else (
                "alpha=sin(pi*t/2)=1/sqrt(1+lambda^2);"
                "beta=cos(pi*t/2)=lambda/sqrt(1+lambda^2);"
                "d_alpha/dt=(pi/2)*beta;d_beta/dt=-(pi/2)*alpha"
            )
        )
    parameterization = canonical_parameterization(spec.parameterization)
    loss_weight_formula = (
        "1/lambda^2"
        if parameterization == "posterior_mean"
        else "1/(alpha*kappa*lambda)^2"
    )
    return {
        "schema_version": AFFINE_FLOW_CONTRACT_SCHEMA_VERSION,
        "family": "independent_affine_flow",
        "source_coupling": "independent_standard_gaussian",
        "interpolant": "Y=alpha(lambda)*X+beta(lambda)*Z",
        "scale_semantics": "noise_ratio_lambda=beta/alpha",
        "variant_id": spec.variant_id,
        "schedule": schedule,
        "native_coordinate": native_coordinate,
        "native_orientation": native_orientation,
        "data_endpoint": data_endpoint,
        "schedule_definition": schedule_definition,
        "parameterization": parameterization,
        "conditioning": spec.conditioning,
        "model_condition": "log(lambda)",
        "neural_condition_transform": "linear",
        "scale_sampling": spec.scale_sampling,
        "loss_weighting": spec.loss_weighting,
        "loss_weight_formula": loss_weight_formula,
        "loss_reduction": "mean(weight*mean_coordinate_squared_error)",
        "noise_ratio_min": float(spec.noise_ratio_min),
        "noise_ratio_max": float(spec.noise_ratio_max),
        "native_response": "alpha*div_y_posterior_mean",
        "network_output": (
            "E[X|Y]"
            if canonical_parameterization(spec.parameterization) == "posterior_mean"
            else "native_schedule_velocity"
        ),
        "gaussian_full": (
            "alpha*div_y_posterior_mean+squared_posterior_bias/noise_ratio_squared"
        ),
        "readouts": ["response", "full", "fm_to_score"],
    }
