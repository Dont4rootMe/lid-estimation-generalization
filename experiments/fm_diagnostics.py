"""Sealed, schedule-invariant diagnostics for independent affine FM models.

The primary benchmark protocol selects a single noise ratio on held-out rows of
the source training split and touches validation/test only after that choice is
frozen.  This module is deliberately restricted to that train-selection split.
It records enough primitive quantities to distinguish a bad velocity field, a
bad trace estimate, an unstable schedule conversion, and finite-sample channel
bias without ever creating retrospective validation or test curves.

All supported schedules describe the same Gaussian channel at matched
``lambda = beta / alpha``.  Consequently the posterior response, the general
native affine response, and the FM-to-score readout must agree up to floating
point error.  The validator below recomputes those identities, the deterministic
subsets, every reported metric, and an empirical Gaussian-channel oracle.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from experiments.run_manifest import canonical_json, sha256_path

FM_DIAGNOSTIC_SCHEMA_VERSION = 2
FM_DIAGNOSTIC_PROTOCOL = "train-selection-independent-affine-fm-debug-v2"
PROBE_KIND = "rademacher"
SUPPORTED_VARIANTS = {
    "direct_rectified_flow": ("rectified", "direct_velocity"),
    "posterior_rectified_flow": ("rectified", "posterior_mean"),
    "direct_log_noise_affine_flow": ("log_noise_affine", "direct_velocity"),
    "posterior_log_noise_affine_flow": ("log_noise_affine", "posterior_mean"),
    "direct_vp_trigonometric_flow": ("vp_trigonometric", "direct_velocity"),
    "posterior_vp_trigonometric_flow": ("vp_trigonometric", "posterior_mean"),
}
_REPORTED_SCHEDULES = {
    "rectified_linear": "rectified",
    "log_noise": "log_noise_affine",
    "vp_trigonometric": "vp_trigonometric",
}

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class AffineSchedulePoint:
    """One interior independent-affine schedule point at a fixed noise ratio."""

    schedule: str
    noise_ratio: float
    native_coordinate: float
    alpha: float
    beta: float
    alpha_derivative: float
    beta_derivative: float
    alpha_log_derivative: float
    log_noise_ratio_derivative: float

    def as_row(self) -> tuple[float, ...]:
        return (
            self.noise_ratio,
            self.native_coordinate,
            self.alpha,
            self.beta,
            self.alpha_derivative,
            self.beta_derivative,
            self.alpha_log_derivative,
            self.log_noise_ratio_derivative,
        )


@dataclass(frozen=True)
class AffinePrimitiveBatch:
    """Primitive predictions required for a semantically auditable readout."""

    velocity: FloatArray
    velocity_divergence: FloatArray
    velocity_divergence_from_posterior: FloatArray
    evaluation_point: FloatArray
    posterior_mean: FloatArray
    posterior_divergence: FloatArray
    score: FloatArray
    score_divergence: FloatArray


class PrimitiveFunction(Protocol):
    def __call__(
        self,
        trained: Any,
        query: npt.ArrayLike,
        scale: float,
        *,
        family: str,
        divergence_backend: str,
        trace_probes: int,
        trace_seed: int,
        batch_size: int,
    ) -> Any: ...


class FMDiagnosticConfigError(ValueError):
    """Raised before inference when the diagnostics contract is incomplete."""


_CONFIG_FIELDS = {
    "schema_version",
    "enabled",
    "source_split",
    "primary_divergence_backend",
    "probe_kind",
    "trace_probes",
    "trace_seed",
    "exact_subset_size",
    "exact_subset_seed",
    "oracle_reference_size",
    "oracle_reference_seed",
    "oracle_chunk_size",
    "batch_size",
}

_SCHEDULE_COLUMNS = (
    "noise_ratio",
    "native_coordinate",
    "alpha",
    "beta",
    "alpha_derivative",
    "beta_derivative",
    "alpha_log_derivative",
    "log_noise_ratio_derivative",
)

_ARRAY_NAMES = {
    "scales",
    "schedule_table",
    "query_model_space",
    "target",
    "oracle_reference_model_space",
    "exact_subset_indices",
    "oracle_reference_indices",
    "posterior_mean",
    "velocity",
    "evaluation_point",
    "velocity_divergence",
    "velocity_divergence_from_posterior",
    "posterior_divergence",
    "reconstructed_posterior_divergence",
    "response_native",
    "response",
    "correction",
    "full",
    "fm_to_score",
    "fm_to_score_ideal",
    "velocity_norm",
    "posterior_residual_sq",
    "score_norm_sq",
    "ideal_score_norm_sq",
    "score_divergence",
    "ideal_score_divergence",
    "score_norm_sq_numeric_residual",
    "score_divergence_numeric_residual",
    "evaluation_point_residual_norm",
    "posterior_velocity_predictor_residual_norm",
    "posterior_velocity_ideal_residual_norm",
    "velocity_posterior_predictor_residual_norm",
    "required_velocity_divergence",
    "endpoint_jacobian_ratio",
    "required_posterior_divergence",
    "posterior_trace_ratio",
    "exact_velocity",
    "exact_velocity_divergence",
    "exact_velocity_divergence_from_posterior",
    "exact_evaluation_point",
    "exact_posterior_mean",
    "exact_posterior_divergence",
    "exact_reconstructed_posterior_divergence",
    "exact_score",
    "exact_score_divergence",
    "exact_ideal_score",
    "exact_ideal_score_divergence",
    "hutch_velocity",
    "hutch_velocity_divergence",
    "hutch_velocity_divergence_from_posterior",
    "hutch_posterior_mean",
    "hutch_posterior_divergence",
    "hutch_reconstructed_posterior_divergence",
    "oracle_posterior_mean",
    "oracle_posterior_divergence",
    "oracle_velocity",
    "oracle_velocity_divergence",
    "oracle_score",
    "oracle_score_divergence",
    "oracle_response",
    "oracle_full",
    "oracle_weight_ess",
    "oracle_max_weight",
}


def _schedule_name(variant_id: str) -> str:
    try:
        return SUPPORTED_VARIANTS[str(variant_id)][0]
    except KeyError as exc:
        raise FMDiagnosticConfigError(
            f"unsupported affine-FM variant_id: {variant_id!r}"
        ) from exc


def affine_schedule_point(schedule: str, noise_ratio: float) -> AffineSchedulePoint:
    """Construct a schedule directly from ``lambda=beta/alpha``.

    This avoids comparing nominal native times that represent different
    Gaussian channels.  Only interior points are accepted.
    """

    lam = float(noise_ratio)
    if not math.isfinite(lam) or lam <= 0.0:
        raise ValueError("noise_ratio must be finite and positive")
    if schedule == "rectified":
        alpha = 1.0 / (1.0 + lam)
        beta = lam * alpha
        return AffineSchedulePoint(
            schedule=schedule,
            noise_ratio=lam,
            native_coordinate=alpha,
            alpha=alpha,
            beta=beta,
            alpha_derivative=1.0,
            beta_derivative=-1.0,
            alpha_log_derivative=1.0 / alpha,
            log_noise_ratio_derivative=-1.0 / (alpha * beta),
        )
    if schedule == "log_noise_affine":
        return AffineSchedulePoint(
            schedule=schedule,
            noise_ratio=lam,
            native_coordinate=math.log(lam),
            alpha=1.0,
            beta=lam,
            alpha_derivative=0.0,
            beta_derivative=lam,
            alpha_log_derivative=0.0,
            log_noise_ratio_derivative=1.0,
        )
    if schedule == "vp_trigonometric":
        alpha = 1.0 / math.sqrt(1.0 + lam * lam)
        beta = lam * alpha
        angular_speed = math.pi / 2.0
        theta = math.atan2(1.0, lam)
        return AffineSchedulePoint(
            schedule=schedule,
            noise_ratio=lam,
            native_coordinate=2.0 * theta / math.pi,
            alpha=alpha,
            beta=beta,
            alpha_derivative=angular_speed * beta,
            beta_derivative=-angular_speed * alpha,
            alpha_log_derivative=angular_speed * beta / alpha,
            log_noise_ratio_derivative=-angular_speed / (alpha * beta),
        )
    raise ValueError(f"unsupported affine schedule: {schedule!r}")


def validate_fm_diagnostic_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an explicit Hydra-owned diagnostic configuration."""

    if not isinstance(config, Mapping):
        raise FMDiagnosticConfigError("diagnostics config must be a mapping")
    value = dict(config)
    missing = _CONFIG_FIELDS - set(value)
    unknown = set(value) - _CONFIG_FIELDS
    if missing or unknown:
        raise FMDiagnosticConfigError(
            f"diagnostics fields mismatch: missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    if value["schema_version"] != 1 or isinstance(value["schema_version"], bool):
        raise FMDiagnosticConfigError("diagnostics.schema_version must be exactly 1")
    if value["enabled"] is not True:
        raise FMDiagnosticConfigError("diagnostics.enabled must be true for a run")
    if value["source_split"] != "train_selection":
        raise FMDiagnosticConfigError(
            "FM diagnostics are restricted to source_split=train_selection"
        )
    if value["primary_divergence_backend"] != "hutchinson":
        raise FMDiagnosticConfigError(
            "primary_divergence_backend must be exactly 'hutchinson'"
        )
    if value["probe_kind"] != PROBE_KIND:
        raise FMDiagnosticConfigError("probe_kind must be exactly 'rademacher'")
    for field in (
        "trace_probes",
        "exact_subset_size",
        "oracle_reference_size",
        "oracle_chunk_size",
        "batch_size",
    ):
        raw = value[field]
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise FMDiagnosticConfigError(f"diagnostics.{field} must be positive")
    for field in ("trace_seed", "exact_subset_seed", "oracle_reference_seed"):
        raw = value[field]
        if isinstance(raw, bool) or not isinstance(raw, int) or not 0 <= raw < 2**64:
            raise FMDiagnosticConfigError(
                f"diagnostics.{field} must be an integer in [0, 2**64)"
            )
    return value


def _trained_compute_dtype(trained: Any) -> str:
    """Recover the field arithmetic dtype without importing torch here."""

    model = getattr(trained, "model", trained)
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        # Injectable test predictors need not be torch modules.  Their numeric
        # contract is the ordinary float64 NumPy path unless declared by a
        # real model parameter.
        return "float64"
    try:
        parameter = next(iter(parameters()))
    except (StopIteration, TypeError):
        return "float64"
    dtype = str(getattr(parameter, "dtype", ""))
    if dtype.endswith("float32"):
        return "float32"
    if dtype.endswith("float64"):
        return "float64"
    raise FMDiagnosticConfigError(
        f"FM diagnostics support float32/float64 model arithmetic; got {dtype!r}"
    )


def _numeric_schedule_scalars(
    point: AffineSchedulePoint, *, compute_dtype: str
) -> dict[str, np.ndarray]:
    """Reproduce schedule and conversion scalars in the model dtype."""

    dtype = np.float32 if compute_dtype == "float32" else np.float64
    lam = np.asarray(point.noise_ratio, dtype=dtype)
    one = np.asarray(1.0, dtype=dtype)
    if point.schedule == "rectified":
        alpha = np.reciprocal(one + lam)
        beta = lam * alpha
        alpha_derivative = np.ones_like(lam)
        beta_derivative = -np.ones_like(lam)
    elif point.schedule == "log_noise_affine":
        alpha = np.ones_like(lam)
        beta = lam
        alpha_derivative = np.zeros_like(lam)
        beta_derivative = beta
    elif point.schedule == "vp_trigonometric":
        alpha = np.reciprocal(np.sqrt(one + np.square(lam)))
        beta = lam * alpha
        angular_rate = np.asarray(math.pi / 2.0, dtype=dtype)
        alpha_derivative = angular_rate * beta
        beta_derivative = -angular_rate * alpha
    else:  # pragma: no cover - AffineSchedulePoint construction is closed.
        raise AssertionError(f"unsupported schedule {point.schedule!r}")
    alpha_log_derivative = alpha_derivative / alpha
    beta_log_derivative = beta_derivative / beta
    log_noise_ratio_derivative = beta_log_derivative - alpha_log_derivative
    return {
        "alpha": alpha,
        "beta": beta,
        "beta_log_derivative": beta_log_derivative,
        "alpha_log_noise_derivative": alpha * log_noise_ratio_derivative,
    }


def _numeric_schedule_alpha_beta(
    point: AffineSchedulePoint, *, compute_dtype: str
) -> tuple[np.ndarray, np.ndarray]:
    state = _numeric_schedule_scalars(point, compute_dtype=compute_dtype)
    return state["alpha"], state["beta"]


def _predictor_field_conversions_from_arrays(
    *,
    velocity: npt.ArrayLike,
    evaluation_point: npt.ArrayLike,
    posterior_mean: npt.ArrayLike,
    point: AffineSchedulePoint,
    compute_dtype: str,
) -> tuple[FloatArray, FloatArray]:
    """Reproduce both velocity/posterior conversions in predictor arithmetic."""

    dtype = np.float32 if compute_dtype == "float32" else np.float64
    state = _numeric_schedule_scalars(point, compute_dtype=compute_dtype)
    numeric_evaluation_point = np.asarray(evaluation_point, dtype=dtype)
    numeric_velocity = np.asarray(velocity, dtype=dtype)
    numeric_posterior = np.asarray(posterior_mean, dtype=dtype)
    beta_log = state["beta_log_derivative"]
    alpha_kappa = state["alpha_log_noise_derivative"]
    posterior_from_velocity = (
        beta_log * numeric_evaluation_point - numeric_velocity
    ) / alpha_kappa
    velocity_from_posterior = (
        beta_log * numeric_evaluation_point - alpha_kappa * numeric_posterior
    )
    return (
        np.asarray(posterior_from_velocity, dtype=np.float64),
        np.asarray(velocity_from_posterior, dtype=np.float64),
    )


def _predictor_field_conversions(
    primitives: AffinePrimitiveBatch,
    point: AffineSchedulePoint,
    *,
    compute_dtype: str,
) -> tuple[FloatArray, FloatArray]:
    return _predictor_field_conversions_from_arrays(
        velocity=primitives.velocity,
        evaluation_point=primitives.evaluation_point,
        posterior_mean=primitives.posterior_mean,
        point=point,
        compute_dtype=compute_dtype,
    )


def _predictor_field_conversion_roundoff_scales(
    primitives: AffinePrimitiveBatch,
    point: AffineSchedulePoint,
    *,
    compute_dtype: str,
) -> tuple[FloatArray, FloatArray]:
    """Return componentwise forward-error scales for the q<->v conversions.

    Both conversions contain a potentially ill-conditioned subtraction.  A
    result-relative tolerance is therefore invalid when the two affine terms
    nearly cancel: the standard floating-point forward-error scale for
    ``a - b`` is ``abs(a) + abs(b)``, not ``abs(a - b)``.  For the posterior
    conversion the numerator scale is additionally divided by
    ``abs(alpha * kappa)`` so it is expressed in posterior-coordinate units.

    The operands are formed in the checkpoint's compute dtype and in the same
    operation order as the predictor.  This bounds CPU/GPU FMA and rounding
    differences without relaxing identities that do not use these affine
    conversions.
    """

    dtype = np.float32 if compute_dtype == "float32" else np.float64
    state = _numeric_schedule_scalars(point, compute_dtype=compute_dtype)
    evaluation_point = np.asarray(primitives.evaluation_point, dtype=dtype)
    velocity = np.asarray(primitives.velocity, dtype=dtype)
    posterior = np.asarray(primitives.posterior_mean, dtype=dtype)
    point_term = state["beta_log_derivative"] * evaluation_point
    posterior_term = state["alpha_log_noise_derivative"] * posterior
    denominator = np.abs(state["alpha_log_noise_derivative"])
    posterior_scale = (np.abs(point_term) + np.abs(velocity)) / denominator
    velocity_scale = np.abs(point_term) + np.abs(posterior_term)
    return (
        np.asarray(posterior_scale, dtype=np.float64),
        np.asarray(velocity_scale, dtype=np.float64),
    )


def _predictor_score_conversion(
    primitives: AffinePrimitiveBatch,
    point: AffineSchedulePoint,
    *,
    ambient_dim: int,
    compute_dtype: str,
) -> tuple[FloatArray, FloatArray]:
    """Recompute the exact operation order used by ``models.training``."""

    dtype = np.float32 if compute_dtype == "float32" else np.float64
    alpha, beta = _numeric_schedule_alpha_beta(point, compute_dtype=compute_dtype)
    posterior = np.asarray(primitives.posterior_mean, dtype=dtype)
    evaluation_point = np.asarray(primitives.evaluation_point, dtype=dtype)
    posterior_divergence = np.asarray(primitives.posterior_divergence, dtype=dtype)
    beta_squared = np.square(beta)
    score = (alpha * posterior - evaluation_point) / beta_squared
    score_divergence = (
        alpha * posterior_divergence - np.asarray(float(ambient_dim), dtype=dtype)
    ) / beta_squared
    return (
        np.asarray(score, dtype=np.float64),
        np.asarray(score_divergence, dtype=np.float64),
    )


def _predictor_score_conversion_roundoff_scales(
    primitives: AffinePrimitiveBatch,
    point: AffineSchedulePoint,
    *,
    ambient_dim: int,
    compute_dtype: str,
) -> tuple[FloatArray, FloatArray]:
    """Return operand-conditioned scales for score and score divergence.

    The Gaussian score conversions divide two cancellation-prone numerators
    by ``beta**2``.  Their componentwise forward-error scales are therefore
    ``(|alpha*q| + |y|) / beta**2`` and
    ``(|alpha*div(q)| + D) / beta**2``.  Using the small post-cancellation
    result as the sole scale is invalid near the data endpoint.
    """

    dtype = np.float32 if compute_dtype == "float32" else np.float64
    alpha, beta = _numeric_schedule_alpha_beta(point, compute_dtype=compute_dtype)
    posterior = np.asarray(primitives.posterior_mean, dtype=dtype)
    evaluation_point = np.asarray(primitives.evaluation_point, dtype=dtype)
    posterior_divergence = np.asarray(primitives.posterior_divergence, dtype=dtype)
    denominator = np.abs(np.square(beta))
    score_scale = (np.abs(alpha * posterior) + np.abs(evaluation_point)) / denominator
    score_divergence_scale = (
        np.abs(alpha * posterior_divergence)
        + np.asarray(float(ambient_dim), dtype=dtype)
    ) / denominator
    return (
        np.asarray(score_scale, dtype=np.float64),
        np.asarray(score_divergence_scale, dtype=np.float64),
    )


def _require_roundoff_consistency(
    actual: FloatArray,
    expected: FloatArray,
    *,
    compute_dtype: str,
    label: str,
    forward_error_scale: FloatArray | None = None,
) -> None:
    """Gate arithmetic identities by a scale-relative floating-point bound."""

    dtype = np.float32 if compute_dtype == "float32" else np.float64
    epsilon = float(np.finfo(dtype).eps)
    scales = [np.ones_like(actual), np.abs(actual), np.abs(expected)]
    if forward_error_scale is not None:
        conditioning_scale = np.asarray(forward_error_scale, dtype=np.float64)
        if conditioning_scale.shape != actual.shape:
            raise ValueError(f"{label} roundoff scale shape mismatch")
        if not np.isfinite(conditioning_scale).all() or np.any(conditioning_scale < 0):
            raise ValueError(f"{label} roundoff scale must be finite and non-negative")
        scales.append(conditioning_scale)
    magnitude = np.maximum.reduce(scales)
    # The conversion has multiply/subtract/square/divide plus schedule
    # evaluation.  Sixty-four unit roundoffs is conservative across CPU/GPU
    # kernels.  For cancellation-prone affine conversions ``magnitude`` also
    # includes their componentwise operand-based forward-error scale.
    allowance = 64.0 * epsilon * magnitude
    error = np.abs(actual - expected)
    if np.all(error <= allowance):
        return
    worst = int(np.argmax(error / allowance))
    raise ValueError(
        f"{label} exceeds the {compute_dtype} roundoff contract: "
        f"error={error.ravel()[worst]:.6g}, "
        f"allowance={allowance.ravel()[worst]:.6g}"
    )


def _matrix(value: npt.ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim < 2 or array.shape[0] <= 0:
        raise ValueError(f"{name} must have shape (nonempty batch, ...)")
    result = np.ascontiguousarray(array.reshape(array.shape[0], -1))
    if result.shape[1] <= 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with nonempty feature dimensions")
    return result


def _vector(value: npt.ArrayLike, *, n: int, name: str) -> FloatArray:
    array = np.ravel(np.asarray(value, dtype=np.float64))
    if array.shape != (n,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape ({n},)")
    return np.ascontiguousarray(array)


def _field(value: npt.ArrayLike, *, shape: tuple[int, int], name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}; got {array.shape}")
    return np.ascontiguousarray(array)


def _attribute(value: Any, *names: str) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    raise ValueError(f"primitive result is missing one of {names!r}")


def _optional_attribute(value: Any, *names: str) -> Any | None:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return None


def _coerce_primitives(value: Any, *, n: int, ambient_dim: int) -> AffinePrimitiveBatch:
    shape = (n, ambient_dim)
    velocity = _field(
        _attribute(value, "velocity", "field"), shape=shape, name="velocity"
    )
    velocity_divergence = _vector(
        _attribute(value, "velocity_divergence", "divergence"),
        n=n,
        name="velocity_divergence",
    )
    velocity_divergence_from_posterior = _vector(
        _attribute(value, "velocity_divergence_from_posterior"),
        n=n,
        name="velocity_divergence_from_posterior",
    )
    evaluation_point = _field(
        _attribute(value, "evaluation_point"), shape=shape, name="evaluation_point"
    )
    posterior_mean = _field(
        _attribute(value, "posterior_mean", "denoiser"),
        shape=shape,
        name="posterior_mean",
    )
    posterior_divergence = _vector(
        _attribute(value, "posterior_divergence", "denoiser_divergence"),
        n=n,
        name="posterior_divergence",
    )
    raw_score = _attribute(value, "score", "gaussian_score", "marginal_score")
    raw_score_divergence = _attribute(
        value,
        "score_divergence",
        "gaussian_score_divergence",
        "marginal_score_divergence",
    )
    score = _field(raw_score, shape=shape, name="score")
    score_divergence = _vector(raw_score_divergence, n=n, name="score_divergence")
    return AffinePrimitiveBatch(
        velocity=velocity,
        velocity_divergence=velocity_divergence,
        velocity_divergence_from_posterior=velocity_divergence_from_posterior,
        evaluation_point=evaluation_point,
        posterior_mean=posterior_mean,
        posterior_divergence=posterior_divergence,
        score=score,
        score_divergence=score_divergence,
    )


def _validate_primitive_contract(
    value: Any,
    *,
    variant_id: str,
    point: AffineSchedulePoint,
    model_query: FloatArray,
    divergence_backend: str,
    trace_probes: int,
    trace_seed: int,
) -> None:
    """Reject a predictor/checkpoint whose scientific identity was relabelled."""

    reported_variant = _optional_attribute(value, "variant_id")
    if reported_variant is None or str(reported_variant) != variant_id:
        raise ValueError(
            f"primitive variant_id mismatch: expected {variant_id!r}, "
            f"got {reported_variant!r}"
        )
    reported_schedule = _optional_attribute(value, "schedule")
    if (
        reported_schedule is None
        or _REPORTED_SCHEDULES.get(str(reported_schedule)) != point.schedule
    ):
        raise ValueError("primitive schedule identity does not match variant_id")
    expected_parameterization = (
        "posterior_mean" if variant_id.startswith("posterior_") else "direct_velocity"
    )
    reported_parameterization = _optional_attribute(value, "parameterization")
    if (
        reported_parameterization is None
        or str(reported_parameterization) != expected_parameterization
    ):
        raise ValueError(
            "primitive parameterization identity does not match variant_id"
        )
    scalar_contract = {
        "noise_ratio": point.noise_ratio,
        "native_time": point.native_coordinate,
        "alpha": point.alpha,
        "beta": point.beta,
        "alpha_derivative": point.alpha_derivative,
        "beta_derivative": point.beta_derivative,
        "alpha_log_derivative": point.alpha_log_derivative,
        "log_noise_ratio_derivative": point.log_noise_ratio_derivative,
    }
    for name, expected in scalar_contract.items():
        actual = _optional_attribute(value, name)
        if actual is None or not math.isclose(
            float(actual), float(expected), rel_tol=1e-11, abs_tol=1e-12
        ):
            raise ValueError(f"primitive schedule scalar {name} is inconsistent")
    channel_point = _optional_attribute(value, "channel_point")
    if channel_point is not None and not np.allclose(
        np.asarray(channel_point, dtype=np.float64),
        model_query,
        rtol=5e-6,
        atol=5e-7,
    ):
        raise ValueError("primitive channel_point is not the model-space query")
    expected_probe_kind = PROBE_KIND if divergence_backend == "hutchinson" else "exact"
    expected_seed: int | None = (
        trace_seed if divergence_backend == "hutchinson" else None
    )
    expected_probes = trace_probes if divergence_backend == "hutchinson" else 0
    metadata_contract = {
        "divergence_backend": divergence_backend,
        "trace_probe_kind": expected_probe_kind,
        "trace_seed": expected_seed,
        "trace_probes": expected_probes,
        "primary_trace_field": "posterior_mean",
        "velocity_divergence_source": (
            "raw_model_trace"
            if expected_parameterization == "direct_velocity"
            else "derived_from_posterior_trace"
        ),
    }
    for name, expected in metadata_contract.items():
        actual = _optional_attribute(value, name)
        if actual != expected:
            raise ValueError(
                f"primitive trace metadata {name} mismatch: "
                f"expected {expected!r}, got {actual!r}"
            )
    expected_shared = (
        divergence_backend == "hutchinson"
        and expected_parameterization == "direct_velocity"
    )
    if (
        _optional_attribute(value, "shared_posterior_velocity_probes")
        is not expected_shared
    ):
        raise ValueError("primitive shared-probe attestation is inconsistent")


def _deterministic_indices(n: int, *, size: int, seed: int) -> npt.NDArray[np.int64]:
    if size <= 0 or size > n:
        raise ValueError(f"subset size must be in [1, {n}]; got {size}")
    indices = np.arange(n, dtype=np.uint64)
    with np.errstate(over="ignore"):
        mixed = indices + np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15)
        mixed = (mixed ^ (mixed >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        mixed = (mixed ^ (mixed >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        mixed ^= mixed >> np.uint64(31)
    order = np.lexsort((indices, mixed))
    return np.sort(order[:size].astype(np.int64, copy=False))


def _distribution(value: npt.ArrayLike) -> dict[str, float | int]:
    array = np.ravel(np.asarray(value, dtype=np.float64))
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("diagnostic distributions must be nonempty and finite")
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "q01": float(np.quantile(array, 0.01)),
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.5)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "q99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def _prediction_metrics(
    prediction: npt.ArrayLike, target: npt.ArrayLike
) -> dict[str, Any]:
    values = np.ravel(np.asarray(prediction, dtype=np.float64))
    truth = np.ravel(np.asarray(target, dtype=np.float64))
    if values.shape != truth.shape or values.size == 0:
        raise ValueError("prediction and target must have equal nonempty shapes")
    if not np.isfinite(values).all() or not np.isfinite(truth).all():
        raise ValueError("diagnostic predictions and targets must be finite")
    error = values - truth
    return {
        "prediction": _distribution(values),
        "target": _distribution(truth),
        "error": _distribution(error),
        "absolute_error": _distribution(np.abs(error)),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "bias": float(error.mean()),
    }


def _paired_metrics(learned: npt.ArrayLike, reference: npt.ArrayLike) -> dict[str, Any]:
    return _prediction_metrics(learned, reference)


def _semantic_json_equal(left: Any, right: Any) -> bool:
    """Compare recomputed JSON while tolerating array-layout reduction roundoff."""

    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if isinstance(left, int) and isinstance(right, int):
            return left == right
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    if isinstance(left, str) or isinstance(right, str) or left is None or right is None:
        return type(left) is type(right) and left == right
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _semantic_json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _semantic_json_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return False


def _field_error_metrics(learned: FloatArray, oracle: FloatArray) -> dict[str, Any]:
    if learned.shape != oracle.shape or learned.ndim != 2:
        raise ValueError("paired fields must have identical rank-2 shapes")
    difference = learned - oracle
    learned_norm = np.linalg.norm(learned, axis=1)
    oracle_norm = np.linalg.norm(oracle, axis=1)
    error_norm = np.linalg.norm(difference, axis=1)
    denominator = learned_norm * oracle_norm
    cosine = np.divide(
        np.einsum("ij,ij->i", learned, oracle),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0.0,
    )
    norm_ratio = np.divide(
        learned_norm,
        oracle_norm,
        out=np.full_like(learned_norm, np.nan),
        where=oracle_norm > 0.0,
    )
    finite_ratio = norm_ratio[np.isfinite(norm_ratio)]
    return {
        "component_mae": float(np.abs(difference).mean()),
        "component_rmse": float(np.sqrt(np.square(difference).mean())),
        "error_norm": _distribution(error_norm),
        "cosine": _distribution(cosine),
        "norm_ratio": (_distribution(finite_ratio) if finite_ratio.size else {"n": 0}),
    }


def _derived_components(
    primitives: AffinePrimitiveBatch,
    query_model_space: FloatArray,
    point: AffineSchedulePoint,
    *,
    compute_dtype: str,
    parameterization: str,
) -> dict[str, FloatArray]:
    ambient_dim = query_model_space.shape[1]
    alpha = point.alpha
    beta = point.beta
    kappa = point.log_noise_ratio_derivative
    alpha_log = point.alpha_log_derivative
    expected_point = alpha * query_model_space
    if not np.allclose(
        primitives.evaluation_point,
        expected_point,
        rtol=5e-6,
        atol=5e-7,
    ):
        raise ValueError("primitive evaluation_point is not exactly alpha*x")
    ideal_reconstructed_q = (
        (alpha_log + kappa) * expected_point - primitives.velocity
    ) / (kappa * alpha)
    reconstructed_q, reconstructed_velocity = _predictor_field_conversions(
        primitives,
        point,
        compute_dtype=compute_dtype,
    )
    q_roundoff_scale, velocity_roundoff_scale = (
        _predictor_field_conversion_roundoff_scales(
            primitives,
            point,
            compute_dtype=compute_dtype,
        )
    )
    if parameterization == "direct_velocity":
        _require_roundoff_consistency(
            primitives.posterior_mean,
            reconstructed_q,
            compute_dtype=compute_dtype,
            label="posterior reconstructed from direct velocity",
            forward_error_scale=q_roundoff_scale,
        )
    elif parameterization == "posterior_mean":
        _require_roundoff_consistency(
            primitives.velocity,
            reconstructed_velocity,
            compute_dtype=compute_dtype,
            label="velocity reconstructed from posterior",
            forward_error_scale=velocity_roundoff_scale,
        )
    else:  # pragma: no cover - variant identity is closed before inference.
        raise AssertionError(f"unsupported parameterization {parameterization!r}")
    reconstructed_q_divergence = (
        (alpha_log + kappa) * ambient_dim - primitives.velocity_divergence
    ) / (kappa * alpha)
    # Evaluate the mathematical channel at the declared y=alpha*x.  The raw
    # predictor point is audited separately because fp32 roundoff must not
    # contaminate schedule-invariant readout identities.
    ideal_score = (
        alpha * (primitives.posterior_mean - query_model_space) / (beta * beta)
    )
    ideal_score_divergence = (alpha * primitives.posterior_divergence - ambient_dim) / (
        beta * beta
    )
    predictor_score, predictor_score_divergence = _predictor_score_conversion(
        primitives,
        point,
        ambient_dim=ambient_dim,
        compute_dtype=compute_dtype,
    )
    score_roundoff_scale, score_divergence_roundoff_scale = (
        _predictor_score_conversion_roundoff_scales(
            primitives,
            point,
            ambient_dim=ambient_dim,
            compute_dtype=compute_dtype,
        )
    )
    _require_roundoff_consistency(
        primitives.score,
        predictor_score,
        compute_dtype=compute_dtype,
        label="reported Gaussian score",
        forward_error_scale=score_roundoff_scale,
    )
    _require_roundoff_consistency(
        primitives.score_divergence,
        predictor_score_divergence,
        compute_dtype=compute_dtype,
        label="reported Gaussian score divergence",
        forward_error_scale=score_divergence_roundoff_scale,
    )
    response_native = (
        ambient_dim + (ambient_dim * alpha_log - primitives.velocity_divergence) / kappa
    )
    response = alpha * primitives.posterior_divergence
    posterior_residual_sq = np.einsum(
        "ij,ij->i",
        query_model_space - primitives.posterior_mean,
        query_model_space - primitives.posterior_mean,
    )
    # Elementwise division before summation is intentional.  It exposes
    # overflow instead of hiding it in a precomputed squared scalar.
    correction = np.square(
        (query_model_space - primitives.posterior_mean) / point.noise_ratio
    ).sum(axis=1)
    full = response + correction
    score_norm_sq = np.einsum("ij,ij->i", primitives.score, primitives.score)
    ideal_score_norm_sq = np.einsum("ij,ij->i", ideal_score, ideal_score)
    fm_to_score = ambient_dim + beta * beta * (
        primitives.score_divergence + score_norm_sq
    )
    fm_to_score_ideal = ambient_dim + beta * beta * (
        ideal_score_divergence + ideal_score_norm_sq
    )
    velocity_norm = np.linalg.norm(primitives.velocity, axis=1)
    evaluation_point_residual_norm = np.linalg.norm(
        primitives.evaluation_point - expected_point, axis=1
    )
    for name, value in {
        "reconstructed_q": reconstructed_q,
        "ideal_reconstructed_q": ideal_reconstructed_q,
        "reconstructed_velocity": reconstructed_velocity,
        "posterior_velocity_predictor_residual_norm": np.linalg.norm(
            primitives.posterior_mean - reconstructed_q, axis=1
        ),
        "posterior_velocity_ideal_residual_norm": np.linalg.norm(
            primitives.posterior_mean - ideal_reconstructed_q, axis=1
        ),
        "velocity_posterior_predictor_residual_norm": np.linalg.norm(
            primitives.velocity - reconstructed_velocity, axis=1
        ),
        "reconstructed_q_divergence": reconstructed_q_divergence,
        "derived_score": primitives.score,
        "derived_score_divergence": primitives.score_divergence,
        "ideal_score": ideal_score,
        "ideal_score_divergence": ideal_score_divergence,
        "predictor_score": predictor_score,
        "predictor_score_divergence": predictor_score_divergence,
        "response_native": response_native,
        "response": response,
        "posterior_residual_sq": posterior_residual_sq,
        "correction": correction,
        "full": full,
        "score_norm_sq": score_norm_sq,
        "ideal_score_norm_sq": ideal_score_norm_sq,
        "score_norm_sq_numeric_residual": score_norm_sq - ideal_score_norm_sq,
        "score_divergence_numeric_residual": (
            primitives.score_divergence - ideal_score_divergence
        ),
        "fm_to_score": fm_to_score,
        "fm_to_score_ideal": fm_to_score_ideal,
        "velocity_norm": velocity_norm,
        "evaluation_point_residual_norm": evaluation_point_residual_norm,
    }.items():
        if not np.isfinite(value).all():
            raise FloatingPointError(f"non-finite FM diagnostic component: {name}")
    return {
        "reconstructed_q": reconstructed_q,
        "ideal_reconstructed_q": ideal_reconstructed_q,
        "reconstructed_velocity": reconstructed_velocity,
        "posterior_velocity_predictor_residual_norm": np.linalg.norm(
            primitives.posterior_mean - reconstructed_q, axis=1
        ),
        "posterior_velocity_ideal_residual_norm": np.linalg.norm(
            primitives.posterior_mean - ideal_reconstructed_q, axis=1
        ),
        "velocity_posterior_predictor_residual_norm": np.linalg.norm(
            primitives.velocity - reconstructed_velocity, axis=1
        ),
        "reconstructed_q_divergence": reconstructed_q_divergence,
        "derived_score": primitives.score,
        "derived_score_divergence": primitives.score_divergence,
        "ideal_score": ideal_score,
        "ideal_score_divergence": ideal_score_divergence,
        "predictor_score": predictor_score,
        "predictor_score_divergence": predictor_score_divergence,
        "response_native": response_native,
        "response": response,
        "posterior_residual_sq": posterior_residual_sq,
        "correction": correction,
        "full": full,
        "score_norm_sq": score_norm_sq,
        "ideal_score_norm_sq": ideal_score_norm_sq,
        "score_norm_sq_numeric_residual": score_norm_sq - ideal_score_norm_sq,
        "score_divergence_numeric_residual": (
            primitives.score_divergence - ideal_score_divergence
        ),
        "fm_to_score": fm_to_score,
        "fm_to_score_ideal": fm_to_score_ideal,
        "velocity_norm": velocity_norm,
        "evaluation_point_residual_norm": evaluation_point_residual_norm,
    }


def _empirical_gaussian_oracle(
    query: FloatArray,
    reference: FloatArray,
    point: AffineSchedulePoint,
    *,
    chunk_size: int,
) -> dict[str, FloatArray]:
    """Exact posterior moments of the finite empirical Gaussian mixture."""

    if query.shape[1] != reference.shape[1]:
        raise ValueError("oracle query/reference ambient dimensions differ")
    n, ambient_dim = query.shape
    posterior_mean = np.empty_like(query)
    posterior_trace_covariance = np.empty(n, dtype=np.float64)
    weight_ess = np.empty(n, dtype=np.float64)
    max_weight = np.empty(n, dtype=np.float64)
    reference_sq = np.einsum("ij,ij->i", reference, reference)
    variance = point.noise_ratio * point.noise_ratio
    for start in range(0, n, chunk_size):
        stop = min(n, start + chunk_size)
        batch = query[start:stop]
        distance_sq = (
            np.einsum("ij,ij->i", batch, batch)[:, None]
            + reference_sq[None, :]
            - 2.0 * batch @ reference.T
        )
        np.maximum(distance_sq, 0.0, out=distance_sq)
        logits = -distance_sq / (2.0 * variance)
        logits -= logits.max(axis=1, keepdims=True)
        weights = np.exp(logits)
        weights /= weights.sum(axis=1, keepdims=True)
        weight_ess[start:stop] = 1.0 / np.square(weights).sum(axis=1)
        max_weight[start:stop] = weights.max(axis=1)
        means = weights @ reference
        second_moment = weights @ reference_sq
        posterior_mean[start:stop] = means
        posterior_trace_covariance[start:stop] = np.maximum(
            second_moment - np.einsum("ij,ij->i", means, means), 0.0
        )
    alpha = point.alpha
    beta = point.beta
    alpha_log = point.alpha_log_derivative
    kappa = point.log_noise_ratio_derivative
    evaluation_point = alpha * query
    posterior_divergence = posterior_trace_covariance / (alpha * variance)
    velocity = (alpha_log + kappa) * evaluation_point - kappa * alpha * posterior_mean
    velocity_divergence = (
        alpha_log + kappa
    ) * ambient_dim - kappa * alpha * posterior_divergence
    score = (alpha * posterior_mean - evaluation_point) / (beta * beta)
    score_divergence = (alpha * posterior_divergence - ambient_dim) / (beta * beta)
    response = alpha * posterior_divergence
    correction = np.square((query - posterior_mean) / point.noise_ratio).sum(axis=1)
    full = response + correction
    return {
        "posterior_mean": posterior_mean,
        "posterior_divergence": posterior_divergence,
        "velocity": velocity,
        "velocity_divergence": velocity_divergence,
        "score": score,
        "score_divergence": score_divergence,
        "response": response,
        "full": full,
        "weight_ess": weight_ess,
        "max_weight": max_weight,
    }


def _array_sha256(value: npt.ArrayLike) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    header = canonical_json(
        {"dtype": array.dtype.str, "shape": [int(size) for size in array.shape]}
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _checkpoint_sha256_from_trained(trained: Any) -> str:
    value = getattr(trained, "checkpoint_sha256", None)
    if not _is_sha256(value):
        raise FMDiagnosticConfigError(
            "trained result must expose a lowercase 64-hex checkpoint_sha256"
        )
    checkpoint_path = getattr(trained, "checkpoint_path", None)
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FMDiagnosticConfigError(
                "trained result checkpoint_path does not identify a file"
            )
        if sha256_path(path) != value:
            raise FMDiagnosticConfigError(
                "trained result checkpoint_sha256 does not match checkpoint_path"
            )
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _save_npy(path: Path, value: npt.ArrayLike) -> None:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"refusing to save invalid numeric array {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    temporary.replace(path)


def _output_inventory(directory: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink is forbidden in FM diagnostics: {path}")
        if not path.is_file() or path.name == "manifest.json":
            continue
        relative = path.relative_to(directory).as_posix()
        records[relative] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_path(path),
        }
    return records


def _stack(columns: Sequence[FloatArray]) -> FloatArray:
    return np.ascontiguousarray(np.stack(columns, axis=1), dtype=np.float64)


def run_fm_diagnostics(
    output_dir: Path,
    *,
    variant_id: str,
    outer_selection_curve_sha256: str,
    trained: Any,
    query: npt.ArrayLike,
    query_model_space: npt.ArrayLike,
    target: npt.ArrayLike,
    oracle_reference_model_space: npt.ArrayLike,
    scales: npt.ArrayLike,
    config: Mapping[str, Any],
    primitive_fn: PrimitiveFunction,
) -> Path:
    """Evaluate and seal exhaustive diagnostics on train-selection rows only."""

    diagnostic_config = validate_fm_diagnostic_config(config)
    if not _is_sha256(outer_selection_curve_sha256):
        raise FMDiagnosticConfigError(
            "outer_selection_curve_sha256 must be lowercase 64-hex"
        )
    schedule_name = _schedule_name(variant_id)
    parameterization = SUPPORTED_VARIANTS[str(variant_id)][1]
    checkpoint_sha256 = _checkpoint_sha256_from_trained(trained)
    compute_dtype = _trained_compute_dtype(trained)
    model_query = _matrix(query_model_space, name="query_model_space")
    raw_query = np.asarray(query)
    if raw_query.shape[0] != model_query.shape[0]:
        raise ValueError("query and query_model_space row counts differ")
    n, ambient_dim = model_query.shape
    truth = _vector(target, n=n, name="target")
    reference_source = _matrix(
        oracle_reference_model_space, name="oracle_reference_model_space"
    )
    if reference_source.shape[1] != ambient_dim:
        raise ValueError("oracle reference ambient dimension differs from query")
    lambda_grid = np.ravel(np.asarray(scales, dtype=np.float64))
    if (
        lambda_grid.size < 2
        or not np.isfinite(lambda_grid).all()
        or np.any(lambda_grid <= 0.0)
        or np.unique(lambda_grid).size != lambda_grid.size
    ):
        raise ValueError("scales must be unique finite positive noise ratios")
    if int(diagnostic_config["exact_subset_size"]) > n:
        raise FMDiagnosticConfigError("exact_subset_size exceeds train-selection rows")
    if int(diagnostic_config["oracle_reference_size"]) > reference_source.shape[0]:
        raise FMDiagnosticConfigError("oracle_reference_size exceeds fit rows")

    exact_indices = _deterministic_indices(
        n,
        size=int(diagnostic_config["exact_subset_size"]),
        seed=int(diagnostic_config["exact_subset_seed"]),
    )
    reference_indices = _deterministic_indices(
        reference_source.shape[0],
        size=int(diagnostic_config["oracle_reference_size"]),
        seed=int(diagnostic_config["oracle_reference_seed"]),
    )
    oracle_reference = np.ascontiguousarray(reference_source[reference_indices])
    exact_raw_query = raw_query[exact_indices]
    exact_model_query = model_query[exact_indices]
    exact_target = truth[exact_indices]
    points = [
        affine_schedule_point(schedule_name, float(scale)) for scale in lambda_grid
    ]

    primary_columns: dict[str, list[FloatArray]] = {
        name: []
        for name in (
            "posterior_mean",
            "velocity",
            "evaluation_point",
            "velocity_divergence",
            "velocity_divergence_from_posterior",
            "posterior_divergence",
            "reconstructed_posterior_divergence",
            "response_native",
            "response",
            "correction",
            "full",
            "fm_to_score",
            "fm_to_score_ideal",
            "velocity_norm",
            "posterior_residual_sq",
            "score_norm_sq",
            "ideal_score_norm_sq",
            "score_divergence",
            "ideal_score_divergence",
            "score_norm_sq_numeric_residual",
            "score_divergence_numeric_residual",
            "evaluation_point_residual_norm",
            "posterior_velocity_predictor_residual_norm",
            "posterior_velocity_ideal_residual_norm",
            "velocity_posterior_predictor_residual_norm",
            "required_velocity_divergence",
            "endpoint_jacobian_ratio",
            "required_posterior_divergence",
            "posterior_trace_ratio",
        )
    }
    subset_columns: dict[str, list[FloatArray]] = {
        name: []
        for name in (
            "exact_velocity",
            "exact_velocity_divergence",
            "exact_velocity_divergence_from_posterior",
            "exact_evaluation_point",
            "exact_posterior_mean",
            "exact_posterior_divergence",
            "exact_reconstructed_posterior_divergence",
            "exact_score",
            "exact_score_divergence",
            "exact_ideal_score",
            "exact_ideal_score_divergence",
            "hutch_velocity",
            "hutch_velocity_divergence",
            "hutch_velocity_divergence_from_posterior",
            "hutch_posterior_mean",
            "hutch_posterior_divergence",
            "hutch_reconstructed_posterior_divergence",
            "oracle_posterior_mean",
            "oracle_posterior_divergence",
            "oracle_velocity",
            "oracle_velocity_divergence",
            "oracle_score",
            "oracle_score_divergence",
            "oracle_response",
            "oracle_full",
            "oracle_weight_ess",
            "oracle_max_weight",
        )
    }
    per_scale: list[dict[str, Any]] = []
    for scale_index, point in enumerate(points):
        call_common = {
            "family": "independent_affine_flow",
            "trace_seed": int(diagnostic_config["trace_seed"]),
            "batch_size": int(diagnostic_config["batch_size"]),
        }
        primary_raw = primitive_fn(
            trained,
            raw_query,
            float(point.noise_ratio),
            divergence_backend="hutchinson",
            trace_probes=int(diagnostic_config["trace_probes"]),
            **call_common,
        )
        _validate_primitive_contract(
            primary_raw,
            variant_id=variant_id,
            point=point,
            model_query=model_query,
            divergence_backend="hutchinson",
            trace_probes=int(diagnostic_config["trace_probes"]),
            trace_seed=int(diagnostic_config["trace_seed"]),
        )
        primary = _coerce_primitives(
            primary_raw,
            n=n,
            ambient_dim=ambient_dim,
        )
        primary_derived = _derived_components(
            primary,
            model_query,
            point,
            compute_dtype=compute_dtype,
            parameterization=parameterization,
        )
        exact_raw = primitive_fn(
            trained,
            exact_raw_query,
            float(point.noise_ratio),
            divergence_backend="exact",
            trace_probes=0,
            **call_common,
        )
        _validate_primitive_contract(
            exact_raw,
            variant_id=variant_id,
            point=point,
            model_query=exact_model_query,
            divergence_backend="exact",
            trace_probes=0,
            trace_seed=int(diagnostic_config["trace_seed"]),
        )
        exact = _coerce_primitives(
            exact_raw,
            n=exact_indices.size,
            ambient_dim=ambient_dim,
        )
        exact_derived = _derived_components(
            exact,
            exact_model_query,
            point,
            compute_dtype=compute_dtype,
            parameterization=parameterization,
        )
        hutch_raw = primitive_fn(
            trained,
            exact_raw_query,
            float(point.noise_ratio),
            divergence_backend="hutchinson",
            trace_probes=int(diagnostic_config["trace_probes"]),
            **call_common,
        )
        _validate_primitive_contract(
            hutch_raw,
            variant_id=variant_id,
            point=point,
            model_query=exact_model_query,
            divergence_backend="hutchinson",
            trace_probes=int(diagnostic_config["trace_probes"]),
            trace_seed=int(diagnostic_config["trace_seed"]),
        )
        hutch = _coerce_primitives(
            hutch_raw,
            n=exact_indices.size,
            ambient_dim=ambient_dim,
        )
        hutch_derived = _derived_components(
            hutch,
            exact_model_query,
            point,
            compute_dtype=compute_dtype,
            parameterization=parameterization,
        )
        oracle = _empirical_gaussian_oracle(
            exact_model_query,
            oracle_reference,
            point,
            chunk_size=int(diagnostic_config["oracle_chunk_size"]),
        )

        required_velocity_divergence = (
            ambient_dim * point.alpha_log_derivative
            - point.log_noise_ratio_derivative * (truth - ambient_dim)
        )
        endpoint_ratio = np.divide(
            primary.velocity_divergence,
            required_velocity_divergence,
            out=np.full_like(required_velocity_divergence, np.nan),
            where=required_velocity_divergence != 0.0,
        )
        if not np.isfinite(endpoint_ratio).all():
            raise FloatingPointError("endpoint required divergence is zero/non-finite")
        required_posterior_divergence = truth / point.alpha
        posterior_trace_ratio = primary.posterior_divergence / (
            required_posterior_divergence
        )

        primary_columns["posterior_mean"].append(primary.posterior_mean)
        primary_columns["velocity"].append(primary.velocity)
        primary_columns["evaluation_point"].append(primary.evaluation_point)
        primary_columns["velocity_divergence"].append(primary.velocity_divergence)
        primary_columns["velocity_divergence_from_posterior"].append(
            primary.velocity_divergence_from_posterior
        )
        primary_columns["posterior_divergence"].append(primary.posterior_divergence)
        primary_columns["reconstructed_posterior_divergence"].append(
            primary_derived["reconstructed_q_divergence"]
        )
        for name in (
            "response_native",
            "response",
            "correction",
            "full",
            "fm_to_score",
            "fm_to_score_ideal",
            "velocity_norm",
            "posterior_residual_sq",
            "score_norm_sq",
            "ideal_score_norm_sq",
            "score_divergence_numeric_residual",
            "score_norm_sq_numeric_residual",
            "evaluation_point_residual_norm",
            "posterior_velocity_predictor_residual_norm",
            "posterior_velocity_ideal_residual_norm",
            "velocity_posterior_predictor_residual_norm",
        ):
            primary_columns[name].append(primary_derived[name])
        primary_columns["score_divergence"].append(primary.score_divergence)
        primary_columns["ideal_score_divergence"].append(
            primary_derived["ideal_score_divergence"]
        )
        primary_columns["required_velocity_divergence"].append(
            required_velocity_divergence
        )
        primary_columns["endpoint_jacobian_ratio"].append(endpoint_ratio)
        primary_columns["required_posterior_divergence"].append(
            required_posterior_divergence
        )
        primary_columns["posterior_trace_ratio"].append(posterior_trace_ratio)

        subset_values = {
            "exact_velocity": exact.velocity,
            "exact_velocity_divergence": exact.velocity_divergence,
            "exact_velocity_divergence_from_posterior": (
                exact.velocity_divergence_from_posterior
            ),
            "exact_evaluation_point": exact.evaluation_point,
            "exact_posterior_mean": exact.posterior_mean,
            "exact_posterior_divergence": exact.posterior_divergence,
            "exact_reconstructed_posterior_divergence": exact_derived[
                "reconstructed_q_divergence"
            ],
            "exact_score": exact_derived["derived_score"],
            "exact_score_divergence": exact_derived["derived_score_divergence"],
            "exact_ideal_score": exact_derived["ideal_score"],
            "exact_ideal_score_divergence": exact_derived["ideal_score_divergence"],
            "hutch_velocity": hutch.velocity,
            "hutch_velocity_divergence": hutch.velocity_divergence,
            "hutch_velocity_divergence_from_posterior": (
                hutch.velocity_divergence_from_posterior
            ),
            "hutch_posterior_mean": hutch.posterior_mean,
            "hutch_posterior_divergence": hutch.posterior_divergence,
            "hutch_reconstructed_posterior_divergence": hutch_derived[
                "reconstructed_q_divergence"
            ],
            "oracle_posterior_mean": oracle["posterior_mean"],
            "oracle_posterior_divergence": oracle["posterior_divergence"],
            "oracle_velocity": oracle["velocity"],
            "oracle_velocity_divergence": oracle["velocity_divergence"],
            "oracle_score": oracle["score"],
            "oracle_score_divergence": oracle["score_divergence"],
            "oracle_response": oracle["response"],
            "oracle_full": oracle["full"],
            "oracle_weight_ess": oracle["weight_ess"],
            "oracle_max_weight": oracle["max_weight"],
        }
        for name, value in subset_values.items():
            subset_columns[name].append(np.asarray(value, dtype=np.float64))

        primary_q_error = primary.posterior_mean[exact_indices] - exact.posterior_mean
        per_scale.append(
            {
                "scale_index": scale_index,
                "noise_ratio": point.noise_ratio,
                "native_coordinate": point.native_coordinate,
                "readouts": {
                    "response": _prediction_metrics(primary_derived["response"], truth),
                    "full": _prediction_metrics(primary_derived["full"], truth),
                    "fm_to_score": _prediction_metrics(
                        primary_derived["fm_to_score"], truth
                    ),
                },
                "components": {
                    "velocity_divergence": _distribution(primary.velocity_divergence),
                    "velocity_divergence_from_posterior": _distribution(
                        primary.velocity_divergence_from_posterior
                    ),
                    "posterior_divergence": _distribution(primary.posterior_divergence),
                    "response": _distribution(primary_derived["response"]),
                    "correction": _distribution(primary_derived["correction"]),
                    "velocity_norm": _distribution(primary_derived["velocity_norm"]),
                    "posterior_residual_sq": _distribution(
                        primary_derived["posterior_residual_sq"]
                    ),
                    "score_norm_sq": _distribution(primary_derived["score_norm_sq"]),
                    "ideal_score_norm_sq": _distribution(
                        primary_derived["ideal_score_norm_sq"]
                    ),
                    "score_divergence": _distribution(primary.score_divergence),
                    "ideal_score_divergence": _distribution(
                        primary_derived["ideal_score_divergence"]
                    ),
                    "evaluation_point_residual_norm": _distribution(
                        primary_derived["evaluation_point_residual_norm"]
                    ),
                },
                "identities": {
                    "native_response_minus_alpha_div_q": _distribution(
                        primary_derived["response_native"] - primary_derived["response"]
                    ),
                    "full_minus_fm_to_score": _distribution(
                        primary_derived["full"] - primary_derived["fm_to_score"]
                    ),
                    "full_minus_fm_to_score_ideal": _distribution(
                        primary_derived["full"] - primary_derived["fm_to_score_ideal"]
                    ),
                    "fm_to_score_minus_ideal": _distribution(
                        primary_derived["fm_to_score"]
                        - primary_derived["fm_to_score_ideal"]
                    ),
                    "score_norm_sq_numeric_residual": _distribution(
                        primary_derived["score_norm_sq_numeric_residual"]
                    ),
                    "score_divergence_numeric_residual": _distribution(
                        primary_derived["score_divergence_numeric_residual"]
                    ),
                    "q_minus_velocity_reconstruction_norm": _distribution(
                        primary_derived["posterior_velocity_predictor_residual_norm"]
                    ),
                    "q_minus_ideal_velocity_reconstruction_norm": _distribution(
                        primary_derived["posterior_velocity_ideal_residual_norm"]
                    ),
                    "velocity_minus_posterior_reconstruction_norm": _distribution(
                        primary_derived["velocity_posterior_predictor_residual_norm"]
                    ),
                    "div_q_minus_velocity_reconstruction": _distribution(
                        primary.posterior_divergence
                        - primary_derived["reconstructed_q_divergence"]
                    ),
                    "raw_velocity_divergence_minus_posterior_reconstruction": (
                        _distribution(
                            primary.velocity_divergence
                            - primary.velocity_divergence_from_posterior
                        )
                    ),
                },
                "endpoint": {
                    "required_velocity_divergence": _distribution(
                        required_velocity_divergence
                    ),
                    "learned_over_required_ratio": _distribution(endpoint_ratio),
                    "required_posterior_divergence": _distribution(
                        required_posterior_divergence
                    ),
                    "posterior_learned_over_required_ratio": _distribution(
                        posterior_trace_ratio
                    ),
                },
                "exact_vs_hutchinson": {
                    "probe_kind": PROBE_KIND,
                    "trace_probes": int(diagnostic_config["trace_probes"]),
                    "posterior_divergence": _paired_metrics(
                        hutch.posterior_divergence, exact.posterior_divergence
                    ),
                    "velocity_divergence": _paired_metrics(
                        hutch.velocity_divergence, exact.velocity_divergence
                    ),
                    "posterior_mean_primary_vs_exact": _distribution(
                        np.linalg.norm(primary_q_error, axis=1)
                    ),
                    "raw_v_vs_reconstructed_q_trace": {
                        "exact": _paired_metrics(
                            exact.velocity_divergence,
                            exact.velocity_divergence_from_posterior,
                        ),
                        "hutchinson": _paired_metrics(
                            hutch.velocity_divergence,
                            hutch.velocity_divergence_from_posterior,
                        ),
                    },
                },
                "empirical_oracle": {
                    "kind": "finite_empirical_reference",
                    "reference_size": int(oracle_reference.shape[0]),
                    "posterior_weight_ess": _distribution(oracle["weight_ess"]),
                    "posterior_max_weight": _distribution(oracle["max_weight"]),
                    "response": _prediction_metrics(oracle["response"], exact_target),
                    "full": _prediction_metrics(oracle["full"], exact_target),
                    "learned_full_vs_oracle": _paired_metrics(
                        exact_derived["full"], oracle["full"]
                    ),
                    "posterior_field_error": _field_error_metrics(
                        exact.posterior_mean, oracle["posterior_mean"]
                    ),
                    "velocity_field_error": _field_error_metrics(
                        exact.velocity, oracle["velocity"]
                    ),
                    "score_field_error": _field_error_metrics(
                        exact_derived["derived_score"], oracle["score"]
                    ),
                    "posterior_divergence_error": _paired_metrics(
                        exact.posterior_divergence, oracle["posterior_divergence"]
                    ),
                    "velocity_divergence_error": _paired_metrics(
                        exact.velocity_divergence, oracle["velocity_divergence"]
                    ),
                },
            }
        )

    arrays: dict[str, np.ndarray] = {
        "scales": lambda_grid,
        "schedule_table": np.asarray([point.as_row() for point in points]),
        "query_model_space": model_query,
        "target": truth,
        "oracle_reference_model_space": oracle_reference,
        "exact_subset_indices": exact_indices,
        "oracle_reference_indices": reference_indices,
    }
    for name, columns in primary_columns.items():
        arrays[name] = _stack(columns)
    for name, columns in subset_columns.items():
        arrays[name] = _stack(columns)
    if set(arrays) != _ARRAY_NAMES:
        raise AssertionError("internal FM diagnostic array schema mismatch")

    directory = Path(output_dir)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"FM diagnostic output must be empty: {directory}")
    array_dir = directory / "arrays"
    array_dir.mkdir(parents=True, exist_ok=True)
    for name, value in arrays.items():
        _save_npy(array_dir / f"{name}.npy", value)
    metadata = {
        "schema_version": FM_DIAGNOSTIC_SCHEMA_VERSION,
        "protocol": FM_DIAGNOSTIC_PROTOCOL,
        "variant_id": str(variant_id),
        "schedule": schedule_name,
        "parameterization": SUPPORTED_VARIANTS[str(variant_id)][1],
        "compute_dtype": compute_dtype,
        "checkpoint_sha256": checkpoint_sha256,
        "outer_selection_curve_sha256": outer_selection_curve_sha256,
        "raw_query_sha256": _array_sha256(raw_query),
        "source_split": "train_selection",
        "ambient_dim": ambient_dim,
        "n_query": n,
        "n_scales": int(lambda_grid.size),
        "n_reference_source": int(reference_source.shape[0]),
        "schedule_columns": list(_SCHEDULE_COLUMNS),
        "config": diagnostic_config,
        "array_sha256": {
            name: _array_sha256(value) for name, value in sorted(arrays.items())
        },
        "attestations": {
            "validation_curves_computed": False,
            "test_curves_computed": False,
            "oracle_uses_train_fit_reference_only": True,
            "oracle_kind": "finite_empirical_reference",
            "exact_hutchinson_probe_kind": PROBE_KIND,
            "matched_scale_coordinate": "noise_ratio_beta_over_alpha",
            "no_clipping": True,
        },
    }
    summary = {
        "schema_version": FM_DIAGNOSTIC_SCHEMA_VERSION,
        "protocol": FM_DIAGNOSTIC_PROTOCOL,
        "variant_id": str(variant_id),
        "schedule": schedule_name,
        "parameterization": SUPPORTED_VARIANTS[str(variant_id)][1],
        "primary_selection_readout": "full",
        "per_scale": per_scale,
    }
    _write_json(directory / "metadata.json", metadata)
    _write_json(directory / "summary.json", summary)
    manifest = {
        "schema_version": FM_DIAGNOSTIC_SCHEMA_VERSION,
        "protocol": FM_DIAGNOSTIC_PROTOCOL,
        "metadata_sha256": sha256_path(directory / "metadata.json"),
        "summary_sha256": sha256_path(directory / "summary.json"),
        "outputs": _output_inventory(directory),
    }
    _write_json(directory / "manifest.json", manifest)
    errors = validate_fm_diagnostics(directory)
    if errors:
        raise RuntimeError(
            "new FM diagnostic artifact failed validation: " + "; ".join(errors)
        )
    return directory


def _load_arrays(directory: Path) -> dict[str, np.ndarray]:
    array_dir = directory / "arrays"
    actual = (
        {path.stem for path in array_dir.glob("*.npy")} if array_dir.is_dir() else set()
    )
    if actual != _ARRAY_NAMES:
        raise ValueError(
            f"diagnostic arrays mismatch: missing={sorted(_ARRAY_NAMES - actual)}, "
            f"unknown={sorted(actual - _ARRAY_NAMES)}"
        )
    arrays: dict[str, np.ndarray] = {}
    for name in sorted(_ARRAY_NAMES):
        arrays[name] = np.load(array_dir / f"{name}.npy", allow_pickle=False)
        if not np.issubdtype(arrays[name].dtype, np.number):
            raise ValueError(f"{name}.npy is not numeric")
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"{name}.npy contains non-finite values")
    return arrays


def _expected_summary(
    *, metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    variant_id = str(metadata["variant_id"])
    schedule_name = _schedule_name(variant_id)
    config = validate_fm_diagnostic_config(metadata["config"])
    scales = np.ravel(np.asarray(arrays["scales"], dtype=np.float64))
    query = np.asarray(arrays["query_model_space"], dtype=np.float64)
    target = np.ravel(np.asarray(arrays["target"], dtype=np.float64))
    exact_indices = np.asarray(arrays["exact_subset_indices"], dtype=np.int64)
    exact_target = target[exact_indices]
    reference = np.asarray(arrays["oracle_reference_model_space"], dtype=np.float64)
    per_scale: list[dict[str, Any]] = []
    for index, scale in enumerate(scales):
        point = affine_schedule_point(schedule_name, float(scale))
        response = arrays["response"][:, index]
        full = arrays["full"][:, index]
        score_readout = arrays["fm_to_score"][:, index]
        ideal_score_readout = arrays["fm_to_score_ideal"][:, index]
        velocity_divergence = arrays["velocity_divergence"][:, index]
        velocity_divergence_from_posterior = arrays[
            "velocity_divergence_from_posterior"
        ][:, index]
        posterior_divergence = arrays["posterior_divergence"][:, index]
        correction = arrays["correction"][:, index]
        velocity_norm = arrays["velocity_norm"][:, index]
        posterior_residual_sq = arrays["posterior_residual_sq"][:, index]
        score_norm_sq = arrays["score_norm_sq"][:, index]
        ideal_score_norm_sq = arrays["ideal_score_norm_sq"][:, index]
        score_divergence = arrays["score_divergence"][:, index]
        ideal_score_divergence = arrays["ideal_score_divergence"][:, index]
        score_norm_sq_numeric_residual = arrays["score_norm_sq_numeric_residual"][
            :, index
        ]
        score_divergence_numeric_residual = arrays["score_divergence_numeric_residual"][
            :, index
        ]
        evaluation_point_residual_norm = arrays["evaluation_point_residual_norm"][
            :, index
        ]
        posterior_velocity_predictor_residual_norm = arrays[
            "posterior_velocity_predictor_residual_norm"
        ][:, index]
        posterior_velocity_ideal_residual_norm = arrays[
            "posterior_velocity_ideal_residual_norm"
        ][:, index]
        velocity_posterior_predictor_residual_norm = arrays[
            "velocity_posterior_predictor_residual_norm"
        ][:, index]
        required = arrays["required_velocity_divergence"][:, index]
        endpoint_ratio = arrays["endpoint_jacobian_ratio"][:, index]
        required_posterior_divergence = arrays["required_posterior_divergence"][
            :, index
        ]
        posterior_trace_ratio = arrays["posterior_trace_ratio"][:, index]
        exact_q = arrays["exact_posterior_mean"][:, index, :]
        exact_v = arrays["exact_velocity"][:, index, :]
        exact_score = arrays["exact_score"][:, index, :]
        exact_q_div = arrays["exact_posterior_divergence"][:, index]
        exact_v_div = arrays["exact_velocity_divergence"][:, index]
        exact_v_div_from_q = arrays["exact_velocity_divergence_from_posterior"][
            :, index
        ]
        hutch_q_div = arrays["hutch_posterior_divergence"][:, index]
        hutch_v_div = arrays["hutch_velocity_divergence"][:, index]
        hutch_v_div_from_q = arrays["hutch_velocity_divergence_from_posterior"][
            :, index
        ]
        oracle = _empirical_gaussian_oracle(
            query[exact_indices],
            reference,
            point,
            chunk_size=int(config["oracle_chunk_size"]),
        )
        per_scale.append(
            {
                "scale_index": index,
                "noise_ratio": point.noise_ratio,
                "native_coordinate": point.native_coordinate,
                "readouts": {
                    "response": _prediction_metrics(response, target),
                    "full": _prediction_metrics(full, target),
                    "fm_to_score": _prediction_metrics(score_readout, target),
                },
                "components": {
                    "velocity_divergence": _distribution(velocity_divergence),
                    "velocity_divergence_from_posterior": _distribution(
                        velocity_divergence_from_posterior
                    ),
                    "posterior_divergence": _distribution(posterior_divergence),
                    "response": _distribution(response),
                    "correction": _distribution(correction),
                    "velocity_norm": _distribution(velocity_norm),
                    "posterior_residual_sq": _distribution(posterior_residual_sq),
                    "score_norm_sq": _distribution(score_norm_sq),
                    "ideal_score_norm_sq": _distribution(ideal_score_norm_sq),
                    "score_divergence": _distribution(score_divergence),
                    "ideal_score_divergence": _distribution(ideal_score_divergence),
                    "evaluation_point_residual_norm": _distribution(
                        evaluation_point_residual_norm
                    ),
                },
                "identities": {
                    "native_response_minus_alpha_div_q": _distribution(
                        arrays["response_native"][:, index] - response
                    ),
                    "full_minus_fm_to_score": _distribution(full - score_readout),
                    "full_minus_fm_to_score_ideal": _distribution(
                        full - ideal_score_readout
                    ),
                    "fm_to_score_minus_ideal": _distribution(
                        score_readout - ideal_score_readout
                    ),
                    "score_norm_sq_numeric_residual": _distribution(
                        score_norm_sq_numeric_residual
                    ),
                    "score_divergence_numeric_residual": _distribution(
                        score_divergence_numeric_residual
                    ),
                    "q_minus_velocity_reconstruction_norm": _distribution(
                        posterior_velocity_predictor_residual_norm
                    ),
                    "q_minus_ideal_velocity_reconstruction_norm": _distribution(
                        posterior_velocity_ideal_residual_norm
                    ),
                    "velocity_minus_posterior_reconstruction_norm": _distribution(
                        velocity_posterior_predictor_residual_norm
                    ),
                    "div_q_minus_velocity_reconstruction": _distribution(
                        posterior_divergence
                        - arrays["reconstructed_posterior_divergence"][:, index]
                    ),
                    "raw_velocity_divergence_minus_posterior_reconstruction": (
                        _distribution(
                            velocity_divergence - velocity_divergence_from_posterior
                        )
                    ),
                },
                "endpoint": {
                    "required_velocity_divergence": _distribution(required),
                    "learned_over_required_ratio": _distribution(endpoint_ratio),
                    "required_posterior_divergence": _distribution(
                        required_posterior_divergence
                    ),
                    "posterior_learned_over_required_ratio": _distribution(
                        posterior_trace_ratio
                    ),
                },
                "exact_vs_hutchinson": {
                    "probe_kind": PROBE_KIND,
                    "trace_probes": int(config["trace_probes"]),
                    "posterior_divergence": _paired_metrics(hutch_q_div, exact_q_div),
                    "velocity_divergence": _paired_metrics(hutch_v_div, exact_v_div),
                    "posterior_mean_primary_vs_exact": _distribution(
                        np.linalg.norm(
                            arrays["posterior_mean"][exact_indices, index, :] - exact_q,
                            axis=1,
                        )
                    ),
                    "raw_v_vs_reconstructed_q_trace": {
                        "exact": _paired_metrics(
                            exact_v_div,
                            exact_v_div_from_q,
                        ),
                        "hutchinson": _paired_metrics(
                            hutch_v_div,
                            hutch_v_div_from_q,
                        ),
                    },
                },
                "empirical_oracle": {
                    "kind": "finite_empirical_reference",
                    "reference_size": int(reference.shape[0]),
                    "posterior_weight_ess": _distribution(oracle["weight_ess"]),
                    "posterior_max_weight": _distribution(oracle["max_weight"]),
                    "response": _prediction_metrics(oracle["response"], exact_target),
                    "full": _prediction_metrics(oracle["full"], exact_target),
                    "learned_full_vs_oracle": _paired_metrics(
                        point.alpha * exact_q_div
                        + np.square(
                            (query[exact_indices] - exact_q) / point.noise_ratio
                        ).sum(axis=1),
                        oracle["full"],
                    ),
                    "posterior_field_error": _field_error_metrics(
                        exact_q, oracle["posterior_mean"]
                    ),
                    "velocity_field_error": _field_error_metrics(
                        exact_v, oracle["velocity"]
                    ),
                    "score_field_error": _field_error_metrics(
                        exact_score, oracle["score"]
                    ),
                    "posterior_divergence_error": _paired_metrics(
                        exact_q_div, oracle["posterior_divergence"]
                    ),
                    "velocity_divergence_error": _paired_metrics(
                        exact_v_div, oracle["velocity_divergence"]
                    ),
                },
            }
        )
    return {
        "schema_version": FM_DIAGNOSTIC_SCHEMA_VERSION,
        "protocol": FM_DIAGNOSTIC_PROTOCOL,
        "variant_id": variant_id,
        "schedule": schedule_name,
        "parameterization": SUPPORTED_VARIANTS[variant_id][1],
        "primary_selection_readout": "full",
        "per_scale": per_scale,
    }


def validate_fm_diagnostics(output_dir: Path) -> list[str]:
    """Strictly validate sealed FM diagnostics, including scientific identities."""

    directory = Path(output_dir)
    errors: list[str] = []
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"invalid FM diagnostic JSON: {exc}"]
    if not all(isinstance(value, dict) for value in (manifest, metadata, summary)):
        return ["FM diagnostic JSON roots must be mappings"]
    expected_manifest_fields = {
        "schema_version",
        "protocol",
        "metadata_sha256",
        "summary_sha256",
        "outputs",
    }
    if set(manifest) != expected_manifest_fields:
        errors.append("FM diagnostic manifest fields mismatch")
    if manifest.get("schema_version") != FM_DIAGNOSTIC_SCHEMA_VERSION:
        errors.append("unsupported FM diagnostic manifest schema")
    if manifest.get("protocol") != FM_DIAGNOSTIC_PROTOCOL:
        errors.append("unsupported FM diagnostic protocol")
    if manifest.get("metadata_sha256") != sha256_path(directory / "metadata.json"):
        errors.append("metadata SHA mismatch")
    if manifest.get("summary_sha256") != sha256_path(directory / "summary.json"):
        errors.append("summary SHA mismatch")
    try:
        actual_inventory = _output_inventory(directory)
    except ValueError as exc:
        errors.append(str(exc))
    else:
        if manifest.get("outputs") != actual_inventory:
            errors.append("FM diagnostic output inventory mismatch")

    expected_metadata_fields = {
        "schema_version",
        "protocol",
        "variant_id",
        "schedule",
        "parameterization",
        "compute_dtype",
        "checkpoint_sha256",
        "outer_selection_curve_sha256",
        "raw_query_sha256",
        "source_split",
        "ambient_dim",
        "n_query",
        "n_scales",
        "n_reference_source",
        "schedule_columns",
        "config",
        "array_sha256",
        "attestations",
    }
    if set(metadata) != expected_metadata_fields:
        errors.append("FM diagnostic metadata fields mismatch")
    if metadata.get("schema_version") != FM_DIAGNOSTIC_SCHEMA_VERSION:
        errors.append("unsupported FM diagnostic metadata schema")
    if metadata.get("protocol") != FM_DIAGNOSTIC_PROTOCOL:
        errors.append("invalid metadata protocol")
    try:
        config = validate_fm_diagnostic_config(metadata.get("config", {}))
    except (TypeError, ValueError) as exc:
        errors.append(f"invalid FM diagnostic config: {exc}")
        config = None
    try:
        schedule_name = _schedule_name(str(metadata.get("variant_id")))
    except ValueError as exc:
        errors.append(str(exc))
        schedule_name = None
    if schedule_name is not None and metadata.get("schedule") != schedule_name:
        errors.append("variant schedule identity mismatch")
    if (
        schedule_name is not None
        and metadata.get("parameterization")
        != SUPPORTED_VARIANTS[str(metadata.get("variant_id"))][1]
    ):
        errors.append("variant parameterization identity mismatch")
    if metadata.get("compute_dtype") not in {"float32", "float64"}:
        errors.append("unsupported FM diagnostic compute_dtype")
    if not _is_sha256(metadata.get("checkpoint_sha256")):
        errors.append("invalid FM diagnostic checkpoint_sha256")
    if not _is_sha256(metadata.get("outer_selection_curve_sha256")):
        errors.append("invalid FM diagnostic outer_selection_curve_sha256")
    if not _is_sha256(metadata.get("raw_query_sha256")):
        errors.append("invalid FM diagnostic raw_query_sha256")
    if metadata.get("source_split") != "train_selection":
        errors.append("diagnostics source split is not train_selection")
    if metadata.get("schedule_columns") != list(_SCHEDULE_COLUMNS):
        errors.append("schedule table column contract mismatch")
    expected_attestations = {
        "validation_curves_computed": False,
        "test_curves_computed": False,
        "oracle_uses_train_fit_reference_only": True,
        "oracle_kind": "finite_empirical_reference",
        "exact_hutchinson_probe_kind": PROBE_KIND,
        "matched_scale_coordinate": "noise_ratio_beta_over_alpha",
        "no_clipping": True,
    }
    if metadata.get("attestations") != expected_attestations:
        errors.append("FM diagnostic attestations mismatch")
    forbidden = [
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and ("validation" in path.name or "test" in path.name)
    ]
    if forbidden:
        errors.append(f"forbidden validation/test diagnostic artifacts: {forbidden}")

    try:
        arrays = _load_arrays(directory)
    except (OSError, TypeError, ValueError) as exc:
        errors.append(f"invalid FM diagnostic arrays: {exc}")
        arrays = None
    if arrays is None:
        return errors
    recorded_hashes = metadata.get("array_sha256")
    expected_hashes = {
        name: _array_sha256(value) for name, value in sorted(arrays.items())
    }
    if recorded_hashes != expected_hashes:
        errors.append("FM diagnostic semantic array hashes mismatch")
    scales = np.ravel(np.asarray(arrays["scales"], dtype=np.float64))
    query = np.asarray(arrays["query_model_space"], dtype=np.float64)
    target = np.ravel(np.asarray(arrays["target"], dtype=np.float64))
    if query.ndim != 2 or target.shape != (query.shape[0],):
        errors.append("query/target shape contract mismatch")
        return errors
    n, ambient_dim = query.shape
    if metadata.get("ambient_dim") != ambient_dim or metadata.get("n_query") != n:
        errors.append("metadata query dimensions mismatch")
    if metadata.get("n_scales") != scales.size:
        errors.append("metadata scale count mismatch")
    if config is not None:
        expected_exact = _deterministic_indices(
            n,
            size=int(config["exact_subset_size"]),
            seed=int(config["exact_subset_seed"]),
        )
        if not np.array_equal(arrays["exact_subset_indices"], expected_exact):
            errors.append("exact/Hutchinson subset indices are not reproducible")
        expected_reference = _deterministic_indices(
            int(metadata.get("n_reference_source", 0)),
            size=int(config["oracle_reference_size"]),
            seed=int(config["oracle_reference_seed"]),
        )
        if not np.array_equal(arrays["oracle_reference_indices"], expected_reference):
            errors.append("oracle reference subset indices are not reproducible")
    if schedule_name is not None:
        expected_schedule = np.asarray(
            [
                affine_schedule_point(schedule_name, float(scale)).as_row()
                for scale in scales
            ]
        )
        if not np.array_equal(arrays["schedule_table"], expected_schedule):
            errors.append("schedule table does not recompute exactly from lambda")

    shape_ns = (n, scales.size)
    for name in (
        "velocity_divergence",
        "velocity_divergence_from_posterior",
        "posterior_divergence",
        "reconstructed_posterior_divergence",
        "response_native",
        "response",
        "correction",
        "full",
        "fm_to_score",
        "fm_to_score_ideal",
        "velocity_norm",
        "posterior_residual_sq",
        "score_norm_sq",
        "ideal_score_norm_sq",
        "score_divergence",
        "ideal_score_divergence",
        "score_norm_sq_numeric_residual",
        "score_divergence_numeric_residual",
        "evaluation_point_residual_norm",
        "posterior_velocity_predictor_residual_norm",
        "posterior_velocity_ideal_residual_norm",
        "velocity_posterior_predictor_residual_norm",
        "required_velocity_divergence",
        "endpoint_jacobian_ratio",
        "required_posterior_divergence",
        "posterior_trace_ratio",
    ):
        if arrays[name].shape != shape_ns:
            errors.append(f"{name}.npy shape mismatch")
    if arrays["posterior_mean"].shape != (n, scales.size, ambient_dim):
        errors.append("posterior_mean.npy shape mismatch")
    if arrays["velocity"].shape != (n, scales.size, ambient_dim):
        errors.append("velocity.npy shape mismatch")
    if arrays["evaluation_point"].shape != (n, scales.size, ambient_dim):
        errors.append("evaluation_point.npy shape mismatch")
    exact_size = int(np.asarray(arrays["exact_subset_indices"]).size)
    for name in (
        "exact_velocity",
        "exact_evaluation_point",
        "exact_posterior_mean",
        "exact_score",
        "exact_ideal_score",
    ):
        if arrays[name].shape != (exact_size, scales.size, ambient_dim):
            errors.append(f"{name}.npy shape mismatch")
    for name in (
        "exact_posterior_divergence",
        "exact_score_divergence",
        "exact_ideal_score_divergence",
    ):
        if arrays[name].shape != (exact_size, scales.size):
            errors.append(f"{name}.npy shape mismatch")
    if errors:
        # Avoid obscure broadcasting errors; inventory/hash checks above still ran.
        return errors

    for index, scale in enumerate(scales):
        point = affine_schedule_point(str(schedule_name), float(scale))
        q = arrays["posterior_mean"][:, index, :]
        velocity = arrays["velocity"][:, index, :]
        evaluation_point = arrays["evaluation_point"][:, index, :]
        q_div = arrays["posterior_divergence"][:, index]
        expected_response = point.alpha * q_div
        expected_correction = np.square((query - q) / point.noise_ratio).sum(axis=1)
        expected_full = expected_response + expected_correction
        ideal_score = (point.alpha * q - point.alpha * query) / (point.beta**2)
        ideal_score_divergence = (point.alpha * q_div - ambient_dim) / (point.beta**2)
        primary_primitives = AffinePrimitiveBatch(
            velocity=np.asarray(velocity),
            velocity_divergence=np.asarray(arrays["velocity_divergence"][:, index]),
            velocity_divergence_from_posterior=np.asarray(
                arrays["velocity_divergence_from_posterior"][:, index]
            ),
            evaluation_point=np.asarray(evaluation_point),
            posterior_mean=np.asarray(q),
            posterior_divergence=np.asarray(q_div),
            # The full primary score field is intentionally not persisted;
            # these fields are unused by the q<->v conversion helpers.
            score=np.asarray(ideal_score),
            score_divergence=np.asarray(arrays["score_divergence"][:, index]),
        )
        expected_score_readout = ambient_dim + point.beta**2 * (
            arrays["score_divergence"][:, index] + arrays["score_norm_sq"][:, index]
        )
        expected_ideal_score_readout = ambient_dim + point.beta**2 * (
            ideal_score_divergence + np.einsum("ij,ij->i", ideal_score, ideal_score)
        )
        expected_required = (
            ambient_dim * point.alpha_log_derivative
            - point.log_noise_ratio_derivative * (target - ambient_dim)
        )
        expected_ratio = arrays["velocity_divergence"][:, index] / expected_required
        expected_required_q = target / point.alpha
        expected_q_ratio = q_div / expected_required_q
        ideal_reconstructed_q = (
            (point.alpha_log_derivative + point.log_noise_ratio_derivative)
            * point.alpha
            * query
            - velocity
        ) / (point.log_noise_ratio_derivative * point.alpha)
        compute_dtype = str(metadata.get("compute_dtype"))
        predictor_reconstructed_q, predictor_reconstructed_velocity = (
            _predictor_field_conversions(
                primary_primitives,
                point,
                compute_dtype=compute_dtype,
            )
        )
        q_roundoff_scale, velocity_roundoff_scale = (
            _predictor_field_conversion_roundoff_scales(
                primary_primitives,
                point,
                compute_dtype=compute_dtype,
            )
        )
        _, primary_score_divergence_roundoff_scale = (
            _predictor_score_conversion_roundoff_scales(
                primary_primitives,
                point,
                ambient_dim=ambient_dim,
                compute_dtype=compute_dtype,
            )
        )
        expected_predictor_q_residual = np.linalg.norm(
            q - predictor_reconstructed_q, axis=1
        )
        expected_ideal_q_residual = np.linalg.norm(q - ideal_reconstructed_q, axis=1)
        expected_predictor_velocity_residual = np.linalg.norm(
            velocity - predictor_reconstructed_velocity, axis=1
        )
        reconstructed_q_divergence = (
            (point.alpha_log_derivative + point.log_noise_ratio_derivative)
            * ambient_dim
            - arrays["velocity_divergence"][:, index]
        ) / (point.log_noise_ratio_derivative * point.alpha)
        expected_velocity_divergence_from_q = (
            point.alpha_log_derivative + point.log_noise_ratio_derivative
        ) * ambient_dim - point.log_noise_ratio_derivative * point.alpha * q_div
        exact_indices = np.asarray(arrays["exact_subset_indices"], dtype=np.int64)
        exact_q = arrays["exact_posterior_mean"][:, index, :]
        exact_query = query[exact_indices]
        expected_exact_ideal_score = (
            point.alpha * (exact_q - exact_query) / (point.beta**2)
        )
        expected_exact_ideal_score_divergence = (
            point.alpha * arrays["exact_posterior_divergence"][:, index] - ambient_dim
        ) / (point.beta**2)
        for name, actual, expected in (
            ("response", arrays["response"][:, index], expected_response),
            ("correction", arrays["correction"][:, index], expected_correction),
            ("full", arrays["full"][:, index], expected_full),
            ("fm_to_score", arrays["fm_to_score"][:, index], expected_score_readout),
            (
                "fm_to_score_ideal",
                arrays["fm_to_score_ideal"][:, index],
                expected_ideal_score_readout,
            ),
            (
                "posterior_residual_sq",
                arrays["posterior_residual_sq"][:, index],
                np.square(query - q).sum(axis=1),
            ),
            (
                "ideal_score_norm_sq",
                arrays["ideal_score_norm_sq"][:, index],
                np.square(ideal_score).sum(axis=1),
            ),
            (
                "ideal_score_divergence",
                arrays["ideal_score_divergence"][:, index],
                ideal_score_divergence,
            ),
            (
                "score_norm_sq_numeric_residual",
                arrays["score_norm_sq_numeric_residual"][:, index],
                arrays["score_norm_sq"][:, index]
                - arrays["ideal_score_norm_sq"][:, index],
            ),
            (
                "score_divergence_numeric_residual",
                arrays["score_divergence_numeric_residual"][:, index],
                arrays["score_divergence"][:, index]
                - arrays["ideal_score_divergence"][:, index],
            ),
            (
                "velocity_norm",
                arrays["velocity_norm"][:, index],
                np.linalg.norm(velocity, axis=1),
            ),
            (
                "evaluation_point_residual_norm",
                arrays["evaluation_point_residual_norm"][:, index],
                np.linalg.norm(evaluation_point - point.alpha * query, axis=1),
            ),
            (
                "posterior_velocity_predictor_residual_norm",
                arrays["posterior_velocity_predictor_residual_norm"][:, index],
                expected_predictor_q_residual,
            ),
            (
                "posterior_velocity_ideal_residual_norm",
                arrays["posterior_velocity_ideal_residual_norm"][:, index],
                expected_ideal_q_residual,
            ),
            (
                "velocity_posterior_predictor_residual_norm",
                arrays["velocity_posterior_predictor_residual_norm"][:, index],
                expected_predictor_velocity_residual,
            ),
            (
                "reconstructed_posterior_divergence",
                arrays["reconstructed_posterior_divergence"][:, index],
                reconstructed_q_divergence,
            ),
            (
                "required_velocity_divergence",
                arrays["required_velocity_divergence"][:, index],
                expected_required,
            ),
            (
                "endpoint_jacobian_ratio",
                arrays["endpoint_jacobian_ratio"][:, index],
                expected_ratio,
            ),
            (
                "required_posterior_divergence",
                arrays["required_posterior_divergence"][:, index],
                expected_required_q,
            ),
            (
                "posterior_trace_ratio",
                arrays["posterior_trace_ratio"][:, index],
                expected_q_ratio,
            ),
        ):
            if not np.allclose(actual, expected, rtol=1e-10, atol=1e-10):
                errors.append(f"{name} formula mismatch at scale index {index}")
        dtype = np.float32 if compute_dtype == "float32" else np.float64
        numeric_alpha, numeric_beta = _numeric_schedule_alpha_beta(
            point, compute_dtype=compute_dtype
        )
        numeric_q_div = np.asarray(q_div, dtype=dtype)
        expected_predictor_score_divergence = np.asarray(
            (
                numeric_alpha * numeric_q_div
                - np.asarray(float(ambient_dim), dtype=dtype)
            )
            / np.square(numeric_beta),
            dtype=np.float64,
        )
        try:
            _require_roundoff_consistency(
                np.asarray(arrays["score_divergence"][:, index], dtype=np.float64),
                expected_predictor_score_divergence,
                compute_dtype=compute_dtype,
                label="saved Gaussian score divergence",
                forward_error_scale=primary_score_divergence_roundoff_scale,
            )
        except ValueError as exc:
            errors.append(f"scale index {index}: {exc}")
        if not np.allclose(
            arrays["exact_ideal_score"][:, index, :],
            expected_exact_ideal_score,
            rtol=1e-10,
            atol=1e-10,
        ):
            errors.append(f"exact ideal score mismatch at scale index {index}")
        if not np.allclose(
            arrays["exact_ideal_score_divergence"][:, index],
            expected_exact_ideal_score_divergence,
            rtol=1e-10,
            atol=1e-10,
        ):
            errors.append(
                f"exact ideal score divergence mismatch at scale index {index}"
            )
        exact_primitives = AffinePrimitiveBatch(
            velocity=np.asarray(arrays["exact_velocity"][:, index, :]),
            velocity_divergence=np.asarray(
                arrays["exact_velocity_divergence"][:, index]
            ),
            velocity_divergence_from_posterior=np.asarray(
                arrays["exact_velocity_divergence_from_posterior"][:, index]
            ),
            evaluation_point=np.asarray(arrays["exact_evaluation_point"][:, index, :]),
            posterior_mean=np.asarray(exact_q),
            posterior_divergence=np.asarray(
                arrays["exact_posterior_divergence"][:, index]
            ),
            score=np.asarray(arrays["exact_score"][:, index, :]),
            score_divergence=np.asarray(arrays["exact_score_divergence"][:, index]),
        )
        expected_exact_score, expected_exact_score_divergence = (
            _predictor_score_conversion(
                exact_primitives,
                point,
                ambient_dim=ambient_dim,
                compute_dtype=compute_dtype,
            )
        )
        exact_score_roundoff_scale, exact_score_divergence_roundoff_scale = (
            _predictor_score_conversion_roundoff_scales(
                exact_primitives,
                point,
                ambient_dim=ambient_dim,
                compute_dtype=compute_dtype,
            )
        )
        for label, actual, expected, forward_error_scale in (
            (
                "exact Gaussian score",
                exact_primitives.score,
                expected_exact_score,
                exact_score_roundoff_scale,
            ),
            (
                "exact Gaussian score divergence",
                exact_primitives.score_divergence,
                expected_exact_score_divergence,
                exact_score_divergence_roundoff_scale,
            ),
        ):
            try:
                _require_roundoff_consistency(
                    actual,
                    expected,
                    compute_dtype=compute_dtype,
                    label=label,
                    forward_error_scale=forward_error_scale,
                )
            except ValueError as exc:
                errors.append(f"scale index {index}: {exc}")
        if not np.allclose(
            arrays["velocity_divergence_from_posterior"][:, index],
            expected_velocity_divergence_from_q,
            rtol=5e-6,
            atol=5e-6,
        ):
            errors.append(
                "velocity_divergence_from_posterior formula mismatch at "
                f"scale index {index}"
            )
        parameterization = str(metadata.get("parameterization"))
        if parameterization == "direct_velocity":
            field_identity_actual = q
            field_identity_expected = predictor_reconstructed_q
            field_identity_label = "posterior reconstructed from direct velocity"
            field_identity_roundoff_scale = q_roundoff_scale
        elif parameterization == "posterior_mean":
            field_identity_actual = velocity
            field_identity_expected = predictor_reconstructed_velocity
            field_identity_label = "velocity reconstructed from posterior"
            field_identity_roundoff_scale = velocity_roundoff_scale
        else:  # Reported separately by the metadata contract above.
            field_identity_actual = None
            field_identity_expected = None
            field_identity_label = "invalid parameterization"
            field_identity_roundoff_scale = None
        if field_identity_actual is not None:
            try:
                _require_roundoff_consistency(
                    field_identity_actual,
                    field_identity_expected,
                    compute_dtype=compute_dtype,
                    label=field_identity_label,
                    forward_error_scale=field_identity_roundoff_scale,
                )
            except ValueError as exc:
                errors.append(f"scale index {index}: {exc}")
        native_from_divergence = (
            ambient_dim
            + (
                ambient_dim * point.alpha_log_derivative
                - arrays["velocity_divergence"][:, index]
            )
            / point.log_noise_ratio_derivative
        )
        if not np.allclose(
            arrays["response_native"][:, index],
            native_from_divergence,
            rtol=1e-10,
            atol=1e-10,
        ):
            errors.append(f"native response formula mismatch at scale index {index}")
    if config is not None:
        try:
            expected_summary = _expected_summary(metadata=metadata, arrays=arrays)
        except (FloatingPointError, TypeError, ValueError) as exc:
            errors.append(f"cannot recompute FM diagnostic summary: {exc}")
        else:
            if not _semantic_json_equal(summary, expected_summary):
                errors.append("FM diagnostic summary does not recompute from arrays")
        # Recomputed oracle arrays are checked independently from summary metrics.
        exact_indices = np.asarray(arrays["exact_subset_indices"], dtype=np.int64)
        reference = np.asarray(arrays["oracle_reference_model_space"], dtype=np.float64)
        for index, scale in enumerate(scales):
            point = affine_schedule_point(str(schedule_name), float(scale))
            oracle = _empirical_gaussian_oracle(
                query[exact_indices],
                reference,
                point,
                chunk_size=int(config["oracle_chunk_size"]),
            )
            for suffix, key in (
                ("posterior_mean", "posterior_mean"),
                ("posterior_divergence", "posterior_divergence"),
                ("velocity", "velocity"),
                ("velocity_divergence", "velocity_divergence"),
                ("score", "score"),
                ("score_divergence", "score_divergence"),
                ("response", "response"),
                ("full", "full"),
                ("weight_ess", "weight_ess"),
                ("max_weight", "max_weight"),
            ):
                actual = arrays[f"oracle_{suffix}"][:, index, ...]
                if not np.allclose(actual, oracle[key], rtol=1e-10, atol=1e-10):
                    errors.append(
                        f"empirical oracle {suffix} mismatch at scale index {index}"
                    )
    return errors


__all__ = [
    "FM_DIAGNOSTIC_PROTOCOL",
    "AffinePrimitiveBatch",
    "AffineSchedulePoint",
    "FMDiagnosticConfigError",
    "affine_schedule_point",
    "run_fm_diagnostics",
    "validate_fm_diagnostic_config",
    "validate_fm_diagnostics",
]
