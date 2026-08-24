"""Pure readout formulas from the paper.

This module deliberately knows nothing about neural architectures.  Learned
models (or analytic oracles) supply primitive fields such as a score, velocity
and divergence; the functions below apply the corresponding mathematical
interface.  Keeping this boundary explicit prevents a training implementation
from silently changing the estimator being evaluated.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.floating[Any]]


def _vectors(value: npt.ArrayLike, *, name: str) -> Array:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (n, D); got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _scalars(value: npt.ArrayLike, *, n: int, name: str) -> Array:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(n, float(array))
    else:
        array = np.ravel(array)
    if array.shape != (n,):
        raise ValueError(f"{name} must be scalar or shape ({n},); got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def diffusion_flipd(
    score: npt.ArrayLike,
    score_divergence: npt.ArrayLike,
    *,
    sigma: float,
    ambient_dim: int,
) -> Array:
    """Gaussian diffusion/FLIPD density readout, paper Eq. (33)."""

    score_array = _vectors(score, name="score")
    divergence = _scalars(
        score_divergence, n=score_array.shape[0], name="score_divergence"
    )
    if sigma <= 0 or not np.isfinite(sigma):
        raise ValueError("sigma must be finite and positive")
    if score_array.shape[1] != ambient_dim:
        raise ValueError("score ambient dimension does not match ambient_dim")
    return ambient_dim + sigma**2 * (
        divergence + np.einsum("ij,ij->i", score_array, score_array)
    )


def affine_fm_response(
    velocity_divergence: npt.ArrayLike,
    *,
    ambient_dim: int,
    alpha_log_derivative: float,
    log_noise_ratio_derivative: float,
) -> Array:
    """Native independent-affine flow-matching response, paper Eq. (36)."""

    divergence = np.ravel(np.asarray(velocity_divergence, dtype=np.float64))
    if not np.isfinite(divergence).all():
        raise ValueError("velocity_divergence contains non-finite values")
    kappa = float(log_noise_ratio_derivative)
    if not np.isfinite(kappa) or kappa == 0:
        raise ValueError("log_noise_ratio_derivative must be finite and non-zero")
    return ambient_dim + (
        ambient_dim * float(alpha_log_derivative) - divergence
    ) / kappa


def affine_fm_full(
    velocity: npt.ArrayLike,
    velocity_divergence: npt.ArrayLike,
    score: npt.ArrayLike,
    evaluation_point: npt.ArrayLike,
    *,
    ambient_dim: int,
    alpha_log_derivative: float,
    log_noise_ratio_derivative: float,
) -> Array:
    """Boundary-safe affine-FM density branch, paper Eq. (38)."""

    velocity_array = _vectors(velocity, name="velocity")
    score_array = _vectors(score, name="score")
    point_array = _vectors(evaluation_point, name="evaluation_point")
    if not (velocity_array.shape == score_array.shape == point_array.shape):
        raise ValueError("velocity, score and evaluation_point shapes must match")
    response = affine_fm_response(
        velocity_divergence,
        ambient_dim=ambient_dim,
        alpha_log_derivative=alpha_log_derivative,
        log_noise_ratio_derivative=log_noise_ratio_derivative,
    )
    if response.shape[0] != velocity_array.shape[0]:
        raise ValueError("velocity_divergence batch size does not match fields")
    correction = np.einsum(
        "ij,ij->i",
        float(alpha_log_derivative) * point_array - velocity_array,
        score_array,
    ) / float(log_noise_ratio_derivative)
    return response + correction


def rectified_flow_response(
    velocity_divergence: npt.ArrayLike,
    *,
    t: float,
    ambient_dim: int,
) -> Array:
    """Rectified-flow specialization evaluated at ``y=t*x``, Eq. (37)."""

    if not 0 < t < 1:
        raise ValueError("t must lie strictly between 0 and 1")
    divergence = np.ravel(np.asarray(velocity_divergence, dtype=np.float64))
    return t * (ambient_dim + (1.0 - t) * divergence)


def rectified_flow_full(
    velocity: npt.ArrayLike,
    velocity_divergence: npt.ArrayLike,
    data_point: npt.ArrayLike,
    *,
    t: float,
    ambient_dim: int,
) -> Array:
    """Full Gaussian rectified-flow branch, paper Eq. (52)."""

    velocity_array = _vectors(velocity, name="velocity")
    data_array = _vectors(data_point, name="data_point")
    if velocity_array.shape != data_array.shape:
        raise ValueError("velocity and data_point shapes must match")
    response = rectified_flow_response(
        velocity_divergence, t=t, ambient_dim=ambient_dim
    )
    if response.shape[0] != velocity_array.shape[0]:
        raise ValueError("velocity_divergence batch size does not match velocity")
    return response + t**2 * np.einsum(
        "ij,ij->i", velocity_array - data_array, velocity_array - data_array
    )


def sb_forward_response(
    forward_drift_divergence: npt.ArrayLike,
    *,
    time_to_go: float,
    ambient_dim: int,
) -> Array:
    """Terminal Brownian Schrödinger-bridge response, paper Eq. (42)."""

    if time_to_go <= 0 or not np.isfinite(time_to_go):
        raise ValueError("time_to_go must be finite and positive")
    divergence = np.ravel(np.asarray(forward_drift_divergence, dtype=np.float64))
    return ambient_dim + time_to_go * divergence


def sb_forward_full(
    forward_drift: npt.ArrayLike,
    forward_drift_divergence: npt.ArrayLike,
    *,
    time_to_go: float,
    diffusivity: float,
    ambient_dim: int,
) -> Array:
    """Boundary-safe terminal bridge readout, paper Eq. (43)."""

    drift = _vectors(forward_drift, name="forward_drift")
    if diffusivity <= 0 or not np.isfinite(diffusivity):
        raise ValueError("diffusivity must be finite and positive")
    response = sb_forward_response(
        forward_drift_divergence,
        time_to_go=time_to_go,
        ambient_dim=ambient_dim,
    )
    if response.shape[0] != drift.shape[0]:
        raise ValueError("forward_drift_divergence batch size does not match drift")
    return response + (time_to_go / diffusivity) * np.einsum(
        "ij,ij->i", drift, drift
    )


def sb_current_full(
    current_velocity: npt.ArrayLike,
    current_velocity_divergence: npt.ArrayLike,
    score: npt.ArrayLike,
    *,
    time_to_go: float,
    ambient_dim: int,
) -> Array:
    """Score-flow/current bridge density readout with factor two, Eq. (83)."""

    velocity = _vectors(current_velocity, name="current_velocity")
    score_array = _vectors(score, name="score")
    if velocity.shape != score_array.shape:
        raise ValueError("current_velocity and score shapes must match")
    divergence = _scalars(
        current_velocity_divergence,
        n=velocity.shape[0],
        name="current_velocity_divergence",
    )
    return ambient_dim + 2.0 * time_to_go * (
        divergence + np.einsum("ij,ij->i", velocity, score_array)
    )


def nf_fixed_density(
    scale_velocity: npt.ArrayLike,
    scale_velocity_divergence: npt.ArrayLike,
    score: npt.ArrayLike,
    *,
    ambient_dim: int,
) -> Array:
    """Gauge-invariant fixed-point scale-conditioned NF readout, Eq. (46)."""

    velocity = _vectors(scale_velocity, name="scale_velocity")
    score_array = _vectors(score, name="score")
    if velocity.shape != score_array.shape:
        raise ValueError("scale_velocity and score shapes must match")
    divergence = _scalars(
        scale_velocity_divergence,
        n=velocity.shape[0],
        name="scale_velocity_divergence",
    )
    return ambient_dim - divergence - np.einsum(
        "ij,ij->i", velocity, score_array
    )


def nf_calibrated_native(
    scale_velocity_divergence: npt.ArrayLike, *, ambient_dim: int
) -> Array:
    """Native calibrated singular-flow trace readout, paper Eq. (49)."""

    divergence = np.ravel(np.asarray(scale_velocity_divergence, dtype=np.float64))
    return ambient_dim - divergence


def cnf_calibrated_native(
    velocity_divergence: npt.ArrayLike,
    *,
    log_scale_derivative: float,
    ambient_dim: int,
) -> Array:
    """Time-parameterized calibrated CNF readout, paper Eq. (50)."""

    kappa = float(log_scale_derivative)
    if not np.isfinite(kappa) or kappa == 0:
        raise ValueError("log_scale_derivative must be finite and non-zero")
    divergence = np.ravel(np.asarray(velocity_divergence, dtype=np.float64))
    return ambient_dim - divergence / kappa

