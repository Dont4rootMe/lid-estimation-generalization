from __future__ import annotations

import json
import math

import numpy as np
import pytest

from models.oracle import (
    READOUT_IDS,
    EmpiricalGaussianChannel,
    readout_branch,
    select_stable_scale,
)
from models.readouts import (
    affine_fm_full,
    affine_fm_response,
    cnf_calibrated_native,
    diffusion_flipd,
    nf_calibrated_native,
    nf_fixed_density,
    rectified_flow_full,
    rectified_flow_response,
    sb_current_full,
    sb_forward_full,
    sb_forward_response,
)


def _stable_naive_posterior(
    reference: np.ndarray,
    sample_weight: np.ndarray,
    query: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    squared_distance = np.sum(
        np.square(query[:, None, :] - reference[None, :, :]), axis=2
    )
    normalized_mass = sample_weight / sample_weight.sum()
    log_weight = np.full(squared_distance.shape, -np.inf)
    positive = normalized_mass > 0
    log_weight[:, positive] = (
        np.log(normalized_mass[positive])[None, :]
        - squared_distance[:, positive] / (2.0 * scale**2)
    )
    maximum = np.max(log_weight, axis=1, keepdims=True)
    weight = np.exp(log_weight - maximum)
    weight /= weight.sum(axis=1, keepdims=True)
    mean = weight @ reference
    trace_covariance = np.sum(
        weight * np.sum(np.square(reference[None, :, :] - mean[:, None, :]), axis=2),
        axis=1,
    )
    squared_bias = np.sum(np.square(mean - query), axis=1)
    log_density = (
        maximum[:, 0]
        + np.log(np.exp(log_weight - maximum).sum(axis=1))
        - reference.shape[1] * math.log(scale)
        - 0.5 * reference.shape[1] * math.log(2.0 * math.pi)
    )
    ess = 1.0 / np.sum(np.square(weight), axis=1)
    return mean, trace_covariance, squared_bias, log_density, ess


def test_chunked_posterior_matches_direct_stable_computation() -> None:
    rng = np.random.default_rng(8)
    reference = rng.normal(size=(37, 4)) + 1.0e7
    query = rng.normal(size=(7, 4)) + 1.0e7
    sample_weight = rng.lognormal(size=37)
    sample_weight[[2, 19]] = 0.0
    scale = 0.7

    expected = _stable_naive_posterior(reference, sample_weight, query, scale)
    channel = EmpiricalGaussianChannel(
        reference, sample_weight=sample_weight, reference_chunk_size=5
    )
    moments = channel.posterior(query, scale, query_chunk_size=2)

    np.testing.assert_allclose(moments.mean, expected[0], rtol=1e-11, atol=1e-8)
    np.testing.assert_allclose(
        moments.trace_covariance, expected[1], rtol=2e-8, atol=2e-9
    )
    np.testing.assert_allclose(moments.squared_bias, expected[2], rtol=2e-8)
    np.testing.assert_allclose(
        moments.full_second_moment,
        moments.trace_covariance + moments.squared_bias,
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(moments.log_density, expected[3], rtol=1e-10)
    np.testing.assert_allclose(moments.effective_sample_size, expected[4])


def test_streaming_log_weights_remain_finite_when_raw_weights_underflow() -> None:
    reference = np.array([[-10_000.0], [0.0], [10_000.0]])
    channel = EmpiricalGaussianChannel(reference, reference_chunk_size=1)
    moments = channel.posterior([[5_000.0]], 1.0e-3, query_chunk_size=1)

    # The two nearest atoms have equal weights despite both unnormalised
    # exponentials being exactly zero in float64.
    np.testing.assert_allclose(moments.mean, [[5_000.0]])
    np.testing.assert_allclose(moments.trace_covariance, [25_000_000.0])
    np.testing.assert_allclose(moments.effective_sample_size, [2.0])
    assert np.isfinite(moments.log_density).all()


def test_every_family_readout_matches_its_exact_channel_formula() -> None:
    rng = np.random.default_rng(31)
    reference = rng.normal(size=(101, 3))
    query = rng.normal(size=(6, 3))
    scale = 0.4
    moments = EmpiricalGaussianChannel(
        reference, reference_chunk_size=13
    ).posterior(query, scale, query_chunk_size=2)
    ambient_dim = reference.shape[1]

    np.testing.assert_allclose(
        diffusion_flipd(
            moments.score,
            moments.score_divergence,
            sigma=scale,
            ambient_dim=ambient_dim,
        ),
        moments.full,
    )

    alpha = 0.8
    beta = alpha * scale
    alpha_dot = 0.3
    beta_dot = -0.7
    a = alpha_dot / alpha
    kappa = beta_dot / beta - a
    posterior_noise_mean = (query - moments.mean) / scale
    velocity = alpha_dot * moments.mean + beta_dot * posterior_noise_mean
    velocity_divergence = ambient_dim * a + kappa * (
        ambient_dim - moments.response
    )
    marginal_score = moments.score / alpha
    evaluation_point = alpha * query
    np.testing.assert_allclose(
        affine_fm_response(
            velocity_divergence,
            ambient_dim=ambient_dim,
            alpha_log_derivative=a,
            log_noise_ratio_derivative=kappa,
        ),
        moments.response,
    )
    np.testing.assert_allclose(
        affine_fm_full(
            velocity,
            velocity_divergence,
            marginal_score,
            evaluation_point,
            ambient_dim=ambient_dim,
            alpha_log_derivative=a,
            log_noise_ratio_derivative=kappa,
        ),
        moments.full,
    )

    t = 1.0 / (1.0 + scale)
    rectified_noise_mean = (query - moments.mean) / scale
    rectified_velocity = moments.mean - rectified_noise_mean
    rectified_a = 1.0 / t
    rectified_kappa = -1.0 / (t * (1.0 - t))
    rectified_divergence = ambient_dim * rectified_a + rectified_kappa * (
        ambient_dim - moments.response
    )
    np.testing.assert_allclose(
        rectified_flow_response(
            rectified_divergence, t=t, ambient_dim=ambient_dim
        ),
        moments.response,
    )
    np.testing.assert_allclose(
        rectified_flow_full(
            rectified_velocity,
            rectified_divergence,
            query,
            t=t,
            ambient_dim=ambient_dim,
        ),
        moments.full,
    )

    diffusivity = 1.7
    time_to_go = scale**2 / diffusivity
    forward_drift = (moments.mean - query) / time_to_go
    forward_divergence = (moments.response - ambient_dim) / time_to_go
    np.testing.assert_allclose(
        sb_forward_response(
            forward_divergence,
            time_to_go=time_to_go,
            ambient_dim=ambient_dim,
        ),
        moments.response,
    )
    np.testing.assert_allclose(
        sb_forward_full(
            forward_drift,
            forward_divergence,
            time_to_go=time_to_go,
            diffusivity=diffusivity,
            ambient_dim=ambient_dim,
        ),
        moments.full,
    )

    # Canonical empirical bridge: constant backward factor, hence b^- = 0,
    # current velocity b^+/2 and marginal score b^+/gamma.
    np.testing.assert_allclose(
        sb_current_full(
            0.5 * forward_drift,
            0.5 * forward_divergence,
            moments.score,
            time_to_go=time_to_go,
            ambient_dim=ambient_dim,
        ),
        moments.full,
    )

    np.testing.assert_allclose(
        nf_fixed_density(
            moments.scale_velocity,
            moments.scale_velocity_divergence,
            moments.score,
            ambient_dim=ambient_dim,
        ),
        moments.full,
    )
    np.testing.assert_allclose(
        nf_calibrated_native(
            moments.scale_velocity_divergence, ambient_dim=ambient_dim
        ),
        moments.response,
    )
    np.testing.assert_allclose(
        cnf_calibrated_native(
            moments.scale_velocity_divergence,
            log_scale_derivative=1.0,
            ambient_dim=ambient_dim,
        ),
        moments.response,
    )

    outputs = moments.all_readouts()
    assert tuple(outputs) == READOUT_IDS
    for readout_id in READOUT_IDS:
        expected = (
            moments.response
            if readout_branch(readout_id) == "response"
            else moments.full
        )
        np.testing.assert_allclose(outputs[readout_id], expected)


def test_half_line_boundary_has_variance_defect_but_full_identity() -> None:
    # Midpoint quadrature for Lebesgue measure on the half-line.  At the
    # boundary, its Gaussian posterior is half-normal: variance 1 - 2/pi,
    # squared mean 2/pi, and second moment 1.
    count = 30_000
    cutoff = 9.0
    reference = ((np.arange(count) + 0.5) * cutoff / count)[:, None]
    moments = EmpiricalGaussianChannel(
        reference, reference_chunk_size=257
    ).posterior([[0.0]], scale=1.0)

    np.testing.assert_allclose(moments.response, [1.0 - 2.0 / math.pi], atol=2e-7)
    np.testing.assert_allclose(
        moments.normalized_squared_bias, [2.0 / math.pi], atol=2e-7
    )
    np.testing.assert_allclose(moments.full, [1.0], atol=2e-7)


def test_label_free_scale_selector_finds_interior_plateau_and_serializes() -> None:
    scales = np.array([1.6, 0.1, 0.8, 0.2, 0.4])  # intentionally unsorted
    curves = np.array(
        [
            [3.0, 0.0, 1.02, 1.0, 1.01],
            [4.0, 0.0, 2.04, 2.0, 2.02],
            [5.0, np.nan, 3.06, 3.0, 3.03],
        ]
    )
    index, diagnostics = select_stable_scale(
        scales, curves, window=1, min_valid_fraction=2.0 / 3.0
    )

    assert index == 4
    assert scales[index] == pytest.approx(0.4)
    assert diagnostics["uses_ground_truth"] is False
    assert diagnostics["selected_index"] == index
    assert diagnostics["stability_scores"][1] is None  # smallest-scale edge
    json.dumps(diagnostics, allow_nan=False)


def test_scale_selector_can_mask_unreliable_validation_values() -> None:
    scales = [0.1, 0.2, 0.4, 0.8, 1.6]
    curves = np.array([[0.0, 0.0, 0.0, 2.0, 2.01]] * 4)
    reliable = np.ones_like(curves, dtype=bool)
    reliable[:, :3] = False

    with pytest.raises(ValueError, match="no scale"):
        select_stable_scale(
            scales,
            curves,
            valid_mask=reliable,
            min_valid_fraction=1.0,
        )


def test_oracle_rejects_unknown_readout_and_invalid_empirical_mass() -> None:
    with pytest.raises(ValueError, match="positive mass"):
        EmpiricalGaussianChannel([[0.0], [1.0]], sample_weight=[0.0, 0.0])
    channel = EmpiricalGaussianChannel([[0.0], [1.0]])
    with pytest.raises(ValueError, match="unknown oracle readout_id"):
        channel.readout("not_a_readout", [[0.0]], 1.0)
