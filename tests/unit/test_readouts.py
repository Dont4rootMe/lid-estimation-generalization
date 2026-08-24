from __future__ import annotations

import numpy as np
import pytest

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


def test_diffusion_formula() -> None:
    score = np.array([[1.0, 2.0], [0.0, 1.0]])
    divergence = np.array([-3.0, -2.0])
    np.testing.assert_allclose(
        diffusion_flipd(score, divergence, sigma=0.5, ambient_dim=2),
        [2.5, 1.75],
    )


def test_affine_full_adds_continuity_correction() -> None:
    velocity = np.array([[0.25, -0.5]])
    score = np.array([[2.0, 1.0]])
    point = np.array([[1.0, 2.0]])
    response = affine_fm_response(
        [1.5],
        ambient_dim=2,
        alpha_log_derivative=0.25,
        log_noise_ratio_derivative=-2.0,
    )
    full = affine_fm_full(
        velocity,
        [1.5],
        score,
        point,
        ambient_dim=2,
        alpha_log_derivative=0.25,
        log_noise_ratio_derivative=-2.0,
    )
    expected_correction = np.dot(0.25 * point[0] - velocity[0], score[0]) / -2.0
    np.testing.assert_allclose(full, response + expected_correction)


def test_rectified_full_adds_squared_bias() -> None:
    velocity = np.array([[1.0, -1.0]])
    data = np.array([[0.0, 0.0]])
    response = rectified_flow_response([2.0], t=0.75, ambient_dim=2)
    full = rectified_flow_full(
        velocity, [2.0], data, t=0.75, ambient_dim=2
    )
    np.testing.assert_allclose(full, response + 0.75**2 * 2.0)


def test_bridge_response_full_and_current_factor_two() -> None:
    drift = np.array([[2.0, 0.0]])
    response = sb_forward_response([-4.0], time_to_go=0.25, ambient_dim=2)
    full = sb_forward_full(
        drift,
        [-4.0],
        time_to_go=0.25,
        diffusivity=2.0,
        ambient_dim=2,
    )
    np.testing.assert_allclose(full, response + 0.5)
    current = sb_current_full(
        [[1.0, 0.0]],
        [-1.0],
        [[2.0, 0.0]],
        time_to_go=0.25,
        ambient_dim=2,
    )
    np.testing.assert_allclose(current, [2.5])


def test_nf_gauge_invariant_and_calibrated_forms() -> None:
    fixed = nf_fixed_density(
        [[1.0, 2.0]], [-3.0], [[0.5, -0.5]], ambient_dim=2
    )
    np.testing.assert_allclose(fixed, [5.5])
    np.testing.assert_allclose(
        nf_calibrated_native([1.5], ambient_dim=4), [2.5]
    )
    np.testing.assert_allclose(
        cnf_calibrated_native(
            [3.0], log_scale_derivative=-2.0, ambient_dim=4
        ),
        [5.5],
    )


@pytest.mark.parametrize("bad_t", [0.0, 1.0, -0.1, 1.1])
def test_rectified_flow_rejects_endpoint_evaluation(bad_t: float) -> None:
    with pytest.raises(ValueError):
        rectified_flow_response([0.0], t=bad_t, ambient_dim=2)
