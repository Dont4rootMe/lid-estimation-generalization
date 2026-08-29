from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from models.normalizing_flow import (
    NF_DENSITY_CONTRACT,
    ConditionalFlowConfig,
    ScaleConditionedRealNVP,
    conditional_smoothed_nll,
    fixed_point_lid,
    fixed_point_lid_local_ols,
    fixed_point_lid_symmetric_fd,
    fixed_point_likelihood_readouts,
    fixed_point_log_likelihood_curve,
)


def _flow(*, dtype: torch.dtype = torch.float64) -> ScaleConditionedRealNVP:
    model = ScaleConditionedRealNVP(
        ConditionalFlowConfig(
            ambient_dim=4,
            hidden_dim=12,
            num_coupling_layers=3,
            conditioner_depth=2,
            condition_dim=8,
            fourier_features=3,
            max_condition_frequency=5.0,
            log_scale_limit=1.5,
        )
    ).to(dtype=dtype)
    generator = torch.Generator(device="cpu").manual_seed(91)
    with torch.no_grad():
        for coupling in model.couplings:
            output = coupling.conditioner[-1]
            output.weight.copy_(
                0.04
                * torch.randn(
                    output.weight.shape,
                    generator=generator,
                    dtype=output.weight.dtype,
                )
            )
            output.bias.copy_(
                0.04
                * torch.randn(
                    output.bias.shape,
                    generator=generator,
                    dtype=output.bias.dtype,
                )
            )
    return model.eval()


class _AnalyticLogPolynomial(torch.nn.Module):
    """Conditional density-shaped oracle with a known log-scale derivative."""

    def __init__(self, *, ambient_dim: int, linear: float, quadratic: float) -> None:
        super().__init__()
        self.config = SimpleNamespace(ambient_dim=ambient_dim)
        self.linear = float(linear)
        self.quadratic = float(quadratic)

    def log_prob(
        self, observations: torch.Tensor, epsilon: torch.Tensor | float
    ) -> torch.Tensor:
        value = torch.as_tensor(
            epsilon,
            device=observations.device,
            dtype=observations.dtype,
        )
        if value.ndim == 0:
            value = value.expand(observations.shape[0])
        log_epsilon = torch.log(value)
        flattened = observations.reshape(observations.shape[0], -1)
        fixed_term = -0.25 * flattened.square().sum(dim=1)
        return (
            fixed_term
            + self.linear * log_epsilon
            + self.quadratic * log_epsilon.square()
        )


class _SmoothedLinearSubspaceDensity(torch.nn.Module):
    """Exact Gaussian smoothing of N(0,I_d) embedded in ambient R^D."""

    def __init__(self, *, ambient_dim: int, intrinsic_dim: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(ambient_dim=ambient_dim)
        self.intrinsic_dim = intrinsic_dim

    def log_prob(
        self, observations: torch.Tensor, epsilon: torch.Tensor | float
    ) -> torch.Tensor:
        value = torch.as_tensor(
            epsilon,
            device=observations.device,
            dtype=observations.dtype,
        )
        if value.ndim == 0:
            value = value.expand(observations.shape[0])
        intrinsic_variance = 1.0 + value.square()
        normal_variance = value.square()
        ambient_dim = int(self.config.ambient_dim)
        normal_dim = ambient_dim - self.intrinsic_dim
        intrinsic = observations[:, : self.intrinsic_dim]
        normal = observations[:, self.intrinsic_dim :]
        return -0.5 * (
            self.intrinsic_dim
            * (math.log(2.0 * math.pi) + torch.log(intrinsic_variance))
            + intrinsic.square().sum(dim=1) / intrinsic_variance
            + normal_dim * (math.log(2.0 * math.pi) + torch.log(normal_variance))
            + normal.square().sum(dim=1) / normal_variance
        )


class _RecordingDensity(torch.nn.Module):
    """Minimal objective oracle that records its exact noisy training batch."""

    def __init__(self, *, ambient_dim: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(ambient_dim=ambient_dim)
        self.observations: torch.Tensor | None = None
        self.epsilon: torch.Tensor | None = None

    def log_prob(
        self, observations: torch.Tensor, epsilon: torch.Tensor | float
    ) -> torch.Tensor:
        self.observations = observations.detach().clone()
        self.epsilon = torch.as_tensor(epsilon).detach().clone()
        return -observations.square().sum(dim=1)


def test_conditional_flow_is_invertible_and_log_determinants_cancel() -> None:
    model = _flow()
    latent = torch.tensor(
        [[0.2, -0.4, 0.8, 1.1], [-1.0, 0.3, 0.1, -0.2]],
        dtype=torch.float64,
    )
    epsilon = torch.tensor([0.07, 0.3], dtype=torch.float64)
    observations, forward_log_det = model.decode(latent, epsilon)
    recovered, inverse_log_det = model.encode(observations, epsilon)

    torch.testing.assert_close(recovered, latent, rtol=1.0e-10, atol=1.0e-10)
    torch.testing.assert_close(
        forward_log_det + inverse_log_det,
        torch.zeros_like(forward_log_det),
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_reported_change_of_variables_jacobian_is_exact() -> None:
    model = _flow()
    latent = torch.tensor([0.4, -0.7, 0.2, 1.3], dtype=torch.float64)
    epsilon = 0.17

    def transform(value: torch.Tensor) -> torch.Tensor:
        output, _ = model.decode(value[None, :], epsilon)
        return output[0]

    jacobian = torch.autograd.functional.jacobian(transform, latent)
    _, expected = torch.linalg.slogdet(jacobian)
    _, reported = model.decode(latent[None, :], epsilon)
    torch.testing.assert_close(reported[0], expected, rtol=1.0e-9, atol=1.0e-9)


def test_fixed_point_readout_is_exact_log_scale_autograd_derivative() -> None:
    model = _flow()
    observations = torch.tensor(
        [[0.2, -0.3, 0.1, 0.7], [-0.8, 0.4, 0.6, -0.2]],
        dtype=torch.float64,
    )
    epsilon = 0.13
    step = 1.0e-5
    plus = model.log_prob(observations, epsilon * math.exp(step))
    minus = model.log_prob(observations, epsilon * math.exp(-step))
    finite_difference = (plus - minus) / (2.0 * step)
    predicted = fixed_point_lid(model, observations, epsilon)

    torch.testing.assert_close(
        predicted - model.config.ambient_dim,
        finite_difference,
        rtol=2.0e-5,
        atol=2.0e-6,
    )
    assert NF_DENSITY_CONTRACT["readout"] == (
        "ambient_dim + d_log_epsilon log_p_epsilon_at_fixed_x"
    )


def test_likelihood_curve_matches_direct_fixed_point_evaluation() -> None:
    model = _flow()
    observations = torch.tensor(
        [[0.2, -0.3, 0.1, 0.7], [-0.8, 0.4, 0.6, -0.2]],
        dtype=torch.float64,
        requires_grad=True,
    )
    epsilons = torch.tensor([0.08, 0.13, 0.31], dtype=torch.float64)
    curve = fixed_point_log_likelihood_curve(model, observations, epsilons)
    expected = torch.column_stack(
        [model.log_prob(observations.detach(), epsilon) for epsilon in epsilons]
    )

    assert curve.shape == (2, 3)
    assert curve.requires_grad is False
    torch.testing.assert_close(curve, expected)
    assert observations.grad is None


def test_all_readouts_recover_known_log_polynomial_slope() -> None:
    model = _AnalyticLogPolynomial(
        ambient_dim=7,
        linear=-2.3,
        quadratic=0.4,
    ).double()
    observations = torch.tensor(
        [[0.2, -0.3], [-0.8, 0.4]],
        dtype=torch.float64,
    )
    epsilon = 0.2
    expected = torch.full(
        (2,),
        7.0 - 2.3 + 0.8 * math.log(epsilon),
        dtype=torch.float64,
    )
    prediction = fixed_point_likelihood_readouts(
        model,
        observations,
        epsilon,
        finite_difference_log_step=0.01,
        ols_log_step=0.05,
    )

    for value in (
        prediction.lid_autograd,
        prediction.lid_symmetric_fd,
        prediction.lid_ols3,
        prediction.lid_ols5,
        prediction.lid_ols9,
    ):
        torch.testing.assert_close(value, expected, rtol=1.0e-11, atol=1.0e-11)
    assert prediction.finite_difference_log_likelihood.shape == (2, 2)
    assert prediction.ols_log_likelihood.shape == (2, 9)


def test_symmetric_fd_equals_ols3_when_they_share_the_same_grid() -> None:
    model = _flow()
    observations = torch.tensor(
        [[0.2, -0.3, 0.1, 0.7], [-0.8, 0.4, 0.6, -0.2]],
        dtype=torch.float64,
    )
    finite_difference = fixed_point_lid_symmetric_fd(
        model,
        observations,
        0.2,
        log_step=0.04,
    )
    ols3 = fixed_point_lid_local_ols(
        model,
        observations,
        0.2,
        log_step=0.04,
        window_size=3,
    )
    torch.testing.assert_close(finite_difference, ols3, rtol=1.0e-12, atol=1.0e-12)


def test_readouts_match_smoothed_subspace_oracle_at_the_origin() -> None:
    ambient_dim = 6
    intrinsic_dim = 2
    model = _SmoothedLinearSubspaceDensity(
        ambient_dim=ambient_dim,
        intrinsic_dim=intrinsic_dim,
    ).double()
    observations = torch.zeros((3, ambient_dim), dtype=torch.float64)
    epsilon = 0.2
    exact = torch.full(
        (3,),
        intrinsic_dim / (1.0 + epsilon**2),
        dtype=torch.float64,
    )
    autograd = fixed_point_lid(model, observations, epsilon)
    coarse = fixed_point_lid_symmetric_fd(
        model,
        observations,
        epsilon,
        log_step=0.1,
    )
    fine = fixed_point_lid_symmetric_fd(
        model,
        observations,
        epsilon,
        log_step=0.01,
    )

    torch.testing.assert_close(autograd, exact, rtol=1.0e-12, atol=1.0e-12)
    assert torch.max(torch.abs(fine - exact)) < torch.max(torch.abs(coarse - exact))
    torch.testing.assert_close(fine, exact, rtol=2.0e-6, atol=2.0e-6)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda model, observations: fixed_point_log_likelihood_curve(
                model, observations, [0.2, 0.1]
            ),
            "strictly increasing",
        ),
        (
            lambda model, observations: fixed_point_lid_symmetric_fd(
                model, observations, 0.2, log_step=0.0
            ),
            "log_step",
        ),
        (
            lambda model, observations: fixed_point_lid_local_ols(
                model,
                observations,
                0.2,
                log_step=0.05,
                window_size=7,
            ),
            "window_size",
        ),
    ],
)
def test_likelihood_readout_inputs_are_strictly_validated(call, message: str) -> None:
    model = _flow()
    observations = torch.zeros((2, 4), dtype=torch.float64)
    with pytest.raises((TypeError, ValueError), match=message):
        call(model, observations)


def test_fixed_epsilon_objective_uses_exact_condition_and_noise_scale() -> None:
    model = _RecordingDensity(ambient_dim=3)
    clean = torch.tensor(
        [[0.1, -0.2, 0.3], [0.7, 0.4, -0.5]],
        dtype=torch.float64,
    )
    epsilon = 0.17
    generator = torch.Generator(device="cpu").manual_seed(812)
    expected_generator = torch.Generator(device="cpu").manual_seed(812)
    expected_noise = torch.randn(
        clean.shape,
        dtype=clean.dtype,
        generator=expected_generator,
    )
    expected_noisy = clean + epsilon * expected_noise

    loss = conditional_smoothed_nll(
        model,
        clean,
        epsilon_min=epsilon,
        epsilon_max=epsilon,
        generator=generator,
    )

    assert model.epsilon is not None
    assert model.observations is not None
    torch.testing.assert_close(
        model.epsilon,
        torch.full((clean.shape[0],), epsilon, dtype=clean.dtype),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        model.observations,
        expected_noisy,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(generator.get_state(), expected_generator.get_state())
    torch.testing.assert_close(
        loss,
        expected_noisy.square().sum(dim=1).mean() / 3.0,
    )


def test_smoothed_nll_rejects_reversed_or_nonfinite_epsilon_bounds() -> None:
    model = _RecordingDensity(ambient_dim=2)
    clean = torch.zeros((2, 2), dtype=torch.float32)
    generator = torch.Generator(device="cpu").manual_seed(3)

    for epsilon_min, epsilon_max in ((0.2, 0.1), (0.1, math.inf)):
        with pytest.raises(ValueError, match="0 < min <= max"):
            conditional_smoothed_nll(
                model,
                clean,
                epsilon_min=epsilon_min,
                epsilon_max=epsilon_max,
                generator=generator,
            )


def test_zero_initialized_flow_is_standard_normal_at_every_scale() -> None:
    model = ScaleConditionedRealNVP(
        ConditionalFlowConfig(
            ambient_dim=3,
            hidden_dim=8,
            num_coupling_layers=2,
            conditioner_depth=1,
            condition_dim=4,
            fourier_features=2,
        )
    ).double()
    observations = torch.tensor([[0.1, -0.2, 0.3]], dtype=torch.float64)
    expected = -0.5 * (observations.square() + math.log(2.0 * math.pi)).sum(dim=-1)
    torch.testing.assert_close(model.log_prob(observations, 0.01), expected)
    torch.testing.assert_close(model.log_prob(observations, 0.8), expected)
    torch.testing.assert_close(
        fixed_point_lid(model, observations, 0.2),
        torch.full((1,), 3.0, dtype=torch.float64),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    readouts = fixed_point_likelihood_readouts(
        model,
        observations,
        0.2,
        finite_difference_log_step=0.01,
        ols_log_step=0.05,
    )
    for value in (
        readouts.lid_autograd,
        readouts.lid_symmetric_fd,
        readouts.lid_ols3,
        readouts.lid_ols5,
        readouts.lid_ols9,
    ):
        torch.testing.assert_close(
            value,
            torch.full((1,), 3.0, dtype=value.dtype),
            rtol=1.0e-10,
            atol=1.0e-10,
        )
