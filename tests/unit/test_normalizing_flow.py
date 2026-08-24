from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from models.normalizing_flow import (
    NF_DENSITY_CONTRACT,
    ConditionalFlowConfig,
    ScaleConditionedRealNVP,
    fixed_point_lid,
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
