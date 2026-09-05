import math

import pytest

pytest.importorskip("torch")
import torch
from torch import nn

from models.vp_baseline import (
    VPSchedule,
    VPBottleneckMLP,
    denoise,
    vp_primitives,
    vp_loss,
)
from models.readouts import diffusion_flipd


def flipd_components(model, query, lam, probes=0):
    p = vp_primitives(model, query, lam, probes=probes)
    response = query.shape[1] + p.sigma**2 * p.score_divergence
    correction = p.sigma**2 * p.score.square().sum(dim=1)
    full = diffusion_flipd(
        p.score.numpy(),
        p.score_divergence.numpy(),
        sigma=p.sigma,
        ambient_dim=query.shape[1],
    )
    return torch.as_tensor(full), response, correction


def test_schedule_roundtrip_and_endpoints():
    s = VPSchedule()
    for lam in (1e-5, 0.01, 0.1, 1, 16, 100):
        alpha, sigma = s.coefficients(
            torch.tensor(s.time_for_lambda(lam), dtype=torch.float64)
        )
        assert float(sigma / alpha) == pytest.approx(lam, rel=1e-10)
        assert float(alpha**2 + sigma**2) == pytest.approx(1)
    alpha, sigma = s.coefficients(torch.tensor(0.0))
    assert alpha == 1 and sigma == 0
    assert float(s.coefficients(torch.tensor(1.0))[0]) < 0.007
    with pytest.raises(ValueError):
        s.time_for_lambda(1000)


class GaussianOracle(nn.Module):
    def __init__(self, variances):
        super().__init__()
        self.register_buffer(
            "variances", torch.as_tensor(variances, dtype=torch.float64)
        )

    def forward(self, y, time):
        alpha, sigma = VPSchedule().coefficients(time)
        return (
            -sigma[:, None]
            * y
            / (alpha[:, None] ** 2 * self.variances + sigma[:, None] ** 2)
        )


def test_oracle_flipd_native_query_and_divergence():
    variance = torch.tensor([1.0, 2.0, 0.0, 0.0], dtype=torch.float64)
    model = GaussianOracle(variance)
    query = torch.tensor([[0.5, -0.7, 0.0, 0.0]], dtype=torch.float64)
    for lam in (0.002, 0.1, 1.0, 8.0):
        expected_response = (variance / (variance + lam**2)).sum()
        expected_correction = (lam**2 * query**2 / (variance + lam**2) ** 2).sum()
        full, response, correction = flipd_components(model, query, lam)
        torch.testing.assert_close(
            response, expected_response.expand(1), rtol=1e-9, atol=1e-9
        )
        torch.testing.assert_close(correction, expected_correction.expand(1))
        torch.testing.assert_close(full, response + correction)
        approx = flipd_components(model, query, lam, probes=4)[0]
        torch.testing.assert_close(approx, full)
    assert float(flipd_components(model, query, 0.002)[0]) == pytest.approx(2, abs=1e-5)


def test_upstream_topology_parameter_count_and_checkpoint_roundtrip():
    with torch.device("meta"):
        model = VPBottleneckMLP(784, (4096, 2048, 1024, 1024, 512, 512))
    assert sum(p.numel() for p in model.parameters()) == 55_652_880
    small = VPBottleneckMLP(5, (16, 8, 4), time_dim=8).double()
    clone = VPBottleneckMLP(5, (16, 8, 4), time_dim=8).double()
    clone.load_state_dict(small.state_dict())
    x, t = torch.randn(3, 5, dtype=torch.float64), torch.rand(3, dtype=torch.float64)
    torch.testing.assert_close(small(x, t), clone(x, t))
    loss = vp_loss(small, x, t, torch.randn_like(x))
    loss.backward()
    assert math.isfinite(float(loss.detach()))
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in small.parameters()
    )


def test_denoiser_sign():
    class FixedNoise(nn.Module):
        def forward(self, y, time):
            return -noise

    clean = torch.randn(3, 4, dtype=torch.float64)
    noise = torch.randn_like(clean)
    time = torch.tensor([0.01, 0.1, 0.9], dtype=torch.float64)
    alpha, sigma = VPSchedule().coefficients(time)
    noisy = alpha[:, None] * clean + sigma[:, None] * noise
    torch.testing.assert_close(denoise(FixedNoise(), noisy, time), clean)
    assert vp_loss(FixedNoise(), clean, time, noise) == 0
