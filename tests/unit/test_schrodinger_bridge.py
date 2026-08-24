from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from models.neural_fields import NeuralFieldConfig, ScaleConditionedNeuralField
from models.schrodinger_bridge import (
    CANONICAL_BRIDGE_CONDITIONING,
    CANONICAL_BRIDGE_CONSTRUCTION,
    CANONICAL_FACTOR_F,
    CANONICAL_FACTOR_G,
    CANONICAL_INITIAL_MARGINAL,
    CANONICAL_REFERENCE_PROCESS,
    CANONICAL_TERMINAL_MARGINAL,
    BrownianBridgeSpec,
    brownian_bridge_contract,
    brownian_sb_terminal_denoising_loss,
    denoiser_to_forward_drift,
)
from models.training import EpochMetrics, TrainingConfig, TrainingResult, predict_lid


def _spec(*, diffusivity: float = 1.0) -> BrownianBridgeSpec:
    return BrownianBridgeSpec(
        construction=CANONICAL_BRIDGE_CONSTRUCTION,
        reference_process=CANONICAL_REFERENCE_PROCESS,
        initial_marginal=CANONICAL_INITIAL_MARGINAL,
        terminal_marginal=CANONICAL_TERMINAL_MARGINAL,
        factor_f=CANONICAL_FACTOR_F,
        factor_g=CANONICAL_FACTOR_G,
        conditioning=CANONICAL_BRIDGE_CONDITIONING,
        diffusivity=diffusivity,
        terminal_time=1.0,
    )


def _training_config() -> TrainingConfig:
    spec = _spec()
    return TrainingConfig(
        device="cpu",
        bridge_construction=spec.construction,
        bridge_reference_process=spec.reference_process,
        bridge_initial_marginal=spec.initial_marginal,
        bridge_terminal_marginal=spec.terminal_marginal,
        bridge_factor_f=spec.factor_f,
        bridge_factor_g=spec.factor_g,
        bridge_conditioning=spec.conditioning,
        bridge_diffusivity=spec.diffusivity,
        bridge_terminal_time=spec.terminal_time,
        bridge_tau_min=0.01,
        bridge_tau_max=1.0,
    )


def test_canonical_bridge_spec_rejects_ambiguous_constructions() -> None:
    spec = _spec()
    assert spec.construction == CANONICAL_BRIDGE_CONSTRUCTION
    assert spec.diffusivity == 1.0
    assert spec.terminal_time == 1.0

    with pytest.raises(ValueError, match="bridge_construction"):
        BrownianBridgeSpec(
            **{
                **spec.__dict__,
                "construction": "independent-endpoints",
            }
        )
    with pytest.raises(ValueError, match="bridge_diffusivity"):
        BrownianBridgeSpec(**{**spec.__dict__, "diffusivity": 0.0})


def test_bridge_contract_records_endpoint_factorization_and_tau_support() -> None:
    contract = brownian_bridge_contract(_spec(), tau_min=0.01, tau_max=0.5)
    assert contract == {
        "schema_version": 1,
        "family": "brownian_schrodinger_bridge",
        "construction": "terminal-data-lebesgue-factor-v1",
        "reference_process": "brownian-motion",
        "initial_marginal": "gaussian-convolution-of-terminal-data",
        "terminal_marginal": "dataset-terminal-law",
        "factor_f": "lebesgue-measure",
        "factor_g": "dataset-terminal-law",
        "conditioning": "time-to-go-tau",
        "diffusivity": 1.0,
        "terminal_time": 1.0,
        "tau_min": 0.01,
        "tau_max": 0.5,
        "trainable_primitive": "terminal-denoiser-to-forward-drift",
        "readout": "brownian-sb-forward-full",
    }


def test_terminal_denoising_loss_uses_tau_and_brownian_variance() -> None:
    class RecordingZeroDenoiser(nn.Module):
        inputs: torch.Tensor | None = None
        condition: torch.Tensor | None = None

        def forward(self, inputs, condition):  # type: ignore[no-untyped-def]
            assert inputs.shape[0] == condition.shape[0]
            assert torch.all((0.1 <= condition) & (condition <= 0.2))
            self.inputs = inputs.detach().clone()
            self.condition = condition.detach().clone()
            return torch.zeros_like(inputs)

    terminal = torch.zeros(12, 3)
    model = RecordingZeroDenoiser()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(17)
    loss = brownian_sb_terminal_denoising_loss(
        model,
        terminal,
        tau_min=0.1,
        tau_max=0.2,
        spec=_spec(diffusivity=2.0),
        generator=generator,
    )
    torch.testing.assert_close(loss, torch.zeros_like(loss))
    assert model.inputs is not None
    assert model.condition is not None

    replay = torch.Generator(device="cpu")
    replay.manual_seed(17)
    uniform = torch.rand(12, generator=replay)
    expected_tau = torch.exp(np.log(0.1) + uniform * (np.log(0.2) - np.log(0.1)))
    expected_noise = torch.randn((12, 3), generator=replay)
    expected_state = torch.sqrt(2.0 * expected_tau[:, None]) * expected_noise
    torch.testing.assert_close(model.condition, expected_tau)
    torch.testing.assert_close(model.inputs, expected_state)


def test_denoiser_conversion_recovers_forward_drift_and_trace() -> None:
    state = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
    denoised = 0.75 * state
    denoiser_divergence = torch.full((2,), 1.5)
    drift, divergence = denoiser_to_forward_drift(
        denoised,
        denoiser_divergence,
        state,
        tau=0.25,
        ambient_dim=2,
    )
    torch.testing.assert_close(drift, -state)
    torch.testing.assert_close(divergence, torch.full((2,), -2.0))


def test_identity_terminal_denoiser_gives_ambient_bridge_lid() -> None:
    class IdentityDenoiser(ScaleConditionedNeuralField):
        def __init__(self) -> None:
            super().__init__(
                NeuralFieldConfig(
                    ambient_dim=3,
                    hidden_dim=4,
                    depth=1,
                    condition_dim=4,
                    fourier_features=2,
                )
            )

        def forward(self, inputs, condition):  # type: ignore[no-untyped-def]
            del condition
            return inputs

    config = _training_config()
    metric = EpochMetrics(1, 0.0, 0.0, 1.0e-3)
    result = TrainingResult(
        family="brownian_schrodinger_bridge",
        model=IdentityDenoiser(),
        config=config,
        history=(metric,),
        best_epoch=1,
        best_validation_loss=0.0,
        checkpoint_path=Path("unused.pt"),
        checkpoint_sha256="a" * 64,
        normalization_mean=torch.zeros(3),
        normalization_scale=1.0,
        preprocessing={},
        preprocessing_sha256="b" * 64,
    )
    query = np.array([[1.0, -2.0, 3.0], [0.0, 4.0, -1.0]], dtype=np.float32)
    prediction = predict_lid(
        result,
        query,
        0.1,
        divergence_backend="exact",
        batch_size=1,
    )
    np.testing.assert_allclose(prediction, np.full(2, 3.0))
