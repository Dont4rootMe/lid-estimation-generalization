from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from models.neural_fields import (  # noqa: E402
    NeuralFieldConfig,
    ScaleConditionedNeuralField,
    exact_divergence,
    hutchinson_divergence,
)
from models.training import (  # noqa: E402
    EpochMetrics,
    TrainingConfig,
    TrainingResult,
    load_checkpoint,
    predict_lid,
    train_model,
)


def _tiny_config(seed: int = 7) -> TrainingConfig:
    return TrainingConfig(
        seed=seed,
        device="cpu",
        epochs=2,
        batch_size=8,
        learning_rate=2.0e-3,
        weight_decay=0.0,
        hidden_dim=16,
        depth=1,
        time_embedding_dim=8,
        early_stopping_patience=None,
        gradient_clip_norm=1.0,
        sigma_min=0.1,
        sigma_max=0.5,
        fourier_features=4,
        max_condition_frequency=10.0,
    )


def test_scale_conditioned_field_flattens_and_restores_images() -> None:
    config = NeuralFieldConfig(
        ambient_dim=12,
        hidden_dim=16,
        depth=2,
        condition_dim=8,
        fourier_features=4,
    )
    model = ScaleConditionedNeuralField(config)
    images = torch.randn(5, 1, 3, 4)
    output = model(images, torch.linspace(0.1, 0.9, 5))
    assert output.shape == images.shape
    assert torch.isfinite(output).all()


def test_exact_and_hutchinson_divergence_match_diagonal_reference() -> None:
    class DiagonalField(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("diagonal", torch.tensor([1.0, -2.0, 3.0, 4.0]))

        def forward(self, inputs, condition):  # type: ignore[no-untyped-def]
            del condition
            return inputs * self.diagonal.reshape(1, 2, 2)

    field = DiagonalField()
    inputs = torch.randn(6, 2, 2)
    expected = torch.full((6,), 6.0)
    exact = exact_divergence(field, inputs, 0.5)
    estimate_a = hutchinson_divergence(
        field, inputs, 0.5, num_probes=3, seed=19
    )
    estimate_b = hutchinson_divergence(
        field, inputs, 0.5, num_probes=3, seed=19
    )
    torch.testing.assert_close(exact, expected)
    torch.testing.assert_close(estimate_a, expected)
    torch.testing.assert_close(estimate_b, estimate_a)


@pytest.mark.parametrize("family", ["diffusion", "rectified_flow"])
def test_train_model_writes_loadable_checkpoint_for_both_families(
    family: str, tmp_path: Path
) -> None:
    rng = np.random.default_rng(42)
    train = rng.normal(size=(24, 3)).astype(np.float32)
    validation = rng.normal(size=(12, 3)).astype(np.float32)
    checkpoint = tmp_path / family / "model.ckpt"
    result = train_model(
        family, train, validation, _tiny_config(), checkpoint
    )

    assert checkpoint.is_file()
    assert len(result.checkpoint_sha256) == 64
    assert len(result.preprocessing_sha256) == 64
    assert result.preprocessing["kind"] == "train_mean_global_rms_v1"
    assert result.metrics["epochs_completed"] == 2
    assert np.isfinite(result.metrics["best_validation_loss"])

    loaded = load_checkpoint(checkpoint)
    assert loaded.family == result.family
    assert loaded.preprocessing_sha256 == result.preprocessing_sha256
    assert loaded.checkpoint_sha256 == result.checkpoint_sha256
    for name, value in result.model.state_dict().items():
        torch.testing.assert_close(value.cpu(), loaded.model.state_dict()[name].cpu())

    lid = predict_lid(
        loaded,
        validation[:4],
        0.25 if family == "rectified_flow" else 0.2,
        divergence_backend="exact",
        batch_size=2,
    )
    assert lid.shape == (4,)
    assert np.isfinite(lid).all()


def test_same_seed_reproduces_history_and_weights(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    train = rng.normal(size=(16, 2)).astype(np.float32)
    validation = rng.normal(size=(8, 2)).astype(np.float32)
    config = _tiny_config(seed=101)
    first = train_model(
        "rectified_flow", train, validation, config, tmp_path / "first.ckpt"
    )
    second = train_model(
        "rectified_flow", train, validation, config, tmp_path / "second.ckpt"
    )
    assert first.history == second.history
    for name, value in first.model.state_dict().items():
        torch.testing.assert_close(value, second.model.state_dict()[name])


def test_diffusion_denoiser_is_converted_to_score_for_readout() -> None:
    class IdentityDenoiser(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = nn.Parameter(torch.tensor(0.0))
            self.config = SimpleNamespace(ambient_dim=3)

        def forward(self, inputs, condition):  # type: ignore[no-untyped-def]
            del condition
            return inputs + self.anchor * 0.0

    query = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]], dtype=np.float32)
    lid = predict_lid(
        IdentityDenoiser(),
        query,
        0.2,
        family="diffusion",
        divergence_backend="exact",
    )
    np.testing.assert_allclose(lid, np.full(2, 3.0), rtol=1e-6, atol=1e-6)


def test_diffusion_readout_is_invariant_to_checkpointed_scalar_affine_transform(
    tmp_path: Path,
) -> None:
    class ContractiveDenoiser(nn.Module):
        def __init__(self, factor: float) -> None:
            super().__init__()
            self.anchor = nn.Parameter(torch.tensor(0.0))
            self.factor = factor
            self.config = SimpleNamespace(ambient_dim=2)

        def forward(self, inputs, condition):  # type: ignore[no-untyped-def]
            del condition
            return self.factor * inputs + self.anchor * 0.0

    model = ContractiveDenoiser(0.75)
    metric = EpochMetrics(1, 1.0, 1.0, 1.0e-3)

    def result(mean: np.ndarray, scale: float, name: str) -> TrainingResult:
        return TrainingResult(
            family="gaussian_diffusion",
            model=model,  # type: ignore[arg-type]
            config=TrainingConfig(device="cpu", epochs=1),
            history=(metric,),
            best_epoch=1,
            best_validation_loss=1.0,
            checkpoint_path=tmp_path / name,
            checkpoint_sha256="a" * 64,
            normalization_mean=torch.tensor(mean, dtype=torch.float32),
            normalization_scale=scale,
            preprocessing={},
            preprocessing_sha256="b" * 64,
        )

    query = np.array([[2.0, -1.0], [0.5, 3.0]], dtype=np.float32)
    mean = np.array([0.25, -0.5], dtype=np.float32)
    first = predict_lid(
        result(mean, 2.0, "first.ckpt"),
        query,
        0.2,
        divergence_backend="exact",
    )
    multiplier = 3.5
    translation = np.array([10.0, -4.0], dtype=np.float32)
    transformed = multiplier * query + translation
    transformed_mean = multiplier * mean + translation
    second = predict_lid(
        result(transformed_mean, multiplier * 2.0, "second.ckpt"),
        transformed,
        0.2,
        divergence_backend="exact",
    )
    np.testing.assert_allclose(second, first, rtol=2e-5, atol=2e-5)


def test_training_config_accepts_pilot_yaml_keys() -> None:
    config = TrainingConfig.from_mapping(
        {
            "seed": 0,
            "device": "cpu",
            "epochs": 2,
            "batch_size": 8,
            "learning_rate": 1.0e-3,
            "weight_decay": 0.0,
            "hidden_dim": 16,
            "depth": 1,
            "time_embedding_dim": 8,
            "validation_interval": 1,
            "early_stopping_patience": None,
            "gradient_clip_norm": 1.0,
            "num_workers": 0,
            "deterministic": True,
        }
    )
    assert config.time_embedding_dim == 8
