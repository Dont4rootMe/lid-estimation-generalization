from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn

import models.training as training_module
from models.neural_fields import (
    NeuralFieldConfig,
    ScaleConditionedNeuralField,
    exact_divergence,
    hutchinson_divergence,
)
from models.normalizing_flow import (
    NF_DENSITY_CONTRACT,
    ScaleConditionedRealNVP,
)
from models.schrodinger_bridge import (
    CANONICAL_BRIDGE_CONDITIONING,
    CANONICAL_BRIDGE_CONSTRUCTION,
    CANONICAL_FACTOR_F,
    CANONICAL_FACTOR_G,
    CANONICAL_INITIAL_MARGINAL,
    CANONICAL_REFERENCE_PROCESS,
    CANONICAL_TERMINAL_MARGINAL,
)
from models.training import (
    EpochMetrics,
    TrainingConfig,
    TrainingResult,
    load_checkpoint,
    predict_lid,
    predict_nf_log_likelihood,
    predict_nf_readouts,
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
        time_min=0.1,
        time_max=0.5,
        fourier_features=4,
        max_condition_frequency=10.0,
    )


def _tiny_bridge_config(seed: int = 7) -> TrainingConfig:
    return replace(
        _tiny_config(seed),
        bridge_construction=CANONICAL_BRIDGE_CONSTRUCTION,
        bridge_reference_process=CANONICAL_REFERENCE_PROCESS,
        bridge_initial_marginal=CANONICAL_INITIAL_MARGINAL,
        bridge_terminal_marginal=CANONICAL_TERMINAL_MARGINAL,
        bridge_factor_f=CANONICAL_FACTOR_F,
        bridge_factor_g=CANONICAL_FACTOR_G,
        bridge_conditioning=CANONICAL_BRIDGE_CONDITIONING,
        bridge_diffusivity=1.0,
        bridge_terminal_time=1.0,
        bridge_tau_min=0.1,
        bridge_tau_max=0.5,
    )


def _tiny_nf_config(seed: int = 7) -> TrainingConfig:
    return replace(
        _tiny_config(seed),
        depth=None,
        sigma_min=None,
        sigma_max=None,
        time_min=None,
        time_max=None,
        num_coupling_layers=2,
        conditioner_depth=1,
        log_scale_limit=1.25,
        epsilon_min=0.1,
        epsilon_max=0.5,
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
    estimate_a = hutchinson_divergence(field, inputs, 0.5, num_probes=3, seed=19)
    estimate_b = hutchinson_divergence(field, inputs, 0.5, num_probes=3, seed=19)
    torch.testing.assert_close(exact, expected)
    torch.testing.assert_close(estimate_a, expected)
    torch.testing.assert_close(estimate_b, estimate_a)


@pytest.mark.parametrize(
    "family", ["diffusion", "rectified_flow", "schrodinger_bridge"]
)
def test_train_model_writes_loadable_checkpoint_for_all_field_families(
    family: str, tmp_path: Path
) -> None:
    rng = np.random.default_rng(42)
    train = rng.normal(size=(24, 3)).astype(np.float32)
    validation = rng.normal(size=(12, 3)).astype(np.float32)
    checkpoint = tmp_path / family / "model.ckpt"
    config = _tiny_bridge_config() if family == "schrodinger_bridge" else _tiny_config()
    result = train_model(family, train, validation, config, checkpoint)

    assert checkpoint.is_file()
    assert len(result.checkpoint_sha256) == 64
    assert len(result.preprocessing_sha256) == 64
    assert result.preprocessing["kind"] == "train_mean_global_rms_v1"
    assert result.metrics["epochs_completed"] == 2
    assert np.isfinite(result.metrics["best_validation_loss"])

    loaded = load_checkpoint(checkpoint)
    assert loaded.family == result.family
    assert loaded.history == result.history
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


def test_train_model_resumes_from_atomic_validated_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(training_module, "TRAINING_PROGRESS_INTERVAL_EPOCHS", 1)
    rng = np.random.default_rng(31)
    train = rng.normal(size=(24, 3)).astype(np.float32)
    validation = rng.normal(size=(12, 3)).astype(np.float32)
    config = replace(_tiny_config(seed=113), epochs=4)
    uninterrupted = train_model(
        "diffusion",
        train,
        validation,
        config,
        tmp_path / "uninterrupted.ckpt",
    )

    progress = tmp_path / "resumed" / "training-progress.pt"
    checkpoint = tmp_path / "resumed" / "model.ckpt"
    original_save = training_module._atomic_torch_save
    interrupted = False

    def save_then_interrupt(path: Path, payload: dict[str, object]) -> None:
        nonlocal interrupted
        original_save(path, payload)
        if not interrupted:
            interrupted = True
            raise RuntimeError("simulated scheduler interruption")

    monkeypatch.setattr(training_module, "_atomic_torch_save", save_then_interrupt)
    with pytest.raises(RuntimeError, match="scheduler interruption"):
        train_model(
            "diffusion",
            train,
            validation,
            config,
            checkpoint,
            progress_checkpoint_path=progress,
        )
    assert progress.is_file()
    assert not checkpoint.exists()

    monkeypatch.setattr(training_module, "_atomic_torch_save", original_save)
    resumed = train_model(
        "diffusion",
        train,
        validation,
        config,
        checkpoint,
        progress_checkpoint_path=progress,
    )
    assert not progress.exists()
    assert resumed.history == uninterrupted.history
    assert resumed.best_epoch == uninterrupted.best_epoch
    assert resumed.best_validation_loss == uninterrupted.best_validation_loss
    for name, value in uninterrupted.model.state_dict().items():
        torch.testing.assert_close(value, resumed.model.state_dict()[name])


def test_train_model_rejects_tampered_progress_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(training_module, "TRAINING_PROGRESS_INTERVAL_EPOCHS", 1)
    rng = np.random.default_rng(32)
    train = rng.normal(size=(16, 2)).astype(np.float32)
    validation = rng.normal(size=(8, 2)).astype(np.float32)
    config = replace(_tiny_config(seed=117), epochs=2)
    progress = tmp_path / "progress.pt"
    original_save = training_module._atomic_torch_save

    def save_then_interrupt(path: Path, payload: dict[str, object]) -> None:
        original_save(path, payload)
        raise RuntimeError("simulated scheduler interruption")

    monkeypatch.setattr(training_module, "_atomic_torch_save", save_then_interrupt)
    with pytest.raises(RuntimeError, match="scheduler interruption"):
        train_model(
            "rectified_flow",
            train,
            validation,
            config,
            tmp_path / "model.ckpt",
            progress_checkpoint_path=progress,
        )
    payload = torch.load(progress, map_location="cpu", weights_only=True)
    payload["training_config"]["seed"] = 999
    torch.save(payload, progress)
    monkeypatch.setattr(training_module, "_atomic_torch_save", original_save)
    with pytest.raises(ValueError, match="config mismatch"):
        train_model(
            "rectified_flow",
            train,
            validation,
            config,
            tmp_path / "model.ckpt",
            progress_checkpoint_path=progress,
        )


def test_training_progress_rejects_different_validation_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(training_module, "TRAINING_PROGRESS_INTERVAL_EPOCHS", 1)
    rng = np.random.default_rng(33)
    train = rng.normal(size=(16, 2)).astype(np.float32)
    validation = rng.normal(size=(8, 2)).astype(np.float32)
    config = replace(_tiny_config(seed=119), epochs=3)
    progress = tmp_path / "progress.pt"
    original_save = training_module._atomic_torch_save

    def save_then_interrupt(path: Path, payload: dict[str, object]) -> None:
        original_save(path, payload)
        raise RuntimeError("simulated scheduler interruption")

    monkeypatch.setattr(training_module, "_atomic_torch_save", save_then_interrupt)
    with pytest.raises(RuntimeError, match="scheduler interruption"):
        train_model(
            "diffusion",
            train,
            validation,
            config,
            tmp_path / "model.ckpt",
            progress_checkpoint_path=progress,
        )
    monkeypatch.setattr(training_module, "_atomic_torch_save", original_save)
    with pytest.raises(ValueError, match="data identity mismatch"):
        train_model(
            "diffusion",
            train,
            np.full_like(validation, 17.0),
            config,
            tmp_path / "model.ckpt",
            progress_checkpoint_path=progress,
        )


def test_training_progress_path_must_differ_from_final_checkpoint(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(34)
    train = rng.normal(size=(16, 2)).astype(np.float32)
    validation = rng.normal(size=(8, 2)).astype(np.float32)
    shared_path = tmp_path / "shared.pt"

    with pytest.raises(ValueError, match="must differ"):
        train_model(
            "diffusion",
            train,
            validation,
            _tiny_config(seed=121),
            shared_path,
            progress_checkpoint_path=shared_path,
        )
    assert not shared_path.exists()


def test_cpu_state_dict_is_a_frozen_snapshot() -> None:
    model = nn.Linear(2, 2)
    snapshot = training_module._cpu_state_dict(model)
    expected = {name: value.clone() for name, value in snapshot.items()}

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(7.0)

    for name, value in snapshot.items():
        torch.testing.assert_close(value, expected[name])


def test_scale_conditioned_nf_trains_loads_and_predicts_exact_fixed_likelihood(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(54)
    train = rng.normal(size=(24, 3)).astype(np.float32)
    validation = rng.normal(size=(12, 3)).astype(np.float32)
    checkpoint = tmp_path / "scale-conditioned-nf.ckpt"
    result = train_model(
        "scale_conditioned_nf",
        train,
        validation,
        _tiny_nf_config(),
        checkpoint,
    )

    assert result.family == "scale_conditioned_normalizing_flow"
    assert isinstance(result.model, ScaleConditionedRealNVP)
    assert result.model.config.num_coupling_layers == 2
    assert result.model.config.conditioner_depth == 1
    assert result.model.config.log_scale_limit == pytest.approx(1.25)
    assert result.config.depth is None
    assert result.config.sigma_min is None
    assert result.config.time_min is None
    assert result.model_contract == {
        **dict(NF_DENSITY_CONTRACT),
        "epsilon_min": 0.1,
        "epsilon_max": 0.5,
    }

    loaded = load_checkpoint(checkpoint)
    assert isinstance(loaded.model, ScaleConditionedRealNVP)
    assert loaded.model_contract == result.model_contract
    for name, value in result.model.state_dict().items():
        torch.testing.assert_close(value.cpu(), loaded.model.state_dict()[name].cpu())
    lid = predict_lid(
        loaded,
        validation[:5],
        0.2,
        readout="fixed_likelihood",
        divergence_backend="exact",
        trace_probes=0,
        batch_size=2,
    )
    assert lid.shape == (5,)
    assert np.isfinite(lid).all()
    readouts = predict_nf_readouts(
        loaded,
        validation[:5],
        0.2,
        finite_difference_log_step=0.01,
        ols_log_step=0.05,
        batch_size=2,
    )
    assert readouts.finite_difference_epsilons.shape == (2,)
    assert readouts.finite_difference_log_likelihood.shape == (5, 2)
    assert readouts.ols_epsilons.shape == (9,)
    assert readouts.ols_log_likelihood.shape == (5, 9)
    assert set(readouts.lid_by_readout) == {
        "autograd",
        "symmetric_fd",
        "ols3",
        "ols5",
        "ols9",
    }
    for prediction in readouts.lid_by_readout.values():
        assert prediction.shape == (5,)
        assert np.isfinite(prediction).all()
    np.testing.assert_array_equal(readouts.lid_autograd, lid)

    unbatched = predict_nf_readouts(
        loaded,
        validation[:5],
        0.2,
        finite_difference_log_step=0.01,
        ols_log_step=0.05,
        batch_size=5,
    )
    np.testing.assert_array_equal(
        unbatched.finite_difference_log_likelihood,
        readouts.finite_difference_log_likelihood,
    )
    np.testing.assert_array_equal(
        unbatched.ols_log_likelihood,
        readouts.ols_log_likelihood,
    )
    for name, prediction in readouts.lid_by_readout.items():
        np.testing.assert_array_equal(unbatched.lid_by_readout[name], prediction)

    with pytest.raises(ValueError, match="outside.*training interval"):
        predict_nf_readouts(
            loaded,
            validation[:2],
            0.1,
            finite_difference_log_step=0.01,
            ols_log_step=0.05,
        )


def test_scale_conditioned_nf_requires_complete_explicit_config() -> None:
    with pytest.raises(ValueError, match="complete block"):
        replace(_tiny_config(), num_coupling_layers=2)
    with pytest.raises(ValueError, match="epsilon bounds"):
        replace(
            _tiny_nf_config(),
            epsilon_min=0.5,
            epsilon_max=0.1,
        )
    with pytest.raises(ValueError, match="dropout.*exactly 0"):
        replace(_tiny_nf_config(), dropout=0.1)


def test_fixed_epsilon_nf_training_and_checkpoint_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(541)
    train = rng.normal(size=(20, 3)).astype(np.float32)
    validation = rng.normal(size=(8, 3)).astype(np.float32)
    fixed_epsilon = 0.2
    config = replace(
        _tiny_nf_config(seed=19),
        epochs=1,
        epsilon_min=fixed_epsilon,
        epsilon_max=fixed_epsilon,
    )
    assert TrainingConfig.from_mapping(config.to_dict()) == config
    checkpoint = tmp_path / "fixed-epsilon-nf.ckpt"

    trained = train_model(
        "scale_conditioned_nf",
        train,
        validation,
        config,
        checkpoint,
    )
    assert trained.config.epsilon_min == fixed_epsilon
    assert trained.config.epsilon_max == fixed_epsilon
    assert trained.model_contract == {
        **dict(NF_DENSITY_CONTRACT),
        "epsilon_min": fixed_epsilon,
        "epsilon_max": fixed_epsilon,
    }

    loaded = load_checkpoint(checkpoint)
    assert loaded.config == config
    assert loaded.model_contract == trained.model_contract
    for name, value in trained.model.state_dict().items():
        torch.testing.assert_close(value.cpu(), loaded.model.state_dict()[name].cpu())
    prediction = predict_lid(
        loaded,
        validation[:3],
        fixed_epsilon,
        readout="fixed_likelihood",
        divergence_backend="exact",
        trace_probes=0,
    )
    assert prediction.shape == (3,)
    assert np.isfinite(prediction).all()
    likelihood = predict_nf_log_likelihood(
        loaded,
        validation[:3],
        fixed_epsilon,
        batch_size=2,
    )
    normalized = (
        torch.as_tensor(validation[:3]) - loaded.normalization_mean.reshape(1, -1)
    ) / loaded.normalization_scale
    with torch.no_grad():
        expected_likelihood = loaded.model.log_prob(normalized, fixed_epsilon).numpy()
    np.testing.assert_array_equal(likelihood, expected_likelihood)
    with pytest.raises(ValueError, match="outside.*training interval"):
        predict_lid(
            loaded,
            validation[:2],
            fixed_epsilon * 1.01,
            readout="fixed_likelihood",
            divergence_backend="exact",
            trace_probes=0,
        )
    with pytest.raises(ValueError, match="outside.*training interval"):
        predict_nf_log_likelihood(
            loaded,
            validation[:2],
            fixed_epsilon * 1.01,
        )


def test_scale_conditioned_nf_predictor_rejects_wrong_readout_contract(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(55)
    train = rng.normal(size=(12, 2)).astype(np.float32)
    result = train_model(
        "scale_conditioned_nf",
        train,
        train[:6],
        replace(_tiny_nf_config(), epochs=1),
        tmp_path / "nf.ckpt",
    )
    with pytest.raises(ValueError, match="fixed_likelihood"):
        predict_lid(
            result,
            train[:2],
            0.2,
            readout="full",
            divergence_backend="exact",
            trace_probes=0,
        )
    with pytest.raises(ValueError, match="exact autograd"):
        predict_lid(
            result,
            train[:2],
            0.2,
            readout="fixed_likelihood",
            divergence_backend="hutchinson",
        )
    with pytest.raises(ValueError, match="training interval"):
        predict_lid(
            result,
            train[:2],
            0.05,
            readout="fixed_likelihood",
            divergence_backend="exact",
            trace_probes=0,
        )


def test_scale_conditioned_nf_checkpoint_rejects_contract_tampering(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(56)
    train = rng.normal(size=(12, 2)).astype(np.float32)
    checkpoint = tmp_path / "nf.ckpt"
    train_model(
        "scale_conditioned_nf",
        train,
        train[:6],
        replace(_tiny_nf_config(), epochs=1),
        checkpoint,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["model_contract"]["readout"] = "untrusted-replacement"
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="model_contract mismatch"):
        load_checkpoint(checkpoint)


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
