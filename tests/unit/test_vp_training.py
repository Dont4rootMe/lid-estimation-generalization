from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import models.training as training_module
from models.training import (
    StepMetrics,
    TrainingConfig,
    evaluate_native_loss,
    field_parameter_count,
    load_checkpoint,
    predict_lid,
    predict_primitives,
    train_model,
)
from models.vp_baseline import ConditionedBottleneckMLP, VPBottleneckMLP, VPSchedule


def _vp_config(*, seed: int = 37) -> TrainingConfig:
    return TrainingConfig(
        seed=seed,
        device="cpu",
        training_mode="fixed_steps_v1",
        steps=6,
        warmup_steps=2,
        validation_interval_steps=2,
        batch_size=4,
        learning_rate=1.0e-3,
        weight_decay=0.01,
        time_embedding_dim=8,
        early_stopping_patience=None,
        gradient_clip_norm=None,
        deterministic=True,
        vp_hidden_sizes=(16, 8, 4),
        vp_beta_min=0.1,
        vp_beta_max=20.0,
    )


def _data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(81)
    return (
        rng.normal(size=(24, 3)).astype(np.float32),
        rng.normal(size=(8, 3)).astype(np.float32),
    )


def test_vp_fixed_steps_checkpoint_exposes_best_final_and_native_readout(
    tmp_path: Path,
) -> None:
    train, validation = _data()
    result = train_model(
        "vp_diffusion", train, validation, _vp_config(), tmp_path / "vp.ckpt"
    )

    assert result.family == "vp_diffusion"
    assert isinstance(result.model, VPBottleneckMLP)
    assert all(isinstance(metric, StepMetrics) for metric in result.history)
    assert [metric.step for metric in result.history] == [2, 4, 6]
    assert [metric.examples_seen for metric in result.history] == [8, 16, 24]
    assert result.metrics["steps_completed"] == 6
    assert result.metrics["examples_seen_total"] == 24
    assert result.final_model_state is not None
    assert result.weights_metadata["initial"]["step"] == 0
    assert result.weights_metadata["initial"]["examples_seen"] == 0
    assert np.isfinite(result.weights_metadata["initial"]["validation_loss"])
    assert result.weights_metadata["selected"]["step"] == result.best_epoch
    assert result.weights_metadata["final"]["step"] == 6
    assert result.weights_metadata["final"]["examples_seen"] == 24
    assert len(result.weights_metadata["selected"]["state_sha256"]) == 64
    assert len(result.weights_metadata["final"]["state_sha256"]) == 64

    loaded = load_checkpoint(tmp_path / "vp.ckpt")
    assert loaded.history == result.history
    assert loaded.weights_metadata == result.weights_metadata
    assert loaded.final_model_state is not None
    for name, value in result.model.state_dict().items():
        torch.testing.assert_close(value, loaded.model.state_dict()[name])
    for name, value in result.final_model_state.items():
        torch.testing.assert_close(value, loaded.final_model_state[name])

    loss = evaluate_native_loss(loaded, validation, seed=901)
    assert np.isfinite(loss)
    lid = predict_lid(
        loaded,
        validation[:3],
        0.5,
        divergence_backend="exact",
        batch_size=2,
    )
    assert lid.shape == (3,)
    assert np.isfinite(lid).all()
    primitives = predict_primitives(
        loaded,
        validation[:3],
        0.5,
        divergence_backend="exact",
        batch_size=2,
    )
    schedule = VPSchedule(0.1, 20.0)
    time = torch.tensor(schedule.time_for_lambda(0.5))
    alpha, sigma = schedule.coefficients(time)
    normalized = (
        torch.as_tensor(validation[:3]) - loaded.normalization_mean
    ) / loaded.normalization_scale
    np.testing.assert_allclose(
        primitives.evaluation_point,
        (alpha * normalized).numpy(),
        rtol=1e-6,
        atol=1e-6,
    )
    assert primitives.condition == pytest.approx(float(sigma))
    with pytest.raises(ValueError, match="beyond"):
        predict_lid(loaded, validation[:1], 1000, divergence_backend="exact")


def test_vp_fixed_step_resume_is_identical_to_uninterrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train, validation = _data()
    config = _vp_config(seed=41)
    uninterrupted = train_model(
        "vp_diffusion",
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
            "vp_diffusion",
            train,
            validation,
            config,
            checkpoint,
            progress_checkpoint_path=progress,
        )
    payload = torch.load(progress, map_location="cpu", weights_only=True)
    assert payload["schema_version"] == 2
    assert payload["global_step"] == 2
    assert payload["scheduler_state"]["last_completed_step"] == 2
    assert payload["sampler"]["kind"] == "uniform_with_replacement_v1"
    assert payload["counters"] == {"examples_seen_total": 8}
    assert np.isfinite(payload["initial_validation_loss"])

    monkeypatch.setattr(training_module, "_atomic_torch_save", original_save)
    resumed = train_model(
        "vp_diffusion",
        train,
        validation,
        config,
        checkpoint,
        progress_checkpoint_path=progress,
    )
    assert progress.is_file()
    assert resumed.history == uninterrupted.history
    assert resumed.best_epoch == uninterrupted.best_epoch
    assert resumed.weights_metadata == uninterrupted.weights_metadata
    for name, value in uninterrupted.model.state_dict().items():
        torch.testing.assert_close(value, resumed.model.state_dict()[name])
    assert uninterrupted.final_model_state is not None
    assert resumed.final_model_state is not None
    for name, value in uninterrupted.final_model_state.items():
        torch.testing.assert_close(value, resumed.final_model_state[name])


def test_shared_bottleneck_has_declared_exact_capacity() -> None:
    widths = (1024, 512, 256, 256, 128, 128)
    vp = _vp_config()
    vp = TrainingConfig.from_mapping(
        {**vp.to_dict(), "vp_hidden_sizes": widths, "time_embedding_dim": 128}
    )
    assert field_parameter_count("vp_diffusion", vp, 30) == 3_200_034

    ve = TrainingConfig(
        depth=None,
        time_embedding_dim=128,
        field_hidden_sizes=widths,
        sigma_min=1 / 256,
        sigma_max=64,
    )
    assert field_parameter_count("gaussian_diffusion", ve, 30) == 3_200_034


@pytest.mark.parametrize(
    ("family", "native"),
    [
        ("gaussian_diffusion", {"sigma_min": 0.1, "sigma_max": 0.5}),
        ("rectified_flow", {"time_min": 0.1, "time_max": 0.9}),
    ],
)
def test_non_vp_fixed_step_vector_fields_use_shared_bottleneck(
    family: str, native: dict[str, float], tmp_path: Path
) -> None:
    train, validation = _data()
    config = TrainingConfig(
        seed=51,
        device="cpu",
        training_mode="fixed_steps_v1",
        steps=2,
        warmup_steps=0,
        validation_interval_steps=1,
        batch_size=4,
        learning_rate=1.0e-3,
        weight_decay=0.0,
        time_embedding_dim=8,
        early_stopping_patience=None,
        gradient_clip_norm=1.0,
        field_hidden_sizes=(16, 8, 4),
        **native,
    )
    trained = train_model(
        family, train, validation, config, tmp_path / f"{family}.ckpt"
    )
    assert isinstance(trained.model, ConditionedBottleneckMLP)
    assert trained.model.config.condition_transform == (
        "log" if family == "gaussian_diffusion" else "linear"
    )
    assert all(
        metric.learning_rate == config.learning_rate for metric in trained.history
    )
    assert field_parameter_count(family, config, 3) == sum(
        parameter.numel() for parameter in trained.model.parameters()
    )
    loaded = load_checkpoint(tmp_path / f"{family}.ckpt")
    assert loaded.history == trained.history
    assert loaded.weights_metadata == trained.weights_metadata


def test_fixed_step_nf_keeps_native_architecture_and_both_states(
    tmp_path: Path,
) -> None:
    train, validation = _data()
    config = TrainingConfig(
        seed=53,
        device="cpu",
        training_mode="fixed_steps_v1",
        steps=2,
        warmup_steps=0,
        validation_interval_steps=1,
        batch_size=4,
        learning_rate=1.0e-3,
        weight_decay=0.0,
        hidden_dim=16,
        depth=None,
        time_embedding_dim=8,
        early_stopping_patience=None,
        gradient_clip_norm=1.0,
        fourier_features=4,
        max_condition_frequency=10.0,
        num_coupling_layers=2,
        conditioner_depth=1,
        log_scale_limit=1.25,
        epsilon_min=0.1,
        epsilon_max=0.5,
    )
    trained = train_model(
        "scale_conditioned_nf",
        train,
        validation,
        config,
        tmp_path / "nf.ckpt",
    )
    assert trained.final_model_state is not None
    assert trained.metrics["examples_seen_total"] == 8
    loaded = load_checkpoint(tmp_path / "nf.ckpt")
    assert loaded.final_model_state is not None
    assert loaded.weights_metadata == trained.weights_metadata


def test_vp_requires_explicit_fixed_step_contract() -> None:
    with pytest.raises(ValueError, match="complete block"):
        TrainingConfig(vp_beta_min=0.1)
    with pytest.raises(ValueError, match="fixed_steps_v1"):
        train_model(
            "vp_diffusion",
            *_data(),
            TrainingConfig(
                depth=None,
                time_embedding_dim=8,
                early_stopping_patience=None,
                gradient_clip_norm=None,
                vp_hidden_sizes=(16, 8, 4),
                vp_beta_min=0.1,
                vp_beta_max=20.0,
            ),
            Path("unused.ckpt"),
        )
