"""Completed-run finalization and weight integrity, with no optimizer updates."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments import nf_ablation
from experiments.run_manifest import sha256_path
from models import training


def test_finalizing_completed_progress_retains_weights_and_resume_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = training.TrainingConfig(
        device="cpu", hidden_dim=8, depth=1, time_embedding_dim=4,
        fourier_features=2, sigma_min=0.01, sigma_max=2.0,
        training_mode="fixed_steps_v1", steps=2, warmup_steps=0,
        validation_interval_steps=1, batch_size=4, early_stopping_patience=None,
    )
    checkpoint = tmp_path / "model.pt"
    progress = tmp_path / "progress.pt"
    torch.save({"marker": "completed resume fixture"}, progress)
    original = progress.read_bytes()

    def completed(path, **kwargs):
        assert path == progress
        loss = kwargs["expected_initial_validation_loss"]
        row = training.StepMetrics(
            step=2, examples_seen=8, train_loss=loss,
            validation_loss=loss, learning_rate=cfg.learning_rate,
        )
        return 3, [row], 2, loss, loss, training._cpu_state_dict(kwargs["model"])

    def forbidden(*args, **kwargs):
        raise AssertionError("optimizer updates are forbidden")

    monkeypatch.setattr(training, "_load_fixed_training_progress", completed)
    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)
    data = np.arange(24, dtype=np.float32).reshape(12, 2) / 10
    result = training.train_model(
        "gaussian_diffusion", data[:8], data[8:], cfg, checkpoint,
        progress_checkpoint_path=progress,
    )
    assert checkpoint.is_file() and progress.read_bytes() == original
    restored = training.load_checkpoint(checkpoint, device="cpu")
    for name, value in result.model.state_dict().items():
        torch.testing.assert_close(
            value, restored.model.state_dict()[name], atol=0, rtol=0
        )


def test_nf_retained_weight_integrity_and_legacy_records(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"saved evaluation weights")
    record = {"retention": "retain", "checkpoint_sha256": sha256_path(checkpoint)}
    attestation = tmp_path / "training_attestation.json"
    attestation.write_text(json.dumps(record))
    assert not nf_ablation._retained_checkpoint_errors(tmp_path)
    checkpoint.write_bytes(b"changed weights")
    assert "retained NF checkpoint SHA differs" in (
        nf_ablation._retained_checkpoint_errors(tmp_path)
    )
    checkpoint.unlink()
    assert "retained NF checkpoint is missing" in (
        nf_ablation._retained_checkpoint_errors(tmp_path)
    )
    record["retention"] = "pruned_after_inline_evaluation"
    attestation.write_text(json.dumps(record))
    assert not nf_ablation._retained_checkpoint_errors(tmp_path)
