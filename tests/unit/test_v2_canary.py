from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from experiments.v2_canary import (
    EXPECTED_CELL_IDS,
    EXPECTED_VARIANTS,
    PROTOCOL_ID,
    REQUIRED_GATE_IDS,
    CanaryError,
    _load_or_create_preseal_runtime,
    _load_runtime_sidecar,
    _MeasuredTrain,
    _reference_selector_quality,
    _resume_or_reset_diagnostics,
    _trace_quality,
    file_sha256,
    load_and_validate_canary_report,
    validate_canary_config,
    validate_canary_report,
)

ROOT = Path(__file__).resolve().parents[2]
SHA = "a" * 64
COMMIT = "b" * 40


def canary_config() -> dict:
    return yaml.safe_load((ROOT / "configs/v2_canary.yaml").read_text())


def valid_report(tmp_path: Path) -> dict:
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"ok":true}\n', encoding="utf-8")
    digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
    contact_sheet = tmp_path / "arrows_contact_sheet.png"
    contact_sheet.write_bytes(b"test-png-evidence")
    contact_digest = hashlib.sha256(contact_sheet.read_bytes()).hexdigest()
    preprocessing = {
        "schema_version": 1,
        "kind": "train_mean_global_rms_v1",
        "ambient_dim": 3072,
        "mean_sha256": SHA,
        "scalar_scale": 40.0,
    }
    preprocessing_digest = hashlib.sha256(
        json.dumps(preprocessing, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    devices = [
        {
            "worker_index": index,
            "logical_index": index,
            "physical_id": str(index),
            "uuid": f"GPU-{index:02d}",
            "name": "NVIDIA H100 80GB HBM3",
            "memory_total_mib": 81559.0,
        }
        for index in range(8)
    ]
    telemetry = [
        {
            "worker_index": index,
            "uuid": f"GPU-{index:02d}",
            "samples": 10,
            "mean_gpu_utilization_percent": 91.0,
            "p50_gpu_utilization_percent": 93.0,
            "p95_gpu_utilization_percent": 99.0,
            "peak_memory_mib": 50000.0,
            "memory_total_mib": 81559.0,
            "examples_per_second": 1234.5,
        }
        for index in range(8)
    ]
    cells = []
    for worker, cell_id in enumerate(EXPECTED_CELL_IDS):
        variant, suite, dataset, representation = cell_id.split("/")
        is_nf = variant == "scale_conditioned_nf"
        cells.append(
            {
                "cell_id": cell_id,
                "variant_id": variant,
                "family": {
                    "vp_diffusion": "vp_diffusion",
                    "ve_diffusion": "gaussian_diffusion",
                    "posterior_log_noise_affine_flow": "independent_affine_flow",
                    "scale_conditioned_nf": "scale_conditioned_normalizing_flow",
                }[variant],
                "suite_id": suite,
                "dataset": dataset,
                "representation": representation,
                "status": "passed",
                "worker_index": worker,
                "input_sha256": SHA,
                "partition_sha256": SHA,
                "preprocessing_sha256": SHA,
                "training_config_sha256": SHA,
                "checkpoint_sha256": SHA,
                "production_identity_sha256": SHA,
                "reusable_by_full_campaign": True,
                "production_preseal_wall_seconds": 3000.0,
                "native_loss": {"status": "passed"},
                "reconstruction": (
                    {"status": "not_applicable"} if is_nf else {"status": "passed"}
                ),
                "pointwise_lid": {"status": "passed"},
                "reference_selector": {"status": "passed"},
                "nf_scale_bin_nll": (
                    {"status": "passed"} if is_nf else {"status": "not_applicable"}
                ),
                "trace_agreement": (
                    {"status": "not_applicable"} if is_nf else {"status": "passed"}
                ),
                "artifacts": [{"path": "sealed/cell.json", "sha256": SHA}],
            }
        )
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "passed",
        "source": {
            "git_commit": COMMIT,
            "git_tree_sha256": SHA,
            "declared_source_sha256": SHA,
            "worktree_clean": True,
        },
        "config": {
            "path": "configs/v2_canary.yaml",
            "sha256": SHA,
            "batch_size": 256,
            "worker_count": 8,
            "campaign_identity": SHA,
            "campaign_config_sha256": SHA,
            "input_inventory_sha256": SHA,
        },
        "preflight": {
            "status": "passed",
            "archive_sha256": SHA,
            "arrows": {
                "status": "passed",
                "dataset": "e2_arrows",
                "layout": "NHWC_RGB",
                "shape": [100000, 32, 32, 3],
                "dtype": "|u1",
                "raw_min": 0.0,
                "raw_max": 255.0,
                "normalization_rms": 40.0,
                "preprocessing_sha256": preprocessing_digest,
                "preprocessing": preprocessing,
                "production_partition_sha256": SHA,
                "optimizer_fit_indices_sha256": SHA,
                "selection_indices_sha256": SHA,
                "visual_sample_seed": 20260906,
                "display_transform": "production_standardized_clip_minus3_plus3_v1",
                "source_dataset_sha256": SHA,
                "source_target_sha256": SHA,
                "sample_indices_sha256": SHA,
                "sample_images_sha256": SHA,
                "target_sha256": SHA,
                "pillow_version": "11.3.0",
                "contact_sheet": {
                    "path": contact_sheet.name,
                    "sha256": contact_digest,
                    "size_bytes": contact_sheet.stat().st_size,
                },
                "human_review": {
                    "status": "reviewed",
                    "reviewer": "test-reviewer",
                    "reviewed_at": "2026-09-06T00:00:00Z",
                    "reviewed_gate_sha256": SHA,
                },
            },
            "devices": devices,
        },
        "utilization_probe": {
            "status": "passed",
            "scientific_result": True,
            "batch_size": 256,
            "devices": telemetry,
        },
        "runtime_projection": {
            "status": "passed",
            "physical_cell_count": 429,
            "worker_count": 8,
            "waves": 54,
            "maximum_canary_preseal_wall_seconds": 3000.0,
            "safety_factor": 2.0,
            "projected_campaign_hours": 90.0,
            "maximum_allowed_hours": 96.0,
        },
        "cells": cells,
        "gates": [
            {"id": gate, "status": "passed", "evidence": {"checked": True}}
            for gate in sorted(REQUIRED_GATE_IDS)
        ],
        "outputs": [
            {
                "path": evidence.name,
                "sha256": digest,
                "size_bytes": evidence.stat().st_size,
            },
            {
                "path": contact_sheet.name,
                "sha256": contact_digest,
                "size_bytes": contact_sheet.stat().st_size,
            },
        ],
        "failures": [],
    }


def test_versioned_config_selects_exact_eight_production_cells() -> None:
    config = validate_canary_config(canary_config())
    assert tuple(config["training"]["variant_ids"]) == EXPECTED_VARIANTS
    assert len(EXPECTED_CELL_IDS) == 8
    assert {value.rsplit("/", 1)[-1] for value in EXPECTED_CELL_IDS} == {
        "coefficients",
        "dataset",
    }
    assert {"/".join(value.split("/")[1:]) for value in EXPECTED_CELL_IDS} == {
        "e2/e2_uniform_pca/coefficients",
        "e2/e2_arrows/dataset",
    }
    assert config["data"]["selection"]["seed"] == 0
    assert config["data"]["visual_sample_seed"] == 20260906


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("execution", "worker_count"), 7),
        (("execution", "batch_size"), 4096),
        (("training", "steps"), 31000),
        (("training", "field_hidden_sizes"), [512, 256]),
    ],
)
def test_config_rejects_contract_drift(path: tuple[str, str], value: object) -> None:
    config = canary_config()
    config[path[0]][path[1]] = value
    with pytest.raises(CanaryError):
        validate_canary_config(config)


def test_pass_attestation_checks_exact_devices_cells_and_output_hashes(
    tmp_path: Path,
) -> None:
    report = valid_report(tmp_path)
    assert (
        validate_canary_report(
            report,
            report_dir=tmp_path,
            expected_commit=COMMIT,
            expected_git_tree_sha256=SHA,
            expected_declared_source_sha256=SHA,
            expected_config_sha256=SHA,
            expected_campaign_identity=SHA,
            expected_campaign_config_sha256=SHA,
            expected_input_inventory_sha256=SHA,
        )["status"]
        == "passed"
    )

    duplicate_device = copy.deepcopy(report)
    duplicate_device["preflight"]["devices"][1]["uuid"] = "GPU-00"
    with pytest.raises(CanaryError, match="one-to-one"):
        validate_canary_report(duplicate_device, report_dir=tmp_path)

    not_reusable = copy.deepcopy(report)
    not_reusable["cells"][0]["reusable_by_full_campaign"] = False
    with pytest.raises(CanaryError, match="not reusable"):
        validate_canary_report(not_reusable, report_dir=tmp_path)

    wrong_safety_factor = copy.deepcopy(report)
    wrong_safety_factor["runtime_projection"]["safety_factor"] = 1.0
    with pytest.raises(CanaryError, match="projection contract differs"):
        validate_canary_report(wrong_safety_factor, report_dir=tmp_path)

    over_budget = copy.deepcopy(report)
    over_budget["runtime_projection"]["maximum_canary_preseal_wall_seconds"] = 3201.0
    over_budget["runtime_projection"]["projected_campaign_hours"] = (
        54 * 3201.0 * 2.0 / 3600.0
    )
    with pytest.raises(CanaryError, match="exceeds the 96-hour gate"):
        validate_canary_report(over_budget, report_dir=tmp_path)

    (tmp_path / "evidence.json").write_text('{"ok":false}\n', encoding="utf-8")
    with pytest.raises(CanaryError, match="size-mismatched|hash mismatch"):
        validate_canary_report(report, report_dir=tmp_path)


def test_old_pilot_report_cannot_be_used_as_canary_gate(tmp_path: Path) -> None:
    source = ROOT / "docs/results/vp_pilot_20260905.json"
    copied = tmp_path / "canary_report.json"
    copied.write_bytes(source.read_bytes())
    with pytest.raises(CanaryError):
        load_and_validate_canary_report(copied)


def test_loader_rejects_stale_source_binding(tmp_path: Path) -> None:
    report = valid_report(tmp_path)
    path = tmp_path / "canary_report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(CanaryError, match="commit"):
        load_and_validate_canary_report(path, expected_commit="c" * 40)


def test_diagnostics_resume_reuses_complete_or_cleans_known_partial(
    tmp_path: Path,
) -> None:
    identity = {
        "cell_id": "vp_diffusion/e2/e2_uniform_pca/coefficients",
        "selection_query_sha256": SHA,
        "trace_query_sha256": SHA,
        "candidate_scales_sha256": SHA,
        "cell_identity_sha256": SHA,
        "preprocessing_sha256": SHA,
    }
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "lid_curve.npz").write_bytes(b"interrupted")
    assert _resume_or_reset_diagnostics(partial, expected_identity=identity) is None
    assert list(partial.iterdir()) == []

    complete = tmp_path / "complete"
    complete.mkdir()
    artifacts = []
    for index in range(3):
        path = complete / f"artifact-{index}.npz"
        path.write_bytes(f"artifact-{index}".encode())
        artifacts.append(
            {
                "path": path.name,
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    record = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "passed",
        **identity,
        "artifacts": artifacts,
    }
    (complete / "diagnostics.json").write_text(json.dumps(record), encoding="utf-8")
    assert _resume_or_reset_diagnostics(complete, expected_identity=identity) == record
    (complete / "artifact-0.npz").write_bytes(b"changed")
    with pytest.raises(CanaryError, match="changed"):
        _resume_or_reset_diagnostics(complete, expected_identity=identity)


def test_partial_training_resume_cannot_attest_shortened_full_cell_wall_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import models.training

    class Sampler:
        def summarize(self, **kwargs: object) -> dict[str, object]:
            raise AssertionError("shortened partial timing must not be summarized")

    called = False

    def resumed_train(
        family: object,
        train_data: object,
        selection_data: object,
        config: object,
        checkpoint_path: object,
        callback: object,
        *,
        progress_checkpoint_path: object,
    ) -> dict[str, object]:
        nonlocal called
        called = True
        callback({"step": 31500})
        callback({"step": 32000})
        return {"config": {"validation_interval_steps": 500}}

    monkeypatch.setattr(models.training, "train_model", resumed_train)
    measured = _MeasuredTrain(Sampler(), worker_index=0)
    checkpoint = tmp_path / "checkpoint.pt"
    with pytest.raises(CanaryError, match="full 0-to-32000-step training"):
        measured(
            "vp_diffusion",
            object(),
            object(),
            object(),
            checkpoint,
            progress_checkpoint_path=tmp_path / "absent-progress.pt",
        )
    assert called is True
    assert not (tmp_path / "canary_training_runtime.json").exists()

    progress = tmp_path / "partial-progress.pt"
    progress.write_bytes(b"partial")
    called = False
    with pytest.raises(CanaryError, match="partial progress cannot attest"):
        measured(
            "vp_diffusion",
            object(),
            object(),
            object(),
            checkpoint,
            progress_checkpoint_path=progress,
        )
    assert called is False


def test_checkpoint_only_resume_preseal_includes_original_training_and_downtime(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"complete")
    assert not (tmp_path / "training_progress.pt").exists()
    processed_examples = 32000 * 256
    training_start_ns = 1_000_000_000
    training_end_ns = 5_000_000_000
    sidecar = {
        "schema_version": 1,
        "scientific_result": True,
        "batch_size": 256,
        "starting_step": 0,
        "final_step": 32000,
        "processed_examples": processed_examples,
        "training_start_time_ns": training_start_ns,
        "training_end_time_ns": training_end_ns,
        "device": {
            "worker_index": 0,
            "examples_per_second": processed_examples / 4.0,
        },
    }
    (tmp_path / "canary_training_runtime.json").write_text(
        json.dumps(sidecar), encoding="utf-8"
    )
    loaded = _load_runtime_sidecar(checkpoint, worker_index=0)
    diagnostics = tmp_path / "quality_diagnostics"
    diagnostics.mkdir()
    record = _load_or_create_preseal_runtime(
        diagnostics,
        cell_id="vp_diffusion/e2/e2_uniform_pca/coefficients",
        cell_identity_sha256=SHA,
        training_runtime=loaded,
        worker_start_ns=9_000_000_000,
        hook_end_ns=10_000_000_000,
    )
    assert record["preseal_start_time_ns"] == training_start_ns
    assert record["preseal_end_time_ns"] == 10_000_000_000
    assert record["production_preseal_wall_seconds"] == 9.0

    tampered = dict(record)
    tampered["preseal_start_time_ns"] = training_end_ns + 1
    (diagnostics / "preseal_runtime.json").write_text(
        json.dumps(tampered), encoding="utf-8"
    )
    with pytest.raises(CanaryError, match="full training interval"):
        _load_or_create_preseal_runtime(
            diagnostics,
            cell_id="vp_diffusion/e2/e2_uniform_pca/coefficients",
            cell_identity_sha256=SHA,
            training_runtime=loaded,
            worker_start_ns=9_000_000_000,
            hook_end_ns=10_000_000_000,
        )


def test_image_trace_gate_uses_common_16_64_probes_without_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import models.training

    calls: list[tuple[str, int, int]] = []

    def predict_lid(*args: object, **kwargs: object) -> np.ndarray:
        backend = str(kwargs["divergence_backend"])
        probes = int(kwargs["trace_probes"])
        seed = int(kwargs["trace_seed"])
        calls.append((backend, probes, seed))
        return np.full(4, 1.0 + probes / 6400.0)

    monkeypatch.setattr(models.training, "predict_lid", predict_lid)
    result = _trace_quality(
        object(),
        np.zeros((4, 8), dtype=np.float32),
        np.ones(4),
        variant_id="ve_diffusion",
        family="gaussian_diffusion",
        representation="dataset",
        scale=1.0,
        trace_seed=123,
        batch_size=4,
        maximum_mae=2.0,
        maximum_ratio=1.25,
        output_dir=tmp_path,
    )
    assert result["status"] == "passed"
    assert result["comparison"] == "common_probe_hutchinson16_vs64_no_exact_claim"
    assert calls == [("hutchinson", 16, 123), ("hutchinson", 64, 123)]
    assert result["hutchinson16_mae_vs_exact"] is None


def test_reference_selector_failure_is_a_gate_not_a_type_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.global_campaign_v2
    import models.training

    monkeypatch.setattr(
        experiments.global_campaign_v2,
        "unknown_reference_lambdas",
        lambda: np.asarray([0.5, 1.0, 2.0]),
    )
    monkeypatch.setattr(
        experiments.global_campaign_v2,
        "select_unknown_reference_kneedle",
        lambda scales, curve: (
            None,
            {"status": "selection_failed", "failure_reason": "no_knee"},
        ),
    )
    monkeypatch.setattr(
        models.training,
        "predict_lid",
        lambda *args, **kwargs: np.ones(4, dtype=np.float64),
    )
    result = _reference_selector_quality(
        object(),
        np.zeros((4, 3), dtype=np.float32),
        variant_id="ve_diffusion",
        family="gaussian_diffusion",
        representation="coefficients",
        trace_seed=9,
        batch_size=4,
        output_dir=tmp_path,
    )
    assert result["status"] == "failed"
    assert result["failure_reason"] == "no_knee"
