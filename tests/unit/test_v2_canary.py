from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from experiments.v2_canary import (
    EXPECTED_ALL_CELL_IDS,
    EXPECTED_CELL_IDS,
    EXPECTED_COMPANION_CELL_IDS,
    EXPECTED_LANES_PER_DEVICE,
    EXPECTED_PROCESS_COUNT,
    EXPECTED_PROCESS_DEVICE_INDICES,
    EXPECTED_VARIANTS,
    PROTOCOL_ID,
    REQUIRED_GATE_IDS,
    SCHEMA_VERSION,
    CanaryDiagnostics,
    CanaryError,
    _assert_fresh_physical_canary_cells,
    _diagnostic_subset_seed,
    _lid_quality,
    _load_or_create_preseal_runtime,
    _load_runtime_sidecar,
    _MeasuredTrain,
    _reconstruction_quality,
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


@pytest.mark.parametrize("tail", [[8.0, 6.0, 4.0], [4.0, 4.0, 4.0]])
def test_improving_loss_is_reported_without_failing_campaign(tail: list[float]) -> None:
    from experiments.v2_canary import _history_quality

    trained = {
        "history": [{"validation_loss": value} for value in [9.0, *tail]],
        "weights_metadata": {
            "initial": {"step": 0, "examples_seen": 0, "validation_loss": 10.0}
        },
    }
    result = _history_quality(trained, canary_config()["gates"])
    assert result["status"] == "diagnostic_only"
    assert result["plateau_gate_applied"] is False
    assert result["convergence_status"] == (
        "still_improving_at_budget" if tail[0] > tail[-1] else "plateau_detected"
    )


def test_native_loss_records_nonlearning_but_rejects_nonfinite() -> None:
    from experiments.v2_canary import _history_quality

    trained = {
        "history": [{"validation_loss": 10.0}] * 4,
        "weights_metadata": {
            "initial": {"step": 0, "examples_seen": 0, "validation_loss": 10.0}
        },
    }
    result = _history_quality(trained, canary_config()["gates"])
    assert result["status"] == "diagnostic_only"
    assert result["quality_gate_applied"] is False
    assert result["historical_threshold_met"] is False
    trained["history"][-1] = {"validation_loss": float("nan")}
    with pytest.raises(CanaryError, match="non-finite"):
        _history_quality(trained, canary_config()["gates"])


@pytest.mark.parametrize("initial_loss", [0.0, -1.0, 1.0])
def test_native_loss_finite_zero_or_signed_initial_is_diagnostic(
    initial_loss: float,
) -> None:
    from experiments.v2_canary import _history_quality

    result = _history_quality(
        {
            "history": [{"validation_loss": 10.0}] * 4,
            "weights_metadata": {
                "initial": {
                    "step": 0,
                    "examples_seen": 0,
                    "validation_loss": initial_loss,
                }
            },
        },
        canary_config()["gates"],
    )
    assert result["status"] == "diagnostic_only"
    assert np.isfinite(result["best_to_initial_ratio"])


@pytest.mark.parametrize("nonfinite", [False, True])
def test_nf_losing_every_bin_is_recorded_but_nonfinite_is_rejected(
    tmp_path: Path, nonfinite: bool
) -> None:
    import torch

    from experiments.v2_canary import _nf_scale_bin_quality

    class PoorNF(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def log_prob(self, query, epsilon):
            return torch.full((len(query),), float("nan") if nonfinite else -1e6)

    query = np.ones((8, 3), dtype=np.float32)
    trained = {
        "model": PoorNF(),
        "normalization_mean": np.zeros(3),
        "normalization_scale": 1.0,
    }

    def run():
        return _nf_scale_bin_quality(
            trained,
            query,
            query,
            bins=canary_config()["evaluation"]["nf_scale_bins"],
            seed=7,
            minimum_improved_fraction=0.5,
            output_dir=tmp_path,
        )

    if nonfinite:
        with pytest.raises(CanaryError, match="non-finite"):
            run()
    else:
        result = run()
        assert result["status"] == "diagnostic_only"
        assert result["quality_gate_applied"] is False
        assert result["historical_threshold_met"] is False
        assert result["improved_bin_fraction"] == 0.0


def test_poor_reconstruction_does_not_stop_campaign(tmp_path: Path) -> None:
    import torch

    class PoorDenoiser(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def forward(self, query, condition):
            return torch.full_like(query, 100.0)

    result = _reconstruction_quality(
        {
            "model": PoorDenoiser(),
            "normalization_mean": np.zeros(3),
            "normalization_scale": 1.0,
            "config": {},
        },
        np.ones((8, 3), dtype=np.float32),
        variant_id="ve_diffusion",
        noise_ratio=0.5,
        seed=7,
        maximum_ratio=0.9,
        output_dir=tmp_path,
    )
    assert result["status"] == "diagnostic_only"
    assert result["historical_threshold_met"] is False
    assert result["reconstruction_mse"] == 9801.0


def valid_report(tmp_path: Path, *, weak_models: bool = False) -> dict:
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"ok":true}\n', encoding="utf-8")
    digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
    evidence_record = {
        "path": evidence.name,
        "sha256": digest,
        "size_bytes": evidence.stat().st_size,
    }
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
    cell_outputs = []
    for worker, cell_id in enumerate(EXPECTED_CELL_IDS):
        variant, suite, dataset, representation = cell_id.split("/")
        is_nf = variant == "scale_conditioned_nf"
        is_low_dimensional = representation == "coefficients"
        backend = "exact" if is_low_dimensional or is_nf else "hutchinson"
        probes = 0 if backend == "exact" else 16
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
                "production_selected_lambda": 1.0,
                "reusable_by_full_campaign": True,
                "production_preseal_wall_seconds": 3000.0,
                "native_loss": {
                    "status": "diagnostic_only",
                    "quality_gate_applied": False,
                },
                "reconstruction": (
                    {"status": "not_applicable"}
                    if is_nf
                    else {"status": "diagnostic_only", "quality_gate_applied": False}
                ),
                "pointwise_lid": {
                    "status": "diagnostic_only",
                    "assessment_scope": (
                        "diagnostic_low_dim_benchmark_outcome"
                        if is_low_dimensional
                        else "diagnostic_high_dim_benchmark_outcome"
                    ),
                    "accuracy_gate_applied": False,
                    "maximum_pointwise_mae": 5.0 if is_low_dimensional else None,
                    "selection_role": "canary_oracle_envelope_not_production_selector",
                    "query_sha256": SHA,
                    "target_sha256": SHA,
                    "query_count": 128 if is_low_dimensional else 32,
                    "divergence_backend": backend,
                    "trace_probes": probes,
                    "selected_index": 1,
                    "selected_lambda": 1.0,
                    "selected_pointwise_mae": 1.0 if is_low_dimensional else 23.4259,
                    "boundary_selected": False,
                    "finite_fraction": 1.0,
                    "arrays": evidence_record,
                },
                "reference_selector": {
                    "status": "passed",
                    "protocol_id": "held_out_reference_mean_kneedle_v2",
                    "selection_uses_lid_targets": False,
                    "query_sha256": SHA,
                    "query_count": 128 if is_low_dimensional else 32,
                    "divergence_backend": backend,
                    "trace_probes": probes,
                    "scales_sha256": SHA,
                    "mean_curve_sha256": SHA,
                    "selected_index": 1,
                    "selected_lambda": 1.0,
                    "diagnostics": {"status": "selected"},
                    "arrays": evidence_record,
                },
                "nf_scale_bin_nll": (
                    {"status": "diagnostic_only", "quality_gate_applied": False}
                    if is_nf
                    else {"status": "not_applicable"}
                ),
                "trace_agreement": (
                    {"status": "not_applicable"}
                    if is_nf
                    else {
                        "status": ("diagnostic_only"),
                        "assessment_scope": (
                            "diagnostic_low_dim_exact_comparison"
                            if is_low_dimensional
                            else "diagnostic_high_dim_no_exact_claim"
                        ),
                        "accuracy_gate_applied": False,
                        "maximum_trace64_mae": 2.0 if is_low_dimensional else None,
                        "maximum_trace64_to_trace16_ratio": (
                            1.25 if is_low_dimensional else None
                        ),
                        "query_sha256": SHA,
                        "target_sha256": SHA,
                        "query_count": 4,
                        "lambda": 1.0,
                        "trace_seed": 123,
                        "probe_prefix_shared": True,
                        "single_batch_prefix_verified": True,
                        "comparison": (
                            "exact_vs_hutchinson16_64"
                            if is_low_dimensional
                            else "common_probe_hutchinson16_vs64_no_exact_claim"
                        ),
                        "hutchinson16_mae_vs_exact": (
                            1.5 if is_low_dimensional else None
                        ),
                        "hutchinson64_mae_vs_exact": (
                            1.0 if is_low_dimensional else None
                        ),
                        "trace64_to_trace16_mae_ratio": (
                            2.0 / 3.0 if is_low_dimensional else None
                        ),
                        "hutchinson64_mae_vs_hutchinson16": 2.53735,
                        "arrays": evidence_record,
                    }
                ),
                "artifacts": [evidence_record],
            }
        )
        cell = cells[-1]
        if weak_models:
            cell["native_loss"]["historical_threshold_met"] = False
            cell["pointwise_lid"]["selected_pointwise_mae"] = 1e6
            cell["pointwise_lid"]["boundary_selected"] = True
            if is_nf:
                cell["nf_scale_bin_nll"]["improved_bin_fraction"] = 0.2
                cell["nf_scale_bin_nll"]["historical_threshold_met"] = False
            else:
                cell["reconstruction"]["ratio_to_better_trivial"] = 100.0
                cell["reconstruction"]["historical_threshold_met"] = False
                if is_low_dimensional:
                    cell["trace_agreement"]["hutchinson64_mae_vs_exact"] = 1e6
                    cell["trace_agreement"]["trace64_to_trace16_mae_ratio"] = 1e6 / 1.5
        quality = {
            "production_selected_lambda": cell["production_selected_lambda"],
            "native_loss": cell["native_loss"],
            "reconstruction": cell["reconstruction"],
            "pointwise_lid": cell["pointwise_lid"],
            "reference_selector": cell["reference_selector"],
            "nf_scale_bin_nll": cell["nf_scale_bin_nll"],
            "trace_agreement": cell["trace_agreement"],
        }
        summary = {
            "model_variant": variant,
            "suite_id": suite,
            "dataset": dataset,
            "representation": representation,
            "selected_scale": cell["production_selected_lambda"],
            "quality_diagnostics": quality,
        }
        cell_dir = tmp_path / f"cell-{worker}"
        cell_dir.mkdir()
        for filename, payload in (
            ("summary.json", summary),
            ("quality_diagnostics.json", quality),
        ):
            path = cell_dir / filename
            path.write_text(json.dumps(payload), encoding="utf-8")
            record = {
                "path": path.relative_to(tmp_path).as_posix(),
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
            cell["artifacts"].append(record)
            cell_outputs.append(record)
    companion_cells = []
    companion_outputs = []
    for process_index, cell_id in enumerate(
        EXPECTED_COMPANION_CELL_IDS, start=len(EXPECTED_CELL_IDS)
    ):
        variant, suite, dataset, representation = cell_id.split("/")
        artifact = tmp_path / f"companion-{process_index}.json"
        artifact.write_text(json.dumps({"cell_id": cell_id}), encoding="utf-8")
        artifact_record = {
            "path": artifact.name,
            "sha256": file_sha256(artifact),
            "size_bytes": artifact.stat().st_size,
        }
        companion_outputs.append(artifact_record)
        companion_cells.append(
            {
                "cell_id": cell_id,
                "variant_id": variant,
                "suite_id": suite,
                "dataset": dataset,
                "representation": representation,
                "status": "passed",
                "process_index": process_index,
                "physical_device_index": EXPECTED_PROCESS_DEVICE_INDICES[process_index],
                "input_sha256": SHA,
                "partition_sha256": SHA,
                "training_config_sha256": SHA,
                "checkpoint_sha256": SHA,
                "production_identity_sha256": SHA,
                "reusable_by_full_campaign": True,
                "production_wall_seconds": 3000.0,
                "artifacts": [artifact_record],
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
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
            "physical_device_count": 8,
            "lanes_per_device": EXPECTED_LANES_PER_DEVICE,
            "logical_worker_count": EXPECTED_PROCESS_COUNT,
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
            "physical_device_count": 8,
            "lanes_per_device": EXPECTED_LANES_PER_DEVICE,
            "logical_worker_count": EXPECTED_PROCESS_COUNT,
            "devices": telemetry,
            "companion_cells": companion_cells,
        },
        "runtime_projection": {
            "status": "passed",
            "physical_cell_count": 429,
            "logical_worker_count": EXPECTED_PROCESS_COUNT,
            "physical_device_count": 8,
            "lanes_per_device": EXPECTED_LANES_PER_DEVICE,
            "waves": 18,
            "maximum_observed_cell_wall_seconds": 3000.0,
            "safety_factor": 2.0,
            "projected_campaign_hours": 30.0,
            "maximum_allowed_hours": 96.0,
        },
        "cells": cells,
        "gates": [
            {"id": gate, "status": "passed", "evidence": {"checked": True}}
            for gate in sorted(REQUIRED_GATE_IDS)
        ],
        "outputs": [
            evidence_record,
            {
                "path": contact_sheet.name,
                "sha256": contact_digest,
                "size_bytes": contact_sheet.stat().st_size,
            },
            *cell_outputs,
            *companion_outputs,
        ],
        "failures": [],
    }


def test_versioned_config_selects_quality_and_multiplexing_cells() -> None:
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
    assert config["execution"]["lanes_per_device"] == 3
    assert len(EXPECTED_COMPANION_CELL_IDS) == 16
    assert EXPECTED_PROCESS_COUNT == 24
    assert len(set(EXPECTED_CELL_IDS + EXPECTED_COMPANION_CELL_IDS)) == 24
    assert np.bincount(EXPECTED_PROCESS_DEVICE_INDICES, minlength=8).tolist() == [3] * 8
    low_seeds = {
        _diagnostic_subset_seed(cell_id, 17)
        for cell_id in EXPECTED_CELL_IDS
        if cell_id.endswith("/coefficients")
    }
    image_seeds = {
        _diagnostic_subset_seed(cell_id, 17)
        for cell_id in EXPECTED_CELL_IDS
        if cell_id.endswith("/dataset")
    }
    assert low_seeds == {17}
    assert image_seeds == {1026}


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("execution", "physical_device_count"), 7),
        (("execution", "lanes_per_device"), 2),
        (("execution", "logical_worker_count"), 23),
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


def _freshness_fixture() -> SimpleNamespace:
    cell_keys = tuple(
        dict.fromkeys(
            "/".join(cell_id.split("/")[1:]) for cell_id in EXPECTED_ALL_CELL_IDS
        )
    )
    return SimpleNamespace(
        plans=tuple(
            SimpleNamespace(variant_id=variant)
            for variant in reversed(EXPECTED_VARIANTS)
        ),
        cells=tuple(SimpleNamespace(key=key) for key in reversed(cell_keys)),
    )


def test_freshness_preflight_checks_exact_24_physical_cell_path_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.global_campaign_v2

    prepared = _freshness_fixture()
    calls: list[tuple[str, str]] = []

    def expected_cell_directory(
        *, campaign_root: Path, prepared: object, model_index: int, cell_index: int
    ) -> tuple[Path, dict]:
        variant = prepared.plans[model_index].variant_id
        cell_key = prepared.cells[cell_index].key
        calls.append((variant, cell_key))
        label = cell_key.replace("/", "__")
        return campaign_root / "runs" / variant / label, {}

    monkeypatch.setattr(
        experiments.global_campaign_v2,
        "_expected_cell_directory",
        expected_cell_directory,
    )
    _assert_fresh_physical_canary_cells(
        campaign_root=tmp_path,
        prepared=prepared,
    )

    expected = {
        (cell_id.split("/", 1)[0], cell_id.split("/", 1)[1])
        for cell_id in EXPECTED_ALL_CELL_IDS
    }
    assert len(calls) == EXPECTED_PROCESS_COUNT == 24
    assert set(calls) == expected


@pytest.mark.parametrize("existing_kind", ["final", "incomplete"])
def test_freshness_preflight_requires_new_root_for_existing_physical_cell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_kind: str,
) -> None:
    import experiments.global_campaign_v2

    prepared = _freshness_fixture()

    def expected_cell_directory(
        *, campaign_root: Path, prepared: object, model_index: int, cell_index: int
    ) -> tuple[Path, dict]:
        variant = prepared.plans[model_index].variant_id
        cell_key = prepared.cells[cell_index].key
        label = cell_key.replace("/", "__")
        return campaign_root / "runs" / variant / label, {}

    monkeypatch.setattr(
        experiments.global_campaign_v2,
        "_expected_cell_directory",
        expected_cell_directory,
    )
    first_id = EXPECTED_ALL_CELL_IDS[0]
    variant, cell_key = first_id.split("/", 1)
    model_index = next(
        index for index, plan in enumerate(prepared.plans) if plan.variant_id == variant
    )
    cell_index = next(
        index for index, cell in enumerate(prepared.cells) if cell.key == cell_key
    )
    final_dir, _ = expected_cell_directory(
        campaign_root=tmp_path,
        prepared=prepared,
        model_index=model_index,
        cell_index=cell_index,
    )
    existing = (
        final_dir
        if existing_kind == "final"
        else final_dir.with_name(f".{final_dir.name}.incomplete")
    )
    existing.mkdir(parents=True)

    with pytest.raises(CanaryError, match="fresh output root.*timing/telemetry"):
        _assert_fresh_physical_canary_cells(
            campaign_root=tmp_path,
            prepared=prepared,
        )


def test_report_accepts_weak_models_low_utilization_and_long_runtime(
    tmp_path: Path,
) -> None:
    report = valid_report(tmp_path, weak_models=True)
    for device in report["utilization_probe"]["devices"]:
        device["mean_gpu_utilization_percent"] = 5.0
        device["p50_gpu_utilization_percent"] = 0.0
        device["peak_memory_mib"] = device["memory_total_mib"]
    report["cells"][0]["production_preseal_wall_seconds"] = 10000.0
    report["runtime_projection"]["maximum_observed_cell_wall_seconds"] = 10000.0
    report["runtime_projection"]["projected_campaign_hours"] = 100.0
    assert validate_canary_report(report, report_dir=tmp_path)["status"] == "passed"


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

    wrong_companion_binding = copy.deepcopy(report)
    wrong_companion_binding["utilization_probe"]["companion_cells"][0][
        "physical_device_index"
    ] = 7
    with pytest.raises(CanaryError, match="companion identity"):
        validate_canary_report(wrong_companion_binding, report_dir=tmp_path)

    wrong_safety_factor = copy.deepcopy(report)
    wrong_safety_factor["runtime_projection"]["safety_factor"] = 1.0
    with pytest.raises(CanaryError, match="projection contract differs"):
        validate_canary_report(wrong_safety_factor, report_dir=tmp_path)

    over_budget = copy.deepcopy(report)
    over_budget["runtime_projection"]["maximum_observed_cell_wall_seconds"] = 9601.0
    over_budget["runtime_projection"]["projected_campaign_hours"] = (
        18 * 9601.0 * 2.0 / 3600.0
    )
    with pytest.raises(CanaryError, match="not derived from canary cells"):
        validate_canary_report(over_budget, report_dir=tmp_path)

    high_dimensional_quality_claim = copy.deepcopy(report)
    image_cell = next(
        cell
        for cell in high_dimensional_quality_claim["cells"]
        if cell["dataset"] == "e2_arrows"
    )
    image_cell["pointwise_lid"]["status"] = "passed"
    with pytest.raises(CanaryError, match="pointwise LID assessment"):
        validate_canary_report(high_dimensional_quality_claim, report_dir=tmp_path)

    missing_trace_diagnostic = copy.deepcopy(report)
    image_field_cell = next(
        cell
        for cell in missing_trace_diagnostic["cells"]
        if cell["dataset"] == "e2_arrows"
        and cell["variant_id"] != "scale_conditioned_nf"
    )
    image_field_cell["trace_agreement"]["status"] = "passed"
    with pytest.raises(CanaryError, match="trace assessment"):
        validate_canary_report(missing_trace_diagnostic, report_dir=tmp_path)

    false_low_dimensional_mae = copy.deepcopy(report)
    coefficient_cell = next(
        cell
        for cell in false_low_dimensional_mae["cells"]
        if cell["representation"] == "coefficients"
    )
    coefficient_cell["pointwise_lid"]["selected_pointwise_mae"] = 1.0e99
    with pytest.raises(CanaryError, match="differs from retained evidence"):
        validate_canary_report(false_low_dimensional_mae, report_dir=tmp_path)

    false_low_dimensional_trace = copy.deepcopy(report)
    coefficient_field_cell = next(
        cell
        for cell in false_low_dimensional_trace["cells"]
        if cell["representation"] == "coefficients"
        and cell["variant_id"] != "scale_conditioned_nf"
    )
    coefficient_field_cell["trace_agreement"]["hutchinson64_mae_vs_exact"] = 1.0e99
    with pytest.raises(CanaryError, match="differs from retained evidence"):
        validate_canary_report(false_low_dimensional_trace, report_dir=tmp_path)

    nonfinite_high_dimensional_trace = copy.deepcopy(report)
    image_field_cell = next(
        cell
        for cell in nonfinite_high_dimensional_trace["cells"]
        if cell["dataset"] == "e2_arrows"
        and cell["variant_id"] != "scale_conditioned_nf"
    )
    image_field_cell["trace_agreement"]["hutchinson64_mae_vs_hutchinson16"] = float(
        "nan"
    )
    with pytest.raises(CanaryError, match="must be finite"):
        validate_canary_report(nonfinite_high_dimensional_trace, report_dir=tmp_path)

    unbound_trace_scale = copy.deepcopy(report)
    image_field_cell = next(
        cell
        for cell in unbound_trace_scale["cells"]
        if cell["dataset"] == "e2_arrows"
        and cell["variant_id"] != "scale_conditioned_nf"
    )
    image_field_cell["trace_agreement"]["lambda"] = 2.0
    with pytest.raises(CanaryError, match="trace assessment contract"):
        validate_canary_report(unbound_trace_scale, report_dir=tmp_path)

    coordinated_report_scale_tamper = copy.deepcopy(report)
    image_field_cell = next(
        cell
        for cell in coordinated_report_scale_tamper["cells"]
        if cell["dataset"] == "e2_arrows"
        and cell["variant_id"] != "scale_conditioned_nf"
    )
    image_field_cell["production_selected_lambda"] = 2.0
    image_field_cell["trace_agreement"]["lambda"] = 2.0
    with pytest.raises(CanaryError, match="retained production summary"):
        validate_canary_report(coordinated_report_scale_tamper, report_dir=tmp_path)

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
        callback({"step": 127500})
        callback({"step": 128000})
        return {"config": {"validation_interval_steps": 500}}

    monkeypatch.setattr(models.training, "train_model", resumed_train)
    measured = _MeasuredTrain(Sampler(), worker_index=0)
    checkpoint = tmp_path / "checkpoint.pt"
    with pytest.raises(CanaryError, match="full 0-to-128000-step training"):
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
    processed_examples = 128000 * 256
    training_start_ns = 1_000_000_000
    training_end_ns = 5_000_000_000
    sidecar = {
        "schema_version": 1,
        "scientific_result": True,
        "batch_size": 256,
        "starting_step": 0,
        "final_step": 128000,
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
        return np.full(4, 1.0 if probes == 16 else 3.53735)

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
        maximum_mae=None,
        maximum_ratio=None,
        output_dir=tmp_path,
    )
    assert result["status"] == "diagnostic_only"
    assert result["accuracy_gate_applied"] is False
    assert result["comparison"] == "common_probe_hutchinson16_vs64_no_exact_claim"
    assert calls == [("hutchinson", 16, 123), ("hutchinson", 64, 123)]
    assert result["hutchinson16_mae_vs_exact"] is None
    assert result["hutchinson64_mae_vs_hutchinson16"] == pytest.approx(2.53735)


def test_low_dimensional_exact_trace_accuracy_is_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import models.training

    def predict_lid(*args: object, **kwargs: object) -> np.ndarray:
        del args
        return (
            np.zeros(4) if kwargs["divergence_backend"] == "exact" else np.full(4, 3.0)
        )

    monkeypatch.setattr(models.training, "predict_lid", predict_lid)
    result = _trace_quality(
        object(),
        np.zeros((4, 8), dtype=np.float32),
        np.zeros(4),
        variant_id="vp_diffusion",
        family="vp_diffusion",
        representation="coefficients",
        scale=1.0,
        trace_seed=123,
        batch_size=4,
        maximum_mae=2.0,
        maximum_ratio=1.25,
        output_dir=tmp_path,
    )
    assert result["status"] == "diagnostic_only"
    assert result["accuracy_gate_applied"] is False
    assert result["hutchinson64_mae_vs_exact"] == 3.0


def test_trace_prefix_assessment_rejects_multiple_inference_batches(
    tmp_path: Path,
) -> None:
    with pytest.raises(CanaryError, match="one shared inference minibatch"):
        _trace_quality(
            object(),
            np.zeros((5, 8), dtype=np.float32),
            np.zeros(5),
            variant_id="vp_diffusion",
            family="vp_diffusion",
            representation="dataset",
            scale=1.0,
            trace_seed=123,
            batch_size=4,
            maximum_mae=None,
            maximum_ratio=None,
            output_dir=tmp_path,
        )


@pytest.mark.parametrize(
    ("representation", "maximum_mae", "expected_status"),
    [
        ("coefficients", 5.0, "diagnostic_only"),
        ("dataset", None, "diagnostic_only"),
    ],
)
def test_pointwise_accuracy_never_blocks_a_finite_benchmark_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    representation: str,
    maximum_mae: float | None,
    expected_status: str,
) -> None:
    import models.training

    def predict_lid(
        _trained: object,
        query: np.ndarray,
        scale: float,
        **_kwargs: object,
    ) -> np.ndarray:
        error = {1.0: 30.0, 2.0: 23.4259, 3.0: 40.0}[scale]
        return np.full(len(query), error, dtype=np.float64)

    monkeypatch.setattr(models.training, "predict_lid", predict_lid)
    result, selected_scale = _lid_quality(
        object(),
        np.zeros((4, 8), dtype=np.float32),
        np.zeros(4, dtype=np.float64),
        variant_id="vp_diffusion",
        family="vp_diffusion",
        representation=representation,
        scales=(1.0, 2.0, 3.0),
        trace_seed=123,
        batch_size=4,
        maximum_mae=maximum_mae,
        output_dir=tmp_path,
    )
    assert selected_scale == 2.0
    assert result["selected_pointwise_mae"] == pytest.approx(23.4259)
    assert result["status"] == expected_status
    assert result["accuracy_gate_applied"] is False
    assert result["boundary_selected"] is False


@pytest.mark.parametrize("representation", ["coefficients", "dataset"])
def test_pointwise_boundary_is_preserved_without_censoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, representation: str
) -> None:
    import models.training

    monkeypatch.setattr(
        models.training,
        "predict_lid",
        lambda _trained, query, scale, **_kwargs: np.full(
            len(query), scale, dtype=np.float64
        ),
    )
    result, _ = _lid_quality(
        object(),
        np.zeros((4, 8), dtype=np.float32),
        np.zeros(4, dtype=np.float64),
        variant_id="vp_diffusion",
        family="vp_diffusion",
        representation=representation,
        scales=(1.0, 2.0, 3.0),
        trace_seed=123,
        batch_size=4,
        maximum_mae=5.0 if representation == "coefficients" else None,
        output_dir=tmp_path,
    )
    assert result["status"] == "diagnostic_only"
    assert result["boundary_selected"] is True


def test_trace_assessment_uses_production_selected_lambda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from experiments import v2_canary

    observed: dict[str, float] = {}
    monkeypatch.setattr(
        v2_canary,
        "_history_quality",
        lambda trained, gates: {"status": "passed"},
    )
    monkeypatch.setattr(
        v2_canary,
        "_lid_quality",
        lambda *args, **kwargs: ({"status": "diagnostic_only"}, 2.0),
    )
    monkeypatch.setattr(
        v2_canary,
        "_reconstruction_quality",
        lambda *args, **kwargs: {"status": "passed"},
    )

    def trace_quality(*args: object, **kwargs: object) -> dict[str, str]:
        del args
        observed["scale"] = float(kwargs["scale"])
        return {"status": "diagnostic_only"}

    monkeypatch.setattr(v2_canary, "_trace_quality", trace_quality)
    monkeypatch.setattr(
        v2_canary,
        "_reference_selector_quality",
        lambda *args, **kwargs: {"status": "passed"},
    )
    config = canary_config()
    scales = np.asarray(config["evaluation"]["lambda_grid"], dtype=np.float64)
    production_selected_lambda = 0.5
    result = CanaryDiagnostics(config)(
        {
            "trained": SimpleNamespace(preprocessing_sha256=SHA),
            "model_variant": "vp_diffusion",
            "family": "vp_diffusion",
            "cell": SimpleNamespace(
                suite_id="e2", dataset="e2_arrows", representation="dataset"
            ),
            "eval_batch_size": 4,
            "partition": SimpleNamespace(
                selection_features=np.zeros((40, 8), dtype=np.float32),
                selection_target=np.full(40, 14.0),
                fit_features=np.zeros((40, 8), dtype=np.float32),
            ),
            "candidate_scales": scales,
            "production_selected_lambda": production_selected_lambda,
            "cell_identity": {"sha256": SHA},
        },
        tmp_path,
    )
    assert result["status"] == "passed"
    assert result["production_selected_lambda"] == production_selected_lambda
    assert observed["scale"] == production_selected_lambda


def test_ve_reconstruction_uses_x0_denoiser_output_directly(tmp_path: Path) -> None:
    import torch

    class ExactDenoiser(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def forward(self, noisy: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            del sigma
            return torch.ones_like(noisy) + self.anchor * 0

    result = _reconstruction_quality(
        {
            "model": ExactDenoiser(),
            "config": object(),
            "normalization_mean": np.zeros(3, dtype=np.float32),
            "normalization_scale": 1.0,
        },
        np.ones((8, 3), dtype=np.float32),
        variant_id="ve_diffusion",
        noise_ratio=0.5,
        seed=7,
        maximum_ratio=0.9,
        output_dir=tmp_path,
    )
    assert result["status"] == "diagnostic_only"
    assert result["reconstruction_mse"] == 0.0
    with np.load(tmp_path / "reconstruction.npz") as arrays:
        np.testing.assert_array_equal(arrays["predicted"], np.ones((8, 3)))


def test_reference_selector_no_knee_is_preserved_without_censoring(
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
    assert result["status"] == "diagnostic_only"
    assert result["failure_reason"] == "no_knee"


def test_reference_selector_structural_error_remains_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.global_campaign_v2
    import models.training

    monkeypatch.setattr(
        experiments.global_campaign_v2,
        "unknown_reference_lambdas",
        lambda: np.asarray([0.5, 1.0, 2.0]),
    )

    def fail_selector(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise experiments.global_campaign_v2.GlobalCampaignError("invalid curve")

    monkeypatch.setattr(
        experiments.global_campaign_v2,
        "select_unknown_reference_kneedle",
        fail_selector,
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
    assert result["error_type"] == "GlobalCampaignError"


def test_reference_selector_uses_exact_common_lambda_endpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.global_campaign_v2
    import models.training

    observed: list[float] = []

    def strict_predict(
        _trained: object,
        query: np.ndarray,
        scale: float,
        **_kwargs: object,
    ) -> np.ndarray:
        assert experiments.global_campaign_v2.COMMON_LAMBDA_MIN <= scale
        assert scale <= experiments.global_campaign_v2.COMMON_LAMBDA_MAX
        observed.append(scale)
        return np.ones(len(query), dtype=np.float64)

    monkeypatch.setattr(models.training, "predict_lid", strict_predict)
    monkeypatch.setattr(
        experiments.global_campaign_v2,
        "select_unknown_reference_kneedle",
        lambda scales, curve: (
            None,
            {"status": "selection_failed", "failure_reason": "no_knee"},
        ),
    )
    result = _reference_selector_quality(
        object(),
        np.zeros((4, 3), dtype=np.float32),
        variant_id="posterior_log_noise_affine_flow",
        family="independent_affine_flow",
        representation="coefficients",
        trace_seed=9,
        batch_size=4,
        output_dir=tmp_path,
    )

    assert result["status"] == "diagnostic_only"
    assert observed[0] == experiments.global_campaign_v2.COMMON_LAMBDA_MIN
    assert observed[-1] == experiments.global_campaign_v2.COMMON_LAMBDA_MAX
