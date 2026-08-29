"""Integration coverage for the staged NF ablation campaign."""

from __future__ import annotations

import csv
import json
import math
import os
import threading
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml

from experiments import global_campaign, nf_ablation

SOURCE_RECORD = {"kind": "sealed_fixture_source", "revision": 1}
RAW_INPUT_RECORD = {"kind": "nf_ablation_fixture", "revision": 1}


def _known_cell() -> global_campaign.CampaignCell:
    return global_campaign.CampaignCell(
        inventory_id="fixture-e4-dataset",
        source_kind="integration_fixture",
        exact_archive=False,
        dataset_config="fixture.yaml",
        data_root="fixture",
        registry="fixture.json",
        registry_overlay=None,
        upstream_revision="fixture-v1",
        archive_sha256=None,
        provenance_label="nf-ablation-test",
        suite_id="e4",
        dataset="e4_sphere_pca_radius1",
        representation="dataset",
        target_policy="known_lid",
        selection_protocol=global_campaign.KNOWN_SELECTION_PROTOCOL,
        comparison_group="e4-sphere",
        reference_dataset=None,
        expected_lid_delta=None,
    )


def _coordinator_cells() -> tuple[global_campaign.CampaignCell, ...]:
    cells: list[global_campaign.CampaignCell] = []
    for index, key in enumerate(global_campaign.APPROVED_GLOBAL_CELL_KEYS):
        suite_id, dataset, representation = key.split("/")
        if key.startswith("e1/e1_sampled"):
            target_policy = "sample_size"
            reference_dataset = "e1_sampled_fmnist_step1"
        elif key.startswith("e5/"):
            target_policy = "paired_delta"
            reference_dataset = "e5_downscaled_fmnist"
        else:
            target_policy = "known_lid"
            reference_dataset = None
        cells.append(
            global_campaign.CampaignCell(
                inventory_id=f"coordinator-{index:02d}",
                source_kind="integration_fixture",
                exact_archive=False,
                dataset_config="fixture.yaml",
                data_root="fixture",
                registry="fixture.json",
                registry_overlay=None,
                upstream_revision="fixture-v1",
                archive_sha256=None,
                provenance_label="nf-ablation-coordinator-test",
                suite_id=suite_id,
                dataset=dataset,
                representation=representation,
                target_policy=target_policy,
                selection_protocol=(
                    global_campaign.KNOWN_SELECTION_PROTOCOL
                    if target_policy == "known_lid"
                    else global_campaign.UNKNOWN_SELECTION_PROTOCOL
                ),
                comparison_group=f"fixture-{suite_id}",
                reference_dataset=reference_dataset,
                expected_lid_delta=(None if target_policy == "known_lid" else 0.0),
            )
        )
    return tuple(cells)


def _coordinator_preflight(tmp_path: Path) -> nf_ablation.BaselinePreflight:
    cells = _coordinator_cells()
    return nf_ablation.BaselinePreflight(
        baseline_root=tmp_path / "sealed-baseline",
        baseline_campaign_identity="c" * 64,
        baseline_unified_sha256="b" * 64,
        baseline_row_count=780,
        config=_selection_config(),
        cells=cells,
        input_sha256_by_key={
            cell.key: f"{index + 1:064x}"[-64:] for index, cell in enumerate(cells)
        },
        source_records={
            cell.inventory_id: {"kind": "coordinator_fixture", "cell_key": cell.key}
            for cell in cells
        },
    )


def _prepared_fixture(tmp_path: Path) -> nf_ablation._PreparedNFAblation:
    preflight = _coordinator_preflight(tmp_path)
    return nf_ablation._PreparedNFAblation(
        project_root=str(tmp_path),
        campaign_root=str(tmp_path / "nf-campaign"),
        campaign_identity="d" * 64,
        identity_record={"fixture": True},
        preflight=preflight,
        worker_count=2,
    )


def _cell_data(input_sha256: str) -> global_campaign.CellData:
    train = np.column_stack(
        (
            np.linspace(1.0, 2.1, 12, dtype=np.float64),
            np.linspace(-1.0, 1.0, 12, dtype=np.float64),
        )
    )
    validation = np.asarray([[1.25, -0.5], [1.75, 0.5]], dtype=np.float64)
    test = np.asarray([[1.5, -0.25], [2.0, 0.75]], dtype=np.float64)
    return global_campaign.CellData(
        train=train,
        validation=validation,
        test=test,
        train_target=train[:, 0] + 0.1,
        validation_target=validation[:, 0] + 0.1,
        test_target=test[:, 0] + 0.1,
        validation_labels=np.arange(validation.shape[0], dtype=np.int64),
        test_labels=np.arange(test.shape[0], dtype=np.int64),
        input_record=RAW_INPUT_RECORD,
        input_sha256=input_sha256,
    )


def _selection_config() -> dict[str, Any]:
    return {
        "campaign": {
            "selection": {
                "fraction": 0.25,
                "minimum_fit": 4,
                "minimum_selection": 2,
                "maximum_selection": 3,
                "tie_tolerance": 0.0,
                "stability_window": 3,
                "stability_min_valid_fraction": 1.0,
            }
        }
    }


def _training_history(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "epoch": 1,
            "train_loss": 2.0,
            "validation_loss": 1.0,
            "learning_rate": float(config["learning_rate"]),
        }
    ]


def _train_success(
    family: str,
    train: Any,
    validation: Any,
    config: dict[str, Any],
    checkpoint_path: Path,
    log_callback: Any = None,
    *,
    progress_checkpoint_path: Path | None = None,
) -> SimpleNamespace:
    del family, train, validation, log_callback
    checkpoint = Path(checkpoint_path)
    progress = Path(progress_checkpoint_path)  # type: ignore[arg-type]
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    canonical = global_campaign._canonical_training_config_record(
        config, field="NF ablation fake trainer config"
    )
    checkpoint.write_text(json.dumps({"config": canonical}), encoding="utf-8")
    progress.write_text("resumable", encoding="utf-8")
    return _load_fake_checkpoint(checkpoint, device=str(canonical["device"]))


def _load_fake_checkpoint(checkpoint_path: Path, *, device: str) -> SimpleNamespace:
    del device
    checkpoint = Path(checkpoint_path)
    config = json.loads(checkpoint.read_text(encoding="utf-8"))["config"]
    return SimpleNamespace(
        config=config,
        checkpoint_sha256=global_campaign.sha256_path(checkpoint),
        history=_training_history(config),
        best_epoch=1,
        best_validation_loss=1.0,
    )


def _predict_readouts(
    trained: Any,
    query: Any,
    epsilon: float,
    **kwargs: Any,
) -> dict[str, Any]:
    del trained
    features = np.asarray(query, dtype=np.float64)
    prediction = features[:, 0] + float(epsilon)
    finite_difference_log_step = float(kwargs["finite_difference_log_step"])
    ols_log_step = float(kwargs["ols_log_step"])
    finite_difference_epsilons = float(epsilon) * np.exp(
        np.asarray((-1.0, 1.0)) * finite_difference_log_step
    )
    ols_epsilons = float(epsilon) * np.exp(np.arange(-4.0, 5.0) * ols_log_step)
    intercept = features[:, :1]
    return {
        "epsilon": float(epsilon),
        "finite_difference_log_step": finite_difference_log_step,
        "ols_log_step": ols_log_step,
        "finite_difference_epsilons": finite_difference_epsilons,
        "finite_difference_log_likelihood": intercept
        + np.log(finite_difference_epsilons)[None, :],
        "ols_epsilons": ols_epsilons,
        "ols_log_likelihood": intercept + np.log(ols_epsilons)[None, :],
        "lid_by_readout": {
            readout: prediction.copy() for readout in nf_ablation.READOUTS
        },
    }


def _predict_log_likelihood(
    trained: Any,
    query: Any,
    epsilon: float,
    **kwargs: Any,
) -> np.ndarray:
    del trained, kwargs
    features = np.asarray(query, dtype=np.float64)
    ambient_dim = int(features.shape[1])
    desired_lid = features[:, 0] + 0.1
    return 7.0 + (desired_lid - float(ambient_dim)) * np.log(float(epsilon))


def _dependencies(
    *,
    data: global_campaign.CellData,
    train_fn: Any = _train_success,
) -> nf_ablation.NFDependencies:
    return nf_ablation.NFDependencies(
        inventory_loader=lambda *args, **kwargs: (),
        source_preflight_fn=lambda *args, **kwargs: {},
        cell_loader=lambda *args, **kwargs: data,
        train_fn=train_fn,
        load_checkpoint_fn=_load_fake_checkpoint,
        predict_readouts_fn=_predict_readouts,
        predict_log_likelihood_fn=_predict_log_likelihood,
    )


def _task(
    tmp_path: Path,
    *,
    candidate_id: str = "C0",
    evaluate_test: bool = False,
) -> nf_ablation.NFCellTask:
    cell = _known_cell()
    bound_record = {**RAW_INPUT_RECORD, "source_preflight": SOURCE_RECORD}
    expected_input_sha256 = global_campaign.sha256_bytes(
        global_campaign.canonical_json(bound_record).encode("utf-8")
    )
    return nf_ablation.NFCellTask(
        project_root=str(tmp_path),
        campaign_root=str(tmp_path / "campaign"),
        baseline_unified_sha256="b" * 64,
        config=_selection_config(),
        cell=cell,
        candidate_id=candidate_id,
        seed=0,
        evaluate_test=evaluate_test,
        expected_input_sha256=expected_input_sha256,
        source_record=SOURCE_RECORD,
    )


def _bomb(*args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    raise AssertionError("a sealed cell must not invoke runtime dependencies")


def _bomb_dependencies() -> nf_ablation.NFDependencies:
    return nf_ablation.NFDependencies(
        inventory_loader=_bomb,
        source_preflight_fn=_bomb,
        cell_loader=_bomb,
        train_fn=_bomb,
        load_checkpoint_fn=_bomb,
        predict_readouts_fn=_bomb,
        predict_log_likelihood_fn=_bomb,
    )


def test_conditional_cell_seals_prunes_and_recovers_post_prune_window(
    tmp_path: Path,
) -> None:
    task = _task(tmp_path)
    data = _cell_data(task.expected_input_sha256)

    result = nf_ablation._run_cell_task(task, _dependencies(data=data))

    sealed = Path(result["directory"])
    assert result["reused"] is False
    assert sealed.is_dir()
    assert not list(sealed.rglob("checkpoint.pt"))
    assert not list(sealed.rglob("training_progress.pt"))
    assert len(list(sealed.glob("validation_prediction__*.npy"))) == len(
        nf_ablation.READOUTS
    )
    assert not list(sealed.glob("test_prediction__*.npy"))
    likelihood_files = sorted(sealed.glob("likelihood__*.npy"))
    assert len(likelihood_files) == 4 * (
        len(nf_ablation.CONDITIONAL_SELECTION_SCALES) + 1
    )
    summary = json.loads((sealed / "summary.json").read_text(encoding="utf-8"))
    assert len(summary["run_id"]) == 64
    assert summary["run_id"][:20] == summary["cell_id"]
    assert summary["run_id_contract"] == "full_sha256_of_cell_identity_v1"
    evidence = summary["likelihood_path_evidence"]
    assert len(evidence["train_selection"]) == len(
        nf_ablation.CONDITIONAL_SELECTION_SCALES
    )
    assert len(evidence["evaluated_splits"]) == 1
    for record in (*evidence["train_selection"], *evidence["evaluated_splits"]):
        for artifact in record["files"].values():
            path = sealed / artifact["path"]
            assert path.is_file()
            assert global_campaign.sha256_path(path) == artifact["sha256"]
    model = nf_ablation._candidate_model(
        nf_ablation.candidate_by_id(task.candidate_id), seed=task.seed
    )
    identity = nf_ablation._task_identity(task, model)
    assert nf_ablation._validate_cell(sealed, identity) == []

    # Reuse of a final seal is entirely artifact-driven.
    reused = nf_ablation._run_cell_task(task, _bomb_dependencies())
    assert reused["reused"] is True
    assert reused["directory"] == str(sealed)

    # The atomic post-prune/pre-rename recovery window must also avoid all
    # loading, training, and prediction calls.
    final_dir, work_dir = nf_ablation._task_paths(task, identity)
    os.replace(final_dir, work_dir)
    recovered = nf_ablation._run_cell_task(task, _bomb_dependencies())
    assert recovered["reused"] is True
    assert Path(recovered["directory"]) == final_dir
    assert final_dir.is_dir() and not work_dir.exists()


def test_p0_resumes_independent_components_and_prunes_each_checkpoint(
    tmp_path: Path,
) -> None:
    task = _task(tmp_path, candidate_id="P0")
    data = _cell_data(task.expected_input_sha256)
    calls = 0

    @wraps(_train_success)
    def fail_on_third_component(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 3:
            progress = Path(kwargs["progress_checkpoint_path"])
            progress.parent.mkdir(parents=True, exist_ok=True)
            progress.write_text("interrupted", encoding="utf-8")
            raise RuntimeError("fixture interruption")
        return _train_success(*args, **kwargs)

    with pytest.raises(RuntimeError, match="fixture interruption"):
        nf_ablation._run_cell_task(
            task,
            _dependencies(data=data, train_fn=fail_on_third_component),
        )
    assert calls == 3

    # Components 0 and 1 are already fully evaluated and pruned.  Resume must
    # train only components 2..8, including replacement of component 2's
    # progress artifact.
    resumed_calls = 0

    @wraps(_train_success)
    def counting_success(*args: Any, **kwargs: Any) -> Any:
        nonlocal resumed_calls
        resumed_calls += 1
        return _train_success(*args, **kwargs)

    result = nf_ablation._run_cell_task(
        task,
        _dependencies(data=data, train_fn=counting_success),
    )
    assert resumed_calls == 7
    sealed = Path(result["directory"])
    components = sorted((sealed / "components").glob("epsilon-*"))
    assert len(components) == 9
    assert all((directory / "attestation.json").is_file() for directory in components)
    assert not list(sealed.rglob("checkpoint.pt"))
    assert not list(sealed.rglob("training_progress.pt"))
    prediction = np.load(
        sealed / f"validation_prediction__{nf_ablation.PAPER_PARITY_READOUT}.npy",
        allow_pickle=False,
    )
    np.testing.assert_allclose(prediction, data.validation_target, atol=1.0e-12)


def _fresh_source_record(cell: SimpleNamespace) -> dict[str, Any]:
    return {"kind": "fresh_source", "cell_key": cell.key}


def _fresh_raw_data(cell: SimpleNamespace) -> global_campaign.CellData:
    empty = np.zeros((2, 1), dtype=np.float64)
    return global_campaign.CellData(
        train=empty,
        validation=empty,
        test=empty,
        train_target=None,
        validation_target=None,
        test_target=None,
        validation_labels=None,
        test_labels=None,
        input_record={"kind": "raw_fixture", "cell_key": cell.key},
        input_sha256="0" * 64,
    )


def _write_baseline_fixture(root: Path) -> tuple[list[SimpleNamespace], dict[str, str]]:
    root.mkdir(parents=True)
    cells: list[SimpleNamespace] = []
    input_sha_by_key: dict[str, str] = {}
    inventory: list[dict[str, Any]] = []
    for index, key in enumerate(global_campaign.APPROVED_GLOBAL_CELL_KEYS):
        suite, dataset, representation = key.split("/")
        inventory_id = f"inventory-{index:02d}"
        cell = SimpleNamespace(key=key, inventory_id=inventory_id)
        cells.append(cell)
        bound_record = {
            "kind": "raw_fixture",
            "cell_key": key,
            "source_preflight": _fresh_source_record(cell),
        }
        digest = global_campaign.sha256_bytes(
            global_campaign.canonical_json(bound_record).encode("utf-8")
        )
        input_sha_by_key[key] = digest
        inventory.append(
            {
                "cell": {
                    "suite_id": suite,
                    "dataset": dataset,
                    "representation": representation,
                },
                "input_sha256": digest,
            }
        )
    (root / "input_inventory.json").write_text(json.dumps(inventory), encoding="utf-8")
    (root / "resolved_config.yaml").write_text(
        yaml.safe_dump({"fixture": True}), encoding="utf-8"
    )
    with (root / "unified_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=global_campaign._UNIFIED_TABLE_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        for variant in global_campaign.APPROVED_MODEL_VARIANTS:
            row = {field: "" for field in global_campaign._UNIFIED_TABLE_FIELDS}
            row.update(
                {
                    "analysis": "known_lid",
                    "model_variant": variant,
                    "suite_id": "e4",
                    "dataset": "fixture",
                    "representation": "dataset",
                    "split": "validation",
                    "readout": "fixture",
                }
            )
            writer.writerow(row)
    unified_sha = global_campaign.sha256_path(root / "unified_results.csv")
    manifest = {
        "complete": True,
        "expected_models": 10,
        "expected_cells_per_model": 39,
        "cells": [{} for _ in range(390)],
        "unified_results_sha256": unified_sha,
        "campaign_identity": "c" * 64,
    }
    (root / "campaign.json").write_text(json.dumps(manifest), encoding="utf-8")
    return cells, input_sha_by_key


def test_baseline_preflight_reloads_exact_sources_and_never_mutates_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = tmp_path / "baseline"
    cells, input_sha_by_key = _write_baseline_fixture(baseline)
    before = {
        path.name: path.read_bytes() for path in baseline.iterdir() if path.is_file()
    }
    monkeypatch.setattr(
        global_campaign,
        "validate_global_campaign_config",
        lambda raw: {"validated": raw},
    )
    dependencies = nf_ablation.NFDependencies(
        inventory_loader=lambda config, root: tuple(cells),
        source_preflight_fn=lambda config, root, source_cells: {
            cell.inventory_id: _fresh_source_record(cell) for cell in source_cells
        },
        cell_loader=lambda cell, config, root: _fresh_raw_data(cell),
        train_fn=_bomb,
        load_checkpoint_fn=_bomb,
        predict_readouts_fn=_bomb,
        predict_log_likelihood_fn=_bomb,
    )

    result = nf_ablation.preflight_baseline(
        baseline,
        project_root=tmp_path,
        dependencies=dependencies,
    )

    assert (
        tuple(cell.key for cell in result.cells)
        == global_campaign.APPROVED_GLOBAL_CELL_KEYS
    )
    assert result.input_sha256_by_key == input_sha_by_key
    assert result.baseline_row_count == len(global_campaign.APPROVED_MODEL_VARIANTS)
    assert result.baseline_unified_sha256 == global_campaign.sha256_path(
        baseline / "unified_results.csv"
    )
    assert {
        path.name: path.read_bytes() for path in baseline.iterdir() if path.is_file()
    } == before


def test_baseline_preflight_rejects_a_fresh_input_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = tmp_path / "baseline"
    cells, _input_sha_by_key = _write_baseline_fixture(baseline)
    monkeypatch.setattr(
        global_campaign,
        "validate_global_campaign_config",
        lambda raw: {"validated": raw},
    )
    mismatch_key = global_campaign.APPROVED_GLOBAL_CELL_KEYS[-1]
    dependencies = nf_ablation.NFDependencies(
        inventory_loader=lambda config, root: tuple(cells),
        source_preflight_fn=lambda config, root, source_cells: {
            cell.inventory_id: _fresh_source_record(cell) for cell in source_cells
        },
        cell_loader=lambda cell, config, root: global_campaign.CellData(
            **{
                **_fresh_raw_data(cell).__dict__,
                "input_record": (
                    {"kind": "tampered", "cell_key": cell.key}
                    if cell.key == mismatch_key
                    else _fresh_raw_data(cell).input_record
                ),
            }
        ),
        train_fn=_bomb,
        load_checkpoint_fn=_bomb,
        predict_readouts_fn=_bomb,
        predict_log_likelihood_fn=_bomb,
    )

    with pytest.raises(nf_ablation.NFAblationError, match="source input differs"):
        nf_ablation.preflight_baseline(
            baseline,
            project_root=tmp_path,
            dependencies=dependencies,
        )


def test_coordinator_builds_exact_stages_and_reference_dependency_closure(
    tmp_path: Path,
) -> None:
    prepared = _prepared_fixture(tmp_path)
    stage1 = nf_ablation._stage1_tasks(prepared)

    assert len(stage1) == len(nf_ablation.STAGE1_CANDIDATES) * 8 == 56
    assert {task.candidate_id for task in stage1} == {
        candidate.candidate_id for candidate in nf_ablation.STAGE1_CANDIDATES
    }
    assert {task.cell.key for task in stage1} == set(nf_ablation.STAGE1_SENTINEL_KEYS)
    assert all(task.seed == 0 and task.evaluate_test is False for task in stage1)

    dependencies = nf_ablation._task_dependency_ids(stage1)
    task_by_id = {nf_ablation._task_id(task): task for task in stage1}
    for task_id, dependency_id in dependencies.items():
        task = task_by_id[task_id]
        if task.cell.dataset in {
            "e1_sampled_fmnist_step7",
            "e5_padded_fmnist_adddim8",
        }:
            assert dependency_id is not None
            dependency = task_by_id[dependency_id]
            assert dependency.candidate_id == task.candidate_id
            assert dependency.seed == task.seed
            assert dependency.cell.dataset == task.cell.reference_dataset
        else:
            assert dependency_id is None

    promotions = [
        nf_ablation.Promotion(
            candidate_id="C1",
            readout="ols5",
            median_log_ratio=math.log(0.8),
            win_rate=0.8,
            stratum_median_ratios={"known_dataset": 0.8},
            dataset_to_coefficients_ratio=1.0,
        )
    ]
    stage2 = nf_ablation._stage2_tasks(prepared, promotions)
    assert len(stage2) == 2 * 2 * 19
    assert all(task.evaluate_test is False for task in stage2)
    winner = nf_ablation.Winner(
        candidate_id="C1",
        readout="ols5",
        validation_median_mae=1.0,
        validation_mean_mae=1.0,
        canonical_geometric_mean_ratio=0.8,
        canonical_wins=12,
        canonical_regressions_over_25pct=0,
        generated_geometric_mean_ratio=0.9,
    )
    stage3 = nf_ablation._stage3_tasks(prepared, winner)
    assert len(stage3) == 3 * 39
    assert {(task.candidate_id, task.seed) for task in stage3} == {
        ("C0", 2),
        ("C1", 2),
        ("C1", 3),
    }
    assert all(task.evaluate_test is True for task in stage3)


def test_cpu_preflight_only_coordinator_writes_contract_without_running_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight = _coordinator_preflight(tmp_path)
    lifecycle: list[str] = []

    class FakeMonitor:
        def __init__(self, path: Path, **kwargs: Any) -> None:
            del kwargs
            self.path = path

        def start(self) -> dict[str, Any]:
            lifecycle.append("monitor-start")
            return {"status": "disabled_cpu_run", "gpus": []}

        def stop(self) -> None:
            lifecycle.append("monitor-stop")

    class FakePool:
        def __init__(
            self, *, worker_count: int, require_cuda: bool, **kwargs: Any
        ) -> None:
            del kwargs
            assert worker_count == 2
            assert require_cuda is False

        def start(self) -> tuple[dict[str, Any], ...]:
            lifecycle.append("pool-start")
            return tuple(
                {
                    "kind": "ready",
                    "worker_slot": slot,
                    "visible_device": "cpu-test-worker",
                    "device_name": None,
                }
                for slot in range(2)
            )

        def close(self) -> None:
            lifecycle.append("pool-close")

        def abort(self) -> None:
            lifecycle.append("pool-abort")

    monkeypatch.setattr(
        nf_ablation, "preflight_baseline", lambda *args, **kwargs: preflight
    )
    monkeypatch.setattr(nf_ablation, "_NvidiaSmiMonitor", FakeMonitor)
    monkeypatch.setattr(nf_ablation, "_PersistentNFPool", FakePool)

    root = nf_ablation.run_nf_ablation_campaign(
        baseline_root=preflight.baseline_root,
        output_root=tmp_path / "output",
        project_root=Path(__file__).resolve().parents[2],
        worker_count=2,
        dependencies=_bomb_dependencies(),
        require_cuda=False,
        preflight_only=True,
    )

    assert lifecycle == ["monitor-start", "pool-start", "pool-close", "monitor-stop"]
    assert not (root / "campaign.json").exists()
    assert (root / "ablation_identity.json").is_file()
    assert (root / "baseline_provenance.json").is_file()
    assert (root / "resolved_candidates.yaml").is_file()
    preflight_record = json.loads((root / "preflight.json").read_text(encoding="utf-8"))
    assert preflight_record["status"] == "ready"
    assert preflight_record["worker_count"] == 2
    assert preflight_record["require_cuda"] is False
    assert preflight_record["planned_stage1_tasks"] == 56
    assert preflight_record["planned_stage2_known_cells"] == 19
    assert preflight_record["planned_stage3_tasks"] == 117


def test_cli_parsing_and_main_forward_cpu_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    baseline = tmp_path / "baseline"
    output = tmp_path / "output"
    project = tmp_path / "project"
    arguments = [
        "--baseline-root",
        str(baseline),
        "--output-root",
        str(output),
        "--project-root",
        str(project),
        "--worker-count",
        "2",
        "--allow-cpu",
        "--preflight-only",
    ]
    parsed = nf_ablation._parse_cli(arguments)
    assert parsed.baseline_root == baseline
    assert parsed.output_root == output
    assert parsed.project_root == project
    assert parsed.worker_count == 2
    assert parsed.allow_cpu is True
    assert parsed.preflight_only is True

    observed: dict[str, Any] = {}

    def fake_run(**kwargs: Any) -> Path:
        observed.update(kwargs)
        return output / "prepared-campaign"

    monkeypatch.setattr(nf_ablation, "run_nf_ablation_campaign", fake_run)
    nf_ablation.main(arguments)

    assert observed == {
        "baseline_root": baseline,
        "output_root": output,
        "project_root": project,
        "worker_count": 2,
        "require_cuda": False,
        "preflight_only": True,
    }
    assert capsys.readouterr().out.strip() == str(output / "prepared-campaign")


def test_coordinator_production_worker_preflight_fails_before_source_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(nf_ablation, "preflight_baseline", _bomb)
    with pytest.raises(nf_ablation.NFAblationError, match="exactly 8 GPU workers"):
        nf_ablation.run_nf_ablation_campaign(
            baseline_root=tmp_path / "baseline",
            output_root=tmp_path / "output",
            project_root=tmp_path,
            worker_count=7,
            require_cuda=True,
        )

    assert nf_ablation._visible_device_tokens(3, require_cuda=False) == (
        None,
        None,
        None,
    )


def test_production_configures_exact_deterministic_cublas_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    nf_ablation._configure_deterministic_cublas(require_cuda=True)
    assert (
        os.environ["CUBLAS_WORKSPACE_CONFIG"]
        == nf_ablation.DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG
    )

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with pytest.raises(nf_ablation.NFAblationError, match="differs"):
        nf_ablation._configure_deterministic_cublas(require_cuda=True)

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "cpu-sentinel")
    nf_ablation._configure_deterministic_cublas(require_cuda=False)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == "cpu-sentinel"


def test_nvidia_smi_preflight_uses_exact_fields_and_parses_one_gpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_query = (
        "index,uuid,name,utilization.gpu,memory.used,memory.total,power.draw,pstate"
    )

    def fake_run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        assert command[1] == f"--query-gpu={expected_query}"
        assert command[2] == "--format=csv,noheader,nounits"
        assert kwargs == {
            "check": True,
            "capture_output": True,
            "text": True,
            "timeout": 30,
        }
        return SimpleNamespace(
            stdout="0, GPU-deadbeef, NVIDIA H100 80GB HBM3, 97, 1234, 81559, 612.5, P0\n"
        )

    monkeypatch.setattr(
        nf_ablation.shutil, "which", lambda executable: "/fake/nvidia-smi"
    )
    monkeypatch.setattr(nf_ablation.subprocess, "run", fake_run)
    monitor = nf_ablation._NvidiaSmiMonitor(
        tmp_path / "telemetry.jsonl",
        enabled=True,
        expected_gpu_count=1,
    )

    sample = monitor._sample()

    assert sample["gpus"] == [
        {
            "index": 0,
            "uuid": "GPU-deadbeef",
            "name": "NVIDIA H100 80GB HBM3",
            "utilization_gpu_percent": 97.0,
            "memory_used_mib": 1234.0,
            "memory_total_mib": 81559.0,
            "power_draw_w": 612.5,
            "pstate": "P0",
        }
    ]


def test_nvidia_smi_first_sample_is_strict_h100_but_periodic_errors_are_nonfatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "telemetry.jsonl"
    monitor = nf_ablation._NvidiaSmiMonitor(
        path,
        enabled=True,
        expected_gpu_count=1,
        interval_seconds=0.001,
    )
    calls = 0
    recovered = threading.Event()

    def sample() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("transient nvidia-smi timeout")
        if calls >= 3:
            recovered.set()
        return {
            "schema_version": 1,
            "session_id": monitor.session_id,
            "recorded_at_utc": "2026-08-29T00:00:00+00:00",
            "gpus": [{"name": "NVIDIA H100 80GB HBM3"}],
        }

    monkeypatch.setattr(monitor, "_sample", sample)
    monitor.start()
    assert recovered.wait(timeout=1.0)
    monitor.stop()
    monitor.raise_if_failed()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert any(row.get("status") == "periodic_sampling_error" for row in records)

    wrong_gpu = nf_ablation._NvidiaSmiMonitor(
        tmp_path / "wrong-gpu.jsonl",
        enabled=True,
        expected_gpu_count=1,
    )
    monkeypatch.setattr(
        wrong_gpu,
        "_sample",
        lambda: {
            "schema_version": 1,
            "session_id": wrong_gpu.session_id,
            "recorded_at_utc": "2026-08-29T00:00:00+00:00",
            "gpus": [{"name": "NVIDIA A100-SXM4-80GB"}],
        },
    )
    with pytest.raises(nf_ablation.NFAblationError, match="H100"):
        wrong_gpu.start()
