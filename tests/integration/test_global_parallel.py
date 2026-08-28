from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from experiments import global_campaign
from experiments.global_campaign import (
    APPROVED_GLOBAL_CELL_KEYS,
    APPROVED_MODEL_VARIANTS,
    CellData,
    ModelPlan,
    canonical_json,
    compose_global_campaign_config,
    sha256_bytes,
    sha256_path,
)
from experiments.global_parallel import (
    H100_PROFILE,
    TELEMETRY_EPOCH_INTERVAL,
    ParallelDependencies,
    _cell_dependencies,
    _prepare_campaign,
    _run_cell_task,
    _ThrottledEventForwarder,
    run_global_parallel_campaign,
    with_cell_dag_profile,
    with_h100_profile,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_SHA = "7" * 64


def _source_hash(root: Path) -> str:
    del root
    return SOURCE_SHA


def _tiny_plans(config: dict[str, Any]) -> tuple[ModelPlan, ...]:
    override = config["execution"]["training_batch_size_override"]
    model = {
        "name": "test_diffusion",
        "family": "diffusion",
        "readout": "full",
        "selection_prefer": "smaller",
        "derivative_backend": "exact",
        "trace_probes": 0,
        "training": {
            "device": "cpu",
            "seed": 0,
            "epochs": 1,
            "batch_size": 256 if override is None else int(override),
            "validation_interval": 1,
            "early_stopping_patience": None,
        },
        "scales": [0.1, 0.2, 0.4],
    }
    return tuple(
        ModelPlan(
            variant_id=row["id"],
            experiment_name=row["experiment_name"],
            model=model,
        )
        for row in config["campaign"]["models"]
    )


def _telemetry_plans(config: dict[str, Any]) -> tuple[ModelPlan, ...]:
    return tuple(
        ModelPlan(
            variant_id=plan.variant_id,
            experiment_name=plan.experiment_name,
            model={
                **plan.model,
                "training": {
                    **plan.model["training"],
                    "epochs": 45,
                    "early_stopping_patience": None,
                },
            },
        )
        for plan in _tiny_plans(config)
    )


def _source_preflight(config, root, cells):  # type: ignore[no-untyped-def]
    del config, root
    return {
        cell.inventory_id: {
            "schema_version": 1,
            "kind": "parallel_integration_test_seal",
            "inventory_id": cell.inventory_id,
        }
        for cell in cells
    }


def _cell_loader(cell, config, root):  # type: ignore[no-untyped-def]
    del config, root
    ordinal = APPROVED_GLOBAL_CELL_KEYS.index(cell.key)
    if cell.target_policy == "paired_delta":
        offset = float(cell.expected_lid_delta)
    elif cell.target_policy == "sample_size":
        offset = float(ordinal) / 100.0
    else:
        offset = float(ordinal) / 1000.0
    train = np.column_stack((np.full(12, offset), np.linspace(0.0, 1.0, 12)))
    validation = np.column_stack((np.full(4, offset), np.arange(4)))
    test = np.column_stack((np.full(4, offset), np.arange(4)))
    known = cell.target_policy == "known_lid"
    train_target = train[:, 0] + 0.2 if known else None
    validation_target = validation[:, 0] + 0.2 if known else None
    test_target = test[:, 0] + 0.2 if known else None
    labels = np.arange(4, dtype=np.int64)
    record = {
        "schema_version": 1,
        "cell_key": cell.key,
        "fixture_revision": 1,
    }
    return CellData(
        train=train,
        validation=validation,
        test=test,
        train_target=train_target,
        validation_target=validation_target,
        test_target=test_target,
        validation_labels=labels,
        test_labels=labels,
        input_record=record,
        input_sha256=sha256_bytes(canonical_json(record).encode()),
    )


def _append_training_call(status: str, checkpoint_path: Path) -> None:
    log_path = Path(os.environ["LID_PARALLEL_TEST_CALL_LOG"])
    payload = f"{status}\t{checkpoint_path}\n".encode()
    descriptor = os.open(log_path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, payload)
    finally:
        os.close(descriptor)


def _training_result(config, checkpoint_path: Path):  # type: ignore[no-untyped-def]
    config_record = global_campaign._canonical_training_config_record(
        config, field="parallel fake trainer config"
    )
    history = tuple(_fake_history(config_record))
    best = history[-1]
    return SimpleNamespace(
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=sha256_path(checkpoint_path),
        config=config_record,
        history=history,
        best_epoch=best["epoch"],
        best_validation_loss=best["validation_loss"],
    )


def _fake_history(config: dict[str, Any]) -> list[dict[str, Any]]:
    epochs = int(config["epochs"])
    interval = int(config["validation_interval"])
    schedule = list(range(interval, epochs + 1, interval))
    if not schedule or schedule[-1] != epochs:
        schedule.append(epochs)
    return [
        {
            "epoch": epoch,
            "train_loss": 1.0 / epoch,
            "validation_loss": 1.0 / (epoch + 1),
            "learning_rate": 1.0e-3,
        }
        for epoch in schedule
    ]


def _train_success(
    family,
    train,
    validation,
    config,
    checkpoint_path,
    log_callback=None,
    *,
    progress_checkpoint_path=None,
):  # type: ignore[no-untyped-def]
    del family, train, validation, log_callback
    checkpoint = Path(checkpoint_path)
    progress = Path(progress_checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    config_record = global_campaign._canonical_training_config_record(
        config, field="parallel fake trainer config"
    )
    history = _fake_history(config_record)
    best = history[-1]
    checkpoint.write_text(
        canonical_json(
            {
                "config": config_record,
                "history": history,
                "best_epoch": best["epoch"],
                "best_validation_loss": best["validation_loss"],
            }
        ),
        encoding="utf-8",
    )
    progress.unlink(missing_ok=True)
    _append_training_call("success", checkpoint)
    return _training_result(config, checkpoint)


def _train_success_with_epoch_logs(
    family,
    train,
    validation,
    config,
    checkpoint_path,
    log_callback=None,
    *,
    progress_checkpoint_path=None,
):  # type: ignore[no-untyped-def]
    if log_callback is not None:
        for epoch in range(1, int(config["epochs"]) + 1):
            log_callback(
                {
                    "epoch": epoch,
                    "train_loss": 1.0 / epoch,
                    "validated": epoch % int(config["validation_interval"]) == 0,
                }
            )
    return _train_success(
        family,
        train,
        validation,
        config,
        checkpoint_path,
        None,
        progress_checkpoint_path=progress_checkpoint_path,
    )


def _train_fail_once(
    family,
    train,
    validation,
    config,
    checkpoint_path,
    log_callback=None,
    *,
    progress_checkpoint_path=None,
):  # type: ignore[no-untyped-def]
    if os.environ["LID_PARALLEL_TEST_FAILURE_TARGET"] not in str(checkpoint_path):
        return _train_success(
            family,
            train,
            validation,
            config,
            checkpoint_path,
            log_callback,
            progress_checkpoint_path=progress_checkpoint_path,
        )
    marker = Path(os.environ["LID_PARALLEL_TEST_FAILURE_MARKER"])
    try:
        descriptor = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return _train_success(
            family,
            train,
            validation,
            config,
            checkpoint_path,
            log_callback,
            progress_checkpoint_path=progress_checkpoint_path,
        )
    os.close(descriptor)
    progress = Path(progress_checkpoint_path)
    progress.parent.mkdir(parents=True, exist_ok=True)
    progress.write_bytes(b"resumable-progress")
    _append_training_call("failed", Path(checkpoint_path))
    raise RuntimeError("adversarial worker failure")


def _load_checkpoint(path, *, device):  # type: ignore[no-untyped-def]
    del device
    checkpoint = Path(path)
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    return SimpleNamespace(
        checkpoint_path=checkpoint,
        checkpoint_sha256=sha256_path(checkpoint),
        config=payload["config"],
        history=tuple(payload["history"]),
        best_epoch=payload["best_epoch"],
        best_validation_loss=payload["best_validation_loss"],
    )


def _predict(
    trained,
    query,
    scale,
    *,
    family,
    readout,
    divergence_backend,
    trace_probes,
    trace_seed,
    batch_size,
):  # type: ignore[no-untyped-def]
    del (
        trained,
        family,
        readout,
        divergence_backend,
        trace_probes,
        trace_seed,
        batch_size,
    )
    values = np.asarray(query)
    return values[:, 0] + float(scale)


def _dependencies(train_fn, model_plans_fn=_tiny_plans) -> ParallelDependencies:  # type: ignore[no-untyped-def]
    return ParallelDependencies(
        source_preflight_fn=_source_preflight,
        cell_loader=_cell_loader,
        train_fn=train_fn,
        predict_fn=_predict,
        load_checkpoint_fn=_load_checkpoint,
        source_hash_fn=_source_hash,
        model_plans_fn=model_plans_fn,
    )


def test_h100_profile_is_explicit_and_legacy_default_is_unchanged() -> None:
    base = global_campaign.validate_global_campaign_config(
        compose_global_campaign_config(("logging.backend=none",))
    )
    assert base["execution"] == {
        "profile": "legacy_sequential",
        "strategy": "sequential",
        "worker_count": 1,
        "training_batch_size_override": None,
        "evaluation_batch_size_override": None,
    }

    h100 = with_h100_profile(base)
    assert h100["execution"] == dict(H100_PROFILE)
    plans = global_campaign.model_plans(h100)
    assert {plan.model["training"]["batch_size"] for plan in plans} == {4096}
    assert h100["campaign"]["evaluation"]["batch_size"] == 128
    assert h100["execution"]["evaluation_batch_size_override"] == 512
    affine_plans = tuple(
        plan for plan in plans if plan.model["family"] == "independent_affine_flow"
    )
    assert len(affine_plans) == 6
    assert {plan.model["diagnostics"]["batch_size"] for plan in affine_plans} == {512}

    legacy_affine_plans = tuple(
        plan
        for plan in global_campaign.model_plans(base)
        if plan.model["family"] == "independent_affine_flow"
    )
    assert {
        plan.model["diagnostics"]["batch_size"] for plan in legacy_affine_plans
    } == {128}


def test_cell_dag_dependencies_follow_declared_references_including_paired_delta() -> (
    None
):
    config = global_campaign.validate_global_campaign_config(
        compose_global_campaign_config(("logging.backend=none",))
    )
    cells = tuple(global_campaign.load_campaign_inventory(config, PROJECT_ROOT))
    dependencies = _cell_dependencies(cells)
    assert len(dependencies) == 39
    for cell_index, cell in enumerate(cells):
        dependency = dependencies[cell_index]
        if cell.reference_dataset in {None, cell.dataset}:
            assert dependency is None
            continue
        assert dependency is not None
        reference = cells[dependency]
        assert reference == global_campaign._reference_cell(cells, cell)
        assert reference.suite_id == cell.suite_id
        assert reference.dataset == cell.reference_dataset
        assert reference.representation == cell.representation
        if cell.target_policy == "paired_delta":
            assert reference.target_policy == "paired_delta"


def test_epoch_telemetry_is_throttled_without_touching_scientific_payload() -> None:
    forwarded: list[tuple[str, dict[str, Any]]] = []
    forwarder = _ThrottledEventForwarder(
        lambda event, payload: forwarded.append((event, dict(payload))),
        epoch_interval=TELEMETRY_EPOCH_INTERVAL,
    )
    scientific = {
        "selected_scale": 0.2,
        "validation": {"mae": 0.1},
        "test": {"mae": 0.2},
    }
    scientific_sha = sha256_bytes(canonical_json(scientific).encode())
    forwarder("cell.started", {"cell_id": "cell"})
    for epoch in range(1, 46):
        forwarder(
            "cell.training.epoch",
            {"step": epoch, "training": {"epoch": epoch, "train_loss": 1.0}},
        )
    forwarder("cell.completed", scientific)

    assert [
        payload["step"]
        for event, payload in forwarded
        if event == "cell.training.epoch"
    ] == [1, 20, 40, 45]
    assert forwarded[0][0] == "cell.started"
    assert forwarded[-1][0] == "cell.completed"
    assert sha256_bytes(canonical_json(scientific).encode()) == scientific_sha


def test_cell_scientific_artifacts_are_independent_of_telemetry_throttling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "LID_PARALLEL_TEST_CALL_LOG", str(tmp_path / "training-calls.log")
    )
    config = with_cell_dag_profile(
        compose_global_campaign_config(("logging.backend=none",)),
        worker_count=1,
        training_batch_size_override=4096,
        evaluation_batch_size_override=512,
        profile="cpu-telemetry-independence-test",
    )
    dependencies = _dependencies(
        _train_success_with_epoch_logs, model_plans_fn=_telemetry_plans
    )
    raw = _prepare_campaign(
        config,
        root=PROJECT_ROOT,
        output_root=tmp_path / "raw",
        dependencies=dependencies,
    )
    throttled = _prepare_campaign(
        config,
        root=PROJECT_ROOT,
        output_root=tmp_path / "throttled",
        dependencies=dependencies,
    )
    raw_events: list[tuple[str, dict[str, Any]]] = []
    throttled_events: list[tuple[str, dict[str, Any]]] = []
    raw_result = _run_cell_task(
        raw,
        model_index=0,
        cell_index=13,
        reference_summary=None,
        dependencies=dependencies,
        event_callback=lambda event, payload: raw_events.append((event, dict(payload))),
    )
    forwarder = _ThrottledEventForwarder(
        lambda event, payload: throttled_events.append((event, dict(payload))),
        epoch_interval=TELEMETRY_EPOCH_INTERVAL,
    )
    throttled_result = _run_cell_task(
        throttled,
        model_index=0,
        cell_index=13,
        reference_summary=None,
        dependencies=dependencies,
        event_callback=forwarder,
    )
    forwarder.flush()

    def artifact_hashes(directory: str) -> dict[str, str]:
        root = Path(directory)
        return {
            path.relative_to(root).as_posix(): sha256_path(path)
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

    assert artifact_hashes(raw_result["directory"]) == artifact_hashes(
        throttled_result["directory"]
    )
    assert sum(event == "cell.training.epoch" for event, _ in raw_events) == 45
    assert [
        payload["step"]
        for event, payload in throttled_events
        if event == "cell.training.epoch"
    ] == [1, 20, 40, 45]


def test_two_spawn_workers_fail_fast_resume_and_complete_exact_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    call_log = tmp_path / "training-calls.log"
    failure_marker = tmp_path / "fail-once.marker"
    monkeypatch.setenv("LID_PARALLEL_TEST_CALL_LOG", str(call_log))
    monkeypatch.setenv("LID_PARALLEL_TEST_FAILURE_MARKER", str(failure_marker))
    monkeypatch.setenv(
        "LID_PARALLEL_TEST_FAILURE_TARGET",
        "posterior-vp-trigonometric-flow/e5/.e5-upscaled-fmnist__dataset__",
    )
    monkeypatch.setattr(global_campaign, "hash_declared_sources", _source_hash)
    monkeypatch.setattr(global_campaign, "model_plans", _tiny_plans)
    config = with_cell_dag_profile(
        compose_global_campaign_config(("logging.backend=none",)),
        worker_count=2,
        training_batch_size_override=4096,
        evaluation_batch_size_override=512,
        profile="cpu-2-worker-test",
    )

    preflight_root = run_global_parallel_campaign(
        config,
        root=PROJECT_ROOT,
        output_root=tmp_path / "campaigns",
        dependencies=_dependencies(_train_success),
        require_cuda=False,
        preflight_only=True,
    )
    preflight = json.loads(
        (preflight_root / "state" / "parallel" / "preflight.json").read_text()
    )
    assert preflight["status"] == "ready"
    assert preflight["execution"]["worker_count"] == 2
    assert preflight["task_count"] == 390
    assert len(preflight["workers"]) == 2
    assert not call_log.exists()

    with pytest.raises(
        global_campaign.GlobalCampaignError, match="adversarial worker failure"
    ):
        run_global_parallel_campaign(
            config,
            root=PROJECT_ROOT,
            output_root=tmp_path / "campaigns",
            dependencies=_dependencies(_train_fail_once),
            require_cuda=False,
        )
    first_lines = call_log.read_text(encoding="utf-8").splitlines()
    assert sum(line.startswith("failed\t") for line in first_lines) == 1
    assert sum(line.startswith("success\t") for line in first_lines) < 390
    assert not list((tmp_path / "campaigns").rglob("campaign.json"))
    assert list((tmp_path / "campaigns").rglob("training_progress.pt"))
    assert list(
        (tmp_path / "campaigns").glob("*/runs/*/e1/e1-sampled-fmnist-step2__dataset__*")
    ), "failure must leave at least one sealed dependent cell for resume"

    campaign_root = run_global_parallel_campaign(
        config,
        root=PROJECT_ROOT,
        output_root=tmp_path / "campaigns",
        dependencies=_dependencies(_train_success),
        require_cuda=False,
    )
    lines = call_log.read_text(encoding="utf-8").splitlines()
    assert sum(line.startswith("failed\t") for line in lines) == 1
    assert sum(line.startswith("success\t") for line in lines) == 390

    manifest = json.loads((campaign_root / "campaign.json").read_text())
    assert len(manifest["models"]) == 10
    assert len(manifest["cells"]) == 390
    assert tuple(row["model_variant"] for row in manifest["models"]) == (
        APPROVED_MODEL_VARIANTS
    )
    assignments = json.loads(
        (campaign_root / "state" / "parallel" / "assignments.json").read_text()
    )
    assert assignments["strategy"] == "cell_dag_pool"
    assert assignments["worker_count"] == 2
    assert len(assignments["assignments"]) == 390
    assert (
        len(
            {
                (row["model_index"], row["cell_index"])
                for row in assignments["assignments"]
            }
        )
        == 390
    )
    assert {row["model_variant"] for row in assignments["assignments"]} == set(
        APPROVED_MODEL_VARIANTS
    )
    completed_assignments = [
        row for row in assignments["assignments"] if row["status"] == "completed"
    ]
    reconstructed_assignments = [
        row for row in assignments["assignments"] if row["status"] == "reconstructed"
    ]
    assert {row["worker_slot"] for row in completed_assignments} <= {0, 1}
    assert all(row["worker_slot"] is None for row in reconstructed_assignments)
    dispatch_sequences = [row["dispatch_sequence"] for row in completed_assignments]
    assert len(dispatch_sequences) == len(set(dispatch_sequences))
    assert max(row["in_flight_after_dispatch"] for row in completed_assignments) == 2
    assert all(
        row["in_flight_after_dispatch"] in {1, 2} for row in completed_assignments
    )

    ledgers = sorted((campaign_root / "state" / "models").glob("*/ledger.json"))
    assert len(ledgers) == 10
    for path in ledgers:
        ledger = json.loads(path.read_text())
        assert ledger["complete"] is True
        assert len(ledger["completed_cells"]) == 39
        assert len({row["cell_key"] for row in ledger["completed_cells"]}) == 39
    assert not list(campaign_root.rglob("checkpoint.pt"))
    assert not list(campaign_root.rglob("training_progress.pt"))
    assert (
        global_campaign.validate_global_campaign(
            campaign_root,
            project_root=PROJECT_ROOT,
            source_preflight_fn=_source_preflight,
            cell_loader=_cell_loader,
        )
        == []
    )

    calls_before = len(lines)
    assert (
        run_global_parallel_campaign(
            config,
            root=PROJECT_ROOT,
            output_root=tmp_path / "campaigns",
            dependencies=_dependencies(_train_success),
            require_cuda=False,
        )
        == campaign_root
    )
    assert len(call_log.read_text(encoding="utf-8").splitlines()) == calls_before
