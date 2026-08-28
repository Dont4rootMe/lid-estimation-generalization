"""Spawn-based one-job cell-DAG runner for the global LID campaign.

The scientific unit of independence is one model/cell pair.  Dependencies are
derived from each cell's declared ``reference_dataset`` and never hard-coded.
A work-conserving coordinator dynamically assigns ready cells to a fixed pool
of workers.  Production workers see exactly one CUDA device each; there is
deliberately no DDP and no cross-model gradient synchronization.

Only the coordinator writes telemetry, ledgers, aggregates and manifests.
Workers own only disjoint cell directories and send structured events over the
result queue.  Consequently a failed process can be terminated without
invalidating cells already atomically sealed by any worker.
"""

from __future__ import annotations

import copy
import json
import multiprocessing as mp
import os
import pickle
import queue
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

H100_PROFILE: Mapping[str, Any] = {
    "profile": "h100_8gpu_cell_dag",
    "strategy": "cell_dag_pool",
    "worker_count": 8,
    "training_batch_size_override": 4096,
    "evaluation_batch_size_override": 512,
}

TELEMETRY_EPOCH_INTERVAL = 20


@dataclass(frozen=True)
class ParallelDependencies:
    """Spawn-picklable dependency overrides used by integration tests."""

    inventory_loader: Any = None
    source_preflight_fn: Any = None
    cell_loader: Any = None
    train_fn: Any = None
    predict_fn: Any = None
    load_checkpoint_fn: Any = None
    logger_factory: Any = None
    affine_diagnostics_fn: Any = None
    source_hash_fn: Any = None
    model_plans_fn: Any = None


@dataclass(frozen=True)
class _PreparedCampaign:
    config: dict[str, Any]
    project_root: str
    campaign_root: str
    state_dir: str
    campaign_id: str
    campaign_identity: str
    config_sha: str
    source_sha: str
    input_inventory_sha: str
    cells: tuple[Any, ...]
    plans: tuple[Any, ...]
    source_records: dict[str, Mapping[str, Any]]
    preflight_inputs: dict[str, dict[str, Any]]


def with_cell_dag_profile(
    hydra_config: Any,
    *,
    worker_count: int,
    training_batch_size_override: int | None,
    evaluation_batch_size_override: int | None,
    profile: str = "custom_cell_dag",
) -> dict[str, Any]:
    """Return a plain config with an explicit, identity-bearing DAG profile."""

    from experiments import global_campaign as campaign

    value = copy.deepcopy(campaign._mapping(hydra_config, field="global campaign"))
    value["execution"] = {
        "profile": profile,
        "strategy": campaign.EXECUTION_STRATEGY_CELL_DAG,
        "worker_count": worker_count,
        "training_batch_size_override": training_batch_size_override,
        "evaluation_batch_size_override": evaluation_batch_size_override,
    }
    return campaign.validate_global_campaign_config(value)


def with_h100_profile(hydra_config: Any) -> dict[str, Any]:
    """Install the immutable eight-H100 production profile."""

    return with_cell_dag_profile(
        hydra_config,
        worker_count=int(H100_PROFILE["worker_count"]),
        training_batch_size_override=int(H100_PROFILE["training_batch_size_override"]),
        evaluation_batch_size_override=int(
            H100_PROFILE["evaluation_batch_size_override"]
        ),
        profile=str(H100_PROFILE["profile"]),
    )


def _prepare_campaign(
    hydra_config: Any,
    *,
    root: Path | None,
    output_root: Path | None,
    dependencies: ParallelDependencies,
) -> _PreparedCampaign:
    from experiments import global_campaign as campaign

    config = campaign.validate_global_campaign_config(hydra_config)
    if config["execution"]["strategy"] != campaign.EXECUTION_STRATEGY_CELL_DAG:
        raise campaign.GlobalCampaignError(
            "parallel runner requires execution.strategy=cell_dag_pool"
        )
    project_root = (
        campaign.repository_root() if root is None else Path(root)
    ).resolve()
    configured_output = campaign._safe_path(config["output_root"], field="output_root")
    selected_output = configured_output if output_root is None else Path(output_root)
    if not selected_output.is_absolute():
        selected_output = project_root / selected_output
    selected_output = selected_output.resolve()
    source_hash_fn = (
        campaign.hash_declared_sources
        if dependencies.source_hash_fn is None
        else dependencies.source_hash_fn
    )
    source_sha = source_hash_fn(project_root)
    config_sha = campaign.sha256_bytes(campaign.canonical_json(config).encode("utf-8"))
    campaign_id = str(config["campaign"]["campaign_id"])

    inventory_loader = (
        campaign.load_campaign_inventory
        if dependencies.inventory_loader is None
        else dependencies.inventory_loader
    )
    source_preflight_fn = (
        campaign.validate_campaign_sources
        if dependencies.source_preflight_fn is None
        else dependencies.source_preflight_fn
    )
    cell_loader = (
        campaign.load_campaign_cell_data
        if dependencies.cell_loader is None
        else dependencies.cell_loader
    )
    cells = tuple(inventory_loader(config, project_root))
    if not cells:
        raise campaign.GlobalCampaignError("campaign inventory contains no cells")
    if {cell.suite_id for cell in cells} != set(campaign.REQUIRED_SUITE_IDS):
        raise campaign.GlobalCampaignError(
            "campaign cells do not cover suites e1 through e8"
        )
    if tuple(cell.key for cell in cells) != campaign.APPROVED_GLOBAL_CELL_KEYS:
        raise campaign.GlobalCampaignError(
            "campaign inventory differs from the exact approved ordered 39 cells"
        )
    source_records = dict(source_preflight_fn(config, project_root, cells))
    if set(source_records) != {cell.inventory_id for cell in cells}:
        raise campaign.GlobalCampaignError(
            "source preflight does not cover exact inventories"
        )

    preflight_inputs: dict[str, dict[str, Any]] = {}
    for cell in cells:
        data = campaign._bind_source_preflight(
            cell_loader(cell, config, project_root), cell, source_records
        )
        if cell.key in preflight_inputs:
            raise campaign.GlobalCampaignError(
                f"preflight repeated campaign cell {cell.key}"
            )
        preflight_inputs[cell.key] = {
            "input_sha256": data.input_sha256,
            "input_record": campaign._plain(data.input_record),
            "source_evidence": campaign._source_evidence(data, config),
        }
        del data
    for cell in cells:
        if cell.target_policy != "paired_delta":
            continue
        reference = campaign._reference_cell(cells, cell)
        current = preflight_inputs[cell.key]["source_evidence"]
        baseline = preflight_inputs[reference.key]["source_evidence"]
        for split in ("validation", "test"):
            current_labels = current[f"{split}_labels_sha256"]
            baseline_labels = baseline[f"{split}_labels_sha256"]
            if (
                current_labels is None
                or baseline_labels is None
                or current_labels != baseline_labels
                or current[f"{split}_n"] != baseline[f"{split}_n"]
            ):
                raise campaign.GlobalCampaignError(
                    f"paired-delta source rows are not aligned: {cell.key}/{split}"
                )

    input_inventory = [
        {
            "cell": campaign._plain(cell),
            "input_sha256": preflight_inputs[cell.key]["input_sha256"],
            "input_record": preflight_inputs[cell.key]["input_record"],
        }
        for cell in cells
    ]
    input_inventory_sha = campaign.sha256_bytes(
        campaign.canonical_json(input_inventory).encode("utf-8")
    )
    identity_record = {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "config_sha256": config_sha,
        "source_tree_sha256": source_sha,
        "input_inventory_sha256": input_inventory_sha,
    }
    campaign_identity = campaign.sha256_bytes(
        campaign.canonical_json(identity_record).encode("utf-8")
    )
    campaign_root = selected_output / f"{campaign_id}__{campaign_identity[:20]}"
    campaign_root.mkdir(parents=True, exist_ok=True)
    state_dir = campaign_root / "state"
    state_dir.mkdir(exist_ok=True)
    inventory_path = campaign_root / "input_inventory.json"
    if inventory_path.exists():
        try:
            existing = json.loads(inventory_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise campaign.GlobalCampaignError(
                "existing campaign input inventory is unreadable"
            ) from exc
        if not campaign._same_json(existing, input_inventory):
            raise campaign.GlobalCampaignError(
                "existing campaign input inventory differs from current preflight"
            )
    else:
        campaign._write_json(inventory_path, input_inventory)

    return _PreparedCampaign(
        config=config,
        project_root=str(project_root),
        campaign_root=str(campaign_root),
        state_dir=str(state_dir),
        campaign_id=campaign_id,
        campaign_identity=campaign_identity,
        config_sha=config_sha,
        source_sha=source_sha,
        input_inventory_sha=input_inventory_sha,
        cells=cells,
        plans=(
            campaign.model_plans(config)
            if dependencies.model_plans_fn is None
            else dependencies.model_plans_fn(config)
        ),
        source_records=source_records,
        preflight_inputs=preflight_inputs,
    )


def _cell_dependencies(cells: Sequence[Any]) -> tuple[int | None, ...]:
    """Derive the DAG only from each cell's declared reference dataset."""

    from experiments import global_campaign as campaign

    indices = {cell.key: index for index, cell in enumerate(cells)}
    dependencies: list[int | None] = []
    for cell in cells:
        if cell.reference_dataset in {None, cell.dataset}:
            dependencies.append(None)
            continue
        reference = campaign._reference_cell(cells, cell)
        dependencies.append(indices[reference.key])
    for start in range(len(cells)):
        visited: set[int] = set()
        current: int | None = start
        while current is not None:
            if current in visited:
                raise campaign.GlobalCampaignError(
                    "campaign cell references form a cycle"
                )
            visited.add(current)
            current = dependencies[current]
    return tuple(dependencies)


class _CellIdentityInput:
    def __init__(self, input_sha256: str) -> None:
        self.input_sha256 = input_sha256


def _expected_cell(
    prepared: _PreparedCampaign, *, model_index: int, cell_index: int
) -> tuple[Path, dict[str, Any]]:
    from experiments import global_campaign as campaign

    plan = prepared.plans[model_index]
    cell = prepared.cells[cell_index]
    identity = campaign._cell_identity(
        campaign_id=prepared.campaign_id,
        campaign_config_sha=prepared.config_sha,
        source_sha=prepared.source_sha,
        model_plan=plan,
        cell=cell,
        cell_data=_CellIdentityInput(
            str(prepared.preflight_inputs[cell.key]["input_sha256"])
        ),
        selection_contract=prepared.config["campaign"]["selection"],
        evaluation_contract=prepared.config["campaign"]["evaluation"],
    )
    cell_id = campaign.sha256_bytes(campaign.canonical_json(identity).encode("utf-8"))[
        :20
    ]
    directory = (
        Path(prepared.campaign_root)
        / "runs"
        / campaign._safe_component(plan.variant_id)
        / campaign._safe_component(cell.suite_id)
        / (
            f"{campaign._safe_component(cell.dataset)}__"
            f"{campaign._safe_component(cell.representation)}__{cell_id}"
        )
    )
    return directory, identity


def _model_ledger_path(prepared: _PreparedCampaign, model_index: int) -> Path:
    from experiments import global_campaign as campaign

    return (
        Path(prepared.state_dir)
        / "models"
        / campaign._safe_component(prepared.plans[model_index].variant_id)
        / "ledger.json"
    )


def _validate_dag_ledger(
    path: Path, *, prepared: _PreparedCampaign, model_index: int
) -> None:
    if not path.exists():
        return
    from experiments import global_campaign as campaign

    ledger = campaign._load_json(path)
    completed_cells = ledger.get("completed_cells")
    if (
        set(ledger)
        != {
            "schema_version",
            "campaign_identity",
            "model_variant",
            "completed_cells",
            "complete",
        }
        or ledger.get("schema_version") != 1
        or ledger.get("campaign_identity") != prepared.campaign_identity
        or ledger.get("model_variant") != prepared.plans[model_index].variant_id
        or not isinstance(completed_cells, list)
        or ledger.get("complete") not in {True, False}
    ):
        raise campaign.GlobalCampaignError(f"durable model ledger is invalid: {path}")
    assert isinstance(completed_cells, list)
    expected_fields = {
        "model_variant",
        "cell_key",
        "ordinal",
        "path",
        "manifest_sha256",
        "summary_sha256",
    }
    seen_ordinals: set[int] = set()
    for record in completed_cells:
        if not isinstance(record, Mapping) or set(record) != expected_fields:
            raise campaign.GlobalCampaignError(
                f"durable model ledger row is invalid: {path}"
            )
        ordinal = record.get("ordinal")
        if (
            type(ordinal) is not int
            or not 0 <= ordinal < len(prepared.cells)
            or ordinal in seen_ordinals
        ):
            raise campaign.GlobalCampaignError(
                f"durable model ledger ordinal is invalid: {path}"
            )
        seen_ordinals.add(ordinal)
        expected_directory, _identity = _expected_cell(
            prepared, model_index=model_index, cell_index=ordinal
        )
        if (
            record.get("model_variant") != prepared.plans[model_index].variant_id
            or record.get("cell_key") != prepared.cells[ordinal].key
            or record.get("path")
            != expected_directory.relative_to(prepared.campaign_root).as_posix()
            or not isinstance(record.get("manifest_sha256"), str)
            or not campaign._SHA256.fullmatch(record["manifest_sha256"])
            or not isinstance(record.get("summary_sha256"), str)
            or not campaign._SHA256.fullmatch(record["summary_sha256"])
        ):
            raise campaign.GlobalCampaignError(
                f"durable model ledger row differs from its cell: {path}"
            )
    if [record["ordinal"] for record in completed_cells] != sorted(seen_ordinals):
        raise campaign.GlobalCampaignError(
            f"durable model ledger rows are not ordered: {path}"
        )
    if bool(ledger["complete"]) != (len(completed_cells) == len(prepared.cells)):
        raise campaign.GlobalCampaignError(
            f"durable model ledger completion flag is invalid: {path}"
        )


def _write_dag_ledger(
    prepared: _PreparedCampaign,
    *,
    model_index: int,
    records: Mapping[int, Mapping[str, Any]],
) -> None:
    from experiments import global_campaign as campaign

    ordered = [dict(records[index]) for index in sorted(records)]
    campaign._write_json(
        _model_ledger_path(prepared, model_index),
        {
            "schema_version": 1,
            "campaign_identity": prepared.campaign_identity,
            "model_variant": prepared.plans[model_index].variant_id,
            "completed_cells": ordered,
            "complete": len(ordered) == len(prepared.cells),
        },
    )


def _reconstruct_completed(
    prepared: _PreparedCampaign,
    dag_dependencies: Sequence[int | None],
) -> list[dict[str, dict[int, Any]]]:
    """Strictly discover sealed cells; artifacts, not ledgers, are truth."""

    from experiments import global_campaign as campaign

    states: list[dict[str, dict[int, Any]]] = []
    for model_index, plan in enumerate(prepared.plans):
        _validate_dag_ledger(
            _model_ledger_path(prepared, model_index),
            prepared=prepared,
            model_index=model_index,
        )
        summaries: dict[int, Any] = {}
        directories: dict[int, Any] = {}
        records: dict[int, Any] = {}
        for cell_index, cell in enumerate(prepared.cells):
            directory, identity = _expected_cell(
                prepared, model_index=model_index, cell_index=cell_index
            )
            if not directory.exists():
                continue
            dependency = dag_dependencies[cell_index]
            reference_summary = None
            if dependency is not None:
                reference_summary = summaries.get(dependency)
                if reference_summary is None:
                    raise campaign.GlobalCampaignError(
                        "sealed dependent cell lacks its sealed reference: "
                        f"{plan.variant_id}/{cell.key}"
                    )
            errors = campaign.validate_global_cell(
                directory,
                expected_identity=identity,
                reference_summary=reference_summary,
                expected_source_evidence=prepared.preflight_inputs[cell.key][
                    "source_evidence"
                ],
            )
            if errors:
                raise campaign.GlobalCampaignError(
                    f"refusing to resume invalid sealed cell {directory}: {errors}"
                )
            summary = campaign._load_json(directory / "summary.json")
            summary["summary_sha256"] = campaign.sha256_path(directory / "summary.json")
            summaries[cell_index] = summary
            directories[cell_index] = directory
            records[cell_index] = {
                "model_variant": plan.variant_id,
                "cell_key": cell.key,
                "ordinal": cell_index,
                "path": directory.relative_to(prepared.campaign_root).as_posix(),
                "manifest_sha256": campaign.sha256_path(directory / "manifest.json"),
                "summary_sha256": campaign.sha256_path(directory / "summary.json"),
            }
        _write_dag_ledger(prepared, model_index=model_index, records=records)
        states.append(
            {"summaries": summaries, "directories": directories, "records": records}
        )
    return states


def _run_cell_task(
    prepared: _PreparedCampaign,
    *,
    model_index: int,
    cell_index: int,
    reference_summary: Mapping[str, Any] | None,
    dependencies: ParallelDependencies,
    event_callback: Any,
) -> dict[str, Any]:
    from experiments import global_campaign as campaign

    project_root = Path(prepared.project_root)
    source_hash_fn = (
        campaign.hash_declared_sources
        if dependencies.source_hash_fn is None
        else dependencies.source_hash_fn
    )
    if source_hash_fn(project_root) != prepared.source_sha:
        raise campaign.GlobalCampaignError("source tree changed before campaign cell")
    cell_loader = (
        campaign.load_campaign_cell_data
        if dependencies.cell_loader is None
        else dependencies.cell_loader
    )
    train_fn = dependencies.train_fn
    predict_fn = dependencies.predict_fn
    load_checkpoint_fn = dependencies.load_checkpoint_fn
    if train_fn is None or predict_fn is None or load_checkpoint_fn is None:
        from models.training import load_checkpoint, predict_lid, train_model

        train_fn = train_model if train_fn is None else train_fn
        predict_fn = predict_lid if predict_fn is None else predict_fn
        load_checkpoint_fn = (
            load_checkpoint if load_checkpoint_fn is None else load_checkpoint_fn
        )
    affine_diagnostics_fn = (
        campaign.run_known_affine_diagnostics
        if dependencies.affine_diagnostics_fn is None
        else dependencies.affine_diagnostics_fn
    )
    plan = prepared.plans[model_index]
    cell = prepared.cells[cell_index]
    data = campaign._bind_source_preflight(
        cell_loader(cell, prepared.config, project_root),
        cell,
        prepared.source_records,
    )
    if data.input_sha256 != prepared.preflight_inputs[cell.key]["input_sha256"]:
        raise campaign.GlobalCampaignError(
            f"input changed after campaign preflight: {cell.key}"
        )
    try:
        final_dir, summary = campaign._run_cell(
            campaign_root=Path(prepared.campaign_root),
            campaign_id=prepared.campaign_id,
            campaign_config_sha=prepared.config_sha,
            source_sha=prepared.source_sha,
            config=prepared.config,
            model_plan=plan,
            cell=cell,
            data=data,
            reference_summary=reference_summary,
            train_fn=train_fn,
            predict_fn=predict_fn,
            load_checkpoint_fn=load_checkpoint_fn,
            affine_diagnostics_fn=affine_diagnostics_fn,
            callback=event_callback,
        )
    finally:
        del data
        campaign._clear_accelerator_cache()
    return {
        "model_index": model_index,
        "cell_index": cell_index,
        "summary": summary,
        "directory": str(final_dir),
        "record": {
            "model_variant": plan.variant_id,
            "cell_key": cell.key,
            "ordinal": cell_index,
            "path": final_dir.relative_to(prepared.campaign_root).as_posix(),
            "manifest_sha256": campaign.sha256_path(final_dir / "manifest.json"),
            "summary_sha256": campaign.sha256_path(final_dir / "summary.json"),
        },
    }


class _ThrottledEventForwarder:
    """Bound epoch traffic while preserving all lifecycle events and final epoch."""

    def __init__(self, callback: Any, *, epoch_interval: int) -> None:
        if epoch_interval <= 0:
            raise ValueError("epoch telemetry interval must be positive")
        self._callback = callback
        self._epoch_interval = epoch_interval
        self._pending_epoch: tuple[str, dict[str, Any]] | None = None

    def __call__(self, event: str, payload: Mapping[str, Any]) -> None:
        record = dict(payload)
        if event != "cell.training.epoch":
            self.flush()
            self._callback(event, record)
            return
        step = record.get("step")
        if type(step) is not int:
            self.flush()
            self._callback(event, record)
            return
        self._pending_epoch = (event, record)
        if step == 1 or step % self._epoch_interval == 0:
            self.flush()

    def flush(self) -> None:
        if self._pending_epoch is None:
            return
        event, payload = self._pending_epoch
        self._pending_epoch = None
        self._callback(event, payload)


def _dag_worker_main(
    worker_slot: int,
    device_token: str | None,
    prepared: _PreparedCampaign,
    dependencies: ParallelDependencies,
    task_queue: Any,
    result_queue: Any,
    stop_event: Any,
    require_cuda: bool,
) -> None:
    if device_token is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        visible_device = "cpu-test-worker"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = device_token
        visible_device = device_token
    task: Mapping[str, Any] | None = None
    try:
        if require_cuda:
            import torch

            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError(
                    "production worker must see exactly one available CUDA device"
                )
            torch.cuda.set_device(0)
            visible_device = f"{device_token}:{torch.cuda.get_device_name(0)}"
        while not stop_event.is_set():
            try:
                task = task_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if task is None:
                return
            model_index = int(task["model_index"])
            cell_index = int(task["cell_index"])

            def send_event(
                event: str,
                payload: Mapping[str, Any],
                *,
                _model_index: int = model_index,
                _cell_index: int = cell_index,
            ) -> None:
                result_queue.put(
                    {
                        "kind": "event",
                        "model_index": _model_index,
                        "cell_index": _cell_index,
                        "event": event,
                        "payload": dict(payload),
                    }
                )

            events = _ThrottledEventForwarder(
                send_event, epoch_interval=TELEMETRY_EPOCH_INTERVAL
            )
            try:
                result = _run_cell_task(
                    prepared,
                    model_index=model_index,
                    cell_index=cell_index,
                    reference_summary=task.get("reference_summary"),
                    dependencies=dependencies,
                    event_callback=events,
                )
            finally:
                events.flush()
            result_queue.put(
                {
                    "kind": "completed",
                    "worker_slot": worker_slot,
                    "visible_device": visible_device,
                    "payload": result,
                }
            )
            task = None
    except BaseException as exc:  # noqa: BLE001 - process boundary reports all
        stop_event.set()
        result_queue.put(
            {
                "kind": "failed",
                "worker_slot": worker_slot,
                "task": dict(task) if isinstance(task, Mapping) else None,
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


def _device_tokens(worker_count: int, *, require_cuda: bool) -> list[str | None]:
    if not require_cuda:
        return [None] * worker_count
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        tokens = [str(index) for index in range(worker_count)]
    else:
        tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if len(tokens) != worker_count or len(set(tokens)) != worker_count:
        from experiments.global_campaign import GlobalCampaignError

        raise GlobalCampaignError(
            f"H100 profile requires exactly {worker_count} unique visible devices"
        )
    return tokens


def _terminate_workers(processes: Sequence[Any]) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=10)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)


def _assignment_path(prepared: _PreparedCampaign) -> Path:
    return Path(prepared.state_dir) / "parallel" / "assignments.json"


def _persist_assignments(
    prepared: _PreparedCampaign,
    assignments: Mapping[tuple[int, int], Mapping[str, Any]],
) -> None:
    from experiments import global_campaign as campaign

    campaign._write_json(
        _assignment_path(prepared),
        {
            "schema_version": 1,
            "campaign_identity": prepared.campaign_identity,
            "strategy": prepared.config["execution"]["strategy"],
            "worker_count": int(prepared.config["execution"]["worker_count"]),
            "assignments": [dict(assignments[key]) for key in sorted(assignments)],
        },
    )


def _resume_assignments(
    prepared: _PreparedCampaign,
    completed: set[tuple[int, int]],
) -> dict[tuple[int, int], dict[str, Any]]:
    from experiments import global_campaign as campaign

    path = _assignment_path(prepared)
    assignments: dict[tuple[int, int], dict[str, Any]] = {}
    seen: set[tuple[int, int]] = set()
    if path.exists():
        payload = campaign._load_json(path)
        if (
            set(payload)
            != {
                "schema_version",
                "campaign_identity",
                "strategy",
                "worker_count",
                "assignments",
            }
            or payload.get("schema_version") != 1
            or payload.get("campaign_identity") != prepared.campaign_identity
            or payload.get("strategy") != prepared.config["execution"]["strategy"]
            or payload.get("worker_count")
            != prepared.config["execution"]["worker_count"]
            or not isinstance(payload.get("assignments"), list)
        ):
            raise campaign.GlobalCampaignError(
                f"parallel assignment ledger is invalid: {path}"
            )
        for row in payload["assignments"]:
            expected_fields = {
                "model_index",
                "model_variant",
                "cell_index",
                "cell_key",
                "worker_slot",
                "visible_device",
                "dispatch_sequence",
                "in_flight_after_dispatch",
                "ready_after_dispatch",
                "status",
            }
            if not isinstance(row, Mapping) or set(row) != expected_fields:
                raise campaign.GlobalCampaignError("parallel assignment row is invalid")
            model_index = row.get("model_index")
            cell_index = row.get("cell_index")
            if (
                type(model_index) is not int
                or type(cell_index) is not int
                or not 0 <= model_index < len(prepared.plans)
                or not 0 <= cell_index < len(prepared.cells)
            ):
                raise campaign.GlobalCampaignError(
                    "parallel assignment indices are invalid"
                )
            key = (model_index, cell_index)
            if key in seen:
                raise campaign.GlobalCampaignError(
                    "parallel assignment ledger repeats a cell"
                )
            seen.add(key)
            status = row.get("status")
            worker_slot = row.get("worker_slot")
            dispatch_sequence = row.get("dispatch_sequence")
            in_flight = row.get("in_flight_after_dispatch")
            ready = row.get("ready_after_dispatch")
            completed_row = (
                status == "completed"
                and type(worker_slot) is int
                and 0 <= worker_slot < prepared.config["execution"]["worker_count"]
                and type(dispatch_sequence) is int
                and dispatch_sequence > 0
                and type(in_flight) is int
                and 1 <= in_flight <= prepared.config["execution"]["worker_count"]
                and type(ready) is int
                and ready >= 0
            )
            reconstructed_row = (
                status == "reconstructed"
                and worker_slot is None
                and dispatch_sequence is None
                and in_flight is None
                and ready is None
            )
            if (
                row.get("model_variant") != prepared.plans[model_index].variant_id
                or row.get("cell_key") != prepared.cells[cell_index].key
                or not isinstance(row.get("visible_device"), str)
                or not row["visible_device"]
                or not (completed_row or reconstructed_row)
            ):
                raise campaign.GlobalCampaignError(
                    "parallel assignment row differs from its cell"
                )
            if key in completed:
                assignments[key] = dict(row)
    for key in sorted(completed):
        if key in assignments:
            continue
        model_index, cell_index = key
        assignments[key] = {
            "model_index": model_index,
            "model_variant": prepared.plans[model_index].variant_id,
            "cell_index": cell_index,
            "cell_key": prepared.cells[cell_index].key,
            "worker_slot": None,
            "visible_device": "reconstructed-sealed-cell",
            "dispatch_sequence": None,
            "in_flight_after_dispatch": None,
            "ready_after_dispatch": None,
            "status": "reconstructed",
        }
    _persist_assignments(prepared, assignments)
    return assignments


def _run_cell_dag_pool(
    prepared: _PreparedCampaign,
    *,
    dependencies: ParallelDependencies,
    require_cuda: bool,
    handles: Sequence[Any],
) -> tuple[list[dict[str, dict[int, Any]]], dict[tuple[int, int], dict[str, Any]]]:
    """Run the 390-task DAG while keeping the worker pool work-conserving."""

    from collections import deque

    from experiments import global_campaign as campaign

    try:
        pickle.dumps((prepared, dependencies))
    except Exception as exc:
        raise campaign.GlobalCampaignError(
            "parallel campaign state must be spawn-picklable"
        ) from exc
    dag_dependencies = _cell_dependencies(prepared.cells)
    states = _reconstruct_completed(prepared, dag_dependencies)
    all_tasks = {
        (model_index, cell_index)
        for model_index in range(len(prepared.plans))
        for cell_index in range(len(prepared.cells))
    }
    completed = {
        (model_index, cell_index)
        for model_index, state in enumerate(states)
        for cell_index in state["summaries"]
    }
    assignments = _resume_assignments(prepared, completed)
    pending = all_tasks - completed
    if not pending:
        return states, assignments

    worker_count = int(prepared.config["execution"]["worker_count"])
    tokens = _device_tokens(worker_count, require_cuda=require_cuda)
    context = mp.get_context("spawn")
    task_queue = context.Queue()
    result_queue = context.Queue()
    stop_event = context.Event()
    processes = [
        context.Process(
            target=_dag_worker_main,
            name=f"lid-cell-worker-{slot}",
            args=(
                slot,
                tokens[slot],
                prepared,
                dependencies,
                task_queue,
                result_queue,
                stop_event,
                require_cuda,
            ),
        )
        for slot in range(worker_count)
    ]
    for process in processes:
        process.start()

    ready: deque[tuple[int, int]] = deque()
    ready_set: set[tuple[int, int]] = set()
    in_flight: dict[tuple[int, int], dict[str, Any]] = {}
    dispatch_sequence = max(
        (
            int(row["dispatch_sequence"])
            for row in assignments.values()
            if row.get("dispatch_sequence") is not None
        ),
        default=0,
    )

    def discover_ready() -> None:
        for key in sorted(pending):
            if key in ready_set or key in in_flight:
                continue
            model_index, cell_index = key
            dependency = dag_dependencies[cell_index]
            if dependency is None or (model_index, dependency) in completed:
                ready.append(key)
                ready_set.add(key)

    def fill_workers() -> None:
        nonlocal dispatch_sequence
        while ready and len(in_flight) < worker_count:
            key = ready.popleft()
            ready_set.remove(key)
            model_index, cell_index = key
            dependency = dag_dependencies[cell_index]
            reference_summary = (
                None
                if dependency is None
                else states[model_index]["summaries"][dependency]
            )
            dispatch_sequence += 1
            info = {
                "dispatch_sequence": dispatch_sequence,
                "in_flight_after_dispatch": len(in_flight) + 1,
                "ready_after_dispatch": len(ready),
            }
            in_flight[key] = info
            task_queue.put(
                {
                    "model_index": model_index,
                    "cell_index": cell_index,
                    "reference_summary": reference_summary,
                }
            )
        if ready and len(in_flight) < worker_count:
            raise campaign.GlobalCampaignError("cell-DAG scheduler left a worker idle")

    discover_ready()
    fill_workers()
    if not in_flight:
        _terminate_workers(processes)
        raise campaign.GlobalCampaignError("cell-DAG has no ready task")

    try:
        while pending:
            try:
                message = result_queue.get(timeout=1.0)
            except queue.Empty:
                exited = [
                    process for process in processes if process.exitcode is not None
                ]
                if exited and not stop_event.is_set():
                    raise campaign.GlobalCampaignError(
                        "cell worker exited before coordinator shutdown: "
                        + ", ".join(
                            f"{process.name}={process.exitcode}" for process in exited
                        )
                    )
                if len(exited) == len(processes):
                    raise campaign.GlobalCampaignError(
                        "cell workers exited before reporting campaign failure"
                    )
                continue
            if not isinstance(message, Mapping):
                raise campaign.GlobalCampaignError("invalid cell-worker message")
            kind = message.get("kind")
            if kind == "event":
                model_index = int(message.get("model_index", -1))
                cell_index = int(message.get("cell_index", -1))
                if (
                    not 0 <= model_index < len(handles)
                    or (model_index, cell_index) not in in_flight
                    or not isinstance(message.get("event"), str)
                    or not isinstance(message.get("payload"), Mapping)
                ):
                    raise campaign.GlobalCampaignError(
                        "worker event does not match an in-flight cell"
                    )
                handles[model_index].callback(message["event"], message["payload"])
                continue
            if kind == "failed":
                raise campaign.GlobalCampaignError(
                    "cell worker failed "
                    f"(slot={message.get('worker_slot')}, "
                    f"task={message.get('task')}, "
                    f"type={message.get('exception_type')}): "
                    f"{message.get('message')}\n{message.get('traceback')}"
                )
            if kind != "completed" or not isinstance(message.get("payload"), Mapping):
                raise campaign.GlobalCampaignError("invalid cell-worker result")
            result = dict(message["payload"])
            key = (int(result["model_index"]), int(result["cell_index"]))
            if key not in in_flight or key not in pending:
                raise campaign.GlobalCampaignError(
                    "cell worker completed an unassigned or duplicate task"
                )
            dispatch = in_flight.pop(key)
            pending.remove(key)
            completed.add(key)
            model_index, cell_index = key
            state = states[model_index]
            summary = dict(result["summary"])
            state["summaries"][cell_index] = summary
            state["directories"][cell_index] = Path(str(result["directory"]))
            state["records"][cell_index] = dict(result["record"])
            _write_dag_ledger(
                prepared,
                model_index=model_index,
                records=state["records"],
            )
            assignments[key] = {
                "model_index": model_index,
                "model_variant": prepared.plans[model_index].variant_id,
                "cell_index": cell_index,
                "cell_key": prepared.cells[cell_index].key,
                "worker_slot": int(message["worker_slot"]),
                "visible_device": str(message["visible_device"]),
                **dispatch,
                "status": "completed",
            }
            _persist_assignments(prepared, assignments)
            discover_ready()
            fill_workers()
        for _ in range(worker_count):
            task_queue.put(None)
        for process in processes:
            process.join(timeout=30)
        if any(process.exitcode != 0 for process in processes):
            raise campaign.GlobalCampaignError("one or more cell workers failed")
    except BaseException:
        stop_event.set()
        _terminate_workers(processes)
        raise
    if set(assignments) != all_tasks:
        raise campaign.GlobalCampaignError(
            "cell-DAG assignment ledger does not cover the exact matrix"
        )
    return states, assignments


def _finalize_campaign(
    prepared: _PreparedCampaign,
    results: Sequence[Mapping[str, Any]],
    *,
    dependencies: ParallelDependencies,
) -> Path:
    from experiments import global_campaign as campaign

    campaign_root = Path(prepared.campaign_root)
    state_dir = Path(prepared.state_dir)
    cells = prepared.cells
    plans = prepared.plans
    model_records = [dict(result["model_record"]) for result in results]
    model_aggregates = [dict(result["model_aggregate"]) for result in results]
    cell_records = [
        dict(record) for result in results for record in result["cell_records"]
    ]
    if len(cell_records) != len(plans) * len(cells):
        raise campaign.GlobalCampaignError(
            "parallel campaign did not complete its exact matrix"
        )
    campaign._write_json(
        state_dir / "ledger.json",
        {
            "schema_version": 1,
            "campaign_identity": prepared.campaign_identity,
            "completed_cells": cell_records,
        },
    )
    aggregate = campaign._campaign_aggregate(
        prepared.campaign_identity, model_aggregates
    )
    aggregate_path = campaign_root / "aggregate.json"
    campaign._write_json(aggregate_path, aggregate)
    table_path = campaign_root / "unified_results.csv"
    campaign._write_text(table_path, campaign.render_unified_results_csv(aggregate))

    logger_factory = (
        campaign.open_model_logger
        if dependencies.logger_factory is None
        else dependencies.logger_factory
    )
    for model_index, model_plan in enumerate(plans):
        handle = campaign._safe_model_logger(
            logger_factory,
            model_plan,
            prepared.campaign_identity,
            state_dir,
            prepared.config["logging"],
        )
        handle.asset(aggregate_path, name="global-aggregate.json")
        handle.asset(table_path, name="global-unified-results.csv")
        handle.close()
        record = model_records[model_index]
        record["comet_telemetry"] = handle.telemetry_record()
        manifest_path = campaign_root / str(record["manifest_path"])
        campaign._write_json(
            manifest_path,
            {
                key: value
                for key, value in record.items()
                if key not in {"manifest_path", "manifest_sha256"}
            },
        )
        record["manifest_sha256"] = campaign.sha256_path(manifest_path)

    final_path = campaign_root / "campaign.json"
    final_manifest = {
        "schema_version": campaign.GLOBAL_FINAL_MANIFEST_SCHEMA_VERSION,
        "campaign_identity": prepared.campaign_identity,
        "campaign_id": prepared.campaign_id,
        "config_sha256": prepared.config_sha,
        "source_tree_sha256": prepared.source_sha,
        "input_inventory_sha256": prepared.input_inventory_sha,
        "inventory_cells": [
            {
                **campaign._plain(cell),
                "input_sha256": prepared.preflight_inputs[cell.key]["input_sha256"],
                "input_record": prepared.preflight_inputs[cell.key]["input_record"],
            }
            for cell in cells
        ],
        "approved_model_variants": list(campaign.APPROVED_MODEL_VARIANTS),
        "model_contracts": [
            {
                "variant_id": plan.variant_id,
                "experiment_name": plan.experiment_name,
                "model": campaign._plain(plan.model),
            }
            for plan in plans
        ],
        "input_inventory_path": "input_inventory.json",
        "input_inventory_file_sha256": campaign.sha256_path(
            campaign_root / "input_inventory.json"
        ),
        "aggregate_path": "aggregate.json",
        "aggregate_sha256": campaign.sha256_path(aggregate_path),
        "unified_results_path": "unified_results.csv",
        "unified_results_sha256": campaign.sha256_path(table_path),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "models": model_records,
        "cells": cell_records,
        "expected_models": len(plans),
        "expected_cells_per_model": len(cells),
        "complete": True,
    }
    campaign._write_json(final_path, final_manifest)
    errors = campaign.validate_global_campaign(
        campaign_root,
        expected_campaign_identity=prepared.campaign_identity,
        project_root=Path(prepared.project_root),
        source_preflight_fn=dependencies.source_preflight_fn,
        cell_loader=dependencies.cell_loader,
    )
    if errors:
        final_path.unlink(missing_ok=True)
        raise campaign.GlobalCampaignError(
            f"new parallel global campaign failed validation: {errors}"
        )
    return campaign_root


def _open_coordinator_loggers(
    prepared: _PreparedCampaign, dependencies: ParallelDependencies
) -> list[Any]:
    from experiments import global_campaign as campaign

    factory = (
        campaign.open_model_logger
        if dependencies.logger_factory is None
        else dependencies.logger_factory
    )
    handles: list[Any] = []
    try:
        for plan in prepared.plans:
            handle = campaign._safe_model_logger(
                factory,
                plan,
                prepared.campaign_identity,
                Path(prepared.state_dir),
                prepared.config["logging"],
            )
            handles.append(handle)
            campaign._emit(
                handle.callback,
                "model.started",
                model_plan=plan,
                campaign_identity=prepared.campaign_identity,
                expected_cells=len(prepared.cells),
            )
    except BaseException:
        for handle in handles:
            handle.close()
        raise
    return handles


def _seal_dag_models(
    prepared: _PreparedCampaign,
    states: Sequence[Mapping[str, Mapping[int, Any]]],
    handles: Sequence[Any],
) -> list[dict[str, Any]]:
    """Coordinator-only per-model aggregate, telemetry and manifest phase."""

    from experiments import global_campaign as campaign

    results: list[dict[str, Any]] = []
    for model_index, plan in enumerate(prepared.plans):
        state = states[model_index]
        if any(len(state[field]) != len(prepared.cells) for field in state):
            raise campaign.GlobalCampaignError(
                f"model DAG state is incomplete for {plan.variant_id}"
            )
        directories = {
            cell.key: Path(state["directories"][cell_index])
            for cell_index, cell in enumerate(prepared.cells)
        }
        aggregate = campaign.recompute_model_aggregate(
            plan.variant_id, prepared.cells, directories
        )
        aggregate_path = (
            Path(prepared.campaign_root)
            / "models"
            / campaign._safe_component(plan.variant_id)
            / "aggregate.json"
        )
        campaign._write_json(aggregate_path, aggregate)
        handle = handles[model_index]
        handle.asset(
            aggregate_path,
            name=f"{campaign._safe_component(plan.variant_id)}-aggregate.json",
        )
        manifest_path = aggregate_path.with_name("manifest.json")
        campaign._emit(
            handle.callback,
            "model.completed",
            model_plan=plan,
            complete_cells=len(prepared.cells),
            model_manifest_path=str(manifest_path),
            aggregate=campaign._aggregate_macros(aggregate),
        )
        handle.close()
        model_manifest = {
            "schema_version": 1,
            "campaign_identity": prepared.campaign_identity,
            "model_variant": plan.variant_id,
            "experiment_name": plan.experiment_name,
            "experiment_key": handle.experiment_key,
            "comet_telemetry": handle.telemetry_record(),
            "model_contract": {
                "variant_id": plan.variant_id,
                "experiment_name": plan.experiment_name,
                "model": campaign._plain(plan.model),
            },
            "expected_cells": len(prepared.cells),
            "complete_cells": len(prepared.cells),
            "aggregate_path": aggregate_path.relative_to(
                prepared.campaign_root
            ).as_posix(),
            "aggregate_sha256": campaign.sha256_path(aggregate_path),
            "complete": True,
        }
        campaign._write_json(manifest_path, model_manifest)
        model_record = {
            **model_manifest,
            "manifest_path": manifest_path.relative_to(
                prepared.campaign_root
            ).as_posix(),
            "manifest_sha256": campaign.sha256_path(manifest_path),
        }
        results.append(
            {
                "model_record": model_record,
                "model_aggregate": aggregate,
                "cell_records": [
                    dict(state["records"][index])
                    for index in range(len(prepared.cells))
                ],
            }
        )
    return results


def _probe_worker(
    slot: int, token: str | None, require_cuda: bool, result_queue: Any
) -> None:
    if token is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = token
    try:
        record: dict[str, Any] = {
            "slot": slot,
            "visible_token": token,
            "cuda_required": require_cuda,
        }
        if require_cuda:
            import torch

            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError("preflight worker must see exactly one CUDA GPU")
            torch.cuda.set_device(0)
            record["device_count"] = torch.cuda.device_count()
            record["device_name"] = torch.cuda.get_device_name(0)
        else:
            record["device_count"] = 0
            record["device_name"] = None
        result_queue.put({"kind": "ok", "record": record})
    except BaseException as exc:  # noqa: BLE001 - process boundary reports all
        result_queue.put(
            {
                "kind": "failed",
                "slot": slot,
                "exception_type": type(exc).__name__,
                "message": str(exc),
            }
        )


def _run_preflight_only(
    prepared: _PreparedCampaign,
    *,
    dependencies: ParallelDependencies,
    require_cuda: bool,
) -> Path:
    from experiments import global_campaign as campaign

    try:
        pickle.dumps((prepared, dependencies))
    except Exception as exc:
        raise campaign.GlobalCampaignError(
            "parallel campaign state is not spawn-picklable"
        ) from exc
    worker_count = int(prepared.config["execution"]["worker_count"])
    tokens = _device_tokens(worker_count, require_cuda=require_cuda)
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_probe_worker,
            name=f"lid-preflight-worker-{slot}",
            args=(slot, tokens[slot], require_cuda, result_queue),
        )
        for slot in range(worker_count)
    ]
    for process in processes:
        process.start()
    records: list[dict[str, Any]] = []
    seen_slots: set[int] = set()
    deadline = time.monotonic() + 60.0
    try:
        while len(records) < worker_count:
            timeout = min(1.0, deadline - time.monotonic())
            if timeout <= 0:
                raise campaign.GlobalCampaignError("parallel preflight timed out")
            try:
                message = result_queue.get(timeout=timeout)
            except queue.Empty:
                exited = [
                    process for process in processes if process.exitcode is not None
                ]
                if exited:
                    raise campaign.GlobalCampaignError(
                        "preflight worker exited without a valid report: "
                        + ", ".join(
                            f"{process.name}={process.exitcode}" for process in exited
                        )
                    )
                continue
            if not isinstance(message, Mapping):
                raise campaign.GlobalCampaignError("invalid preflight worker report")
            if message.get("kind") != "ok":
                raise campaign.GlobalCampaignError(
                    "parallel preflight worker failed: "
                    f"{message.get('exception_type')}: {message.get('message')}"
                )
            record = message.get("record")
            if not isinstance(record, Mapping) or type(record.get("slot")) is not int:
                raise campaign.GlobalCampaignError("invalid preflight worker record")
            slot = int(record["slot"])
            if not 0 <= slot < worker_count or slot in seen_slots:
                raise campaign.GlobalCampaignError("duplicate preflight worker record")
            seen_slots.add(slot)
            records.append(dict(record))
        for process in processes:
            process.join(timeout=10)
        if any(process.exitcode != 0 for process in processes):
            raise campaign.GlobalCampaignError("parallel preflight process failed")
    except BaseException:
        _terminate_workers(processes)
        raise
    campaign._write_json(
        Path(prepared.state_dir) / "parallel" / "preflight.json",
        {
            "schema_version": 1,
            "campaign_identity": prepared.campaign_identity,
            "execution": prepared.config["execution"],
            "workers": sorted(records, key=lambda row: int(row["slot"])),
            "cell_count": len(prepared.cells),
            "model_count": len(prepared.plans),
            "task_count": len(prepared.cells) * len(prepared.plans),
            "status": "ready",
        },
    )
    return Path(prepared.campaign_root)


def run_global_parallel_campaign(
    hydra_config: Any,
    *,
    root: Path | None = None,
    output_root: Path | None = None,
    dependencies: ParallelDependencies | None = None,
    require_cuda: bool | None = None,
    preflight_only: bool = False,
) -> Path:
    """Execute the 390-cell DAG on a spawn pool and seal one campaign."""

    from experiments import global_campaign as campaign

    dependencies = ParallelDependencies() if dependencies is None else dependencies
    prepared = _prepare_campaign(
        hydra_config,
        root=root,
        output_root=output_root,
        dependencies=dependencies,
    )
    if require_cuda is None:
        require_cuda = (
            prepared.config["execution"]["profile"] == campaign.EXECUTION_PROFILE_H100
        )
    if (
        prepared.config["execution"]["profile"] == campaign.EXECUTION_PROFILE_H100
        and require_cuda is not True
    ):
        raise campaign.GlobalCampaignError("H100 profile cannot disable CUDA checks")
    state_dir = Path(prepared.state_dir)
    campaign_root = Path(prepared.campaign_root)
    lock_path = state_dir / str(prepared.config["campaign"]["resume"]["lock_filename"])
    final_path = campaign_root / "campaign.json"
    logger_factory = (
        campaign.open_model_logger
        if dependencies.logger_factory is None
        else dependencies.logger_factory
    )
    if final_path.exists():
        errors = campaign.validate_global_campaign(
            campaign_root,
            expected_campaign_identity=prepared.campaign_identity,
            project_root=Path(prepared.project_root),
            source_preflight_fn=dependencies.source_preflight_fn,
            cell_loader=dependencies.cell_loader,
        )
        if errors:
            raise campaign.GlobalCampaignError(
                f"existing final global campaign is invalid: {errors}"
            )
        with campaign._exclusive_campaign_lock(lock_path):
            for model_plan in prepared.plans:
                handle = campaign._safe_model_logger(
                    logger_factory,
                    model_plan,
                    prepared.campaign_identity,
                    state_dir,
                    prepared.config["logging"],
                )
                handle.close()
        return campaign_root

    with campaign._exclusive_campaign_lock(lock_path):
        campaign._write_yaml(campaign_root / "resolved_config.yaml", prepared.config)
        if preflight_only:
            return _run_preflight_only(
                prepared,
                dependencies=dependencies,
                require_cuda=bool(require_cuda),
            )
        handles = _open_coordinator_loggers(prepared, dependencies)
        try:
            states, _assignments = _run_cell_dag_pool(
                prepared,
                dependencies=dependencies,
                require_cuda=bool(require_cuda),
                handles=handles,
            )
            results = _seal_dag_models(prepared, states, handles)
            return _finalize_campaign(prepared, results, dependencies=dependencies)
        except BaseException:
            for handle in handles:
                if getattr(handle, "_close_status", "not_closed") == "not_closed":
                    handle.close()
            raise


def _hydra_main() -> None:
    from experiments.global_campaign import compose_global_campaign_config

    preflight_only = "--preflight-only" in sys.argv[1:]
    overrides = tuple(
        argument for argument in sys.argv[1:] if argument != "--preflight-only"
    )
    config = compose_global_campaign_config(overrides)
    output = run_global_parallel_campaign(
        with_h100_profile(config), preflight_only=preflight_only
    )
    print(output)


def main() -> None:
    _hydra_main()


if __name__ == "__main__":
    main()
