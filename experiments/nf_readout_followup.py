"""Readout-only NF follow-up selected from the completed stage-2 evidence.

The original NF ablation intentionally stopped because its promoted training
candidate did not pass the predeclared stage-2 quality gates.  This module does
not reinterpret that failure and does not relax those gates.  It performs a
separate, explicitly post-hoc follow-up:

* verify the complete sealed stage-2 evidence and its failed training gate;
* freeze OLS5 only when the C0/OLS5 readout beats C0/autograd under the same
  paired canonical-cell gate;
* train only the unchanged C0 architecture for seeds 2 and 3 on all 39 cells;
* emit C0/s2/autograd, C0/s2/OLS5 and C0/s3/OLS5 as 234 validation/test rows;
* append those rows while preserving the baseline CSV as an exact byte prefix.

Seed 2 is trained once.  Its worker evaluates both frozen test readouts from the
same checkpoint, avoiding a scientifically pointless duplicate training run.
Every checkpoint is pruned immediately after inline evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from experiments import global_campaign as campaign
from experiments import nf_ablation as nf
from experiments.metrics import (
    known_lid_metrics,
    paired_delta_metrics,
    prediction_summary,
)

SCHEMA_VERSION = 1
FOLLOWUP_ID = "nf-c0-ols5-readout-followup-v1"
CONTROL_CANDIDATE = "C0"
FROZEN_READOUT = "ols5"
REFERENCE_READOUT = "autograd"
CANONICAL_RATIO_LIMIT = 0.85
CANONICAL_MINIMUM_WINS = 10
CANONICAL_MAXIMUM_REGRESSIONS = 2
EXPECTED_KNOWN_CELLS = 19
EXPECTED_ALL_CELLS = 39
EXPECTED_TRAINING_TASKS = 2 * EXPECTED_ALL_CELLS
EXPECTED_EXTENSION_ROWS = 3 * EXPECTED_ALL_CELLS * 2
EXPECTED_BASELINE_ROWS = 1872
EXPECTED_COMBINED_ROWS = EXPECTED_BASELINE_ROWS + EXPECTED_EXTENSION_ROWS
EXPECTED_STAGE2_RECORDS = 2 * 2 * EXPECTED_KNOWN_CELLS * len(nf.READOUTS)
EXPECTED_STAGE1_TASKS = len(nf.STAGE1_CANDIDATES) * len(nf.STAGE1_SENTINEL_KEYS)
EXPECTED_STAGE1_RECORDS = (len(nf.CANDIDATES) * len(nf.READOUTS) + 1) * len(
    nf.STAGE1_SENTINEL_KEYS
)
ROW_KEY_FIELDS = (
    "model_variant",
    "analysis",
    "suite_id",
    "dataset",
    "representation",
    "split",
    "readout",
)
BUNDLED_EVIDENCE_PATHS = {
    "identity_sha256": "evidence/source_ablation_identity.json",
    "stage1_ledger_sha256": "evidence/stage1_ledger.json",
    "stage1_scores_sha256": "evidence/stage1_validation_scores.json",
    "promotions_sha256": "evidence/stage1_promotions.json",
    "ledger_sha256": "evidence/stage2_ledger.json",
    "scores_sha256": "evidence/stage2_validation_scores.json",
}


class NFReadoutFollowupError(nf.NFAblationError):
    """Raised before a readout-only follow-up can become ambiguous."""


@dataclass(frozen=True)
class GateAudit:
    candidate_id: str
    readout: str
    baseline_candidate_id: str
    baseline_readout: str
    canonical_geometric_mean_ratio: float
    canonical_wins: int
    canonical_regressions_over_25pct: int
    generated_geometric_mean_ratio: float
    passed: bool


@dataclass(frozen=True)
class ReviewedEvidence:
    source_campaign_identity: str
    source_identity_sha256: str
    source_baseline_campaign_identity: str
    source_baseline_unified_sha256: str
    stage1_ledger_sha256: str
    stage1_scores_sha256: str
    promotions_sha256: str
    stage2_ledger_sha256: str
    stage2_scores_sha256: str


@dataclass(frozen=True)
class Stage2Evidence:
    source_root: Path
    source_campaign_identity: str
    source_baseline_campaign_identity: str
    source_baseline_unified_sha256: str
    identity_sha256: str
    stage1_ledger_sha256: str
    stage1_scores_sha256: str
    stage2_ledger_sha256: str
    stage2_scores_sha256: str
    promotions_sha256: str
    completed_tasks: int
    validation_records: int
    promoted_candidate_id: str
    promoted_readout: str
    failed_training_gate: GateAudit
    frozen_readout_gate: GateAudit


@dataclass(frozen=True)
class OutputVariant:
    variant_id: str
    candidate_id: str
    seed: int
    readout: str


OUTPUT_VARIANTS = (
    OutputVariant("c0_seed2_autograd_reference", "C0", 2, "autograd"),
    OutputVariant("c0_seed2_ols5", "C0", 2, "ols5"),
    OutputVariant("c0_seed3_ols5", "C0", 3, "ols5"),
)


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NFReadoutFollowupError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, Mapping):
        raise NFReadoutFollowupError(f"{label} is not a JSON object: {path}")
    return value


def _finite_loss(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise NFReadoutFollowupError(f"{label} is not numeric") from exc
    if not math.isfinite(result) or result < 0.0:
        raise NFReadoutFollowupError(f"{label} is not a finite non-negative loss")
    return result


def load_reviewed_evidence(path: str | Path) -> ReviewedEvidence:
    """Load the independently reviewed hashes that authorize this follow-up."""

    source = Path(path).expanduser().resolve()
    payload = _read_json(source, label="reviewed evidence manifest")
    expected_fields = {
        "schema_version",
        *(field.name for field in ReviewedEvidence.__dataclass_fields__.values()),
    }
    if (
        set(payload) != expected_fields
        or payload.get("schema_version") != SCHEMA_VERSION
    ):
        raise NFReadoutFollowupError("reviewed evidence manifest schema differs")
    values = {
        field: str(payload[field]) for field in expected_fields - {"schema_version"}
    }
    if any(
        len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in values.values()
    ):
        raise NFReadoutFollowupError(
            "reviewed evidence manifest contains an invalid SHA"
        )
    return ReviewedEvidence(**values)


def _gate_audit(
    indexed: Mapping[tuple[str, str, int, str], float],
    *,
    candidate_id: str,
    readout: str,
    baseline_candidate_id: str,
    baseline_readout: str,
) -> GateAudit:
    known_cells = tuple(
        key
        for key in campaign.APPROVED_GLOBAL_CELL_KEYS
        if not key.startswith("e1/e1_sampled") and not key.startswith("e5/")
    )
    if len(known_cells) != EXPECTED_KNOWN_CELLS:
        raise NFReadoutFollowupError("known-LID inventory differs from the contract")
    canonical = tuple(key for key in known_cells if not key.startswith(("e3/", "e4/")))
    generated = tuple(key for key in known_cells if key not in canonical)
    if len(canonical) != 15 or len(generated) != 4:
        raise NFReadoutFollowupError("canonical/generated stage-2 split differs")
    epsilon = 1.0e-12
    log_ratio_by_cell: dict[str, float] = {}
    for cell_key in known_cells:
        values: list[float] = []
        for seed in (0, 1):
            candidate_key = (candidate_id, readout, seed, cell_key)
            baseline_key = (
                baseline_candidate_id,
                baseline_readout,
                seed,
                cell_key,
            )
            try:
                candidate_loss = indexed[candidate_key]
                baseline_loss = indexed[baseline_key]
            except KeyError as exc:
                raise NFReadoutFollowupError(
                    "stage-2 evidence lacks paired gate coverage"
                ) from exc
            values.append(
                math.log(max(candidate_loss, epsilon))
                - math.log(max(baseline_loss, epsilon))
            )
        log_ratio_by_cell[cell_key] = float(np.mean(values))
    canonical_ratios = [math.exp(log_ratio_by_cell[key]) for key in canonical]
    generated_logs = [log_ratio_by_cell[key] for key in generated]
    ratio = math.exp(float(np.mean([log_ratio_by_cell[key] for key in canonical])))
    wins = sum(value < 1.0 for value in canonical_ratios)
    regressions = sum(value > 1.25 for value in canonical_ratios)
    passed = (
        ratio <= CANONICAL_RATIO_LIMIT
        and wins >= CANONICAL_MINIMUM_WINS
        and regressions <= CANONICAL_MAXIMUM_REGRESSIONS
    )
    return GateAudit(
        candidate_id=candidate_id,
        readout=readout,
        baseline_candidate_id=baseline_candidate_id,
        baseline_readout=baseline_readout,
        canonical_geometric_mean_ratio=ratio,
        canonical_wins=wins,
        canonical_regressions_over_25pct=regressions,
        generated_geometric_mean_ratio=math.exp(float(np.mean(generated_logs))),
        passed=passed,
    )


def load_stage2_evidence(
    source_ablation_root: str | Path,
    *,
    reviewed: ReviewedEvidence,
    expected_baseline_campaign_identity: str | None = None,
    expected_baseline_unified_sha256: str | None = None,
) -> Stage2Evidence:
    """Strictly verify stage 2 and freeze OLS5 without changing training gates."""

    root = Path(source_ablation_root).expanduser().resolve()
    identity_path = root / "ablation_identity.json"
    stage1_ledger_path = root / "state" / "stage1" / "ledger.json"
    stage1_scores_path = root / "state" / "stage1" / "validation_scores.json"
    ledger_path = root / "state" / "stage2" / "ledger.json"
    scores_path = root / "state" / "stage2" / "validation_scores.json"
    promotions_path = root / "state" / "stage1" / "promotions.json"
    reviewed_paths = {
        "source_identity_sha256": identity_path,
        "stage1_ledger_sha256": stage1_ledger_path,
        "stage1_scores_sha256": stage1_scores_path,
        "promotions_sha256": promotions_path,
        "stage2_ledger_sha256": ledger_path,
        "stage2_scores_sha256": scores_path,
    }
    for field, path in reviewed_paths.items():
        if not path.is_file() or nf._sha256_path(path) != getattr(reviewed, field):
            raise NFReadoutFollowupError(
                f"source evidence differs from reviewed hash: {field}"
            )
    identity = _read_json(identity_path, label="source ablation identity")
    stage1_ledger = _read_json(stage1_ledger_path, label="source stage-1 ledger")
    stage1_scores = _read_json(stage1_scores_path, label="source stage-1 scores")
    ledger = _read_json(ledger_path, label="source stage-2 ledger")
    score_payload = _read_json(scores_path, label="source stage-2 scores")
    promotion_payload = _read_json(promotions_path, label="source promotions")
    source_identity = nf._canonical_sha(identity)
    if source_identity != reviewed.source_campaign_identity:
        raise NFReadoutFollowupError("source campaign identity differs from input")
    if any(
        payload.get("campaign_identity") != source_identity
        for payload in (
            stage1_ledger,
            stage1_scores,
            ledger,
            score_payload,
            promotion_payload,
        )
    ):
        raise NFReadoutFollowupError("source stage evidence identity differs")
    baseline_identity = str(identity.get("baseline_campaign_identity", ""))
    baseline_sha = str(identity.get("baseline_unified_sha256", ""))
    if (
        not baseline_identity
        or len(baseline_sha) != 64
        or baseline_identity != reviewed.source_baseline_campaign_identity
        or baseline_sha != reviewed.source_baseline_unified_sha256
        or (
            expected_baseline_campaign_identity is not None
            and baseline_identity != expected_baseline_campaign_identity
        )
        or (
            expected_baseline_unified_sha256 is not None
            and baseline_sha != expected_baseline_unified_sha256
        )
    ):
        raise NFReadoutFollowupError(
            "source stage evidence belongs to another baseline"
        )

    promotions = promotion_payload.get("promotions")
    if not isinstance(promotions, list) or len(promotions) != 1:
        raise NFReadoutFollowupError(
            "readout follow-up requires the exact single stage-1 promotion"
        )
    promotion = promotions[0]
    if not isinstance(promotion, Mapping):
        raise NFReadoutFollowupError("source promotion row is malformed")
    promoted_id = str(promotion.get("candidate_id"))
    promoted_readout = str(promotion.get("readout"))
    if promoted_id != "C1" or promoted_readout != FROZEN_READOUT:
        raise NFReadoutFollowupError(
            "source promotion is not the reviewed C1/OLS5 decision"
        )

    stage1_completed = stage1_ledger.get("completed_tasks")
    expected_stage1_tasks = {
        (candidate.candidate_id, 0, cell_key)
        for candidate in nf.STAGE1_CANDIDATES
        for cell_key in nf.STAGE1_SENTINEL_KEYS
    }
    if (
        len(expected_stage1_tasks) != EXPECTED_STAGE1_TASKS
        or stage1_ledger.get("complete") is not True
        or stage1_ledger.get("expected_task_count") != EXPECTED_STAGE1_TASKS
        or not isinstance(stage1_completed, list)
        or len(stage1_completed) != EXPECTED_STAGE1_TASKS
    ):
        raise NFReadoutFollowupError("source stage-1 ledger is not complete")
    observed_stage1 = {
        (
            str(row.get("candidate_id")),
            int(row.get("seed", -1)),
            str(row.get("cell_key")),
        )
        for row in stage1_completed
        if isinstance(row, Mapping)
    }
    if (
        len(observed_stage1) != len(stage1_completed)
        or observed_stage1 != expected_stage1_tasks
    ):
        raise NFReadoutFollowupError("source stage-1 task coverage differs")
    for raw in stage1_completed:
        assert isinstance(raw, Mapping)
        try:
            directory = (root / str(raw["path"])).resolve()
            directory.relative_to(root)
            if (
                not directory.is_dir()
                or nf._sha256_path(directory / "manifest.json")
                != raw["manifest_sha256"]
                or nf._sha256_path(directory / "summary.json") != raw["summary_sha256"]
                or raw.get("evaluate_test") is not False
                or raw.get("test_readout") is not None
                or list(directory.glob("test_*"))
                or list(directory.rglob("checkpoint.pt"))
                or list(directory.rglob("training_progress.pt"))
            ):
                raise NFReadoutFollowupError("source stage-1 seal differs")
        except (KeyError, OSError, TypeError, ValueError) as exc:
            if isinstance(exc, NFReadoutFollowupError):
                raise
            raise NFReadoutFollowupError("source stage-1 seal is malformed") from exc
    stage1_records = stage1_scores.get("records")
    expected_stage1_scores = {
        (candidate.candidate_id, readout, 0, cell_key)
        for candidate in nf.STAGE1_CANDIDATES
        for readout in (
            (nf.PAPER_PARITY_READOUT,)
            if candidate.independent_fixed_epsilon
            else nf.READOUTS
        )
        for cell_key in nf.STAGE1_SENTINEL_KEYS
    }
    if (
        len(expected_stage1_scores) != EXPECTED_STAGE1_RECORDS
        or not isinstance(stage1_records, list)
        or len(stage1_records) != EXPECTED_STAGE1_RECORDS
    ):
        raise NFReadoutFollowupError("source stage-1 score coverage is incomplete")
    observed_stage1_scores: set[tuple[str, str, int, str]] = set()
    for raw in stage1_records:
        if not isinstance(raw, Mapping):
            raise NFReadoutFollowupError("source stage-1 score row is malformed")
        key = (
            str(raw.get("candidate_id")),
            str(raw.get("readout")),
            int(raw.get("seed", -1)),
            str(raw.get("cell_key")),
        )
        if (
            key in observed_stage1_scores
            or raw.get("split") != "validation"
            or float(raw.get("finite_fraction", 0.0)) != 1.0
        ):
            raise NFReadoutFollowupError("source stage-1 score row differs")
        _finite_loss(raw.get("loss"), label="stage-1 loss")
        observed_stage1_scores.add(key)
    if observed_stage1_scores != expected_stage1_scores:
        raise NFReadoutFollowupError("source stage-1 score matrix differs")

    completed = ledger.get("completed_tasks")
    expected_tasks = {
        (candidate_id, seed, cell_key)
        for candidate_id in (CONTROL_CANDIDATE, promoted_id)
        for seed in (0, 1)
        for cell_key in campaign.APPROVED_GLOBAL_CELL_KEYS
        if not cell_key.startswith("e1/e1_sampled") and not cell_key.startswith("e5/")
    }
    if (
        ledger.get("complete") is not True
        or ledger.get("expected_task_count") != len(expected_tasks)
        or not isinstance(completed, list)
        or len(completed) != len(expected_tasks)
    ):
        raise NFReadoutFollowupError("source stage-2 ledger is not complete")
    observed_tasks: set[tuple[str, int, str]] = set()
    for raw in completed:
        if not isinstance(raw, Mapping):
            raise NFReadoutFollowupError("source stage-2 ledger row is malformed")
        key = (
            str(raw.get("candidate_id")),
            int(raw.get("seed", -1)),
            str(raw.get("cell_key")),
        )
        if key in observed_tasks:
            raise NFReadoutFollowupError("source stage-2 ledger repeats a task")
        observed_tasks.add(key)
        if raw.get("evaluate_test") is not False or raw.get("test_readout") is not None:
            raise NFReadoutFollowupError("source stage 2 opened test data")
        try:
            directory = (root / str(raw["path"])).resolve()
            directory.relative_to(root)
            manifest_path = directory / "manifest.json"
            summary_path = directory / "summary.json"
            if (
                not directory.is_dir()
                or nf._sha256_path(manifest_path) != raw["manifest_sha256"]
                or nf._sha256_path(summary_path) != raw["summary_sha256"]
            ):
                raise NFReadoutFollowupError("source stage-2 seal differs")
            summary = _read_json(summary_path, label="source stage-2 summary")
            if (
                summary.get("candidate_id") != key[0]
                or summary.get("seed") != key[1]
                or summary.get("cell_key") != key[2]
                or summary.get("evaluation_splits") != ["validation"]
            ):
                raise NFReadoutFollowupError("source stage-2 summary differs")
            if list(directory.glob("test_*")):
                raise NFReadoutFollowupError("source stage 2 contains test artifacts")
            if list(directory.rglob("checkpoint.pt")) or list(
                directory.rglob("training_progress.pt")
            ):
                raise NFReadoutFollowupError(
                    "source stage-2 cell retained a checkpoint"
                )
        except (KeyError, OSError, TypeError, ValueError) as exc:
            if isinstance(exc, NFReadoutFollowupError):
                raise
            raise NFReadoutFollowupError("source stage-2 seal is malformed") from exc
    if observed_tasks != expected_tasks:
        raise NFReadoutFollowupError("source stage-2 task coverage differs")

    records = score_payload.get("records")
    expected_scores = {
        (candidate_id, readout, seed, cell_key)
        for candidate_id, seed, cell_key in expected_tasks
        for readout in nf.READOUTS
    }
    if (
        len(expected_scores) != EXPECTED_STAGE2_RECORDS
        or not isinstance(records, list)
        or len(records) != EXPECTED_STAGE2_RECORDS
    ):
        raise NFReadoutFollowupError("source stage-2 score coverage is incomplete")
    indexed: dict[tuple[str, str, int, str], float] = {}
    for raw in records:
        if not isinstance(raw, Mapping):
            raise NFReadoutFollowupError("source stage-2 score row is malformed")
        key = (
            str(raw.get("candidate_id")),
            str(raw.get("readout")),
            int(raw.get("seed", -1)),
            str(raw.get("cell_key")),
        )
        if (
            key in indexed
            or raw.get("split") != "validation"
            or raw.get("target_policy") != "known_lid"
            or float(raw.get("finite_fraction", 0.0)) != 1.0
        ):
            raise NFReadoutFollowupError("source stage-2 score row differs")
        indexed[key] = _finite_loss(raw.get("loss"), label="stage-2 loss")
    if set(indexed) != expected_scores:
        raise NFReadoutFollowupError("source stage-2 score matrix differs")

    training_gate = _gate_audit(
        indexed,
        candidate_id=promoted_id,
        readout=promoted_readout,
        baseline_candidate_id=CONTROL_CANDIDATE,
        baseline_readout=promoted_readout,
    )
    if training_gate.passed:
        raise NFReadoutFollowupError(
            "source training candidate passed; use the original stage-3 contract"
        )
    readout_gate = _gate_audit(
        indexed,
        candidate_id=CONTROL_CANDIDATE,
        readout=FROZEN_READOUT,
        baseline_candidate_id=CONTROL_CANDIDATE,
        baseline_readout=REFERENCE_READOUT,
    )
    if not readout_gate.passed:
        raise NFReadoutFollowupError("C0/OLS5 did not pass the readout-only gate")
    return Stage2Evidence(
        source_root=root,
        source_campaign_identity=source_identity,
        source_baseline_campaign_identity=baseline_identity,
        source_baseline_unified_sha256=baseline_sha,
        identity_sha256=nf._sha256_path(identity_path),
        stage1_ledger_sha256=nf._sha256_path(stage1_ledger_path),
        stage1_scores_sha256=nf._sha256_path(stage1_scores_path),
        stage2_ledger_sha256=nf._sha256_path(ledger_path),
        stage2_scores_sha256=nf._sha256_path(scores_path),
        promotions_sha256=nf._sha256_path(promotions_path),
        completed_tasks=len(completed),
        validation_records=len(records),
        promoted_candidate_id=promoted_id,
        promoted_readout=promoted_readout,
        failed_training_gate=training_gate,
        frozen_readout_gate=readout_gate,
    )


def _identity_record(
    preflight: nf.BaselinePreflight,
    evidence: Stage2Evidence,
    *,
    project_root: Path,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "followup_id": FOLLOWUP_ID,
        "scientific_scope": "post_hoc_readout_only_no_training_gate_relaxation",
        "baseline_campaign_identity": preflight.baseline_campaign_identity,
        "baseline_unified_sha256": preflight.baseline_unified_sha256,
        "baseline_input_sha256_by_key": dict(preflight.input_sha256_by_key),
        "declared_source_sha256": campaign.hash_declared_sources(project_root),
        "coordinator_sha256": nf._sha256_path(Path(__file__).resolve()),
        "nf_ablation_engine_sha256": nf._sha256_path(Path(nf.__file__).resolve()),
        "source_stage2_evidence": {
            "campaign_identity": evidence.source_campaign_identity,
            "identity_sha256": evidence.identity_sha256,
            "stage1_ledger_sha256": evidence.stage1_ledger_sha256,
            "stage1_scores_sha256": evidence.stage1_scores_sha256,
            "stage2_ledger_sha256": evidence.stage2_ledger_sha256,
            "stage2_scores_sha256": evidence.stage2_scores_sha256,
            "promotions_sha256": evidence.promotions_sha256,
            "completed_tasks": evidence.completed_tasks,
            "validation_records": evidence.validation_records,
        },
        "failed_training_gate": nf._plain(evidence.failed_training_gate),
        "frozen_readout_gate": nf._plain(evidence.frozen_readout_gate),
        "frozen_contract": {
            "candidate_id": CONTROL_CANDIDATE,
            "training_contract": nf.candidate_by_id(CONTROL_CANDIDATE).contract,
            "reference_readout": REFERENCE_READOUT,
            "frozen_readout": FROZEN_READOUT,
            "selection_status": "frozen_before_followup_test_evaluation",
            "test_data_used_for_selection": False,
        },
        "output_variants": [asdict(value) for value in OUTPUT_VARIANTS],
        "execution": {
            "candidate_seed_tasks": [
                {
                    "candidate_id": "C0",
                    "seed": 2,
                    "test_readouts": ["autograd", "ols5"],
                },
                {"candidate_id": "C0", "seed": 3, "test_readouts": ["ols5"]},
            ],
            "cell_policy": "all_39_cells",
            "training_tasks": EXPECTED_TRAINING_TASKS,
            "output_rows": EXPECTED_EXTENSION_ROWS,
            "expected_combined_rows": EXPECTED_COMBINED_ROWS,
            "checkpoint_retention": "pruned_after_inline_evaluation",
        },
    }


def _prepare(
    preflight: nf.BaselinePreflight,
    evidence: Stage2Evidence,
    *,
    project_root: Path,
    output_root: Path,
    worker_count: int,
) -> nf._PreparedNFAblation:
    if isinstance(worker_count, bool) or not 1 <= worker_count <= nf.WORKER_COUNT:
        raise NFReadoutFollowupError(f"worker_count must be in [1, {nf.WORKER_COUNT}]")
    identity_record = _identity_record(preflight, evidence, project_root=project_root)
    identity = nf._canonical_sha(identity_record)
    selected_output = output_root.expanduser()
    if not selected_output.is_absolute():
        selected_output = project_root / selected_output
    root = selected_output.resolve() / f"{FOLLOWUP_ID}__{identity[:20]}"
    return nf._PreparedNFAblation(
        project_root=str(project_root),
        campaign_root=str(root),
        campaign_identity=identity,
        identity_record=identity_record,
        preflight=preflight,
        worker_count=worker_count,
    )


def _followup_tasks(prepared: nf._PreparedNFAblation) -> tuple[nf.NFCellTask, ...]:
    if len(prepared.preflight.cells) != EXPECTED_ALL_CELLS:
        raise NFReadoutFollowupError("follow-up requires the exact 39-cell inventory")
    task_specs: tuple[tuple[int, str | tuple[str, ...]], ...] = (
        (2, (REFERENCE_READOUT, FROZEN_READOUT)),
        (3, FROZEN_READOUT),
    )
    tasks = tuple(
        nf._make_cell_task(
            prepared,
            cell=cell,
            candidate_id=CONTROL_CANDIDATE,
            seed=seed,
            evaluate_test=True,
            test_readout=readouts,
        )
        for seed, readouts in task_specs
        for cell in prepared.preflight.cells
    )
    if len(tasks) != EXPECTED_TRAINING_TASKS:
        raise NFReadoutFollowupError("follow-up task matrix differs")
    return tasks


def _empty_row() -> dict[str, Any]:
    return {field: "" for field in campaign._UNIFIED_TABLE_FIELDS}


def build_followup_unified_rows(
    results: Sequence[Mapping[str, Any]],
    cells: Sequence[Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Render the three frozen output variants from two training seeds."""

    expected_task_keys = {(seed, cell.key) for seed in (2, 3) for cell in cells}
    summaries: dict[tuple[int, str], Mapping[str, Any]] = {}
    directories: dict[tuple[int, str], Path] = {}
    for result in results:
        if str(result.get("candidate_id")) != CONTROL_CANDIDATE:
            raise NFReadoutFollowupError("follow-up result is not a C0 task")
        key = (int(result.get("seed", -1)), str(result.get("cell_key")))
        if key in summaries:
            raise NFReadoutFollowupError("follow-up repeats a training task")
        summaries[key] = dict(result["summary"])
        directories[key] = Path(str(result["directory"]))
    if set(summaries) != expected_task_keys:
        raise NFReadoutFollowupError("follow-up lacks exact 2 x 39 task coverage")

    rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for variant in OUTPUT_VARIANTS:
        for cell in cells:
            key = (variant.seed, cell.key)
            summary = summaries[key]
            directory = directories[key]
            try:
                selected = summary["readouts"][variant.readout]
            except (KeyError, TypeError) as exc:
                raise NFReadoutFollowupError(
                    f"frozen readout is absent for {variant.variant_id}/{cell.key}"
                ) from exc
            trained_variant = str(summary["model_variant"])
            output_model_variant = f"{trained_variant}-readout-{variant.readout}"
            for split in ("validation", "test"):
                prediction_path = (
                    directory / f"{split}_prediction__{variant.readout}.npy"
                )
                prediction = np.asarray(
                    np.load(prediction_path, allow_pickle=False), dtype=np.float64
                )
                if not np.isfinite(prediction).all():
                    raise NFReadoutFollowupError("final prediction is non-finite")
                row = _empty_row()
                row.update(
                    {
                        "model_variant": output_model_variant,
                        "suite_id": cell.suite_id,
                        "dataset": cell.dataset,
                        "representation": cell.representation,
                        "split": split,
                        "readout": variant.readout,
                        "primary_readout": variant.readout,
                        "is_primary_readout": True,
                        "selected_coordinate_name": "scale",
                        "selected_index": (
                            ""
                            if selected["selected_index"] is None
                            else selected["selected_index"]
                        ),
                        "selected_coordinate": (
                            ""
                            if selected["selected_scale"] is None
                            else selected["selected_scale"]
                        ),
                        "n_source_train": summary["partition"]["n_source_train"],
                        "expected_lid_delta": (
                            ""
                            if cell.expected_lid_delta is None
                            else cell.expected_lid_delta
                        ),
                    }
                )
                row_evidence = {
                    "prediction_path": str(prediction_path),
                    "prediction_sha256": nf._sha256_path(prediction_path),
                    "target_path": None,
                    "target_sha256": None,
                    "reference_prediction_path": None,
                    "reference_prediction_sha256": None,
                    "labels_path": None,
                    "labels_sha256": None,
                    "reference_labels_path": None,
                    "reference_labels_sha256": None,
                }
                if cell.target_policy == "known_lid":
                    row["analysis"] = "known_lid"
                    row["selection_protocol"] = campaign.KNOWN_SELECTION_PROTOCOL
                    target_path = directory / f"{split}_target.npy"
                    target = np.load(target_path, allow_pickle=False)
                    nf._metric_fields(row, known_lid_metrics(prediction, target))
                    row_evidence.update(
                        target_path=str(target_path),
                        target_sha256=nf._sha256_path(target_path),
                    )
                else:
                    reference_cell = campaign._reference_cell(cells, cell)
                    reference_directory = directories[
                        (variant.seed, reference_cell.key)
                    ]
                    reference_prediction = np.asarray(
                        np.load(
                            reference_directory
                            / f"{split}_prediction__{variant.readout}.npy",
                            allow_pickle=False,
                        ),
                        dtype=np.float64,
                    )
                    reference_prediction_path = (
                        reference_directory
                        / f"{split}_prediction__{variant.readout}.npy"
                    )
                    row_evidence.update(
                        reference_prediction_path=str(reference_prediction_path),
                        reference_prediction_sha256=nf._sha256_path(
                            reference_prediction_path
                        ),
                    )
                    row["reference_dataset"] = reference_cell.dataset
                    row["selection_protocol"] = campaign.UNKNOWN_SELECTION_PROTOCOL
                    if cell.target_policy == "sample_size":
                        row["analysis"] = "e1_sample_size_stability"
                        current_metrics = prediction_summary(prediction)
                        reference_metrics = prediction_summary(reference_prediction)
                        nf._metric_fields(row, current_metrics)
                        row["reference_mean"] = reference_metrics["mean"]
                        row["reference_median"] = reference_metrics["median"]
                        row["mean_delta_from_reference"] = float(
                            current_metrics["mean"] - reference_metrics["mean"]
                        )
                        row["median_delta_from_reference"] = float(
                            current_metrics["median"] - reference_metrics["median"]
                        )
                        row["mean_delta_error"] = float(
                            row["mean_delta_from_reference"]
                            - float(cell.expected_lid_delta)
                        )
                    elif cell.target_policy == "paired_delta":
                        row["analysis"] = "e5_paired_delta"
                        metrics = paired_delta_metrics(
                            reference_prediction,
                            prediction,
                            expected_delta=float(cell.expected_lid_delta),
                        )
                        nf._metric_fields(row, metrics)
                        labels_path = directory / f"{split}_labels.npy"
                        reference_labels_path = (
                            reference_directory / f"{split}_labels.npy"
                        )
                        labels = np.load(labels_path, allow_pickle=False)
                        reference_labels = np.load(
                            reference_labels_path, allow_pickle=False
                        )
                        if not np.array_equal(labels, reference_labels):
                            raise NFReadoutFollowupError(
                                f"paired labels differ for {cell.key}/{split}"
                            )
                        row["labels_sha256"] = campaign._array_sha(labels)
                        row_evidence.update(
                            labels_path=str(labels_path),
                            labels_sha256=nf._sha256_path(labels_path),
                            reference_labels_path=str(reference_labels_path),
                            reference_labels_sha256=nf._sha256_path(
                                reference_labels_path
                            ),
                        )
                    else:
                        raise NFReadoutFollowupError(
                            f"unsupported target policy {cell.target_policy}"
                        )
                rows.append(row)
                provenance.append(
                    {
                        "output_variant_id": variant.variant_id,
                        "model_variant": output_model_variant,
                        "trained_model_variant": trained_variant,
                        "analysis": row["analysis"],
                        "suite_id": cell.suite_id,
                        "dataset": cell.dataset,
                        "representation": cell.representation,
                        "split": split,
                        "readout": variant.readout,
                        "candidate_id": variant.candidate_id,
                        "seed": variant.seed,
                        "target_policy": cell.target_policy,
                        "expected_lid_delta": cell.expected_lid_delta,
                        "reference_dataset": cell.reference_dataset,
                        "stage": "readout_followup",
                        "evaluation_role": (
                            "validation_selection_conditioned"
                            if split == "validation"
                            else "test_confirmatory"
                        ),
                        "run_id": summary["run_id"],
                        "run_id_contract": summary["run_id_contract"],
                        "cell_id": summary["cell_id"],
                        "cell_path": str(directory),
                        "manifest_sha256": nf._sha256_path(directory / "manifest.json"),
                        "summary_sha256": nf._sha256_path(directory / "summary.json"),
                        "training_attestation_sha256": summary[
                            "training_attestation_sha256"
                        ],
                        "row_evidence": row_evidence,
                        "validation_status": "sealed_complete",
                    }
                )
    if len(rows) != EXPECTED_EXTENSION_ROWS or len(provenance) != len(rows):
        raise NFReadoutFollowupError("follow-up output matrix is incomplete")
    return rows, provenance


def _row_key(row: Mapping[str, Any]) -> dict[str, str]:
    return {field: str(row[field]) for field in ROW_KEY_FIELDS}


def _csv_row_sha256(row: Mapping[str, Any]) -> str:
    normalized = {field: str(row[field]) for field in campaign._UNIFIED_TABLE_FIELDS}
    return nf._canonical_sha(normalized)


def _build_row_audit(
    *,
    extension_path: Path,
    provenance_path: Path,
    provenance_records: Sequence[Mapping[str, Any]],
    campaign_identity: str,
) -> dict[str, Any]:
    header, csv_rows = nf._read_csv(extension_path)
    if tuple(header) != tuple(campaign._UNIFIED_TABLE_FIELDS) or len(csv_rows) != len(
        provenance_records
    ):
        raise NFReadoutFollowupError("row audit input coverage differs")
    audit_records: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for ordinal, (row, provenance) in enumerate(
        zip(csv_rows, provenance_records, strict=True)
    ):
        row_key = _row_key(row)
        provenance_key = _row_key(provenance)
        key_tuple = tuple(row_key[field] for field in ROW_KEY_FIELDS)
        if row_key != provenance_key or key_tuple in seen:
            raise NFReadoutFollowupError("CSV/provenance row keys differ")
        seen.add(key_tuple)
        evidence = provenance.get("row_evidence")
        if not isinstance(evidence, Mapping):
            raise NFReadoutFollowupError("provenance lacks pointwise evidence hashes")
        audit_records.append(
            {
                "ordinal": ordinal,
                "row_key": row_key,
                "csv_row_sha256": _csv_row_sha256(row),
                "provenance_record_sha256": nf._canonical_sha(provenance),
                "output_variant_id": provenance["output_variant_id"],
                "run_id": provenance["run_id"],
                "cell_id": provenance["cell_id"],
                "evidence": dict(evidence),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_identity": campaign_identity,
        "extension_path": extension_path.name,
        "extension_sha256": nf._sha256_path(extension_path),
        "provenance_path": provenance_path.name,
        "provenance_sha256": nf._sha256_path(provenance_path),
        "row_key_fields": list(ROW_KEY_FIELDS),
        "records": audit_records,
    }


def _aggregate_followup_rows(
    rows: Sequence[Mapping[str, Any]], evidence: Stage2Evidence
) -> dict[str, Any]:
    count_by_analysis: dict[str, int] = {}
    count_by_suite: dict[str, int] = {}
    count_by_variant: dict[str, int] = {}
    summaries: list[dict[str, Any]] = []
    for row in rows:
        for target, field in (
            (count_by_analysis, "analysis"),
            (count_by_suite, "suite_id"),
            (count_by_variant, "model_variant"),
        ):
            key = str(row[field])
            target[key] = target.get(key, 0) + 1
    metric_fields = (
        "mean",
        "median",
        "mae",
        "rmse",
        "bias",
        "median_absolute_error",
        "reference_mean",
        "reference_median",
        "mean_delta_from_reference",
        "median_delta_from_reference",
        "mean_delta_error",
    )
    variants = sorted(count_by_variant)
    for variant in variants:
        for split in ("validation", "test"):
            for analysis in (
                "known_lid",
                "e1_sample_size_stability",
                "e5_paired_delta",
            ):
                selected = [
                    row
                    for row in rows
                    if row["model_variant"] == variant
                    and row["split"] == split
                    and row["analysis"] == analysis
                ]
                metrics: dict[str, Any] = {}
                for field in metric_fields:
                    values: list[float] = []
                    for row in selected:
                        raw = row.get(field, "")
                        if raw in (None, ""):
                            continue
                        value = float(raw)
                        if math.isfinite(value):
                            values.append(value)
                    if values:
                        metrics[field] = {
                            "finite_cells": len(values),
                            "macro_mean": float(np.mean(values)),
                            "macro_median": float(np.median(values)),
                        }
                summaries.append(
                    {
                        "model_variant": variant,
                        "split": split,
                        "evaluation_role": (
                            "validation_selection_conditioned"
                            if split == "validation"
                            else "test_confirmatory"
                        ),
                        "analysis": analysis,
                        "rows": len(selected),
                        "metrics": metrics,
                    }
                )
    return {
        "schema_version": SCHEMA_VERSION,
        "followup_id": FOLLOWUP_ID,
        "scientific_scope": "post_hoc_readout_only_no_training_gate_relaxation",
        "rows": len(rows),
        "expected_combined_rows": EXPECTED_COMBINED_ROWS,
        "physical_training_runs": EXPECTED_TRAINING_TASKS,
        "reported_output_variants": len(OUTPUT_VARIANTS),
        "counts": {
            "by_analysis": count_by_analysis,
            "by_suite": count_by_suite,
            "by_model_variant": count_by_variant,
        },
        "selection_evidence": {
            "source_campaign_identity": evidence.source_campaign_identity,
            "failed_training_gate": nf._plain(evidence.failed_training_gate),
            "frozen_readout_gate": nf._plain(evidence.frozen_readout_gate),
        },
        "macro_summaries": summaries,
        "caveats": [
            "OLS5 was selected post-hoc using validation evidence only.",
            "Test rows are confirmatory and were not used to choose the readout.",
            "C1 failed its original training-quality gate and is excluded.",
            "Generated E3/E4 cells remain a secondary four-cell extension.",
            "OLS9 remains diagnostic; OLS5 is the only frozen improved readout.",
        ],
    }


def _readme_text(evidence: Stage2Evidence) -> str:
    training = evidence.failed_training_gate
    readout = evidence.frozen_readout_gate
    return f"""# NF C0/OLS5 readout follow-up

This folder is a sealed, post-hoc **readout-only** extension of the immutable
1,872-row global results table. `unified_results_with_nf_readout_followup.csv`
contains exactly 2,106 rows and preserves the original CSV as a byte-for-byte
prefix. `nf_readout_followup_results.csv` contains the 234 new rows only.

## What was frozen

Completed validation-only Stage 2 from campaign
`{evidence.source_campaign_identity}` froze OLS5 for the unchanged C0 model.
The C0/OLS5 versus C0/autograd canonical geometric loss ratio was
{readout.canonical_geometric_mean_ratio:.6f}, with {readout.canonical_wins}/15
wins and {readout.canonical_regressions_over_25pct} regressions above 25%.
No test values were opened before this decision.

The promoted C1 training change remains rejected: its OLS5 ratio against
C0/OLS5 was {training.canonical_geometric_mean_ratio:.6f}, with
{training.canonical_wins}/15 wins and
{training.canonical_regressions_over_25pct} regressions above 25%. This
follow-up does not relax, override, or relabel the failed training-quality gate,
and no C1 row is included.

## Matrix and interpretation

- 78 physical C0 trainings: all 39 cells for seeds 2 and 3.
- 3 reported variants: seed-2/autograd reference, seed-2/OLS5, seed-3/OLS5.
- 234 rows: 3 variants x 39 cells x validation/test.
- Validation metrics are selection-conditioned. Test metrics are confirmatory.
- Canonical and generated E3/E4 cells are retained separately in provenance;
  the four generated E3/E4 cells are secondary evidence.
- OLS9 remains a diagnostic readout from the ablation. It was not frozen for
  this extension; OLS5 is the only improved readout reported here.
- Every checkpoint was deleted after inline evaluation. Pointwise evidence is
  sealed in cell directories but is not required for the compact result export.

`aggregate.json` provides compact macro summaries and coverage counts.
`nf_readout_followup_results.provenance.json` binds every row to its physical
run ID, sealed cell, source-input evidence, and reviewed Stage-2 hashes.
"""


def _bundle_source_evidence(
    root: Path, evidence: Stage2Evidence
) -> dict[str, dict[str, Any]]:
    sources = {
        "identity_sha256": evidence.source_root / "ablation_identity.json",
        "stage1_ledger_sha256": evidence.source_root
        / "state"
        / "stage1"
        / "ledger.json",
        "stage1_scores_sha256": evidence.source_root
        / "state"
        / "stage1"
        / "validation_scores.json",
        "promotions_sha256": evidence.source_root
        / "state"
        / "stage1"
        / "promotions.json",
        "ledger_sha256": evidence.source_root / "state" / "stage2" / "ledger.json",
        "scores_sha256": evidence.source_root
        / "state"
        / "stage2"
        / "validation_scores.json",
    }
    expected = {
        "identity_sha256": evidence.identity_sha256,
        "stage1_ledger_sha256": evidence.stage1_ledger_sha256,
        "stage1_scores_sha256": evidence.stage1_scores_sha256,
        "promotions_sha256": evidence.promotions_sha256,
        "ledger_sha256": evidence.stage2_ledger_sha256,
        "scores_sha256": evidence.stage2_scores_sha256,
    }
    records: dict[str, dict[str, Any]] = {}
    for field, source in sources.items():
        destination = root / BUNDLED_EVIDENCE_PATHS[field]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        actual_sha = nf._sha256_path(destination)
        if actual_sha != expected[field]:
            destination.unlink(missing_ok=True)
            raise NFReadoutFollowupError(f"bundled reviewed evidence differs: {field}")
        records[field] = {
            "path": destination.relative_to(root).as_posix(),
            "sha256": actual_sha,
            "size": destination.stat().st_size,
        }
    return records


def finalize_followup_campaign(
    campaign_root: str | Path,
    preflight: nf.BaselinePreflight,
    evidence: Stage2Evidence,
    results: Sequence[Mapping[str, Any]],
) -> Path:
    root = Path(campaign_root).expanduser().resolve()
    identity_path = root / "followup_identity.json"
    identity = _read_json(identity_path, label="follow-up identity")
    campaign_identity = nf._canonical_sha(identity)
    rows, provenance_records = build_followup_unified_rows(results, preflight.cells)
    extension_path = root / "nf_readout_followup_results.csv"
    combined_path = root / "unified_results_with_nf_readout_followup.csv"
    provenance_path = root / "nf_readout_followup_results.provenance.json"
    aggregate_path = root / "aggregate.json"
    readme_path = root / "README.md"
    row_audit_path = root / "row_audit.json"
    baseline_csv = preflight.baseline_root / "unified_results.csv"
    baseline_bytes = baseline_csv.read_bytes()
    nf._write_csv(extension_path, rows)
    merge = nf.merge_unified_results(
        baseline_csv=baseline_csv,
        extension_rows=rows,
        output_csv=combined_path,
    )
    nf._write_json(
        provenance_path,
        {
            "schema_version": SCHEMA_VERSION,
            "campaign_identity": campaign_identity,
            "baseline_campaign_identity": preflight.baseline_campaign_identity,
            "baseline_unified_sha256": preflight.baseline_unified_sha256,
            "source_stage2_campaign_identity": evidence.source_campaign_identity,
            "scientific_scope": "post_hoc_readout_only_no_training_gate_relaxation",
            "failed_training_gate": nf._plain(evidence.failed_training_gate),
            "frozen_readout_gate": nf._plain(evidence.frozen_readout_gate),
            "run_id_contract": {
                "run_id": "full_sha256_of_cell_identity_v1",
                "cell_id": "first_20_hex_characters_of_run_id",
            },
            "output_variants": [asdict(value) for value in OUTPUT_VARIANTS],
            "records": provenance_records,
        },
    )
    nf._write_json(
        row_audit_path,
        _build_row_audit(
            extension_path=extension_path,
            provenance_path=provenance_path,
            provenance_records=provenance_records,
            campaign_identity=campaign_identity,
        ),
    )
    nf._write_json(aggregate_path, _aggregate_followup_rows(rows, evidence))
    campaign._write_text(readme_path, _readme_text(evidence))
    bundled_evidence = _bundle_source_evidence(root, evidence)
    ledger_path = root / "state" / "stage3_readout_followup" / "ledger.json"
    ledger = _read_json(ledger_path, label="follow-up ledger")
    if ledger.get("complete") is not True:
        raise NFReadoutFollowupError("cannot finalize an incomplete follow-up")
    manifest_path = root / "campaign.json"
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "followup_id": FOLLOWUP_ID,
        "campaign_identity": campaign_identity,
        "scientific_scope": "post_hoc_readout_only_no_training_gate_relaxation",
        "baseline": {
            "campaign_root": str(preflight.baseline_root),
            "campaign_identity": preflight.baseline_campaign_identity,
            "unified_results_path": str(
                preflight.baseline_root / "unified_results.csv"
            ),
            "sha256": preflight.baseline_unified_sha256,
            "byte_size": len(baseline_bytes),
            "rows": preflight.baseline_row_count,
            "preservation": "immutable_byte_prefix",
        },
        "source_stage2": {
            "campaign_root": str(evidence.source_root),
            "campaign_identity": evidence.source_campaign_identity,
            "identity_sha256": evidence.identity_sha256,
            "stage1_ledger_sha256": evidence.stage1_ledger_sha256,
            "stage1_scores_sha256": evidence.stage1_scores_sha256,
            "ledger_sha256": evidence.stage2_ledger_sha256,
            "scores_sha256": evidence.stage2_scores_sha256,
            "promotions_sha256": evidence.promotions_sha256,
            "completed_tasks": evidence.completed_tasks,
            "validation_records": evidence.validation_records,
            "training_gate_status": "failed_not_relaxed",
            "bundled_evidence": bundled_evidence,
        },
        "frozen_readout": FROZEN_READOUT,
        "failed_training_gate": nf._plain(evidence.failed_training_gate),
        "frozen_readout_gate": nf._plain(evidence.frozen_readout_gate),
        "extension": {
            "path": extension_path.relative_to(root).as_posix(),
            "sha256": nf._sha256_path(extension_path),
            "rows": len(rows),
            "output_variants": 3,
            "training_tasks": EXPECTED_TRAINING_TASKS,
            "cells_per_variant": EXPECTED_ALL_CELLS,
            "splits_per_cell": 2,
            "expected_combined_rows": EXPECTED_COMBINED_ROWS,
        },
        "combined": {
            "path": combined_path.relative_to(root).as_posix(),
            **merge,
        },
        "provenance": {
            "path": provenance_path.relative_to(root).as_posix(),
            "sha256": nf._sha256_path(provenance_path),
            "records": len(provenance_records),
        },
        "row_audit": {
            "path": row_audit_path.relative_to(root).as_posix(),
            "sha256": nf._sha256_path(row_audit_path),
            "records": len(provenance_records),
        },
        "aggregate": {
            "path": aggregate_path.relative_to(root).as_posix(),
            "sha256": nf._sha256_path(aggregate_path),
        },
        "protocol_readme": {
            "path": readme_path.relative_to(root).as_posix(),
            "sha256": nf._sha256_path(readme_path),
        },
        "stage": {
            "ledger_path": ledger_path.relative_to(root).as_posix(),
            "ledger_sha256": nf._sha256_path(ledger_path),
            "completed_tasks": len(ledger["completed_tasks"]),
        },
        "telemetry": {
            "nvidia_smi_jsonl": "telemetry/nvidia_smi.jsonl",
            "worker_occupancy_jsonl": "telemetry/worker_occupancy.jsonl",
        },
        "created_at_utc": datetime.now(UTC).isoformat(),
        "complete": True,
    }
    output_relatives = {
        "followup_identity.json",
        "baseline_provenance.json",
        "source_stage2_evidence.json",
        "reviewed_evidence.json",
        "resolved_followup.yaml",
        "preflight.json",
        "state/stage3_readout_followup/ledger.json",
        "telemetry/nvidia_smi.jsonl",
        "telemetry/worker_occupancy.jsonl",
        extension_path.relative_to(root).as_posix(),
        combined_path.relative_to(root).as_posix(),
        provenance_path.relative_to(root).as_posix(),
        row_audit_path.relative_to(root).as_posix(),
        aggregate_path.relative_to(root).as_posix(),
        readme_path.relative_to(root).as_posix(),
        *(record["path"] for record in bundled_evidence.values()),
    }
    missing = [
        value for value in sorted(output_relatives) if not (root / value).is_file()
    ]
    if missing:
        raise NFReadoutFollowupError(
            f"final follow-up artifacts are missing: {missing}"
        )
    manifest["outputs"] = [
        {
            "path": relative,
            "size": (root / relative).stat().st_size,
            "sha256": nf._sha256_path(root / relative),
        }
        for relative in sorted(output_relatives)
    ]
    nf._write_json(manifest_path, manifest)
    errors = validate_followup_campaign(root)
    if errors:
        manifest_path.unlink(missing_ok=True)
        raise NFReadoutFollowupError(
            f"new readout follow-up failed final validation: {errors}"
        )
    return root


def _matches_metric_fields(row: Mapping[str, str], expected: Mapping[str, Any]) -> bool:
    metric_fields = (
        "n",
        "finite_n",
        "finite_fraction",
        "mean",
        "std",
        "median",
        "q05",
        "q95",
        "target_finite_n",
        "mae",
        "rmse",
        "bias",
        "median_absolute_error",
        "reference_mean",
        "reference_median",
        "mean_delta_from_reference",
        "median_delta_from_reference",
        "mean_delta_error",
    )
    for field in metric_fields:
        expected_value = expected.get(field, "")
        actual = row.get(field, "")
        if expected_value in (None, ""):
            if actual != "":
                return False
            continue
        try:
            if not math.isclose(
                float(actual), float(expected_value), rel_tol=1.0e-12, abs_tol=1.0e-12
            ):
                return False
        except (TypeError, ValueError):
            return False
    return True


def _expected_extension_row_keys() -> set[tuple[str, ...]]:
    expected: set[tuple[str, ...]] = set()
    for variant in OUTPUT_VARIANTS:
        trained_variant = nf._model_variant(
            nf.candidate_by_id(variant.candidate_id), seed=variant.seed
        )
        model_variant = f"{trained_variant}-readout-{variant.readout}"
        for cell_key in campaign.APPROVED_GLOBAL_CELL_KEYS:
            suite_id, dataset, representation = cell_key.split("/")
            analysis = (
                "e1_sample_size_stability"
                if cell_key.startswith("e1/e1_sampled")
                else "e5_paired_delta"
                if cell_key.startswith("e5/")
                else "known_lid"
            )
            for split in ("validation", "test"):
                values = {
                    "model_variant": model_variant,
                    "analysis": analysis,
                    "suite_id": suite_id,
                    "dataset": dataset,
                    "representation": representation,
                    "split": split,
                    "readout": variant.readout,
                }
                expected.add(tuple(values[field] for field in ROW_KEY_FIELDS))
    if len(expected) != EXPECTED_EXTENSION_ROWS:
        raise NFReadoutFollowupError("expected extension row matrix differs")
    return expected


def _compact_row_contract_errors(
    extension_rows: Sequence[Mapping[str, str]],
) -> tuple[list[str], dict[str, int]]:
    errors: list[str] = []
    keys = {tuple(row[field] for field in ROW_KEY_FIELDS) for row in extension_rows}
    expected_keys = _expected_extension_row_keys()
    if len(keys) != len(extension_rows):
        errors.append("extension result keys are not unique")
    if keys != expected_keys:
        errors.append("extension does not cover the exact 3 x 39 x 2 matrix")
    if len({row["model_variant"] for row in extension_rows}) != 3:
        errors.append("extension does not expose three output variants")
    if sum(row["readout"] == REFERENCE_READOUT for row in extension_rows) != 78:
        errors.append("autograd reference coverage differs")
    if sum(row["readout"] == FROZEN_READOUT for row in extension_rows) != 156:
        errors.append("OLS5 coverage differs")
    analysis_counts = {
        analysis: sum(row["analysis"] == analysis for row in extension_rows)
        for analysis in (
            "known_lid",
            "e1_sample_size_stability",
            "e5_paired_delta",
        )
    }
    if analysis_counts != {
        "known_lid": 114,
        "e1_sample_size_stability": 78,
        "e5_paired_delta": 42,
    }:
        errors.append("analysis coverage differs from 114/78/42")
    summary_fields = ("n", "finite_n", "finite_fraction", "mean", "median")
    known_fields = (
        "target_finite_n",
        "mae",
        "rmse",
        "bias",
        "median_absolute_error",
    )
    e1_fields = (
        "reference_mean",
        "reference_median",
        "mean_delta_from_reference",
        "median_delta_from_reference",
        "mean_delta_error",
    )
    for row in extension_rows:
        if any(row[field] == "" for field in summary_fields):
            errors.append("extension row lacks prediction summary fields")
            break
        if row["analysis"] == "known_lid" and (
            any(row[field] == "" for field in known_fields)
            or any(row[field] != "" for field in e1_fields)
        ):
            errors.append("known-LID metric blank/populated contract differs")
            break
        if row["analysis"] == "e1_sample_size_stability" and (
            any(row[field] == "" for field in e1_fields)
            or any(row[field] != "" for field in known_fields)
        ):
            errors.append("E1 metric blank/populated contract differs")
            break
        if row["analysis"] == "e5_paired_delta" and (
            any(row[field] == "" for field in known_fields)
            or row["labels_sha256"] == ""
        ):
            errors.append("E5 metric blank/populated contract differs")
            break
    return errors, analysis_counts


def validate_followup_compact_bundle(campaign_root: str | Path) -> list[str]:
    """Validate copied compact outputs without remote paths or pointwise arrays."""

    root = Path(campaign_root).expanduser().resolve()
    errors: list[str] = []
    try:
        manifest = _read_json(root / "campaign.json", label="final manifest")
        identity = _read_json(root / "followup_identity.json", label="identity")
        if manifest.get("complete") is not True:
            errors.append("final manifest is not complete")
        if manifest.get("campaign_identity") != nf._canonical_sha(identity):
            errors.append("final manifest identity differs")
        outputs = manifest.get("outputs")
        if not isinstance(outputs, list):
            errors.append("final outputs are absent")
        else:
            for raw in outputs:
                path = root / str(raw["path"])
                if (
                    not path.is_file()
                    or path.stat().st_size != int(raw["size"])
                    or nf._sha256_path(path) != raw["sha256"]
                ):
                    errors.append(f"final output differs: {raw.get('path')}")

        combined_path = root / str(manifest["combined"]["path"])
        combined_bytes = combined_path.read_bytes()
        baseline_size = int(manifest["baseline"]["byte_size"])
        if (
            baseline_size <= 0
            or baseline_size >= len(combined_bytes)
            or hashlib.sha256(combined_bytes[:baseline_size]).hexdigest()
            != manifest["baseline"]["sha256"]
        ):
            errors.append("combined CSV baseline byte prefix differs")
        extension_header, extension_rows = nf._read_csv(
            root / str(manifest["extension"]["path"])
        )
        combined_header, combined_rows = nf._read_csv(combined_path)
        if tuple(extension_header) != tuple(campaign._UNIFIED_TABLE_FIELDS):
            errors.append("extension CSV schema differs")
        if tuple(combined_header) != tuple(campaign._UNIFIED_TABLE_FIELDS):
            errors.append("combined CSV schema differs")
        if len(extension_rows) != EXPECTED_EXTENSION_ROWS:
            errors.append("extension does not contain exact 3 x 39 x 2 rows")
        if (
            manifest["baseline"]["rows"] != EXPECTED_BASELINE_ROWS
            or len(combined_rows) != EXPECTED_COMBINED_ROWS
        ):
            errors.append("combined CSV row count differs")
        elif combined_rows[EXPECTED_BASELINE_ROWS:] != extension_rows:
            errors.append("combined CSV extension suffix differs")
        row_errors, analysis_counts = _compact_row_contract_errors(extension_rows)
        errors.extend(row_errors)

        provenance_path = root / str(manifest["provenance"]["path"])
        provenance = _read_json(provenance_path, label="provenance")
        provenance_records = provenance.get("records")
        audit_path = root / str(manifest["row_audit"]["path"])
        audit = _read_json(audit_path, label="row audit")
        audit_records = audit.get("records")
        if (
            provenance.get("campaign_identity") != manifest.get("campaign_identity")
            or audit.get("campaign_identity") != manifest.get("campaign_identity")
            or audit.get("extension_sha256")
            != nf._sha256_path(root / str(manifest["extension"]["path"]))
            or audit.get("provenance_sha256") != nf._sha256_path(provenance_path)
            or not isinstance(provenance_records, list)
            or not isinstance(audit_records, list)
            or len(provenance_records) != EXPECTED_EXTENSION_ROWS
            or len(audit_records) != EXPECTED_EXTENSION_ROWS
        ):
            errors.append("CSV/provenance/audit coverage differs")
        else:
            for ordinal, (row, provenance_row, audit_row) in enumerate(
                zip(extension_rows, provenance_records, audit_records, strict=True)
            ):
                if (
                    audit_row.get("ordinal") != ordinal
                    or audit_row.get("row_key") != _row_key(row)
                    or _row_key(provenance_row) != _row_key(row)
                    or audit_row.get("csv_row_sha256") != _csv_row_sha256(row)
                    or audit_row.get("provenance_record_sha256")
                    != nf._canonical_sha(provenance_row)
                    or audit_row.get("run_id") != provenance_row.get("run_id")
                    or audit_row.get("cell_id") != provenance_row.get("cell_id")
                    or audit_row.get("evidence") != provenance_row.get("row_evidence")
                ):
                    errors.append("CSV/provenance/audit 1:1 binding differs")
                    break
            run_ids = {str(row["run_id"]) for row in provenance_records}
            run_counts = {
                run_id: sum(row["run_id"] == run_id for row in provenance_records)
                for run_id in run_ids
            }
            if (
                len(run_ids) != EXPECTED_TRAINING_TASKS
                or sorted(run_counts.values()) != [2] * 39 + [4] * 39
                or {row.get("evaluation_role") for row in provenance_records}
                != {"validation_selection_conditioned", "test_confirmatory"}
            ):
                errors.append("physical run/provenance role coverage differs")

        source = manifest["source_stage2"]
        bundled = source.get("bundled_evidence")
        if not isinstance(bundled, Mapping) or set(bundled) != set(
            BUNDLED_EVIDENCE_PATHS
        ):
            errors.append("bundled reviewed evidence coverage differs")
        else:
            for field, raw in bundled.items():
                path = root / str(raw["path"])
                if (
                    raw.get("path") != BUNDLED_EVIDENCE_PATHS[field]
                    or not path.is_file()
                    or path.stat().st_size != int(raw["size"])
                    or nf._sha256_path(path) != raw["sha256"]
                    or raw["sha256"] != source[field]
                ):
                    errors.append(f"bundled reviewed evidence differs: {field}")
        aggregate = _read_json(
            root / str(manifest["aggregate"]["path"]), label="aggregate"
        )
        if (
            aggregate.get("rows") != EXPECTED_EXTENSION_ROWS
            or aggregate.get("expected_combined_rows") != EXPECTED_COMBINED_ROWS
            or aggregate.get("physical_training_runs") != EXPECTED_TRAINING_TASKS
            or aggregate.get("counts", {}).get("by_analysis") != analysis_counts
        ):
            errors.append("aggregate coverage differs")
        readme = (root / str(manifest["protocol_readme"]["path"])).read_text(
            encoding="utf-8"
        )
        for statement in (
            "selection-conditioned",
            "confirmatory",
            "does not relax",
            "OLS9 remains a diagnostic",
            "2,106 rows",
        ):
            if statement not in readme:
                errors.append(f"protocol README lacks caveat: {statement}")
        ledger_path = root / str(manifest["stage"]["ledger_path"])
        ledger = _read_json(ledger_path, label="follow-up ledger")
        if (
            ledger.get("complete") is not True
            or nf._sha256_path(ledger_path) != manifest["stage"]["ledger_sha256"]
            or len(ledger.get("completed_tasks", ())) != EXPECTED_TRAINING_TASKS
        ):
            errors.append("follow-up ledger is incomplete or changed")
    except (
        KeyError,
        OSError,
        UnicodeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        NFReadoutFollowupError,
    ) as exc:
        errors.append(f"compact follow-up validation failed: {exc}")
    if list(root.rglob("checkpoint.pt")) or list(root.rglob("training_progress.pt")):
        errors.append("completed readout follow-up retained a checkpoint")
    return errors


def _remote_ledger_join_errors(root: Path, manifest: Mapping[str, Any]) -> list[str]:
    """Bind every reported row to one exact sealed physical training task."""

    errors: list[str] = []
    _, rows = nf._read_csv(root / str(manifest["extension"]["path"]))
    provenance = _read_json(
        root / str(manifest["provenance"]["path"]), label="provenance"
    )
    provenance_records = provenance.get("records")
    ledger = _read_json(
        root / str(manifest["stage"]["ledger_path"]), label="follow-up ledger"
    )
    completed = ledger.get("completed_tasks")
    if (
        not isinstance(provenance_records, list)
        or not isinstance(completed, list)
        or len(rows) != len(provenance_records)
    ):
        return ["row/provenance/ledger join inputs are incomplete"]

    expected_task_keys = {
        (CONTROL_CANDIDATE, seed, cell_key)
        for seed in (2, 3)
        for cell_key in campaign.APPROVED_GLOBAL_CELL_KEYS
    }
    ledger_by_key: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for raw in completed:
        if not isinstance(raw, Mapping):
            errors.append("follow-up ledger row is malformed")
            continue
        try:
            key = (
                str(raw["candidate_id"]),
                int(raw["seed"]),
                str(raw["cell_key"]),
            )
        except (KeyError, TypeError, ValueError):
            errors.append("follow-up ledger task key is malformed")
            continue
        if key in ledger_by_key:
            errors.append("follow-up ledger repeats a physical task")
            continue
        ledger_by_key[key] = raw
    if set(ledger_by_key) != expected_task_keys:
        errors.append("follow-up ledger does not cover the exact C0 seeds 2/3 matrix")

    variant_by_signature = {
        (value.candidate_id, value.seed, value.readout): value
        for value in OUTPUT_VARIANTS
    }
    provenance_count_by_task: dict[tuple[str, int, str], int] = {}
    for row, provenance_row in zip(rows, provenance_records, strict=True):
        if not isinstance(provenance_row, Mapping):
            errors.append("follow-up provenance row is malformed")
            continue
        try:
            cell_key = "/".join(
                (
                    str(provenance_row["suite_id"]),
                    str(provenance_row["dataset"]),
                    str(provenance_row["representation"]),
                )
            )
            task_key = (
                str(provenance_row["candidate_id"]),
                int(provenance_row["seed"]),
                cell_key,
            )
            readout = str(provenance_row["readout"])
            variant = variant_by_signature[(task_key[0], task_key[1], readout)]
            ledger_row = ledger_by_key[task_key]
            directory = (root / str(ledger_row["path"])).resolve()
            directory.relative_to(root.resolve())
            provenance_directory = Path(str(provenance_row["cell_path"])).resolve()
        except (KeyError, TypeError, ValueError):
            errors.append("provenance row does not identify a frozen output variant")
            continue
        provenance_count_by_task[task_key] = (
            provenance_count_by_task.get(task_key, 0) + 1
        )
        if (
            provenance_row.get("output_variant_id") != variant.variant_id
            or provenance_row.get("stage") != "readout_followup"
            or provenance_row.get("evaluation_role")
            != (
                "validation_selection_conditioned"
                if row["split"] == "validation"
                else "test_confirmatory"
            )
            or provenance_directory != directory
            or provenance_row.get("manifest_sha256")
            != ledger_row.get("manifest_sha256")
            or provenance_row.get("summary_sha256") != ledger_row.get("summary_sha256")
        ):
            errors.append(f"provenance/ledger binding differs: {_row_key(row)}")
            continue
        try:
            identity = _read_json(directory / "identity.json", label="cell identity")
            summary = _read_json(directory / "summary.json", label="cell summary")
            cell_identity = identity["cell"]
            selected = summary["readouts"][readout]
            trained_variant = str(summary["model_variant"])
            expected_run_id = nf._canonical_sha(identity)
            selected_index = selected["selected_index"]
            selected_scale = selected["selected_scale"]
        except (KeyError, TypeError, ValueError, NFReadoutFollowupError):
            errors.append(f"sealed row binding is malformed: {_row_key(row)}")
            continue
        if not isinstance(cell_identity, Mapping):
            errors.append(f"sealed cell identity is malformed: {_row_key(row)}")
            continue
        try:
            selected_index_matches = (
                row["selected_index"] == ""
                if selected_index is None
                else int(row["selected_index"]) == int(selected_index)
            )
            selected_scale_matches = (
                row["selected_coordinate"] == ""
                if selected_scale is None
                else math.isclose(
                    float(row["selected_coordinate"]),
                    float(selected_scale),
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                )
            )
        except (TypeError, ValueError):
            selected_index_matches = False
            selected_scale_matches = False
        if (
            str(cell_identity.get("suite_id")) != row["suite_id"]
            or str(cell_identity.get("dataset")) != row["dataset"]
            or str(cell_identity.get("representation")) != row["representation"]
            or identity.get("candidate_id") != task_key[0]
            or identity.get("training_seed") != task_key[1]
            or ledger_row.get("task_id") != expected_run_id
            or summary.get("run_id") != expected_run_id
            or provenance_row.get("run_id") != expected_run_id
            or provenance_row.get("cell_id") != expected_run_id[:20]
            or summary.get("cell_id") != expected_run_id[:20]
            or provenance_row.get("run_id_contract")
            != "full_sha256_of_cell_identity_v1"
            or provenance_row.get("trained_model_variant") != trained_variant
            or row["model_variant"] != f"{trained_variant}-readout-{readout}"
            or row["primary_readout"] != readout
            or row["is_primary_readout"] != "True"
            or row["selected_coordinate_name"] != "scale"
            or not selected_index_matches
            or not selected_scale_matches
        ):
            errors.append(f"row/sealed-summary binding differs: {_row_key(row)}")

    expected_counts = {key: (4 if key[1] == 2 else 2) for key in expected_task_keys}
    if provenance_count_by_task != expected_counts:
        errors.append("reported rows do not map 4/2 to every seed-2/seed-3 task")
    return errors


def _remote_row_evidence_errors(root: Path, manifest: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    _, rows = nf._read_csv(root / str(manifest["extension"]["path"]))
    provenance = _read_json(
        root / str(manifest["provenance"]["path"]), label="provenance"
    )
    for row, provenance_row in zip(rows, provenance["records"], strict=True):
        evidence = provenance_row["row_evidence"]
        for prefix in (
            "prediction",
            "target",
            "reference_prediction",
            "labels",
            "reference_labels",
        ):
            path_value = evidence[f"{prefix}_path"]
            sha_value = evidence[f"{prefix}_sha256"]
            if path_value is None:
                if sha_value is not None:
                    errors.append("row evidence path/hash nullability differs")
                continue
            path = Path(str(path_value))
            if not path.is_file() or nf._sha256_path(path) != sha_value:
                errors.append(f"pointwise row evidence differs: {_row_key(row)}")
                break
        else:
            prediction = np.asarray(
                np.load(evidence["prediction_path"], allow_pickle=False),
                dtype=np.float64,
            )
            expected: dict[str, Any] = {}
            analysis = row["analysis"]
            if analysis == "known_lid":
                target = np.load(evidence["target_path"], allow_pickle=False)
                expected.update(known_lid_metrics(prediction, target))
            else:
                reference = np.asarray(
                    np.load(evidence["reference_prediction_path"], allow_pickle=False),
                    dtype=np.float64,
                )
                if analysis == "e1_sample_size_stability":
                    current_metrics = prediction_summary(prediction)
                    reference_metrics = prediction_summary(reference)
                    expected.update(current_metrics)
                    expected["reference_mean"] = reference_metrics["mean"]
                    expected["reference_median"] = reference_metrics["median"]
                    expected["mean_delta_from_reference"] = float(
                        current_metrics["mean"] - reference_metrics["mean"]
                    )
                    expected["median_delta_from_reference"] = float(
                        current_metrics["median"] - reference_metrics["median"]
                    )
                    expected["mean_delta_error"] = float(
                        expected["mean_delta_from_reference"]
                        - float(provenance_row["expected_lid_delta"])
                    )
                elif analysis == "e5_paired_delta":
                    expected.update(
                        paired_delta_metrics(
                            reference,
                            prediction,
                            expected_delta=float(provenance_row["expected_lid_delta"]),
                        )
                    )
                    labels = np.load(evidence["labels_path"], allow_pickle=False)
                    reference_labels = np.load(
                        evidence["reference_labels_path"], allow_pickle=False
                    )
                    if not np.array_equal(labels, reference_labels) or row[
                        "labels_sha256"
                    ] != campaign._array_sha(labels):
                        errors.append(f"paired row labels differ: {_row_key(row)}")
                        continue
                else:
                    errors.append(f"unknown analysis in pointwise audit: {analysis}")
                    continue
            if not _matches_metric_fields(row, expected):
                errors.append(f"recomputed row metrics differ: {_row_key(row)}")
    return errors


def validate_followup_campaign(campaign_root: str | Path) -> list[str]:
    """Layer remote baseline, source, cell and pointwise checks on compact QA."""

    root = Path(campaign_root).expanduser().resolve()
    errors = validate_followup_compact_bundle(root)
    try:
        manifest = _read_json(root / "campaign.json", label="final manifest")
        baseline_path = Path(str(manifest["baseline"]["unified_results_path"]))
        baseline_bytes = baseline_path.read_bytes()
        if (
            len(baseline_bytes) != int(manifest["baseline"]["byte_size"])
            or hashlib.sha256(baseline_bytes).hexdigest()
            != manifest["baseline"]["sha256"]
        ):
            errors.append("remote baseline unified CSV changed")
        ledger = _read_json(
            root / str(manifest["stage"]["ledger_path"]), label="follow-up ledger"
        )
        for raw in ledger.get("completed_tasks", ()):
            directory = root / str(raw["path"])
            if (
                not directory.is_dir()
                or nf._sha256_path(directory / "manifest.json")
                != raw["manifest_sha256"]
                or nf._sha256_path(directory / "summary.json") != raw["summary_sha256"]
            ):
                errors.append(f"sealed follow-up cell differs: {raw.get('cell_key')}")
        errors.extend(_remote_ledger_join_errors(root, manifest))
        source = manifest["source_stage2"]
        source_root = Path(str(source["campaign_root"]))
        source_paths = {
            "identity_sha256": source_root / "ablation_identity.json",
            "stage1_ledger_sha256": source_root / "state" / "stage1" / "ledger.json",
            "stage1_scores_sha256": source_root
            / "state"
            / "stage1"
            / "validation_scores.json",
            "ledger_sha256": source_root / "state" / "stage2" / "ledger.json",
            "scores_sha256": source_root
            / "state"
            / "stage2"
            / "validation_scores.json",
            "promotions_sha256": source_root / "state" / "stage1" / "promotions.json",
        }
        for field, path in source_paths.items():
            if not path.is_file() or nf._sha256_path(path) != source[field]:
                errors.append(f"remote source evidence differs: {field}")
        errors.extend(_remote_row_evidence_errors(root, manifest))
    except (
        KeyError,
        OSError,
        UnicodeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        NFReadoutFollowupError,
    ) as exc:
        errors.append(f"remote follow-up validation failed: {exc}")
    return errors


def run_followup_campaign(
    *,
    baseline_root: str | Path,
    source_ablation_root: str | Path,
    reviewed_evidence_manifest: str | Path,
    output_root: str | Path,
    project_root: str | Path | None = None,
    worker_count: int = nf.WORKER_COUNT,
    dependencies: nf.NFDependencies | None = None,
    require_cuda: bool = True,
    preflight_only: bool = False,
) -> Path:
    """Run the evidence-bound readout-only follow-up in one persistent pool."""

    if type(require_cuda) is not bool or type(preflight_only) is not bool:
        raise NFReadoutFollowupError("require_cuda and preflight_only must be booleans")
    if require_cuda and worker_count != nf.WORKER_COUNT:
        raise NFReadoutFollowupError(
            f"production follow-up requires exactly {nf.WORKER_COUNT} GPU workers"
        )
    nf._configure_deterministic_cublas(require_cuda=require_cuda)
    checkout = (
        campaign.repository_root()
        if project_root is None
        else Path(project_root).expanduser().resolve()
    )
    if not checkout.is_dir():
        raise NFReadoutFollowupError(f"project root is not a directory: {checkout}")
    dependencies = nf.NFDependencies() if dependencies is None else dependencies
    preflight = nf.preflight_baseline(
        baseline_root, project_root=checkout, dependencies=dependencies
    )
    if preflight.baseline_row_count != EXPECTED_BASELINE_ROWS:
        raise NFReadoutFollowupError(
            f"baseline must contain exactly {EXPECTED_BASELINE_ROWS} rows"
        )
    reviewed = load_reviewed_evidence(reviewed_evidence_manifest)
    evidence = load_stage2_evidence(
        source_ablation_root,
        reviewed=reviewed,
        expected_baseline_campaign_identity=preflight.baseline_campaign_identity,
        expected_baseline_unified_sha256=preflight.baseline_unified_sha256,
    )
    prepared = _prepare(
        preflight,
        evidence,
        project_root=checkout,
        output_root=Path(output_root),
        worker_count=worker_count,
    )
    root = Path(prepared.campaign_root)
    root.mkdir(parents=True, exist_ok=True)
    final_path = root / "campaign.json"
    if final_path.is_file():
        errors = validate_followup_campaign(root)
        if errors:
            raise NFReadoutFollowupError(
                f"existing final follow-up is invalid: {errors}"
            )
        return root
    with campaign._exclusive_campaign_lock(root / "state" / "campaign.lock"):
        if final_path.is_file():
            errors = validate_followup_campaign(root)
            if errors:
                raise NFReadoutFollowupError(
                    f"existing final follow-up is invalid: {errors}"
                )
            return root
        identity_path = root / "followup_identity.json"
        if identity_path.exists():
            existing = _read_json(identity_path, label="existing follow-up identity")
            if campaign.canonical_json(existing) != campaign.canonical_json(
                prepared.identity_record
            ):
                raise NFReadoutFollowupError(
                    "stable follow-up root has a different identity"
                )
        else:
            nf._write_json(identity_path, prepared.identity_record)
        nf._write_json(
            root / "baseline_provenance.json",
            {
                "schema_version": SCHEMA_VERSION,
                "baseline_root": str(preflight.baseline_root),
                "baseline_campaign_identity": preflight.baseline_campaign_identity,
                "baseline_unified_sha256": preflight.baseline_unified_sha256,
                "baseline_rows": preflight.baseline_row_count,
                "input_sha256_by_key": dict(preflight.input_sha256_by_key),
                "source_records": dict(preflight.source_records),
            },
        )
        nf._write_json(
            root / "source_stage2_evidence.json",
            {
                "schema_version": SCHEMA_VERSION,
                **nf._plain(evidence),
                "selection_statement": (
                    "OLS5 is frozen for C0 only; the failed C1 training gate remains "
                    "failed and is not overridden."
                ),
            },
        )
        nf._write_json(
            root / "reviewed_evidence.json",
            {"schema_version": SCHEMA_VERSION, **nf._plain(reviewed)},
        )
        campaign._write_yaml(
            root / "resolved_followup.yaml",
            {
                "schema_version": SCHEMA_VERSION,
                "followup_id": FOLLOWUP_ID,
                "worker_count": worker_count,
                "candidate": nf._plain(nf.candidate_by_id(CONTROL_CANDIDATE)),
                "frozen_readout": FROZEN_READOUT,
                "reference_readout": REFERENCE_READOUT,
                "output_variants": [asdict(value) for value in OUTPUT_VARIANTS],
                "training_tasks": EXPECTED_TRAINING_TASKS,
                "extension_rows": EXPECTED_EXTENSION_ROWS,
            },
        )
        monitor = nf._NvidiaSmiMonitor(
            root / "telemetry" / "nvidia_smi.jsonl",
            enabled=require_cuda,
            expected_gpu_count=worker_count,
        )
        first_gpu_sample = monitor.start()
        pool = nf._PersistentNFPool(
            worker_count=worker_count,
            dependencies=dependencies,
            require_cuda=require_cuda,
        )
        results: list[dict[str, Any]] | None = None
        try:
            workers = pool.start()
            nf._write_json(
                root / "preflight.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "campaign_identity": prepared.campaign_identity,
                    "baseline_campaign_identity": preflight.baseline_campaign_identity,
                    "source_stage2_campaign_identity": evidence.source_campaign_identity,
                    "failed_training_gate_preserved": True,
                    "frozen_readout": FROZEN_READOUT,
                    "worker_count": worker_count,
                    "require_cuda": require_cuda,
                    "workers": list(workers),
                    "first_gpu_sample": first_gpu_sample,
                    "planned_training_tasks": EXPECTED_TRAINING_TASKS,
                    "planned_output_variants": len(OUTPUT_VARIANTS),
                    "planned_extension_rows": EXPECTED_EXTENSION_ROWS,
                    "status": "ready",
                },
            )
            if not preflight_only:
                results = nf._run_task_stage(
                    prepared,
                    stage="stage3_readout_followup",
                    tasks=_followup_tasks(prepared),
                    pool=pool,
                    monitor=monitor,
                )
            pool.close()
        except BaseException:
            pool.abort()
            raise
        finally:
            monitor.stop()
        if preflight_only:
            return root
        if results is None:
            raise NFReadoutFollowupError("follow-up coordinator lacks final results")
        return finalize_followup_campaign(root, preflight, evidence, results)


def _parse_cli(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the evidence-bound C0/OLS5 NF readout follow-up."
    )
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--source-ablation-root", type=Path, required=True)
    parser.add_argument(
        "--reviewed-evidence-manifest",
        type=Path,
        required=True,
        help="manifest with the exact reviewed source identity and evidence SHAs",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--worker-count", type=int, default=nf.WORKER_COUNT)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="disable CUDA checks for local integration tests only",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _parse_cli(argv)
    result = run_followup_campaign(
        baseline_root=arguments.baseline_root,
        source_ablation_root=arguments.source_ablation_root,
        reviewed_evidence_manifest=arguments.reviewed_evidence_manifest,
        output_root=arguments.output_root,
        project_root=arguments.project_root,
        worker_count=arguments.worker_count,
        require_cuda=not arguments.allow_cpu,
        preflight_only=arguments.preflight_only,
    )
    print(result)


if __name__ == "__main__":
    main()
