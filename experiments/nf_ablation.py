"""Staged, resume-safe normalizing-flow ablation campaign.

This module deliberately lives beside, rather than inside, the sealed global
campaign.  The original 10 x 39 campaign and its unified CSV are immutable
inputs.  Only validation data are consulted in stages 1 and 2; test data are
first evaluated after one global NF configuration/readout has been frozen.

Production execution uses one spawn-based pool with one independent cell per
GPU.  A cell trains, evaluates, seals all pointwise evidence, and removes its
checkpoint before the worker can accept another task.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import io
import json
import math
import multiprocessing as mp
import os
import pickle
import queue
import shutil
import subprocess
import threading
import time
import traceback
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml

from experiments import global_campaign as campaign
from experiments.metrics import (
    known_lid_metrics,
    paired_delta_metrics,
    prediction_summary,
)
from models.oracle import select_stable_scale

NF_ABLATION_SCHEMA_VERSION = 1
NF_ABLATION_ID = "nf-quality-readout-optimizer-architecture-v1"
WORKER_COUNT = 8
PARTITION_SEED = 0
DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
READOUTS = ("autograd", "symmetric_fd", "ols3", "ols5", "ols9")
PAPER_PARITY_READOUT = "global_ols9"
FINITE_DIFFERENCE_LOG_STEP = 0.01
OLS_LOG_STEP = 0.05
SELECTION_SCALES = (
    0.01,
    0.0178,
    0.0316,
    0.0562,
    0.1,
    0.1778,
    0.3162,
    0.5623,
    1.0,
)
CONDITIONAL_SELECTION_SCALES = SELECTION_SCALES[1:-1]

# Eight cells, including both dependency references, exercise high-dimensional
# data, coefficient representations, sample-size stability, and paired deltas.
STAGE1_SENTINEL_KEYS = (
    "e4/e4_sphere_pca_radius1/dataset",
    "e4/e4_sphere_pca_radius1/coefficients",
    "e7/e7_crescent_moon_radius3.0/dataset",
    "e7/e7_crescent_moon_radius3.0/coefficients",
    "e1/e1_sampled_fmnist_step1/dataset",
    "e1/e1_sampled_fmnist_step7/dataset",
    "e5/e5_downscaled_fmnist/dataset",
    "e5/e5_padded_fmnist_adddim8/dataset",
)


class NFAblationError(RuntimeError):
    """Raised before an incomplete or scientifically unsafe run can continue."""


@dataclass(frozen=True)
class NFCandidate:
    candidate_id: str
    contract: str
    training_overrides: tuple[tuple[str, Any], ...]
    selection_scales: tuple[float, ...] = SELECTION_SCALES
    independent_fixed_epsilon: bool = False

    def training_config(
        self, base: Mapping[str, Any], *, seed: int, device: str = "cuda"
    ) -> dict[str, Any]:
        value = dict(base)
        value.update(dict(self.training_overrides))
        value["seed"] = int(seed)
        value["device"] = device
        # Materializing through the checkpoint schema rejects stale/unknown
        # fields and makes the immutable candidate identity explicit.
        return campaign._canonical_training_config_record(
            value, field=f"NF ablation {self.candidate_id} training config"
        )


def _overrides(**values: Any) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted(values.items()))


# C0--C5 are the approved conditional-NF matrix.  Do not generate this matrix
# procedurally: every scientific difference must remain reviewable in source.
CANDIDATES: tuple[NFCandidate, ...] = (
    NFCandidate(
        "C0",
        "conditional_realnvp_current_h100_control_v1",
        _overrides(batch_size=4096, epochs=200),
    ),
    NFCandidate(
        "C1",
        "conditional_realnvp_batch1024_v1",
        _overrides(batch_size=1024, epochs=200),
    ),
    NFCandidate(
        "C2",
        "conditional_realnvp_matched_update_budget_v1",
        _overrides(batch_size=1024, epochs=400, early_stopping_patience=40),
    ),
    NFCandidate(
        "C3",
        "conditional_realnvp_capacity_v1",
        _overrides(
            batch_size=1024,
            epochs=250,
            early_stopping_patience=25,
            hidden_dim=768,
            num_coupling_layers=12,
            conditioner_depth=3,
        ),
    ),
    NFCandidate(
        "C4",
        "conditional_realnvp_smooth_conditioning_v1",
        _overrides(
            batch_size=1024,
            epochs=400,
            early_stopping_patience=40,
            time_embedding_dim=256,
            fourier_features=64,
            max_condition_frequency=32.0,
            epsilon_min=0.005,
            epsilon_max=1.25,
        ),
    ),
    NFCandidate(
        "C5",
        "conditional_realnvp_combined_capacity_conditioning_e250_v1",
        _overrides(
            batch_size=1024,
            epochs=250,
            early_stopping_patience=25,
            hidden_dim=768,
            num_coupling_layers=12,
            conditioner_depth=3,
            time_embedding_dim=256,
            fourier_features=64,
            max_condition_frequency=32.0,
            epsilon_min=0.005,
            epsilon_max=1.25,
        ),
    ),
)

# P0 isolates the most consequential departure from the LIDL paper: it fits
# one density model per declared epsilon and regresses likelihood globally.
PAPER_PARITY_CANDIDATE = NFCandidate(
    "P0",
    "independent_fixed_epsilon_realnvp_global_ols9_v1",
    _overrides(batch_size=1024, epochs=400, early_stopping_patience=40),
    independent_fixed_epsilon=True,
)
STAGE1_CANDIDATES = (*CANDIDATES, PAPER_PARITY_CANDIDATE)


@dataclass(frozen=True)
class Promotion:
    candidate_id: str
    readout: str
    median_log_ratio: float
    win_rate: float
    stratum_median_ratios: Mapping[str, float]
    dataset_to_coefficients_ratio: float | None


@dataclass(frozen=True)
class Winner:
    candidate_id: str
    readout: str
    validation_median_mae: float
    validation_mean_mae: float
    canonical_geometric_mean_ratio: float
    canonical_wins: int
    canonical_regressions_over_25pct: int
    generated_geometric_mean_ratio: float | None


@dataclass(frozen=True)
class NFDependencies:
    """Spawn-picklable overrides for integration tests and production adapters."""

    inventory_loader: Callable[..., Any] | None = None
    source_preflight_fn: Callable[..., Any] | None = None
    cell_loader: Callable[..., Any] | None = None
    train_fn: Callable[..., Any] | None = None
    load_checkpoint_fn: Callable[..., Any] | None = None
    predict_readouts_fn: Callable[..., Any] | None = None
    predict_log_likelihood_fn: Callable[..., Any] | None = None


@dataclass(frozen=True)
class BaselinePreflight:
    baseline_root: Path
    baseline_campaign_identity: str
    baseline_unified_sha256: str
    baseline_row_count: int
    config: Mapping[str, Any]
    cells: tuple[Any, ...]
    input_sha256_by_key: Mapping[str, str]
    source_records: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class NFCellTask:
    project_root: str
    campaign_root: str
    baseline_unified_sha256: str
    config: Mapping[str, Any]
    cell: Any
    candidate_id: str
    seed: int
    evaluate_test: bool
    expected_input_sha256: str
    source_record: Mapping[str, Any]
    test_readout: str | None = None


@dataclass(frozen=True)
class _NFReadoutEvaluation:
    """Validated readouts together with the exact likelihood-path evidence."""

    epsilon: float
    finite_difference_log_step: float
    ols_log_step: float
    finite_difference_epsilons: npt.NDArray[np.float64]
    finite_difference_log_likelihood: npt.NDArray[np.float64]
    ols_epsilons: npt.NDArray[np.float64]
    ols_log_likelihood: npt.NDArray[np.float64]
    lid_by_readout: Mapping[str, npt.NDArray[np.float64]]


def candidate_by_id(candidate_id: str) -> NFCandidate:
    matches = [
        value for value in STAGE1_CANDIDATES if value.candidate_id == candidate_id
    ]
    if len(matches) != 1:
        raise NFAblationError(f"unknown or repeated NF candidate {candidate_id!r}")
    return matches[0]


def validate_candidate_matrix() -> None:
    if tuple(value.candidate_id for value in CANDIDATES) != (
        "C0",
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
    ):
        raise NFAblationError("conditional NF candidate matrix differs from contract")
    if len(set(CANDIDATES)) != len(CANDIDATES):
        raise NFAblationError("conditional NF candidate matrix contains duplicates")
    if len(STAGE1_SENTINEL_KEYS) != 8 or len(set(STAGE1_SENTINEL_KEYS)) != 8:
        raise NFAblationError("stage-1 sentinel inventory must contain exactly 8 cells")
    if PAPER_PARITY_CANDIDATE.candidate_id in {
        value.candidate_id for value in CANDIDATES
    }:
        raise NFAblationError("paper-parity candidate collides with C0--C5")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return campaign.sha256_bytes(campaign.canonical_json(value).encode("utf-8"))


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise NFAblationError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def _resolve_dependencies(dependencies: NFDependencies) -> NFDependencies:
    if (
        dependencies.train_fn is None
        or dependencies.load_checkpoint_fn is None
        or dependencies.predict_readouts_fn is None
        or dependencies.predict_log_likelihood_fn is None
    ):
        from models.training import (
            load_checkpoint,
            predict_nf_log_likelihood,
            predict_nf_readouts,
            train_model,
        )

        train_fn = (
            train_model if dependencies.train_fn is None else dependencies.train_fn
        )
        load_fn = (
            load_checkpoint
            if dependencies.load_checkpoint_fn is None
            else dependencies.load_checkpoint_fn
        )
        predict_fn = (
            predict_nf_readouts
            if dependencies.predict_readouts_fn is None
            else dependencies.predict_readouts_fn
        )
        predict_log_fn = (
            predict_nf_log_likelihood
            if dependencies.predict_log_likelihood_fn is None
            else dependencies.predict_log_likelihood_fn
        )
    else:
        train_fn = dependencies.train_fn
        load_fn = dependencies.load_checkpoint_fn
        predict_fn = dependencies.predict_readouts_fn
        predict_log_fn = dependencies.predict_log_likelihood_fn
    return NFDependencies(
        inventory_loader=(
            campaign.load_campaign_inventory
            if dependencies.inventory_loader is None
            else dependencies.inventory_loader
        ),
        source_preflight_fn=(
            campaign.validate_campaign_sources
            if dependencies.source_preflight_fn is None
            else dependencies.source_preflight_fn
        ),
        cell_loader=(
            campaign.load_campaign_cell_data
            if dependencies.cell_loader is None
            else dependencies.cell_loader
        ),
        train_fn=train_fn,
        load_checkpoint_fn=load_fn,
        predict_readouts_fn=predict_fn,
        predict_log_likelihood_fn=predict_log_fn,
    )


def preflight_baseline(
    baseline_root: str | Path,
    *,
    project_root: str | Path,
    dependencies: NFDependencies | None = None,
) -> BaselinePreflight:
    """Verify the sealed 390-cell baseline and freshly loaded source inputs."""

    validate_candidate_matrix()
    dependencies = NFDependencies() if dependencies is None else dependencies
    root = Path(baseline_root).expanduser().resolve()
    checkout = Path(project_root).expanduser().resolve()
    manifest_path = root / "campaign.json"
    unified_path = root / "unified_results.csv"
    config_path = root / "resolved_config.yaml"
    inventory_path = root / "input_inventory.json"
    for path in (manifest_path, unified_path, config_path, inventory_path):
        if not path.is_file():
            raise NFAblationError(f"baseline artifact is missing: {path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        sealed_inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise NFAblationError("baseline metadata are unreadable") from exc
    if (
        manifest.get("complete") is not True
        or manifest.get("expected_models") != 10
        or manifest.get("expected_cells_per_model") != 39
        or len(manifest.get("cells", ())) != 390
    ):
        raise NFAblationError("baseline manifest is not the complete 10 x 39 campaign")
    unified_sha = _sha256_path(unified_path)
    if manifest.get("unified_results_sha256") != unified_sha:
        raise NFAblationError("baseline unified CSV differs from its sealed SHA")
    header, baseline_rows = _read_csv(unified_path)
    if tuple(header) != tuple(campaign._UNIFIED_TABLE_FIELDS):
        raise NFAblationError(
            "baseline unified CSV schema differs from global campaign"
        )
    variants = {row["model_variant"] for row in baseline_rows}
    if variants != set(campaign.APPROVED_MODEL_VARIANTS):
        raise NFAblationError("baseline unified CSV does not cover all ten models")
    config = campaign.validate_global_campaign_config(raw_config)
    deps = _resolve_dependencies(dependencies)
    assert deps.inventory_loader is not None
    assert deps.source_preflight_fn is not None
    assert deps.cell_loader is not None
    cells = tuple(deps.inventory_loader(config, checkout))
    if tuple(cell.key for cell in cells) != campaign.APPROVED_GLOBAL_CELL_KEYS:
        raise NFAblationError(
            "fresh source inventory differs from the approved 39 cells"
        )
    source_records = dict(deps.source_preflight_fn(config, checkout, cells))
    if set(source_records) != {cell.inventory_id for cell in cells}:
        raise NFAblationError("source preflight does not cover the exact inventories")
    if not isinstance(sealed_inventory, list):
        raise NFAblationError("baseline input inventory is not a list")
    sealed_by_key: dict[str, str] = {}
    for row in sealed_inventory:
        try:
            key = "/".join(
                (
                    str(row["cell"]["suite_id"]),
                    str(row["cell"]["dataset"]),
                    str(row["cell"]["representation"]),
                )
            )
            digest = str(row["input_sha256"])
        except (KeyError, TypeError) as exc:
            raise NFAblationError("baseline input inventory row is malformed") from exc
        if key in sealed_by_key:
            raise NFAblationError(f"baseline input inventory repeats {key}")
        sealed_by_key[key] = digest
    if set(sealed_by_key) != set(campaign.APPROVED_GLOBAL_CELL_KEYS):
        raise NFAblationError("baseline input inventory does not cover exact 39 cells")
    fresh_input_sha: dict[str, str] = {}
    for cell in cells:
        data = campaign._bind_source_preflight(
            deps.cell_loader(cell, config, checkout),
            cell,
            source_records,
        )
        if data.input_sha256 != sealed_by_key[cell.key]:
            raise NFAblationError(f"source input differs from baseline: {cell.key}")
        fresh_input_sha[cell.key] = str(data.input_sha256)
    if _sha256_path(unified_path) != unified_sha:
        raise NFAblationError("baseline unified CSV changed during preflight")
    identity = manifest.get("campaign_identity")
    if not isinstance(identity, str) or len(identity) != 64:
        raise NFAblationError("baseline campaign identity is invalid")
    return BaselinePreflight(
        baseline_root=root,
        baseline_campaign_identity=identity,
        baseline_unified_sha256=unified_sha,
        baseline_row_count=len(baseline_rows),
        config=config,
        cells=cells,
        input_sha256_by_key=fresh_input_sha,
        source_records=source_records,
    )


def _finite_metric(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise NFAblationError("validation score contains a non-finite value")
    return result


def _score_key(record: Mapping[str, Any]) -> tuple[str, str, int, str]:
    return (
        str(record["candidate_id"]),
        str(record["readout"]),
        int(record["seed"]),
        str(record["cell_key"]),
    )


def rank_stage1(
    validation_records: Sequence[Mapping[str, Any]],
    *,
    maximum_promotions: int = 2,
) -> tuple[Promotion, ...]:
    """Rank every candidate/readout globally, then promote at most two configs.

    Scores are cell losses: known-LID MAE, absolute E1 mean-delta error, or E5
    paired MAE.  Each candidate/readout is normalized against the matching C0
    readout; a paper-parity-only readout falls back to C0/autograd.
    """

    if isinstance(maximum_promotions, bool) or not 1 <= maximum_promotions <= 2:
        raise NFAblationError("stage 1 may promote only one or two configurations")
    records = [dict(row) for row in validation_records]
    expected_candidates = {value.candidate_id for value in STAGE1_CANDIDATES}
    if {str(row.get("candidate_id")) for row in records} != expected_candidates:
        raise NFAblationError("stage-1 records do not cover the exact candidate matrix")
    observed_keys = [_score_key(row) for row in records]
    if len(observed_keys) != len(set(observed_keys)):
        raise NFAblationError("stage 1 repeats a candidate/readout/cell/seed")
    expected_keys = {
        (candidate.candidate_id, readout, 0, cell_key)
        for candidate in CANDIDATES
        for readout in READOUTS
        for cell_key in STAGE1_SENTINEL_KEYS
    } | {
        (PAPER_PARITY_CANDIDATE.candidate_id, PAPER_PARITY_READOUT, 0, cell_key)
        for cell_key in STAGE1_SENTINEL_KEYS
    }
    if set(observed_keys) != expected_keys:
        raise NFAblationError(
            "stage 1 does not cover the exact candidate/readout/sentinel matrix"
        )
    for row in records:
        if int(row.get("seed", -1)) != 0 or row.get("split") != "validation":
            raise NFAblationError("stage 1 must be seed-0 validation-only")
        if row.get("cell_key") not in STAGE1_SENTINEL_KEYS:
            raise NFAblationError("stage 1 contains a non-sentinel cell")
        if _finite_metric(row.get("finite_fraction")) != 1.0:
            # Failed readouts remain auditable but cannot pass any gate.
            row["_invalid"] = True

    by_candidate_readout: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in records:
        by_candidate_readout[(str(row["candidate_id"]), str(row["readout"]))].append(
            row
        )
    baseline: dict[tuple[str, str], dict[str, Any]] = {}
    for (candidate_id, readout), rows in by_candidate_readout.items():
        if candidate_id != "C0":
            continue
        for row in rows:
            baseline[(readout, str(row["cell_key"]))] = row
    if not baseline:
        raise NFAblationError("stage 1 lacks the C0 baseline")

    ranked: list[Promotion] = []
    epsilon = 1.0e-12
    for (candidate_id, readout), rows in sorted(by_candidate_readout.items()):
        if candidate_id == "C0" or any(row.get("_invalid") for row in rows):
            continue
        effective = list(rows)
        if not effective:
            continue
        ratios: list[float] = []
        log_ratios: list[float] = []
        by_stratum: dict[str, list[float]] = defaultdict(list)
        dataset_losses: list[float] = []
        coefficient_losses: list[float] = []
        runtime_ratios: list[float] = []
        missing = False
        for row in effective:
            cell_key = str(row["cell_key"])
            base = baseline.get((readout, cell_key)) or baseline.get(
                ("autograd", cell_key)
            )
            if base is None or base.get("_invalid"):
                missing = True
                break
            loss = max(_finite_metric(row["loss"]), epsilon)
            base_loss = max(_finite_metric(base["loss"]), epsilon)
            ratio = loss / base_loss
            ratios.append(ratio)
            log_ratios.append(math.log(ratio))
            stratum = str(row["stratum"])
            by_stratum[stratum].append(ratio)
            candidate_runtime = _finite_metric(row.get("runtime_seconds", 0.0))
            baseline_runtime = _finite_metric(base.get("runtime_seconds", 0.0))
            if baseline_runtime <= 0.0 or candidate_runtime < 0.0:
                missing = True
                break
            runtime_ratios.append(candidate_runtime / baseline_runtime)
            representation = str(row.get("representation", ""))
            if stratum.startswith("known") and representation == "dataset":
                dataset_losses.append(loss)
            elif stratum.startswith("known") and representation == "coefficients":
                coefficient_losses.append(loss)
        if missing or not log_ratios:
            continue
        stratum_medians = {
            key: float(np.median(values)) for key, values in sorted(by_stratum.items())
        }
        if any(value > 1.10 for value in stratum_medians.values()):
            continue
        dataset_coeff_ratio: float | None = None
        if dataset_losses and coefficient_losses:
            dataset_coeff_ratio = float(np.mean(dataset_losses)) / max(
                float(np.mean(coefficient_losses)), epsilon
            )
        median_log_ratio = float(np.median(log_ratios))
        if (
            math.exp(median_log_ratio) > 0.90
            or sum(value < 1.0 for value in ratios) < 5
            or not runtime_ratios
            or float(np.median(runtime_ratios))
            > (12.0 if candidate_id == PAPER_PARITY_CANDIDATE.candidate_id else 3.0)
        ):
            continue
        ranked.append(
            Promotion(
                candidate_id=candidate_id,
                readout=readout,
                median_log_ratio=median_log_ratio,
                win_rate=float(np.mean(np.asarray(ratios) < 1.0)),
                stratum_median_ratios=stratum_medians,
                dataset_to_coefficients_ratio=dataset_coeff_ratio,
            )
        )
    # One configuration is promoted only once, with its best globally ranked
    # readout.  This prevents five correlated readouts from consuming both slots.
    ranked.sort(key=lambda row: (row.median_log_ratio, row.candidate_id, row.readout))
    selected: list[Promotion] = []
    seen_candidates: set[str] = set()
    for row in ranked:
        if row.candidate_id in seen_candidates:
            continue
        seen_candidates.add(row.candidate_id)
        selected.append(row)
        if len(selected) == maximum_promotions:
            break
    if not selected:
        raise NFAblationError("no NF candidate passed the stage-1 validation gates")
    return tuple(selected)


def pick_stage2_winner(
    validation_records: Sequence[Mapping[str, Any]],
    promotions: Sequence[Promotion],
) -> Winner:
    """Freeze one candidate/readout using all 19 known-LID cells and seeds 0/1."""

    promoted_ids = {row.candidate_id for row in promotions}
    if not 1 <= len(promoted_ids) <= 2:
        raise NFAblationError("stage 2 requires one or two promoted configurations")
    promoted_pairs = {(row.candidate_id, row.readout) for row in promotions}
    if len(promoted_pairs) != len(promoted_ids):
        raise NFAblationError("stage 2 promotions repeat a candidate")
    grouped: dict[tuple[str, str], dict[tuple[int, str], float]] = defaultdict(dict)
    for row in validation_records:
        candidate_id = str(row.get("candidate_id"))
        if candidate_id not in {*promoted_ids, "C0"}:
            raise NFAblationError(
                "stage 2 contains a non-promoted/non-control candidate"
            )
        if row.get("split") != "validation" or row.get("target_policy") != "known_lid":
            raise NFAblationError("stage 2 may contain only known-LID validation rows")
        seed = int(row.get("seed", -1))
        if seed not in {0, 1}:
            raise NFAblationError("stage 2 requires exactly seeds 0 and 1")
        if _finite_metric(row.get("finite_fraction")) != 1.0:
            raise NFAblationError("stage 2 contains a non-finite validation record")
        key = (candidate_id, str(row["readout"]))
        # Workers evaluate all readouts from a shared checkpoint.  Only the
        # exact candidate/readout pair promoted in stage 1 remains eligible
        # for stage-2 model selection; the other readouts stay diagnostic.
        if candidate_id != "C0" and key not in promoted_pairs:
            continue
        observation = (seed, str(row["cell_key"]))
        if observation in grouped[key]:
            raise NFAblationError("stage 2 repeats a candidate/readout/cell/seed")
        grouped[key][observation] = _finite_metric(row["loss"])
    expected_cells = {
        key
        for key in campaign.APPROVED_GLOBAL_CELL_KEYS
        if not key.startswith("e1/e1_sampled") and not key.startswith("e5/")
    }
    expected_coverage = {(seed, key) for seed in (0, 1) for key in expected_cells}
    canonical_cells = {
        key for key in expected_cells if not key.startswith(("e3/", "e4/"))
    }
    generated_cells = expected_cells - canonical_cells
    candidates: list[Winner] = []
    epsilon = 1.0e-12
    for key, observations in grouped.items():
        if key[0] == "C0" or set(observations) != expected_coverage:
            continue
        base_key = ("C0", key[1])
        if base_key not in grouped:
            base_key = ("C0", "autograd")
        baseline = grouped.get(base_key)
        if baseline is None or set(baseline) != expected_coverage:
            continue
        per_cell_log_ratio: dict[str, float] = {}
        for cell_key in expected_cells:
            seed_logs = [
                math.log(max(observations[(seed, cell_key)], epsilon))
                - math.log(max(baseline[(seed, cell_key)], epsilon))
                for seed in (0, 1)
            ]
            per_cell_log_ratio[cell_key] = float(np.mean(seed_logs))
        canonical_ratios = [
            math.exp(per_cell_log_ratio[cell_key]) for cell_key in canonical_cells
        ]
        generated_ratios = [
            math.exp(per_cell_log_ratio[cell_key]) for cell_key in generated_cells
        ]
        geometric_ratio = math.exp(
            float(
                np.mean([per_cell_log_ratio[cell_key] for cell_key in canonical_cells])
            )
        )
        wins = sum(value < 1.0 for value in canonical_ratios)
        regressions = sum(value > 1.25 for value in canonical_ratios)
        if geometric_ratio > 0.85 or wins < 10 or regressions > 2:
            continue
        losses = list(observations.values())
        candidates.append(
            Winner(
                candidate_id=key[0],
                readout=key[1],
                validation_median_mae=float(np.median(losses)),
                validation_mean_mae=float(np.mean(losses)),
                canonical_geometric_mean_ratio=geometric_ratio,
                canonical_wins=wins,
                canonical_regressions_over_25pct=regressions,
                generated_geometric_mean_ratio=(
                    float(np.exp(np.mean(np.log(generated_ratios))))
                    if generated_ratios
                    else None
                ),
            )
        )
    if not candidates:
        raise NFAblationError(
            "no promoted candidate has complete finite stage-2 coverage"
        )
    candidates.sort(
        key=lambda row: (
            row.canonical_geometric_mean_ratio,
            row.validation_median_mae,
            row.validation_mean_mae,
            row.candidate_id,
            row.readout,
        )
    )
    return candidates[0]


def merge_unified_results(
    *,
    baseline_csv: str | Path,
    extension_rows: Sequence[Mapping[str, Any]],
    output_csv: str | Path,
) -> dict[str, Any]:
    """Append schema-compatible NF rows without rewriting the sealed baseline."""

    source = Path(baseline_csv).resolve()
    destination = Path(output_csv).resolve()
    if source == destination:
        raise NFAblationError("combined CSV must not overwrite the baseline CSV")
    original_bytes = source.read_bytes()
    original_sha = hashlib.sha256(original_bytes).hexdigest()
    header, baseline_rows = _read_csv(source)
    if tuple(header) != tuple(campaign._UNIFIED_TABLE_FIELDS):
        raise NFAblationError("baseline CSV schema differs from global campaign")
    key_fields = (
        "model_variant",
        "analysis",
        "suite_id",
        "dataset",
        "representation",
        "split",
        "readout",
    )
    seen = {tuple(row[field] for field in key_fields) for row in baseline_rows}
    normalized: list[dict[str, Any]] = []
    for raw in extension_rows:
        if set(raw) != set(header):
            raise NFAblationError("extension row is not schema-compatible")
        row = {field: "" if raw[field] is None else raw[field] for field in header}
        key = tuple(str(row[field]) for field in key_fields)
        if key in seen:
            raise NFAblationError(f"duplicate unified result key: {key}")
        seen.add(key)
        normalized.append(row)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=header,
        extrasaction="raise",
        lineterminator="\n",
    )
    for row in normalized:
        writer.writerow(row)
    suffix = stream.getvalue().encode("utf-8")
    if original_bytes and not original_bytes.endswith(b"\n"):
        suffix = b"\n" + suffix
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_bytes(original_bytes + suffix)
    os.replace(temporary, destination)
    if _sha256_path(source) != original_sha:
        raise NFAblationError("baseline CSV changed while producing the extension")
    return {
        "baseline_sha256": original_sha,
        "baseline_rows": len(baseline_rows),
        "extension_rows": len(normalized),
        "combined_rows": len(baseline_rows) + len(normalized),
        "combined_sha256": _sha256_path(destination),
    }


def _candidate_model(candidate: NFCandidate, *, seed: int) -> dict[str, Any]:
    model = dict(campaign._resolved_pilot_model("scale_conditioned_nf", seed))
    model["training"] = candidate.training_config(model["training"], seed=seed)
    model["scales"] = list(candidate.selection_scales)
    model["ablation_candidate_id"] = candidate.candidate_id
    model["ablation_contract"] = candidate.contract
    return model


def _model_variant(candidate: NFCandidate, *, seed: int) -> str:
    training = dict(candidate.training_overrides)
    if candidate.independent_fixed_epsilon:
        stem = "nf-independent-fixed-epsilon-realnvp-global-ols9"
    else:
        stem = (
            f"nf-realnvp-b{training['batch_size']}-e{training['epochs']}"
            f"-h{training.get('hidden_dim', 512)}"
            f"-l{training.get('num_coupling_layers', 8)}"
            f"-d{training.get('conditioner_depth', 2)}"
            f"-f{int(training.get('max_condition_frequency', 100))}"
        )
    return f"{stem}-seed{seed}"


def _plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise NFAblationError("non-finite values are forbidden in JSON artifacts")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _plain(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(child) for child in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    campaign._write_json(path, _plain(value))


def _safe(value: str) -> str:
    return campaign._safe_component(value)


def _task_identity(task: NFCellTask, model: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": NF_ABLATION_SCHEMA_VERSION,
        "ablation_id": NF_ABLATION_ID,
        "baseline_unified_sha256": task.baseline_unified_sha256,
        "candidate_id": task.candidate_id,
        "candidate_contract": candidate_by_id(task.candidate_id).contract,
        "training_seed": task.seed,
        "partition_seed": PARTITION_SEED,
        "evaluation_splits": (
            ["validation", "test"] if task.evaluate_test else ["validation"]
        ),
        "test_readout": task.test_readout,
        "cell": _plain(task.cell),
        "input_sha256": task.expected_input_sha256,
        "source_record": _plain(task.source_record),
        "model": _plain(model),
        "common_conditional_selection_scales": list(CONDITIONAL_SELECTION_SCALES),
        "paper_parity_selection_scales": list(SELECTION_SCALES),
    }


def _task_paths(task: NFCellTask, identity: Mapping[str, Any]) -> tuple[Path, Path]:
    root = Path(task.campaign_root)
    cell_id = _canonical_sha(identity)[:20]
    base = (
        root
        / "cells"
        / task.candidate_id.lower()
        / f"seed-{task.seed}"
        / _safe(str(task.cell.suite_id))
    )
    label = (
        f"{_safe(str(task.cell.dataset))}__"
        f"{_safe(str(task.cell.representation))}__{cell_id}"
    )
    return base / label, base / f".{label}.incomplete"


def _manifest_outputs(
    directory: Path, *, exclude: Iterable[str] = ()
) -> list[dict[str, Any]]:
    excluded = set(exclude)
    values: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(directory).as_posix()
        if relative in excluded or relative == "manifest.json":
            continue
        values.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": _sha256_path(path),
            }
        )
    return values


def _validate_cell(directory: Path, identity: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        actual_identity = json.loads(
            (directory / "identity.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"cannot read sealed metadata: {exc}"]
    if campaign.canonical_json(actual_identity) != campaign.canonical_json(identity):
        errors.append("identity differs")
    if manifest.get("identity_sha256") != _canonical_sha(identity):
        errors.append("manifest identity SHA differs")
    expected_run_id = _canonical_sha(identity)
    if summary.get("run_id") != expected_run_id:
        errors.append("summary run ID differs")
    if summary.get("cell_id") != expected_run_id[:20]:
        errors.append("summary cell ID differs")
    if summary.get("run_id_contract") != "full_sha256_of_cell_identity_v1":
        errors.append("summary run ID contract differs")
    if summary.get("candidate_id") != identity.get("candidate_id"):
        errors.append("summary candidate differs")
    if summary.get("seed") != identity.get("training_seed"):
        errors.append("summary seed differs")
    if summary.get("input_sha256") != identity.get("input_sha256"):
        errors.append("summary input differs")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list):
        errors.append("manifest outputs are missing")
    else:
        for row in outputs:
            try:
                path = directory / str(row["path"])
                if (
                    not path.is_file()
                    or path.stat().st_size != int(row["size"])
                    or _sha256_path(path) != row["sha256"]
                ):
                    errors.append(f"output differs: {row.get('path')}")
            except (KeyError, OSError, TypeError, ValueError):
                errors.append("manifest output row is malformed")
    if list(directory.rglob("checkpoint.pt")) or list(
        directory.rglob("training_progress.pt")
    ):
        errors.append("sealed cell retained a checkpoint")
    return errors


def _call_trainer(
    train_fn: Callable[..., Any],
    *,
    train: npt.ArrayLike,
    validation: npt.ArrayLike,
    training: Mapping[str, Any],
    checkpoint: Path,
    progress: Path,
) -> Any:
    parameters = inspect.signature(train_fn).parameters
    if "progress_checkpoint_path" not in parameters:
        raise NFAblationError("NF trainer lacks resume progress support")
    return train_fn(
        "scale_conditioned_nf",
        train,
        validation,
        training,
        checkpoint,
        None,
        progress_checkpoint_path=progress,
    )


def _readout_mapping(
    result: Any, *, expected_n: int
) -> dict[str, npt.NDArray[np.float64]]:
    source: Any
    if isinstance(result, Mapping):
        source = result.get("lid_by_readout", result)
    else:
        source = getattr(result, "lid_by_readout", None)
        if source is None:
            source = {
                name: getattr(result, f"lid_{name}")
                for name in READOUTS
                if hasattr(result, f"lid_{name}")
            }
    if not isinstance(source, Mapping):
        raise NFAblationError("predict_nf_readouts did not return a readout mapping")
    values: dict[str, npt.NDArray[np.float64]] = {}
    for name, raw in source.items():
        array = np.ravel(np.asarray(raw, dtype=np.float64))
        if array.shape != (expected_n,) or not np.isfinite(array).all():
            raise NFAblationError(
                f"NF readout {name!r} is not a finite length-{expected_n} vector"
            )
        values[str(name)] = np.ascontiguousarray(array)
    if set(values) != set(READOUTS):
        raise NFAblationError(
            f"NF readout API returned {sorted(values)}, expected {list(READOUTS)}"
        )
    return values


def _result_value(result: Any, name: str) -> Any:
    if isinstance(result, Mapping):
        if name not in result:
            raise NFAblationError(
                f"predict_nf_readouts omitted likelihood evidence field {name!r}"
            )
        return result[name]
    if not hasattr(result, name):
        raise NFAblationError(
            f"predict_nf_readouts omitted likelihood evidence field {name!r}"
        )
    return getattr(result, name)


def _readout_evaluation(
    result: Any,
    *,
    expected_n: int,
    expected_epsilon: float,
) -> _NFReadoutEvaluation:
    """Validate and retain every likelihood value used by the NF readouts."""

    try:
        epsilon = float(_result_value(result, "epsilon"))
        finite_difference_log_step = float(
            _result_value(result, "finite_difference_log_step")
        )
        ols_log_step = float(_result_value(result, "ols_log_step"))
    except (TypeError, ValueError) as exc:
        raise NFAblationError("NF likelihood evidence metadata are invalid") from exc
    if not math.isclose(epsilon, expected_epsilon, rel_tol=0.0, abs_tol=1.0e-12):
        raise NFAblationError("NF likelihood evidence epsilon differs from the request")
    if not math.isclose(
        finite_difference_log_step,
        FINITE_DIFFERENCE_LOG_STEP,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ) or not math.isclose(
        ols_log_step,
        OLS_LOG_STEP,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise NFAblationError("NF likelihood evidence bandwidth differs from contract")

    finite_difference_epsilons = np.ravel(
        np.asarray(
            _result_value(result, "finite_difference_epsilons"), dtype=np.float64
        )
    )
    ols_epsilons = np.ravel(
        np.asarray(_result_value(result, "ols_epsilons"), dtype=np.float64)
    )
    finite_difference_log_likelihood = np.asarray(
        _result_value(result, "finite_difference_log_likelihood"), dtype=np.float64
    )
    ols_log_likelihood = np.asarray(
        _result_value(result, "ols_log_likelihood"), dtype=np.float64
    )
    if (
        finite_difference_epsilons.shape != (2,)
        or ols_epsilons.shape != (9,)
        or finite_difference_log_likelihood.shape != (expected_n, 2)
        or ols_log_likelihood.shape != (expected_n, 9)
    ):
        raise NFAblationError("NF likelihood evidence has an invalid shape")
    arrays = (
        finite_difference_epsilons,
        ols_epsilons,
        finite_difference_log_likelihood,
        ols_log_likelihood,
    )
    if not all(np.isfinite(value).all() for value in arrays):
        raise NFAblationError("NF likelihood evidence contains non-finite values")
    if (
        np.any(finite_difference_epsilons <= 0.0)
        or np.any(ols_epsilons <= 0.0)
        or not np.all(np.diff(finite_difference_epsilons) > 0.0)
        or not np.all(np.diff(ols_epsilons) > 0.0)
    ):
        raise NFAblationError("NF likelihood evidence epsilon grids are invalid")
    expected_fd = expected_epsilon * np.exp(
        np.asarray((-1.0, 1.0)) * FINITE_DIFFERENCE_LOG_STEP
    )
    expected_ols = expected_epsilon * np.exp(np.arange(-4.0, 5.0) * OLS_LOG_STEP)
    if not np.allclose(
        finite_difference_epsilons, expected_fd, rtol=1.0e-5, atol=0.0
    ) or not np.allclose(ols_epsilons, expected_ols, rtol=1.0e-5, atol=0.0):
        raise NFAblationError(
            "NF likelihood evidence epsilon grids differ from contract"
        )
    return _NFReadoutEvaluation(
        epsilon=epsilon,
        finite_difference_log_step=finite_difference_log_step,
        ols_log_step=ols_log_step,
        finite_difference_epsilons=np.ascontiguousarray(finite_difference_epsilons),
        finite_difference_log_likelihood=np.ascontiguousarray(
            finite_difference_log_likelihood
        ),
        ols_epsilons=np.ascontiguousarray(ols_epsilons),
        ols_log_likelihood=np.ascontiguousarray(ols_log_likelihood),
        lid_by_readout=_readout_mapping(result, expected_n=expected_n),
    )


def _predict_readouts(
    predict_fn: Callable[..., Any],
    trained: Any,
    query: npt.ArrayLike,
    epsilon: float,
    *,
    batch_size: int,
) -> _NFReadoutEvaluation:
    result = predict_fn(
        trained,
        query,
        float(epsilon),
        family="scale_conditioned_nf",
        finite_difference_log_step=FINITE_DIFFERENCE_LOG_STEP,
        ols_log_step=OLS_LOG_STEP,
        batch_size=batch_size,
    )
    return _readout_evaluation(
        result,
        expected_n=int(np.asarray(query).shape[0]),
        expected_epsilon=float(epsilon),
    )


def _save_likelihood_evidence(
    work_dir: Path,
    *,
    stem: str,
    evaluation: _NFReadoutEvaluation,
) -> dict[str, Any]:
    """Persist one complete fixed-point likelihood path and bind its hashes."""

    arrays = {
        "finite_difference_epsilons": evaluation.finite_difference_epsilons,
        "finite_difference_log_likelihood": (
            evaluation.finite_difference_log_likelihood
        ),
        "ols_epsilons": evaluation.ols_epsilons,
        "ols_log_likelihood": evaluation.ols_log_likelihood,
    }
    files: dict[str, Any] = {}
    for name, value in arrays.items():
        path = work_dir / f"likelihood__{stem}__{name}.npy"
        campaign._save_npy(path, value)
        files[name] = {
            "path": path.name,
            "shape": list(value.shape),
            "sha256": _sha256_path(path),
        }
    return {
        "epsilon": evaluation.epsilon,
        "finite_difference_log_step": evaluation.finite_difference_log_step,
        "ols_log_step": evaluation.ols_log_step,
        "files": files,
    }


def _select_readout(
    *,
    cell: Any,
    readout: str,
    scales: npt.NDArray[np.float64],
    curve: npt.NDArray[np.float64],
    partition: Any,
    selection_config: Mapping[str, Any],
    reference_summary: Mapping[str, Any] | None,
) -> tuple[int, Mapping[str, Any], Mapping[str, Any] | None]:
    if cell.target_policy == "known_lid":
        if partition.selection_target is None:
            raise NFAblationError(f"known-LID cell lacks train target: {cell.key}")
        index, diagnostics = campaign._select_supervised(
            scales,
            curve,
            partition.selection_target,
            prefer="smaller",
            tolerance=float(selection_config["tie_tolerance"]),
        )
        return index, diagnostics, None
    if cell.reference_dataset not in {None, cell.dataset}:
        if reference_summary is None:
            raise NFAblationError(f"reference summary is unavailable for {cell.key}")
        try:
            reference = reference_summary["readouts"][readout]
            selected_scale = float(reference["selected_scale"])
        except (KeyError, TypeError, ValueError) as exc:
            raise NFAblationError("reference summary lacks a matching readout") from exc
        matches = np.flatnonzero(
            np.isclose(scales, selected_scale, rtol=0.0, atol=1e-12)
        )
        if matches.size != 1:
            raise NFAblationError(
                "reference selected scale is absent from this candidate"
            )
        index = int(matches[0])
        binding = {
            "cell_key": (
                f"{cell.suite_id}/{cell.reference_dataset}/{cell.representation}"
            ),
            "candidate_id": reference_summary["candidate_id"],
            "seed": reference_summary["seed"],
            "readout": readout,
            "selected_scale": selected_scale,
            "summary_sha256": reference_summary["summary_sha256"],
        }
        return (
            index,
            {
                "criterion": "reference_cell_train_stability",
                "reference_selected_scale": selected_scale,
            },
            binding,
        )
    index, diagnostics = select_stable_scale(
        scales,
        curve,
        window=int(selection_config["stability_window"]),
        min_valid_fraction=float(selection_config["stability_min_valid_fraction"]),
        prefer="smaller",
    )
    return index, diagnostics, None


def _find_reference_summary(task: NFCellTask) -> dict[str, Any] | None:
    cell = task.cell
    if cell.reference_dataset in {None, cell.dataset}:
        return None
    parent = (
        Path(task.campaign_root)
        / "cells"
        / task.candidate_id.lower()
        / f"seed-{task.seed}"
        / _safe(str(cell.suite_id))
    )
    prefix = (
        f"{_safe(str(cell.reference_dataset))}__{_safe(str(cell.representation))}__"
    )
    matches = [
        path
        for path in parent.glob(f"{prefix}*")
        if path.is_dir() and not path.name.startswith(".")
    ]
    if len(matches) != 1:
        raise NFAblationError(
            f"expected one sealed reference for {cell.key}, found {len(matches)}"
        )
    summary_path = matches[0] / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["summary_sha256"] = _sha256_path(summary_path)
    return summary


def _training_attestation(
    trained: Any, checkpoint: Path, training: Mapping[str, Any]
) -> dict[str, Any]:
    declared_config = getattr(trained, "config", None)
    if declared_config is None:
        raise NFAblationError("round-tripped NF checkpoint lacks training config")
    campaign._require_matching_training_configs(declared_config, training)
    checkpoint_sha = _sha256_path(checkpoint)
    declared_sha = getattr(trained, "checkpoint_sha256", None)
    if declared_sha != checkpoint_sha:
        raise NFAblationError("round-tripped NF checkpoint SHA differs")
    history = getattr(trained, "history", None)
    if not history:
        raise NFAblationError("round-tripped NF checkpoint lacks training history")
    return {
        "schema_version": 1,
        "checkpoint_sha256": checkpoint_sha,
        "training_config": campaign._canonical_training_config_record(
            declared_config, field="round-tripped NF training config"
        ),
        "training_config_sha256": _canonical_sha(
            campaign._canonical_training_config_record(
                declared_config, field="round-tripped NF training config"
            )
        ),
        "best_epoch": int(trained.best_epoch),
        "best_validation_loss": float(trained.best_validation_loss),
        "history": [_plain(row) for row in history],
        "retention": "pruned_after_inline_evaluation",
    }


def _run_conditional_cell(
    task: NFCellTask,
    *,
    candidate: NFCandidate,
    model: Mapping[str, Any],
    data: Any,
    partition: Any,
    work_dir: Path,
    dependencies: NFDependencies,
) -> dict[str, Any]:
    assert dependencies.train_fn is not None
    assert dependencies.load_checkpoint_fn is not None
    assert dependencies.predict_readouts_fn is not None
    checkpoint = work_dir / "checkpoint.pt"
    progress = work_dir / "training_progress.pt"
    training = model["training"]
    if not (checkpoint.is_file() and not progress.exists()):
        _call_trainer(
            dependencies.train_fn,
            train=partition.fit_features,
            validation=partition.selection_features,
            training=training,
            checkpoint=checkpoint,
            progress=progress,
        )
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise NFAblationError("NF trainer did not write a checkpoint")
    trained = dependencies.load_checkpoint_fn(
        checkpoint, device=str(training["device"])
    )
    attestation = _training_attestation(trained, checkpoint, training)
    _write_json(work_dir / "training_attestation.json", attestation)
    scales = np.asarray(CONDITIONAL_SELECTION_SCALES, dtype=np.float64)
    batch_size = 512
    curve_by_readout = {
        name: np.empty(
            (partition.selection_features.shape[0], scales.size), dtype=np.float64
        )
        for name in READOUTS
    }
    train_likelihood_evidence: list[dict[str, Any]] = []
    for index, scale in enumerate(scales):
        evaluation = _predict_readouts(
            dependencies.predict_readouts_fn,
            trained,
            partition.selection_features,
            float(scale),
            batch_size=batch_size,
        )
        train_likelihood_evidence.append(
            {
                "scope": "train_selection",
                "selected_index": index,
                **_save_likelihood_evidence(
                    work_dir,
                    stem=f"train-selection-scale-{index:02d}",
                    evaluation=evaluation,
                ),
            }
        )
        for readout in READOUTS:
            curve_by_readout[readout][:, index] = evaluation.lid_by_readout[readout]
    reference_summary = _find_reference_summary(task)
    readout_summaries: dict[str, Any] = {}
    split_prediction_cache: dict[tuple[str, int], _NFReadoutEvaluation] = {}
    evaluated_likelihood_evidence: dict[tuple[str, int], dict[str, Any]] = {}
    for readout in READOUTS:
        curve = np.ascontiguousarray(curve_by_readout[readout])
        campaign._save_npy(work_dir / f"train_selection_curve__{readout}.npy", curve)
        selected_index, diagnostics, binding = _select_readout(
            cell=task.cell,
            readout=readout,
            scales=scales,
            curve=curve,
            partition=partition,
            selection_config=task.config["campaign"]["selection"],
            reference_summary=reference_summary,
        )
        selected_scale = float(scales[selected_index])
        split_metrics: dict[str, Any] = {}
        for split, query, target in (
            ("validation", data.validation, data.validation_target),
            ("test", data.test, data.test_target),
        ):
            if split == "test" and not task.evaluate_test:
                continue
            if split == "test" and task.test_readout != readout:
                continue
            cache_key = (split, selected_index)
            evaluation = split_prediction_cache.get(cache_key)
            if evaluation is None:
                evaluation = _predict_readouts(
                    dependencies.predict_readouts_fn,
                    trained,
                    query,
                    selected_scale,
                    batch_size=batch_size,
                )
                split_prediction_cache[cache_key] = evaluation
                evaluated_likelihood_evidence[cache_key] = {
                    "scope": split,
                    "selected_index": selected_index,
                    **_save_likelihood_evidence(
                        work_dir,
                        stem=f"{split}-scale-{selected_index:02d}",
                        evaluation=evaluation,
                    ),
                }
            prediction = evaluation.lid_by_readout[readout]
            campaign._save_npy(
                work_dir / f"{split}_prediction__{readout}.npy", prediction
            )
            split_metrics[split] = (
                prediction_summary(prediction)
                if target is None
                else known_lid_metrics(prediction, target)
            )
        readout_summaries[readout] = {
            "selected_index": int(selected_index),
            "selected_scale": selected_scale,
            "selection": _plain(diagnostics),
            "reference_binding": binding,
            "metrics": split_metrics,
        }
    checkpoint.unlink()
    progress.unlink(missing_ok=True)
    return {
        "candidate_contract": candidate.contract,
        "paper_parity_status": "not_applicable_shared_conditional_density",
        "readouts": readout_summaries,
        "likelihood_path_evidence": {
            "schema_version": 1,
            "contract": "fixed_point_fd2_ols9_likelihood_paths_v1",
            "train_selection": train_likelihood_evidence,
            "evaluated_splits": [
                evaluated_likelihood_evidence[key]
                for key in sorted(evaluated_likelihood_evidence)
            ],
        },
        "training_attestation_sha256": _sha256_path(
            work_dir / "training_attestation.json"
        ),
    }


def _global_ols_lid(
    log_likelihood: npt.ArrayLike,
    *,
    scales: npt.ArrayLike,
    ambient_dim: int,
) -> npt.NDArray[np.float64]:
    values = np.asarray(log_likelihood, dtype=np.float64)
    epsilon = np.ravel(np.asarray(scales, dtype=np.float64))
    if values.ndim != 2 or values.shape[1] != epsilon.size:
        raise NFAblationError("paper-parity likelihood curve has the wrong shape")
    if epsilon.size != 9 or not np.isfinite(values).all() or not np.all(epsilon > 0.0):
        raise NFAblationError("paper-parity likelihood curve is invalid")
    coordinate = np.log(epsilon)
    centered = coordinate - float(np.mean(coordinate))
    denominator = float(np.dot(centered, centered))
    slopes = (values @ centered) / denominator
    prediction = float(ambient_dim) + slopes
    if not np.isfinite(prediction).all():
        raise NFAblationError("paper-parity global OLS produced non-finite LID")
    return np.ascontiguousarray(prediction, dtype=np.float64)


def _component_is_complete(
    component_dir: Path, *, expected_training_sha256: str
) -> bool:
    attestation_path = component_dir / "attestation.json"
    if not attestation_path.is_file():
        return False
    try:
        attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if attestation.get("training_config_sha256") != expected_training_sha256:
        return False
    outputs = attestation.get("likelihood_outputs")
    if not isinstance(outputs, list):
        return False
    for row in outputs:
        try:
            path = component_dir / str(row["path"])
            if not path.is_file() or _sha256_path(path) != row["sha256"]:
                return False
        except (KeyError, OSError, TypeError):
            return False
    return (
        not (component_dir / "checkpoint.pt").exists()
        and not (component_dir / "training_progress.pt").exists()
    )


def _run_paper_parity_cell(
    task: NFCellTask,
    *,
    candidate: NFCandidate,
    model: Mapping[str, Any],
    data: Any,
    partition: Any,
    work_dir: Path,
    dependencies: NFDependencies,
) -> dict[str, Any]:
    assert dependencies.train_fn is not None
    assert dependencies.load_checkpoint_fn is not None
    if dependencies.predict_log_likelihood_fn is None:
        raise NFAblationError(
            "paper-parity P0 requires models.training.predict_nf_log_likelihood"
        )
    scales = np.asarray(SELECTION_SCALES, dtype=np.float64)
    components_root = work_dir / "components"
    components_root.mkdir(exist_ok=True)
    split_queries: list[tuple[str, Any]] = [
        ("train_selection", partition.selection_features),
        ("validation", data.validation),
    ]
    if task.evaluate_test:
        split_queries.append(("test", data.test))
    likelihood_columns: dict[str, list[npt.NDArray[np.float64]]] = {
        split: [] for split, _ in split_queries
    }
    component_attestations: list[dict[str, Any]] = []
    for index, epsilon in enumerate(scales):
        component_dir = components_root / f"epsilon-{index:02d}-{epsilon:.8g}"
        component_dir.mkdir(exist_ok=True)
        training = dict(model["training"])
        # Exact equality is a checkpointed training contract: the objective
        # uses torch.full and consumes no epsilon-sampling RNG draw.
        training["epsilon_min"] = float(epsilon)
        training["epsilon_max"] = float(epsilon)
        training = campaign._canonical_training_config_record(
            training, field=f"P0 fixed-epsilon component {index}"
        )
        training_sha = _canonical_sha(training)
        if not _component_is_complete(
            component_dir, expected_training_sha256=training_sha
        ):
            checkpoint = component_dir / "checkpoint.pt"
            progress = component_dir / "training_progress.pt"
            if not (checkpoint.is_file() and not progress.exists()):
                _call_trainer(
                    dependencies.train_fn,
                    train=partition.fit_features,
                    validation=partition.selection_features,
                    training=training,
                    checkpoint=checkpoint,
                    progress=progress,
                )
            if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
                raise NFAblationError("P0 trainer did not write a component checkpoint")
            trained = dependencies.load_checkpoint_fn(
                checkpoint, device=str(training["device"])
            )
            attestation = _training_attestation(trained, checkpoint, training)
            likelihood_outputs: list[dict[str, Any]] = []
            for split, query in split_queries:
                prediction = np.ravel(
                    np.asarray(
                        dependencies.predict_log_likelihood_fn(
                            trained,
                            query,
                            float(epsilon),
                            family="scale_conditioned_nf",
                            batch_size=512,
                        ),
                        dtype=np.float64,
                    )
                )
                if (
                    prediction.shape != (int(np.asarray(query).shape[0]),)
                    or not np.isfinite(prediction).all()
                ):
                    raise NFAblationError("P0 log likelihood is not a finite vector")
                path = component_dir / f"{split}_log_likelihood.npy"
                campaign._save_npy(path, prediction)
                likelihood_outputs.append(
                    {
                        "path": path.name,
                        "sha256": _sha256_path(path),
                    }
                )
            component_record = {
                **attestation,
                "contract": "exact_fixed_epsilon_realnvp_component_v1",
                "epsilon": float(epsilon),
                "component_index": index,
                "likelihood_outputs": likelihood_outputs,
            }
            _write_json(component_dir / "attestation.json", component_record)
            checkpoint.unlink()
            progress.unlink(missing_ok=True)
            if not _component_is_complete(
                component_dir, expected_training_sha256=training_sha
            ):
                raise NFAblationError("P0 component failed post-prune validation")
            del trained
            campaign._clear_accelerator_cache()
        component_record = json.loads(
            (component_dir / "attestation.json").read_text(encoding="utf-8")
        )
        component_attestations.append(component_record)
        for split, _ in split_queries:
            likelihood_columns[split].append(
                np.asarray(
                    np.load(
                        component_dir / f"{split}_log_likelihood.npy",
                        allow_pickle=False,
                    ),
                    dtype=np.float64,
                )
            )
    ambient_dim = int(np.asarray(partition.fit_features).shape[1])
    predictions: dict[str, npt.NDArray[np.float64]] = {}
    for split, columns in likelihood_columns.items():
        curve = np.ascontiguousarray(np.column_stack(columns), dtype=np.float64)
        campaign._save_npy(work_dir / f"{split}_global_likelihood_curve.npy", curve)
        predictions[split] = _global_ols_lid(
            curve, scales=scales, ambient_dim=ambient_dim
        )
    split_metrics: dict[str, Any] = {}
    for split, target in (
        ("validation", data.validation_target),
        ("test", data.test_target),
    ):
        if split not in predictions:
            continue
        prediction = predictions[split]
        campaign._save_npy(
            work_dir / f"{split}_prediction__{PAPER_PARITY_READOUT}.npy",
            prediction,
        )
        split_metrics[split] = (
            prediction_summary(prediction)
            if target is None
            else known_lid_metrics(prediction, target)
        )
    reference_summary = _find_reference_summary(task)
    reference_binding = None
    if reference_summary is not None:
        reference_binding = {
            "cell_key": (
                f"{task.cell.suite_id}/{task.cell.reference_dataset}/"
                f"{task.cell.representation}"
            ),
            "candidate_id": reference_summary["candidate_id"],
            "seed": reference_summary["seed"],
            "readout": PAPER_PARITY_READOUT,
            "summary_sha256": reference_summary["summary_sha256"],
        }
    component_index = {
        "schema_version": 1,
        "contract": "exact_independent_fixed_epsilon_realnvp_global_ols9_v1",
        "scales": scales.tolist(),
        "components": [
            {
                "component_index": row["component_index"],
                "epsilon": row["epsilon"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "training_config_sha256": row["training_config_sha256"],
            }
            for row in component_attestations
        ],
    }
    _write_json(work_dir / "component_index.json", component_index)
    return {
        "candidate_contract": candidate.contract,
        "paper_parity_status": "exact_fixed_epsilon_components_global_ols9",
        "readouts": {
            PAPER_PARITY_READOUT: {
                "selected_index": None,
                "selected_scale": None,
                "selection": {
                    "criterion": "predeclared_global_ols_over_all_9_fixed_epsilons"
                },
                "reference_binding": reference_binding,
                "metrics": split_metrics,
            }
        },
        "training_attestation_sha256": _sha256_path(work_dir / "component_index.json"),
    }


def _run_cell_task(
    task: NFCellTask,
    dependencies: NFDependencies | None = None,
) -> dict[str, Any]:
    import time

    started = time.monotonic()
    dependencies = NFDependencies() if dependencies is None else dependencies
    deps = _resolve_dependencies(dependencies)
    assert deps.cell_loader is not None
    candidate = candidate_by_id(task.candidate_id)
    model = _candidate_model(candidate, seed=task.seed)
    identity = _task_identity(task, model)
    final_dir, work_dir = _task_paths(task, identity)
    if final_dir.exists():
        errors = _validate_cell(final_dir, identity)
        if errors:
            raise NFAblationError(
                f"refusing to reuse invalid NF ablation cell {final_dir}: {errors}"
            )
        summary = json.loads((final_dir / "summary.json").read_text(encoding="utf-8"))
        return {
            "cell_key": task.cell.key,
            "candidate_id": task.candidate_id,
            "seed": task.seed,
            "directory": str(final_dir),
            "summary": summary,
            "reused": True,
            "runtime_seconds": 0.0,
        }
    work_dir.parent.mkdir(parents=True, exist_ok=True)
    if work_dir.exists():
        identity_path = work_dir / "identity.json"
        if not identity_path.is_file() or campaign.canonical_json(
            json.loads(identity_path.read_text(encoding="utf-8"))
        ) != campaign.canonical_json(identity):
            raise NFAblationError("stable incomplete NF cell has a different identity")
    else:
        work_dir.mkdir()
        _write_json(work_dir / "identity.json", identity)
        campaign._write_yaml(work_dir / "resolved_model.yaml", model)
    # Recover the tiny post-prune/pre-rename window without retraining.
    if (
        (work_dir / "manifest.json").is_file()
        and not list(work_dir.rglob("checkpoint.pt"))
        and not list(work_dir.rglob("training_progress.pt"))
    ):
        errors = _validate_cell(work_dir, identity)
        if errors:
            raise NFAblationError(f"pruned incomplete cell is invalid: {errors}")
        os.replace(work_dir, final_dir)
        summary = json.loads((final_dir / "summary.json").read_text(encoding="utf-8"))
        return {
            "cell_key": task.cell.key,
            "candidate_id": task.candidate_id,
            "seed": task.seed,
            "directory": str(final_dir),
            "summary": summary,
            "reused": True,
            "runtime_seconds": 0.0,
        }
    raw_data = deps.cell_loader(task.cell, task.config, Path(task.project_root))
    data = campaign._bind_source_preflight(
        raw_data,
        task.cell,
        {task.cell.inventory_id: task.source_record},
    )
    if data.input_sha256 != task.expected_input_sha256:
        raise NFAblationError(f"source input changed after preflight: {task.cell.key}")
    partition = campaign.partition_source_train(
        data.train,
        data.train_target,
        selection=task.config["campaign"]["selection"],
        seed=PARTITION_SEED,
    )
    campaign._save_npy(work_dir / "train_fit_indices.npy", partition.fit_indices)
    campaign._save_npy(
        work_dir / "train_selection_indices.npy", partition.selection_indices
    )
    if partition.selection_target is not None:
        campaign._save_npy(
            work_dir / "train_selection_target.npy", partition.selection_target
        )
    for split, target, labels in (
        ("validation", data.validation_target, data.validation_labels),
        ("test", data.test_target, data.test_labels),
    ):
        if split == "test" and not task.evaluate_test:
            continue
        if target is not None:
            campaign._save_npy(work_dir / f"{split}_target.npy", target)
        if labels is not None:
            campaign._save_npy(work_dir / f"{split}_labels.npy", labels)
    if candidate.independent_fixed_epsilon:
        scientific = _run_paper_parity_cell(
            task,
            candidate=candidate,
            model=model,
            data=data,
            partition=partition,
            work_dir=work_dir,
            dependencies=deps,
        )
    else:
        scientific = _run_conditional_cell(
            task,
            candidate=candidate,
            model=model,
            data=data,
            partition=partition,
            work_dir=work_dir,
            dependencies=deps,
        )
    runtime_seconds = float(time.monotonic() - started)
    summary = {
        "schema_version": NF_ABLATION_SCHEMA_VERSION,
        "ablation_id": NF_ABLATION_ID,
        "run_id": _canonical_sha(identity),
        "run_id_contract": "full_sha256_of_cell_identity_v1",
        "cell_id": _canonical_sha(identity)[:20],
        "candidate_id": candidate.candidate_id,
        "model_variant": _model_variant(candidate, seed=task.seed),
        "seed": task.seed,
        "cell_key": task.cell.key,
        "suite_id": task.cell.suite_id,
        "dataset": task.cell.dataset,
        "representation": task.cell.representation,
        "target_policy": task.cell.target_policy,
        "input_sha256": data.input_sha256,
        "partition": _plain(partition.record),
        "evaluation_splits": (
            ["validation", "test"] if task.evaluate_test else ["validation"]
        ),
        "runtime_seconds": runtime_seconds,
        **scientific,
    }
    _write_json(work_dir / "summary.json", summary)
    manifest = {
        "schema_version": NF_ABLATION_SCHEMA_VERSION,
        "identity_sha256": _canonical_sha(identity),
        "outputs": _manifest_outputs(work_dir),
    }
    _write_json(work_dir / "manifest.json", manifest)
    errors = _validate_cell(work_dir, identity)
    if errors:
        raise NFAblationError(f"new NF ablation cell failed validation: {errors}")
    os.replace(work_dir, final_dir)
    return {
        "cell_key": task.cell.key,
        "candidate_id": task.candidate_id,
        "seed": task.seed,
        "directory": str(final_dir),
        "summary": summary,
        "reused": False,
        "runtime_seconds": runtime_seconds,
    }


def validation_score_records(
    results: Sequence[Mapping[str, Any]],
    cells: Sequence[Any],
) -> list[dict[str, Any]]:
    """Recompute validation-only gate losses from sealed pointwise predictions."""

    cell_by_key = {cell.key: cell for cell in cells}
    directories: dict[tuple[str, int, str], Path] = {}
    summaries: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for result in results:
        key = (
            str(result["candidate_id"]),
            int(result["seed"]),
            str(result["cell_key"]),
        )
        if key in directories:
            continue
        directories[key] = Path(str(result["directory"]))
        summaries[key] = dict(result["summary"])
    records: list[dict[str, Any]] = []
    for key, summary in summaries.items():
        candidate_id, seed, cell_key = key
        cell = cell_by_key[cell_key]
        directory = directories[key]
        runtime = float(summary.get("runtime_seconds", 0.0))
        for readout, readout_summary in summary["readouts"].items():
            prediction = np.asarray(
                np.load(
                    directory / f"validation_prediction__{readout}.npy",
                    allow_pickle=False,
                ),
                dtype=np.float64,
            )
            distribution = prediction_summary(prediction)
            if cell.target_policy == "known_lid":
                metrics = readout_summary["metrics"]["validation"]
                loss = float(metrics["mae"])
                stratum = f"known_{cell.representation}"
                include = True
            else:
                reference_key = (
                    cell.suite_id,
                    str(cell.reference_dataset),
                    cell.representation,
                )
                reference_cell_key = "/".join(reference_key)
                reference_record_key = (candidate_id, seed, reference_cell_key)
                if reference_record_key not in directories:
                    raise NFAblationError(
                        f"validation scoring lacks reference {reference_cell_key}"
                    )
                reference_directory = directories[reference_record_key]
                reference_prediction = np.asarray(
                    np.load(
                        reference_directory / f"validation_prediction__{readout}.npy",
                        allow_pickle=False,
                    ),
                    dtype=np.float64,
                )
                if cell.target_policy == "sample_size":
                    reference_distribution = prediction_summary(reference_prediction)
                    mean_delta = float(
                        distribution["mean"] - reference_distribution["mean"]
                    )
                    loss = abs(mean_delta - float(cell.expected_lid_delta))
                    stratum = "e1_sample_size"
                elif cell.target_policy == "paired_delta":
                    metrics = paired_delta_metrics(
                        reference_prediction,
                        prediction,
                        expected_delta=float(cell.expected_lid_delta),
                    )
                    loss = float(metrics["mae"])
                    stratum = "e5_paired_delta"
                else:
                    raise NFAblationError(
                        f"unsupported target policy {cell.target_policy}"
                    )
                include = cell.dataset != cell.reference_dataset
            records.append(
                {
                    "candidate_id": candidate_id,
                    "readout": str(readout),
                    "seed": seed,
                    "cell_key": cell_key,
                    "suite_id": cell.suite_id,
                    "representation": cell.representation,
                    "target_policy": cell.target_policy,
                    "split": "validation",
                    "loss": loss,
                    "finite_fraction": distribution["finite_fraction"],
                    "stratum": stratum,
                    "include_in_macro": include,
                    "runtime_seconds": runtime,
                }
            )
    return records


def _empty_unified_row() -> dict[str, Any]:
    return {field: "" for field in campaign._UNIFIED_TABLE_FIELDS}


def _metric_fields(row: dict[str, Any], metrics: Mapping[str, Any]) -> None:
    for field in (
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
    ):
        row[field] = metrics.get(field, "")


def build_stage3_unified_rows(
    results: Sequence[Mapping[str, Any]],
    cells: Sequence[Any],
    *,
    winner: Winner,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Render three complete final models: C0/s2 and winner/s2,s3."""

    expected_variants = {("C0", 2), (winner.candidate_id, 2), (winner.candidate_id, 3)}
    by_variant_cell: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    directories: dict[tuple[str, int, str], Path] = {}
    for result in results:
        key = (
            str(result["candidate_id"]),
            int(result["seed"]),
            str(result["cell_key"]),
        )
        if key[:2] not in expected_variants:
            raise NFAblationError("stage 3 contains an unexpected candidate/seed")
        if key in by_variant_cell:
            raise NFAblationError("stage 3 repeats a final cell")
        by_variant_cell[key] = dict(result["summary"])
        directories[key] = Path(str(result["directory"]))
    expected = {
        (candidate_id, seed, cell.key)
        for candidate_id, seed in expected_variants
        for cell in cells
    }
    if set(by_variant_cell) != expected:
        raise NFAblationError("stage 3 does not cover exact 3 x 39 final cells")
    rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for candidate_id, seed in sorted(expected_variants):
        primary = "autograd" if candidate_id == "C0" else winner.readout
        for cell in cells:
            key = (candidate_id, seed, cell.key)
            summary = by_variant_cell[key]
            directory = directories[key]
            if primary not in summary["readouts"]:
                raise NFAblationError(
                    f"final primary readout {primary!r} is absent in {cell.key}"
                )
            selected = summary["readouts"][primary]
            for split in ("validation", "test"):
                prediction = np.asarray(
                    np.load(
                        directory / f"{split}_prediction__{primary}.npy",
                        allow_pickle=False,
                    ),
                    dtype=np.float64,
                )
                variant = str(summary["model_variant"])
                row = _empty_unified_row()
                row.update(
                    {
                        "model_variant": variant,
                        "suite_id": cell.suite_id,
                        "dataset": cell.dataset,
                        "representation": cell.representation,
                        "split": split,
                        "readout": primary,
                        "primary_readout": primary,
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
                if cell.target_policy == "known_lid":
                    row["analysis"] = "known_lid"
                    row["selection_protocol"] = campaign.KNOWN_SELECTION_PROTOCOL
                    _metric_fields(row, selected["metrics"][split])
                else:
                    reference_cell = campaign._reference_cell(cells, cell)
                    reference_key = (candidate_id, seed, reference_cell.key)
                    reference_directory = directories[reference_key]
                    reference_prediction = np.asarray(
                        np.load(
                            reference_directory / f"{split}_prediction__{primary}.npy",
                            allow_pickle=False,
                        ),
                        dtype=np.float64,
                    )
                    row["reference_dataset"] = reference_cell.dataset
                    row["selection_protocol"] = campaign.UNKNOWN_SELECTION_PROTOCOL
                    if cell.target_policy == "sample_size":
                        row["analysis"] = "e1_sample_size_stability"
                        current_metrics = prediction_summary(prediction)
                        reference_metrics = prediction_summary(reference_prediction)
                        _metric_fields(row, current_metrics)
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
                        _metric_fields(row, metrics)
                        labels_path = directory / f"{split}_labels.npy"
                        reference_labels_path = (
                            reference_directory / f"{split}_labels.npy"
                        )
                        labels = np.load(labels_path, allow_pickle=False)
                        reference_labels = np.load(
                            reference_labels_path, allow_pickle=False
                        )
                        if not np.array_equal(labels, reference_labels):
                            raise NFAblationError(
                                f"paired labels differ for {cell.key}/{split}"
                            )
                        row["labels_sha256"] = campaign._array_sha(labels)
                    else:
                        raise NFAblationError(
                            f"unsupported final target policy {cell.target_policy}"
                        )
                rows.append(row)
                provenance.append(
                    {
                        "model_variant": variant,
                        "analysis": row["analysis"],
                        "suite_id": cell.suite_id,
                        "dataset": cell.dataset,
                        "representation": cell.representation,
                        "split": split,
                        "readout": primary,
                        "candidate_id": candidate_id,
                        "seed": seed,
                        "stage": 3,
                        "run_id": summary["run_id"],
                        "run_id_contract": summary["run_id_contract"],
                        "cell_id": summary["cell_id"],
                        "cell_path": str(directory),
                        "manifest_sha256": _sha256_path(directory / "manifest.json"),
                        "summary_sha256": _sha256_path(directory / "summary.json"),
                        "training_attestation_sha256": summary[
                            "training_attestation_sha256"
                        ],
                        "validation_status": "sealed_complete",
                    }
                )
    return rows, provenance


@dataclass(frozen=True)
class _PreparedNFAblation:
    project_root: str
    campaign_root: str
    campaign_identity: str
    identity_record: Mapping[str, Any]
    preflight: BaselinePreflight
    worker_count: int


def _ablation_identity_record(
    preflight: BaselinePreflight,
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Build the immutable scientific identity for all staged work."""

    return {
        "schema_version": NF_ABLATION_SCHEMA_VERSION,
        "ablation_id": NF_ABLATION_ID,
        "baseline_campaign_identity": preflight.baseline_campaign_identity,
        "baseline_unified_sha256": preflight.baseline_unified_sha256,
        "baseline_input_sha256_by_key": dict(preflight.input_sha256_by_key),
        "declared_source_sha256": campaign.hash_declared_sources(project_root),
        "coordinator_sha256": _sha256_path(Path(__file__).resolve()),
        "candidates": [_plain(value) for value in STAGE1_CANDIDATES],
        "readouts": list(READOUTS),
        "paper_parity_readout": PAPER_PARITY_READOUT,
        "deterministic_cublas_workspace_config": (
            DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG
        ),
        "finite_difference_log_step": FINITE_DIFFERENCE_LOG_STEP,
        "ols_log_step": OLS_LOG_STEP,
        "selection_scales": list(SELECTION_SCALES),
        "conditional_selection_scales": list(CONDITIONAL_SELECTION_SCALES),
        "stages": {
            "stage1": {
                "candidate_ids": [value.candidate_id for value in STAGE1_CANDIDATES],
                "cell_keys": list(STAGE1_SENTINEL_KEYS),
                "seeds": [0],
                "splits": ["validation"],
            },
            "stage2": {
                "candidate_ids": "C0_plus_stage1_promotions",
                "cell_policy": "all_19_known_lid_cells",
                "seeds": [0, 1],
                "splits": ["validation"],
            },
            "stage3": {
                "candidate_seed_pairs": [
                    ["C0", 2],
                    ["stage2_winner", 2],
                    ["stage2_winner", 3],
                ],
                "cell_policy": "all_39_cells",
                "splits": ["validation", "test"],
            },
        },
    }


def _prepare_nf_ablation(
    preflight: BaselinePreflight,
    *,
    project_root: Path,
    output_root: Path,
    worker_count: int,
) -> _PreparedNFAblation:
    if isinstance(worker_count, bool) or not 1 <= worker_count <= WORKER_COUNT:
        raise NFAblationError(f"worker_count must be in [1, {WORKER_COUNT}]")
    identity_record = _ablation_identity_record(preflight, project_root=project_root)
    identity = _canonical_sha(identity_record)
    selected_output = output_root.expanduser()
    if not selected_output.is_absolute():
        selected_output = project_root / selected_output
    selected_output = selected_output.resolve()
    campaign_root = selected_output / f"{NF_ABLATION_ID}__{identity[:20]}"
    return _PreparedNFAblation(
        project_root=str(project_root),
        campaign_root=str(campaign_root),
        campaign_identity=identity,
        identity_record=identity_record,
        preflight=preflight,
        worker_count=worker_count,
    )


def _make_cell_task(
    prepared: _PreparedNFAblation,
    *,
    cell: Any,
    candidate_id: str,
    seed: int,
    evaluate_test: bool,
    test_readout: str | None = None,
) -> NFCellTask:
    candidate = candidate_by_id(candidate_id)
    allowed_test_readouts = (
        {PAPER_PARITY_READOUT} if candidate.independent_fixed_epsilon else set(READOUTS)
    )
    if evaluate_test != (test_readout is not None):
        raise NFAblationError("test evaluation requires exactly one frozen readout")
    if test_readout is not None and test_readout not in allowed_test_readouts:
        raise NFAblationError(
            f"invalid frozen test readout {test_readout!r} for {candidate_id}"
        )
    try:
        source_record = prepared.preflight.source_records[cell.inventory_id]
        input_sha = prepared.preflight.input_sha256_by_key[cell.key]
    except KeyError as exc:
        raise NFAblationError(f"preflight evidence is missing for {cell.key}") from exc
    return NFCellTask(
        project_root=prepared.project_root,
        campaign_root=prepared.campaign_root,
        baseline_unified_sha256=prepared.preflight.baseline_unified_sha256,
        config=prepared.preflight.config,
        cell=cell,
        candidate_id=candidate_id,
        seed=seed,
        evaluate_test=evaluate_test,
        expected_input_sha256=input_sha,
        source_record=source_record,
        test_readout=test_readout,
    )


def _task_id(task: NFCellTask) -> str:
    model = _candidate_model(candidate_by_id(task.candidate_id), seed=task.seed)
    return _canonical_sha(_task_identity(task, model))


def _task_descriptor(task: NFCellTask) -> dict[str, Any]:
    return {
        "task_id": _task_id(task),
        "candidate_id": task.candidate_id,
        "seed": task.seed,
        "cell_key": task.cell.key,
        "evaluate_test": task.evaluate_test,
        "test_readout": task.test_readout,
    }


def _existing_task_result(task: NFCellTask) -> dict[str, Any] | None:
    model = _candidate_model(candidate_by_id(task.candidate_id), seed=task.seed)
    identity = _task_identity(task, model)
    final_dir, _work_dir = _task_paths(task, identity)
    if not final_dir.exists():
        return None
    errors = _validate_cell(final_dir, identity)
    if errors:
        raise NFAblationError(
            f"existing NF ablation cell is invalid: {final_dir}: {errors}"
        )
    summary = json.loads((final_dir / "summary.json").read_text(encoding="utf-8"))
    return {
        "cell_key": task.cell.key,
        "candidate_id": task.candidate_id,
        "seed": task.seed,
        "directory": str(final_dir),
        "summary": summary,
        "reused": True,
        "runtime_seconds": 0.0,
    }


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_plain(value), allow_nan=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8", newline="") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


class _NvidiaSmiMonitor:
    """Coordinator-owned, durable GPU utilization sampler."""

    _QUERY_FIELDS = (
        "index",
        "uuid",
        "name",
        "utilization.gpu",
        "memory.used",
        "memory.total",
        "power.draw",
        "pstate",
    )

    def __init__(
        self,
        path: Path,
        *,
        enabled: bool,
        expected_gpu_count: int,
        interval_seconds: float = 5.0,
    ) -> None:
        self.path = path
        self.enabled = enabled
        self.expected_gpu_count = expected_gpu_count
        self.interval_seconds = interval_seconds
        self.session_id = hashlib.sha256(
            f"{os.getpid()}:{time.time_ns()}".encode()
        ).hexdigest()[:20]
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._background_error_count = 0
        self._last_sample: dict[str, Any] | None = None

    def _sample(self) -> dict[str, Any]:
        executable = shutil.which("nvidia-smi")
        if executable is None:
            raise NFAblationError("nvidia-smi is unavailable in an eight-GPU run")
        command = [
            executable,
            f"--query-gpu={','.join(self._QUERY_FIELDS)}",
            "--format=csv,noheader,nounits",
        ]
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        rows = list(csv.reader(io.StringIO(completed.stdout)))
        if len(rows) != self.expected_gpu_count:
            raise NFAblationError(
                "nvidia-smi visible GPU count differs from the worker pool: "
                f"{len(rows)} != {self.expected_gpu_count}"
            )
        gpus: list[dict[str, Any]] = []
        for row in rows:
            if len(row) != len(self._QUERY_FIELDS):
                raise NFAblationError("nvidia-smi returned a malformed CSV row")
            values = [field.strip() for field in row]
            try:
                gpus.append(
                    {
                        "index": int(values[0]),
                        "uuid": values[1],
                        "name": values[2],
                        "utilization_gpu_percent": float(values[3]),
                        "memory_used_mib": float(values[4]),
                        "memory_total_mib": float(values[5]),
                        "power_draw_w": float(values[6]),
                        "pstate": values[7],
                    }
                )
            except ValueError as exc:
                raise NFAblationError(
                    "nvidia-smi returned a non-numeric metric"
                ) from exc
        return {
            "schema_version": 1,
            "session_id": self.session_id,
            "recorded_at_utc": datetime.now(UTC).isoformat(),
            "gpus": gpus,
        }

    def sample_once(self) -> Mapping[str, Any]:
        if not self.enabled:
            record = {
                "schema_version": 1,
                "session_id": self.session_id,
                "recorded_at_utc": datetime.now(UTC).isoformat(),
                "status": "disabled_cpu_run",
                "gpus": [],
            }
        else:
            record = self._sample()
        _append_jsonl(self.path, record)
        self._last_sample = dict(record)
        return record

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                record = self._sample()
            except BaseException as exc:  # noqa: BLE001 - telemetry is best-effort
                self._background_error_count += 1
                record = {
                    "schema_version": 1,
                    "session_id": self.session_id,
                    "recorded_at_utc": datetime.now(UTC).isoformat(),
                    "status": "periodic_sampling_error",
                    "error_index": self._background_error_count,
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "gpus": [],
                }
            # Once shutdown begins, do not mutate telemetry that may shortly be
            # hashed into the final campaign manifest.
            if self._stop.is_set():
                return
            try:
                _append_jsonl(self.path, record)
            except BaseException:  # noqa: BLE001 - telemetry must not kill science
                self._background_error_count += 1
                continue
            self._last_sample = dict(record)

    def start(self) -> Mapping[str, Any]:
        # The first sample is a strict production preflight: command errors,
        # malformed output, GPU-count mismatches, and wrong accelerators abort.
        first = self.sample_once()
        if self.enabled:
            for index, gpu in enumerate(first.get("gpus", ())):
                _require_h100_device_name(
                    gpu.get("name") if isinstance(gpu, Mapping) else None,
                    field=f"nvidia-smi GPU {index}",
                )
            self._thread = threading.Thread(
                target=self._run,
                name="nf-ablation-nvidia-smi",
                daemon=True,
            )
            self._thread.start()
        return first

    def raise_if_failed(self) -> None:
        """Compatibility hook; periodic telemetry is deliberately non-fatal."""

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            # `_sample` has a 30-second subprocess timeout.  The stop flag also
            # prevents a late sample from being appended after this method.
            self._thread.join(timeout=max(40.0, self.interval_seconds * 2.0))


def _visible_device_tokens(
    worker_count: int, *, require_cuda: bool
) -> tuple[str | None, ...]:
    if not require_cuda:
        return tuple(None for _ in range(worker_count))
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        tokens = tuple(str(index) for index in range(worker_count))
    else:
        tokens = tuple(token.strip() for token in raw.split(",") if token.strip())
    if len(tokens) != worker_count or len(set(tokens)) != worker_count:
        raise NFAblationError(
            f"expected exactly {worker_count} unique CUDA_VISIBLE_DEVICES tokens"
        )
    return tokens


def _configure_deterministic_cublas(*, require_cuda: bool) -> None:
    """Freeze the CUDA workspace contract before any worker imports PyTorch."""

    if not require_cuda:
        return
    configured = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if configured not in {None, DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG}:
        raise NFAblationError(
            "CUBLAS_WORKSPACE_CONFIG differs from the deterministic campaign "
            f"contract {DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG!r}"
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG


def _require_h100_device_name(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or "H100" not in value.upper():
        raise NFAblationError(f"{field} must identify an NVIDIA H100 GPU")
    return value


def _nf_worker_main(
    worker_slot: int,
    device_token: str | None,
    dependencies: NFDependencies,
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
    active: Mapping[str, Any] | None = None
    try:
        device_name: str | None = None
        if require_cuda:
            import torch

            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError("each NF worker must see exactly one CUDA device")
            torch.cuda.set_device(0)
            device_name = _require_h100_device_name(
                torch.cuda.get_device_name(0), field=f"NF worker {worker_slot} device"
            )
            visible_device = f"{device_token}:{device_name}"
        result_queue.put(
            {
                "kind": "ready",
                "worker_slot": worker_slot,
                "visible_device": visible_device,
                "device_name": device_name,
            }
        )
        while not stop_event.is_set():
            try:
                active = task_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if active is None:
                return
            task = active.get("task")
            if not isinstance(task, NFCellTask):
                raise NFAblationError("worker received an invalid NF cell task")
            result_queue.put(
                {
                    "kind": "started",
                    "stage": active.get("stage"),
                    "task_id": active.get("task_id"),
                    "worker_slot": worker_slot,
                    "visible_device": visible_device,
                }
            )
            result = _run_cell_task(task, dependencies)
            result_queue.put(
                {
                    "kind": "completed",
                    "stage": active.get("stage"),
                    "task_id": active.get("task_id"),
                    "worker_slot": worker_slot,
                    "visible_device": visible_device,
                    "payload": result,
                }
            )
            active = None
    except BaseException as exc:  # noqa: BLE001 - process boundary reports all
        stop_event.set()
        result_queue.put(
            {
                "kind": "failed",
                "worker_slot": worker_slot,
                "active": (
                    {
                        "stage": active.get("stage"),
                        "task_id": active.get("task_id"),
                    }
                    if isinstance(active, Mapping)
                    else None
                ),
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


class _PersistentNFPool:
    """One spawn pool kept alive across screening, promotion and confirmation."""

    def __init__(
        self,
        *,
        worker_count: int,
        dependencies: NFDependencies,
        require_cuda: bool,
    ) -> None:
        try:
            pickle.dumps(dependencies)
        except Exception as exc:
            raise NFAblationError(
                "NF worker dependencies are not spawn-picklable"
            ) from exc
        self.worker_count = worker_count
        self.dependencies = dependencies
        self.require_cuda = require_cuda
        self.context = mp.get_context("spawn")
        self.task_queue = self.context.Queue()
        self.result_queue = self.context.Queue()
        self.stop_event = self.context.Event()
        tokens = _visible_device_tokens(worker_count, require_cuda=require_cuda)
        self.processes = [
            self.context.Process(
                target=_nf_worker_main,
                name=f"nf-ablation-worker-{slot}",
                args=(
                    slot,
                    tokens[slot],
                    dependencies,
                    self.task_queue,
                    self.result_queue,
                    self.stop_event,
                    require_cuda,
                ),
            )
            for slot in range(worker_count)
        ]
        self.worker_records: list[dict[str, Any]] = []
        self._closed = False

    def start(self) -> tuple[Mapping[str, Any], ...]:
        for process in self.processes:
            process.start()
        seen: set[int] = set()
        deadline = time.monotonic() + 120.0
        try:
            while len(seen) < self.worker_count:
                timeout = min(1.0, deadline - time.monotonic())
                if timeout <= 0:
                    raise NFAblationError("NF worker pool preflight timed out")
                try:
                    message = self.result_queue.get(timeout=timeout)
                except queue.Empty:
                    exited = [
                        process
                        for process in self.processes
                        if process.exitcode is not None
                    ]
                    if exited:
                        raise NFAblationError(
                            "NF worker exited during preflight: "
                            + ", ".join(
                                f"{process.name}={process.exitcode}"
                                for process in exited
                            )
                        )
                    continue
                if not isinstance(message, Mapping) or message.get("kind") != "ready":
                    if isinstance(message, Mapping) and message.get("kind") == "failed":
                        raise NFAblationError(
                            "NF worker preflight failed: "
                            f"{message.get('exception_type')}: {message.get('message')}"
                        )
                    raise NFAblationError(
                        "NF worker emitted an invalid preflight report"
                    )
                slot = message.get("worker_slot")
                if (
                    type(slot) is not int
                    or not 0 <= slot < self.worker_count
                    or slot in seen
                ):
                    raise NFAblationError(
                        "NF worker preflight repeated an invalid slot"
                    )
                seen.add(slot)
                self.worker_records.append(dict(message))
        except BaseException:
            self.abort()
            raise
        self.worker_records.sort(key=lambda row: int(row["worker_slot"]))
        return tuple(self.worker_records)

    def assert_healthy(self) -> None:
        failed = [process for process in self.processes if process.exitcode is not None]
        if failed:
            raise NFAblationError(
                "NF worker pool exited unexpectedly: "
                + ", ".join(f"{process.name}={process.exitcode}" for process in failed)
            )

    def close(self) -> None:
        if self._closed:
            return
        for _ in self.processes:
            self.task_queue.put(None)
        for process in self.processes:
            process.join(timeout=30)
        if any(process.exitcode != 0 for process in self.processes):
            self.abort()
            raise NFAblationError("one or more NF workers failed during shutdown")
        self._closed = True

    def abort(self) -> None:
        if self._closed:
            return
        self.stop_event.set()
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        self._closed = True


def _stage_ledger_path(prepared: _PreparedNFAblation, stage: str) -> Path:
    return Path(prepared.campaign_root) / "state" / stage / "ledger.json"


def _stage_expected_sha(tasks: Sequence[NFCellTask]) -> str:
    return _canonical_sha([_task_descriptor(task) for task in tasks])


def _stage_completed_record(
    prepared: _PreparedNFAblation,
    *,
    task: NFCellTask,
    ordinal: int,
    result: Mapping[str, Any],
    assignment: Mapping[str, Any],
) -> dict[str, Any]:
    directory = Path(str(result["directory"])).resolve()
    root = Path(prepared.campaign_root).resolve()
    try:
        relative = directory.relative_to(root).as_posix()
    except ValueError as exc:
        raise NFAblationError("sealed NF cell escapes the ablation root") from exc
    return {
        **_task_descriptor(task),
        "ordinal": ordinal,
        "path": relative,
        "manifest_sha256": _sha256_path(directory / "manifest.json"),
        "summary_sha256": _sha256_path(directory / "summary.json"),
        "worker_slot": assignment.get("worker_slot"),
        "visible_device": assignment.get("visible_device"),
        "dispatch_sequence": assignment.get("dispatch_sequence"),
        "status": assignment.get("status", "reconstructed"),
    }


def _write_stage_ledger(
    prepared: _PreparedNFAblation,
    *,
    stage: str,
    tasks: Sequence[NFCellTask],
    results: Mapping[str, Mapping[str, Any]],
    assignments: Mapping[str, Mapping[str, Any]],
) -> None:
    rows: list[dict[str, Any]] = []
    for ordinal, task in enumerate(tasks):
        task_id = _task_id(task)
        if task_id not in results:
            continue
        rows.append(
            _stage_completed_record(
                prepared,
                task=task,
                ordinal=ordinal,
                result=results[task_id],
                assignment=assignments.get(task_id, {"status": "reconstructed"}),
            )
        )
    _write_json(
        _stage_ledger_path(prepared, stage),
        {
            "schema_version": 1,
            "campaign_identity": prepared.campaign_identity,
            "stage": stage,
            "expected_task_count": len(tasks),
            "expected_tasks_sha256": _stage_expected_sha(tasks),
            "completed_tasks": rows,
            "complete": len(rows) == len(tasks),
        },
    )


def _validate_stage_ledger(
    prepared: _PreparedNFAblation,
    *,
    stage: str,
    tasks: Sequence[NFCellTask],
) -> dict[str, dict[str, Any]]:
    path = _stage_ledger_path(prepared, stage)
    if not path.exists():
        return {}
    try:
        ledger = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NFAblationError(f"stage ledger is unreadable: {path}") from exc
    expected_fields = {
        "schema_version",
        "campaign_identity",
        "stage",
        "expected_task_count",
        "expected_tasks_sha256",
        "completed_tasks",
        "complete",
    }
    completed = ledger.get("completed_tasks")
    if (
        set(ledger) != expected_fields
        or ledger.get("schema_version") != 1
        or ledger.get("campaign_identity") != prepared.campaign_identity
        or ledger.get("stage") != stage
        or ledger.get("expected_task_count") != len(tasks)
        or ledger.get("expected_tasks_sha256") != _stage_expected_sha(tasks)
        or not isinstance(completed, list)
        or ledger.get("complete") not in {True, False}
    ):
        raise NFAblationError(f"stage ledger differs from its contract: {path}")
    assert isinstance(completed, list)
    task_by_id = {_task_id(task): (ordinal, task) for ordinal, task in enumerate(tasks)}
    if len(task_by_id) != len(tasks):
        raise NFAblationError(f"{stage} task matrix contains duplicate identities")
    row_fields = {
        "task_id",
        "candidate_id",
        "seed",
        "cell_key",
        "evaluate_test",
        "test_readout",
        "ordinal",
        "path",
        "manifest_sha256",
        "summary_sha256",
        "worker_slot",
        "visible_device",
        "dispatch_sequence",
        "status",
    }
    assignments: dict[str, dict[str, Any]] = {}
    seen_ordinals: set[int] = set()
    root = Path(prepared.campaign_root)
    for row in completed:
        if not isinstance(row, Mapping) or set(row) != row_fields:
            raise NFAblationError(f"stage ledger row is malformed: {path}")
        task_id = str(row.get("task_id"))
        if task_id not in task_by_id:
            raise NFAblationError(f"stage ledger contains an unknown task: {path}")
        ordinal, task = task_by_id[task_id]
        expected_descriptor = _task_descriptor(task)
        if any(row.get(key) != value for key, value in expected_descriptor.items()):
            raise NFAblationError(f"stage ledger task descriptor differs: {path}")
        if row.get("ordinal") != ordinal or ordinal in seen_ordinals:
            raise NFAblationError(f"stage ledger task ordinal differs: {path}")
        seen_ordinals.add(ordinal)
        model = _candidate_model(candidate_by_id(task.candidate_id), seed=task.seed)
        final_dir, _work_dir = _task_paths(task, _task_identity(task, model))
        expected_relative = final_dir.relative_to(root).as_posix()
        directory = root / str(row.get("path"))
        if (
            row.get("path") != expected_relative
            or not directory.is_dir()
            or _sha256_path(directory / "manifest.json") != row.get("manifest_sha256")
            or _sha256_path(directory / "summary.json") != row.get("summary_sha256")
            or row.get("status") not in {"completed", "reconstructed"}
        ):
            raise NFAblationError(f"stage ledger artifact differs: {path}")
        worker_slot = row.get("worker_slot")
        visible_device = row.get("visible_device")
        dispatch_sequence = row.get("dispatch_sequence")
        if row["status"] == "completed":
            if (
                type(worker_slot) is not int
                or not 0 <= worker_slot < prepared.worker_count
                or not isinstance(visible_device, str)
                or not visible_device
                or type(dispatch_sequence) is not int
                or dispatch_sequence <= 0
            ):
                raise NFAblationError(f"stage ledger assignment is invalid: {path}")
        elif (
            worker_slot is not None
            or visible_device != "reconstructed-sealed-cell"
            or dispatch_sequence is not None
        ):
            raise NFAblationError(f"reconstructed stage assignment is invalid: {path}")
        errors = _validate_cell(final_dir, _task_identity(task, model))
        if errors:
            raise NFAblationError(f"stage ledger points to an invalid cell: {errors}")
        assignments[task_id] = {
            "worker_slot": row.get("worker_slot"),
            "visible_device": row.get("visible_device"),
            "dispatch_sequence": row.get("dispatch_sequence"),
            "status": row.get("status"),
        }
    if [int(row["ordinal"]) for row in completed] != sorted(seen_ordinals):
        raise NFAblationError(f"stage ledger rows are out of order: {path}")
    if bool(ledger["complete"]) != (len(completed) == len(tasks)):
        raise NFAblationError(f"stage ledger completion flag differs: {path}")
    return assignments


def _task_dependency_ids(tasks: Sequence[NFCellTask]) -> dict[str, str | None]:
    by_key = {
        (task.candidate_id, task.seed, task.cell.key): _task_id(task) for task in tasks
    }
    dependencies: dict[str, str | None] = {}
    for task in tasks:
        task_id = _task_id(task)
        cell = task.cell
        if cell.reference_dataset in {None, cell.dataset}:
            dependencies[task_id] = None
            continue
        reference_key = "/".join(
            (str(cell.suite_id), str(cell.reference_dataset), str(cell.representation))
        )
        dependency = by_key.get((task.candidate_id, task.seed, reference_key))
        if dependency is None:
            raise NFAblationError(
                f"{cell.key} requires absent stage dependency {reference_key}"
            )
        dependencies[task_id] = dependency
    return dependencies


def _occupancy_record(
    path: Path,
    *,
    stage: str,
    event: str,
    worker_count: int,
    ready_count: int,
    in_flight_count: int,
    completed_count: int,
    total_count: int,
    task_id: str | None = None,
    worker_slot: int | None = None,
) -> None:
    _append_jsonl(
        path,
        {
            "schema_version": 1,
            "recorded_at_utc": datetime.now(UTC).isoformat(),
            "stage": stage,
            "event": event,
            "task_id": task_id,
            "worker_slot": worker_slot,
            "worker_count": worker_count,
            "ready_count": ready_count,
            "in_flight_count": in_flight_count,
            "busy_fraction": in_flight_count / worker_count,
            "completed_count": completed_count,
            "total_count": total_count,
        },
    )


def _run_task_stage(
    prepared: _PreparedNFAblation,
    *,
    stage: str,
    tasks: Sequence[NFCellTask],
    pool: _PersistentNFPool,
    monitor: _NvidiaSmiMonitor,
) -> list[dict[str, Any]]:
    if not tasks:
        raise NFAblationError(f"{stage} has no tasks")
    task_ids = [_task_id(task) for task in tasks]
    if len(set(task_ids)) != len(tasks):
        raise NFAblationError(f"{stage} repeats a task identity")
    task_by_id = dict(zip(task_ids, tasks, strict=True))
    ordinal_by_id = {task_id: index for index, task_id in enumerate(task_ids)}
    dependency_by_id = _task_dependency_ids(tasks)
    assignments = _validate_stage_ledger(prepared, stage=stage, tasks=tasks)
    results: dict[str, dict[str, Any]] = {}
    for task_id, task in task_by_id.items():
        result = _existing_task_result(task)
        if result is None:
            continue
        results[task_id] = result
        assignments.setdefault(
            task_id,
            {
                "worker_slot": None,
                "visible_device": "reconstructed-sealed-cell",
                "dispatch_sequence": None,
                "status": "reconstructed",
            },
        )
    _write_stage_ledger(
        prepared,
        stage=stage,
        tasks=tasks,
        results=results,
        assignments=assignments,
    )
    pending = set(task_ids) - set(results)
    in_flight: dict[str, NFCellTask] = {}
    active_by_slot: dict[int, str] = {}
    dispatch_sequence = max(
        (
            int(row["dispatch_sequence"])
            for row in assignments.values()
            if row.get("dispatch_sequence") is not None
        ),
        default=0,
    )
    occupancy_path = (
        Path(prepared.campaign_root) / "telemetry" / "worker_occupancy.jsonl"
    )

    def ready_ids() -> list[str]:
        return sorted(
            (
                task_id
                for task_id in pending
                if task_id not in in_flight
                and (
                    dependency_by_id[task_id] is None
                    or dependency_by_id[task_id] in results
                )
            ),
            key=ordinal_by_id.__getitem__,
        )

    def fill_workers() -> None:
        nonlocal dispatch_sequence
        ready = ready_ids()
        while ready and len(in_flight) < pool.worker_count:
            task_id = ready.pop(0)
            task = task_by_id[task_id]
            dispatch_sequence += 1
            in_flight[task_id] = task
            assignments[task_id] = {
                "worker_slot": None,
                "visible_device": None,
                "dispatch_sequence": dispatch_sequence,
                "status": "completed",
            }
            pool.task_queue.put({"stage": stage, "task_id": task_id, "task": task})
            _occupancy_record(
                occupancy_path,
                stage=stage,
                event="dispatched",
                worker_count=pool.worker_count,
                ready_count=len(ready),
                in_flight_count=len(in_flight),
                completed_count=len(results),
                total_count=len(tasks),
                task_id=task_id,
            )
        if ready and len(in_flight) < pool.worker_count:
            raise NFAblationError(f"{stage} scheduler left a worker idle")

    _occupancy_record(
        occupancy_path,
        stage=stage,
        event="stage_started",
        worker_count=pool.worker_count,
        ready_count=len(ready_ids()),
        in_flight_count=0,
        completed_count=len(results),
        total_count=len(tasks),
    )
    fill_workers()
    if pending and not in_flight:
        raise NFAblationError(f"{stage} dependency graph has no ready task")
    while pending:
        monitor.raise_if_failed()
        try:
            message = pool.result_queue.get(timeout=1.0)
        except queue.Empty:
            pool.assert_healthy()
            continue
        if not isinstance(message, Mapping):
            raise NFAblationError("NF worker emitted a non-mapping message")
        kind = message.get("kind")
        if kind == "failed":
            raise NFAblationError(
                "NF worker failed fast "
                f"(slot={message.get('worker_slot')}, active={message.get('active')}, "
                f"type={message.get('exception_type')}): {message.get('message')}\n"
                f"{message.get('traceback')}"
            )
        message_stage = message.get("stage")
        task_id = str(message.get("task_id"))
        if message_stage != stage or task_id not in in_flight:
            raise NFAblationError("NF worker message does not match an in-flight task")
        slot = message.get("worker_slot")
        if type(slot) is not int or not 0 <= slot < pool.worker_count:
            raise NFAblationError("NF worker reported an invalid slot")
        if kind == "started":
            if slot in active_by_slot:
                raise NFAblationError("NF worker started two concurrent tasks")
            active_by_slot[slot] = task_id
            assignments[task_id]["worker_slot"] = slot
            assignments[task_id]["visible_device"] = str(message["visible_device"])
            _occupancy_record(
                occupancy_path,
                stage=stage,
                event="worker_started",
                worker_count=pool.worker_count,
                ready_count=len(ready_ids()),
                in_flight_count=len(in_flight),
                completed_count=len(results),
                total_count=len(tasks),
                task_id=task_id,
                worker_slot=slot,
            )
            continue
        if kind != "completed" or not isinstance(message.get("payload"), Mapping):
            raise NFAblationError("NF worker emitted an invalid completion")
        if active_by_slot.pop(slot, None) != task_id:
            raise NFAblationError("NF worker completion differs from its start")
        payload = dict(message["payload"])
        task = in_flight.pop(task_id)
        if (
            payload.get("candidate_id") != task.candidate_id
            or payload.get("seed") != task.seed
            or payload.get("cell_key") != task.cell.key
        ):
            raise NFAblationError("NF worker result differs from its task")
        results[task_id] = payload
        pending.remove(task_id)
        assignments[task_id].update(
            {
                "worker_slot": slot,
                "visible_device": str(message["visible_device"]),
                "status": "completed",
            }
        )
        _write_stage_ledger(
            prepared,
            stage=stage,
            tasks=tasks,
            results=results,
            assignments=assignments,
        )
        _occupancy_record(
            occupancy_path,
            stage=stage,
            event="completed",
            worker_count=pool.worker_count,
            ready_count=len(ready_ids()),
            in_flight_count=len(in_flight),
            completed_count=len(results),
            total_count=len(tasks),
            task_id=task_id,
            worker_slot=slot,
        )
        fill_workers()
        if pending and not in_flight:
            raise NFAblationError(f"{stage} dependency graph became blocked")
    if active_by_slot or in_flight or set(results) != set(task_ids):
        raise NFAblationError(f"{stage} coordinator state is incomplete")
    monitor.raise_if_failed()
    _write_stage_ledger(
        prepared,
        stage=stage,
        tasks=tasks,
        results=results,
        assignments=assignments,
    )
    _occupancy_record(
        occupancy_path,
        stage=stage,
        event="stage_completed",
        worker_count=pool.worker_count,
        ready_count=0,
        in_flight_count=0,
        completed_count=len(results),
        total_count=len(tasks),
    )
    return [results[task_id] for task_id in task_ids]


def _stage1_tasks(prepared: _PreparedNFAblation) -> tuple[NFCellTask, ...]:
    cell_by_key = {cell.key: cell for cell in prepared.preflight.cells}
    if not set(STAGE1_SENTINEL_KEYS) <= set(cell_by_key):
        raise NFAblationError("stage-1 sentinel cells are absent from the inventory")
    return tuple(
        _make_cell_task(
            prepared,
            cell=cell_by_key[cell_key],
            candidate_id=candidate.candidate_id,
            seed=0,
            evaluate_test=False,
        )
        for candidate in STAGE1_CANDIDATES
        for cell_key in STAGE1_SENTINEL_KEYS
    )


def _known_lid_cells(cells: Sequence[Any]) -> tuple[Any, ...]:
    values = tuple(cell for cell in cells if cell.target_policy == "known_lid")
    if len(values) != 19:
        raise NFAblationError(
            f"stage 2 expected 19 known-LID cells, found {len(values)}"
        )
    return values


def _stage2_tasks(
    prepared: _PreparedNFAblation,
    promotions: Sequence[Promotion],
) -> tuple[NFCellTask, ...]:
    promoted_ids = tuple(value.candidate_id for value in promotions)
    if not 1 <= len(set(promoted_ids)) <= 2 or "C0" in promoted_ids:
        raise NFAblationError("stage 2 received an invalid promotion set")
    candidate_ids = ("C0", *promoted_ids)
    return tuple(
        _make_cell_task(
            prepared,
            cell=cell,
            candidate_id=candidate_id,
            seed=seed,
            evaluate_test=False,
        )
        for candidate_id in candidate_ids
        for seed in (0, 1)
        for cell in _known_lid_cells(prepared.preflight.cells)
    )


def _stage3_tasks(
    prepared: _PreparedNFAblation,
    winner: Winner,
) -> tuple[NFCellTask, ...]:
    pairs = (("C0", 2), (winner.candidate_id, 2), (winner.candidate_id, 3))
    if winner.candidate_id == "C0":
        raise NFAblationError("stage-3 winner must be a promoted non-control candidate")
    tasks = tuple(
        _make_cell_task(
            prepared,
            cell=cell,
            candidate_id=candidate_id,
            seed=seed,
            evaluate_test=True,
            test_readout=("autograd" if candidate_id == "C0" else winner.readout),
        )
        for candidate_id, seed in pairs
        for cell in prepared.preflight.cells
    )
    if len(tasks) != 3 * 39:
        raise NFAblationError("stage 3 must contain exactly 3 x 39 cells")
    return tasks


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    header = list(campaign._UNIFIED_TABLE_FIELDS)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=header,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for raw in rows:
        if set(raw) != set(header):
            raise NFAblationError("NF extension row differs from unified schema")
        writer.writerow(
            {field: "" if raw[field] is None else raw[field] for field in header}
        )
    campaign._write_text(path, stream.getvalue())


def validate_nf_ablation_campaign(campaign_root: str | Path) -> list[str]:
    """Validate final tables, provenance, stage ledgers and baseline immutability."""

    root = Path(campaign_root).expanduser().resolve()
    manifest_path = root / "campaign.json"
    identity_path = root / "ablation_identity.json"
    errors: list[str] = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        identity_record = json.loads(identity_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"final NF ablation metadata are unreadable: {exc}"]
    if manifest.get("complete") is not True:
        errors.append("final manifest is not complete")
    if manifest.get("campaign_identity") != _canonical_sha(identity_record):
        errors.append("final manifest identity differs")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list):
        errors.append("final manifest outputs are missing")
    else:
        for row in outputs:
            try:
                path = root / str(row["path"])
                if (
                    not path.is_file()
                    or path.stat().st_size != int(row["size"])
                    or _sha256_path(path) != row["sha256"]
                ):
                    errors.append(f"final output differs: {row.get('path')}")
            except (KeyError, OSError, TypeError, ValueError):
                errors.append("final output manifest row is malformed")
    try:
        baseline = Path(str(manifest["baseline"]["unified_results_path"]))
        extension = root / str(manifest["extension"]["path"])
        combined = root / str(manifest["combined"]["path"])
        provenance_path = root / str(manifest["provenance"]["path"])
        baseline_bytes = baseline.read_bytes()
        combined_bytes = combined.read_bytes()
        if hashlib.sha256(baseline_bytes).hexdigest() != manifest["baseline"]["sha256"]:
            errors.append("baseline unified CSV changed")
        if combined_bytes[: len(baseline_bytes)] != baseline_bytes:
            errors.append("combined CSV does not preserve the baseline byte prefix")
        extension_header, extension_rows = _read_csv(extension)
        combined_header, combined_rows = _read_csv(combined)
        if tuple(extension_header) != tuple(campaign._UNIFIED_TABLE_FIELDS):
            errors.append("extension CSV schema differs")
        if tuple(combined_header) != tuple(campaign._UNIFIED_TABLE_FIELDS):
            errors.append("combined CSV schema differs")
        if len(extension_rows) != 3 * 39 * 2:
            errors.append("extension CSV does not contain exact 3 x 39 x 2 rows")
        if len(combined_rows) != manifest["baseline"]["rows"] + len(extension_rows):
            errors.append("combined CSV row count differs")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("campaign_identity") != manifest.get(
            "campaign_identity"
        ) or len(provenance.get("records", ())) != len(extension_rows):
            errors.append("extension provenance coverage differs")
    except (KeyError, OSError, UnicodeError, json.JSONDecodeError, TypeError) as exc:
        errors.append(f"final table validation failed: {exc}")
    for stage in ("stage1", "stage2", "stage3"):
        try:
            row = manifest["stages"][stage]
            ledger = root / str(row["ledger_path"])
            payload = json.loads(ledger.read_text(encoding="utf-8"))
            if (
                payload.get("complete") is not True
                or _sha256_path(ledger) != row["ledger_sha256"]
                or len(payload.get("completed_tasks", ()))
                != int(row["completed_tasks"])
            ):
                errors.append(f"{stage} ledger is incomplete or changed")
            for cell_record in payload.get("completed_tasks", ()):
                cell_directory = root / str(cell_record["path"])
                if (
                    not cell_directory.is_dir()
                    or _sha256_path(cell_directory / "manifest.json")
                    != cell_record["manifest_sha256"]
                    or _sha256_path(cell_directory / "summary.json")
                    != cell_record["summary_sha256"]
                ):
                    errors.append(
                        f"{stage} sealed cell differs: {cell_record.get('cell_key')}"
                    )
        except (KeyError, OSError, UnicodeError, json.JSONDecodeError, TypeError):
            errors.append(f"{stage} ledger is unreadable")
    if list(root.rglob("checkpoint.pt")) or list(root.rglob("training_progress.pt")):
        errors.append("completed NF ablation retained a checkpoint")
    return errors


def finalize_nf_ablation_campaign(
    campaign_root: str | Path,
    preflight: BaselinePreflight,
    stage3_results: Sequence[Mapping[str, Any]],
    winner: Winner,
    promotions: Sequence[Promotion],
) -> Path:
    """Write the extension, byte-preserving combined CSV and provenance seal."""

    root = Path(campaign_root).expanduser().resolve()
    identity_path = root / "ablation_identity.json"
    try:
        identity_record = json.loads(identity_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NFAblationError(
            "ablation identity is unreadable during finalization"
        ) from exc
    campaign_identity = _canonical_sha(identity_record)
    extension_rows, provenance_records = build_stage3_unified_rows(
        stage3_results,
        preflight.cells,
        winner=winner,
    )
    if len(extension_rows) != 3 * 39 * 2 or len(provenance_records) != len(
        extension_rows
    ):
        raise NFAblationError("final NF row/provenance matrix is incomplete")
    extension_path = root / "nf_ablation_results.csv"
    combined_path = root / "unified_results_with_nf_ablation.csv"
    provenance_path = root / "nf_ablation_results.provenance.json"
    _write_csv(extension_path, extension_rows)
    merge_record = merge_unified_results(
        baseline_csv=preflight.baseline_root / "unified_results.csv",
        extension_rows=extension_rows,
        output_csv=combined_path,
    )
    _write_json(
        provenance_path,
        {
            "schema_version": 1,
            "campaign_identity": campaign_identity,
            "baseline_campaign_identity": preflight.baseline_campaign_identity,
            "baseline_unified_sha256": preflight.baseline_unified_sha256,
            "run_id_contract": {
                "run_id": "full_sha256_of_cell_identity_v1",
                "cell_id": "first_20_hex_characters_of_run_id",
            },
            "winner": _plain(winner),
            "promotions": [_plain(value) for value in promotions],
            "records": provenance_records,
        },
    )
    stage_records: dict[str, Any] = {}
    for stage in ("stage1", "stage2", "stage3"):
        ledger = root / "state" / stage / "ledger.json"
        payload = json.loads(ledger.read_text(encoding="utf-8"))
        if payload.get("complete") is not True:
            raise NFAblationError(f"cannot finalize with incomplete {stage}")
        stage_records[stage] = {
            "ledger_path": ledger.relative_to(root).as_posix(),
            "ledger_sha256": _sha256_path(ledger),
            "completed_tasks": len(payload["completed_tasks"]),
        }
    manifest_path = root / "campaign.json"
    manifest = {
        "schema_version": NF_ABLATION_SCHEMA_VERSION,
        "ablation_id": NF_ABLATION_ID,
        "campaign_identity": campaign_identity,
        "baseline": {
            "campaign_root": str(preflight.baseline_root),
            "campaign_identity": preflight.baseline_campaign_identity,
            "unified_results_path": str(
                preflight.baseline_root / "unified_results.csv"
            ),
            "sha256": preflight.baseline_unified_sha256,
            "rows": preflight.baseline_row_count,
            "preservation": "immutable_byte_prefix",
        },
        "extension": {
            "path": extension_path.relative_to(root).as_posix(),
            "sha256": _sha256_path(extension_path),
            "rows": len(extension_rows),
            "models": 3,
            "cells_per_model": 39,
            "splits_per_cell": 2,
        },
        "combined": {
            "path": combined_path.relative_to(root).as_posix(),
            **merge_record,
        },
        "provenance": {
            "path": provenance_path.relative_to(root).as_posix(),
            "sha256": _sha256_path(provenance_path),
            "records": len(provenance_records),
        },
        "stages": stage_records,
        "promotions": [_plain(value) for value in promotions],
        "winner": _plain(winner),
        "telemetry": {
            "nvidia_smi_jsonl": "telemetry/nvidia_smi.jsonl",
            "worker_occupancy_jsonl": "telemetry/worker_occupancy.jsonl",
        },
        "created_at_utc": datetime.now(UTC).isoformat(),
        "complete": True,
    }
    output_relatives = {
        "ablation_identity.json",
        "baseline_provenance.json",
        "resolved_candidates.yaml",
        extension_path.relative_to(root).as_posix(),
        combined_path.relative_to(root).as_posix(),
        provenance_path.relative_to(root).as_posix(),
        "state/stage1/ledger.json",
        "state/stage1/validation_scores.json",
        "state/stage1/promotions.json",
        "state/stage2/ledger.json",
        "state/stage2/validation_scores.json",
        "state/stage2/winner.json",
        "state/stage3/ledger.json",
        "telemetry/nvidia_smi.jsonl",
        "telemetry/worker_occupancy.jsonl",
        "preflight.json",
    }
    missing = [
        relative
        for relative in sorted(output_relatives)
        if not (root / relative).is_file()
    ]
    if missing:
        raise NFAblationError(f"final NF artifacts are missing: {missing}")
    manifest["outputs"] = [
        {
            "path": relative,
            "size": (root / relative).stat().st_size,
            "sha256": _sha256_path(root / relative),
        }
        for relative in sorted(output_relatives)
    ]
    _write_json(manifest_path, manifest)
    errors = validate_nf_ablation_campaign(root)
    if errors:
        manifest_path.unlink(missing_ok=True)
        raise NFAblationError(f"new NF ablation failed final validation: {errors}")
    return root


def run_nf_ablation_campaign(
    *,
    baseline_root: str | Path,
    output_root: str | Path,
    project_root: str | Path | None = None,
    worker_count: int = WORKER_COUNT,
    dependencies: NFDependencies | None = None,
    require_cuda: bool = True,
    preflight_only: bool = False,
) -> Path:
    """Execute the full three-stage NF campaign in one persistent worker pool."""

    if type(require_cuda) is not bool or type(preflight_only) is not bool:
        raise NFAblationError("require_cuda and preflight_only must be booleans")
    if require_cuda and worker_count != WORKER_COUNT:
        raise NFAblationError(
            f"production NF ablation requires exactly {WORKER_COUNT} GPU workers"
        )
    _configure_deterministic_cublas(require_cuda=require_cuda)
    checkout = (
        campaign.repository_root()
        if project_root is None
        else Path(project_root).expanduser().resolve()
    )
    if not checkout.is_dir():
        raise NFAblationError(f"project root is not a directory: {checkout}")
    dependencies = NFDependencies() if dependencies is None else dependencies
    preflight = preflight_baseline(
        baseline_root,
        project_root=checkout,
        dependencies=dependencies,
    )
    prepared = _prepare_nf_ablation(
        preflight,
        project_root=checkout,
        output_root=Path(output_root),
        worker_count=worker_count,
    )
    root = Path(prepared.campaign_root)
    root.mkdir(parents=True, exist_ok=True)
    final_path = root / "campaign.json"
    if final_path.is_file():
        errors = validate_nf_ablation_campaign(root)
        if errors:
            raise NFAblationError(f"existing final NF ablation is invalid: {errors}")
        return root
    lock_path = root / "state" / "campaign.lock"
    with campaign._exclusive_campaign_lock(lock_path):
        if final_path.is_file():
            errors = validate_nf_ablation_campaign(root)
            if errors:
                raise NFAblationError(
                    f"existing final NF ablation is invalid: {errors}"
                )
            return root
        identity_path = root / "ablation_identity.json"
        if identity_path.exists():
            try:
                existing_identity = json.loads(
                    identity_path.read_text(encoding="utf-8")
                )
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise NFAblationError(
                    "existing ablation identity is unreadable"
                ) from exc
            if campaign.canonical_json(existing_identity) != campaign.canonical_json(
                prepared.identity_record
            ):
                raise NFAblationError("stable ablation root has a different identity")
        else:
            _write_json(identity_path, prepared.identity_record)
        _write_json(
            root / "baseline_provenance.json",
            {
                "schema_version": 1,
                "baseline_root": str(preflight.baseline_root),
                "baseline_campaign_identity": preflight.baseline_campaign_identity,
                "baseline_unified_sha256": preflight.baseline_unified_sha256,
                "baseline_rows": preflight.baseline_row_count,
                "input_sha256_by_key": dict(preflight.input_sha256_by_key),
                "source_records": dict(preflight.source_records),
            },
        )
        campaign._write_yaml(
            root / "resolved_candidates.yaml",
            {
                "schema_version": NF_ABLATION_SCHEMA_VERSION,
                "ablation_id": NF_ABLATION_ID,
                "worker_count": worker_count,
                "partition_seed": PARTITION_SEED,
                "candidates": [_plain(value) for value in STAGE1_CANDIDATES],
                "stage1_sentinels": list(STAGE1_SENTINEL_KEYS),
                "readouts": list(READOUTS),
                "selection_scales": list(SELECTION_SCALES),
            },
        )
        monitor = _NvidiaSmiMonitor(
            root / "telemetry" / "nvidia_smi.jsonl",
            enabled=require_cuda,
            expected_gpu_count=worker_count,
        )
        first_gpu_sample = monitor.start()
        pool = _PersistentNFPool(
            worker_count=worker_count,
            dependencies=dependencies,
            require_cuda=require_cuda,
        )
        stage3_results: list[dict[str, Any]] | None = None
        promotions: tuple[Promotion, ...] | None = None
        winner: Winner | None = None
        try:
            worker_records = pool.start()
            _write_json(
                root / "preflight.json",
                {
                    "schema_version": 1,
                    "campaign_identity": prepared.campaign_identity,
                    "baseline_campaign_identity": preflight.baseline_campaign_identity,
                    "baseline_unified_sha256": preflight.baseline_unified_sha256,
                    "worker_count": worker_count,
                    "require_cuda": require_cuda,
                    "workers": list(worker_records),
                    "first_gpu_sample": first_gpu_sample,
                    "planned_stage1_tasks": len(STAGE1_CANDIDATES)
                    * len(STAGE1_SENTINEL_KEYS),
                    "planned_stage2_known_cells": len(
                        _known_lid_cells(preflight.cells)
                    ),
                    "planned_stage3_tasks": 3 * len(preflight.cells),
                    "status": "ready",
                },
            )
            if not preflight_only:
                stage1_results = _run_task_stage(
                    prepared,
                    stage="stage1",
                    tasks=_stage1_tasks(prepared),
                    pool=pool,
                    monitor=monitor,
                )
                stage1_scores = validation_score_records(
                    stage1_results, preflight.cells
                )
                _write_json(
                    root / "state" / "stage1" / "validation_scores.json",
                    {
                        "schema_version": 1,
                        "campaign_identity": prepared.campaign_identity,
                        "records": stage1_scores,
                    },
                )
                promotions = rank_stage1(stage1_scores)
                _write_json(
                    root / "state" / "stage1" / "promotions.json",
                    {
                        "schema_version": 1,
                        "campaign_identity": prepared.campaign_identity,
                        "promotions": [_plain(value) for value in promotions],
                    },
                )
                stage2_results = _run_task_stage(
                    prepared,
                    stage="stage2",
                    tasks=_stage2_tasks(prepared, promotions),
                    pool=pool,
                    monitor=monitor,
                )
                stage2_scores = validation_score_records(
                    stage2_results, preflight.cells
                )
                _write_json(
                    root / "state" / "stage2" / "validation_scores.json",
                    {
                        "schema_version": 1,
                        "campaign_identity": prepared.campaign_identity,
                        "records": stage2_scores,
                    },
                )
                winner = pick_stage2_winner(stage2_scores, promotions)
                _write_json(
                    root / "state" / "stage2" / "winner.json",
                    {
                        "schema_version": 1,
                        "campaign_identity": prepared.campaign_identity,
                        "winner": _plain(winner),
                        "test_data_opened": False,
                    },
                )
                stage3_results = _run_task_stage(
                    prepared,
                    stage="stage3",
                    tasks=_stage3_tasks(prepared, winner),
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
        if stage3_results is None or promotions is None or winner is None:
            raise NFAblationError(
                "NF coordinator reached finalization without final state"
            )
        return finalize_nf_ablation_campaign(
            root,
            preflight,
            stage3_results,
            winner,
            promotions,
        )


def _parse_cli(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the staged eight-GPU normalizing-flow quality ablation."
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        required=True,
        help="sealed root of the completed 10 x 39 global campaign",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="parent directory for the deterministic NF ablation root",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="repository checkout; defaults to the current repository root",
    )
    parser.add_argument(
        "--worker-count",
        type=int,
        default=WORKER_COUNT,
        help=f"persistent one-device worker count (production requires {WORKER_COUNT})",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="verify baseline, source inputs, GPUs and spawn workers without training",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="disable CUDA checks for local integration tests only",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _parse_cli(argv)
    output = run_nf_ablation_campaign(
        baseline_root=arguments.baseline_root,
        output_root=arguments.output_root,
        project_root=arguments.project_root,
        worker_count=arguments.worker_count,
        require_cuda=not arguments.allow_cpu,
        preflight_only=arguments.preflight_only,
    )
    print(output)


if __name__ == "__main__":
    main()
