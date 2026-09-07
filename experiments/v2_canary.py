"""Fail-closed pre-full integrity and multiplexed eight-H100 utilization canary.

The canary trains 24 *production* cells as three independent lanes on each of
eight H100s.  Eight cells (four model variants on D30 ``e2_uniform_pca``
coefficients and D3072 ``e2_arrows`` images) carry scientific diagnostics.
Both low- and high-dimensional accuracy and trace stability are non-blocking
outcomes so the canary cannot censor a weak model before the
comparison is run.  Sixteen additional production cells provide representative
co-load and are sealed for reuse by the same 429-cell campaign.  The resulting
attestation is consumed by the v2 full campaign; this module never starts the
remaining campaign itself.  A failed or incomplete gate therefore stops the
shell chain before any remaining full-matrix cell is scheduled.

This file deliberately owns only the canary/report boundary.  Training and
global-campaign identities remain authoritative in their respective modules.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
import queue
import re
import shutil
import statistics
import subprocess
import threading
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

SCHEMA_VERSION = 5
PROTOCOL_ID = "vp-ve-fm-nf-integrity-canary-v5"
REPORT_FILENAME = "canary_report.json"
EXPECTED_BATCH_SIZE = 256
EXPECTED_WORKER_COUNT = 8
EXPECTED_LANES_PER_DEVICE = 3
EXPECTED_PROCESS_COUNT = EXPECTED_WORKER_COUNT * EXPECTED_LANES_PER_DEVICE
EXPECTED_TRAINING_STEPS = 128000
EXPECTED_SELECTION_QUERY_COUNTS = {"coefficients": 128, "dataset": 32}
EXPECTED_TRACE_QUERY_COUNT = 4
MAXIMUM_LOW_DIM_POINTWISE_LID_MAE = 5.0
MAXIMUM_LOW_DIM_TRACE64_MAE = 2.0
MAXIMUM_LOW_DIM_TRACE64_TO_TRACE16_RATIO = 1.25
EXPECTED_CELL_KEYS = (
    "e2/e2_uniform_pca/coefficients",
    "e2/e2_arrows/dataset",
)
EXPECTED_REPRESENTATIONS = ("coefficients", "dataset")
EXPECTED_VARIANTS = (
    "vp_diffusion",
    "ve_diffusion",
    "posterior_log_noise_affine_flow",
    "scale_conditioned_nf",
)
EXPECTED_CELL_IDS = tuple(
    f"{variant}/{cell_key}"
    for cell_key in EXPECTED_CELL_KEYS
    for variant in EXPECTED_VARIANTS
)
EXPECTED_COMPANION_CELL_KEYS = (
    "e3/e3_gaussian_pca/coefficients",
    "e1/e1_sampled_fmnist_step1/dataset",
    "e4/e4_sphere_pca_radius1/coefficients",
    "e2/e2_uniform_pca/dataset",
)
EXPECTED_COMPANION_CELL_IDS = tuple(
    f"{variant}/{cell_key}"
    for cell_key in EXPECTED_COMPANION_CELL_KEYS
    for variant in EXPECTED_VARIANTS
)
EXPECTED_ALL_CELL_IDS = EXPECTED_CELL_IDS + EXPECTED_COMPANION_CELL_IDS


def _physical_device_for_cell(cell_id: str) -> int:
    variant = cell_id.split("/", 1)[0]
    variant_index = EXPECTED_VARIANTS.index(variant)
    representation = cell_id.rsplit("/", 1)[-1]
    return variant_index if representation == "coefficients" else 4 + variant_index


EXPECTED_PROCESS_DEVICE_INDICES = tuple(
    _physical_device_for_cell(cell_id) for cell_id in EXPECTED_ALL_CELL_IDS
)
if EXPECTED_PROCESS_DEVICE_INDICES != tuple(
    process_index % EXPECTED_WORKER_COUNT
    for process_index in range(EXPECTED_PROCESS_COUNT)
):
    raise RuntimeError(
        "canary process-to-device schedule must be three full H100 lanes"
    )
REQUIRED_GATE_IDS = frozenset(
    {
        "source_provenance",
        "canonical_archive",
        "arrows_data_preprocessing",
        "eight_h100_workers",
        "batch_256_contract",
        "gpu_utilization",
        "runtime_projection",
        "native_loss_quality",
        "reconstruction_quality",
        "pointwise_lid_assessment",
        "reference_selector_quality",
        "nf_scale_bin_nll",
        "trace_estimator_assessment",
        "production_cell_reuse",
        "output_integrity",
    }
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_SHA = re.compile(r"[0-9a-f]{40}\Z")


class CanaryError(RuntimeError):
    """The canary contract or evidence is incomplete."""


def _plain(value: Any) -> Any:
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    return value


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    identity = {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "bytes_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }
    return canonical_sha256(identity)


def _exact_keys(value: Any, expected: set[str], *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CanaryError(f"{field} must be a mapping")
    if set(value) != expected:
        raise CanaryError(
            f"{field} fields differ: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )
    return value


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CanaryError(f"{field} must be a positive integer")
    return value


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CanaryError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CanaryError(f"{field} must be finite")
    return result


def _sha(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise CanaryError(f"{field} must be a lowercase SHA-256")
    return value


def _resolve(root: Path, configured: Any, *, field: str) -> Path:
    if not isinstance(configured, str) or not configured:
        raise CanaryError(f"{field} must be a non-empty path")
    path = Path(configured).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def validate_canary_config(config: Mapping[str, Any] | DictConfig) -> dict[str, Any]:
    """Validate the immutable canary design before touching data or CUDA."""

    raw = _plain(config)
    if isinstance(raw, Mapping) and "hydra" in raw:
        raw = {key: value for key, value in raw.items() if key != "hydra"}
    top = _exact_keys(
        raw,
        {
            "schema_version",
            "protocol_id",
            "output_root",
            "data",
            "execution",
            "training",
            "evaluation",
            "gates",
        },
        field="config",
    )
    if top["schema_version"] != SCHEMA_VERSION or top["protocol_id"] != PROTOCOL_ID:
        raise CanaryError("unsupported canary schema or protocol")
    if not isinstance(top["output_root"], str) or not top["output_root"]:
        raise CanaryError("output_root must be a non-empty path")

    data = _exact_keys(
        top["data"],
        {
            "archive",
            "archive_sha256",
            "benchmark_root",
            "registry",
            "reviewed_arrows_gate",
            "canary_cell_keys",
            "companion_cell_keys",
            "arrows_dataset",
            "visual_sample_seed",
            "selection",
        },
        field="data",
    )
    _sha(data["archive_sha256"], field="data.archive_sha256")
    if not isinstance(data["reviewed_arrows_gate"], str):
        raise CanaryError("data.reviewed_arrows_gate must be a path string")
    if tuple(data["canary_cell_keys"]) != EXPECTED_CELL_KEYS:
        raise CanaryError("canary must use the exact D30/D3072 production cells")
    if tuple(data["companion_cell_keys"]) != EXPECTED_COMPANION_CELL_KEYS:
        raise CanaryError("canary utilization companions differ from the contract")
    if data["arrows_dataset"] != "e2_arrows":
        raise CanaryError("Arrows gate must inspect canonical e2_arrows")
    if isinstance(data["visual_sample_seed"], bool) or not isinstance(
        data["visual_sample_seed"], int
    ):
        raise CanaryError("data.visual_sample_seed must be an integer")
    selection = _exact_keys(
        data["selection"],
        {
            "protocol",
            "fraction",
            "maximum_selection",
            "minimum_selection",
            "minimum_fit",
            "seed",
        },
        field="data.selection",
    )
    if selection["protocol"] != "splitmix64_rank_v1":
        raise CanaryError("canary must reuse splitmix64_rank_v1")
    fraction = _finite_number(selection["fraction"], field="data.selection.fraction")
    if not 0 < fraction < 1:
        raise CanaryError("selection fraction must lie in (0, 1)")
    for name in ("maximum_selection", "minimum_selection", "minimum_fit"):
        _positive_int(selection[name], field=f"data.selection.{name}")
    if isinstance(selection["seed"], bool) or not isinstance(selection["seed"], int):
        raise CanaryError("data.selection.seed must be an integer")
    if selection["seed"] != 0:
        raise CanaryError("production canary partition seed must be exactly zero")

    execution = _exact_keys(
        top["execution"],
        {
            "physical_device_count",
            "lanes_per_device",
            "logical_worker_count",
            "batch_size",
            "require_cuda",
            "accelerator_name_pattern",
            "telemetry_interval_seconds",
        },
        field="execution",
    )
    if execution["physical_device_count"] != EXPECTED_WORKER_COUNT:
        raise CanaryError("canary requires exactly eight physical H100 devices")
    if execution["lanes_per_device"] != EXPECTED_LANES_PER_DEVICE:
        raise CanaryError("canary requires exactly three lanes per H100")
    if execution["logical_worker_count"] != EXPECTED_PROCESS_COUNT:
        raise CanaryError("canary requires exactly 24 logical workers")
    if execution["batch_size"] != EXPECTED_BATCH_SIZE:
        raise CanaryError("canary scientific batch_size must remain exactly 256")
    if execution["require_cuda"] is not True:
        raise CanaryError("canary requires CUDA")
    if execution["accelerator_name_pattern"] != "H100":
        raise CanaryError("canary requires full H100 devices")
    interval = _finite_number(
        execution["telemetry_interval_seconds"],
        field="execution.telemetry_interval_seconds",
    )
    if not 0.1 <= interval <= 10:
        raise CanaryError("telemetry interval must lie in [0.1, 10] seconds")

    training = _exact_keys(
        top["training"],
        {
            "campaign_config",
            "mode",
            "steps",
            "vp_warmup_steps",
            "other_warmup_steps",
            "validation_interval_steps",
            "model_seed",
            "variant_ids",
            "field_hidden_sizes",
            "nf_hidden_dim_by_ambient",
        },
        field="training",
    )
    if training["mode"] != "fixed_steps_v1":
        raise CanaryError("canary must use fixed_steps_v1")
    if (
        not isinstance(training["campaign_config"], str)
        or not training["campaign_config"]
    ):
        raise CanaryError("training.campaign_config must name the v2 production config")
    steps = _positive_int(training["steps"], field="training.steps")
    if steps != EXPECTED_TRAINING_STEPS:
        raise CanaryError("canary must use the exact 128000-step production budget")
    vp_warmup = training["vp_warmup_steps"]
    if (
        isinstance(vp_warmup, bool)
        or not isinstance(vp_warmup, int)
        or not 0 <= vp_warmup < steps
    ):
        raise CanaryError("training.vp_warmup_steps must lie in [0, steps)")
    if training["other_warmup_steps"] != 0:
        raise CanaryError("non-VP native optimizers must use zero warmup")
    _positive_int(
        training["validation_interval_steps"],
        field="training.validation_interval_steps",
    )
    if isinstance(training["model_seed"], bool) or not isinstance(
        training["model_seed"], int
    ):
        raise CanaryError("training.model_seed must be an integer")
    if tuple(training["variant_ids"]) != EXPECTED_VARIANTS:
        raise CanaryError(
            "training.variant_ids must match the four production variants"
        )
    if tuple(training["field_hidden_sizes"]) != (1024, 512, 256, 256, 128, 128):
        raise CanaryError("canary must require the reduced shared bottleneck widths")
    nf_widths = _exact_keys(
        training["nf_hidden_dim_by_ambient"], {"30", "3072"}, field="NF widths"
    )
    if nf_widths != {"30": 544, "3072": 448}:
        raise CanaryError("NF canary widths must match the predeclared capacity table")

    evaluation = _exact_keys(
        top["evaluation"],
        {
            "lambda_grid",
            "selection_query_count",
            "trace_query_count",
            "trace_probes",
            "trace_seed",
            "reconstruction_lambda",
            "nf_scale_bins",
        },
        field="evaluation",
    )
    grid = tuple(
        _finite_number(item, field="evaluation.lambda_grid")
        for item in evaluation["lambda_grid"]
    )
    if (
        len(grid) < 3
        or any(item <= 0 for item in grid)
        or any(a >= b for a, b in pairwise(grid))
    ):
        raise CanaryError("lambda_grid must be strictly increasing and positive")
    selection_counts = _exact_keys(
        evaluation["selection_query_count"],
        set(EXPECTED_REPRESENTATIONS),
        field="evaluation.selection_query_count",
    )
    for name, count in selection_counts.items():
        _positive_int(count, field=f"evaluation.selection_query_count.{name}")
    if selection_counts != EXPECTED_SELECTION_QUERY_COUNTS:
        raise CanaryError("selection diagnostic sample sizes differ from the protocol")
    if (
        _positive_int(
            evaluation["trace_query_count"], field="evaluation.trace_query_count"
        )
        != EXPECTED_TRACE_QUERY_COUNT
    ):
        raise CanaryError("trace diagnostic sample size differs from the protocol")
    if tuple(evaluation["trace_probes"]) != (16, 64):
        raise CanaryError("trace gate must compare exact against 16 and 64 probes")
    if isinstance(evaluation["trace_seed"], bool) or not isinstance(
        evaluation["trace_seed"], int
    ):
        raise CanaryError("evaluation.trace_seed must be an integer")
    if (
        _finite_number(
            evaluation["reconstruction_lambda"],
            field="evaluation.reconstruction_lambda",
        )
        <= 0
    ):
        raise CanaryError("reconstruction lambda must be positive")
    bins = evaluation["nf_scale_bins"]
    if not isinstance(bins, Sequence) or len(bins) < 2:
        raise CanaryError("NF scale-bin gate requires at least two bins")
    previous_upper = None
    for index, pair in enumerate(bins):
        if not isinstance(pair, Sequence) or len(pair) != 2:
            raise CanaryError(f"NF scale bin {index} must contain two bounds")
        lower, upper = (_finite_number(item, field="NF scale bin") for item in pair)
        if not 0 < lower < upper or (
            previous_upper is not None and lower < previous_upper
        ):
            raise CanaryError("NF scale bins must be positive, ordered, and disjoint")
        previous_upper = upper

    gates = _exact_keys(
        top["gates"],
        {
            "minimum_mean_gpu_utilization_percent",
            "minimum_p50_gpu_utilization_percent",
            "maximum_memory_fraction",
            "maximum_projected_campaign_hours",
            "runtime_projection_safety_factor",
            "maximum_best_to_initial_loss_ratio",
            "maximum_tail_improvement_fraction",
            "maximum_reconstruction_ratio",
            "maximum_low_dim_pointwise_lid_mae",
            "minimum_nf_improved_bin_fraction",
            "maximum_low_dim_trace64_mae",
            "maximum_low_dim_trace64_to_trace16_ratio",
        },
        field="gates",
    )
    bounded = {
        "minimum_mean_gpu_utilization_percent": (0, 100),
        "minimum_p50_gpu_utilization_percent": (0, 100),
        "maximum_memory_fraction": (0, 1),
        "maximum_projected_campaign_hours": (0, math.inf),
        "runtime_projection_safety_factor": (0, math.inf),
        "maximum_best_to_initial_loss_ratio": (0, 1),
        "maximum_tail_improvement_fraction": (0, 1),
        "maximum_reconstruction_ratio": (0, 1),
        "maximum_low_dim_pointwise_lid_mae": (0, math.inf),
        "minimum_nf_improved_bin_fraction": (0, 1),
        "maximum_low_dim_trace64_mae": (0, math.inf),
        "maximum_low_dim_trace64_to_trace16_ratio": (0, math.inf),
    }
    for name, (lower, upper) in bounded.items():
        value = _finite_number(gates[name], field=f"gates.{name}")
        if not lower < value <= upper:
            raise CanaryError(f"gates.{name} lies outside ({lower}, {upper}]")
    if gates["runtime_projection_safety_factor"] != 2.0:
        raise CanaryError("runtime projection safety factor must be exactly 2.0")
    if (
        gates["maximum_low_dim_pointwise_lid_mae"] != MAXIMUM_LOW_DIM_POINTWISE_LID_MAE
        or gates["maximum_low_dim_trace64_mae"] != MAXIMUM_LOW_DIM_TRACE64_MAE
        or gates["maximum_low_dim_trace64_to_trace16_ratio"]
        != MAXIMUM_LOW_DIM_TRACE64_TO_TRACE16_RATIO
    ):
        raise CanaryError("exact low-dimensional quality thresholds differ")
    return dict(raw)


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        if name not in value:
            raise CanaryError(f"canary context is missing {name}")
        return value[name]
    if not hasattr(value, name):
        raise CanaryError(f"canary context is missing {name}")
    return getattr(value, name)


def _diagnostic_subset_seed(cell_id: str, base_seed: int) -> int:
    _, suite, dataset, representation = cell_id.split("/")
    cell_key = f"{suite}/{dataset}/{representation}"
    if cell_key not in EXPECTED_CELL_KEYS:
        raise CanaryError("diagnostic cell is outside the paired quality design")
    return base_seed + EXPECTED_CELL_KEYS.index(cell_key) * 1009


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float) and not math.isfinite(value):
        raise CanaryError("non-finite value cannot enter canary evidence")
    return value


def _deterministic_subset(count: int, subset_size: int, seed: int) -> np.ndarray:
    if count < 1 or subset_size < 1:
        raise CanaryError("canary subset sizes must be positive")
    size = min(count, subset_size)
    generator = np.random.default_rng(seed)
    return np.sort(generator.choice(count, size=size, replace=False)).astype(np.int64)


def _history_quality(trained: Any, gates: Mapping[str, Any]) -> dict[str, Any]:
    history = []
    for item in _field(trained, "history"):
        record = item.to_dict() if hasattr(item, "to_dict") else dict(item)
        history.append(record)
    if len(history) < 4:
        raise CanaryError("fixed-step canary requires at least four validations")
    validation = np.asarray(
        [float(item["validation_loss"]) for item in history], dtype=np.float64
    )
    if not np.isfinite(validation).all():
        raise CanaryError("canary validation history contains non-finite values")
    metadata = dict(_field(trained, "weights_metadata"))
    initial = metadata.get("initial")
    if not isinstance(initial, Mapping) or set(initial) != {
        "step",
        "examples_seen",
        "validation_loss",
    }:
        raise CanaryError("fixed-step checkpoint lacks exact step-0 held-out loss")
    if initial["step"] != 0 or initial["examples_seen"] != 0:
        raise CanaryError("initial held-out loss is not bound to step zero")
    initial_loss = _finite_number(
        initial["validation_loss"], field="weights_metadata.initial.validation_loss"
    )
    best = float(validation.min())
    final = float(validation[-1])
    best_ratio = best / max(abs(initial_loss), 1e-12)
    tail_size = max(3, math.ceil(len(validation) / 4))
    tail = validation[-tail_size:]
    tail_improvement = max(0.0, float((tail[0] - tail[-1]) / max(abs(tail[0]), 1e-12)))
    passed = best_ratio <= float(gates["maximum_best_to_initial_loss_ratio"])
    plateau_threshold = float(gates["maximum_tail_improvement_fraction"])
    return {
        "status": "diagnostic_only",
        "quality_gate_applied": False,
        "historical_threshold_met": passed,
        "initial_step": 0,
        "initial_validation_loss": initial_loss,
        "best_validation_loss": best,
        "final_validation_loss": final,
        "best_to_initial_ratio": best_ratio,
        "tail_count": tail_size,
        "tail_improvement_fraction": tail_improvement,
        "convergence_status": (
            "plateau_detected"
            if tail_improvement <= plateau_threshold
            else "still_improving_at_budget"
        ),
        "plateau_gate_applied": False,
        "plateau_diagnostic_threshold": plateau_threshold,
        "history_sha256": canonical_sha256(history),
    }


def _normalized_queries(trained: Any, query: Any):
    import torch

    values = torch.as_tensor(np.asarray(query), dtype=torch.float32).flatten(1)
    mean = torch.as_tensor(_field(trained, "normalization_mean"), dtype=torch.float32)
    scale = _finite_number(
        _field(trained, "normalization_scale"), field="normalization_scale"
    )
    if values.shape[1] != mean.numel() or scale <= 0:
        raise CanaryError("training preprocessing does not match canary query")
    model = _field(trained, "model")
    parameter = next(model.parameters())
    return ((values - mean.reshape(1, -1)) / scale).to(
        device=parameter.device, dtype=parameter.dtype
    )


def _reconstruction_quality(
    trained: Any,
    query: Any,
    *,
    variant_id: str,
    noise_ratio: float,
    seed: int,
    maximum_ratio: float,
    output_dir: Path,
) -> dict[str, Any]:
    import torch

    if variant_id == "scale_conditioned_nf":
        return {"status": "not_applicable"}
    clean = _normalized_queries(trained, query)
    generator = torch.Generator(device=clean.device).manual_seed(seed)
    noise = torch.randn(
        clean.shape, device=clean.device, dtype=clean.dtype, generator=generator
    )
    model = _field(trained, "model")
    model.eval()
    config = _field(trained, "config")
    with torch.no_grad():
        if variant_id == "vp_diffusion":
            from models.vp_baseline import VPSchedule, denoise

            schedule = VPSchedule(
                float(_field(config, "vp_beta_min")),
                float(_field(config, "vp_beta_max")),
            )
            native_time = schedule.time_for_lambda(noise_ratio)
            time_batch = torch.full(
                (len(clean),), native_time, device=clean.device, dtype=clean.dtype
            )
            alpha, sigma = schedule.coefficients(time_batch)
            noisy = alpha[:, None] * clean + sigma[:, None] * noise
            predicted = denoise(model, noisy, time_batch, schedule)
            identity = noisy / alpha[:, None]
        elif variant_id == "ve_diffusion":
            condition = torch.full(
                (len(clean),), noise_ratio, device=clean.device, dtype=clean.dtype
            )
            noisy = clean + noise_ratio * noise
            predicted = model(noisy, condition)
            identity = noisy
        elif variant_id == "posterior_log_noise_affine_flow":
            if (
                _field(config, "flow_schedule") != "log_noise"
                or _field(config, "flow_parameterization") != "posterior_mean"
                or _field(config, "flow_conditioning") != "log_noise_ratio"
            ):
                raise CanaryError(
                    "FM reconstruction requires posterior log-noise contract"
                )
            condition = torch.full(
                (len(clean),),
                math.log(noise_ratio),
                device=clean.device,
                dtype=clean.dtype,
            )
            noisy = clean + noise_ratio * noise
            predicted = model(noisy, condition)
            identity = noisy
        else:
            raise CanaryError(f"unsupported reconstruction variant {variant_id}")
    reconstruction_mse = float((predicted - clean).square().mean().cpu())
    identity_mse = float((identity - clean).square().mean().cpu())
    train_mean_mse = float(clean.square().mean().cpu())
    denominator = min(identity_mse, train_mean_mse)
    if denominator <= 0 or not all(
        math.isfinite(value)
        for value in (reconstruction_mse, identity_mse, train_mean_mse)
    ):
        raise CanaryError("invalid reconstruction diagnostic")
    ratio = reconstruction_mse / denominator
    arrays_path = output_dir / "reconstruction.npz"
    np.savez_compressed(
        arrays_path,
        query=np.asarray(clean.detach().cpu(), dtype=np.float32),
        noisy=np.asarray(noisy.detach().cpu(), dtype=np.float32),
        predicted=np.asarray(predicted.detach().cpu(), dtype=np.float32),
    )
    return {
        "status": "diagnostic_only",
        "quality_gate_applied": False,
        "historical_threshold_met": ratio <= maximum_ratio,
        "lambda": float(noise_ratio),
        "reconstruction_mse": reconstruction_mse,
        "identity_mse": identity_mse,
        "train_mean_mse": train_mean_mse,
        "ratio_to_better_trivial": ratio,
        "arrays": {
            "path": arrays_path.name,
            "sha256": file_sha256(arrays_path),
            "size_bytes": arrays_path.stat().st_size,
        },
    }


def _lid_quality(
    trained: Any,
    query: np.ndarray,
    target: np.ndarray,
    *,
    variant_id: str,
    family: str,
    representation: str,
    scales: Sequence[float],
    trace_seed: int,
    batch_size: int,
    maximum_mae: float | None,
    output_dir: Path,
) -> tuple[dict[str, Any], float]:
    from models.training import predict_lid, predict_nf_lid_ols5

    if representation == "coefficients":
        if maximum_mae is None:
            raise CanaryError("low-dimensional pointwise gate requires a threshold")
    elif representation == "dataset":
        if maximum_mae is not None:
            raise CanaryError("high-dimensional pointwise assessment cannot hard-gate")
    else:
        raise CanaryError("pointwise assessment representation is unsupported")
    readout = (
        "fixed_likelihood_ols5" if variant_id == "scale_conditioned_nf" else "full"
    )
    backend = (
        "exact"
        if representation == "coefficients" or variant_id == "scale_conditioned_nf"
        else "hutchinson"
    )
    probes = 0 if backend == "exact" else 16
    curves = []
    for scale in scales:
        if variant_id == "scale_conditioned_nf":
            prediction_value = predict_nf_lid_ols5(
                trained,
                query,
                float(scale),
                family=family,
                ols_log_step=0.05,
                batch_size=batch_size,
            )
        else:
            prediction_value = predict_lid(
                trained,
                query,
                float(scale),
                family=family,
                readout=readout,
                divergence_backend=backend,
                trace_probes=probes,
                trace_seed=trace_seed,
                batch_size=batch_size,
            )
        prediction = np.ravel(
            np.asarray(
                prediction_value,
                dtype=np.float64,
            )
        )
        if prediction.shape != target.shape or not np.isfinite(prediction).all():
            raise CanaryError("pointwise LID inference returned invalid output")
        curves.append(prediction)
    curve = np.column_stack(curves)
    maes = np.mean(np.abs(curve - target[:, None]), axis=0)
    selected = int(np.argmin(maes))
    boundary = selected in {0, len(scales) - 1}
    selected_mae = float(maes[selected])
    arrays_path = output_dir / "lid_curve.npz"
    np.savez_compressed(
        arrays_path,
        query=np.asarray(query),
        target=np.asarray(target, dtype=np.float64),
        scales=np.asarray(scales, dtype=np.float64),
        predictions=curve,
        candidate_mae=maes,
    )
    accuracy_gate_applied = False
    status = "diagnostic_only"
    return (
        {
            "status": status,
            "assessment_scope": (
                "diagnostic_low_dim_benchmark_outcome"
                if representation == "coefficients"
                else "diagnostic_high_dim_benchmark_outcome"
            ),
            "accuracy_gate_applied": accuracy_gate_applied,
            "maximum_pointwise_mae": maximum_mae,
            "selection_role": "canary_oracle_envelope_not_production_selector",
            "query_sha256": array_sha256(query),
            "target_sha256": array_sha256(target),
            "query_count": len(query),
            "divergence_backend": backend,
            "trace_probes": probes,
            "selected_index": selected,
            "selected_lambda": float(scales[selected]),
            "selected_pointwise_mae": selected_mae,
            "boundary_selected": boundary,
            "finite_fraction": float(np.isfinite(curve).mean()),
            "arrays": {
                "path": arrays_path.name,
                "sha256": file_sha256(arrays_path),
                "size_bytes": arrays_path.stat().st_size,
            },
        },
        float(scales[selected]),
    )


def _trace_quality(
    trained: Any,
    query: np.ndarray,
    target: np.ndarray,
    *,
    variant_id: str,
    family: str,
    representation: str,
    scale: float,
    trace_seed: int,
    batch_size: int,
    maximum_mae: float | None,
    maximum_ratio: float | None,
    output_dir: Path,
) -> dict[str, Any]:
    from models.training import predict_lid

    if variant_id == "scale_conditioned_nf":
        return {"status": "not_applicable"}
    if len(query) > batch_size:
        raise CanaryError(
            "trace prefix assessment requires one shared inference minibatch"
        )
    methods = [
        ("hutchinson16", "hutchinson", 16),
        ("hutchinson64", "hutchinson", 64),
    ]
    if representation == "coefficients":
        methods.insert(0, ("exact", "exact", 0))
    predictions = {}
    for name, backend, probes in methods:
        predictions[name] = np.ravel(
            np.asarray(
                predict_lid(
                    trained,
                    query,
                    scale,
                    family=family,
                    readout="full",
                    divergence_backend=backend,
                    trace_probes=probes,
                    trace_seed=trace_seed,
                    batch_size=batch_size,
                ),
                dtype=np.float64,
            )
        )
    if any(
        value.shape != target.shape or not np.isfinite(value).all()
        for value in predictions.values()
    ):
        raise CanaryError("trace diagnostic produced invalid pointwise output")
    h16_h64_mae = float(
        np.mean(np.abs(predictions["hutchinson64"] - predictions["hutchinson16"]))
    )
    if representation == "coefficients":
        if maximum_mae is None or maximum_ratio is None:
            raise CanaryError("low-dimensional trace gate requires exact thresholds")
        mae16 = float(
            np.mean(np.abs(predictions["hutchinson16"] - predictions["exact"]))
        )
        mae64 = float(
            np.mean(np.abs(predictions["hutchinson64"] - predictions["exact"]))
        )
        ratio = mae64 / max(mae16, 1e-12)
        comparison = "exact_vs_hutchinson16_64"
    else:
        if maximum_mae is not None or maximum_ratio is not None:
            raise CanaryError("high-dimensional trace assessment cannot hard-gate")
        mae16 = None
        mae64 = None
        ratio = None
        comparison = "common_probe_hutchinson16_vs64_no_exact_claim"
    arrays_path = output_dir / "trace_agreement.npz"
    np.savez_compressed(
        arrays_path,
        query=np.asarray(query),
        target=np.asarray(target, dtype=np.float64),
        **predictions,
    )
    return {
        "status": "diagnostic_only",
        "assessment_scope": (
            "diagnostic_low_dim_exact_comparison"
            if representation == "coefficients"
            else "diagnostic_high_dim_no_exact_claim"
        ),
        "accuracy_gate_applied": False,
        "maximum_trace64_mae": maximum_mae,
        "maximum_trace64_to_trace16_ratio": maximum_ratio,
        "query_sha256": array_sha256(query),
        "target_sha256": array_sha256(target),
        "query_count": len(query),
        "lambda": scale,
        "trace_seed": trace_seed,
        "probe_prefix_shared": True,
        "single_batch_prefix_verified": True,
        "comparison": comparison,
        "hutchinson16_mae_vs_exact": mae16,
        "hutchinson64_mae_vs_exact": mae64,
        "trace64_to_trace16_mae_ratio": ratio,
        "hutchinson64_mae_vs_hutchinson16": h16_h64_mae,
        "arrays": {
            "path": arrays_path.name,
            "sha256": file_sha256(arrays_path),
            "size_bytes": arrays_path.stat().st_size,
        },
    }


def _reference_selector_quality(
    trained: Any,
    query: np.ndarray,
    *,
    variant_id: str,
    family: str,
    representation: str,
    trace_seed: int,
    batch_size: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Exercise the production target-free E1/E5 selector without LID labels."""

    from experiments.global_campaign_v2 import (
        GlobalCampaignError,
        select_unknown_reference_kneedle,
        unknown_reference_lambdas,
    )
    from models.training import predict_lid, predict_nf_lid_ols5

    scales = np.asarray(unknown_reference_lambdas(), dtype=np.float64)
    backend = (
        "exact"
        if representation == "coefficients" or variant_id == "scale_conditioned_nf"
        else "hutchinson"
    )
    probes = 0 if backend == "exact" else 16
    columns: list[np.ndarray] = []
    for scale in scales:
        if variant_id == "scale_conditioned_nf":
            raw = predict_nf_lid_ols5(
                trained,
                query,
                float(scale),
                family=family,
                ols_log_step=0.05,
                batch_size=batch_size,
            )
        else:
            raw = predict_lid(
                trained,
                query,
                float(scale),
                family=family,
                readout="full",
                divergence_backend=backend,
                trace_probes=probes,
                trace_seed=trace_seed,
                batch_size=batch_size,
            )
        values = np.ravel(np.asarray(raw, dtype=np.float64))
        if values.shape != (len(query),) or not np.isfinite(values).all():
            raise CanaryError("reference-selector curve is not finite pointwise output")
        columns.append(values)
    curve = np.column_stack(columns)
    mean_curve = curve.mean(axis=0)
    arrays_path = output_dir / "reference_selector.npz"
    np.savez_compressed(
        arrays_path,
        query=np.asarray(query),
        scales=scales,
        pointwise_curve=curve,
        mean_curve=mean_curve,
    )
    try:
        selected_index, diagnostics = select_unknown_reference_kneedle(scales, curve)
    except GlobalCampaignError as exc:
        return {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "selection_uses_lid_targets": False,
            "query_sha256": array_sha256(query),
            "scales_sha256": array_sha256(scales),
            "mean_curve_sha256": array_sha256(mean_curve),
            "arrays": {
                "path": arrays_path.name,
                "sha256": file_sha256(arrays_path),
                "size_bytes": arrays_path.stat().st_size,
            },
        }
    if selected_index is None:
        return {
            "status": "diagnostic_only",
            "assessment_scope": "non_blocking_model_selection_outcome",
            "failure_reason": diagnostics.get("failure_reason", "selection_failed"),
            "selection_uses_lid_targets": False,
            "query_sha256": array_sha256(query),
            "query_count": len(query),
            "divergence_backend": backend,
            "trace_probes": probes,
            "scales_sha256": array_sha256(scales),
            "mean_curve_sha256": array_sha256(mean_curve),
            "diagnostics": _json_safe(diagnostics),
            "arrays": {
                "path": arrays_path.name,
                "sha256": file_sha256(arrays_path),
                "size_bytes": arrays_path.stat().st_size,
            },
        }
    if selected_index in {0, len(scales) - 1}:
        raise CanaryError(
            "reference selector returned a boundary despite fail-closed contract"
        )
    return {
        "status": "passed",
        "protocol_id": "held_out_reference_mean_kneedle_v2",
        "selection_uses_lid_targets": False,
        "query_sha256": array_sha256(query),
        "query_count": len(query),
        "divergence_backend": backend,
        "trace_probes": probes,
        "scales_sha256": array_sha256(scales),
        "mean_curve_sha256": array_sha256(mean_curve),
        "selected_index": int(selected_index),
        "selected_lambda": float(scales[selected_index]),
        "diagnostics": _json_safe(diagnostics),
        "arrays": {
            "path": arrays_path.name,
            "sha256": file_sha256(arrays_path),
            "size_bytes": arrays_path.stat().st_size,
        },
    }


def _streaming_diagonal_variance(
    values: np.ndarray, mean: np.ndarray, scale: float
) -> np.ndarray:
    flattened = np.asarray(values).reshape(len(values), -1)
    total = np.zeros(flattened.shape[1], dtype=np.float64)
    total_square = np.zeros_like(total)
    for start in range(0, len(flattened), 1024):
        batch = np.asarray(flattened[start : start + 1024], dtype=np.float64)
        normalized = (batch - mean) / scale
        total += normalized.sum(axis=0)
        total_square += np.square(normalized).sum(axis=0)
    empirical_mean = total / len(flattened)
    variance = total_square / len(flattened) - np.square(empirical_mean)
    return np.maximum(variance, 1e-6)


def _nf_scale_bin_quality(
    trained: Any,
    fit: np.ndarray,
    query: np.ndarray,
    *,
    bins: Sequence[Sequence[float]],
    seed: int,
    minimum_improved_fraction: float,
    output_dir: Path,
) -> dict[str, Any]:
    import torch

    model = _field(trained, "model")
    model.eval()
    clean = _normalized_queries(trained, query)
    mean = np.asarray(_field(trained, "normalization_mean"), dtype=np.float64)
    scale = float(_field(trained, "normalization_scale"))
    diagonal_variance = _streaming_diagonal_variance(fit, mean, scale)
    generator = torch.Generator(device=clean.device).manual_seed(seed)
    records = []
    all_nf, all_gaussian = [], []
    for index, bounds in enumerate(bins):
        lower, upper = map(float, bounds)
        log_eps = torch.linspace(
            math.log(lower),
            math.log(upper),
            len(clean),
            device=clean.device,
            dtype=clean.dtype,
        )
        epsilon = torch.exp(log_eps)
        noise = torch.randn(
            clean.shape, device=clean.device, dtype=clean.dtype, generator=generator
        )
        noisy = clean + epsilon[:, None] * noise
        with torch.no_grad():
            nf_nll = -model.log_prob(noisy, epsilon) / clean.shape[1]
        variance = (
            torch.as_tensor(diagonal_variance, device=clean.device, dtype=clean.dtype)[
                None, :
            ]
            + epsilon[:, None].square()
        )
        gaussian_nll = 0.5 * (
            noisy.square() / variance + torch.log(2.0 * math.pi * variance)
        ).mean(dim=1)
        nf_values = np.asarray(nf_nll.detach().cpu(), dtype=np.float64)
        gaussian_values = np.asarray(gaussian_nll.detach().cpu(), dtype=np.float64)
        if not np.isfinite(nf_values).all() or not np.isfinite(gaussian_values).all():
            raise CanaryError("NF scale-bin NLL contains non-finite values")
        nf_mean = float(nf_values.mean())
        gaussian_mean = float(gaussian_values.mean())
        records.append(
            {
                "index": index,
                "epsilon_min": lower,
                "epsilon_max": upper,
                "nf_nll_per_dimension": nf_mean,
                "diagonal_gaussian_nll_per_dimension": gaussian_mean,
                "improved": nf_mean < gaussian_mean,
            }
        )
        all_nf.append(nf_values)
        all_gaussian.append(gaussian_values)
    improved_fraction = sum(record["improved"] for record in records) / len(records)
    arrays_path = output_dir / "nf_scale_bin_nll.npz"
    np.savez_compressed(
        arrays_path,
        nf_nll=np.vstack(all_nf),
        diagonal_gaussian_nll=np.vstack(all_gaussian),
        bins=np.asarray(bins, dtype=np.float64),
    )
    return {
        "status": "diagnostic_only",
        "quality_gate_applied": False,
        "historical_threshold_met": improved_fraction >= minimum_improved_fraction,
        "reference": "train_fit_diagonal_gaussian_with_matching_noise",
        "query_sha256": array_sha256(query),
        "query_count": len(query),
        "bins": records,
        "improved_bin_fraction": improved_fraction,
        "arrays": {
            "path": arrays_path.name,
            "sha256": file_sha256(arrays_path),
            "size_bytes": arrays_path.stat().st_size,
        },
    }


_DIAGNOSTIC_OUTPUT_NAMES = frozenset(
    {
        "diagnostics.json",
        "diagnostics.json.tmp",
        "lid_curve.npz",
        "nf_scale_bin_nll.npz",
        "reconstruction.npz",
        "reference_selector.npz",
        "trace_agreement.npz",
    }
)


def _verify_nested_artifacts(value: Any, *, root: Path) -> int:
    count = 0
    if isinstance(value, Mapping):
        if set(value) == {"path", "sha256", "size_bytes"}:
            relative = Path(str(value["path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise CanaryError("cached diagnostic artifact path is unsafe")
            path = (root / relative).resolve()
            try:
                path.relative_to(root.resolve())
            except ValueError as exc:
                raise CanaryError(
                    "cached diagnostic artifact escapes its cell"
                ) from exc
            if (
                not path.is_file()
                or path.stat().st_size != value["size_bytes"]
                or file_sha256(path) != value["sha256"]
            ):
                raise CanaryError("cached diagnostic artifact changed")
            return 1
        for item in value.values():
            count += _verify_nested_artifacts(item, root=root)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            count += _verify_nested_artifacts(item, root=root)
    return count


def _resume_or_reset_diagnostics(
    output_dir: Path,
    *,
    expected_identity: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        return None
    entries = list(output_dir.iterdir())
    if not entries:
        return None
    record_path = output_dir / "diagnostics.json"
    if record_path.is_file():
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CanaryError("cached diagnostics record is unreadable") from exc
        if not isinstance(record, Mapping) or any(
            record.get(key) != value for key, value in expected_identity.items()
        ):
            raise CanaryError("cached diagnostics identity differs")
        if record.get("protocol_id") != PROTOCOL_ID or record.get("status") != "passed":
            raise CanaryError("cached diagnostics did not previously pass")
        if _verify_nested_artifacts(record, root=output_dir) < 3:
            raise CanaryError("cached diagnostics artifact coverage is incomplete")
        return dict(record)
    for path in entries:
        if path.name not in _DIAGNOSTIC_OUTPUT_NAMES or not path.is_file():
            raise CanaryError("partial diagnostics directory contains unknown output")
    for path in entries:
        path.unlink()
    return None


@dataclass(frozen=True)
class CanaryDiagnostics:
    """Pickle-safe pre-seal diagnostics hook for ``global_parallel_v2``."""

    config: Mapping[str, Any]

    def __call__(
        self, context: Mapping[str, Any], output_dir: Path
    ) -> Mapping[str, Any]:
        config = validate_canary_config(self.config)
        trained = _field(context, "trained")
        variant_id = str(_field(context, "model_variant"))
        family = str(_field(context, "family"))
        cell = _field(context, "cell")
        suite_id = str(_field(cell, "suite_id"))
        dataset = str(_field(cell, "dataset"))
        representation = str(_field(cell, "representation"))
        cell_id = f"{variant_id}/{suite_id}/{dataset}/{representation}"
        if cell_id not in EXPECTED_CELL_IDS:
            raise CanaryError(f"diagnostics received non-canary cell {cell_id}")
        eval_batch_size = _positive_int(
            int(_field(context, "eval_batch_size")), field="context.eval_batch_size"
        )
        partition = _field(context, "partition")
        selection = np.asarray(_field(partition, "selection_features"))
        target = np.ravel(
            np.asarray(_field(partition, "selection_target"), dtype=np.float64)
        )
        fit = np.asarray(_field(partition, "fit_features"))
        if target.shape != (len(selection),) or not np.isfinite(target).all():
            raise CanaryError(
                "canary requires finite known-LID train-selection targets"
            )
        ev = config["evaluation"]
        gates = config["gates"]
        candidate_scales = np.asarray(
            _field(context, "candidate_scales"), dtype=np.float64
        )
        production_selected_lambda = _finite_number(
            _field(context, "production_selected_lambda"),
            field="context.production_selected_lambda",
        )
        if (
            candidate_scales.ndim != 1
            or candidate_scales.size < 3
            or not np.isfinite(candidate_scales).all()
            or np.any(candidate_scales <= 0)
            or np.any(np.diff(candidate_scales) <= 0)
        ):
            raise CanaryError("production candidate scales are invalid")
        if production_selected_lambda <= 0 or production_selected_lambda not in set(
            candidate_scales.tolist()
        ):
            raise CanaryError("production selected lambda is not a candidate scale")
        declared_grid = np.asarray(ev["lambda_grid"], dtype=np.float64)
        if not set(declared_grid.tolist()).issubset(set(candidate_scales.tolist())):
            raise CanaryError(
                "production cell omitted declared canary lambda candidates"
            )
        subset_seed = _diagnostic_subset_seed(cell_id, int(ev["trace_seed"]))
        selection_count = int(ev["selection_query_count"][representation])
        selected_indices = _deterministic_subset(
            len(selection), selection_count, subset_seed
        )
        query = np.ascontiguousarray(selection[selected_indices])
        query_target = np.ascontiguousarray(target[selected_indices])
        trace_indices = _deterministic_subset(
            len(query), int(ev["trace_query_count"]), subset_seed + 1
        )
        trace_query = np.ascontiguousarray(query[trace_indices])
        trace_target = np.ascontiguousarray(query_target[trace_indices])
        diagnostic_identity = {
            "cell_id": cell_id,
            "selection_query_sha256": array_sha256(query),
            "trace_query_sha256": array_sha256(trace_query),
            "candidate_scales_sha256": array_sha256(candidate_scales),
            "production_selected_lambda": production_selected_lambda,
            "cell_identity_sha256": canonical_sha256(_field(context, "cell_identity")),
            "preprocessing_sha256": _sha(
                _field(trained, "preprocessing_sha256"),
                field="trained.preprocessing_sha256",
            ),
        }
        cached = _resume_or_reset_diagnostics(
            output_dir, expected_identity=diagnostic_identity
        )
        if cached is not None:
            return cached
        native_loss = _history_quality(trained, gates)
        lid_quality, _ = _lid_quality(
            trained,
            query,
            query_target,
            variant_id=variant_id,
            family=family,
            representation=representation,
            scales=tuple(float(value) for value in candidate_scales),
            trace_seed=subset_seed + 2,
            batch_size=eval_batch_size,
            maximum_mae=(
                float(gates["maximum_low_dim_pointwise_lid_mae"])
                if representation == "coefficients"
                else None
            ),
            output_dir=output_dir,
        )
        reconstruction = _reconstruction_quality(
            trained,
            query,
            variant_id=variant_id,
            noise_ratio=float(ev["reconstruction_lambda"]),
            seed=subset_seed + 3,
            maximum_ratio=float(gates["maximum_reconstruction_ratio"]),
            output_dir=output_dir,
        )
        trace = _trace_quality(
            trained,
            trace_query,
            trace_target,
            variant_id=variant_id,
            family=family,
            representation=representation,
            scale=production_selected_lambda,
            trace_seed=subset_seed + 4,
            batch_size=eval_batch_size,
            maximum_mae=(
                float(gates["maximum_low_dim_trace64_mae"])
                if representation == "coefficients"
                else None
            ),
            maximum_ratio=(
                float(gates["maximum_low_dim_trace64_to_trace16_ratio"])
                if representation == "coefficients"
                else None
            ),
            output_dir=output_dir,
        )
        reference_selector = _reference_selector_quality(
            trained,
            query,
            variant_id=variant_id,
            family=family,
            representation=representation,
            trace_seed=subset_seed + 6,
            batch_size=eval_batch_size,
            output_dir=output_dir,
        )
        nf_nll = (
            _nf_scale_bin_quality(
                trained,
                fit,
                query,
                bins=ev["nf_scale_bins"],
                seed=subset_seed + 5,
                minimum_improved_fraction=float(
                    gates["minimum_nf_improved_bin_fraction"]
                ),
                output_dir=output_dir,
            )
            if variant_id == "scale_conditioned_nf"
            else {"status": "not_applicable"}
        )
        result = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "cell_id": cell_id,
            "variant_id": variant_id,
            "family": family,
            "suite_id": suite_id,
            "dataset": dataset,
            "representation": representation,
            "selection_subset_indices_sha256": array_sha256(selected_indices),
            "selection_query_sha256": diagnostic_identity["selection_query_sha256"],
            "trace_subset_indices_sha256": array_sha256(trace_indices),
            "trace_query_sha256": diagnostic_identity["trace_query_sha256"],
            "candidate_scales": candidate_scales.tolist(),
            "candidate_scales_sha256": diagnostic_identity["candidate_scales_sha256"],
            "production_selected_lambda": production_selected_lambda,
            "cell_identity_sha256": diagnostic_identity["cell_identity_sha256"],
            "preprocessing_sha256": diagnostic_identity["preprocessing_sha256"],
            "native_loss": native_loss,
            "reconstruction": reconstruction,
            "pointwise_lid": lid_quality,
            "reference_selector": reference_selector,
            "nf_scale_bin_nll": nf_nll,
            "trace_agreement": trace,
        }
        if any(
            value["status"] == "failed"
            for value in (
                native_loss,
                reconstruction,
                lid_quality,
                reference_selector,
                nf_nll,
                trace,
            )
        ):
            result["status"] = "failed"
        else:
            result["status"] = "passed"
        safe = _json_safe(result)
        _atomic_json(output_dir / "diagnostics.json", safe)
        return safe


def inspect_arrows_gate(
    project_root: Path,
    config: Mapping[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Validate raw NHWC/RGB Arrows and the exact scalar-RMS preprocessing."""

    import torch
    from PIL import Image, ImageDraw, ImageFont

    from datasets.registry import load_registry, load_split
    from experiments.global_campaign_v2 import (
        compose_global_campaign_v2_config,
        partition_source_train,
        validate_global_campaign_config,
    )
    from models.training import _normalization

    pillow_version = importlib.metadata.version("pillow")
    if pillow_version != "11.3.0":
        raise CanaryError(
            f"Arrows contact sheet requires pillow==11.3.0, got {pillow_version}"
        )

    validated = validate_canary_config(config)
    data = validated["data"]
    registry_path = _resolve(project_root, data["registry"], field="data.registry")
    benchmark_root = _resolve(
        project_root, data["benchmark_root"], field="data.benchmark_root"
    )
    registry = load_registry(registry_path, validate_official_coverage=False)
    split = load_split(
        benchmark_root,
        registry[data["arrows_dataset"]],
        "train",
        representation="dataset",
        mmap_mode="r",
    )
    values = np.asarray(split.features)
    if values.shape != (100000, 32, 32, 3):
        raise CanaryError(f"unexpected Arrows source-train shape: {values.shape}")
    if not np.issubdtype(values.dtype, np.number):
        raise CanaryError("Arrows pixels must have numeric dtype")
    flat = values.reshape(len(values), -1)
    raw_min, raw_max = math.inf, -math.inf
    for start in range(0, len(flat), 512):
        batch = np.asarray(flat[start : start + 512], dtype=np.float64)
        if not np.isfinite(batch).all():
            raise CanaryError("Arrows contains non-finite pixels")
        raw_min = min(raw_min, float(batch.min()))
        raw_max = max(raw_max, float(batch.max()))
    production_config = validate_global_campaign_config(
        compose_global_campaign_v2_config(root=project_root)
    )
    torch.set_num_threads(4)
    production_selection = production_config["campaign"]["selection"]
    for name in (
        "fraction",
        "maximum_selection",
        "minimum_selection",
        "minimum_fit",
    ):
        if data["selection"][name] != production_selection[name]:
            raise CanaryError("Arrows data gate partition differs from production")
    if data["selection"]["protocol"] != production_selection["index_algorithm"]:
        raise CanaryError("Arrows data gate index algorithm differs from production")
    if data["selection"]["seed"] != production_config["seed"]:
        raise CanaryError("Arrows data gate seed differs from production")
    if split.lid is None:
        raise CanaryError("Arrows construction targets are absent")
    partition = partition_source_train(
        flat,
        split.lid,
        selection=production_selection,
        seed=int(production_config["seed"]),
    )
    fit_tensor = torch.as_tensor(partition.fit_features, dtype=torch.float32)
    mean_tensor, rms, preprocessing, preprocessing_sha = _normalization(
        fit_tensor,
        enabled=True,
        epsilon=1.0e-8,
    )
    mean = np.asarray(mean_tensor, dtype=np.float32)
    del fit_tensor, mean_tensor
    if not math.isfinite(rms) or rms <= 1e-8 or raw_max <= raw_min:
        raise CanaryError("Arrows scalar-RMS preprocessing is degenerate")
    indices = _deterministic_subset(len(values), 16, int(data["visual_sample_seed"]))
    sample = np.ascontiguousarray(values[indices])
    normalized_sample = (
        sample.reshape(len(sample), -1).astype(np.float32) - mean
    ) / rms
    if not np.isfinite(normalized_sample).all():
        raise CanaryError("Arrows normalized sample is non-finite")
    output_dir.mkdir(parents=True, exist_ok=False)
    source_dataset_sha = file_sha256(split.source_paths["dataset"])
    source_target_sha = file_sha256(split.source_paths["lid"])
    scaled = np.clip(
        np.rint((normalized_sample.reshape(sample.shape) + 3.0) * (255.0 / 6.0)),
        0.0,
        255.0,
    ).astype(np.uint8)
    scale_factor = 4
    tile_width = 32 * scale_factor
    label_height = 18
    margin = 8
    header_height = 62
    sheet = Image.new(
        "RGB",
        (
            4 * tile_width + 5 * margin,
            header_height + 4 * (tile_width + label_height) + 5 * margin,
        ),
        color=(245, 245, 245),
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text(
        (margin, 6),
        "e2_arrows/train | source=" + source_dataset_sha[:16],
        fill=(0, 0, 0),
        font=font,
    )
    draw.text(
        (margin, 24),
        f"optimizer-fit scalar_rms={rms:.8g} | standardized display [-3,3]",
        fill=(0, 0, 0),
        font=font,
    )
    draw.text(
        (margin, 42),
        "construction labels 6*k; inspect quantization, clipping and overlaps",
        fill=(0, 0, 0),
        font=font,
    )
    lids = np.asarray(split.lid)[indices]
    for position, (index, image, lid) in enumerate(zip(indices, scaled, lids)):
        row, column = divmod(position, 4)
        left = margin + column * (tile_width + margin)
        top = header_height + margin + row * (tile_width + label_height + margin)
        tile = Image.fromarray(image, mode="RGB").resize(
            (tile_width, tile_width), resample=Image.Resampling.NEAREST
        )
        sheet.paste(tile, (left, top))
        draw.text(
            (left, top + tile_width + 2),
            f"idx={int(index)} lid={float(lid):g}",
            fill=(0, 0, 0),
            font=font,
        )
    contact_sheet = output_dir / "arrows_contact_sheet.png"
    sheet.save(
        contact_sheet,
        format="PNG",
        optimize=False,
        compress_level=9,
        pnginfo=None,
    )
    return {
        "status": "passed",
        "dataset": "e2_arrows",
        "layout": "NHWC_RGB",
        "shape": list(values.shape),
        "dtype": values.dtype.str,
        "raw_min": raw_min,
        "raw_max": raw_max,
        "normalization_rms": rms,
        "preprocessing_sha256": preprocessing_sha,
        "preprocessing": preprocessing,
        "production_partition_sha256": partition.record["partition_sha256"],
        "optimizer_fit_indices_sha256": partition.record["fit_indices_sha256"],
        "selection_indices_sha256": partition.record["selection_indices_sha256"],
        "visual_sample_seed": int(data["visual_sample_seed"]),
        "display_transform": "production_standardized_clip_minus3_plus3_v1",
        "source_dataset_sha256": source_dataset_sha,
        "source_target_sha256": source_target_sha,
        "sample_indices_sha256": array_sha256(indices),
        "sample_images_sha256": array_sha256(sample),
        "target_sha256": array_sha256(split.lid),
        "pillow_version": pillow_version,
        "contact_sheet": {
            "path": contact_sheet.name,
            "sha256": file_sha256(contact_sheet),
            "size_bytes": contact_sheet.stat().st_size,
        },
    }


def write_arrows_gate_bundle(
    project_root: Path,
    config: Mapping[str, Any],
    *,
    config_path: Path,
    output_dir: Path,
) -> Path:
    """Create automated evidence for a subsequent explicit visual review."""

    validated = validate_canary_config(config)
    source = _git_source_record(project_root)
    arrows = inspect_arrows_gate(project_root, validated, output_dir=output_dir)
    gate = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "automated_checks_passed_pending_human_review",
        "source": source,
        "config": {
            "path": config_path.resolve()
            .relative_to(project_root.resolve())
            .as_posix(),
            "sha256": file_sha256(config_path),
        },
        "arrows": arrows,
        "human_review": {"status": "pending"},
        "outputs": [arrows["contact_sheet"]],
    }
    path = output_dir / "arrows_gate.json"
    _atomic_json(path, gate)
    return path


def approve_arrows_gate(
    gate_path: Path,
    *,
    reviewer: str,
    reviewed_at: str,
) -> Path:
    """Seal an agent/human visual review after the PNG was actually inspected."""

    if not reviewer.strip() or not reviewed_at.strip():
        raise CanaryError("reviewer and reviewed_at must be non-empty")
    try:
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CanaryError("cannot read Arrows data-gate bundle") from exc
    if (
        not isinstance(gate, Mapping)
        or gate.get("schema_version") != 1
        or gate.get("protocol_id") != PROTOCOL_ID
        or gate.get("status") != "automated_checks_passed_pending_human_review"
        or gate.get("human_review") != {"status": "pending"}
    ):
        raise CanaryError("Arrows gate is not awaiting review")
    contact = gate.get("arrows", {}).get("contact_sheet")
    if not isinstance(contact, Mapping) or set(contact) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise CanaryError("Arrows contact-sheet record is invalid")
    contact_path = gate_path.parent / str(contact["path"])
    if (
        not contact_path.is_file()
        or contact_path.stat().st_size != contact["size_bytes"]
        or file_sha256(contact_path) != contact["sha256"]
    ):
        raise CanaryError("Arrows contact sheet changed before review sealing")
    reviewed = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "reviewed",
        "reviewer": reviewer.strip(),
        "reviewed_at": reviewed_at.strip(),
        "gate": {
            "path": gate_path.name,
            "sha256": file_sha256(gate_path),
            "size_bytes": gate_path.stat().st_size,
        },
        "contact_sheet": dict(contact),
        "source": gate["source"],
        "config": gate["config"],
        "arrows": gate["arrows"],
    }
    destination = gate_path.parent / "arrows_gate_reviewed.json"
    if destination.exists():
        raise CanaryError("reviewed Arrows gate already exists")
    _atomic_json(destination, reviewed)
    return destination


def load_reviewed_arrows_gate(
    reviewed_path: Path,
    *,
    expected_source: Mapping[str, Any],
    expected_config_sha256: str,
) -> dict[str, Any]:
    """Verify visual-review evidence against the exact canary source/config."""

    try:
        reviewed = json.loads(reviewed_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CanaryError("cannot read reviewed Arrows gate") from exc
    reviewed = _exact_keys(
        reviewed,
        {
            "schema_version",
            "protocol_id",
            "status",
            "reviewer",
            "reviewed_at",
            "gate",
            "contact_sheet",
            "source",
            "config",
            "arrows",
        },
        field="reviewed Arrows gate",
    )
    if (
        reviewed["schema_version"] != 1
        or reviewed["protocol_id"] != PROTOCOL_ID
        or reviewed["status"] != "reviewed"
        or not str(reviewed["reviewer"]).strip()
        or not str(reviewed["reviewed_at"]).strip()
    ):
        raise CanaryError("Arrows gate has no explicit completed visual review")
    if reviewed["source"] != dict(expected_source):
        raise CanaryError("reviewed Arrows source differs from canary source")
    if reviewed["config"].get("sha256") != expected_config_sha256:
        raise CanaryError("reviewed Arrows config differs from canary config")
    for name in ("gate", "contact_sheet"):
        artifact = _exact_keys(
            reviewed[name], {"path", "sha256", "size_bytes"}, field=f"reviewed.{name}"
        )
        path = reviewed_path.parent / str(artifact["path"])
        if (
            not path.is_file()
            or path.stat().st_size != artifact["size_bytes"]
            or file_sha256(path) != artifact["sha256"]
        ):
            raise CanaryError(f"reviewed Arrows {name} artifact changed")
    arrows = reviewed["arrows"]
    if arrows.get("contact_sheet") != reviewed["contact_sheet"]:
        raise CanaryError("reviewed contact sheet is not bound to Arrows evidence")
    return dict(reviewed)


def _validate_diagnostic_artifact(value: Any, *, field: str) -> Mapping[str, Any]:
    record = _exact_keys(value, {"path", "sha256", "size_bytes"}, field=field)
    relative = Path(str(record["path"]))
    if relative.is_absolute() or ".." in relative.parts or relative.parent != Path("."):
        raise CanaryError(f"{field} path must be a local relative filename")
    _sha(record["sha256"], field=f"{field}.sha256")
    _positive_int(record["size_bytes"], field=f"{field}.size_bytes")
    return record


def _validate_pointwise_assessment(
    value: Any,
    *,
    variant_id: str,
    representation: str,
) -> Mapping[str, Any]:
    record = _exact_keys(
        value,
        {
            "status",
            "assessment_scope",
            "accuracy_gate_applied",
            "maximum_pointwise_mae",
            "selection_role",
            "query_sha256",
            "target_sha256",
            "query_count",
            "divergence_backend",
            "trace_probes",
            "selected_index",
            "selected_lambda",
            "selected_pointwise_mae",
            "boundary_selected",
            "finite_fraction",
            "arrays",
        },
        field="pointwise LID assessment",
    )
    is_low_dimensional = representation == "coefficients"
    expected_backend = (
        "exact"
        if is_low_dimensional or variant_id == "scale_conditioned_nf"
        else "hutchinson"
    )
    expected_probes = 0 if expected_backend == "exact" else 16
    expected_status = "diagnostic_only"
    expected_scope = (
        "diagnostic_low_dim_benchmark_outcome"
        if is_low_dimensional
        else "diagnostic_high_dim_benchmark_outcome"
    )
    if (
        record["status"] != expected_status
        or record["assessment_scope"] != expected_scope
        or record["accuracy_gate_applied"] is not False
        or record["selection_role"] != "canary_oracle_envelope_not_production_selector"
        or record["divergence_backend"] != expected_backend
        or record["trace_probes"] != expected_probes
        or record["query_count"] != EXPECTED_SELECTION_QUERY_COUNTS[representation]
        or record["finite_fraction"] != 1.0
        or not isinstance(record["boundary_selected"], bool)
    ):
        raise CanaryError("pointwise LID assessment contract differs")
    _sha(record["query_sha256"], field="pointwise query SHA")
    _sha(record["target_sha256"], field="pointwise target SHA")
    if (
        isinstance(record["selected_index"], bool)
        or not isinstance(record["selected_index"], int)
        or record["selected_index"] < 0
        or _finite_number(record["selected_lambda"], field="pointwise selected lambda")
        <= 0
    ):
        raise CanaryError("pointwise selected scale is invalid")
    selected_mae = _finite_number(
        record["selected_pointwise_mae"], field="pointwise selected MAE"
    )
    if selected_mae < 0:
        raise CanaryError("pointwise selected MAE must be non-negative")
    if is_low_dimensional:
        maximum_mae = _finite_number(
            record["maximum_pointwise_mae"], field="pointwise maximum MAE"
        )
        if maximum_mae != MAXIMUM_LOW_DIM_POINTWISE_LID_MAE:
            raise CanaryError("pointwise historical diagnostic threshold differs")
    elif record["maximum_pointwise_mae"] is not None:
        raise CanaryError("high-dimensional pointwise assessment has a hard threshold")
    return _validate_diagnostic_artifact(record["arrays"], field="pointwise arrays")


def _validate_trace_assessment(
    value: Any,
    *,
    representation: str,
    production_selected_lambda: float,
) -> Mapping[str, Any]:
    record = _exact_keys(
        value,
        {
            "status",
            "assessment_scope",
            "accuracy_gate_applied",
            "maximum_trace64_mae",
            "maximum_trace64_to_trace16_ratio",
            "query_sha256",
            "target_sha256",
            "query_count",
            "lambda",
            "trace_seed",
            "probe_prefix_shared",
            "single_batch_prefix_verified",
            "comparison",
            "hutchinson16_mae_vs_exact",
            "hutchinson64_mae_vs_exact",
            "trace64_to_trace16_mae_ratio",
            "hutchinson64_mae_vs_hutchinson16",
            "arrays",
        },
        field="trace assessment",
    )
    is_low_dimensional = representation == "coefficients"
    expected_status = "diagnostic_only"
    expected_scope = (
        "diagnostic_low_dim_exact_comparison"
        if is_low_dimensional
        else "diagnostic_high_dim_no_exact_claim"
    )
    expected_comparison = (
        "exact_vs_hutchinson16_64"
        if is_low_dimensional
        else "common_probe_hutchinson16_vs64_no_exact_claim"
    )
    if (
        record["status"] != expected_status
        or record["assessment_scope"] != expected_scope
        or record["accuracy_gate_applied"] is not False
        or record["query_count"] != EXPECTED_TRACE_QUERY_COUNT
        or record["probe_prefix_shared"] is not True
        or record["single_batch_prefix_verified"] is not True
        or record["comparison"] != expected_comparison
        or _finite_number(record["lambda"], field="trace lambda")
        != production_selected_lambda
        or isinstance(record["trace_seed"], bool)
        or not isinstance(record["trace_seed"], int)
        or record["trace_seed"] < 0
    ):
        raise CanaryError("trace assessment contract differs")
    _sha(record["query_sha256"], field="trace query SHA")
    _sha(record["target_sha256"], field="trace target SHA")
    disagreement = _finite_number(
        record["hutchinson64_mae_vs_hutchinson16"],
        field="H64 versus H16 MAE",
    )
    if disagreement < 0:
        raise CanaryError("H64 versus H16 MAE must be non-negative")
    if is_low_dimensional:
        mae16 = _finite_number(
            record["hutchinson16_mae_vs_exact"], field="H16 exact MAE"
        )
        mae64 = _finite_number(
            record["hutchinson64_mae_vs_exact"], field="H64 exact MAE"
        )
        ratio = _finite_number(
            record["trace64_to_trace16_mae_ratio"], field="H64/H16 ratio"
        )
        maximum_mae = _finite_number(
            record["maximum_trace64_mae"], field="maximum H64 MAE"
        )
        maximum_ratio = _finite_number(
            record["maximum_trace64_to_trace16_ratio"],
            field="maximum H64/H16 ratio",
        )
        if (
            min(mae16, mae64, ratio) < 0
            or maximum_mae != MAXIMUM_LOW_DIM_TRACE64_MAE
            or maximum_ratio != MAXIMUM_LOW_DIM_TRACE64_TO_TRACE16_RATIO
        ):
            raise CanaryError("trace diagnostic values or historical thresholds differ")
    elif any(
        record[name] is not None
        for name in (
            "maximum_trace64_mae",
            "maximum_trace64_to_trace16_ratio",
            "hutchinson16_mae_vs_exact",
            "hutchinson64_mae_vs_exact",
            "trace64_to_trace16_mae_ratio",
        )
    ):
        raise CanaryError("high-dimensional trace assessment makes an exact claim")
    return _validate_diagnostic_artifact(record["arrays"], field="trace arrays")


def _validate_reference_selector_assessment(
    value: Any,
    *,
    variant_id: str,
    representation: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CanaryError("reference-selector assessment must be a mapping")
    common = {
        "status",
        "selection_uses_lid_targets",
        "query_sha256",
        "query_count",
        "divergence_backend",
        "trace_probes",
        "scales_sha256",
        "mean_curve_sha256",
        "diagnostics",
        "arrays",
    }
    if value.get("status") == "passed":
        expected = common | {"protocol_id", "selected_index", "selected_lambda"}
    elif value.get("status") == "diagnostic_only":
        expected = common | {"assessment_scope", "failure_reason"}
    else:
        raise CanaryError("reference-selector assessment did not execute cleanly")
    record = _exact_keys(value, expected, field="reference-selector assessment")
    expected_backend = (
        "exact"
        if representation == "coefficients" or variant_id == "scale_conditioned_nf"
        else "hutchinson"
    )
    expected_probes = 0 if expected_backend == "exact" else 16
    if (
        record["selection_uses_lid_targets"] is not False
        or record["query_count"] != EXPECTED_SELECTION_QUERY_COUNTS[representation]
        or record["divergence_backend"] != expected_backend
        or record["trace_probes"] != expected_probes
        or not isinstance(record["diagnostics"], Mapping)
    ):
        raise CanaryError("reference-selector assessment contract differs")
    for name in ("query_sha256", "scales_sha256", "mean_curve_sha256"):
        _sha(record[name], field=f"reference selector {name}")
    if record["status"] == "passed":
        if (
            record["protocol_id"] != "held_out_reference_mean_kneedle_v2"
            or isinstance(record["selected_index"], bool)
            or not isinstance(record["selected_index"], int)
            or record["selected_index"] <= 0
            or _finite_number(
                record["selected_lambda"], field="reference selected lambda"
            )
            <= 0
            or record["diagnostics"].get("status") != "selected"
        ):
            raise CanaryError("reference-selector selected result is invalid")
    elif (
        record["assessment_scope"] != "non_blocking_model_selection_outcome"
        or not isinstance(record["failure_reason"], str)
        or not record["failure_reason"]
        or record["diagnostics"].get("status") != "selection_failed"
    ):
        raise CanaryError("reference-selector diagnostic outcome is invalid")
    return _validate_diagnostic_artifact(
        record["arrays"], field="reference-selector arrays"
    )


def _load_retained_cell_json(
    artifacts: Any,
    *,
    filename: str,
    report_root: Path,
) -> Mapping[str, Any]:
    if not isinstance(artifacts, list):
        raise CanaryError("cell artifacts must be a list")
    matches = [
        item
        for item in artifacts
        if isinstance(item, Mapping)
        and Path(str(item.get("path", ""))).name == filename
    ]
    if len(matches) != 1:
        raise CanaryError(f"cell must retain exactly one {filename}")
    record = _exact_keys(
        matches[0], {"path", "sha256", "size_bytes"}, field=f"cell {filename}"
    )
    relative = Path(str(record["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise CanaryError(f"cell {filename} path is unsafe")
    path = (report_root / relative).resolve()
    try:
        path.relative_to(report_root.resolve())
    except ValueError as exc:
        raise CanaryError(f"cell {filename} escapes report root") from exc
    if (
        not path.is_file()
        or path.stat().st_size != record["size_bytes"]
        or file_sha256(path) != record["sha256"]
    ):
        raise CanaryError(f"retained cell {filename} changed")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CanaryError(f"retained cell {filename} is unreadable") from exc
    if not isinstance(value, Mapping):
        raise CanaryError(f"retained cell {filename} must contain a mapping")
    return value


def _validate_cell_report_artifact_binding(
    cell: Mapping[str, Any], *, report_root: Path
) -> None:
    summary = _load_retained_cell_json(
        cell["artifacts"], filename="summary.json", report_root=report_root
    )
    quality = _load_retained_cell_json(
        cell["artifacts"],
        filename="quality_diagnostics.json",
        report_root=report_root,
    )
    embedded_quality = summary.get("quality_diagnostics")
    if not isinstance(embedded_quality, Mapping) or embedded_quality != quality:
        raise CanaryError("retained summary and quality diagnostics differ")
    production_selected_lambda = cell["production_selected_lambda"]
    if (
        summary.get("model_variant") != cell["variant_id"]
        or summary.get("suite_id") != cell["suite_id"]
        or summary.get("dataset") != cell["dataset"]
        or summary.get("representation") != cell["representation"]
        or summary.get("selected_scale") != production_selected_lambda
        or quality.get("production_selected_lambda") != production_selected_lambda
    ):
        raise CanaryError("canary report is not bound to retained production summary")
    for field in (
        "native_loss",
        "reconstruction",
        "pointwise_lid",
        "reference_selector",
        "nf_scale_bin_nll",
        "trace_agreement",
    ):
        if quality.get(field) != cell[field]:
            raise CanaryError(f"canary report {field} differs from retained evidence")
    trace = quality["trace_agreement"]
    if cell["variant_id"] != "scale_conditioned_nf" and (
        not isinstance(trace, Mapping)
        or trace.get("lambda") != production_selected_lambda
    ):
        raise CanaryError("retained trace is not bound to production selected scale")


def validate_canary_report(
    report: Mapping[str, Any],
    *,
    report_dir: str | Path,
    expected_commit: str | None = None,
    expected_git_tree_sha256: str | None = None,
    expected_declared_source_sha256: str | None = None,
    expected_config_sha256: str | None = None,
    expected_campaign_identity: str | None = None,
    expected_campaign_config_sha256: str | None = None,
    expected_input_inventory_sha256: str | None = None,
) -> dict[str, Any]:
    """Strictly verify a PASS attestation and every compact output hash."""

    root = Path(report_dir).resolve()
    top = _exact_keys(
        report,
        {
            "schema_version",
            "protocol_id",
            "status",
            "source",
            "config",
            "preflight",
            "utilization_probe",
            "runtime_projection",
            "cells",
            "gates",
            "outputs",
            "failures",
        },
        field="report",
    )
    if top["schema_version"] != SCHEMA_VERSION or top["protocol_id"] != PROTOCOL_ID:
        raise CanaryError("report schema/protocol mismatch")
    if top["status"] != "passed" or top["failures"] != []:
        raise CanaryError("canary report did not pass cleanly")

    source = _exact_keys(
        top["source"],
        {
            "git_commit",
            "git_tree_sha256",
            "declared_source_sha256",
            "worktree_clean",
        },
        field="source",
    )
    if (
        not isinstance(source["git_commit"], str)
        or _GIT_SHA.fullmatch(source["git_commit"]) is None
    ):
        raise CanaryError("source.git_commit must be a full Git SHA")
    _sha(source["git_tree_sha256"], field="source.git_tree_sha256")
    _sha(
        source["declared_source_sha256"],
        field="source.declared_source_sha256",
    )
    if source["worktree_clean"] is not True:
        raise CanaryError("canary source worktree was not clean")
    if expected_commit is not None and source["git_commit"] != expected_commit:
        raise CanaryError("canary commit does not match the full campaign commit")
    if (
        expected_git_tree_sha256 is not None
        and source["git_tree_sha256"] != expected_git_tree_sha256
    ):
        raise CanaryError("canary Git tree does not match the launch assertion")
    if (
        expected_declared_source_sha256 is not None
        and source["declared_source_sha256"] != expected_declared_source_sha256
    ):
        raise CanaryError(
            "canary declared source does not match the full campaign source"
        )

    config = _exact_keys(
        top["config"],
        {
            "path",
            "sha256",
            "batch_size",
            "physical_device_count",
            "lanes_per_device",
            "logical_worker_count",
            "campaign_identity",
            "campaign_config_sha256",
            "input_inventory_sha256",
        },
        field="report.config",
    )
    _sha(config["sha256"], field="report.config.sha256")
    _sha(config["campaign_identity"], field="report.config.campaign_identity")
    _sha(
        config["campaign_config_sha256"],
        field="report.config.campaign_config_sha256",
    )
    _sha(
        config["input_inventory_sha256"],
        field="report.config.input_inventory_sha256",
    )
    if (
        expected_config_sha256 is not None
        and config["sha256"] != expected_config_sha256
    ):
        raise CanaryError("canary config hash mismatch")
    if (
        config["batch_size"] != EXPECTED_BATCH_SIZE
        or config["physical_device_count"] != EXPECTED_WORKER_COUNT
        or config["lanes_per_device"] != EXPECTED_LANES_PER_DEVICE
        or config["logical_worker_count"] != EXPECTED_PROCESS_COUNT
    ):
        raise CanaryError(
            "report does not attest the required 8x3-lane batch-256 execution"
        )
    for expected, field in (
        (expected_campaign_identity, "campaign_identity"),
        (expected_campaign_config_sha256, "campaign_config_sha256"),
        (expected_input_inventory_sha256, "input_inventory_sha256"),
    ):
        if expected is not None and config[field] != expected:
            raise CanaryError(f"canary {field} differs from the full campaign")

    preflight = _exact_keys(
        top["preflight"],
        {"status", "archive_sha256", "arrows", "devices"},
        field="preflight",
    )
    if preflight["status"] != "passed":
        raise CanaryError("preflight did not pass")
    _sha(preflight["archive_sha256"], field="preflight.archive_sha256")
    arrows = _exact_keys(
        preflight["arrows"],
        {
            "status",
            "dataset",
            "layout",
            "shape",
            "dtype",
            "raw_min",
            "raw_max",
            "normalization_rms",
            "preprocessing_sha256",
            "preprocessing",
            "production_partition_sha256",
            "optimizer_fit_indices_sha256",
            "selection_indices_sha256",
            "visual_sample_seed",
            "display_transform",
            "source_dataset_sha256",
            "source_target_sha256",
            "sample_indices_sha256",
            "sample_images_sha256",
            "target_sha256",
            "pillow_version",
            "contact_sheet",
            "human_review",
        },
        field="preflight.arrows",
    )
    if (
        arrows["status"] != "passed"
        or arrows["dataset"] != "e2_arrows"
        or arrows["layout"] != "NHWC_RGB"
    ):
        raise CanaryError("Arrows data/preprocessing gate did not pass")
    if arrows["shape"] != [100000, 32, 32, 3]:
        raise CanaryError("Arrows source-train shape mismatch")
    for name in ("raw_min", "raw_max", "normalization_rms"):
        _finite_number(arrows[name], field=f"preflight.arrows.{name}")
    if arrows["raw_max"] <= arrows["raw_min"] or arrows["normalization_rms"] <= 0:
        raise CanaryError("Arrows has degenerate intensity/preprocessing statistics")
    if arrows["pillow_version"] != "11.3.0":
        raise CanaryError("Arrows contact sheet used an unpinned Pillow version")
    for name in (
        "source_dataset_sha256",
        "source_target_sha256",
        "sample_indices_sha256",
        "sample_images_sha256",
        "target_sha256",
        "preprocessing_sha256",
        "production_partition_sha256",
        "optimizer_fit_indices_sha256",
        "selection_indices_sha256",
    ):
        _sha(arrows[name], field=f"preflight.arrows.{name}")
    if (
        not isinstance(arrows["preprocessing"], Mapping)
        or arrows["preprocessing"].get("kind") != "train_mean_global_rms_v1"
        or canonical_sha256(arrows["preprocessing"]) != arrows["preprocessing_sha256"]
        or arrows["preprocessing"].get("scalar_scale") != arrows["normalization_rms"]
        or arrows["visual_sample_seed"] != 20260906
        or arrows["display_transform"] != "production_standardized_clip_minus3_plus3_v1"
    ):
        raise CanaryError("Arrows production preprocessing evidence differs")
    contact_sheet = _exact_keys(
        arrows["contact_sheet"],
        {"path", "sha256", "size_bytes"},
        field="Arrows contact sheet",
    )
    _sha(contact_sheet["sha256"], field="Arrows contact sheet SHA")
    review = _exact_keys(
        arrows["human_review"],
        {"status", "reviewer", "reviewed_at", "reviewed_gate_sha256"},
        field="Arrows human review",
    )
    if (
        review["status"] != "reviewed"
        or not str(review["reviewer"]).strip()
        or not str(review["reviewed_at"]).strip()
    ):
        raise CanaryError("Arrows contact sheet lacks an explicit visual review")
    _sha(review["reviewed_gate_sha256"], field="Arrows reviewed gate SHA")

    devices = preflight["devices"]
    if not isinstance(devices, list) or len(devices) != EXPECTED_WORKER_COUNT:
        raise CanaryError("preflight must attest eight devices")
    logical, uuids = set(), set()
    for index, item in enumerate(devices):
        device = _exact_keys(
            item,
            {
                "worker_index",
                "logical_index",
                "physical_id",
                "uuid",
                "name",
                "memory_total_mib",
            },
            field=f"device[{index}]",
        )
        if device["worker_index"] != index:
            raise CanaryError("worker indices must be ordered and contiguous")
        logical.add(device["logical_index"])
        uuids.add(device["uuid"])
        if "H100" not in str(device["name"]):
            raise CanaryError("non-H100 device in canary report")
        if _finite_number(device["memory_total_mib"], field="device memory") <= 0:
            raise CanaryError("invalid device memory")
    if len(logical) != EXPECTED_WORKER_COUNT or len(uuids) != EXPECTED_WORKER_COUNT:
        raise CanaryError("worker/device binding is not one-to-one")

    telemetry = _exact_keys(
        top["utilization_probe"],
        {
            "status",
            "scientific_result",
            "batch_size",
            "physical_device_count",
            "lanes_per_device",
            "logical_worker_count",
            "devices",
            "companion_cells",
        },
        field="utilization_probe",
    )
    if telemetry["status"] != "passed" or telemetry["scientific_result"] is not True:
        raise CanaryError("GPU telemetry must come from the eight scientific cells")
    if (
        telemetry["batch_size"] != EXPECTED_BATCH_SIZE
        or telemetry["physical_device_count"] != EXPECTED_WORKER_COUNT
        or telemetry["lanes_per_device"] != EXPECTED_LANES_PER_DEVICE
        or telemetry["logical_worker_count"] != EXPECTED_PROCESS_COUNT
    ):
        raise CanaryError("telemetry batch or multiplexing contract mismatch")
    telemetry_devices = telemetry["devices"]
    if (
        not isinstance(telemetry_devices, list)
        or len(telemetry_devices) != EXPECTED_WORKER_COUNT
    ):
        raise CanaryError("telemetry must cover eight devices")
    telemetry_uuids = set()
    for item in telemetry_devices:
        device = _exact_keys(
            item,
            {
                "worker_index",
                "uuid",
                "samples",
                "mean_gpu_utilization_percent",
                "p50_gpu_utilization_percent",
                "p95_gpu_utilization_percent",
                "peak_memory_mib",
                "memory_total_mib",
                "examples_per_second",
            },
            field="telemetry device",
        )
        telemetry_uuids.add(device["uuid"])
        _positive_int(device["samples"], field="telemetry samples")
        for name in (
            "mean_gpu_utilization_percent",
            "p50_gpu_utilization_percent",
            "p95_gpu_utilization_percent",
            "peak_memory_mib",
            "memory_total_mib",
            "examples_per_second",
        ):
            value = _finite_number(device[name], field=f"telemetry.{name}")
            if value < 0:
                raise CanaryError(f"telemetry.{name} must be non-negative")
        if (
            device["examples_per_second"] <= 0
            or not 0 <= device["mean_gpu_utilization_percent"] <= 100
        ):
            raise CanaryError("telemetry contains impossible throughput/utilization")
        if (
            not 0 <= device["p50_gpu_utilization_percent"] <= 100
            or not 0 <= device["p95_gpu_utilization_percent"] <= 100
        ):
            raise CanaryError("telemetry percentile lies outside [0, 100]")
        if not 0 <= device["peak_memory_mib"] <= device["memory_total_mib"]:
            raise CanaryError("telemetry peak memory exceeds device memory")
    if telemetry_uuids != uuids:
        raise CanaryError("telemetry UUIDs differ from preflight bindings")

    companion_cells = telemetry["companion_cells"]
    if not isinstance(companion_cells, list) or len(companion_cells) != len(
        EXPECTED_COMPANION_CELL_IDS
    ):
        raise CanaryError("utilization probe must contain 16 production companions")
    for offset, item in enumerate(companion_cells, start=len(EXPECTED_CELL_IDS)):
        companion = _exact_keys(
            item,
            {
                "cell_id",
                "variant_id",
                "suite_id",
                "dataset",
                "representation",
                "status",
                "process_index",
                "physical_device_index",
                "input_sha256",
                "partition_sha256",
                "training_config_sha256",
                "checkpoint_sha256",
                "production_identity_sha256",
                "reusable_by_full_campaign",
                "production_wall_seconds",
                "artifacts",
            },
            field="utilization companion",
        )
        expected_id = EXPECTED_ALL_CELL_IDS[offset]
        variant, suite, dataset, representation = expected_id.split("/")
        if (
            companion["cell_id"] != expected_id
            or companion["variant_id"] != variant
            or companion["suite_id"] != suite
            or companion["dataset"] != dataset
            or companion["representation"] != representation
            or companion["status"] != "passed"
            or companion["process_index"] != offset
            or companion["physical_device_index"]
            != EXPECTED_PROCESS_DEVICE_INDICES[offset]
            or companion["reusable_by_full_campaign"] is not True
        ):
            raise CanaryError("utilization companion identity/status mismatch")
        for name in (
            "input_sha256",
            "partition_sha256",
            "training_config_sha256",
            "checkpoint_sha256",
            "production_identity_sha256",
        ):
            _sha(companion[name], field=f"companion.{name}")
        if (
            _finite_number(
                companion["production_wall_seconds"],
                field="companion production wall time",
            )
            <= 0
            or not isinstance(companion["artifacts"], list)
            or not companion["artifacts"]
        ):
            raise CanaryError("utilization companion evidence is incomplete")

    projection = _exact_keys(
        top["runtime_projection"],
        {
            "status",
            "physical_cell_count",
            "logical_worker_count",
            "physical_device_count",
            "lanes_per_device",
            "waves",
            "maximum_observed_cell_wall_seconds",
            "safety_factor",
            "projected_campaign_hours",
            "maximum_allowed_hours",
        },
        field="runtime_projection",
    )
    if (
        projection["status"] != "passed"
        or projection["physical_cell_count"] != 429
        or projection["logical_worker_count"] != EXPECTED_PROCESS_COUNT
        or projection["physical_device_count"] != EXPECTED_WORKER_COUNT
        or projection["lanes_per_device"] != EXPECTED_LANES_PER_DEVICE
        or projection["waves"] != math.ceil(429 / EXPECTED_PROCESS_COUNT)
        or projection["safety_factor"] != 2.0
        or projection["maximum_allowed_hours"] != 96.0
    ):
        raise CanaryError("runtime projection contract differs")
    maximum_wall = _finite_number(
        projection["maximum_observed_cell_wall_seconds"],
        field="runtime projection maximum wall time",
    )
    projected_hours = _finite_number(
        projection["projected_campaign_hours"],
        field="runtime projected hours",
    )
    expected_hours = (
        projection["waves"] * maximum_wall * projection["safety_factor"] / 3600.0
    )
    if maximum_wall <= 0 or not math.isclose(
        projected_hours, expected_hours, rel_tol=1.0e-12
    ):
        raise CanaryError("runtime projection is not derived from measured time")

    cells = top["cells"]
    if not isinstance(cells, list) or len(cells) != len(EXPECTED_CELL_IDS):
        raise CanaryError("report must contain exactly eight production cells")
    observed_ids = []
    for item in cells:
        cell = _exact_keys(
            item,
            {
                "cell_id",
                "variant_id",
                "family",
                "suite_id",
                "dataset",
                "representation",
                "status",
                "worker_index",
                "input_sha256",
                "partition_sha256",
                "preprocessing_sha256",
                "training_config_sha256",
                "checkpoint_sha256",
                "production_identity_sha256",
                "production_selected_lambda",
                "reusable_by_full_campaign",
                "production_preseal_wall_seconds",
                "native_loss",
                "reconstruction",
                "pointwise_lid",
                "reference_selector",
                "nf_scale_bin_nll",
                "trace_agreement",
                "artifacts",
            },
            field="cell",
        )
        observed_ids.append(cell["cell_id"])
        expected_id = (
            f"{cell['variant_id']}/{cell['suite_id']}/"
            f"{cell['dataset']}/{cell['representation']}"
        )
        if cell["cell_id"] != expected_id or cell["status"] != "passed":
            raise CanaryError("cell identity/status mismatch")
        if (
            f"{cell['suite_id']}/{cell['dataset']}/{cell['representation']}"
            not in EXPECTED_CELL_KEYS
        ):
            raise CanaryError("cell is outside the approved paired canary dataset")
        if cell["worker_index"] not in range(EXPECTED_WORKER_COUNT):
            raise CanaryError("cell worker index is invalid")
        for name in (
            "input_sha256",
            "partition_sha256",
            "preprocessing_sha256",
            "training_config_sha256",
            "checkpoint_sha256",
            "production_identity_sha256",
        ):
            _sha(cell[name], field=f"cell.{name}")
        production_selected_lambda = _finite_number(
            cell["production_selected_lambda"],
            field="cell.production_selected_lambda",
        )
        if production_selected_lambda <= 0:
            raise CanaryError("cell production selected lambda must be positive")
        if cell["reusable_by_full_campaign"] is not True:
            raise CanaryError("canary cell is not reusable by the full campaign")
        if (
            _finite_number(
                cell["production_preseal_wall_seconds"],
                field="cell.production_preseal_wall_seconds",
            )
            <= 0
        ):
            raise CanaryError("cell production pre-seal wall time is invalid")
        native_loss = cell["native_loss"]
        if (
            not isinstance(native_loss, Mapping)
            or native_loss.get("status") != "diagnostic_only"
            or native_loss.get("quality_gate_applied") is not False
        ):
            raise CanaryError(f"cell {cell['cell_id']} has invalid native loss gate")
        pointwise_artifact = _validate_pointwise_assessment(
            cell["pointwise_lid"],
            variant_id=cell["variant_id"],
            representation=cell["representation"],
        )
        reference_artifact = _validate_reference_selector_assessment(
            cell["reference_selector"],
            variant_id=cell["variant_id"],
            representation=cell["representation"],
        )
        trace = cell["trace_agreement"]
        if cell["variant_id"] == "scale_conditioned_nf":
            if trace != {"status": "not_applicable"}:
                raise CanaryError(
                    "NF trace assessment must be explicitly not applicable"
                )
            trace_artifact = None
        else:
            trace_artifact = _validate_trace_assessment(
                trace,
                representation=cell["representation"],
                production_selected_lambda=production_selected_lambda,
            )
        if cell["variant_id"] in {
            "vp_diffusion",
            "ve_diffusion",
            "posterior_log_noise_affine_flow",
        }:
            if (
                not isinstance(cell["reconstruction"], Mapping)
                or cell["reconstruction"].get("status") != "diagnostic_only"
                or cell["reconstruction"].get("quality_gate_applied") is not False
            ):
                raise CanaryError(
                    f"cell {cell['cell_id']} lacks reconstruction evidence"
                )
        elif cell["reconstruction"] != {"status": "not_applicable"}:
            raise CanaryError("NF reconstruction must be explicitly not applicable")
        if cell["variant_id"] == "scale_conditioned_nf":
            if (
                not isinstance(cell["nf_scale_bin_nll"], Mapping)
                or cell["nf_scale_bin_nll"].get("status") != "diagnostic_only"
                or cell["nf_scale_bin_nll"].get("quality_gate_applied") is not False
            ):
                raise CanaryError(f"cell {cell['cell_id']} lacks NF NLL-bin evidence")
        elif cell["nf_scale_bin_nll"] != {"status": "not_applicable"}:
            raise CanaryError("non-NF cell must mark NLL-bin gate not applicable")
        artifacts = cell["artifacts"]
        if not isinstance(artifacts, list) or not artifacts:
            raise CanaryError("cell must retain hashed reusable artifacts")
        artifact_fingerprints = {
            (
                Path(str(item.get("path", ""))).name,
                item.get("sha256"),
                item.get("size_bytes"),
            )
            for item in artifacts
            if isinstance(item, Mapping)
        }
        for embedded in (pointwise_artifact, reference_artifact, trace_artifact):
            if embedded is None:
                continue
            fingerprint = (
                Path(str(embedded["path"])).name,
                embedded["sha256"],
                embedded["size_bytes"],
            )
            if fingerprint not in artifact_fingerprints:
                raise CanaryError("cell diagnostic artifact is not retained by report")
        _validate_cell_report_artifact_binding(cell, report_root=root)
    if tuple(observed_ids) != EXPECTED_CELL_IDS:
        raise CanaryError("cell order/coverage differs from approved canary design")
    if len({cell["worker_index"] for cell in cells}) != EXPECTED_WORKER_COUNT:
        raise CanaryError("scientific cells were not bound one per H100 worker")
    if not math.isclose(
        maximum_wall,
        max(
            [float(cell["production_preseal_wall_seconds"]) for cell in cells]
            + [float(cell["production_wall_seconds"]) for cell in companion_cells]
        ),
        rel_tol=1.0e-12,
    ):
        raise CanaryError("runtime projection is not derived from canary cells")

    gates = top["gates"]
    if not isinstance(gates, list):
        raise CanaryError("gates must be a list")
    gate_ids = set()
    for gate in gates:
        record = _exact_keys(gate, {"id", "status", "evidence"}, field="gate")
        if record["id"] in gate_ids:
            raise CanaryError("duplicate gate id")
        gate_ids.add(record["id"])
        if record["status"] != "passed" or not isinstance(record["evidence"], Mapping):
            raise CanaryError(f"gate {record['id']} did not pass with evidence")
    if gate_ids != REQUIRED_GATE_IDS:
        raise CanaryError("required canary gate coverage mismatch")

    outputs = top["outputs"]
    if not isinstance(outputs, list) or not outputs:
        raise CanaryError("report must hash compact outputs")
    seen_paths = set()
    for item in outputs:
        output = _exact_keys(item, {"path", "sha256", "size_bytes"}, field="output")
        relative = Path(str(output["path"]))
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() in seen_paths
        ):
            raise CanaryError("output path must be unique and relative")
        seen_paths.add(relative.as_posix())
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise CanaryError("output escapes report directory") from exc
        if not path.is_file() or path.stat().st_size != output["size_bytes"]:
            raise CanaryError(f"missing or size-mismatched output: {relative}")
        _sha(output["sha256"], field="output.sha256")
        if file_sha256(path) != output["sha256"]:
            raise CanaryError(f"output hash mismatch: {relative}")
    if not any(
        item["path"] == contact_sheet["path"]
        and item["sha256"] == contact_sheet["sha256"]
        and item["size_bytes"] == contact_sheet["size_bytes"]
        for item in outputs
    ):
        raise CanaryError(
            "reviewed Arrows contact sheet is absent from output inventory"
        )
    output_by_path = {str(item["path"]): dict(item) for item in outputs}
    for owner in [*cells, *companion_cells]:
        for item in owner["artifacts"]:
            artifact = _exact_keys(
                item, {"path", "sha256", "size_bytes"}, field="cell artifact"
            )
            if output_by_path.get(str(artifact["path"])) != dict(artifact):
                raise CanaryError(
                    "cell artifact is absent from the hashed output inventory"
                )

    report_path = root / REPORT_FILENAME
    if report_path.exists() and REPORT_FILENAME in seen_paths:
        raise CanaryError("report cannot recursively hash itself")
    return dict(report)


def load_and_validate_canary_report(
    path: str | Path,
    **expected: Any,
) -> dict[str, Any]:
    report_path = Path(path).resolve()
    try:
        value = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CanaryError(f"cannot read canary report {report_path}") from exc
    return validate_canary_report(value, report_dir=report_path.parent, **expected)


def _git_source_record(root: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True
        )
        return completed.stdout.strip()

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=all")
    tree_listing = git("ls-tree", "-r", "--full-tree", "HEAD")
    if status:
        raise CanaryError("canary requires a clean worktree")
    try:
        from experiments.global_campaign_v2 import hash_declared_sources
    except (ImportError, AttributeError) as exc:
        raise CanaryError("v2 declared-source hasher is unavailable") from exc
    return {
        "git_commit": commit,
        "git_tree_sha256": hashlib.sha256(
            (tree_listing + "\n").encode("utf-8")
        ).hexdigest(),
        "declared_source_sha256": hash_declared_sources(root),
        "worktree_clean": True,
    }


def _validate_training_interface(config: Mapping[str, Any]) -> None:
    """Reject a launch until the v2 trainer exposes the declared contracts."""

    try:
        from models.training import evaluate_native_loss, field_parameter_count
    except (ImportError, AttributeError) as exc:
        raise CanaryError("v2 training interface is incomplete") from exc
    if not callable(evaluate_native_loss):
        raise CanaryError("evaluate_native_loss is not callable")
    if not callable(field_parameter_count):
        raise CanaryError("field_parameter_count is not callable")


def _nvidia_smi_rows(
    fields: Sequence[str], *, device: str | None = None
) -> list[list[str]]:
    command = ["nvidia-smi"]
    if device is not None:
        command.append(f"--id={device}")
    command.extend(
        [
            "--query-gpu=" + ",".join(fields),
            "--format=csv,noheader,nounits",
        ]
    )
    try:
        completed = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CanaryError("nvidia-smi query failed") from exc
    rows = [
        [field.strip() for field in line.split(",")]
        for line in completed.stdout.splitlines()
        if line.strip()
    ]
    if not rows or any(len(row) != len(fields) for row in rows):
        raise CanaryError("nvidia-smi returned malformed device data")
    return rows


def _visible_h100_devices() -> tuple[list[str], list[dict[str, Any]]]:
    rows = _nvidia_smi_rows(("index", "uuid", "name", "memory.total"))
    inventory = []
    for row in rows:
        try:
            memory_total = float(row[3])
        except ValueError as exc:
            raise CanaryError("nvidia-smi returned non-numeric memory") from exc
        inventory.append(
            {
                "physical_id": row[0],
                "uuid": row[1],
                "name": row[2],
                "memory_total_mib": memory_total,
            }
        )
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    tokens = (
        [str(index) for index in range(len(inventory))]
        if raw is None
        else [token.strip() for token in raw.split(",") if token.strip()]
    )
    if (
        len(tokens) != EXPECTED_WORKER_COUNT
        or len(set(tokens)) != EXPECTED_WORKER_COUNT
    ):
        raise CanaryError("canary requires exactly eight unique visible device tokens")

    selected: list[dict[str, Any]] = []
    used_uuids: set[str] = set()
    for worker_index, token in enumerate(tokens):
        matches = [
            item
            for item in inventory
            if token == item["physical_id"]
            or token == item["uuid"]
            or item["uuid"].startswith(token)
        ]
        if len(matches) != 1:
            raise CanaryError(f"visible CUDA token {token!r} is not a unique GPU")
        item = matches[0]
        if item["uuid"] in used_uuids or "H100" not in item["name"]:
            raise CanaryError(
                "canary visible devices are not eight unique full H100 GPUs"
            )
        used_uuids.add(item["uuid"])
        selected.append(
            {
                "worker_index": worker_index,
                "logical_index": worker_index,
                **item,
            }
        )
    return tokens, selected


class _DeviceSampler:
    def __init__(
        self,
        *,
        device: Mapping[str, Any],
        interval_seconds: float,
    ) -> None:
        self.device = dict(device)
        self.interval_seconds = interval_seconds
        self._samples: list[tuple[int, float, float, float]] = []
        self._errors: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise CanaryError("GPU sampler cannot be started twice")
        self._thread = threading.Thread(
            target=self._loop,
            name=f"canary-gpu-{self.device['worker_index']}",
            daemon=True,
        )
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                rows = _nvidia_smi_rows(
                    ("uuid", "utilization.gpu", "memory.used", "memory.total"),
                    device=str(self.device["uuid"]),
                )
                if len(rows) != 1 or rows[0][0] != self.device["uuid"]:
                    raise CanaryError("GPU telemetry UUID changed")
                self._samples.append(
                    (
                        time.time_ns(),
                        float(rows[0][1]),
                        float(rows[0][2]),
                        float(rows[0][3]),
                    )
                )
            except (CanaryError, ValueError) as exc:
                self._errors.append(str(exc))
            self._stop.wait(self.interval_seconds)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(10.0, self.interval_seconds * 3))
            if self._thread.is_alive():
                raise CanaryError("GPU sampler thread did not stop")

    def summarize(
        self,
        *,
        worker_index: int,
        start_ns: int,
        end_ns: int,
        processed_examples: int,
    ) -> dict[str, Any]:
        if end_ns <= start_ns or processed_examples <= 0:
            raise CanaryError("invalid training throughput interval")
        selected = [
            sample for sample in self._samples if start_ns <= sample[0] <= end_ns
        ]
        if len(selected) < 3:
            raise CanaryError(
                "fewer than three GPU samples overlap scientific training"
            )
        if self._errors:
            raise CanaryError("GPU telemetry contained query failures")
        utilization = [sample[1] for sample in selected]
        memory_used = [sample[2] for sample in selected]
        memory_total = [sample[3] for sample in selected]
        if any(value != memory_total[0] for value in memory_total):
            raise CanaryError("GPU total memory changed during canary")
        duration = (end_ns - start_ns) / 1_000_000_000
        return {
            "worker_index": worker_index,
            "uuid": str(self.device["uuid"]),
            "samples": len(selected),
            "mean_gpu_utilization_percent": statistics.fmean(utilization),
            "p50_gpu_utilization_percent": float(np.percentile(utilization, 50)),
            "p95_gpu_utilization_percent": float(np.percentile(utilization, 95)),
            "peak_memory_mib": max(memory_used),
            "memory_total_mib": memory_total[0],
            "examples_per_second": processed_examples / duration,
        }


class _MeasuredTrain:
    def __init__(self, sampler: _DeviceSampler, worker_index: int) -> None:
        self.sampler = sampler
        self.worker_index = worker_index

    def __call__(
        self,
        family: str,
        train_data: Any,
        selection_data: Any,
        config: Any,
        checkpoint_path: Path,
        log_callback: Any = None,
        *,
        progress_checkpoint_path: Path | None = None,
    ) -> Any:
        from models.training import train_model

        if (
            progress_checkpoint_path is not None
            and Path(progress_checkpoint_path).exists()
        ):
            raise CanaryError(
                "canary ETA requires a fresh full 0-to-128000-step training; "
                "partial progress cannot attest full-cell runtime"
            )
        first_step: int | None = None
        final_step: int | None = None

        def callback(payload: Mapping[str, Any]) -> None:
            nonlocal first_step, final_step
            step = payload.get("step")
            if type(step) is int:
                first_step = step if first_step is None else first_step
                final_step = step
            if log_callback is not None:
                log_callback(payload)

        start_ns = time.time_ns()
        trained = train_model(
            family,
            train_data,
            selection_data,
            config,
            checkpoint_path,
            callback,
            progress_checkpoint_path=progress_checkpoint_path,
        )
        end_ns = time.time_ns()
        if first_step is None or final_step is None:
            raise CanaryError("fixed-step trainer emitted no measured validation steps")
        interval = int(_field(_field(trained, "config"), "validation_interval_steps"))
        starting_step = max(0, first_step - interval)
        if starting_step != 0 or final_step != EXPECTED_TRAINING_STEPS:
            raise CanaryError(
                "canary ETA requires a measured full 0-to-128000-step training"
            )
        processed_examples = (final_step - starting_step) * EXPECTED_BATCH_SIZE
        runtime = self.sampler.summarize(
            worker_index=self.worker_index,
            start_ns=start_ns,
            end_ns=end_ns,
            processed_examples=processed_examples,
        )
        sidecar = {
            "schema_version": 1,
            "scientific_result": True,
            "batch_size": EXPECTED_BATCH_SIZE,
            "starting_step": starting_step,
            "final_step": final_step,
            "processed_examples": processed_examples,
            "training_start_time_ns": start_ns,
            "training_end_time_ns": end_ns,
            "device": runtime,
        }
        _atomic_json(
            Path(checkpoint_path).parent / "canary_training_runtime.json", sidecar
        )
        return trained


def _load_runtime_sidecar(
    checkpoint_path: Path, *, worker_index: int
) -> dict[str, Any]:
    path = checkpoint_path.parent / "canary_training_runtime.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CanaryError("canary training runtime sidecar is absent") from exc
    device = value.get("device") if isinstance(value, Mapping) else None
    if (
        not isinstance(value, Mapping)
        or value.get("schema_version") != 1
        or value.get("scientific_result") is not True
        or value.get("batch_size") != EXPECTED_BATCH_SIZE
        or value.get("starting_step") != 0
        or value.get("final_step") != EXPECTED_TRAINING_STEPS
        or value.get("processed_examples")
        != EXPECTED_TRAINING_STEPS * EXPECTED_BATCH_SIZE
        or not isinstance(device, Mapping)
        or device.get("worker_index") != worker_index
    ):
        raise CanaryError(
            "canary training runtime sidecar is not a full 0-to-128000-step timing"
        )
    start_ns = value.get("training_start_time_ns")
    end_ns = value.get("training_end_time_ns")
    if (
        isinstance(start_ns, bool)
        or not isinstance(start_ns, int)
        or isinstance(end_ns, bool)
        or not isinstance(end_ns, int)
        or start_ns <= 0
        or end_ns <= start_ns
    ):
        raise CanaryError("canary training runtime timestamps are invalid")
    duration_seconds = (end_ns - start_ns) / 1_000_000_000
    observed_throughput = _finite_number(
        device.get("examples_per_second"), field="training examples per second"
    )
    expected_throughput = value["processed_examples"] / duration_seconds
    if not math.isclose(observed_throughput, expected_throughput, rel_tol=1.0e-12):
        raise CanaryError("canary training runtime throughput is inconsistent")
    return dict(value)


def _load_or_create_preseal_runtime(
    output_dir: Path,
    *,
    cell_id: str,
    cell_identity_sha256: str,
    training_runtime: Mapping[str, Any],
    worker_start_ns: int,
    hook_end_ns: int | None = None,
) -> dict[str, Any]:
    runtime_path = output_dir / "preseal_runtime.json"
    training_start_ns = training_runtime.get("training_start_time_ns")
    training_end_ns = training_runtime.get("training_end_time_ns")
    if (
        isinstance(training_start_ns, bool)
        or not isinstance(training_start_ns, int)
        or isinstance(training_end_ns, bool)
        or not isinstance(training_end_ns, int)
        or training_start_ns <= 0
        or training_end_ns <= training_start_ns
    ):
        raise CanaryError("pre-seal runtime received invalid training timestamps")
    created = False
    if runtime_path.exists():
        try:
            value = json.loads(runtime_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CanaryError("pre-seal runtime record is unreadable") from exc
        record = _exact_keys(
            value,
            {
                "schema_version",
                "cell_id",
                "cell_identity_sha256",
                "timing_basis",
                "preseal_start_time_ns",
                "training_start_time_ns",
                "training_end_time_ns",
                "preseal_end_time_ns",
                "production_preseal_wall_seconds",
            },
            field="pre-seal runtime record",
        )
        if (
            record["schema_version"] != 2
            or record["cell_id"] != cell_id
            or record["cell_identity_sha256"] != cell_identity_sha256
            or record["timing_basis"] != "full_training_start_through_preseal_hook_v1"
            or record["training_start_time_ns"] != training_start_ns
            or record["training_end_time_ns"] != training_end_ns
        ):
            raise CanaryError("pre-seal runtime identity or training binding differs")
    else:
        end_ns = time.time_ns() if hook_end_ns is None else hook_end_ns
        start_ns = min(worker_start_ns, training_start_ns)
        record = {
            "schema_version": 2,
            "cell_id": cell_id,
            "cell_identity_sha256": cell_identity_sha256,
            "timing_basis": "full_training_start_through_preseal_hook_v1",
            "preseal_start_time_ns": start_ns,
            "training_start_time_ns": training_start_ns,
            "training_end_time_ns": training_end_ns,
            "preseal_end_time_ns": end_ns,
            "production_preseal_wall_seconds": (end_ns - start_ns) / 1_000_000_000,
        }
        created = True
    preseal_start_ns = record["preseal_start_time_ns"]
    preseal_end_ns = record["preseal_end_time_ns"]
    for name, value in (
        ("preseal_start_time_ns", preseal_start_ns),
        ("preseal_end_time_ns", preseal_end_ns),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise CanaryError(f"pre-seal runtime {name} is invalid")
    if not (preseal_start_ns <= training_start_ns < training_end_ns <= preseal_end_ns):
        raise CanaryError(
            "pre-seal runtime does not contain the full training interval"
        )
    observed_seconds = _finite_number(
        record["production_preseal_wall_seconds"],
        field="production pre-seal wall time",
    )
    expected_seconds = (preseal_end_ns - preseal_start_ns) / 1_000_000_000
    if observed_seconds <= 0 or not math.isclose(
        observed_seconds, expected_seconds, rel_tol=1.0e-12
    ):
        raise CanaryError("pre-seal runtime duration is inconsistent")
    if created:
        _atomic_json(runtime_path, record)
    return dict(record)


def _canary_worker_main(
    process_index: int,
    cell_id: str,
    quality_worker_index: int | None,
    device_token: str,
    device: Mapping[str, Any],
    prepared: Any,
    config: Mapping[str, Any],
    start_event: Any,
    result_queue: Any,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = device_token
    os.environ["LID_WORKER_SLOT"] = str(process_index)
    try:
        import torch

        from experiments.global_parallel import ParallelDependencies
        from experiments.global_parallel_v2 import run_prepared_v2_cell

        if (
            not torch.cuda.is_available()
            or torch.cuda.device_count() != 1
            or "H100" not in torch.cuda.get_device_name(0)
        ):
            raise CanaryError("worker does not see exactly one H100 CUDA device")
        torch.cuda.set_device(0)
        visible = f"{device_token}:{torch.cuda.get_device_name(0)}"
        os.environ["LID_VISIBLE_DEVICE"] = visible
        result_queue.put(
            {
                "kind": "ready",
                "process_index": process_index,
                "cell_id": cell_id,
                "visible_device": visible,
                "torch_version": torch.__version__,
            }
        )
        if not start_event.wait(timeout=120):
            raise CanaryError("worker start barrier timed out")
        cell_start_ns = time.time_ns()
        sampler = None
        dependencies = ParallelDependencies()
        hook = None
        if quality_worker_index is not None:
            sampler = _DeviceSampler(
                device=device,
                interval_seconds=float(
                    config["execution"]["telemetry_interval_seconds"]
                ),
            )
            sampler.start()
            measured_train = _MeasuredTrain(sampler, quality_worker_index)
            dependencies = ParallelDependencies(train_fn=measured_train)

            def quality_hook(
                context: Mapping[str, Any], output_dir: Path
            ) -> Mapping[str, Any]:
                evidence = dict(CanaryDiagnostics(config)(context, output_dir))
                training_runtime = _load_runtime_sidecar(
                    Path(_field(context, "checkpoint_path")),
                    worker_index=quality_worker_index,
                )
                evidence["training_runtime"] = training_runtime
                runtime_record = _load_or_create_preseal_runtime(
                    output_dir,
                    cell_id=str(evidence["cell_id"]),
                    cell_identity_sha256=str(evidence["cell_identity_sha256"]),
                    training_runtime=training_runtime,
                    worker_start_ns=cell_start_ns,
                )
                evidence["production_preseal_wall_seconds"] = runtime_record[
                    "production_preseal_wall_seconds"
                ]
                return evidence

            hook = quality_hook

        def event_callback(event: str, payload: Mapping[str, Any]) -> None:
            result_queue.put(
                {
                    "kind": "event",
                    "process_index": process_index,
                    "event": event,
                    "payload": _json_safe(payload),
                }
            )

        variant, suite, dataset, representation = cell_id.split("/")
        try:
            result = run_prepared_v2_cell(
                prepared,
                model_variant=variant,
                cell_key=f"{suite}/{dataset}/{representation}",
                dependencies=dependencies,
                event_callback=event_callback,
                pre_seal_hook=hook,
            )
        finally:
            if sampler is not None:
                sampler.stop()
        cell_end_ns = time.time_ns()
        result_queue.put(
            {
                "kind": "completed",
                "process_index": process_index,
                "cell_id": cell_id,
                "quality_worker_index": quality_worker_index,
                "physical_device_index": int(device["worker_index"]),
                "production_wall_seconds": (cell_end_ns - cell_start_ns)
                / 1_000_000_000,
                "result": result,
            }
        )
    except BaseException as exc:  # noqa: BLE001 - process boundary must report all
        result_queue.put(
            {
                "kind": "failed",
                "process_index": process_index,
                "cell_id": cell_id,
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


def _run_multiplexed_cells(
    prepared: Any,
    config: Mapping[str, Any],
    *,
    device_tokens: Sequence[str],
    devices: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    context = mp.get_context("spawn")
    start_event = context.Event()
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_canary_worker_main,
            name=f"v2-canary-{index}",
            args=(
                index,
                EXPECTED_ALL_CELL_IDS[index],
                index if index < len(EXPECTED_CELL_IDS) else None,
                device_tokens[EXPECTED_PROCESS_DEVICE_INDICES[index]],
                devices[EXPECTED_PROCESS_DEVICE_INDICES[index]],
                prepared,
                config,
                start_event,
                result_queue,
            ),
        )
        for index in range(EXPECTED_PROCESS_COUNT)
    ]
    for process in processes:
        process.start()
    ready: set[int] = set()
    completed: dict[int, dict[str, Any]] = {}
    try:
        while len(ready) < EXPECTED_PROCESS_COUNT:
            try:
                message = result_queue.get(timeout=120)
            except queue.Empty as exc:
                raise CanaryError("eight-H100 worker preflight timed out") from exc
            if message.get("kind") == "failed":
                raise CanaryError(
                    f"canary process {message.get('process_index')} failed preflight: "
                    f"{message.get('exception_type')}: {message.get('message')}"
                )
            if message.get("kind") != "ready":
                raise CanaryError("worker emitted data before the start barrier")
            worker = int(message["process_index"])
            if (
                worker in ready
                or message.get("cell_id") != EXPECTED_ALL_CELL_IDS[worker]
                or message.get("torch_version") != "2.7.1+cu126"
            ):
                raise CanaryError("worker preflight identity/runtime differs")
            ready.add(worker)
        start_event.set()
        while len(completed) < EXPECTED_PROCESS_COUNT:
            try:
                message = result_queue.get(timeout=60)
            except queue.Empty:
                exited = [
                    process
                    for process in processes
                    if process.exitcode is not None and process.exitcode != 0
                ]
                if exited:
                    raise CanaryError("canary worker exited without a result")
                continue
            kind = message.get("kind")
            if kind == "event":
                print(
                    json.dumps(
                        {
                            "worker": message["process_index"],
                            "event": message["event"],
                            "payload": message["payload"],
                        },
                        allow_nan=False,
                    ),
                    flush=True,
                )
                continue
            if kind == "failed":
                raise CanaryError(
                    f"canary process {message.get('process_index')} failed: "
                    f"{message.get('exception_type')}: {message.get('message')}\n"
                    f"{message.get('traceback', '')}"
                )
            if kind != "completed":
                raise CanaryError("canary worker emitted an unknown message")
            worker = int(message["process_index"])
            if worker in completed:
                raise CanaryError("canary worker completed twice")
            completed[worker] = dict(message)
        for process in processes:
            process.join(timeout=30)
        if any(process.exitcode != 0 for process in processes):
            raise CanaryError("one or more canary workers exited non-zero")
    except BaseException:
        start_event.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        raise
    return [completed[index] for index in range(EXPECTED_PROCESS_COUNT)]


def _assert_expected_identity(name: str, actual: str) -> None:
    expected = os.environ.get(name)
    if expected is None or _SHA256.fullmatch(expected) is None:
        raise CanaryError(f"{name} must contain an exact lowercase SHA-256")
    if actual != expected:
        raise CanaryError(f"{name} differs from prepared v2 campaign")


def _assert_fresh_physical_canary_cells(*, campaign_root: Path, prepared: Any) -> None:
    """Require fresh storage for all 24 physical canary trainings.

    Logical readouts may share one sealed training, but timing and telemetry are
    valid only when every physical canary cell starts without a reusable final
    directory or its stable resume directory.
    """

    try:
        from experiments.global_campaign_v2 import _expected_cell_directory
    except (ImportError, AttributeError) as exc:
        raise CanaryError(
            "exact v2 production cell paths are unavailable; refusing canary launch"
        ) from exc

    plans = tuple(_field(prepared, "plans"))
    cells = tuple(_field(prepared, "cells"))
    plan_indices = {
        str(_field(plan, "variant_id")): index for index, plan in enumerate(plans)
    }
    cell_indices = {str(_field(cell, "key")): index for index, cell in enumerate(cells)}
    if len(plan_indices) != len(plans) or len(cell_indices) != len(cells):
        raise CanaryError("v2 production matrix contains duplicate plan or cell keys")

    expected_tasks = []
    for cell_id in EXPECTED_ALL_CELL_IDS:
        variant, suite, dataset, representation = cell_id.split("/")
        expected_tasks.append((variant, f"{suite}/{dataset}/{representation}"))
    if (
        len(expected_tasks) != EXPECTED_PROCESS_COUNT
        or len(set(expected_tasks)) != EXPECTED_PROCESS_COUNT
    ):
        raise CanaryError("internal physical canary cell inventory differs")

    root = Path(campaign_root).resolve()
    locations: dict[Path, str] = {}
    conflicts: list[Path] = []
    for variant, cell_key in expected_tasks:
        try:
            model_index = plan_indices[variant]
            cell_index = cell_indices[cell_key]
        except KeyError as exc:
            raise CanaryError(
                f"physical canary cell is absent from production matrix: "
                f"{variant}/{cell_key}"
            ) from exc
        final_dir, _ = _expected_cell_directory(
            campaign_root=root,
            prepared=prepared,
            model_index=model_index,
            cell_index=cell_index,
        )
        final_dir = Path(final_dir)
        try:
            final_dir.relative_to(root)
        except ValueError as exc:
            raise CanaryError(
                "exact production canary cell path escapes campaign root"
            ) from exc
        incomplete_dir = final_dir.with_name(f".{final_dir.name}.incomplete")
        for location in (final_dir, incomplete_dir):
            if location in locations:
                raise CanaryError(
                    "physical canary cells resolve to a duplicate production path"
                )
            locations[location] = f"{variant}/{cell_key}"
            if os.path.lexists(location):
                conflicts.append(location)

    if len(locations) != 2 * EXPECTED_PROCESS_COUNT:
        raise CanaryError(
            "physical canary path coverage differs from the 24-cell design"
        )
    if conflicts:
        rendered = [
            path.relative_to(root).as_posix()
            if path.is_relative_to(root)
            else str(path)
            for path in conflicts
        ]
        raise CanaryError(
            "canary report is absent but exact physical canary cell storage already "
            f"exists: {rendered}; use a fresh output root so reused sealed or partial "
            "cells cannot undercount canary timing/telemetry"
        )


def _copy_exact(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise CanaryError(f"required evidence file is absent: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or file_sha256(destination) != file_sha256(source):
            raise CanaryError(f"existing evidence differs: {destination}")
        return
    shutil.copyfile(source, destination)
    if file_sha256(destination) != file_sha256(source):
        raise CanaryError(f"copied evidence hash differs: {destination}")


def _materialize_reviewed_arrows(
    reviewed_path: Path,
    *,
    campaign_root: Path,
    source: Mapping[str, Any],
    config_sha256: str,
) -> tuple[dict[str, Any], list[Path]]:
    reviewed = load_reviewed_arrows_gate(
        reviewed_path,
        expected_source=source,
        expected_config_sha256=config_sha256,
    )
    destination = campaign_root / "canary_evidence" / "arrows"
    source_gate = reviewed_path.parent / str(reviewed["gate"]["path"])
    source_contact = reviewed_path.parent / str(reviewed["contact_sheet"]["path"])
    copied_gate = destination / source_gate.name
    copied_contact = destination / source_contact.name
    copied_reviewed = destination / reviewed_path.name
    for source_path, destination_path in (
        (source_gate, copied_gate),
        (source_contact, copied_contact),
        (reviewed_path, copied_reviewed),
    ):
        _copy_exact(source_path, destination_path)
    load_reviewed_arrows_gate(
        copied_reviewed,
        expected_source=source,
        expected_config_sha256=config_sha256,
    )
    contact_record = {
        "path": copied_contact.relative_to(campaign_root).as_posix(),
        "sha256": file_sha256(copied_contact),
        "size_bytes": copied_contact.stat().st_size,
    }
    arrows = dict(reviewed["arrows"])
    arrows["status"] = "passed"
    arrows["contact_sheet"] = contact_record
    arrows["human_review"] = {
        "status": "reviewed",
        "reviewer": str(reviewed["reviewer"]),
        "reviewed_at": str(reviewed["reviewed_at"]),
        "reviewed_gate_sha256": file_sha256(copied_reviewed),
    }
    return arrows, [copied_gate, copied_contact, copied_reviewed]


def _output_record(path: Path, *, root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise CanaryError(f"canary output escapes campaign root: {path}") from exc
    if not resolved.is_file():
        raise CanaryError(f"canary output is absent: {path}")
    return {
        "path": relative.as_posix(),
        "sha256": file_sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _cell_report(
    result: Mapping[str, Any],
    *,
    worker_index: int,
    campaign_root: Path,
) -> tuple[dict[str, Any], list[Path]]:
    from experiments.global_campaign_v2 import validate_global_cell

    directory = Path(str(result.get("directory", ""))).resolve()
    try:
        directory.relative_to(campaign_root.resolve())
    except ValueError as exc:
        raise CanaryError(
            "production cell was sealed outside the v2 campaign root"
        ) from exc
    if not directory.is_dir():
        raise CanaryError("production canary cell directory is absent")
    errors = validate_global_cell(directory)
    if errors:
        raise CanaryError(f"sealed production canary cell is invalid: {errors}")
    try:
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        identity = json.loads((directory / "identity.json").read_text(encoding="utf-8"))
        attestation = json.loads(
            (directory / "training_attestation.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise CanaryError("sealed production cell evidence is unreadable") from exc
    quality = summary.get("quality_diagnostics")
    if not isinstance(quality, Mapping) or quality.get("status") != "passed":
        raise CanaryError("sealed production cell lacks passing canary diagnostics")
    expected_cell_id = EXPECTED_CELL_IDS[worker_index]
    variant, suite, dataset, representation = expected_cell_id.split("/")
    if (
        summary.get("model_variant") != variant
        or summary.get("suite_id") != suite
        or summary.get("dataset") != dataset
        or summary.get("representation") != representation
        or quality.get("cell_id") != expected_cell_id
    ):
        raise CanaryError(
            "sealed production cell identity differs from worker assignment"
        )
    production_selected_lambda = _finite_number(
        quality.get("production_selected_lambda"),
        field="quality production selected lambda",
    )
    summary_selected_lambda = _finite_number(
        summary.get("selected_scale"), field="summary selected scale"
    )
    if (
        production_selected_lambda <= 0
        or summary_selected_lambda != production_selected_lambda
    ):
        raise CanaryError(
            "quality diagnostics are not bound to production selected scale"
        )
    trace_quality = quality.get("trace_agreement")
    if variant != "scale_conditioned_nf" and (
        not isinstance(trace_quality, Mapping)
        or _finite_number(trace_quality.get("lambda"), field="trace lambda")
        != production_selected_lambda
    ):
        raise CanaryError("trace assessment is not bound to production selected scale")
    runtime = quality.get("training_runtime")
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("scientific_result") is not True
        or runtime.get("batch_size") != EXPECTED_BATCH_SIZE
        or runtime.get("starting_step") != 0
        or runtime.get("final_step") != EXPECTED_TRAINING_STEPS
        or runtime.get("processed_examples")
        != EXPECTED_TRAINING_STEPS * EXPECTED_BATCH_SIZE
    ):
        raise CanaryError("sealed production cell lacks scientific runtime telemetry")
    runtime_device = runtime.get("device")
    if (
        not isinstance(runtime_device, Mapping)
        or runtime_device.get("worker_index") != worker_index
    ):
        raise CanaryError("sealed production cell telemetry worker differs")
    partition = summary.get("partition")
    partition_sha = (
        partition.get("partition_sha256") if isinstance(partition, Mapping) else None
    )
    artifacts = [
        directory / "identity.json",
        directory / "manifest.json",
        directory / "summary.json",
        directory / "training_attestation.json",
        directory / "canary_training_runtime.json",
        directory / "quality_diagnostics.json",
    ]
    diagnostics_dir = directory / "quality_diagnostics"
    if not diagnostics_dir.is_dir():
        raise CanaryError("sealed production cell diagnostics directory is absent")
    artifacts.extend(
        sorted(path for path in diagnostics_dir.iterdir() if path.is_file())
    )
    records = [_output_record(path, root=campaign_root) for path in artifacts]
    cell = {
        "cell_id": expected_cell_id,
        "variant_id": variant,
        "family": str(quality.get("family")),
        "suite_id": suite,
        "dataset": dataset,
        "representation": representation,
        "status": "passed",
        "worker_index": worker_index,
        "input_sha256": _sha(summary.get("input_sha256"), field="cell input"),
        "partition_sha256": _sha(partition_sha, field="cell partition"),
        "preprocessing_sha256": _sha(
            quality.get("preprocessing_sha256"), field="cell preprocessing"
        ),
        "training_config_sha256": _sha(
            attestation.get("training_config_sha256"), field="cell training config"
        ),
        "checkpoint_sha256": _sha(
            summary.get("checkpoint_sha256"), field="cell checkpoint"
        ),
        "production_identity_sha256": canonical_sha256(identity),
        "production_selected_lambda": production_selected_lambda,
        "reusable_by_full_campaign": True,
        "production_preseal_wall_seconds": _finite_number(
            quality.get("production_preseal_wall_seconds"),
            field="production pre-seal wall time",
        ),
        "native_loss": dict(quality["native_loss"]),
        "reconstruction": dict(quality["reconstruction"]),
        "pointwise_lid": dict(quality["pointwise_lid"]),
        "reference_selector": dict(quality["reference_selector"]),
        "nf_scale_bin_nll": dict(quality["nf_scale_bin_nll"]),
        "trace_agreement": dict(quality["trace_agreement"]),
        "artifacts": records,
    }
    return cell, artifacts


def _companion_cell_report(
    message: Mapping[str, Any],
    *,
    process_index: int,
    campaign_root: Path,
) -> tuple[dict[str, Any], list[Path]]:
    from experiments.global_campaign_v2 import validate_global_cell

    result = _field(message, "result")
    if not isinstance(result, Mapping):
        raise CanaryError("companion process result is not a mapping")
    directory = Path(str(result.get("directory", ""))).resolve()
    try:
        directory.relative_to(campaign_root.resolve())
    except ValueError as exc:
        raise CanaryError(
            "companion cell was sealed outside the campaign root"
        ) from exc
    errors = validate_global_cell(directory)
    if errors:
        raise CanaryError(f"sealed utilization companion is invalid: {errors}")
    try:
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        identity = json.loads((directory / "identity.json").read_text(encoding="utf-8"))
        attestation = json.loads(
            (directory / "training_attestation.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise CanaryError("sealed utilization companion is unreadable") from exc
    expected_id = EXPECTED_ALL_CELL_IDS[process_index]
    variant, suite, dataset, representation = expected_id.split("/")
    if (
        process_index < len(EXPECTED_CELL_IDS)
        or message.get("cell_id") != expected_id
        or message.get("quality_worker_index") is not None
        or message.get("physical_device_index")
        != EXPECTED_PROCESS_DEVICE_INDICES[process_index]
        or summary.get("model_variant") != variant
        or summary.get("suite_id") != suite
        or summary.get("dataset") != dataset
        or summary.get("representation") != representation
        or summary.get("quality_diagnostics") != {"status": "not_requested"}
    ):
        raise CanaryError("utilization companion identity differs from assignment")
    wall_seconds = _finite_number(
        message.get("production_wall_seconds"), field="companion production wall time"
    )
    if wall_seconds <= 0:
        raise CanaryError("companion production wall time must be positive")
    partition = summary.get("partition")
    partition_sha = (
        partition.get("partition_sha256") if isinstance(partition, Mapping) else None
    )
    artifacts = sorted(path for path in directory.rglob("*") if path.is_file())
    if not artifacts:
        raise CanaryError("sealed utilization companion has no artifacts")
    return (
        {
            "cell_id": expected_id,
            "variant_id": variant,
            "suite_id": suite,
            "dataset": dataset,
            "representation": representation,
            "status": "passed",
            "process_index": process_index,
            "physical_device_index": int(message["physical_device_index"]),
            "input_sha256": _sha(summary.get("input_sha256"), field="companion input"),
            "partition_sha256": _sha(partition_sha, field="companion partition"),
            "training_config_sha256": _sha(
                attestation.get("training_config_sha256"),
                field="companion training config",
            ),
            "checkpoint_sha256": _sha(
                summary.get("checkpoint_sha256"), field="companion checkpoint"
            ),
            "production_identity_sha256": canonical_sha256(identity),
            "reusable_by_full_campaign": True,
            "production_wall_seconds": wall_seconds,
            "artifacts": [
                _output_record(path, root=campaign_root) for path in artifacts
            ],
        },
        artifacts,
    )


def _gate_records(
    cells: Sequence[Mapping[str, Any]],
    *,
    companion_cells: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
    archive_sha256: str,
    arrows: Mapping[str, Any],
    devices: Sequence[Mapping[str, Any]],
    telemetry: Sequence[Mapping[str, Any]],
    runtime_projection: Mapping[str, Any],
    outputs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    evidence: dict[str, Mapping[str, Any]] = {
        "source_provenance": {
            "git_commit": source["git_commit"],
            "git_tree_sha256": source["git_tree_sha256"],
            "declared_source_sha256": source["declared_source_sha256"],
        },
        "canonical_archive": {"sha256": archive_sha256},
        "arrows_data_preprocessing": {
            "sample_images_sha256": arrows["sample_images_sha256"],
            "production_partition_sha256": arrows["production_partition_sha256"],
            "preprocessing_sha256": arrows["preprocessing_sha256"],
            "contact_sheet_sha256": arrows["contact_sheet"]["sha256"],
            "reviewed_gate_sha256": arrows["human_review"]["reviewed_gate_sha256"],
        },
        "eight_h100_workers": {
            "physical_device_count": len(devices),
            "lanes_per_device": EXPECTED_LANES_PER_DEVICE,
            "logical_worker_count": EXPECTED_PROCESS_COUNT,
            "uuids": [row["uuid"] for row in devices],
        },
        "batch_256_contract": {
            "batch_size": EXPECTED_BATCH_SIZE,
            "quality_cells": len(cells),
            "companion_cells": len(companion_cells),
        },
        "gpu_utilization": {"devices": list(telemetry)},
        "runtime_projection": dict(runtime_projection),
        "native_loss_quality": {
            "cell_evidence": [cell["native_loss"] for cell in cells]
        },
        "reconstruction_quality": {
            "cell_evidence": [cell["reconstruction"] for cell in cells]
        },
        "pointwise_lid_assessment": {
            "cell_evidence": [cell["pointwise_lid"] for cell in cells]
        },
        "reference_selector_quality": {
            "cell_evidence": [cell["reference_selector"] for cell in cells]
        },
        "nf_scale_bin_nll": {
            "cell_evidence": [cell["nf_scale_bin_nll"] for cell in cells]
        },
        "trace_estimator_assessment": {
            "cell_evidence": [cell["trace_agreement"] for cell in cells]
        },
        "production_cell_reuse": {
            "production_identity_sha256": [
                cell["production_identity_sha256"] for cell in cells
            ],
            "companion_identity_sha256": [
                cell["production_identity_sha256"] for cell in companion_cells
            ],
        },
        "output_integrity": {
            "output_count": len(outputs),
            "inventory_sha256": canonical_sha256(outputs),
        },
    }
    if set(evidence) != REQUIRED_GATE_IDS:
        raise CanaryError("internal canary gate inventory differs")
    return [
        {"id": gate_id, "status": "passed", "evidence": dict(evidence[gate_id])}
        for gate_id in sorted(REQUIRED_GATE_IDS)
    ]


def _run(config: Mapping[str, Any], *, config_path: Path) -> dict[str, Any]:
    """Execute preflight and the 24 exact physical production cells.

    The full-cell materialization/reuse API is intentionally supplied by the
    v2 campaign module.  Import failure is a hard gate: running an independent
    look-alike trainer here would create duplicate, non-reusable checkpoints.
    """

    validated = validate_canary_config(config)
    root = Path(__file__).resolve().parents[1]
    output_parent = _resolve(root, validated["output_root"], field="output_root")
    output_parent.mkdir(parents=True, exist_ok=True)
    source = _git_source_record(root)
    _validate_training_interface(validated)
    try:
        from experiments.global_parallel_v2 import prepare_global_v2_campaign
    except (ImportError, AttributeError) as exc:
        raise CanaryError(
            "global v2 production-cell reuse interface is unavailable; refusing "
            "to train duplicate canary checkpoints"
        ) from exc
    prepared = prepare_global_v2_campaign(root=root, output_root=output_parent)
    campaign_root = Path(prepared.campaign_root).resolve()
    _assert_expected_identity(
        "LID_V2_EXPECTED_CAMPAIGN_IDENTITY", prepared.campaign_identity
    )
    _assert_expected_identity(
        "LID_V2_EXPECTED_CAMPAIGN_CONFIG_SHA256", prepared.config_sha
    )
    _assert_expected_identity(
        "LID_V2_EXPECTED_INPUT_INVENTORY_SHA256", prepared.input_inventory_sha
    )
    _assert_expected_identity(
        "LID_V2_EXPECTED_DECLARED_SOURCE_SHA256", prepared.source_sha
    )
    if prepared.source_sha != source["declared_source_sha256"]:
        raise CanaryError("prepared v2 source differs from canary source attestation")
    report_path = campaign_root / REPORT_FILENAME
    expected = {
        "expected_commit": source["git_commit"],
        "expected_git_tree_sha256": source["git_tree_sha256"],
        "expected_declared_source_sha256": prepared.source_sha,
        "expected_config_sha256": file_sha256(config_path),
        "expected_campaign_identity": prepared.campaign_identity,
        "expected_campaign_config_sha256": prepared.config_sha,
        "expected_input_inventory_sha256": prepared.input_inventory_sha,
    }
    if report_path.exists():
        report = load_and_validate_canary_report(report_path, **expected)
        print(
            json.dumps({"status": "passed", "report": report_path.as_posix()}),
            flush=True,
        )
        return report

    _assert_fresh_physical_canary_cells(
        campaign_root=campaign_root,
        prepared=prepared,
    )

    archive = _resolve(root, validated["data"]["archive"], field="data.archive")
    archive_sha = file_sha256(archive)
    if archive_sha != validated["data"]["archive_sha256"]:
        raise CanaryError("canonical benchmark archive hash differs")
    reviewed_value = validated["data"]["reviewed_arrows_gate"]
    if not reviewed_value:
        raise CanaryError("full canary requires a pre-reviewed Arrows data gate")
    reviewed_path = _resolve(root, reviewed_value, field="data.reviewed_arrows_gate")
    arrows, arrow_outputs = _materialize_reviewed_arrows(
        reviewed_path,
        campaign_root=campaign_root,
        source=source,
        config_sha256=file_sha256(config_path),
    )
    device_tokens, devices = _visible_h100_devices()
    results = _run_multiplexed_cells(
        prepared,
        validated,
        device_tokens=device_tokens,
        devices=devices,
    )
    quality_results = results[: len(EXPECTED_CELL_IDS)]
    companion_results = results[len(EXPECTED_CELL_IDS) :]
    cells: list[dict[str, Any]] = []
    companion_cells: list[dict[str, Any]] = []
    artifact_paths: list[Path] = list(arrow_outputs)
    for worker_index, message in enumerate(quality_results):
        cell, cell_artifacts = _cell_report(
            _field(message, "result"),
            worker_index=worker_index,
            campaign_root=campaign_root,
        )
        cells.append(cell)
        artifact_paths.extend(cell_artifacts)
    for offset, message in enumerate(companion_results, start=len(EXPECTED_CELL_IDS)):
        companion, companion_artifacts = _companion_cell_report(
            message,
            process_index=offset,
            campaign_root=campaign_root,
        )
        companion_cells.append(companion)
        artifact_paths.extend(companion_artifacts)
    for cell in cells:
        if cell["dataset"] != "e2_arrows":
            continue
        if (
            cell["partition_sha256"] != arrows["production_partition_sha256"]
            or cell["preprocessing_sha256"] != arrows["preprocessing_sha256"]
        ):
            raise CanaryError(
                "Arrows training partition/preprocessing differs from reviewed data gate"
            )
    telemetry = [
        dict(
            cell["artifacts"]
            and cell_result["result"]["summary"]["quality_diagnostics"][
                "training_runtime"
            ]["device"]
        )
        for cell, cell_result in zip(cells, quality_results)
    ]
    gates = validated["gates"]
    maximum_wall_seconds = max(
        [float(cell["production_preseal_wall_seconds"]) for cell in cells]
        + [float(cell["production_wall_seconds"]) for cell in companion_cells]
    )
    waves = math.ceil(429 / EXPECTED_PROCESS_COUNT)
    safety_factor = float(gates["runtime_projection_safety_factor"])
    projected_hours = waves * maximum_wall_seconds * safety_factor / 3600.0
    maximum_allowed_hours = float(gates["maximum_projected_campaign_hours"])
    runtime_projection = {
        "status": "passed",
        "physical_cell_count": 429,
        "logical_worker_count": EXPECTED_PROCESS_COUNT,
        "physical_device_count": EXPECTED_WORKER_COUNT,
        "lanes_per_device": EXPECTED_LANES_PER_DEVICE,
        "waves": waves,
        "maximum_observed_cell_wall_seconds": maximum_wall_seconds,
        "safety_factor": safety_factor,
        "projected_campaign_hours": projected_hours,
        "maximum_allowed_hours": maximum_allowed_hours,
    }
    artifact_paths.append(campaign_root / "input_inventory.json")
    unique_paths = sorted({path.resolve() for path in artifact_paths})
    outputs = [_output_record(path, root=campaign_root) for path in unique_paths]
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "passed",
        "source": source,
        "config": {
            "path": config_path.resolve().relative_to(root.resolve()).as_posix(),
            "sha256": file_sha256(config_path),
            "batch_size": EXPECTED_BATCH_SIZE,
            "physical_device_count": EXPECTED_WORKER_COUNT,
            "lanes_per_device": EXPECTED_LANES_PER_DEVICE,
            "logical_worker_count": EXPECTED_PROCESS_COUNT,
            "campaign_identity": prepared.campaign_identity,
            "campaign_config_sha256": prepared.config_sha,
            "input_inventory_sha256": prepared.input_inventory_sha,
        },
        "preflight": {
            "status": "passed",
            "archive_sha256": archive_sha,
            "arrows": arrows,
            "devices": devices,
        },
        "utilization_probe": {
            "status": "passed",
            "scientific_result": True,
            "batch_size": EXPECTED_BATCH_SIZE,
            "physical_device_count": EXPECTED_WORKER_COUNT,
            "lanes_per_device": EXPECTED_LANES_PER_DEVICE,
            "logical_worker_count": EXPECTED_PROCESS_COUNT,
            "devices": telemetry,
            "companion_cells": companion_cells,
        },
        "runtime_projection": runtime_projection,
        "cells": cells,
        "gates": _gate_records(
            cells,
            companion_cells=companion_cells,
            source=source,
            archive_sha256=archive_sha,
            arrows=arrows,
            devices=devices,
            telemetry=telemetry,
            runtime_projection=runtime_projection,
            outputs=outputs,
        ),
        "outputs": outputs,
        "failures": [],
    }
    _atomic_json(report_path, report)
    report = load_and_validate_canary_report(report_path, **expected)
    print(
        json.dumps({"status": "passed", "report": report_path.as_posix()}), flush=True
    )
    return report


@hydra.main(version_base=None, config_path="../configs", config_name="v2_canary")
def _hydra_main(config: DictConfig) -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "v2_canary.yaml"
    _run(_plain(config), config_path=config_path)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--data-gate-only", action="store_true")
    mode.add_argument("--approve-arrows-gate", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reviewer")
    parser.add_argument("--reviewed-at")
    known, remaining = parser.parse_known_args()
    if known.data_gate_only:
        if remaining or known.output_dir is None or known.reviewer or known.reviewed_at:
            raise CanaryError(
                "--data-gate-only requires only --output-dir <new-directory>"
            )
        root = Path(__file__).resolve().parents[1]
        config_path = root / "configs" / "v2_canary.yaml"
        raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False)
        assert isinstance(raw, Mapping)
        gate_path = write_arrows_gate_bundle(
            root,
            raw,
            config_path=config_path,
            output_dir=known.output_dir.resolve(),
        )
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        print(
            json.dumps(
                {
                    "status": gate["status"],
                    "gate": gate_path.as_posix(),
                    "contact_sheet": (
                        gate_path.parent / gate["arrows"]["contact_sheet"]["path"]
                    ).as_posix(),
                }
            ),
            flush=True,
        )
        return
    if known.approve_arrows_gate is not None:
        if remaining or known.output_dir is not None or not known.reviewer:
            raise CanaryError(
                "--approve-arrows-gate requires --reviewer and no output directory"
            )
        reviewed_at = known.reviewed_at or datetime.now(UTC).isoformat()
        reviewed = approve_arrows_gate(
            known.approve_arrows_gate.resolve(),
            reviewer=known.reviewer,
            reviewed_at=reviewed_at,
        )
        print(
            json.dumps({"status": "reviewed", "gate": reviewed.as_posix()}),
            flush=True,
        )
        return
    if any(
        value is not None
        for value in (known.output_dir, known.reviewer, known.reviewed_at)
    ):
        raise CanaryError("data-gate flags require an explicit data-gate mode")
    _hydra_main()


if __name__ == "__main__":
    main()
