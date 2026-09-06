"""Versioned VP/VE global campaign with two predeclared LID selectors.

This module deliberately reuses the v1 data, provenance and logging primitives
without weakening their allowlists.  Its model matrix, cell identity, scale
support and result schema are independent from the sealed v1 campaign.
"""

from __future__ import annotations

import csv
import importlib.metadata
import io
import json
import math
import os
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from kneed import KneeLocator
from omegaconf import DictConfig, OmegaConf

from experiments import global_campaign as v1
from experiments.metrics import (
    known_lid_metrics,
    paired_delta_metrics,
    prediction_summary,
)
from experiments.run_manifest import (
    canonical_json,
    hash_declared_sources,
    sha256_bytes,
    sha256_path,
)

PROJECT_NAME = v1.PROJECT_NAME
WORKSPACE_NAME = v1.WORKSPACE_NAME
APPROVED_CAMPAIGN_ID = (
    "lid-global-e1-e8-canonical-plus-generated-e3-e4-vp-ve-all-models-v2"
)
GLOBAL_CAMPAIGN_SCHEMA_VERSION = 2
GLOBAL_CELL_MANIFEST_SCHEMA_VERSION = 2
GLOBAL_FINAL_MANIFEST_SCHEMA_VERSION = 2
EXPECTED_GLOBAL_CELL_COUNT = v1.EXPECTED_GLOBAL_CELL_COUNT
APPROVED_GLOBAL_CELL_KEYS = v1.APPROVED_GLOBAL_CELL_KEYS
REQUIRED_SUITE_IDS = v1.REQUIRED_SUITE_IDS
APPROVED_MODEL_VARIANTS = (
    "vp_diffusion",
    "ve_diffusion",
    "rectified_flow",
    "scale_conditioned_nf",
    "schrodinger_bridge",
    "direct_rectified_flow",
    "posterior_rectified_flow",
    "direct_log_noise_affine_flow",
    "posterior_log_noise_affine_flow",
    "direct_vp_trigonometric_flow",
    "posterior_vp_trigonometric_flow",
)
EXPECTED_PHYSICAL_TRAININGS = len(APPROVED_MODEL_VARIANTS) * EXPECTED_GLOBAL_CELL_COUNT
KNOWN_SUPERVISED_PROTOCOL = "held_out_source_train_supervised_mae_bounded_v2"
KNOWN_KNEEDLE_PROTOCOL = "flipd_pointwise_kneedle_v1"
UNKNOWN_REFERENCE_PROTOCOL = "held_out_reference_mean_kneedle_v2"
FROZEN_EVALUATION_PROTOCOL = "multi_protocol_train_selected_lambda_v2"
EXECUTION_STRATEGY_SEQUENTIAL = v1.EXECUTION_STRATEGY_SEQUENTIAL
EXECUTION_STRATEGY_CELL_DAG = v1.EXECUTION_STRATEGY_CELL_DAG
EXECUTION_PROFILE_LEGACY = "legacy_sequential_v2"
EXECUTION_PROFILE_H100 = "h100_8gpu_3lane_cell_dag_v2"
H100_PHYSICAL_DEVICE_COUNT = 8
H100_LANES_PER_DEVICE = 3
H100_LOGICAL_WORKER_COUNT = H100_PHYSICAL_DEVICE_COUNT * H100_LANES_PER_DEVICE
CHECKPOINT_RETENTION_RETAIN = v1.CHECKPOINT_RETENTION_RETAIN
CHECKPOINT_RETENTION_PRUNE_AFTER_EVALUATION = (
    v1.CHECKPOINT_RETENTION_PRUNE_AFTER_EVALUATION
)
SUPPORTED_CHECKPOINT_RETENTION_POLICIES = v1.SUPPORTED_CHECKPOINT_RETENTION_POLICIES
FLIPD_SOURCE_REVISION = "05ab170c2c9bcada3f9286c3ef86db31b80925fa"
PINNED_KNEED_VERSION = "0.8.6"
COMMON_LAMBDA_MIN = 1.0 / 256.0
COMMON_LAMBDA_MAX = 64.0
FIELD_HIDDEN_SIZES = (1024, 512, 256, 256, 128, 128)
TIME_EMBEDDING_DIM = 128
NF_WIDTH_MULTIPLE = 32
NF_MAX_RELATIVE_GAP = 0.10
FM_EXACT_TRACE_MAX_AMBIENT_DIM = 64
FM_HIGH_DIM_QUERY_SUBSET_SIZE = 8
FM_HIGH_DIM_TRACE_PROBES = (16, 64)
FM_HIGH_DIM_PROTOCOL = "hutchinson_prefix_stability_high_dimensional_v1"
CANARY_PROTOCOL_ID = "vp-ve-fm-nf-quality-canary-v2"
CANARY_CELL_KEYS = (
    "e2/e2_uniform_pca/coefficients",
    "e2/e2_arrows/dataset",
)
CANARY_MODEL_VARIANTS = (
    "vp_diffusion",
    "ve_diffusion",
    "posterior_log_noise_affine_flow",
    "scale_conditioned_nf",
)
CANARY_TASKS = tuple(
    (variant, key) for key in CANARY_CELL_KEYS for variant in CANARY_MODEL_VARIANTS
)
CANARY_COMPANION_CELL_KEYS = (
    "e3/e3_gaussian_pca/coefficients",
    "e1/e1_sampled_fmnist_step1/dataset",
    "e4/e4_sphere_pca_radius1/coefficients",
    "e2/e2_uniform_pca/dataset",
)
CANARY_COMPANION_TASKS = tuple(
    (variant, key)
    for key in CANARY_COMPANION_CELL_KEYS
    for variant in CANARY_MODEL_VARIANTS
)
CANARY_REUSE_TASKS = CANARY_TASKS + CANARY_COMPANION_TASKS

GlobalCampaignError = v1.GlobalCampaignError
ModelPlan = v1.ModelPlan
CampaignCell = v1.CampaignCell
CellData = v1.CellData
HoldoutPartition = v1.HoldoutPartition
ModelLoggerHandle = v1.ModelLoggerHandle
TrainFunction = v1.TrainFunction
PredictFunction = v1.PredictFunction

repository_root = v1.repository_root
load_campaign_inventory = v1.load_campaign_inventory
load_campaign_cell_data = v1.load_campaign_cell_data
validate_campaign_sources = v1.validate_campaign_sources
partition_source_train = v1.partition_source_train
open_model_logger = v1.open_model_logger
_safe_model_logger = v1._safe_model_logger
_emit = v1._emit
_bind_source_preflight = v1._bind_source_preflight
_plain = v1._plain
_mapping = v1._mapping
_safe_path = v1._safe_path
_same_json = v1._same_json
_safe_component = v1._safe_component
_write_json = v1._write_json
_write_yaml = v1._write_yaml
_write_text = v1._write_text
_save_npy = v1._save_npy
_load_json = v1._load_json
_load_numeric_array = v1._load_numeric_array
_output_inventory = v1._output_inventory
_reference_cell = v1._reference_cell
_clear_accelerator_cache = v1._clear_accelerator_cache
_exclusive_campaign_lock = v1._exclusive_campaign_lock
_SHA256 = v1._SHA256


def _config_dir(root: Path | None = None) -> Path:
    return (repository_root() if root is None else Path(root)) / "configs"


def compose_global_campaign_v2_config(
    overrides: Sequence[str] = (), *, root: Path | None = None
) -> DictConfig:
    config_dir = _config_dir(root).resolve()
    global_hydra = GlobalHydra.instance()
    if global_hydra.is_initialized():
        search_path = global_hydra.hydra.config_loader.get_search_path()
        actual_search_path = tuple(
            (str(entry.provider), str(entry.path))
            for entry in search_path.config_search_path
        )
        approved_library_search_path = (
            ("hydra", "pkg://hydra.conf"),
            ("main", str(config_dir)),
            ("schema", "structured://"),
        )
        approved_cli_search_path = (
            ("hydra", "pkg://hydra.conf"),
            ("main", "pkg://experiments"),
            ("command-line", config_dir.as_uri()),
            ("schema", "structured://"),
        )
        approved_module_cli_search_path = (
            ("hydra", "pkg://hydra.conf"),
            ("command-line", config_dir.as_uri()),
            ("schema", "structured://"),
        )
        approved_layouts = {
            approved_library_search_path: "main",
            approved_cli_search_path: "command-line",
            approved_module_cli_search_path: "command-line",
        }
        expected_provider = approved_layouts.get(actual_search_path)
        if expected_provider is None:
            raise GlobalCampaignError(
                "active Hydra search path differs from the exact approved v2 "
                "library and CLI layouts"
            )
        expected_path = config_dir.as_uri()
        repository = global_hydra.hydra.config_loader.repository
        for config_name in (
            "global_campaign_v2.yaml",
            "campaign/all_suites_all_models_v2.yaml",
        ):
            selected = repository.load_config(config_name)
            if (
                selected is None
                or selected.provider != expected_provider
                or selected.path != expected_path
            ):
                raise GlobalCampaignError(
                    f"active Hydra selected an unapproved source for {config_name!r}"
                )
        config = compose(config_name="global_campaign_v2", overrides=list(overrides))
    else:
        with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
            config = compose(
                config_name="global_campaign_v2", overrides=list(overrides)
            )
    OmegaConf.set_struct(config, True)
    return config


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = {str(key): _plain(value) for key, value in base.items()}
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[str(key)] = _deep_merge(result[str(key)], value)
        else:
            result[str(key)] = _plain(value)
    return result


def _load_model_fragment(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise GlobalCampaignError(f"cannot load v2 model config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GlobalCampaignError(f"v2 model config {path} must be a mapping")
    return value


def _resolved_model(variant_id: str, path_value: str, seed: int) -> dict[str, Any]:
    if seed != 0:
        raise GlobalCampaignError("v2 model seed must be exactly zero")
    root = repository_root()
    base_path = root / "configs/pilot_model_v2/base.yaml"
    path = root / _safe_path(path_value, field=f"{variant_id}.config")
    expected = root / f"configs/pilot_model_v2/{variant_id}.yaml"
    if path.resolve() != expected.resolve():
        raise GlobalCampaignError(f"{variant_id} model config path is not approved")
    model = _deep_merge(_load_model_fragment(base_path), _load_model_fragment(path))
    if model.get("id") != variant_id:
        raise GlobalCampaignError(
            f"v2 model fragment identity differs for {variant_id}"
        )
    allowed = {
        "schema_version",
        "id",
        "name",
        "family",
        "readout",
        "primary_readout",
        "selection_prefer",
        "derivative_backend",
        "trace_probes",
        "native_coordinate",
        "lambda_support",
        "capacity_policy",
        "training",
        "scales",
        "diagnostics",
    }
    v1._reject_unknown(model, allowed, field=f"v2 model {variant_id}")
    if model.get("schema_version") != 2:
        raise GlobalCampaignError(f"v2 model {variant_id} schema differs")
    if tuple(float(x) for x in model.get("lambda_support", ())) != (
        COMMON_LAMBDA_MIN,
        COMMON_LAMBDA_MAX,
    ):
        raise GlobalCampaignError(f"v2 model {variant_id} lambda support differs")
    scales = np.asarray(model.get("scales"), dtype=np.float64)
    expected_scales = initial_supervised_lambdas()
    if scales.shape != expected_scales.shape or not np.array_equal(
        scales, expected_scales
    ):
        raise GlobalCampaignError(f"v2 model {variant_id} initial grid differs")
    training = _mapping(model.get("training"), field=f"{variant_id}.training")
    training["seed"] = seed
    try:
        from models.training import TrainingConfig

        materialized = TrainingConfig.from_mapping(training).to_dict()
    except (TypeError, ValueError) as exc:
        raise GlobalCampaignError(
            f"invalid v2 training config for {variant_id}"
        ) from exc
    if (
        materialized["training_mode"] != "fixed_steps_v1"
        or materialized["steps"] != 32000
        or materialized["batch_size"] != 256
        or materialized["warmup_steps"] != (500 if variant_id == "vp_diffusion" else 0)
        or materialized["gradient_clip_norm"]
        != (None if variant_id == "vp_diffusion" else 1.0)
        or materialized["validation_interval_steps"] != 500
        or materialized["early_stopping_patience"] is not None
    ):
        raise GlobalCampaignError(f"v2 model {variant_id} budget differs")
    model["training"] = training
    _validate_model_support(model)
    return model


def validate_global_campaign_config(
    config: DictConfig | Mapping[str, Any],
) -> dict[str, Any]:
    value = _mapping(config, field="global v2 campaign")
    v1._reject_unknown(
        value,
        {
            "schema_version",
            "project",
            "seed",
            "output_root",
            "execution",
            "data",
            "logging",
            "campaign",
        },
        field="v2 top-level",
    )
    if value.get("schema_version") != 2 or isinstance(
        value.get("schema_version"), bool
    ):
        raise GlobalCampaignError("global v2 schema_version must be exactly 2")
    if value.get("project") != PROJECT_NAME or value.get("seed") != 0:
        raise GlobalCampaignError("global v2 project/seed differs from contract")
    if v1._contains_secret(value):
        raise GlobalCampaignError("credentials are forbidden in campaign config")
    _safe_path(value.get("output_root"), field="output_root")

    execution = _mapping(value.get("execution"), field="execution")
    base_execution_fields = {
        "profile",
        "strategy",
        "worker_count",
        "training_batch_size_override",
        "evaluation_batch_size_override",
    }
    expected_execution_fields = (
        base_execution_fields | {"physical_device_count", "lanes_per_device"}
        if execution.get("profile") == EXECUTION_PROFILE_H100
        else base_execution_fields
    )
    if set(execution) != expected_execution_fields:
        raise GlobalCampaignError("v2 execution fields differ")
    if execution["strategy"] not in {
        EXECUTION_STRATEGY_SEQUENTIAL,
        EXECUTION_STRATEGY_CELL_DAG,
    }:
        raise GlobalCampaignError("v2 execution strategy is unsupported")
    v1._positive_int(execution["worker_count"], field="execution.worker_count")
    if execution["training_batch_size_override"] is not None:
        raise GlobalCampaignError("v2 forbids hidden training batch overrides")
    if execution["evaluation_batch_size_override"] is not None:
        v1._positive_int(
            execution["evaluation_batch_size_override"],
            field="execution.evaluation_batch_size_override",
        )
    if execution["profile"] == EXECUTION_PROFILE_H100:
        physical_device_count = v1._positive_int(
            execution["physical_device_count"],
            field="execution.physical_device_count",
        )
        lanes_per_device = v1._positive_int(
            execution["lanes_per_device"],
            field="execution.lanes_per_device",
        )
        if physical_device_count * lanes_per_device != execution["worker_count"]:
            raise GlobalCampaignError(
                "v2 H100 worker_count does not match its device topology"
            )
    if execution["profile"] == EXECUTION_PROFILE_H100 and execution != {
        "profile": EXECUTION_PROFILE_H100,
        "strategy": EXECUTION_STRATEGY_CELL_DAG,
        "worker_count": H100_LOGICAL_WORKER_COUNT,
        "physical_device_count": H100_PHYSICAL_DEVICE_COUNT,
        "lanes_per_device": H100_LANES_PER_DEVICE,
        "training_batch_size_override": None,
        "evaluation_batch_size_override": 512,
    }:
        raise GlobalCampaignError("v2 H100 profile is immutable")
    if execution["strategy"] == EXECUTION_STRATEGY_SEQUENTIAL and execution != {
        "profile": EXECUTION_PROFILE_LEGACY,
        "strategy": EXECUTION_STRATEGY_SEQUENTIAL,
        "worker_count": 1,
        "training_batch_size_override": None,
        "evaluation_batch_size_override": None,
    }:
        raise GlobalCampaignError("v2 sequential profile differs")

    data = _mapping(value.get("data"), field="data")
    expected_data = {
        "canonical_archive",
        "canonical_extracted_root",
        "root",
        "generated_root",
        "canonical_pca",
        "generated_manifest",
        "registry",
        "mmap_mode",
    }
    if set(data) != expected_data:
        raise GlobalCampaignError("v2 data fields differ")
    if data["registry"] != "configs/datasets/registry/paper_benchmarks.yaml":
        raise GlobalCampaignError("v2 registry differs from canonical registry")
    if data["mmap_mode"] not in {None, "r"}:
        raise GlobalCampaignError("v2 mmap mode is invalid")
    for field in expected_data - {"mmap_mode"}:
        _safe_path(data[field], field=f"data.{field}")

    logging = _mapping(value.get("logging"), field="logging")
    if set(logging) != {"backend", "project", "workspace"}:
        raise GlobalCampaignError("v2 logging fields differ")
    if logging["backend"] not in {"none", "comet"}:
        raise GlobalCampaignError("v2 logging backend is unsupported")
    if logging["project"] != PROJECT_NAME or logging["workspace"] != WORKSPACE_NAME:
        raise GlobalCampaignError("v2 logging identity differs")

    campaign = _mapping(value.get("campaign"), field="campaign")
    if set(campaign) != {
        "campaign_id",
        "inventory",
        "models",
        "selection",
        "evaluation",
        "canary_gate",
        "resume",
        "fm_diagnostics",
    }:
        raise GlobalCampaignError("v2 campaign fields differ")
    if campaign["campaign_id"] != APPROVED_CAMPAIGN_ID:
        raise GlobalCampaignError("v2 campaign id differs")
    inventory = _mapping(campaign["inventory"], field="campaign.inventory")
    if inventory != {
        "canonical": "configs/global_suite/canonical_exact.yaml",
        "generated_e3_e4": "configs/global_suite/generated_e3_e4.yaml",
        "require_suite_ids": list(REQUIRED_SUITE_IDS),
        "include_all_representations": True,
    }:
        raise GlobalCampaignError("v2 inventory differs from exact 35+4 contract")
    rows = campaign["models"]
    if (
        not isinstance(rows, list)
        or tuple(row.get("id") for row in rows if isinstance(row, dict))
        != APPROVED_MODEL_VARIANTS
    ):
        raise GlobalCampaignError("v2 model allowlist/order differs")
    seen_names: set[str] = set()
    for index, row in enumerate(rows):
        model_row = _mapping(row, field=f"campaign.models[{index}]")
        if set(model_row) != {"id", "config", "experiment_name"}:
            raise GlobalCampaignError("v2 model row fields differ")
        name = model_row["experiment_name"]
        if (
            not isinstance(name, str)
            or not v1._EXPERIMENT_NAME.fullmatch(name)
            or name in seen_names
        ):
            raise GlobalCampaignError("v2 experiment names are invalid or repeated")
        seen_names.add(name)
        _resolved_model(str(model_row["id"]), str(model_row["config"]), 0)
    _validate_selection_config(_mapping(campaign["selection"], field="selection"))
    _validate_evaluation_config(_mapping(campaign["evaluation"], field="evaluation"))
    _validate_canary_gate_config(_mapping(campaign["canary_gate"], field="canary_gate"))
    if campaign["resume"] != {
        "schema_version": 2,
        "cell_attempt_policy": "stable_identity_directory",
        "training_progress_filename": "training_progress.pt",
        "invalid_existing_policy": "fail",
        "source_tree_check_before_every_cell": True,
        "lock_filename": "campaign.lock",
    }:
        raise GlobalCampaignError("v2 resume contract differs")
    if campaign["fm_diagnostics"] != {
        "policy": "dimension_aware_known_lid_v2",
        "exact_trace_max_ambient_dim": FM_EXACT_TRACE_MAX_AMBIENT_DIM,
        "high_dim_query_subset_size": FM_HIGH_DIM_QUERY_SUBSET_SIZE,
        "high_dim_trace_probes": list(FM_HIGH_DIM_TRACE_PROBES),
        "high_dim_scale_policy": "actual_selection_grid",
        "high_dim_exact_status": "skipped_cost_prohibitive",
        "high_dim_oracle_status": "skipped_cost_prohibitive",
        "unknown_lid_status": "not_applicable_no_lid_targets",
    }:
        raise GlobalCampaignError("v2 FM diagnostics contract differs")
    return value


def _validate_selection_config(selection: Mapping[str, Any]) -> None:
    expected = {
        "known_supervised_protocol": KNOWN_SUPERVISED_PROTOCOL,
        "known_label_free_protocol": KNOWN_KNEEDLE_PROTOCOL,
        "unknown_reference_protocol": UNKNOWN_REFERENCE_PROTOCOL,
        "index_algorithm": "splitmix64_rank_v1",
        "fraction": 0.2,
        "minimum_selection": 2,
        "maximum_selection": 1000,
        "minimum_fit": 8,
        "criterion": "mae",
        "tie_tolerance": 1.0e-12,
        "lambda_min": COMMON_LAMBDA_MIN,
        "lambda_max": COMMON_LAMBDA_MAX,
        "initial_log2_half_step_min": -12,
        "initial_log2_half_step_max": 8,
        "boundary_extension_half_steps": 4,
        "flipd_source_revision": FLIPD_SOURCE_REVISION,
        "kneed_version": PINNED_KNEED_VERSION,
        "flipd_grid_count": 50,
        "flipd_ignore_time_left": 0.05,
        "flipd_ignore_time_right": 0.5,
        "flipd_sensitivity": 1.0,
        "unknown_grid_count": 50,
        "vp_beta_min": 0.1,
        "vp_beta_max": 20.0,
        "minimum_kneedle_segment": 3,
    }
    if not _same_json(selection, expected):
        raise GlobalCampaignError("v2 selection contract differs")
    actual_kneed = importlib.metadata.version("kneed")
    if actual_kneed != PINNED_KNEED_VERSION:
        raise GlobalCampaignError(
            f"v2 requires kneed {PINNED_KNEED_VERSION}, found {actual_kneed}"
        )


def _validate_evaluation_config(evaluation: Mapping[str, Any]) -> None:
    expected = {
        "batch_size": 128,
        "checkpoint_retention": CHECKPOINT_RETENTION_PRUNE_AFTER_EVALUATION,
        "supervised_validation_candidate_count": 1,
        "supervised_test_candidate_count": 1,
        "kneedle_uses_targets": False,
        "save_pointwise_protocol_curves": True,
        "nf_primary_readout": "ols5",
        "nf_ols_log_step": 0.05,
    }
    if not _same_json(evaluation, expected):
        raise GlobalCampaignError("v2 evaluation contract differs")


def _validate_canary_gate_config(gate: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": 2,
        "required": True,
        "protocol_id": CANARY_PROTOCOL_ID,
        "report_filename": "canary_report.json",
        "required_physical_device_count": H100_PHYSICAL_DEVICE_COUNT,
        "required_lanes_per_device": H100_LANES_PER_DEVICE,
        "required_logical_worker_count": H100_LOGICAL_WORKER_COUNT,
        "required_batch_size": 256,
        "required_cell_keys": list(CANARY_CELL_KEYS),
        "required_companion_cell_keys": list(CANARY_COMPANION_CELL_KEYS),
        "required_model_variants": list(CANARY_MODEL_VARIANTS),
    }
    if not _same_json(gate, expected):
        raise GlobalCampaignError("v2 canary gate contract differs")


def model_plans(config: Mapping[str, Any]) -> tuple[ModelPlan, ...]:
    execution = _mapping(config["execution"], field="execution")
    plans: list[ModelPlan] = []
    for row in config["campaign"]["models"]:
        model = _resolved_model(str(row["id"]), str(row["config"]), int(config["seed"]))
        if (
            execution["evaluation_batch_size_override"] is not None
            and model.get("family") == "independent_affine_flow"
        ):
            diagnostics = _mapping(model["diagnostics"], field="affine diagnostics")
            diagnostics["batch_size"] = int(execution["evaluation_batch_size_override"])
            model["diagnostics"] = diagnostics
        plans.append(
            ModelPlan(
                variant_id=str(row["id"]),
                experiment_name=str(row["experiment_name"]),
                model=model,
            )
        )
    return tuple(plans)


def initial_supervised_lambdas() -> npt.NDArray[np.float64]:
    return np.ascontiguousarray(
        np.asarray([2.0 ** (k / 2.0) for k in range(-12, 9)], dtype=np.float64)
    )


def flipd_timesteps() -> npt.NDArray[np.float64]:
    complete = np.linspace(0.0, 1.0, 50, dtype=np.float64)
    return np.ascontiguousarray(complete[(complete > 0.05) & (complete < 0.5)])


def vp_lambda_from_time(
    time: npt.ArrayLike, *, beta_min: float = 0.1, beta_max: float = 20.0
) -> npt.NDArray[np.float64]:
    values = np.asarray(time, dtype=np.float64)
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise GlobalCampaignError("VP time must be finite and lie in [0, 1]")
    integral = beta_min * values + 0.5 * (beta_max - beta_min) * values**2
    return np.ascontiguousarray(np.sqrt(np.expm1(integral)), dtype=np.float64)


def vp_time_from_lambda(
    noise_ratio: npt.ArrayLike, *, beta_min: float = 0.1, beta_max: float = 20.0
) -> npt.NDArray[np.float64]:
    values = np.asarray(noise_ratio, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise GlobalCampaignError("lambda must be finite and positive")
    integral = np.log1p(values**2)
    denominator = beta_min + np.sqrt(
        beta_min**2 + 2.0 * (beta_max - beta_min) * integral
    )
    result = 2.0 * integral / denominator
    if np.any(result > 1.0):
        raise GlobalCampaignError("lambda lies outside the VP native interval")
    return np.ascontiguousarray(result, dtype=np.float64)


def unknown_reference_lambdas() -> npt.NDArray[np.float64]:
    endpoints = vp_time_from_lambda((COMMON_LAMBDA_MIN, COMMON_LAMBDA_MAX))
    return vp_lambda_from_time(np.linspace(endpoints[0], endpoints[1], 50))


def _native_coordinate(model: Mapping[str, Any], noise_ratio: float) -> float:
    _assert_lambdas_supported(model, (noise_ratio,))
    coordinate = str(model["native_coordinate"])
    if coordinate in {"lambda", "ve_sigma", "epsilon", "vp_time_from_lambda"}:
        # The VP trainer/predictor accepts physical lambda and owns the t map.
        return float(noise_ratio)
    if coordinate == "rectified_time_from_lambda":
        return float(1.0 / (1.0 + noise_ratio))
    if coordinate == "brownian_tau":
        return float(noise_ratio**2)
    raise GlobalCampaignError(f"unsupported v2 native coordinate {coordinate!r}")


def _validate_model_support(model: Mapping[str, Any]) -> None:
    training = _mapping(model["training"], field="model.training")
    family = str(model["family"])
    lo, hi = (float(x) for x in model["lambda_support"])
    if (lo, hi) != (COMMON_LAMBDA_MIN, COMMON_LAMBDA_MAX):
        raise GlobalCampaignError("model support differs from common lambda interval")
    if family == "vp_diffusion":
        if training.get("vp_beta_min") != 0.1 or training.get("vp_beta_max") != 20.0:
            raise GlobalCampaignError("VP schedule differs from v2 contract")
        vp_time_from_lambda((lo, hi))
    elif family == "diffusion":
        if (training.get("sigma_min"), training.get("sigma_max")) != (lo, hi):
            raise GlobalCampaignError("VE sigma support differs from lambda support")
    elif family == "rectified_flow":
        expected = (1.0 / (1.0 + hi), 1.0 / (1.0 + lo))
        if not np.allclose(
            (training.get("time_min"), training.get("time_max")),
            expected,
            rtol=0,
            atol=1e-15,
        ):
            raise GlobalCampaignError("rectified-flow time support differs")
    elif family == "independent_affine_flow":
        if (
            training.get("flow_noise_ratio_min"),
            training.get("flow_noise_ratio_max"),
        ) != (lo, hi):
            raise GlobalCampaignError("affine-FM lambda support differs")
    elif family == "schrodinger_bridge":
        if (
            training.get("bridge_terminal_time") != hi**2
            or training.get("bridge_tau_min") != lo**2
            or training.get("bridge_tau_max") != hi**2
        ):
            raise GlobalCampaignError("bridge tau support differs")
    elif family == "scale_conditioned_nf":
        padded = (lo * math.exp(-0.1), hi * math.exp(0.1))
        if not np.allclose(
            (training.get("epsilon_min"), training.get("epsilon_max")),
            padded,
            rtol=0,
            atol=1e-15,
        ):
            raise GlobalCampaignError("NF OLS5-padded epsilon support differs")
    else:
        raise GlobalCampaignError(f"unsupported v2 family {family!r}")


def _assert_lambdas_supported(
    model: Mapping[str, Any], values: npt.ArrayLike, *, include_nf_stencil: bool = False
) -> None:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all() or np.any(array <= 0.0):
        raise GlobalCampaignError("requested lambdas must be finite and positive")
    lo, hi = (float(x) for x in model["lambda_support"])
    tolerance = 32.0 * np.finfo(np.float64).eps * max(1.0, hi)
    if float(array.min()) < lo - tolerance or float(array.max()) > hi + tolerance:
        raise GlobalCampaignError(
            f"requested estimator center lies outside declared lambda support [{lo}, {hi}]"
        )
    if include_nf_stencil and model.get("family") == "scale_conditioned_nf":
        step = 0.05
        offsets = np.arange(-2, 3, dtype=np.float64)
        array = np.concatenate(
            [array, (array[:, None] * np.exp(step * offsets)).ravel()]
        )
        lo = float(model["training"]["epsilon_min"])
        hi = float(model["training"]["epsilon_max"])
        tolerance = 32.0 * np.finfo(np.float64).eps * max(1.0, hi)
    if float(array.min()) < lo - tolerance or float(array.max()) > hi + tolerance:
        raise GlobalCampaignError(
            f"requested estimator scale lies outside checkpoint support [{lo}, {hi}]"
        )


def _feature_ambient_dim(input_record: Mapping[str, Any]) -> int:
    shape = input_record.get("feature_shape")
    if (
        not isinstance(shape, list)
        or not shape
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in shape
        )
    ):
        raise GlobalCampaignError("cell input record lacks a valid feature shape")
    return math.prod(shape)


def _nf_parameter_count(ambient_dim: int, hidden_dim: int) -> int:
    import torch

    from models.normalizing_flow import ConditionalFlowConfig, ScaleConditionedRealNVP

    with torch.device("meta"):
        model = ScaleConditionedRealNVP(
            ConditionalFlowConfig(
                ambient_dim=ambient_dim,
                hidden_dim=hidden_dim,
                num_coupling_layers=8,
                conditioner_depth=2,
                condition_dim=TIME_EMBEDDING_DIM,
                fourier_features=32,
                max_condition_frequency=100.0,
                dropout=0.0,
                log_scale_limit=2.0,
            )
        )
    return sum(parameter.numel() for parameter in model.parameters())


def _field_parameter_count(ambient_dim: int) -> int:
    from models.vp_baseline import bottleneck_parameter_count

    return bottleneck_parameter_count(
        ambient_dim, FIELD_HIDDEN_SIZES, TIME_EMBEDDING_DIM
    )


def select_nf_width(ambient_dim: int) -> tuple[int, int, int, float]:
    target = _field_parameter_count(ambient_dim)
    candidates: list[tuple[float, int, int]] = []
    for width in range(NF_WIDTH_MULTIPLE, 2048 + NF_WIDTH_MULTIPLE, NF_WIDTH_MULTIPLE):
        count = _nf_parameter_count(ambient_dim, width)
        candidates.append((abs(count - target) / target, width, count))
    gap, width, actual = min(candidates)
    if gap > NF_MAX_RELATIVE_GAP:
        raise GlobalCampaignError(
            f"no NF width matches D={ambient_dim} within {NF_MAX_RELATIVE_GAP:.0%}"
        )
    return width, target, actual, float((actual - target) / target)


def resolve_cell_model(
    model_plan: ModelPlan, input_record: Mapping[str, Any]
) -> dict[str, Any]:
    model = _plain(model_plan.model)
    ambient_dim = _feature_ambient_dim(input_record)
    target = _field_parameter_count(ambient_dim)
    if model["family"] == "scale_conditioned_nf":
        width, target, actual, relative_gap = select_nf_width(ambient_dim)
        model["training"]["hidden_dim"] = width
    else:
        from models.training import field_parameter_count

        actual = field_parameter_count(
            str(model["family"]), model["training"], ambient_dim
        )
        relative_gap = float((actual - target) / target)
        if relative_gap != 0.0:
            raise GlobalCampaignError(
                "cell vector-field capacity differs from the shared bottleneck"
            )
    model["capacity_alignment"] = {
        "schema_version": 1,
        "policy_id": model["capacity_policy"]["id"],
        "ambient_dim": ambient_dim,
        "field_hidden_sizes": list(FIELD_HIDDEN_SIZES),
        "time_embedding_dim": TIME_EMBEDDING_DIM,
        "nf_hidden_dim": (
            int(model["training"]["hidden_dim"])
            if model["family"] == "scale_conditioned_nf"
            else None
        ),
        "reference_parameter_count": target,
        "actual_parameter_count": actual,
        "relative_gap": relative_gap,
        "within_tolerance": abs(relative_gap) <= NF_MAX_RELATIVE_GAP,
    }
    if model["family"] == "independent_affine_flow":
        model["diagnostics_execution"] = {
            "schema_version": 1,
            "policy": "dimension_aware_known_lid_v2",
            "ambient_dim": ambient_dim,
            "mode": (
                "small_dim_exhaustive_exact"
                if ambient_dim <= FM_EXACT_TRACE_MAX_AMBIENT_DIM
                else "high_dim_hutchinson_convergence"
            ),
            "exact_trace_max_ambient_dim": FM_EXACT_TRACE_MAX_AMBIENT_DIM,
            "high_dim_query_subset_size": FM_HIGH_DIM_QUERY_SUBSET_SIZE,
            "high_dim_trace_probes": list(FM_HIGH_DIM_TRACE_PROBES),
            "high_dim_scale_policy": "actual_selection_grid",
            "high_dim_exact_status": "skipped_cost_prohibitive",
            "high_dim_oracle_status": "skipped_cost_prohibitive",
        }
    try:
        from models.training import TrainingConfig

        TrainingConfig.from_mapping(model["training"])
    except (TypeError, ValueError) as exc:
        raise GlobalCampaignError("cell-resolved training config is invalid") from exc
    _validate_model_support(model)
    return model


def _high_dim_subset_indices(n_rows: int) -> npt.NDArray[np.int64]:
    if n_rows <= 0:
        raise GlobalCampaignError("high-dimensional FM diagnostic query is empty")
    size = min(FM_HIGH_DIM_QUERY_SUBSET_SIZE, n_rows)
    if size == n_rows:
        return np.arange(n_rows, dtype=np.int64)
    return v1._splitmix64_indices(n_rows, subset_size=size, seed=0)


def _source_evidence(data: CellData, config: Mapping[str, Any]) -> dict[str, Any]:
    evidence = v1._source_evidence(data, config)
    partition = partition_source_train(
        data.train,
        data.train_target,
        selection=config["campaign"]["selection"],
        seed=int(config["seed"]),
    )
    indices = _high_dim_subset_indices(partition.selection_features.shape[0])
    evidence["fm_high_dim_query_subset_sha256"] = v1._array_sha256(
        np.ascontiguousarray(partition.selection_features[indices])
    )
    return evidence


def _hutchinson_stability_metrics(
    low: npt.ArrayLike, high: npt.ArrayLike
) -> dict[str, Any]:
    left = np.asarray(low, dtype=np.float64)
    right = np.asarray(high, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2:
        raise GlobalCampaignError("Hutchinson stability arrays must share a 2-D shape")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise GlobalCampaignError("Hutchinson stability arrays must be finite")
    difference = left - right
    absolute = np.abs(difference)
    return {
        "n": int(difference.size),
        "mean_signed_difference": float(difference.mean()),
        "mean_absolute_difference": float(absolute.mean()),
        "rmse_difference": float(np.sqrt(np.mean(difference**2))),
        "maximum_absolute_difference": float(absolute.max()),
    }


def _high_dim_fm_attestation(output_dir: Path) -> dict[str, Any]:
    metadata = _load_json(output_dir / "metadata.json")
    return {
        "status": "completed_hutchinson_prefix_stability_v1",
        "protocol_id": FM_HIGH_DIM_PROTOCOL,
        "path": "fm_diagnostics",
        "manifest_sha256": sha256_path(output_dir / "manifest.json"),
        "metadata_sha256": sha256_path(output_dir / "metadata.json"),
        "summary_sha256": sha256_path(output_dir / "summary.json"),
        "outer_selection_curve_sha256": metadata["outer_selection_curve_sha256"],
        "exact_trace_status": "skipped_cost_prohibitive",
        "oracle_status": "skipped_cost_prohibitive",
    }


def _validate_high_dim_fm_diagnostics(
    output_dir: Path,
    *,
    model: Mapping[str, Any],
    checkpoint_sha256: Any,
    scales: npt.ArrayLike,
    selection_curve: npt.ArrayLike,
    partition: Mapping[str, Any],
    fm: Mapping[str, Any],
    expected_query_subset_sha256: str | None = None,
) -> list[str]:
    errors: list[str] = []
    try:
        manifest = _load_json(output_dir / "manifest.json")
        metadata = _load_json(output_dir / "metadata.json")
        summary = _load_json(output_dir / "summary.json")
        subset_indices = np.asarray(
            _load_numeric_array(output_dir / "arrays/query_subset_indices.npy", ndim=1),
            dtype=np.int64,
        )
        query_subset = np.asarray(
            _load_numeric_array(output_dir / "arrays/query_subset.npy", ndim=2)
        )
        diagnostic_scales = np.asarray(
            _load_numeric_array(output_dir / "arrays/scales.npy", ndim=1),
            dtype=np.float64,
        )
        low = np.asarray(
            _load_numeric_array(output_dir / "arrays/lid_hutchinson_16.npy", ndim=2),
            dtype=np.float64,
        )
        high = np.asarray(
            _load_numeric_array(output_dir / "arrays/lid_hutchinson_64.npy", ndim=2),
            dtype=np.float64,
        )
    except (GlobalCampaignError, OSError, KeyError, ValueError) as exc:
        return [f"cannot validate high-dimensional FM diagnostics: {exc}"]
    expected_manifest_fields = {"schema_version", "protocol_id", "outputs"}
    if (
        set(manifest) != expected_manifest_fields
        or manifest.get("schema_version") != 1
        or manifest.get("protocol_id") != FM_HIGH_DIM_PROTOCOL
    ):
        errors.append("high-dimensional FM diagnostic manifest differs")
    try:
        actual_outputs = _output_inventory(output_dir)
    except GlobalCampaignError as exc:
        errors.append(str(exc))
    else:
        if not _same_json(manifest.get("outputs"), actual_outputs):
            errors.append("high-dimensional FM diagnostic outputs differ")
    expected_metadata_fields = {
        "schema_version",
        "protocol_id",
        "variant_id",
        "ambient_dim",
        "checkpoint_sha256",
        "outer_selection_curve_sha256",
        "raw_query_sha256",
        "query_subset_sha256",
        "query_subset_indices_sha256",
        "execution_contract",
        "trace_seed",
        "trace_probe_counts",
        "shared_rademacher_prefix",
        "scale_policy",
        "exact_trace_status",
        "oracle_status",
        "batch_size",
    }
    execution = model.get("diagnostics_execution")
    diagnostics = model.get("diagnostics")
    training = model.get("training")
    expected_variant = (
        training.get("flow_variant_id") if isinstance(training, Mapping) else None
    )
    outer_curve_sha = v1._array_sha256(selection_curve)
    if set(metadata) != expected_metadata_fields:
        errors.append("high-dimensional FM diagnostic metadata fields differ")
    if (
        metadata.get("schema_version") != 1
        or metadata.get("protocol_id") != FM_HIGH_DIM_PROTOCOL
        or metadata.get("variant_id") != expected_variant
        or metadata.get("ambient_dim")
        != model.get("capacity_alignment", {}).get("ambient_dim")
        or metadata.get("checkpoint_sha256") != checkpoint_sha256
        or metadata.get("outer_selection_curve_sha256") != outer_curve_sha
        or metadata.get("raw_query_sha256")
        != partition.get("selection_features_fm_sha256")
        or not _same_json(metadata.get("execution_contract"), execution)
        or metadata.get("trace_seed") != 0
        or tuple(metadata.get("trace_probe_counts") or ()) != FM_HIGH_DIM_TRACE_PROBES
        or metadata.get("shared_rademacher_prefix") is not True
        or metadata.get("scale_policy") != "actual_selection_grid"
        or metadata.get("exact_trace_status") != "skipped_cost_prohibitive"
        or metadata.get("oracle_status") != "skipped_cost_prohibitive"
        or not isinstance(diagnostics, Mapping)
        or metadata.get("batch_size") != diagnostics.get("batch_size")
    ):
        errors.append("high-dimensional FM diagnostic binding differs")
    expected_scales = np.asarray(scales, dtype=np.float64)
    if not np.array_equal(diagnostic_scales, expected_scales):
        errors.append("high-dimensional FM diagnostic scale grid differs")
    n_selection = partition.get("n_train_selection")
    try:
        expected_indices = _high_dim_subset_indices(int(n_selection))
    except (TypeError, ValueError, GlobalCampaignError):
        expected_indices = np.asarray([], dtype=np.int64)
        errors.append("high-dimensional FM diagnostic subset contract is invalid")
    if not np.array_equal(subset_indices, expected_indices):
        errors.append("high-dimensional FM diagnostic subset indices differ")
    if query_subset.shape != (
        subset_indices.size,
        int(model.get("capacity_alignment", {}).get("ambient_dim", -1)),
    ):
        errors.append("high-dimensional FM diagnostic query subset shape differs")
    if metadata.get("query_subset_sha256") != v1._array_sha256(query_subset):
        errors.append("high-dimensional FM diagnostic query subset SHA differs")
    if (
        expected_query_subset_sha256 is not None
        and metadata.get("query_subset_sha256") != expected_query_subset_sha256
    ):
        errors.append("high-dimensional FM query subset differs from fresh source")
    if metadata.get("query_subset_indices_sha256") != v1._array_sha256(subset_indices):
        errors.append("high-dimensional FM diagnostic subset-index SHA differs")
    expected_shape = (subset_indices.size, diagnostic_scales.size)
    if low.shape != expected_shape or high.shape != expected_shape:
        errors.append("high-dimensional FM diagnostic prediction shape differs")
    try:
        aggregate_metrics = _hutchinson_stability_metrics(low, high)
        per_scale = [
            {
                "scale_index": index,
                "noise_ratio": float(diagnostic_scales[index]),
                "metrics": _hutchinson_stability_metrics(
                    low[:, index : index + 1], high[:, index : index + 1]
                ),
            }
            for index in range(diagnostic_scales.size)
        ]
    except (GlobalCampaignError, ValueError) as exc:
        errors.append(f"cannot recompute high-dimensional FM stability: {exc}")
        aggregate_metrics = None
        per_scale = None
    expected_summary = {
        "schema_version": 1,
        "protocol_id": FM_HIGH_DIM_PROTOCOL,
        "status": "completed_hutchinson_prefix_stability_v1",
        "ambient_dim": model.get("capacity_alignment", {}).get("ambient_dim"),
        "query_subset_size": int(subset_indices.size),
        "scale_count": int(diagnostic_scales.size),
        "trace_probe_counts": list(FM_HIGH_DIM_TRACE_PROBES),
        "shared_rademacher_prefix": True,
        "exact_trace_status": "skipped_cost_prohibitive",
        "oracle_status": "skipped_cost_prohibitive",
        "aggregate_metrics": aggregate_metrics,
        "per_scale": per_scale,
    }
    if not _same_json(summary, expected_summary):
        errors.append("high-dimensional FM diagnostic summary differs")
    try:
        expected_fm = _high_dim_fm_attestation(output_dir)
    except (GlobalCampaignError, OSError, KeyError) as exc:
        errors.append(f"cannot attest high-dimensional FM diagnostics: {exc}")
    else:
        if not _same_json(fm, expected_fm):
            errors.append("high-dimensional FM summary attestation differs")
    return errors


def run_known_affine_diagnostics(
    output_dir: Path,
    *,
    trained: Any,
    partition: HoldoutPartition,
    scales: npt.NDArray[np.float64],
    selection_curve: npt.NDArray[np.float64],
    model: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Run exact low-D diagnostics or a bounded Hutchinson prefix audit."""

    execution = model.get("diagnostics_execution")
    if not isinstance(execution, Mapping):
        raise GlobalCampaignError("v2 affine model lacks diagnostic execution identity")
    if execution.get("mode") == "small_dim_exhaustive_exact":
        return v1.run_known_affine_diagnostics(
            output_dir,
            trained=trained,
            partition=partition,
            scales=scales,
            selection_curve=selection_curve,
            model=model,
        )
    if execution.get("mode") != "high_dim_hutchinson_convergence":
        raise GlobalCampaignError("v2 affine diagnostic execution mode is invalid")
    if partition.selection_target is None:
        raise GlobalCampaignError("known-LID FM diagnostics require train targets")
    if output_dir.exists():
        fm = _high_dim_fm_attestation(output_dir)
        validation = _validate_high_dim_fm_diagnostics(
            output_dir,
            model=model,
            checkpoint_sha256=getattr(trained, "checkpoint_sha256", None),
            scales=scales,
            selection_curve=selection_curve,
            partition=partition.record,
            fm=fm,
        )
        if validation:
            raise GlobalCampaignError(
                f"high-dimensional FM diagnostics failed validation: {validation}"
            )
        return fm

    staging = output_dir.with_name(f".{output_dir.name}.incomplete")
    if staging.exists():
        if staging.is_symlink() or not staging.is_dir():
            raise GlobalCampaignError("FM diagnostic staging path is unsafe")
        shutil.rmtree(staging)
    arrays_dir = staging / "arrays"
    arrays_dir.mkdir(parents=True)
    indices = _high_dim_subset_indices(partition.selection_features.shape[0])
    query = np.ascontiguousarray(partition.selection_features[indices])
    lambda_grid = np.ascontiguousarray(np.asarray(scales, dtype=np.float64))
    _assert_lambdas_supported(model, lambda_grid)
    diagnostics = _mapping(model.get("diagnostics"), field="v2 FM diagnostics")
    from models.training import predict_lid

    predictions: dict[int, npt.NDArray[np.float64]] = {}
    for probes in FM_HIGH_DIM_TRACE_PROBES:
        predictions[probes] = np.ascontiguousarray(
            np.column_stack(
                [
                    predict_lid(
                        trained,
                        query,
                        float(scale),
                        family="independent_affine_flow",
                        readout="full",
                        divergence_backend="hutchinson",
                        trace_probes=probes,
                        trace_seed=0,
                        batch_size=int(diagnostics["batch_size"]),
                    )
                    for scale in lambda_grid
                ]
            ),
            dtype=np.float64,
        )
    checkpoint_sha = getattr(trained, "checkpoint_sha256", None)
    if not isinstance(checkpoint_sha, str) or not _SHA256.fullmatch(checkpoint_sha):
        raise GlobalCampaignError(
            "high-dimensional FM diagnostic checkpoint SHA is invalid"
        )
    _save_npy(arrays_dir / "query_subset_indices.npy", indices)
    _save_npy(arrays_dir / "query_subset.npy", query)
    _save_npy(arrays_dir / "scales.npy", lambda_grid)
    _save_npy(arrays_dir / "lid_hutchinson_16.npy", predictions[16])
    _save_npy(arrays_dir / "lid_hutchinson_64.npy", predictions[64])
    metadata = {
        "schema_version": 1,
        "protocol_id": FM_HIGH_DIM_PROTOCOL,
        "variant_id": model["training"]["flow_variant_id"],
        "ambient_dim": model["capacity_alignment"]["ambient_dim"],
        "checkpoint_sha256": checkpoint_sha,
        "outer_selection_curve_sha256": v1._array_sha256(selection_curve),
        "raw_query_sha256": partition.record["selection_features_fm_sha256"],
        "query_subset_sha256": v1._array_sha256(query),
        "query_subset_indices_sha256": v1._array_sha256(indices),
        "execution_contract": _plain(execution),
        "trace_seed": 0,
        "trace_probe_counts": list(FM_HIGH_DIM_TRACE_PROBES),
        "shared_rademacher_prefix": True,
        "scale_policy": "actual_selection_grid",
        "exact_trace_status": "skipped_cost_prohibitive",
        "oracle_status": "skipped_cost_prohibitive",
        "batch_size": int(diagnostics["batch_size"]),
    }
    summary = {
        "schema_version": 1,
        "protocol_id": FM_HIGH_DIM_PROTOCOL,
        "status": "completed_hutchinson_prefix_stability_v1",
        "ambient_dim": model["capacity_alignment"]["ambient_dim"],
        "query_subset_size": int(indices.size),
        "scale_count": int(lambda_grid.size),
        "trace_probe_counts": list(FM_HIGH_DIM_TRACE_PROBES),
        "shared_rademacher_prefix": True,
        "exact_trace_status": "skipped_cost_prohibitive",
        "oracle_status": "skipped_cost_prohibitive",
        "aggregate_metrics": _hutchinson_stability_metrics(
            predictions[16], predictions[64]
        ),
        "per_scale": [
            {
                "scale_index": index,
                "noise_ratio": float(lambda_grid[index]),
                "metrics": _hutchinson_stability_metrics(
                    predictions[16][:, index : index + 1],
                    predictions[64][:, index : index + 1],
                ),
            }
            for index in range(lambda_grid.size)
        ],
    }
    _write_json(staging / "metadata.json", metadata)
    _write_json(staging / "summary.json", summary)
    _write_json(
        staging / "manifest.json",
        {
            "schema_version": 1,
            "protocol_id": FM_HIGH_DIM_PROTOCOL,
            "outputs": _output_inventory(staging),
        },
    )
    fm = _high_dim_fm_attestation(staging)
    validation = _validate_high_dim_fm_diagnostics(
        staging,
        model=model,
        checkpoint_sha256=checkpoint_sha,
        scales=lambda_grid,
        selection_curve=selection_curve,
        partition=partition.record,
        fm=fm,
    )
    if validation:
        raise GlobalCampaignError(
            f"high-dimensional FM diagnostics failed validation: {validation}"
        )
    os.replace(staging, output_dir)
    return _high_dim_fm_attestation(output_dir)


def select_supervised_bounded(
    initial_lambdas: npt.ArrayLike,
    initial_curve: npt.ArrayLike,
    target: npt.ArrayLike,
    *,
    evaluate: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    tolerance: float = 1.0e-12,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], int, dict[str, Any]]:
    lambdas = np.asarray(initial_lambdas, dtype=np.float64)
    curve = np.asarray(initial_curve, dtype=np.float64)
    target_array = np.ravel(np.asarray(target, dtype=np.float64))
    if curve.shape != (target_array.size, lambdas.size):
        raise GlobalCampaignError("supervised selection curve shape differs")

    def choose() -> tuple[int, npt.NDArray[np.float64]]:
        errors = np.mean(np.abs(curve - target_array[:, None]), axis=0)
        minimum = float(errors.min())
        # Protocol v2 always breaks a scientific tie toward lower physical lambda.
        return int(np.flatnonzero(errors <= minimum + tolerance)[0]), errors

    index, errors = choose()
    initial_index = index
    extended_side: str | None = None
    if index in {0, lambdas.size - 1}:
        if index == 0:
            proposed = lambdas[0] / np.sqrt(2.0) ** np.arange(4, 0, -1)
            proposed = proposed[proposed >= COMMON_LAMBDA_MIN - 1e-15]
            extended_side = "lower"
        else:
            proposed = lambdas[-1] * np.sqrt(2.0) ** np.arange(1, 5)
            proposed = proposed[proposed <= COMMON_LAMBDA_MAX + 1e-12]
            extended_side = "upper"
        if proposed.size:
            extension_curve = np.asarray(
                evaluate(np.asarray(proposed, dtype=np.float64)), dtype=np.float64
            )
            if extension_curve.shape != (target_array.size, proposed.size):
                raise GlobalCampaignError("bounded extension curve shape differs")
            if extended_side == "lower":
                lambdas = np.concatenate((proposed, lambdas))
                curve = np.column_stack((extension_curve, curve))
            else:
                lambdas = np.concatenate((lambdas, proposed))
                curve = np.column_stack((curve, extension_curve))
            index, errors = choose()
    unresolved = index in {0, lambdas.size - 1}
    diagnostics = {
        "criterion": "mae",
        "candidate_mae": [float(value) for value in errors],
        "minimum_mae": float(errors[index]),
        "tie_candidates": [
            int(value) for value in np.flatnonzero(errors <= errors[index] + tolerance)
        ],
        "tie_break": "smaller_lambda",
        "initial_selected_index": initial_index,
        "extended_side": extended_side,
        "scale_unresolved": unresolved,
        "status": "scale_unresolved" if unresolved else "selected",
    }
    return (
        np.ascontiguousarray(lambdas),
        np.ascontiguousarray(curve),
        index,
        diagnostics,
    )


def flipd_kneedle_pointwise(
    curve: npt.ArrayLike, ambient_dim: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    values = np.asarray(curve, dtype=np.float64)
    times = flipd_timesteps()
    if (
        values.ndim != 2
        or values.shape[1] != times.size
        or not np.isfinite(values).all()
    ):
        raise GlobalCampaignError("FLIPD Kneedle input curves must be finite N x 22")
    lids = np.empty(values.shape[0], dtype=np.float64)
    selected_times = np.full(values.shape[0], np.nan, dtype=np.float64)
    fallback = np.zeros(values.shape[0], dtype=np.bool_)
    for row_index, row in enumerate(values):
        locator = KneeLocator(
            times,
            row,
            S=1.0,
            curve="convex",
            direction="decreasing",
        )
        if locator.knee is None:
            lids[row_index] = float(ambient_dim)
            fallback[row_index] = True
        else:
            selected_times[row_index] = float(locator.knee)
            lids[row_index] = float(locator.knee_y)
    return lids, selected_times, fallback


def select_unknown_reference_kneedle(
    lambdas: npt.ArrayLike, curve: npt.ArrayLike
) -> tuple[int | None, dict[str, Any]]:
    scales = np.asarray(lambdas, dtype=np.float64)
    values = np.asarray(curve, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != scales.size
        or not np.isfinite(values).all()
    ):
        raise GlobalCampaignError("reference Kneedle curve shape differs")
    mean_curve = values.mean(axis=0)
    maximum_index = int(np.argmax(mean_curve))
    segment_scales = scales[maximum_index:]
    segment_curve = mean_curve[maximum_index:]
    if segment_scales.size < 3:
        return None, {
            "criterion": "reference_mean_kneedle_after_global_maximum",
            "global_maximum_index": maximum_index,
            "segment_size": int(segment_scales.size),
            "knee_vp_time": None,
            "selected_index": None,
            "selected_lambda": None,
            "status": "selection_failed",
            "failure_reason": "remaining_segment_too_short",
        }
    coordinates = vp_time_from_lambda(segment_scales)
    locator = KneeLocator(
        coordinates,
        segment_curve,
        S=1.0,
        curve="convex",
        direction="decreasing",
        online=False,
        interp_method="interp1d",
    )
    if locator.knee is None:
        return None, {
            "criterion": "reference_mean_kneedle_after_global_maximum",
            "global_maximum_index": maximum_index,
            "segment_size": int(segment_scales.size),
            "knee_vp_time": None,
            "selected_index": None,
            "selected_lambda": None,
            "status": "selection_failed",
            "failure_reason": "no_knee",
        }
    local = int(np.argmin(np.abs(coordinates - float(locator.knee))))
    if local in {0, segment_scales.size - 1}:
        return None, {
            "criterion": "reference_mean_kneedle_after_global_maximum",
            "global_maximum_index": maximum_index,
            "segment_size": int(segment_scales.size),
            "knee_vp_time": float(locator.knee),
            "selected_index": None,
            "selected_lambda": None,
            "status": "selection_failed",
            "failure_reason": "knee_at_boundary",
        }
    selected = maximum_index + local
    return selected, {
        "criterion": "reference_mean_kneedle_after_global_maximum",
        "global_maximum_index": maximum_index,
        "segment_size": int(segment_scales.size),
        "knee_vp_time": float(locator.knee),
        "selected_index": selected,
        "selected_lambda": float(scales[selected]),
        "status": "selected",
    }


def validate_canary_report(
    path: Path,
    *,
    campaign_identity: str,
    declared_source_sha256: str,
    campaign_config_sha256: str,
    input_inventory_sha256: str,
    canary_config_sha256: str | None = None,
) -> dict[str, Any]:
    try:
        from experiments.v2_canary import load_and_validate_canary_report

        return load_and_validate_canary_report(
            path,
            expected_declared_source_sha256=declared_source_sha256,
            expected_campaign_identity=campaign_identity,
            expected_campaign_config_sha256=campaign_config_sha256,
            expected_input_inventory_sha256=input_inventory_sha256,
            expected_config_sha256=canary_config_sha256,
        )
    except Exception as exc:
        raise GlobalCampaignError(f"v2 canary attestation is invalid: {exc}") from exc


def _cell_identity(
    *,
    campaign_id: str,
    campaign_config_sha: str,
    source_sha: str,
    model_plan: ModelPlan,
    cell: CampaignCell,
    cell_data: Any,
    selection_contract: Mapping[str, Any],
    evaluation_contract: Mapping[str, Any],
) -> dict[str, Any]:
    model = resolve_cell_model(model_plan, cell_data.input_record)
    return {
        "schema_version": 2,
        "campaign_id": campaign_id,
        "campaign_config_sha256": campaign_config_sha,
        "source_tree_sha256": source_sha,
        "model": {
            "variant_id": model_plan.variant_id,
            "experiment_name": model_plan.experiment_name,
            "model": model,
        },
        "cell": _plain(cell),
        "input_sha256": cell_data.input_sha256,
        "selection_contract": _plain(selection_contract),
        "evaluation_contract": _plain(evaluation_contract),
    }


def _model_readouts(model: Mapping[str, Any]) -> tuple[str, ...]:
    family = model["family"]
    if family == "independent_affine_flow":
        return ("response", "full", "fm_to_score")
    if family in {"rectified_flow", "schrodinger_bridge"}:
        return ("response", "full")
    return (str(model["primary_readout"]),)


def _predict(
    predict_fn: PredictFunction,
    trained: Any,
    query: npt.ArrayLike,
    noise_ratio: float,
    *,
    model: Mapping[str, Any],
    seed: int,
    batch_size: int,
    readout: str,
) -> npt.NDArray[np.float64]:
    include_nf_stencil = model["family"] == "scale_conditioned_nf" and readout == "ols5"
    _assert_lambdas_supported(
        model, (noise_ratio,), include_nf_stencil=include_nf_stencil
    )
    if model["family"] == "scale_conditioned_nf" and readout == "ols5":
        from models.training import predict_nf_lid_ols5

        prediction = predict_nf_lid_ols5(
            trained,
            query,
            float(noise_ratio),
            family="scale_conditioned_nf",
            ols_log_step=0.05,
            batch_size=batch_size,
        )
    else:
        prediction = predict_fn(
            trained,
            query,
            _native_coordinate(model, float(noise_ratio)),
            family=str(model["family"]),
            readout=(str(model["readout"]) if readout == "ols5" else readout),
            divergence_backend=str(model["derivative_backend"]),
            trace_probes=int(model["trace_probes"]),
            trace_seed=seed,
            batch_size=batch_size,
        )
    values = np.ravel(np.asarray(prediction, dtype=np.float64))
    if values.shape != (np.asarray(query).shape[0],) or not np.isfinite(values).all():
        raise GlobalCampaignError("v2 prediction is not a finite pointwise vector")
    return np.ascontiguousarray(values)


def _prediction_curve(
    predict_fn: PredictFunction,
    trained: Any,
    query: npt.ArrayLike,
    lambdas: npt.ArrayLike,
    *,
    model: Mapping[str, Any],
    seed: int,
    batch_size: int,
    readout: str,
) -> npt.NDArray[np.float64]:
    scales = np.asarray(lambdas, dtype=np.float64)
    _assert_lambdas_supported(
        model,
        scales,
        include_nf_stencil=model["family"] == "scale_conditioned_nf"
        and readout == "ols5",
    )
    return np.ascontiguousarray(
        np.column_stack(
            [
                _predict(
                    predict_fn,
                    trained,
                    query,
                    float(scale),
                    model=model,
                    seed=seed,
                    batch_size=batch_size,
                    readout=readout,
                )
                for scale in scales
            ]
        ),
        dtype=np.float64,
    )


def _training_history_attestation(trained: Any) -> dict[str, Any]:
    raw_history = getattr(trained, "history", None)
    metadata = getattr(trained, "weights_metadata", None)
    if isinstance(raw_history, (str, bytes)) or not isinstance(raw_history, Sequence):
        raise GlobalCampaignError("v2 trainer history must be a sequence")
    if not isinstance(metadata, Mapping):
        raise GlobalCampaignError("v2 trainer lacks weights metadata")
    rows: list[dict[str, Any]] = []
    for raw in raw_history:
        value = raw.to_dict() if callable(getattr(raw, "to_dict", None)) else raw
        row = _mapping(value, field="v2 training history row")
        if set(row) != {
            "step",
            "examples_seen",
            "train_loss",
            "validation_loss",
            "learning_rate",
        }:
            raise GlobalCampaignError("v2 training history row fields differ")
        if type(row["step"]) is not int or type(row["examples_seen"]) is not int:
            raise GlobalCampaignError("v2 training step counters are invalid")
        if any(
            isinstance(row[field], bool)
            or not isinstance(row[field], (int, float))
            or not math.isfinite(float(row[field]))
            for field in ("train_loss", "validation_loss", "learning_rate")
        ):
            raise GlobalCampaignError("v2 training metrics must be finite")
        rows.append(_plain(row))
    if not rows or any(left["step"] >= right["step"] for left, right in pairwise(rows)):
        raise GlobalCampaignError("v2 training steps must strictly increase")
    expected_metadata_fields = {
        "schema_version",
        "selection",
        "initial",
        "selected",
        "final",
    }
    if set(metadata) != expected_metadata_fields:
        raise GlobalCampaignError("v2 weights metadata fields differ")
    if (
        metadata.get("schema_version") != 1
        or metadata.get("selection") != "minimum_train_selection_native_loss_v1"
    ):
        raise GlobalCampaignError("v2 weight selection metadata differs")
    for field, kind in (("selected", "validation_best"), ("final", "final")):
        record = metadata.get(field)
        if not isinstance(record, Mapping) or set(record) != {
            "kind",
            "step",
            "examples_seen",
            "state_sha256",
        }:
            raise GlobalCampaignError(f"v2 {field} weight record differs")
        if (
            record["kind"] != kind
            or not isinstance(record["state_sha256"], str)
            or not _SHA256.fullmatch(record["state_sha256"])
        ):
            raise GlobalCampaignError(f"v2 {field} weight identity is invalid")
    return {"status": "complete", "steps": rows, "weights": _plain(metadata)}


def _training_attestation(
    trained: Any,
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
    model: Mapping[str, Any],
) -> dict[str, Any]:
    training = v1._canonical_training_config_record(
        model["training"], field="v2 Hydra training config"
    )
    history = _training_history_attestation(trained)
    weights = history["weights"]
    total_examples = 32000 * 256
    if (
        weights["final"]["step"] != 32000
        or weights["final"]["examples_seen"] != total_examples
    ):
        raise GlobalCampaignError("v2 final weights do not exhaust the fixed budget")
    if weights["selected"]["examples_seen"] != weights["selected"]["step"] * 256:
        raise GlobalCampaignError("v2 selected examples-seen counter differs")
    return {
        "schema_version": 2,
        "model_family": str(model["family"]),
        "training_config_sha256": sha256_bytes(canonical_json(training).encode()),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint_retention": CHECKPOINT_RETENTION_PRUNE_AFTER_EVALUATION,
        "capacity_alignment": _plain(model["capacity_alignment"]),
        "history": history,
    }


def _run_cell(
    *,
    campaign_root: Path,
    campaign_id: str,
    campaign_config_sha: str,
    source_sha: str,
    config: Mapping[str, Any],
    model_plan: ModelPlan,
    cell: CampaignCell,
    data: CellData,
    reference_summary: Mapping[str, Any] | None,
    train_fn: TrainFunction,
    predict_fn: PredictFunction,
    load_checkpoint_fn: Callable[..., Any],
    affine_diagnostics_fn: Callable[..., Mapping[str, Any]],
    callback: Callable[[str, Mapping[str, Any]], None] | None,
    cell_diagnostics_fn: Callable[[Mapping[str, Any], Path], Mapping[str, Any]]
    | None = None,
) -> tuple[Path, dict[str, Any]]:
    model = resolve_cell_model(model_plan, data.input_record)
    cell_plan = replace(model_plan, model=model)
    identity = _cell_identity(
        campaign_id=campaign_id,
        campaign_config_sha=campaign_config_sha,
        source_sha=source_sha,
        model_plan=model_plan,
        cell=cell,
        cell_data=data,
        selection_contract=config["campaign"]["selection"],
        evaluation_contract=config["campaign"]["evaluation"],
    )
    cell_id = sha256_bytes(canonical_json(identity).encode())[:20]
    suite_dir = (
        campaign_root
        / "runs"
        / _safe_component(model_plan.variant_id)
        / _safe_component(cell.suite_id)
    )
    label = f"{_safe_component(cell.dataset)}__{_safe_component(cell.representation)}__{cell_id}"
    final_dir = suite_dir / label
    if final_dir.exists():
        errors = validate_global_cell(
            final_dir, expected_identity=identity, reference_summary=reference_summary
        )
        if errors:
            raise GlobalCampaignError(f"refusing to reuse invalid v2 cell: {errors}")
        summary = _load_json(final_dir / "summary.json")
        summary["summary_sha256"] = sha256_path(final_dir / "summary.json")
        _emit(callback, "cell.reused", model_plan=cell_plan, cell=cell, cell_id=cell_id)
        return final_dir, summary

    suite_dir.mkdir(parents=True, exist_ok=True)
    work_dir = suite_dir / f".{label}.incomplete"
    identity_path = work_dir / "identity.json"
    if work_dir.exists():
        if not identity_path.is_file() or not _same_json(
            _load_json(identity_path), identity
        ):
            raise GlobalCampaignError("stable v2 incomplete cell has another identity")
    else:
        work_dir.mkdir()
        _write_json(identity_path, identity)
        _write_yaml(work_dir / "resolved_model.yaml", model)
    _write_json(work_dir / "input_record.json", data.input_record)

    partition = partition_source_train(
        data.train,
        data.train_target,
        selection=config["campaign"]["selection"],
        seed=int(config["seed"]),
    )
    _save_npy(work_dir / "train_fit_indices.npy", partition.fit_indices)
    _save_npy(work_dir / "train_selection_indices.npy", partition.selection_indices)
    if partition.selection_target is not None:
        _save_npy(work_dir / "train_selection_target.npy", partition.selection_target)

    checkpoint = work_dir / "checkpoint.pt"
    progress = work_dir / str(
        config["campaign"]["resume"]["training_progress_filename"]
    )
    if (
        (work_dir / "manifest.json").is_file()
        and not checkpoint.exists()
        and not progress.exists()
    ):
        errors = validate_global_cell(
            work_dir, expected_identity=identity, reference_summary=reference_summary
        )
        if errors:
            raise GlobalCampaignError(f"pruned v2 recovery failed: {errors}")
        os.replace(work_dir, final_dir)
        summary = _load_json(final_dir / "summary.json")
        summary["summary_sha256"] = sha256_path(final_dir / "summary.json")
        return final_dir, summary

    def training_log(payload: Mapping[str, Any]) -> None:
        step = payload.get("step", payload.get("epoch"))
        _emit(
            callback,
            "cell.training.epoch",
            model_plan=cell_plan,
            cell=cell,
            step=step,
            training=dict(payload),
        )

    if not (checkpoint.is_file() and not progress.exists()):
        _emit(
            callback,
            "cell.started",
            model_plan=cell_plan,
            cell=cell,
            cell_id=cell_id,
            resumed=progress.is_file(),
        )
        v1._training_call(
            train_fn,
            family=str(model["family"]),
            partition=partition,
            training=model["training"],
            checkpoint=checkpoint,
            progress=progress,
            callback=training_log if callback is not None else None,
        )
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise GlobalCampaignError("v2 trainer did not write a checkpoint")
    trained = load_checkpoint_fn(checkpoint, device=str(model["training"]["device"]))
    v1._require_matching_training_configs(trained.config, model["training"])
    checkpoint_sha = sha256_path(checkpoint)
    if getattr(trained, "checkpoint_sha256", None) != checkpoint_sha:
        raise GlobalCampaignError("v2 checkpoint roundtrip SHA differs")
    attestation = _training_attestation(
        trained, checkpoint=checkpoint, checkpoint_sha256=checkpoint_sha, model=model
    )
    _write_json(work_dir / "training_attestation.json", attestation)

    execution = config["execution"]
    batch_size = int(
        execution["evaluation_batch_size_override"]
        if execution["evaluation_batch_size_override"] is not None
        else config["campaign"]["evaluation"]["batch_size"]
    )
    seed = int(config["seed"])
    primary = str(model["primary_readout"])
    readouts = _model_readouts(model)
    metrics: dict[str, Any] = {}
    protocol_records: dict[str, Any] = {}
    all_candidate_lambdas: list[float] = []
    reference_binding: Mapping[str, Any] | None = None

    if cell.target_policy == "known_lid":
        if partition.selection_target is None:
            raise GlobalCampaignError("known-LID v2 cell lacks selection target")
        initial = initial_supervised_lambdas()
        initial_curve = _prediction_curve(
            predict_fn,
            trained,
            partition.selection_features,
            initial,
            model=model,
            seed=seed,
            batch_size=batch_size,
            readout=primary,
        )

        def evaluate_extension(
            scales: npt.NDArray[np.float64],
        ) -> npt.NDArray[np.float64]:
            return _prediction_curve(
                predict_fn,
                trained,
                partition.selection_features,
                scales,
                model=model,
                seed=seed,
                batch_size=batch_size,
                readout=primary,
            )

        selection_lambdas, selection_curve, selected_index, diagnostics = (
            select_supervised_bounded(
                initial,
                initial_curve,
                partition.selection_target,
                evaluate=evaluate_extension,
                tolerance=float(config["campaign"]["selection"]["tie_tolerance"]),
            )
        )
        selected_lambda = float(selection_lambdas[selected_index])
        _save_npy(work_dir / "selection_a_lambdas.npy", selection_lambdas)
        _save_npy(work_dir / "train_selection_curve__supervised.npy", selection_curve)
        protocol_records[KNOWN_SUPERVISED_PROTOCOL] = {
            "selected_index": selected_index,
            "selected_lambda": selected_lambda,
            "selection": diagnostics,
            "selection_uses_lid_targets": True,
        }
        metrics[KNOWN_SUPERVISED_PROTOCOL] = {"validation": {}, "test": {}}
        metrics[KNOWN_KNEEDLE_PROTOCOL] = {"validation": {}, "test": {}}
        kneedle_times = flipd_timesteps()
        kneedle_lambdas = vp_lambda_from_time(kneedle_times)
        _assert_lambdas_supported(
            model,
            kneedle_lambdas,
            include_nf_stencil=model["family"] == "scale_conditioned_nf",
        )
        _save_npy(work_dir / "selection_b_vp_times.npy", kneedle_times)
        _save_npy(work_dir / "selection_b_lambdas.npy", kneedle_lambdas)
        protocol_records[KNOWN_KNEEDLE_PROTOCOL] = {
            "source_revision": FLIPD_SOURCE_REVISION,
            "kneed_version": PINNED_KNEED_VERSION,
            "pointwise": True,
            "selection_uses_lid_targets": False,
        }
        for split, query, target in (
            ("validation", data.validation, data.validation_target),
            ("test", data.test, data.test_target),
        ):
            if target is None:
                raise GlobalCampaignError("known-LID v2 split lacks targets")
            for readout in readouts:
                prediction = _predict(
                    predict_fn,
                    trained,
                    query,
                    selected_lambda,
                    model=model,
                    seed=seed,
                    batch_size=batch_size,
                    readout=readout,
                )
                _save_npy(
                    work_dir / f"{split}_prediction__{readout}__supervised.npy",
                    prediction,
                )
                metrics[KNOWN_SUPERVISED_PROTOCOL][split][readout] = known_lid_metrics(
                    prediction, target
                )
            kneedle_curve = _prediction_curve(
                predict_fn,
                trained,
                query,
                kneedle_lambdas,
                model=model,
                seed=seed,
                batch_size=batch_size,
                readout=primary,
            )
            knee_lid, knee_time, fallback = flipd_kneedle_pointwise(
                kneedle_curve, _feature_ambient_dim(data.input_record)
            )
            _save_npy(
                work_dir / f"{split}_curve__{primary}__kneedle.npy", kneedle_curve
            )
            _save_npy(
                work_dir / f"{split}_prediction__{primary}__kneedle.npy", knee_lid
            )
            _save_npy(
                work_dir / f"{split}_knee_time__{primary}.npy",
                np.where(fallback, -1.0, knee_time),
            )
            _save_npy(
                work_dir / f"{split}_knee_fallback__{primary}.npy",
                fallback.astype(np.uint8),
            )
            metrics[KNOWN_KNEEDLE_PROTOCOL][split][primary] = {
                **known_lid_metrics(knee_lid, target),
                "no_knee_n": int(fallback.sum()),
                "no_knee_fraction": float(fallback.mean()),
            }
        _save_npy(work_dir / "validation_target.npy", data.validation_target)
        _save_npy(work_dir / "test_target.npy", data.test_target)
        all_candidate_lambdas.extend(selection_lambdas.tolist())
        all_candidate_lambdas.extend(kneedle_lambdas.tolist())
        selection_protocol = KNOWN_SUPERVISED_PROTOCOL
    elif cell.target_policy in {"sample_size", "paired_delta"}:
        selection_lambdas = unknown_reference_lambdas()
        if cell.reference_dataset not in {None, cell.dataset}:
            if reference_summary is None:
                raise GlobalCampaignError("v2 dependent cell lacks reference summary")
            reference_selection = reference_summary["protocols"][
                UNKNOWN_REFERENCE_PROTOCOL
            ]["selection"]
            if reference_selection.get("status") == "selection_failed":
                selected_index = None
                selected_lambda = None
                diagnostics = {
                    "criterion": "reuse_reference_mean_kneedle",
                    "reference_dataset": cell.reference_dataset,
                    "selected_index": None,
                    "selected_lambda": None,
                    "status": "selection_failed",
                    "failure_reason": "reference_selection_failed",
                    "reference_failure_reason": reference_selection.get(
                        "failure_reason"
                    ),
                }
            else:
                selected_index = int(reference_summary["selected_index"])
                selected_lambda = float(reference_summary["selected_scale"])
                if selected_lambda != float(selection_lambdas[selected_index]):
                    raise GlobalCampaignError(
                        "v2 reference lambda differs from common grid"
                    )
                diagnostics = {
                    "criterion": "reuse_reference_mean_kneedle",
                    "reference_dataset": cell.reference_dataset,
                    "selected_index": selected_index,
                    "selected_lambda": selected_lambda,
                    "status": "selected",
                }
            reference_binding = {
                "cell_key": f"{cell.suite_id}/{cell.reference_dataset}/{cell.representation}",
                "dataset": cell.reference_dataset,
                "representation": cell.representation,
                "selected_index": selected_index,
                "selected_lambda": selected_lambda,
                "summary_sha256": str(reference_summary["summary_sha256"]),
            }
            # Still retain this cell's target-free curve as evidence, but it cannot retune.
            selection_curve = _prediction_curve(
                predict_fn,
                trained,
                partition.selection_features,
                selection_lambdas,
                model=model,
                seed=seed,
                batch_size=batch_size,
                readout=primary,
            )
        else:
            selection_curve = _prediction_curve(
                predict_fn,
                trained,
                partition.selection_features,
                selection_lambdas,
                model=model,
                seed=seed,
                batch_size=batch_size,
                readout=primary,
            )
            selected_index, diagnostics = select_unknown_reference_kneedle(
                selection_lambdas, selection_curve
            )
            selected_lambda = (
                None
                if selected_index is None
                else float(selection_lambdas[selected_index])
            )
        _save_npy(work_dir / "reference_lambdas.npy", selection_lambdas)
        _save_npy(
            work_dir / "train_selection_curve__reference_kneedle.npy", selection_curve
        )
        protocol_records[UNKNOWN_REFERENCE_PROTOCOL] = {
            "selected_index": selected_index,
            "selected_lambda": selected_lambda,
            "selection": diagnostics,
            "selection_uses_lid_targets": False,
        }
        metrics[UNKNOWN_REFERENCE_PROTOCOL] = {"validation": {}, "test": {}}
        for split, query in (("validation", data.validation), ("test", data.test)):
            for readout in readouts:
                if selected_lambda is None:
                    metrics[UNKNOWN_REFERENCE_PROTOCOL][split][readout] = {
                        "status": "selection_failed",
                        "failure_reason": diagnostics["failure_reason"],
                    }
                    continue
                prediction = _predict(
                    predict_fn,
                    trained,
                    query,
                    selected_lambda,
                    model=model,
                    seed=seed,
                    batch_size=batch_size,
                    readout=readout,
                )
                _save_npy(
                    work_dir / f"{split}_prediction__{readout}__reference.npy",
                    prediction,
                )
                metrics[UNKNOWN_REFERENCE_PROTOCOL][split][readout] = (
                    prediction_summary(prediction)
                )
        all_candidate_lambdas.extend(selection_lambdas.tolist())
        selection_protocol = UNKNOWN_REFERENCE_PROTOCOL
    else:
        raise GlobalCampaignError("v2 cell target policy is unsupported")

    for split, labels in (
        ("validation", data.validation_labels),
        ("test", data.test_labels),
    ):
        if labels is not None:
            _save_npy(work_dir / f"{split}_labels.npy", labels)

    if (
        model["family"] == "independent_affine_flow"
        and cell.target_policy == "known_lid"
    ):
        fm_diagnostics = dict(
            affine_diagnostics_fn(
                work_dir / "fm_diagnostics",
                trained=trained,
                partition=partition,
                scales=selection_lambdas,
                selection_curve=selection_curve,
                model=model,
            )
        )
    elif model["family"] == "independent_affine_flow":
        fm_diagnostics = {
            "status": config["campaign"]["fm_diagnostics"]["unknown_lid_status"]
        }
    else:
        fm_diagnostics = {"status": "not_applicable_non_affine_model"}

    quality_diagnostics: Mapping[str, Any] = {"status": "not_requested"}
    if cell_diagnostics_fn is not None and (model_plan.variant_id, cell.key) in set(
        CANARY_TASKS
    ):
        diagnostic_dir = work_dir / "quality_diagnostics"
        diagnostic_dir.mkdir()
        context = {
            "trained": trained,
            "model_variant": model_plan.variant_id,
            "family": model["family"],
            "cell": cell,
            "cell_data": data,
            "partition": partition,
            "training_config": model["training"],
            "checkpoint_path": checkpoint,
            "worker_index": int(os.environ.get("LID_WORKER_SLOT", "-1")),
            "device": os.environ.get(
                "LID_VISIBLE_DEVICE", str(model["training"]["device"])
            ),
            "candidate_scales": np.asarray(
                sorted(set(all_candidate_lambdas)), dtype=np.float64
            ),
            "trace_seed": seed,
            "eval_batch_size": batch_size,
            "cell_identity": identity,
        }
        quality_diagnostics = _plain(cell_diagnostics_fn(context, diagnostic_dir))
        if (
            not isinstance(quality_diagnostics, Mapping)
            or quality_diagnostics.get("status") != "passed"
        ):
            raise GlobalCampaignError("v2 canary cell diagnostics did not pass")
        _write_json(work_dir / "quality_diagnostics.json", quality_diagnostics)

    summary = {
        "schema_version": 2,
        "campaign_id": campaign_id,
        "cell_id": cell_id,
        "model_variant": model_plan.variant_id,
        "suite_id": cell.suite_id,
        "dataset": cell.dataset,
        "representation": cell.representation,
        "target_policy": cell.target_policy,
        "primary_readout": primary,
        "selection_protocol": selection_protocol,
        "protocols": protocol_records,
        "selected_index": selected_index,
        "selected_scale": selected_lambda,
        "reference_binding": reference_binding,
        "partition": partition.record,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_retention": CHECKPOINT_RETENTION_PRUNE_AFTER_EVALUATION,
        "training_attestation_sha256": sha256_path(
            work_dir / "training_attestation.json"
        ),
        "input_sha256": data.input_sha256,
        "evaluation_protocol": FROZEN_EVALUATION_PROTOCOL,
        "selection_uses_validation_targets": False,
        "selection_uses_test_targets": False,
        "metrics": metrics,
        "fm_diagnostics": fm_diagnostics,
        "quality_diagnostics": quality_diagnostics,
    }
    _write_json(work_dir / "summary.json", summary)
    summary["summary_sha256"] = sha256_path(work_dir / "summary.json")
    manifest = {
        "schema_version": 2,
        "identity": identity,
        "selection_protocols": list(protocol_records),
        "evaluation_protocol": FROZEN_EVALUATION_PROTOCOL,
        "outputs": _output_inventory(
            work_dir, excluded_relative_paths=frozenset({"checkpoint.pt"})
        ),
    }
    _write_json(work_dir / "manifest.json", manifest)
    errors = validate_global_cell(
        work_dir,
        expected_identity=identity,
        reference_summary=reference_summary,
        allow_transient_prunable_checkpoint=True,
    )
    if errors:
        raise GlobalCampaignError(f"new v2 cell failed validation: {errors}")
    checkpoint.unlink()
    errors = validate_global_cell(
        work_dir, expected_identity=identity, reference_summary=reference_summary
    )
    if errors:
        raise GlobalCampaignError(f"pruned v2 cell failed validation: {errors}")
    os.replace(work_dir, final_dir)
    summary["summary_sha256"] = sha256_path(final_dir / "summary.json")
    _emit(
        callback,
        "cell.completed",
        model_plan=cell_plan,
        cell=cell,
        cell_id=cell_id,
        selected_scale=selected_lambda,
        metrics=metrics,
        shared_filesystem_cell_dir=str(final_dir),
    )
    return final_dir, summary


def validate_global_cell(
    directory: Path,
    *,
    expected_identity: Mapping[str, Any] | None = None,
    reference_summary: Mapping[str, Any] | None = None,
    expected_source_evidence: Mapping[str, Any] | None = None,
    allow_transient_prunable_checkpoint: bool = False,
) -> list[str]:
    root = Path(directory)
    errors: list[str] = []
    try:
        manifest = _load_json(root / "manifest.json")
        summary = _load_json(root / "summary.json")
        identity = manifest["identity"]
        input_record = _load_json(root / "input_record.json")
        resolved_model = yaml.safe_load(
            (root / "resolved_model.yaml").read_text(encoding="utf-8")
        )
        fit_indices = np.asarray(
            _load_numeric_array(root / "train_fit_indices.npy", ndim=1), dtype=np.int64
        )
        selection_indices = np.asarray(
            _load_numeric_array(root / "train_selection_indices.npy", ndim=1),
            dtype=np.int64,
        )
    except (
        GlobalCampaignError,
        OSError,
        KeyError,
        UnicodeError,
        yaml.YAMLError,
    ) as exc:
        return [f"cannot validate v2 core cell artifacts: {exc}"]
    if set(manifest) != {
        "schema_version",
        "identity",
        "selection_protocols",
        "evaluation_protocol",
        "outputs",
    }:
        errors.append("v2 cell manifest fields differ")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("evaluation_protocol") != FROZEN_EVALUATION_PROTOCOL
    ):
        errors.append("v2 cell manifest protocol/schema differs")
    if not isinstance(identity, Mapping):
        return errors + ["v2 cell identity is invalid"]
    if (
        set(identity)
        != {
            "schema_version",
            "campaign_id",
            "campaign_config_sha256",
            "source_tree_sha256",
            "model",
            "cell",
            "input_sha256",
            "selection_contract",
            "evaluation_contract",
        }
        or identity.get("schema_version") != 2
    ):
        errors.append("v2 cell identity fields/schema differ")
    if expected_identity is not None and not _same_json(identity, expected_identity):
        errors.append("v2 cell identity differs from expected")
    if identity.get("input_sha256") != sha256_bytes(
        canonical_json(input_record).encode()
    ):
        errors.append("v2 input record does not recompute identity")
    try:
        model = identity["model"]["model"]
        cell = identity["cell"]
    except (KeyError, TypeError):
        return errors + ["v2 cell model/inventory identity is invalid"]
    if not isinstance(model, Mapping) or not isinstance(cell, Mapping):
        return errors + ["v2 cell model/inventory identity is invalid"]
    if not _same_json(resolved_model, model):
        errors.append("v2 resolved model differs from identity")
    if summary.get("input_sha256") != identity.get("input_sha256"):
        errors.append("v2 summary input SHA differs")
    expected_summary_fields = {
        "schema_version",
        "campaign_id",
        "cell_id",
        "model_variant",
        "suite_id",
        "dataset",
        "representation",
        "target_policy",
        "primary_readout",
        "selection_protocol",
        "protocols",
        "selected_index",
        "selected_scale",
        "reference_binding",
        "partition",
        "checkpoint_sha256",
        "checkpoint_retention",
        "training_attestation_sha256",
        "input_sha256",
        "evaluation_protocol",
        "selection_uses_validation_targets",
        "selection_uses_test_targets",
        "metrics",
        "fm_diagnostics",
        "quality_diagnostics",
    }
    if set(summary) != expected_summary_fields or summary.get("schema_version") != 2:
        errors.append("v2 cell summary fields/schema differ")
    expected_cell_id = sha256_bytes(canonical_json(identity).encode("utf-8"))[:20]
    if summary.get("cell_id") != expected_cell_id:
        errors.append("v2 cell id does not recompute")
    for field in ("campaign_id",):
        if summary.get(field) != identity.get(field):
            errors.append(f"v2 summary {field} differs from identity")
    for field in ("suite_id", "dataset", "representation", "target_policy"):
        if summary.get(field) != cell.get(field):
            errors.append(f"v2 summary {field} differs from identity")
    if summary.get("model_variant") != identity.get("model", {}).get("variant_id"):
        errors.append("v2 summary model variant differs from identity")
    if summary.get("evaluation_protocol") != FROZEN_EVALUATION_PROTOCOL:
        errors.append("v2 summary evaluation protocol differs")
    if (
        summary.get("selection_uses_validation_targets") is not False
        or summary.get("selection_uses_test_targets") is not False
    ):
        errors.append("v2 summary permits benchmark-target tuning")
    if summary.get("primary_readout") != model.get("primary_readout"):
        errors.append("v2 primary readout differs from model")
    if (
        summary.get("checkpoint_retention")
        != CHECKPOINT_RETENTION_PRUNE_AFTER_EVALUATION
    ):
        errors.append("v2 checkpoint retention differs")
    checkpoint = root / "checkpoint.pt"
    if checkpoint.exists() and not allow_transient_prunable_checkpoint:
        errors.append("sealed v2 cell retains a checkpoint")
    if checkpoint.exists() and sha256_path(checkpoint) != summary.get(
        "checkpoint_sha256"
    ):
        errors.append("transient v2 checkpoint SHA differs")
    if any(root.glob("*training_progress*.pt")):
        errors.append("sealed v2 cell retains training progress")
    recorded_outputs = manifest.get("outputs")
    if not isinstance(recorded_outputs, Mapping):
        errors.append("v2 output inventory is invalid")
    else:
        try:
            actual_outputs = _output_inventory(
                root, excluded_relative_paths=frozenset({"checkpoint.pt"})
            )
        except GlobalCampaignError as exc:
            errors.append(str(exc))
        else:
            if not _same_json(recorded_outputs, actual_outputs):
                errors.append("v2 output inventory differs")

    if (
        fit_indices.size == 0
        or selection_indices.size == 0
        or np.intersect1d(fit_indices, selection_indices).size
        or np.unique(fit_indices).size != fit_indices.size
        or np.unique(selection_indices).size != selection_indices.size
    ):
        errors.append("v2 train fit/selection partition is invalid")
    partition = summary.get("partition")
    if not isinstance(partition, Mapping):
        errors.append("v2 partition record is invalid")
    else:
        if partition.get("fit_indices_sha256") != v1._array_sha(fit_indices):
            errors.append("v2 fit-index SHA differs")
        if partition.get("selection_indices_sha256") != v1._array_sha(
            selection_indices
        ):
            errors.append("v2 selection-index SHA differs")
        if partition.get("n_source_train") != fit_indices.size + selection_indices.size:
            errors.append("v2 partition sample count differs")
    if expected_source_evidence is not None:
        if not _same_json(partition, expected_source_evidence.get("partition")):
            errors.append("v2 partition differs from fresh source")
        if v1._array_sha(fit_indices) != expected_source_evidence.get(
            "fit_indices_sha256"
        ):
            errors.append("v2 fit indices differ from fresh source")
        if v1._array_sha(selection_indices) != expected_source_evidence.get(
            "selection_indices_sha256"
        ):
            errors.append("v2 selection indices differ from fresh source")
        target_artifacts = {
            "train_selection_target_sha256": root / "train_selection_target.npy",
            "validation_target_sha256": root / "validation_target.npy",
            "test_target_sha256": root / "test_target.npy",
            "validation_labels_sha256": root / "validation_labels.npy",
            "test_labels_sha256": root / "test_labels.npy",
        }
        for evidence_field, artifact in target_artifacts.items():
            actual = (
                v1._array_sha(_load_numeric_array(artifact, ndim=1))
                if artifact.is_file()
                else None
            )
            if actual != expected_source_evidence.get(evidence_field):
                errors.append(f"v2 {evidence_field} differs from fresh source")

    attestation_path = root / "training_attestation.json"
    try:
        attestation = _load_json(attestation_path)
    except GlobalCampaignError as exc:
        errors.append(f"cannot read v2 training attestation: {exc}")
    else:
        expected_attestation_fields = {
            "schema_version",
            "model_family",
            "training_config_sha256",
            "checkpoint_sha256",
            "checkpoint_size_bytes",
            "checkpoint_retention",
            "capacity_alignment",
            "history",
        }
        if set(attestation) != expected_attestation_fields:
            errors.append("v2 training attestation fields differ")
        if summary.get("training_attestation_sha256") != sha256_path(attestation_path):
            errors.append("v2 training attestation SHA differs")
        if (
            attestation.get("schema_version") != 2
            or attestation.get("checkpoint_sha256") != summary.get("checkpoint_sha256")
            or attestation.get("model_family") != model.get("family")
            or attestation.get("checkpoint_retention")
            != CHECKPOINT_RETENTION_PRUNE_AFTER_EVALUATION
            or type(attestation.get("checkpoint_size_bytes")) is not int
            or attestation.get("checkpoint_size_bytes", 0) <= 0
        ):
            errors.append("v2 training attestation identity differs")
        try:
            training = v1._canonical_training_config_record(
                model["training"], field="v2 cell training config"
            )
        except (GlobalCampaignError, KeyError):
            errors.append("v2 training config is invalid")
        else:
            if attestation.get("training_config_sha256") != sha256_bytes(
                canonical_json(training).encode()
            ):
                errors.append("v2 attested training config SHA differs")
        if not _same_json(
            attestation.get("capacity_alignment"), model.get("capacity_alignment")
        ):
            errors.append("v2 capacity attestation differs")
        history = attestation.get("history")
        if not isinstance(history, Mapping) or set(history) != {
            "status",
            "steps",
            "weights",
        }:
            errors.append("v2 training history attestation fields differ")
        else:
            steps = history.get("steps")
            weights = history.get("weights")
            if (
                history.get("status") != "complete"
                or not isinstance(steps, list)
                or not steps
            ):
                errors.append("v2 training history is incomplete")
            elif any(
                not isinstance(row, Mapping)
                or set(row)
                != {
                    "step",
                    "examples_seen",
                    "train_loss",
                    "validation_loss",
                    "learning_rate",
                }
                for row in steps
            ):
                errors.append("v2 training history row fields differ")
            if not isinstance(weights, Mapping) or set(weights) != {
                "schema_version",
                "selection",
                "initial",
                "selected",
                "final",
            }:
                errors.append("v2 weight metadata fields differ")
            else:
                selected = weights.get("selected")
                final = weights.get("final")
                if (
                    weights.get("schema_version") != 1
                    or weights.get("selection")
                    != "minimum_train_selection_native_loss_v1"
                    or not isinstance(selected, Mapping)
                    or not isinstance(final, Mapping)
                    or selected.get("kind") != "validation_best"
                    or final.get("kind") != "final"
                    or final.get("step") != 32000
                    or final.get("examples_seen") != 32000 * 256
                    or selected.get("examples_seen") != selected.get("step", -1) * 256
                ):
                    errors.append("v2 fixed-step weight metadata differs")

    protocols = summary.get("protocols")
    metrics = summary.get("metrics")
    if not isinstance(protocols, Mapping) or not isinstance(metrics, Mapping):
        return errors + ["v2 protocols/metrics are invalid"]
    expected_readouts = set(_model_readouts(model))
    primary = str(model["primary_readout"])
    target_policy = cell.get("target_policy")
    if target_policy == "known_lid":
        if set(protocols) != {KNOWN_SUPERVISED_PROTOCOL, KNOWN_KNEEDLE_PROTOCOL} or set(
            metrics
        ) != set(protocols):
            errors.append("known-LID v2 protocols differ")
        try:
            lambdas = np.asarray(
                _load_numeric_array(root / "selection_a_lambdas.npy", ndim=1),
                dtype=np.float64,
            )
            curve = np.asarray(
                _load_numeric_array(
                    root / "train_selection_curve__supervised.npy", ndim=2
                ),
                dtype=np.float64,
            )
            target = np.asarray(
                _load_numeric_array(root / "train_selection_target.npy", ndim=1),
                dtype=np.float64,
            )
            errors_by_scale = np.mean(np.abs(curve - target[:, None]), axis=0)
            expected_index = int(
                np.flatnonzero(errors_by_scale <= errors_by_scale.min() + 1.0e-12)[0]
            )
        except (GlobalCampaignError, ValueError, IndexError) as exc:
            errors.append(f"cannot recompute v2 supervised selection: {exc}")
            lambdas = np.asarray([], dtype=np.float64)
            expected_index = -1
        if summary.get("selected_index") != expected_index:
            errors.append("v2 supervised selected index does not recompute")
        elif lambdas.size and summary.get("selected_scale") != float(
            lambdas[expected_index]
        ):
            errors.append("v2 supervised selected lambda differs")
        record = protocols.get(KNOWN_SUPERVISED_PROTOCOL)
        if (
            not isinstance(record, Mapping)
            or record.get("selection_uses_lid_targets") is not True
        ):
            errors.append("v2 supervised protocol metadata differs")
        knee_record = protocols.get(KNOWN_KNEEDLE_PROTOCOL)
        if (
            not isinstance(knee_record, Mapping)
            or knee_record.get("selection_uses_lid_targets") is not False
        ):
            errors.append("v2 Kneedle protocol incorrectly uses targets")
        try:
            times = np.asarray(
                _load_numeric_array(root / "selection_b_vp_times.npy", ndim=1),
                dtype=np.float64,
            )
            knee_lambdas = np.asarray(
                _load_numeric_array(root / "selection_b_lambdas.npy", ndim=1),
                dtype=np.float64,
            )
            if not np.array_equal(times, flipd_timesteps()) or not np.allclose(
                knee_lambdas, vp_lambda_from_time(times), rtol=0, atol=0
            ):
                errors.append("v2 FLIPD grid differs from pinned source wrapper")
        except GlobalCampaignError as exc:
            errors.append(f"cannot validate v2 FLIPD grid: {exc}")
        for split in ("validation", "test"):
            try:
                split_target = np.asarray(
                    _load_numeric_array(root / f"{split}_target.npy", ndim=1),
                    dtype=np.float64,
                )
            except GlobalCampaignError as exc:
                errors.append(f"cannot load v2 {split} target: {exc}")
                continue
            declared_a = metrics.get(KNOWN_SUPERVISED_PROTOCOL, {}).get(split, {})
            if (
                not isinstance(declared_a, Mapping)
                or set(declared_a) != expected_readouts
            ):
                errors.append(f"v2 supervised {split} readouts differ")
            else:
                for readout in expected_readouts:
                    try:
                        prediction = np.asarray(
                            _load_numeric_array(
                                root / f"{split}_prediction__{readout}__supervised.npy",
                                ndim=1,
                            ),
                            dtype=np.float64,
                        )
                        expected_metrics = known_lid_metrics(prediction, split_target)
                    except (GlobalCampaignError, ValueError) as exc:
                        errors.append(
                            f"cannot validate v2 supervised {split}/{readout}: {exc}"
                        )
                    else:
                        if not _same_json(declared_a.get(readout), expected_metrics):
                            errors.append(
                                f"v2 supervised {split}/{readout} metrics differ"
                            )
            try:
                knee_curve = np.asarray(
                    _load_numeric_array(
                        root / f"{split}_curve__{primary}__kneedle.npy", ndim=2
                    ),
                    dtype=np.float64,
                )
                knee_prediction = np.asarray(
                    _load_numeric_array(
                        root / f"{split}_prediction__{primary}__kneedle.npy", ndim=1
                    ),
                    dtype=np.float64,
                )
                saved_time = np.asarray(
                    _load_numeric_array(
                        root / f"{split}_knee_time__{primary}.npy", ndim=1
                    ),
                    dtype=np.float64,
                )
                saved_fallback = np.asarray(
                    _load_numeric_array(
                        root / f"{split}_knee_fallback__{primary}.npy", ndim=1
                    ),
                    dtype=np.uint8,
                )
                expected_prediction, expected_time, expected_fallback = (
                    flipd_kneedle_pointwise(
                        knee_curve, int(model["capacity_alignment"]["ambient_dim"])
                    )
                )
                if (
                    not np.array_equal(knee_prediction, expected_prediction)
                    or not np.array_equal(
                        saved_fallback, expected_fallback.astype(np.uint8)
                    )
                    or not np.array_equal(
                        saved_time, np.where(expected_fallback, -1.0, expected_time)
                    )
                ):
                    errors.append(f"v2 pointwise Kneedle {split} outputs differ")
                expected_metrics = {
                    **known_lid_metrics(expected_prediction, split_target),
                    "no_knee_n": int(expected_fallback.sum()),
                    "no_knee_fraction": float(expected_fallback.mean()),
                }
                declared_b = metrics.get(KNOWN_KNEEDLE_PROTOCOL, {}).get(split, {})
                if not isinstance(declared_b, Mapping) or not _same_json(
                    declared_b.get(primary), expected_metrics
                ):
                    errors.append(f"v2 pointwise Kneedle {split} metrics differ")
            except (GlobalCampaignError, ValueError, KeyError) as exc:
                errors.append(f"cannot validate v2 pointwise Kneedle {split}: {exc}")
    elif target_policy in {"sample_size", "paired_delta"}:
        if set(protocols) != {UNKNOWN_REFERENCE_PROTOCOL} or set(metrics) != set(
            protocols
        ):
            errors.append("target-free v2 protocols differ")
        try:
            lambdas = np.asarray(
                _load_numeric_array(root / "reference_lambdas.npy", ndim=1),
                dtype=np.float64,
            )
            curve = np.asarray(
                _load_numeric_array(
                    root / "train_selection_curve__reference_kneedle.npy", ndim=2
                ),
                dtype=np.float64,
            )
            if not np.array_equal(lambdas, unknown_reference_lambdas()):
                errors.append("v2 reference grid differs")
        except GlobalCampaignError as exc:
            errors.append(f"cannot load v2 reference selection: {exc}")
            lambdas = np.asarray([], dtype=np.float64)
            curve = np.empty((0, 0))
        if cell.get("reference_dataset") in {None, cell.get("dataset")}:
            try:
                expected_index, expected_selection = select_unknown_reference_kneedle(
                    lambdas, curve
                )
            except (GlobalCampaignError, ValueError) as exc:
                errors.append(f"cannot recompute v2 reference Kneedle: {exc}")
            else:
                expected_scale = (
                    None if expected_index is None else float(lambdas[expected_index])
                )
                if (
                    summary.get("selected_index") != expected_index
                    or summary.get("selected_scale") != expected_scale
                    or not _same_json(
                        protocols[UNKNOWN_REFERENCE_PROTOCOL].get("selection"),
                        expected_selection,
                    )
                ):
                    errors.append("v2 reference Kneedle selection differs")
                if summary.get("reference_binding") is not None:
                    errors.append("v2 reference cell binds another cell")
        else:
            binding = summary.get("reference_binding")
            if not isinstance(binding, Mapping):
                errors.append("v2 dependent cell lacks reference binding")
            elif reference_summary is not None:
                reference_selection = (
                    reference_summary.get("protocols", {})
                    .get(UNKNOWN_REFERENCE_PROTOCOL, {})
                    .get("selection", {})
                )
                expected_status = (
                    "selection_failed"
                    if reference_selection.get("status") == "selection_failed"
                    else "selected"
                )
                declared_selection = protocols[UNKNOWN_REFERENCE_PROTOCOL].get(
                    "selection", {}
                )
                if (
                    binding.get("summary_sha256")
                    != reference_summary.get("summary_sha256")
                    or summary.get("selected_scale")
                    != reference_summary.get("selected_scale")
                    or summary.get("selected_index")
                    != reference_summary.get("selected_index")
                    or declared_selection.get("status") != expected_status
                ):
                    errors.append("v2 dependent reference binding differs")
        declared = metrics.get(UNKNOWN_REFERENCE_PROTOCOL, {})
        selection = protocols.get(UNKNOWN_REFERENCE_PROTOCOL, {}).get("selection", {})
        selection_failed = selection.get("status") == "selection_failed"
        if selection.get("status") not in {"selected", "selection_failed"}:
            errors.append("v2 reference selection status is invalid")
        for split in ("validation", "test"):
            declared_split = (
                declared.get(split, {}) if isinstance(declared, Mapping) else {}
            )
            if (
                not isinstance(declared_split, Mapping)
                or set(declared_split) != expected_readouts
            ):
                errors.append(f"v2 target-free {split} readouts differ")
                continue
            for readout in expected_readouts:
                prediction_path = root / f"{split}_prediction__{readout}__reference.npy"
                if selection_failed:
                    expected_failure = {
                        "status": "selection_failed",
                        "failure_reason": selection.get("failure_reason"),
                    }
                    if prediction_path.exists():
                        errors.append(
                            f"failed v2 selection stores {split}/{readout} prediction"
                        )
                    if not _same_json(declared_split.get(readout), expected_failure):
                        errors.append(
                            f"v2 target-free {split}/{readout} failure differs"
                        )
                    continue
                try:
                    prediction = np.asarray(
                        _load_numeric_array(prediction_path, ndim=1), dtype=np.float64
                    )
                    expected_metrics = prediction_summary(prediction)
                except (GlobalCampaignError, ValueError) as exc:
                    errors.append(
                        f"cannot validate v2 target-free {split}/{readout}: {exc}"
                    )
                else:
                    if not _same_json(declared_split.get(readout), expected_metrics):
                        errors.append(
                            f"v2 target-free {split}/{readout} metrics differ"
                        )
    else:
        errors.append("v2 target policy is unsupported")

    family = model.get("family")
    fm = summary.get("fm_diagnostics")
    if family == "independent_affine_flow" and target_policy == "known_lid":
        execution = model.get("diagnostics_execution")
        if not isinstance(fm, Mapping) or not isinstance(execution, Mapping):
            errors.append("known-LID v2 affine cell lacks FM diagnostics identity")
        elif execution.get("mode") == "small_dim_exhaustive_exact":
            if (
                fm.get("status") != "completed_strict_v2"
                or fm.get("path") != "fm_diagnostics"
            ):
                errors.append("small-D v2 affine cell lacks exact FM diagnostics")
            else:
                try:
                    from experiments.fm_diagnostics import validate_fm_diagnostics

                    diagnostic_errors = validate_fm_diagnostics(root / "fm_diagnostics")
                except Exception as exc:  # noqa: BLE001 - validator reports corruption
                    errors.append(
                        f"cannot validate v2 FM diagnostics: {type(exc).__name__}"
                    )
                else:
                    errors.extend(
                        f"v2 FM diagnostics: {error}" for error in diagnostic_errors
                    )
                    if "lambdas" in locals() and "curve" in locals():
                        errors.extend(
                            v1._validate_affine_diagnostic_binding(
                                root,
                                fm=fm,
                                model=model,
                                checkpoint_sha256=summary.get("checkpoint_sha256"),
                                scales=lambdas,
                                selection_curve=curve,
                                partition=(
                                    partition if isinstance(partition, Mapping) else {}
                                ),
                            )
                        )
        elif execution.get("mode") == "high_dim_hutchinson_convergence":
            if "lambdas" not in locals() or "curve" not in locals():
                errors.append(
                    "high-dimensional FM diagnostic selection inputs are missing"
                )
            else:
                errors.extend(
                    _validate_high_dim_fm_diagnostics(
                        root / "fm_diagnostics",
                        model=model,
                        checkpoint_sha256=summary.get("checkpoint_sha256"),
                        scales=lambdas,
                        selection_curve=curve,
                        partition=(partition if isinstance(partition, Mapping) else {}),
                        fm=fm,
                        expected_query_subset_sha256=(
                            None
                            if expected_source_evidence is None
                            else expected_source_evidence.get(
                                "fm_high_dim_query_subset_sha256"
                            )
                        ),
                    )
                )
        else:
            errors.append("known-LID v2 affine diagnostic execution mode differs")
    elif family == "independent_affine_flow":
        if not isinstance(fm, Mapping) or fm.get("status") != (
            "not_applicable_no_lid_targets"
        ):
            errors.append("target-free v2 affine diagnostic status differs")
    elif not isinstance(fm, Mapping) or fm.get("status") != (
        "not_applicable_non_affine_model"
    ):
        errors.append("non-affine v2 FM diagnostic status differs")

    quality = summary.get("quality_diagnostics")
    quality_path = root / "quality_diagnostics.json"
    if isinstance(quality, Mapping) and quality.get("status") != "not_requested":
        if not quality_path.is_file():
            errors.append("v2 quality diagnostics summary is missing")
        else:
            try:
                if not _same_json(quality, _load_json(quality_path)):
                    errors.append("v2 quality diagnostics record differs")
            except GlobalCampaignError as exc:
                errors.append(str(exc))
    elif quality_path.exists():
        errors.append("v2 unexpected quality diagnostics file")
    return errors


def _inventory_origin(cell: CampaignCell) -> str:
    return "canonical_35" if cell.exact_archive else "generated_e3_e4_extension_4"


def _protocol_predictions(
    directory: Path, split: str, suffix: str
) -> dict[str, npt.NDArray[np.float64]]:
    prefix = f"{split}_prediction__"
    ending = f"__{suffix}.npy"
    result: dict[str, npt.NDArray[np.float64]] = {}
    for path in sorted(directory.glob(f"{prefix}*{ending}")):
        readout = path.name[len(prefix) : -len(ending)]
        if not readout or readout in result:
            raise GlobalCampaignError(f"invalid v2 aggregate artifact {path}")
        result[readout] = np.asarray(
            _load_numeric_array(path, ndim=1), dtype=np.float64
        )
    if not result:
        raise GlobalCampaignError(
            f"aggregate cell {directory} has no {split}/{suffix} predictions"
        )
    return result


def recompute_model_aggregate(
    model_variant: str,
    cells: Sequence[CampaignCell],
    cell_directories: Mapping[str, Path],
) -> dict[str, Any]:
    """Recompute v2 logical analyses from the 39 sealed physical trainings."""

    expected_keys = [cell.key for cell in cells]
    if len(expected_keys) != EXPECTED_GLOBAL_CELL_COUNT or set(cell_directories) != set(
        expected_keys
    ):
        raise GlobalCampaignError(
            f"model {model_variant} aggregate does not cover exact 39-cell inventory"
        )
    summaries: dict[str, dict[str, Any]] = {}
    reference_predictions: dict[
        tuple[str, str], dict[str, npt.NDArray[np.float64]]
    ] = {}
    labels: dict[tuple[str, str], npt.NDArray[Any] | None] = {}
    known_lid: list[dict[str, Any]] = []
    primary_readout: str | None = None
    for cell in cells:
        directory = Path(cell_directories[cell.key])
        summary = _load_json(directory / "summary.json")
        if summary.get("model_variant") != model_variant:
            raise GlobalCampaignError(f"aggregate model identity differs in {cell.key}")
        summaries[cell.key] = summary
        current_primary = str(summary.get("primary_readout"))
        if primary_readout is None:
            primary_readout = current_primary
        elif primary_readout != current_primary:
            raise GlobalCampaignError("aggregate primary readout differs across cells")
        for split in ("validation", "test"):
            label_path = directory / f"{split}_labels.npy"
            labels[(cell.key, split)] = (
                _load_numeric_array(label_path, ndim=1) if label_path.exists() else None
            )
            if cell.target_policy == "known_lid":
                target = np.asarray(
                    _load_numeric_array(directory / f"{split}_target.npy", ndim=1),
                    dtype=np.float64,
                )
                supervised = _protocol_predictions(directory, split, "supervised")
                declared_a = summary["metrics"][KNOWN_SUPERVISED_PROTOCOL][split]
                for readout, prediction in supervised.items():
                    metrics = known_lid_metrics(prediction, target)
                    if not _same_json(declared_a.get(readout), metrics):
                        raise GlobalCampaignError(
                            f"aggregate supervised metrics differ: {cell.key}/{split}/{readout}"
                        )
                    known_lid.append(
                        {
                            "cell_key": cell.key,
                            "cell_id": summary["cell_id"],
                            "inventory_origin": _inventory_origin(cell),
                            "suite_id": cell.suite_id,
                            "dataset": cell.dataset,
                            "representation": cell.representation,
                            "split": split,
                            "readout": readout,
                            "is_primary_readout": readout == current_primary,
                            "selection_protocol": KNOWN_SUPERVISED_PROTOCOL,
                            "selected_index": summary["protocols"][
                                KNOWN_SUPERVISED_PROTOCOL
                            ]["selected_index"],
                            "selected_lambda": summary["protocols"][
                                KNOWN_SUPERVISED_PROTOCOL
                            ]["selected_lambda"],
                            "selection_status": summary["protocols"][
                                KNOWN_SUPERVISED_PROTOCOL
                            ]["selection"]["status"],
                            "pointwise_selected_time_mean": None,
                            "pointwise_selected_time_median": None,
                            "metrics": metrics,
                        }
                    )
                kneedle = _protocol_predictions(directory, split, "kneedle")
                if set(kneedle) != {current_primary}:
                    raise GlobalCampaignError(
                        "FLIPD protocol must expose only primary readout"
                    )
                fallback = np.asarray(
                    _load_numeric_array(
                        directory / f"{split}_knee_fallback__{current_primary}.npy",
                        ndim=1,
                    ),
                    dtype=np.uint8,
                ).astype(bool)
                selected_time = np.asarray(
                    _load_numeric_array(
                        directory / f"{split}_knee_time__{current_primary}.npy",
                        ndim=1,
                    ),
                    dtype=np.float64,
                )
                finite_time = selected_time[~fallback]
                metrics_b = {
                    **known_lid_metrics(kneedle[current_primary], target),
                    "no_knee_n": int(fallback.sum()),
                    "no_knee_fraction": float(fallback.mean()),
                }
                declared_b = summary["metrics"][KNOWN_KNEEDLE_PROTOCOL][split]
                if not _same_json(declared_b.get(current_primary), metrics_b):
                    raise GlobalCampaignError(
                        f"aggregate Kneedle metrics differ: {cell.key}/{split}"
                    )
                known_lid.append(
                    {
                        "cell_key": cell.key,
                        "cell_id": summary["cell_id"],
                        "inventory_origin": _inventory_origin(cell),
                        "suite_id": cell.suite_id,
                        "dataset": cell.dataset,
                        "representation": cell.representation,
                        "split": split,
                        "readout": current_primary,
                        "is_primary_readout": True,
                        "selection_protocol": KNOWN_KNEEDLE_PROTOCOL,
                        "selected_index": None,
                        "selected_lambda": None,
                        "selection_status": "pointwise_kneedle_with_ambient_fallback",
                        "pointwise_selected_time_mean": (
                            float(finite_time.mean()) if finite_time.size else None
                        ),
                        "pointwise_selected_time_median": (
                            float(np.median(finite_time)) if finite_time.size else None
                        ),
                        "metrics": metrics_b,
                    }
                )
            else:
                selection = summary["protocols"][UNKNOWN_REFERENCE_PROTOCOL][
                    "selection"
                ]
                reference_predictions[(cell.key, split)] = (
                    {}
                    if selection["status"] == "selection_failed"
                    else _protocol_predictions(directory, split, "reference")
                )

    sample_size: list[dict[str, Any]] = []
    paired_delta: list[dict[str, Any]] = []
    for cell in cells:
        if cell.target_policy not in {"sample_size", "paired_delta"}:
            continue
        if cell.expected_lid_delta is None:
            raise GlobalCampaignError(f"aggregate cell {cell.key} lacks LID delta")
        reference = _reference_cell(cells, cell)
        current_summary = summaries[cell.key]
        reference_summary = summaries[reference.key]
        if current_summary["selected_scale"] != reference_summary["selected_scale"]:
            raise GlobalCampaignError(
                f"aggregate cell {cell.key} did not reuse frozen reference lambda"
            )
        if cell.dataset != reference.dataset:
            binding = current_summary.get("reference_binding")
            if (
                not isinstance(binding, Mapping)
                or binding.get("cell_key") != reference.key
                or binding.get("summary_sha256")
                != sha256_path(Path(cell_directories[reference.key]) / "summary.json")
            ):
                raise GlobalCampaignError(
                    f"aggregate cell {cell.key} lacks exact reference binding"
                )
        for split in ("validation", "test"):
            current = reference_predictions[(cell.key, split)]
            baseline = reference_predictions[(reference.key, split)]
            selection = current_summary["protocols"][UNKNOWN_REFERENCE_PROTOCOL][
                "selection"
            ]
            reference_selection = reference_summary["protocols"][
                UNKNOWN_REFERENCE_PROTOCOL
            ]["selection"]
            failed = (
                selection["status"] == "selection_failed"
                or reference_selection["status"] == "selection_failed"
            )
            if failed:
                if current or baseline:
                    raise GlobalCampaignError(
                        f"selection-failed aggregate stores predictions: {cell.key}/{split}"
                    )
                for readout in _model_readouts(
                    _load_json(Path(cell_directories[cell.key]) / "manifest.json")[
                        "identity"
                    ]["model"]["model"]
                ):
                    common_failure = {
                        "cell_key": cell.key,
                        "cell_id": current_summary["cell_id"],
                        "inventory_origin": _inventory_origin(cell),
                        "suite_id": cell.suite_id,
                        "comparison_group": cell.comparison_group,
                        "dataset": cell.dataset,
                        "reference_dataset": reference.dataset,
                        "representation": cell.representation,
                        "split": split,
                        "readout": readout,
                        "is_primary_readout": readout == primary_readout,
                        "selection_protocol": UNKNOWN_REFERENCE_PROTOCOL,
                        "frozen_selected_index": None,
                        "frozen_selected_lambda": None,
                        "selection_status": "selection_failed",
                        "failure_reason": selection.get("failure_reason"),
                        "expected_lid_delta": cell.expected_lid_delta,
                        "n_source_train": (
                            current_summary["partition"]["n_source_train"]
                            if cell.target_policy == "sample_size"
                            else None
                        ),
                        "metrics": {
                            "status": "selection_failed",
                            "failure_reason": selection.get("failure_reason"),
                        },
                    }
                    (
                        sample_size
                        if cell.target_policy == "sample_size"
                        else paired_delta
                    ).append(common_failure)
                continue
            if set(current) != set(baseline):
                raise GlobalCampaignError(
                    f"aggregate readouts differ for {cell.key}/{split}"
                )
            for readout in sorted(current):
                common = {
                    "cell_key": cell.key,
                    "cell_id": current_summary["cell_id"],
                    "inventory_origin": _inventory_origin(cell),
                    "suite_id": cell.suite_id,
                    "comparison_group": cell.comparison_group,
                    "dataset": cell.dataset,
                    "reference_dataset": reference.dataset,
                    "representation": cell.representation,
                    "split": split,
                    "readout": readout,
                    "is_primary_readout": readout == primary_readout,
                    "selection_protocol": UNKNOWN_REFERENCE_PROTOCOL,
                    "selection_status": "selected",
                    "failure_reason": None,
                    "frozen_selected_index": current_summary["selected_index"],
                    "frozen_selected_lambda": current_summary["selected_scale"],
                    "expected_lid_delta": cell.expected_lid_delta,
                }
                if cell.target_policy == "sample_size":
                    current_distribution = prediction_summary(current[readout])
                    reference_distribution = prediction_summary(baseline[readout])
                    mean_delta = float(
                        current_distribution["mean"] - reference_distribution["mean"]
                    )
                    median_delta = float(
                        current_distribution["median"]
                        - reference_distribution["median"]
                    )
                    sample_size.append(
                        {
                            **common,
                            "n_source_train": current_summary["partition"][
                                "n_source_train"
                            ],
                            "prediction_summary": current_distribution,
                            "reference_prediction_summary": reference_distribution,
                            "mean_delta_from_reference": mean_delta,
                            "median_delta_from_reference": median_delta,
                            "mean_delta_error": float(
                                mean_delta - cell.expected_lid_delta
                            ),
                        }
                    )
                    continue
                current_labels = labels[(cell.key, split)]
                reference_labels = labels[(reference.key, split)]
                if (
                    current_labels is None
                    or reference_labels is None
                    or current[readout].shape != baseline[readout].shape
                    or current_labels.shape != reference_labels.shape
                    or current_labels.shape != current[readout].shape
                    or not np.array_equal(current_labels, reference_labels)
                ):
                    raise GlobalCampaignError(
                        f"paired-delta rows are not aligned: {cell.key}/{split}"
                    )
                paired_delta.append(
                    {
                        **common,
                        "n": int(current[readout].size),
                        "labels_sha256": v1._array_sha(current_labels),
                        "metrics": paired_delta_metrics(
                            baseline[readout],
                            current[readout],
                            expected_delta=cell.expected_lid_delta,
                        ),
                    }
                )
    assert primary_readout is not None
    return {
        "schema_version": 2,
        "model_variant": model_variant,
        "primary_readout": primary_readout,
        "physical_training_count": len(cells),
        "logical_known_lid_protocols": [
            KNOWN_SUPERVISED_PROTOCOL,
            KNOWN_KNEEDLE_PROTOCOL,
        ],
        "coverage": {
            "cells": len(cells),
            "canonical_cells": sum(cell.exact_archive for cell in cells),
            "generated_extension_cells": sum(not cell.exact_archive for cell in cells),
            "known_lid_records": len(known_lid),
            "e1_sample_size_records": len(sample_size),
            "e5_paired_delta_records": len(paired_delta),
        },
        "known_lid": known_lid,
        "e1_sample_size_stability": sample_size,
        "e5_paired_delta": paired_delta,
    }


def _campaign_aggregate(
    campaign_identity: str, model_aggregates: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    known: list[dict[str, Any]] = []
    sample: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    models: list[dict[str, Any]] = []
    for aggregate in model_aggregates:
        variant = str(aggregate["model_variant"])
        models.append(
            {
                "model_variant": variant,
                "primary_readout": aggregate["primary_readout"],
                "physical_training_count": aggregate["physical_training_count"],
                "coverage": _plain(aggregate["coverage"]),
            }
        )
        for source, destination in (
            (aggregate["known_lid"], known),
            (aggregate["e1_sample_size_stability"], sample),
            (aggregate["e5_paired_delta"], paired),
        ):
            destination.extend(
                {"model_variant": variant, **_plain(record)} for record in source
            )
    return {
        "schema_version": 2,
        "campaign_identity": campaign_identity,
        "physical_training_count": sum(
            row["physical_training_count"] for row in models
        ),
        "logical_known_lid_protocols": [
            KNOWN_SUPERVISED_PROTOCOL,
            KNOWN_KNEEDLE_PROTOCOL,
        ],
        "models": models,
        "coverage": {
            "models": len(models),
            "known_lid_records": len(known),
            "e1_sample_size_records": len(sample),
            "e5_paired_delta_records": len(paired),
        },
        "known_lid": known,
        "e1_sample_size_stability": sample,
        "e5_paired_delta": paired,
    }


_UNIFIED_TABLE_FIELDS = (
    "analysis",
    "model_variant",
    "cell_key",
    "cell_id",
    "inventory_origin",
    "suite_id",
    "dataset",
    "reference_dataset",
    "representation",
    "split",
    "readout",
    "primary_readout",
    "is_primary_readout",
    "selection_protocol",
    "selected_coordinate_name",
    "selected_index",
    "selected_coordinate",
    "selection_status",
    "failure_reason",
    "pointwise_selected_time_mean",
    "pointwise_selected_time_median",
    "no_knee_n",
    "no_knee_fraction",
    "expected_lid_delta",
    "n_source_train",
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
    "labels_sha256",
)


def render_unified_results_csv(aggregate: Mapping[str, Any]) -> str:
    rows: list[dict[str, Any]] = []
    primary_by_variant = {
        str(row["model_variant"]): str(row["primary_readout"])
        for row in aggregate["models"]
    }
    for analysis, source in (
        ("known_lid", aggregate["known_lid"]),
        ("e1_sample_size_stability", aggregate["e1_sample_size_stability"]),
        ("e5_paired_delta", aggregate["e5_paired_delta"]),
    ):
        for record in source:
            metrics = record.get("metrics")
            if not isinstance(metrics, Mapping):
                metrics = record.get("prediction_summary", {})
            reference_metrics = record.get("reference_prediction_summary", {})
            selected_index = record.get(
                "selected_index", record.get("frozen_selected_index")
            )
            selected_coordinate = record.get(
                "selected_lambda", record.get("frozen_selected_lambda")
            )
            row = {
                "analysis": analysis,
                "model_variant": record["model_variant"],
                "cell_key": record["cell_key"],
                "cell_id": record["cell_id"],
                "inventory_origin": record["inventory_origin"],
                "suite_id": record["suite_id"],
                "dataset": record["dataset"],
                "reference_dataset": record.get("reference_dataset"),
                "representation": record["representation"],
                "split": record["split"],
                "readout": record["readout"],
                "primary_readout": primary_by_variant[str(record["model_variant"])],
                "is_primary_readout": record["is_primary_readout"],
                "selection_protocol": record["selection_protocol"],
                "selected_coordinate_name": (
                    "vp_time_pointwise"
                    if record["selection_protocol"] == KNOWN_KNEEDLE_PROTOCOL
                    else "lambda"
                ),
                "selected_index": selected_index,
                "selected_coordinate": selected_coordinate,
                "selection_status": record.get("selection_status"),
                "failure_reason": record.get(
                    "failure_reason", metrics.get("failure_reason")
                ),
                "pointwise_selected_time_mean": record.get(
                    "pointwise_selected_time_mean"
                ),
                "pointwise_selected_time_median": record.get(
                    "pointwise_selected_time_median"
                ),
                "no_knee_n": metrics.get("no_knee_n"),
                "no_knee_fraction": metrics.get("no_knee_fraction"),
                "expected_lid_delta": record.get("expected_lid_delta"),
                "n_source_train": record.get("n_source_train"),
                "reference_mean": reference_metrics.get("mean"),
                "reference_median": reference_metrics.get("median"),
                "mean_delta_from_reference": record.get("mean_delta_from_reference"),
                "median_delta_from_reference": record.get(
                    "median_delta_from_reference"
                ),
                "mean_delta_error": record.get("mean_delta_error"),
                "labels_sha256": record.get("labels_sha256"),
            }
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
                row[field] = metrics.get(field)
            rows.append(row)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=list(_UNIFIED_TABLE_FIELDS), lineterminator="\n"
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {key: "" if value is None else value for key, value in row.items()}
        )
    return stream.getvalue()


def _aggregate_macros(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    def mean(values: Sequence[float]) -> float | None:
        return float(np.mean(values)) if values else None

    primary = str(aggregate["primary_readout"])
    known_by_protocol: dict[str, float | None] = {}
    for protocol in (KNOWN_SUPERVISED_PROTOCOL, KNOWN_KNEEDLE_PROTOCOL):
        known_by_protocol[protocol] = mean(
            [
                float(record["metrics"]["mae"])
                for record in aggregate["known_lid"]
                if record["split"] == "test"
                and record["is_primary_readout"] is True
                and record["selection_protocol"] == protocol
            ]
        )
    return {
        "coverage": _plain(aggregate["coverage"]),
        "physical_training_count": aggregate["physical_training_count"],
        "primary_readout": primary,
        "known_lid_test_mean_mae_by_protocol": known_by_protocol,
        "e1_test_mean_absolute_mean_delta_error": mean(
            [
                abs(float(record["mean_delta_error"]))
                for record in aggregate["e1_sample_size_stability"]
                if record["split"] == "test"
                and record["is_primary_readout"] is True
                and record.get("selection_status") == "selected"
            ]
        ),
        "e5_test_mean_paired_delta_mae": mean(
            [
                float(record["metrics"]["mae"])
                for record in aggregate["e5_paired_delta"]
                if record["split"] == "test"
                and record["is_primary_readout"] is True
                and record.get("selection_status") == "selected"
            ]
        ),
    }


def _expected_cell_directory(
    *, campaign_root: Path, prepared: Any, model_index: int, cell_index: int
) -> tuple[Path, dict[str, Any]]:
    plan = prepared.plans[model_index]
    cell = prepared.cells[cell_index]

    class IdentityInput:
        input_sha256 = str(prepared.preflight_inputs[cell.key]["input_sha256"])
        input_record = prepared.preflight_inputs[cell.key]["input_record"]

    identity = _cell_identity(
        campaign_id=prepared.campaign_id,
        campaign_config_sha=prepared.config_sha,
        source_sha=prepared.source_sha,
        model_plan=plan,
        cell=cell,
        cell_data=IdentityInput(),
        selection_contract=prepared.config["campaign"]["selection"],
        evaluation_contract=prepared.config["campaign"]["evaluation"],
    )
    cell_id = sha256_bytes(canonical_json(identity).encode("utf-8"))[:20]
    directory = (
        campaign_root
        / "runs"
        / _safe_component(plan.variant_id)
        / _safe_component(cell.suite_id)
        / (
            f"{_safe_component(cell.dataset)}__"
            f"{_safe_component(cell.representation)}__{cell_id}"
        )
    )
    return directory, identity


def _load_bound_canary_report(path: Path, prepared: Any) -> dict[str, Any]:
    project_root = Path(getattr(prepared, "project_root", repository_root())).resolve()
    canary_config = project_root / "configs/v2_canary.yaml"
    if not canary_config.is_file():
        raise GlobalCampaignError("v2 canary config is missing from declared source")
    return validate_canary_report(
        path,
        campaign_identity=prepared.campaign_identity,
        declared_source_sha256=prepared.source_sha,
        campaign_config_sha256=prepared.config_sha,
        input_inventory_sha256=prepared.input_inventory_sha,
        canary_config_sha256=sha256_path(canary_config),
    )


def _validate_reusable_canary_cells(campaign_root: Path, prepared: Any) -> list[str]:
    errors: list[str] = []
    plan_indices = {plan.variant_id: index for index, plan in enumerate(prepared.plans)}
    cell_indices = {cell.key: index for index, cell in enumerate(prepared.cells)}
    quality_tasks = set(CANARY_TASKS)
    for variant, key in CANARY_REUSE_TASKS:
        try:
            model_index = plan_indices[variant]
            cell_index = cell_indices[key]
        except KeyError:
            errors.append(
                f"canary task is absent from production matrix: {variant}/{key}"
            )
            continue
        directory, identity = _expected_cell_directory(
            campaign_root=campaign_root,
            prepared=prepared,
            model_index=model_index,
            cell_index=cell_index,
        )
        cell_errors = validate_global_cell(
            directory,
            expected_identity=identity,
            expected_source_evidence=prepared.preflight_inputs[key].get(
                "source_evidence"
            ),
        )
        errors.extend(f"canary {variant}/{key}: {error}" for error in cell_errors)
        if not cell_errors:
            summary = _load_json(directory / "summary.json")
            diagnostics = summary.get("quality_diagnostics")
            expected_status = (
                "passed" if (variant, key) in quality_tasks else "not_requested"
            )
            if (
                not isinstance(diagnostics, Mapping)
                or diagnostics.get("status") != expected_status
            ):
                errors.append(
                    f"canary {variant}/{key}: quality diagnostics status differs"
                )
    return errors


def validate_pre_run_gate(*, campaign_root: Path, prepared: Any) -> None:
    """Refuse the full DAG unless all 24 exact reusable canary cells validate."""

    report_path = Path(campaign_root) / str(
        prepared.config["campaign"]["canary_gate"]["report_filename"]
    )
    _load_bound_canary_report(report_path, prepared)
    errors = _validate_reusable_canary_cells(Path(campaign_root), prepared)
    if errors:
        raise GlobalCampaignError(f"v2 reusable canary cells are invalid: {errors}")


def final_manifest_extras(*, campaign_root: Path, prepared: Any) -> dict[str, Any]:
    report_path = Path(campaign_root) / str(
        prepared.config["campaign"]["canary_gate"]["report_filename"]
    )
    _load_bound_canary_report(report_path, prepared)
    return {
        "physical_training_count": EXPECTED_PHYSICAL_TRAININGS,
        "canonical_cells_per_model": 35,
        "generated_extension_cells_per_model": 4,
        "logical_known_lid_protocols": [
            KNOWN_SUPERVISED_PROTOCOL,
            KNOWN_KNEEDLE_PROTOCOL,
        ],
        "canary_protocol_id": CANARY_PROTOCOL_ID,
        "canary_report_path": report_path.relative_to(campaign_root).as_posix(),
        "canary_report_sha256": sha256_path(report_path),
    }


def validate_global_campaign(
    campaign_root: Path,
    *,
    expected_campaign_identity: str | None = None,
    project_root: Path | None = None,
    verify_inputs: bool = True,
    source_preflight_fn: Callable[..., Any] | None = None,
    cell_loader: Callable[..., Any] | None = None,
) -> list[str]:
    """Validate the complete 429-training v2 campaign from source to table."""

    root = Path(campaign_root).resolve()
    checkout = (
        repository_root() if project_root is None else Path(project_root).resolve()
    )
    errors: list[str] = []
    try:
        manifest = _load_json(root / "campaign.json")
    except GlobalCampaignError as exc:
        return [str(exc)]
    required = {
        "schema_version",
        "campaign_identity",
        "campaign_id",
        "config_sha256",
        "source_tree_sha256",
        "input_inventory_sha256",
        "inventory_cells",
        "approved_model_variants",
        "model_contracts",
        "input_inventory_path",
        "input_inventory_file_sha256",
        "aggregate_path",
        "aggregate_sha256",
        "unified_results_path",
        "unified_results_sha256",
        "created_at_utc",
        "models",
        "cells",
        "expected_models",
        "expected_cells_per_model",
        "complete",
        "physical_training_count",
        "canonical_cells_per_model",
        "generated_extension_cells_per_model",
        "logical_known_lid_protocols",
        "canary_protocol_id",
        "canary_report_path",
        "canary_report_sha256",
    }
    if set(manifest) != required:
        errors.append("v2 final campaign manifest fields differ")
    if manifest.get("schema_version") != GLOBAL_FINAL_MANIFEST_SCHEMA_VERSION:
        errors.append("v2 final campaign schema differs")
    if manifest.get("complete") is not True:
        errors.append("v2 campaign is not complete")
    if (
        manifest.get("expected_models") != len(APPROVED_MODEL_VARIANTS)
        or manifest.get("expected_cells_per_model") != EXPECTED_GLOBAL_CELL_COUNT
        or manifest.get("physical_training_count") != EXPECTED_PHYSICAL_TRAININGS
    ):
        errors.append("v2 campaign is not the exact 11 x 39 physical matrix")
    if (
        manifest.get("canonical_cells_per_model") != 35
        or manifest.get("generated_extension_cells_per_model") != 4
    ):
        errors.append("v2 canonical/generated inventory counts differ")
    if tuple(manifest.get("approved_model_variants") or ()) != APPROVED_MODEL_VARIANTS:
        errors.append("v2 approved model order differs")
    if tuple(manifest.get("logical_known_lid_protocols") or ()) != (
        KNOWN_SUPERVISED_PROTOCOL,
        KNOWN_KNEEDLE_PROTOCOL,
    ):
        errors.append("v2 logical known-LID protocols differ")
    campaign_identity = manifest.get("campaign_identity")
    if (
        expected_campaign_identity is not None
        and campaign_identity != expected_campaign_identity
    ):
        errors.append("v2 campaign identity differs from expected")

    try:
        resolved_raw = yaml.safe_load(
            (root / "resolved_config.yaml").read_text(encoding="utf-8")
        )
        config = validate_global_campaign_config(resolved_raw)
    except (OSError, UnicodeError, yaml.YAMLError, GlobalCampaignError) as exc:
        errors.append(f"cannot validate v2 resolved config: {exc}")
        return errors
    config_sha = sha256_bytes(canonical_json(config).encode("utf-8"))
    if manifest.get("config_sha256") != config_sha:
        errors.append("v2 resolved config SHA differs")
    if manifest.get("campaign_id") != config["campaign"]["campaign_id"]:
        errors.append("v2 campaign id differs from config")
    try:
        source_sha = hash_declared_sources(checkout)
    except Exception as exc:  # noqa: BLE001 - validator returns source audit errors
        errors.append(f"cannot hash v2 declared sources: {type(exc).__name__}")
        source_sha = None
    if source_sha is not None and manifest.get("source_tree_sha256") != source_sha:
        errors.append("v2 declared source tree differs from checkout")

    inventory_rows = manifest.get("inventory_cells")
    cells: list[CampaignCell] = []
    input_inventory: list[dict[str, Any]] = []
    if (
        not isinstance(inventory_rows, list)
        or len(inventory_rows) != EXPECTED_GLOBAL_CELL_COUNT
    ):
        errors.append("v2 final inventory must contain exactly 39 cells")
    else:
        for ordinal, row in enumerate(inventory_rows):
            if not isinstance(row, Mapping):
                errors.append(f"v2 inventory row {ordinal} is invalid")
                continue
            try:
                cell = CampaignCell(
                    **{
                        key: value
                        for key, value in row.items()
                        if key not in {"input_sha256", "input_record"}
                    }
                )
            except (TypeError, ValueError) as exc:
                errors.append(f"v2 inventory row {ordinal} is invalid: {exc}")
                continue
            record = row.get("input_record")
            digest = row.get("input_sha256")
            if not isinstance(record, Mapping) or digest != sha256_bytes(
                canonical_json(_plain(record)).encode("utf-8")
            ):
                errors.append(f"v2 inventory input identity {ordinal} is invalid")
            cells.append(cell)
            input_inventory.append(
                {
                    "cell": _plain(cell),
                    "input_sha256": digest,
                    "input_record": _plain(record),
                }
            )
    if tuple(cell.key for cell in cells) != APPROVED_GLOBAL_CELL_KEYS:
        errors.append("v2 inventory differs from approved 35+4 order")
    if (
        sum(cell.exact_archive for cell in cells) != 35
        or sum(not cell.exact_archive for cell in cells) != 4
    ):
        errors.append("v2 inventory provenance split differs")
    input_sha = sha256_bytes(canonical_json(input_inventory).encode("utf-8"))
    if manifest.get("input_inventory_sha256") != input_sha:
        errors.append("v2 input inventory identity differs")
    expected_campaign_identity = sha256_bytes(
        canonical_json(
            {
                "schema_version": GLOBAL_CAMPAIGN_SCHEMA_VERSION,
                "campaign_id": manifest.get("campaign_id"),
                "config_sha256": config_sha,
                "source_tree_sha256": manifest.get("source_tree_sha256"),
                "input_inventory_sha256": input_sha,
            }
        ).encode("utf-8")
    )
    if campaign_identity != expected_campaign_identity:
        errors.append("v2 campaign identity does not recompute")
    inventory_path = root / str(manifest.get("input_inventory_path", ""))
    try:
        persisted_inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        persisted_inventory = None
    if not _same_json(persisted_inventory, input_inventory):
        errors.append("v2 input_inventory.json differs")
    if inventory_path.is_file() and (
        manifest.get("input_inventory_file_sha256") != sha256_path(inventory_path)
    ):
        errors.append("v2 input inventory file hash differs")

    fresh_evidence: dict[str, Mapping[str, Any]] = {}
    if verify_inputs and len(cells) == EXPECTED_GLOBAL_CELL_COUNT:
        selected_preflight = (
            validate_campaign_sources
            if source_preflight_fn is None
            else source_preflight_fn
        )
        selected_loader = (
            load_campaign_cell_data if cell_loader is None else cell_loader
        )
        try:
            approved = load_campaign_inventory(config, checkout)
            if tuple(approved) != tuple(cells):
                raise GlobalCampaignError("fresh YAML inventory differs")
            source_records = dict(selected_preflight(config, checkout, approved))
            for ordinal, cell in enumerate(approved):
                data = _bind_source_preflight(
                    selected_loader(cell, config, checkout), cell, source_records
                )
                if data.input_sha256 != inventory_rows[ordinal].get(
                    "input_sha256"
                ) or not _same_json(
                    data.input_record, inventory_rows[ordinal].get("input_record")
                ):
                    errors.append(f"fresh v2 input differs for {cell.key}")
                fresh_evidence[cell.key] = _source_evidence(data, config)
        except Exception as exc:  # noqa: BLE001 - validator audits all inputs
            errors.append(f"cannot revalidate v2 inputs: {type(exc).__name__}: {exc}")

    plans = model_plans(config)
    model_contracts = manifest.get("model_contracts")
    expected_contracts = [
        {
            "variant_id": plan.variant_id,
            "experiment_name": plan.experiment_name,
            "model": _plain(plan.model),
        }
        for plan in plans
    ]
    if not _same_json(model_contracts, expected_contracts):
        errors.append("v2 model contracts differ from resolved fragments")
    cell_records = manifest.get("cells")
    if (
        not isinstance(cell_records, list)
        or len(cell_records) != EXPECTED_PHYSICAL_TRAININGS
    ):
        errors.append("v2 final cell ledger does not contain 429 rows")
        cell_records = []
    record_by_pair: dict[tuple[str, str], Mapping[str, Any]] = {}
    expected_cell_record_fields = {
        "model_variant",
        "cell_key",
        "ordinal",
        "path",
        "manifest_sha256",
        "summary_sha256",
    }
    for record in cell_records:
        if (
            not isinstance(record, Mapping)
            or set(record) != expected_cell_record_fields
        ):
            errors.append("v2 final cell ledger row is invalid")
            continue
        pair = (str(record.get("model_variant")), str(record.get("cell_key")))
        if pair in record_by_pair:
            errors.append(f"v2 final cell ledger repeats {pair}")
        try:
            expected_ordinal = APPROVED_GLOBAL_CELL_KEYS.index(pair[1])
        except ValueError:
            expected_ordinal = None
        if (
            pair[0] not in APPROVED_MODEL_VARIANTS
            or expected_ordinal is None
            or record.get("ordinal") != expected_ordinal
            or not isinstance(record.get("manifest_sha256"), str)
            or not _SHA256.fullmatch(record["manifest_sha256"])
            or not isinstance(record.get("summary_sha256"), str)
            or not _SHA256.fullmatch(record["summary_sha256"])
        ):
            errors.append(f"v2 final cell ledger identity differs for {pair}")
        record_by_pair[pair] = record
    try:
        durable_ledger = _load_json(root / "state/ledger.json")
    except GlobalCampaignError as exc:
        errors.append(str(exc))
    else:
        if (
            not isinstance(durable_ledger, Mapping)
            or set(durable_ledger)
            != {"schema_version", "campaign_identity", "completed_cells"}
            or durable_ledger.get("schema_version") != 1
            or durable_ledger.get("campaign_identity") != campaign_identity
            or not _same_json(durable_ledger.get("completed_cells"), cell_records)
        ):
            errors.append("v2 durable campaign ledger differs")
    try:
        assignments = _load_json(root / "state/parallel/assignments.json")
    except GlobalCampaignError as exc:
        errors.append(str(exc))
    else:
        assignment_rows = assignments.get("assignments")
        expected_assignment_fields = {
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
            "physical_device_index",
            "device_lane",
            "visible_device_token",
        }
        assignment_pairs: set[tuple[str, str]] = set()
        assignment_rows_valid = isinstance(assignment_rows, list)
        completed_sequences: list[int] = []
        if isinstance(assignment_rows, list):
            for row in assignment_rows:
                if (
                    not isinstance(row, Mapping)
                    or set(row) != expected_assignment_fields
                ):
                    assignment_rows_valid = False
                    continue
                model_index = row.get("model_index")
                cell_index = row.get("cell_index")
                if (
                    type(model_index) is not int
                    or type(cell_index) is not int
                    or not 0 <= model_index < len(plans)
                    or not 0 <= cell_index < len(cells)
                ):
                    assignment_rows_valid = False
                    continue
                pair = (str(row.get("model_variant")), str(row.get("cell_key")))
                assignment_pairs.add(pair)
                if pair != (plans[model_index].variant_id, cells[cell_index].key):
                    assignment_rows_valid = False
                status = row.get("status")
                if status == "completed":
                    sequence = row.get("dispatch_sequence")
                    worker_slot = row.get("worker_slot")
                    physical_device_index = row.get("physical_device_index")
                    device_lane = row.get("device_lane")
                    visible_device_token = row.get("visible_device_token")
                    if (
                        type(worker_slot) is not int
                        or not 0 <= worker_slot < H100_LOGICAL_WORKER_COUNT
                        or type(physical_device_index) is not int
                        or physical_device_index
                        != worker_slot % H100_PHYSICAL_DEVICE_COUNT
                        or type(device_lane) is not int
                        or device_lane != worker_slot // H100_PHYSICAL_DEVICE_COUNT
                        or not isinstance(visible_device_token, str)
                        or not visible_device_token
                        or not isinstance(row.get("visible_device"), str)
                        or not row["visible_device"].startswith(
                            f"{visible_device_token}:"
                        )
                        or type(sequence) is not int
                        or sequence <= 0
                        or type(row.get("in_flight_after_dispatch")) is not int
                        or not 1
                        <= row["in_flight_after_dispatch"]
                        <= H100_LOGICAL_WORKER_COUNT
                        or type(row.get("ready_after_dispatch")) is not int
                        or row["ready_after_dispatch"] < 0
                    ):
                        assignment_rows_valid = False
                    else:
                        completed_sequences.append(sequence)
                elif status == "reconstructed":
                    if (
                        row.get("worker_slot") is not None
                        or row.get("visible_device") != "reconstructed-sealed-cell"
                        or row.get("physical_device_index") is not None
                        or row.get("device_lane") is not None
                        or row.get("visible_device_token") is not None
                        or row.get("dispatch_sequence") is not None
                        or row.get("in_flight_after_dispatch") is not None
                        or row.get("ready_after_dispatch") is not None
                    ):
                        assignment_rows_valid = False
                else:
                    assignment_rows_valid = False
        if len(completed_sequences) != len(set(completed_sequences)):
            assignment_rows_valid = False
        if (
            not isinstance(assignments, Mapping)
            or set(assignments)
            != {
                "schema_version",
                "campaign_identity",
                "strategy",
                "worker_count",
                "physical_device_count",
                "lanes_per_device",
                "device_mapping_policy",
                "assignments",
            }
            or assignments.get("schema_version") != 2
            or assignments.get("campaign_identity") != campaign_identity
            or assignments.get("strategy") != EXECUTION_STRATEGY_CELL_DAG
            or assignments.get("worker_count") != H100_LOGICAL_WORKER_COUNT
            or assignments.get("physical_device_count") != H100_PHYSICAL_DEVICE_COUNT
            or assignments.get("lanes_per_device") != H100_LANES_PER_DEVICE
            or assignments.get("device_mapping_policy")
            != "worker_slot_modulo_physical_device_count_v1"
            or not isinstance(assignment_rows, list)
            or len(assignment_rows) != EXPECTED_PHYSICAL_TRAININGS
            or assignment_pairs != set(record_by_pair)
            or not assignment_rows_valid
        ):
            errors.append("v2 parallel assignment ledger differs")

    for plan in plans:
        ledger_path = (
            root / "state/models" / _safe_component(plan.variant_id) / "ledger.json"
        )
        try:
            model_ledger = _load_json(ledger_path)
        except GlobalCampaignError as exc:
            errors.append(str(exc))
            continue
        expected_model_rows = [
            record
            for record in cell_records
            if isinstance(record, Mapping)
            and record.get("model_variant") == plan.variant_id
        ]
        if (
            not isinstance(model_ledger, Mapping)
            or set(model_ledger)
            != {
                "schema_version",
                "campaign_identity",
                "model_variant",
                "completed_cells",
                "complete",
            }
            or model_ledger.get("schema_version") != 1
            or model_ledger.get("campaign_identity") != campaign_identity
            or model_ledger.get("model_variant") != plan.variant_id
            or model_ledger.get("complete") is not True
            or not _same_json(model_ledger.get("completed_cells"), expected_model_rows)
        ):
            errors.append(f"v2 durable model ledger differs for {plan.variant_id}")

    directories_by_model: dict[str, dict[str, Path]] = {}
    for model_index, plan in enumerate(plans):
        summaries: dict[str, dict[str, Any]] = {}
        directories: dict[str, Path] = {}
        for cell_index, cell in enumerate(cells):
            prepared_validation = SimpleNamespace(
                plans=plans,
                cells=tuple(cells),
                preflight_inputs={
                    item.key: {
                        "input_sha256": inventory_rows[index]["input_sha256"],
                        "input_record": inventory_rows[index]["input_record"],
                    }
                    for index, item in enumerate(cells)
                },
                campaign_id=manifest.get("campaign_id"),
                config_sha=config_sha,
                source_sha=manifest.get("source_tree_sha256"),
                config=config,
                project_root=str(checkout),
            )
            directory, identity = _expected_cell_directory(
                campaign_root=root,
                prepared=prepared_validation,
                model_index=model_index,
                cell_index=cell_index,
            )
            reference_summary = None
            if cell.reference_dataset not in {None, cell.dataset}:
                reference_key = _reference_cell(cells, cell).key
                reference_summary = summaries.get(reference_key)
                if reference_summary is None:
                    errors.append(f"v2 reference order is unresolved for {cell.key}")
            cell_errors = validate_global_cell(
                directory,
                expected_identity=identity,
                reference_summary=reference_summary,
                expected_source_evidence=fresh_evidence.get(cell.key),
            )
            errors.extend(
                f"{plan.variant_id}/{cell.key}: {error}" for error in cell_errors
            )
            if directory.is_dir() and not cell_errors:
                summary = _load_json(directory / "summary.json")
                summary["summary_sha256"] = sha256_path(directory / "summary.json")
                summaries[cell.key] = summary
            record = record_by_pair.get((plan.variant_id, cell.key))
            if record is None:
                errors.append(f"v2 final ledger lacks {plan.variant_id}/{cell.key}")
            elif (
                record.get("path") != directory.relative_to(root).as_posix()
                or not directory.is_dir()
                or record.get("manifest_sha256")
                != sha256_path(directory / "manifest.json")
                or record.get("summary_sha256")
                != sha256_path(directory / "summary.json")
            ):
                errors.append(
                    f"v2 final ledger hashes differ for {plan.variant_id}/{cell.key}"
                )
            directories[cell.key] = directory
        directories_by_model[plan.variant_id] = directories

    model_records = manifest.get("models")
    if not isinstance(model_records, list) or len(model_records) != len(plans):
        errors.append("v2 model manifest list differs")
        model_records = []
    recomputed_models: list[dict[str, Any]] = []
    for plan in plans:
        if set(directories_by_model.get(plan.variant_id, {})) != set(
            APPROVED_GLOBAL_CELL_KEYS
        ):
            continue
        try:
            recomputed = recompute_model_aggregate(
                plan.variant_id, cells, directories_by_model[plan.variant_id]
            )
        except Exception as exc:  # noqa: BLE001 - validator reports aggregate errors
            errors.append(f"cannot recompute v2 aggregate {plan.variant_id}: {exc}")
            continue
        recomputed_models.append(recomputed)
        model_path = (
            root / "models" / _safe_component(plan.variant_id) / "aggregate.json"
        )
        try:
            persisted = _load_json(model_path)
        except GlobalCampaignError as exc:
            errors.append(str(exc))
        else:
            if not _same_json(persisted, recomputed):
                errors.append(f"v2 model aggregate differs for {plan.variant_id}")
        matches = [
            row
            for row in model_records
            if isinstance(row, Mapping) and row.get("model_variant") == plan.variant_id
        ]
        if len(matches) != 1:
            errors.append(f"v2 model record differs for {plan.variant_id}")
        else:
            record = matches[0]
            expected_record_fields = {
                "schema_version",
                "campaign_identity",
                "model_variant",
                "experiment_name",
                "experiment_key",
                "comet_telemetry",
                "model_contract",
                "expected_cells",
                "complete_cells",
                "aggregate_path",
                "aggregate_sha256",
                "complete",
                "manifest_path",
                "manifest_sha256",
            }
            expected_manifest_relative = (
                Path("models") / _safe_component(plan.variant_id) / "manifest.json"
            ).as_posix()
            model_manifest_path = root / str(record.get("manifest_path", ""))
            try:
                persisted_model_manifest = _load_json(model_manifest_path)
            except GlobalCampaignError:
                persisted_model_manifest = None
            expected_persisted = {
                key: value
                for key, value in record.items()
                if key not in {"manifest_path", "manifest_sha256"}
            }
            if (
                set(record) != expected_record_fields
                or record.get("campaign_identity") != campaign_identity
                or record.get("experiment_name") != plan.experiment_name
                or not _same_json(
                    record.get("model_contract"),
                    {
                        "variant_id": plan.variant_id,
                        "experiment_name": plan.experiment_name,
                        "model": _plain(plan.model),
                    },
                )
                or record.get("expected_cells") != EXPECTED_GLOBAL_CELL_COUNT
                or record.get("complete_cells") != EXPECTED_GLOBAL_CELL_COUNT
                or record.get("complete") is not True
                or record.get("aggregate_path")
                != (
                    Path("models") / _safe_component(plan.variant_id) / "aggregate.json"
                ).as_posix()
                or record.get("manifest_path") != expected_manifest_relative
                or not model_manifest_path.is_file()
                or not _same_json(persisted_model_manifest, expected_persisted)
                or record.get("manifest_sha256") != sha256_path(model_manifest_path)
                or not model_path.is_file()
                or record.get("aggregate_sha256") != sha256_path(model_path)
            ):
                errors.append(f"v2 model manifest hashes differ for {plan.variant_id}")

    if len(recomputed_models) == len(plans):
        campaign_aggregate = _campaign_aggregate(
            str(campaign_identity), recomputed_models
        )
        aggregate_path = root / str(manifest.get("aggregate_path", ""))
        try:
            persisted_aggregate = _load_json(aggregate_path)
        except GlobalCampaignError as exc:
            errors.append(str(exc))
        else:
            if not _same_json(persisted_aggregate, campaign_aggregate):
                errors.append("v2 global aggregate differs from sealed cells")
        if aggregate_path.is_file() and manifest.get("aggregate_sha256") != sha256_path(
            aggregate_path
        ):
            errors.append("v2 global aggregate hash differs")
        table_path = root / str(manifest.get("unified_results_path", ""))
        expected_table = render_unified_results_csv(campaign_aggregate)
        try:
            actual_table = table_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            errors.append(f"cannot read v2 unified table: {exc}")
        else:
            if actual_table != expected_table:
                errors.append("v2 unified results do not recompute")
            if manifest.get("unified_results_sha256") != sha256_path(table_path):
                errors.append("v2 unified results hash differs")

    report_path = root / str(manifest.get("canary_report_path", ""))
    if (
        manifest.get("canary_protocol_id") != CANARY_PROTOCOL_ID
        or not report_path.is_file()
        or manifest.get("canary_report_sha256") != sha256_path(report_path)
    ):
        errors.append("v2 final manifest canary binding differs")
    else:
        prepared = SimpleNamespace(
            source_sha=manifest.get("source_tree_sha256"),
            campaign_identity=campaign_identity,
            config_sha=config_sha,
            input_inventory_sha=input_sha,
            plans=plans,
            cells=tuple(cells),
            preflight_inputs={
                item.key: {
                    "input_sha256": inventory_rows[index]["input_sha256"],
                    "input_record": inventory_rows[index]["input_record"],
                    "source_evidence": fresh_evidence.get(item.key),
                }
                for index, item in enumerate(cells)
            },
            campaign_id=manifest.get("campaign_id"),
            config=config,
            project_root=str(checkout),
        )
        try:
            _load_bound_canary_report(report_path, prepared)
        except GlobalCampaignError as exc:
            errors.append(str(exc))
        errors.extend(_validate_reusable_canary_cells(root, prepared))

    for path in root.rglob("*"):
        if path.is_symlink():
            errors.append(f"v2 campaign contains symlink: {path.relative_to(root)}")
        if path.is_file() and (
            path.name == "checkpoint.pt"
            or path.name.endswith("training_progress.pt")
            or ".incomplete" in path.parts
        ):
            errors.append(
                f"v2 campaign retains transient artifact: {path.relative_to(root)}"
            )
    return errors
