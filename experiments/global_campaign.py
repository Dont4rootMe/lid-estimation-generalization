"""Resumable one-job orchestrator for the complete learned-model matrix.

The scheduler sees one Python process.  This module composes all scientific
settings from Hydra YAML, then executes model/dataset/representation cells in a
fixed order.  Completed cells are immutable and content-addressed; an
interrupted cell keeps a stable training-progress checkpoint and can resume on
the next invocation.

Known-LID cells select one prediction scale on a held-out source-train subset
by MAE.  Unknown-LID benchmark families select a scale on the declared
reference cell by a target-free stability criterion and reuse that frozen
index.  Validation and test are evaluated at that one index only.
"""

from __future__ import annotations

import csv
import fcntl
import gc
import hashlib
import inspect
import io
import json
import math
import os
import re
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any, Protocol

import hydra
import numpy as np
import numpy.typing as npt
import yaml
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from datasets.registry import (
    LoadedSplit,
    apply_registry_overlay,
    load_dataset,
    load_registry,
)
from experiments.metrics import (
    known_lid_metrics,
    paired_delta_metrics,
    prediction_summary,
)
from experiments.pilot import (
    _affine_outer_curve_consistency_error,
    _array_sha256,
    _features_in_model_space,
    compose_pilot_config,
    validate_pilot_config,
)
from experiments.run_manifest import (
    canonical_json,
    hash_declared_sources,
    sha256_bytes,
    sha256_path,
)
from models.oracle import select_stable_scale
from utils.provenance import sha256_file

PROJECT_NAME = "lid-generalization"
WORKSPACE_NAME = "dont4rootme"
APPROVED_CAMPAIGN_ID = (
    "lid-global-e1-e8-canonical-plus-generated-e3-e4-all-trainable-models-v1"
)
GLOBAL_CAMPAIGN_SCHEMA_VERSION = 1
GLOBAL_CELL_MANIFEST_SCHEMA_VERSION = 1
GLOBAL_FINAL_MANIFEST_SCHEMA_VERSION = 1
APPROVED_MODEL_VARIANTS = (
    "diffusion",
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
_GLOBAL_EXPERIMENT_PREFIX = (
    "lid-generalization-global-e1-e8-canonical-plus-generated-e3-e4-"
)
_GLOBAL_EXPERIMENT_SUFFIXES = (
    "diffusion-train-selected-scale-seed-0",
    "rectified-flow-matching-train-selected-time-seed-0",
    "scale-conditioned-normalizing-flow-train-selected-scale-seed-0",
    "brownian-schrodinger-bridge-train-selected-time-seed-0",
    "fm-rectified-direct-velocity-train-selected-lambda-seed-0",
    "fm-rectified-posterior-mean-train-selected-lambda-seed-0",
    "fm-log-noise-direct-velocity-train-selected-lambda-seed-0",
    "fm-log-noise-posterior-mean-train-selected-lambda-seed-0",
    "fm-vp-trigonometric-direct-velocity-train-selected-lambda-seed-0",
    "fm-vp-trigonometric-posterior-mean-train-selected-lambda-seed-0",
)
APPROVED_EXPERIMENT_NAMES = tuple(
    f"{_GLOBAL_EXPERIMENT_PREFIX}{suffix}" for suffix in _GLOBAL_EXPERIMENT_SUFFIXES
)
EXPECTED_GLOBAL_CELL_COUNT = 39
APPROVED_GLOBAL_CELL_KEYS = (
    "e1/e1_sampled_fmnist_step1/dataset",
    "e1/e1_sampled_fmnist_step2/dataset",
    "e1/e1_sampled_fmnist_step3/dataset",
    "e1/e1_sampled_fmnist_step4/dataset",
    "e1/e1_sampled_fmnist_step5/dataset",
    "e1/e1_sampled_fmnist_step6/dataset",
    "e1/e1_sampled_fmnist_step7/dataset",
    "e1/e1_sampled_fmnist_step8/dataset",
    "e1/e1_sampled_fmnist_step9/dataset",
    "e1/e1_sampled_fmnist_step10/dataset",
    "e1/e1_sampled_fmnist_step11/dataset",
    "e1/e1_sampled_fmnist_step12/dataset",
    "e1/e1_sampled_fmnist_step13/dataset",
    "e1/e1_spiral_pca/dataset",
    "e1/e1_spiral_pca/coefficients",
    "e2/e2_arrows/dataset",
    "e2/e2_uniform_pca/dataset",
    "e2/e2_uniform_pca/coefficients",
    "e3/e3_gaussian_pca/dataset",
    "e3/e3_gaussian_pca/coefficients",
    "e4/e4_sphere_pca_radius1/dataset",
    "e4/e4_sphere_pca_radius1/coefficients",
    "e5/e5_downscaled_fmnist/dataset",
    "e5/e5_padded_fmnist_adddim0/dataset",
    "e5/e5_padded_fmnist_adddim4/dataset",
    "e5/e5_padded_fmnist_adddim8/dataset",
    "e5/e5_stretched_power0.25/dataset",
    "e5/e5_stretched_power4/dataset",
    "e5/e5_upscaled_fmnist/dataset",
    "e6/e6_exp_pca/dataset",
    "e6/e6_exp_pca/coefficients",
    "e7/e7_crescent_moon_radius3.0/dataset",
    "e7/e7_crescent_moon_radius3.0/coefficients",
    "e8/e8_gaussian4_pca/dataset",
    "e8/e8_gaussian4_pca/coefficients",
    "e8/e8_spaghetti_pca/dataset",
    "e8/e8_spaghetti_pca/coefficients",
    "e8/e8_sphere4_pca/dataset",
    "e8/e8_sphere4_pca/coefficients",
)
REQUIRED_SUITE_IDS = ("e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8")
KNOWN_SELECTION_PROTOCOL = "held_out_source_train_supervised_mae_v1"
UNKNOWN_SELECTION_PROTOCOL = "held_out_source_train_reference_stability_v1"
FROZEN_EVALUATION_PROTOCOL = "single_train_selected_scale_v1"
CHECKPOINT_RETENTION_RETAIN = "retain"
CHECKPOINT_RETENTION_PRUNE_AFTER_EVALUATION = "prune_after_cell_evaluation"
SUPPORTED_CHECKPOINT_RETENTION_POLICIES = (
    CHECKPOINT_RETENTION_RETAIN,
    CHECKPOINT_RETENTION_PRUNE_AFTER_EVALUATION,
)
TRAINING_ATTESTATION_SCHEMA_VERSION = 1
EXECUTION_STRATEGY_SEQUENTIAL = "sequential"
EXECUTION_STRATEGY_CELL_DAG = "cell_dag_pool"
EXECUTION_PROFILE_LEGACY = "legacy_sequential"
EXECUTION_PROFILE_H100 = "h100_8gpu_cell_dag"
_EXPERIMENT_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class GlobalCampaignError(RuntimeError):
    """Raised before an unsafe or scientifically incomplete run can continue."""


@dataclass(frozen=True)
class ModelPlan:
    variant_id: str
    experiment_name: str
    model: Mapping[str, Any]


@dataclass(frozen=True)
class CampaignCell:
    inventory_id: str
    source_kind: str
    exact_archive: bool
    dataset_config: str
    data_root: str
    registry: str
    registry_overlay: str | None
    upstream_revision: str
    archive_sha256: str | None
    provenance_label: str
    suite_id: str
    dataset: str
    representation: str
    target_policy: str
    selection_protocol: str
    comparison_group: str
    reference_dataset: str | None
    expected_lid_delta: float | None

    @property
    def key(self) -> str:
        return f"{self.suite_id}/{self.dataset}/{self.representation}"


@dataclass(frozen=True)
class CellData:
    train: npt.NDArray[Any]
    validation: npt.NDArray[Any]
    test: npt.NDArray[Any]
    train_target: npt.NDArray[np.float64] | None
    validation_target: npt.NDArray[np.float64] | None
    test_target: npt.NDArray[np.float64] | None
    validation_labels: npt.NDArray[Any] | None
    test_labels: npt.NDArray[Any] | None
    input_record: Mapping[str, Any]
    input_sha256: str


@dataclass(frozen=True)
class HoldoutPartition:
    fit_indices: npt.NDArray[np.int64]
    selection_indices: npt.NDArray[np.int64]
    fit_features: npt.NDArray[Any]
    selection_features: npt.NDArray[Any]
    selection_target: npt.NDArray[np.float64] | None
    record: Mapping[str, Any]


@dataclass(frozen=True)
class ModelLoggerHandle:
    callback: Callable[[str, Mapping[str, Any]], None] | None
    close: Callable[[], None]
    experiment_key: str | None
    connection_status: str = "online"
    log_asset: Callable[[Path, str], None] | None = None


class TrainFunction(Protocol):
    def __call__(
        self,
        family: str,
        train: npt.ArrayLike,
        validation: npt.ArrayLike,
        config: Mapping[str, Any],
        checkpoint_path: Path,
        log_callback: Callable[[Mapping[str, Any]], None] | None = None,
        *,
        progress_checkpoint_path: Path | None = None,
    ) -> Any: ...


class PredictFunction(Protocol):
    def __call__(
        self,
        trained: Any,
        query: npt.ArrayLike,
        scale: float,
        *,
        family: str,
        readout: str,
        divergence_backend: str,
        trace_probes: int,
        trace_seed: int,
        batch_size: int,
    ) -> npt.ArrayLike: ...


CellLoader = Callable[[CampaignCell, Mapping[str, Any], Path], CellData]
InventoryLoader = Callable[[Mapping[str, Any], Path], Sequence[CampaignCell]]
SourcePreflight = Callable[
    [Mapping[str, Any], Path, Sequence[CampaignCell]], Mapping[str, Mapping[str, Any]]
]
LoggerFactory = Callable[[ModelPlan, str, Path, Mapping[str, Any]], ModelLoggerHandle]
AffineDiagnosticsFunction = Callable[..., Mapping[str, Any]]


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _config_dir(root: Path | None = None) -> Path:
    if root is not None:
        return Path(root) / "configs"
    from experiments.cli import _default_config_dir

    return _default_config_dir()


def compose_global_campaign_config(
    overrides: Sequence[str] = (), *, root: Path | None = None
) -> DictConfig:
    """Compose the global campaign exclusively from Hydra YAML."""

    with initialize_config_dir(
        version_base="1.3", config_dir=str(_config_dir(root).resolve())
    ):
        config = compose(config_name="global_campaign", overrides=list(overrides))
    OmegaConf.set_struct(config, True)
    return config


def _plain(value: Any) -> Any:
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True, throw_on_missing=True)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite values are forbidden in campaign artifacts")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _plain(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(child) for child in value]
    return value


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    resolved = _plain(value)
    if not isinstance(resolved, dict):
        raise GlobalCampaignError(f"{field} must be a mapping")
    return resolved


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], *, field: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise GlobalCampaignError(f"unknown {field} fields: {sorted(unknown)}")


def _contains_secret(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            if any(part in normalized for part in ("api_key", "secret", "token")):
                return True
            if _contains_secret(child):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_secret(child) for child in value)
    return False


def _safe_path(value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise GlobalCampaignError(f"{field} must be a non-empty path string")
    path = Path(value)
    if ".." in path.parts:
        raise GlobalCampaignError(f"{field} must not contain '..'")
    return path


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GlobalCampaignError(f"{field} must be a positive integer")
    return value


def _resolved_pilot_model(variant_id: str, seed: int) -> dict[str, Any]:
    try:
        overrides = (f"pilot_model={variant_id}", f"seed={seed}")
        global_hydra = GlobalHydra.instance()
        if global_hydra.is_initialized():
            search_path = global_hydra.hydra.config_loader.get_search_path()
            approved_config_dir = _config_dir().resolve()
            actual_search_path = tuple(
                (str(entry.provider), str(entry.path))
                for entry in search_path.config_search_path
            )
            approved_library_search_path = (
                ("hydra", "pkg://hydra.conf"),
                ("main", str(approved_config_dir)),
                ("schema", "structured://"),
            )
            approved_cli_search_path = (
                ("hydra", "pkg://hydra.conf"),
                ("main", "pkg://experiments"),
                ("command-line", approved_config_dir.as_uri()),
                ("schema", "structured://"),
            )
            approved_module_cli_search_path = (
                ("hydra", "pkg://hydra.conf"),
                ("command-line", approved_config_dir.as_uri()),
                ("schema", "structured://"),
            )
            approved_layouts = {
                approved_library_search_path: "main",
                approved_cli_search_path: "command-line",
                approved_module_cli_search_path: "command-line",
            }
            if actual_search_path not in approved_layouts:
                raise GlobalCampaignError(
                    "active Hydra search path differs from the exact approved library "
                    "and CLI layouts"
                )
            expected_provider = approved_layouts[actual_search_path]
            expected_path = approved_config_dir.as_uri()
            for config_name in (
                "pilot.yaml",
                f"pilot_model/{variant_id}.yaml",
            ):
                selected = global_hydra.hydra.config_loader.repository.load_config(
                    config_name
                )
                if (
                    selected is None
                    or selected.provider != expected_provider
                    or selected.path != expected_path
                ):
                    raise GlobalCampaignError(
                        f"active Hydra selected an unapproved source for {config_name!r}"
                    )
            pilot_config = compose(config_name="pilot", overrides=list(overrides))
        else:
            pilot_config = compose_pilot_config(overrides)
        pilot = validate_pilot_config(pilot_config)
    except Exception as exc:
        raise GlobalCampaignError(
            f"cannot compose approved pilot model {variant_id!r}"
        ) from exc
    return dict(pilot["pilot_model"])


def validate_global_campaign_config(
    config: DictConfig | Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve and fail-closed validate the one-job campaign contract."""

    value = _mapping(config, field="global campaign")
    _reject_unknown(
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
        field="top-level",
    )
    if (
        type(value.get("schema_version")) is not int
        or value.get("schema_version") != GLOBAL_CAMPAIGN_SCHEMA_VERSION
    ):
        raise GlobalCampaignError("global campaign schema_version must be 1")
    if value.get("project") != PROJECT_NAME:
        raise GlobalCampaignError(f"project must be exactly {PROJECT_NAME!r}")
    if _contains_secret(value):
        raise GlobalCampaignError("credentials are forbidden in Hydra campaign config")
    seed = value.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed != 0:
        raise GlobalCampaignError("global campaign seed must be exactly 0")
    _safe_path(value.get("output_root"), field="output_root")

    execution = _mapping(
        value.get(
            "execution",
            {
                "profile": EXECUTION_PROFILE_LEGACY,
                "strategy": EXECUTION_STRATEGY_SEQUENTIAL,
                "worker_count": 1,
                "training_batch_size_override": None,
                "evaluation_batch_size_override": None,
            },
        ),
        field="execution",
    )
    _reject_unknown(
        execution,
        {
            "profile",
            "strategy",
            "worker_count",
            "training_batch_size_override",
            "evaluation_batch_size_override",
        },
        field="execution",
    )
    if set(execution) != {
        "profile",
        "strategy",
        "worker_count",
        "training_batch_size_override",
        "evaluation_batch_size_override",
    }:
        raise GlobalCampaignError("execution fields differ from contract")
    strategy = execution["strategy"]
    if strategy not in {
        EXECUTION_STRATEGY_SEQUENTIAL,
        EXECUTION_STRATEGY_CELL_DAG,
    }:
        raise GlobalCampaignError("execution.strategy is unsupported")
    worker_count = _positive_int(
        execution["worker_count"], field="execution.worker_count"
    )
    for field in ("training_batch_size_override", "evaluation_batch_size_override"):
        override = execution[field]
        if override is not None:
            _positive_int(override, field=f"execution.{field}")
    profile = execution["profile"]
    if not isinstance(profile, str) or not _EXPERIMENT_NAME.fullmatch(
        profile.replace("_", "-")
    ):
        raise GlobalCampaignError("execution.profile must be a descriptive name")
    if strategy == EXECUTION_STRATEGY_SEQUENTIAL:
        if (
            worker_count != 1
            or execution["training_batch_size_override"] is not None
            or execution["evaluation_batch_size_override"] is not None
        ):
            raise GlobalCampaignError(
                "sequential execution forbids worker and batch overrides"
            )
    elif worker_count > len(APPROVED_MODEL_VARIANTS) * EXPECTED_GLOBAL_CELL_COUNT:
        raise GlobalCampaignError(
            "cell-DAG execution cannot use more workers than campaign cells"
        )
    if profile == EXECUTION_PROFILE_H100 and execution != {
        "profile": EXECUTION_PROFILE_H100,
        "strategy": EXECUTION_STRATEGY_CELL_DAG,
        "worker_count": 8,
        "training_batch_size_override": 4096,
        "evaluation_batch_size_override": 512,
    }:
        raise GlobalCampaignError("the H100 execution profile is immutable")
    value["execution"] = execution

    data = _mapping(value.get("data"), field="data")
    _reject_unknown(
        data,
        {
            "root",
            "canonical_archive",
            "canonical_extracted_root",
            "generated_root",
            "canonical_pca",
            "generated_manifest",
            "registry",
            "mmap_mode",
        },
        field="data",
    )
    for field in (
        "root",
        "canonical_archive",
        "canonical_extracted_root",
        "generated_root",
        "canonical_pca",
        "generated_manifest",
        "registry",
    ):
        _safe_path(data.get(field), field=f"data.{field}")
    generated_root = Path(str(data["generated_root"]))
    generated_manifest = Path(str(data["generated_manifest"]))
    if generated_manifest != generated_root / "generated_e3_e4_manifest.json":
        raise GlobalCampaignError(
            "data.generated_manifest must be the sealed manifest inside generated_root"
        )
    if not str(data["registry"]).endswith(".yaml"):
        raise GlobalCampaignError("data.registry must be YAML")
    if data["registry"] != "configs/datasets/registry/paper_benchmarks.yaml":
        raise GlobalCampaignError("data.registry differs from canonical registry")
    if data.get("mmap_mode") not in {None, "r"}:
        raise GlobalCampaignError("data.mmap_mode must be null or 'r'")

    logging = _mapping(value.get("logging"), field="logging")
    _reject_unknown(logging, {"backend", "project", "workspace"}, field="logging")
    if logging.get("backend") not in {"none", "comet"}:
        raise GlobalCampaignError("logging.backend must be none or comet")
    if logging.get("project") != PROJECT_NAME:
        raise GlobalCampaignError("logging.project differs from project")
    if logging.get("workspace") != WORKSPACE_NAME:
        raise GlobalCampaignError(f"logging.workspace must be {WORKSPACE_NAME!r}")

    campaign = _mapping(value.get("campaign"), field="campaign")
    _reject_unknown(
        campaign,
        {
            "campaign_id",
            "inventory",
            "models",
            "selection",
            "evaluation",
            "resume",
            "fm_diagnostics",
        },
        field="campaign",
    )
    campaign_id = campaign.get("campaign_id")
    if not isinstance(campaign_id, str) or not _EXPERIMENT_NAME.fullmatch(campaign_id):
        raise GlobalCampaignError("campaign.campaign_id must be lowercase kebab-case")
    if campaign_id != APPROVED_CAMPAIGN_ID:
        raise GlobalCampaignError("campaign.campaign_id differs from approved campaign")

    inventory = _mapping(campaign.get("inventory"), field="campaign.inventory")
    _reject_unknown(
        inventory,
        {
            "canonical",
            "generated_e3_e4",
            "require_suite_ids",
            "include_all_representations",
        },
        field="campaign.inventory",
    )
    for field in ("canonical", "generated_e3_e4"):
        path = _safe_path(inventory.get(field), field=f"campaign.inventory.{field}")
        if path.suffix != ".yaml":
            raise GlobalCampaignError(f"campaign.inventory.{field} must be YAML")
    if inventory["canonical"] != "configs/global_suite/canonical_exact.yaml":
        raise GlobalCampaignError("canonical global inventory path is not approved")
    if inventory["generated_e3_e4"] != ("configs/global_suite/generated_e3_e4.yaml"):
        raise GlobalCampaignError("generated E3/E4 inventory path is not approved")
    if tuple(inventory.get("require_suite_ids", ())) != REQUIRED_SUITE_IDS:
        raise GlobalCampaignError(
            f"campaign must require exactly {list(REQUIRED_SUITE_IDS)!r}"
        )
    if inventory.get("include_all_representations") is not True:
        raise GlobalCampaignError("global campaign must include all representations")

    raw_models = campaign.get("models")
    if not isinstance(raw_models, list):
        raise GlobalCampaignError("campaign.models must be a list")
    if tuple(row.get("id") for row in raw_models if isinstance(row, dict)) != (
        APPROVED_MODEL_VARIANTS
    ):
        raise GlobalCampaignError(
            "campaign.models must contain all ten approved variants in fixed order"
        )
    seen_names: set[str] = set()
    for index, row in enumerate(raw_models):
        model_row = _mapping(row, field=f"campaign.models[{index}]")
        _reject_unknown(model_row, {"id", "experiment_name"}, field="model row")
        name = model_row.get("experiment_name")
        if (
            not isinstance(name, str)
            or not _EXPERIMENT_NAME.fullmatch(name)
            or name.startswith("ent-block-")
            or name.endswith("-eval")
        ):
            raise GlobalCampaignError(
                f"model {model_row.get('id')!r} has a non-descriptive Comet name"
            )
        if name in seen_names:
            raise GlobalCampaignError("Comet experiment names must be unique")
        if name != APPROVED_EXPERIMENT_NAMES[index]:
            raise GlobalCampaignError(
                f"model {model_row.get('id')!r} Comet name differs from approved name"
            )
        seen_names.add(name)
        _resolved_pilot_model(str(model_row["id"]), seed)

    selection = _mapping(campaign.get("selection"), field="campaign.selection")
    required_selection = {
        "known_lid_protocol",
        "unknown_lid_protocol",
        "index_algorithm",
        "fraction",
        "minimum_selection",
        "maximum_selection",
        "minimum_fit",
        "criterion",
        "stability_window",
        "stability_min_valid_fraction",
        "tie_tolerance",
    }
    if set(selection) != required_selection:
        raise GlobalCampaignError("campaign.selection fields differ from contract")
    if selection["known_lid_protocol"] != KNOWN_SELECTION_PROTOCOL:
        raise GlobalCampaignError("unsupported known-LID selection protocol")
    if selection["unknown_lid_protocol"] != UNKNOWN_SELECTION_PROTOCOL:
        raise GlobalCampaignError("unsupported unknown-LID selection protocol")
    if selection["index_algorithm"] != "splitmix64_rank_v1":
        raise GlobalCampaignError("unsupported holdout index algorithm")
    for field in (
        "minimum_selection",
        "maximum_selection",
        "minimum_fit",
        "stability_window",
    ):
        _positive_int(selection[field], field=f"campaign.selection.{field}")
    fraction = selection["fraction"]
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not 0 < float(fraction) < 1
    ):
        raise GlobalCampaignError("selection.fraction must lie in (0, 1)")
    if selection["minimum_selection"] > selection["maximum_selection"]:
        raise GlobalCampaignError("selection minimum exceeds maximum")
    if selection["criterion"] != "mae":
        raise GlobalCampaignError("selection criterion must be mae")
    valid_fraction = selection["stability_min_valid_fraction"]
    if (
        isinstance(valid_fraction, bool)
        or not isinstance(valid_fraction, (int, float))
        or not 0 < float(valid_fraction) <= 1
    ):
        raise GlobalCampaignError("invalid stability_min_valid_fraction")
    tolerance = selection["tie_tolerance"]
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or float(tolerance) < 0
    ):
        raise GlobalCampaignError("tie_tolerance must be non-negative")

    evaluation = _mapping(campaign.get("evaluation"), field="campaign.evaluation")
    legacy_evaluation_fields = {
        "batch_size",
        "frozen_candidate_count",
        "save_retrospective_validation_curves",
        "save_retrospective_test_curves",
    }
    if frozenset(evaluation) not in {
        frozenset(legacy_evaluation_fields),
        frozenset({*legacy_evaluation_fields, "checkpoint_retention"}),
    }:
        raise GlobalCampaignError("campaign.evaluation fields differ from contract")
    _positive_int(evaluation["batch_size"], field="campaign.evaluation.batch_size")
    if (
        type(evaluation["frozen_candidate_count"]) is not int
        or evaluation["frozen_candidate_count"] != 1
    ):
        raise GlobalCampaignError("exactly one frozen candidate is required")
    if evaluation["save_retrospective_validation_curves"] is not False:
        raise GlobalCampaignError("retrospective validation curves are forbidden")
    if evaluation["save_retrospective_test_curves"] is not False:
        raise GlobalCampaignError("retrospective test curves are forbidden")
    if (
        evaluation.get("checkpoint_retention", CHECKPOINT_RETENTION_RETAIN)
        not in SUPPORTED_CHECKPOINT_RETENTION_POLICIES
    ):
        raise GlobalCampaignError(
            "evaluation.checkpoint_retention must be retain or "
            "prune_after_cell_evaluation"
        )

    resume = _mapping(campaign.get("resume"), field="campaign.resume")
    expected_resume = {
        "schema_version": 1,
        "cell_attempt_policy": "stable_identity_directory",
        "training_progress_filename": "training_progress.pt",
        "invalid_existing_policy": "fail",
        "source_tree_check_before_every_cell": True,
        "lock_filename": "campaign.lock",
    }
    if type(resume.get("schema_version")) is not int or resume != expected_resume:
        raise GlobalCampaignError("campaign.resume differs from the safe contract")
    diagnostics = _mapping(
        campaign.get("fm_diagnostics"), field="campaign.fm_diagnostics"
    )
    if diagnostics != {
        "policy": "known_lid_only",
        "unknown_lid_status": "not_applicable_no_lid_targets",
    }:
        raise GlobalCampaignError("campaign.fm_diagnostics differs from contract")
    return value


def model_plans(config: Mapping[str, Any]) -> tuple[ModelPlan, ...]:
    seed = int(config["seed"])
    execution = config.get("execution", {})
    training_batch_override = (
        execution.get("training_batch_size_override")
        if isinstance(execution, Mapping)
        else None
    )
    evaluation_batch_override = (
        execution.get("evaluation_batch_size_override")
        if isinstance(execution, Mapping)
        else None
    )

    def resolved_model(variant_id: str) -> dict[str, Any]:
        model = _resolved_pilot_model(variant_id, seed)
        resolved = dict(model)
        if training_batch_override is not None:
            training = _mapping(model.get("training"), field=f"{variant_id}.training")
            training["batch_size"] = int(training_batch_override)
            resolved["training"] = training

        # The outer selector and affine-FM diagnostics independently evaluate
        # the same Hutchinson estimator.  Its probe stream is consumed inside
        # the predictor's batch loop, so the batch partition is part of that
        # deterministic evaluation contract.  Keep both paths aligned when a
        # production profile increases the effective evaluation batch size.
        if (
            evaluation_batch_override is not None
            and model.get("family") == "independent_affine_flow"
        ):
            diagnostics = _mapping(
                model.get("diagnostics"), field=f"{variant_id}.diagnostics"
            )
            diagnostics["batch_size"] = int(evaluation_batch_override)
            resolved["diagnostics"] = diagnostics
        return resolved

    return tuple(
        ModelPlan(
            variant_id=str(row["id"]),
            experiment_name=str(row["experiment_name"]),
            model=resolved_model(str(row["id"])),
        )
        for row in config["campaign"]["models"]
    )


def adaptive_holdout_size(n_samples: int, selection: Mapping[str, Any]) -> int:
    """Return the exact declared bounded-fraction holdout size."""

    if isinstance(n_samples, bool) or not isinstance(n_samples, int):
        raise GlobalCampaignError("n_samples must be an integer")
    minimum_fit = int(selection["minimum_fit"])
    minimum = int(selection["minimum_selection"])
    if n_samples < minimum_fit + minimum:
        raise GlobalCampaignError(
            f"source train split {n_samples} cannot leave {minimum_fit} fit rows "
            f"and {minimum} selection rows"
        )
    fractional = math.floor(float(selection["fraction"]) * n_samples)
    return min(
        int(selection["maximum_selection"]),
        max(minimum, fractional),
        n_samples - minimum_fit,
    )


def _splitmix64_indices(
    n_samples: int, *, subset_size: int, seed: int
) -> npt.NDArray[np.int64]:
    if not 0 < subset_size < n_samples:
        raise GlobalCampaignError("holdout size must lie strictly inside train size")
    indices = np.arange(n_samples, dtype=np.uint64)
    with np.errstate(over="ignore"):
        mixed = indices + np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15)
        mixed = (mixed ^ (mixed >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        mixed = (mixed ^ (mixed >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        mixed ^= mixed >> np.uint64(31)
    ranked = np.argsort(mixed, kind="stable")[:subset_size]
    return np.sort(ranked.astype(np.int64, copy=False))


def _array_sha(value: npt.ArrayLike) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    identity = {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "bytes_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }
    return sha256_bytes(canonical_json(identity).encode("utf-8"))


def partition_source_train(
    train: npt.ArrayLike,
    target: npt.ArrayLike | None,
    *,
    selection: Mapping[str, Any],
    seed: int,
) -> HoldoutPartition:
    features = np.asarray(train)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise GlobalCampaignError("source train features must be a finite matrix")
    truth: npt.NDArray[np.float64] | None = None
    if target is not None:
        truth = np.ravel(np.asarray(target, dtype=np.float64))
        if truth.shape != (features.shape[0],) or not np.isfinite(truth).all():
            raise GlobalCampaignError("source train LID target is invalid")
    subset_size = adaptive_holdout_size(int(features.shape[0]), selection)
    selected = _splitmix64_indices(
        int(features.shape[0]), subset_size=subset_size, seed=seed
    )
    mask = np.ones(features.shape[0], dtype=bool)
    mask[selected] = False
    fit = np.flatnonzero(mask).astype(np.int64, copy=False)
    fit_features = np.ascontiguousarray(features[fit])
    selected_features = np.ascontiguousarray(features[selected])
    selected_target = None if truth is None else np.ascontiguousarray(truth[selected])
    record = {
        "schema_version": 1,
        "protocol": "adaptive_held_out_source_train_v1",
        "index_algorithm": "splitmix64_rank_v1",
        "seed": seed,
        "n_source_train": int(features.shape[0]),
        "n_optimizer_fit": int(fit.size),
        "n_train_selection": int(selected.size),
        "optimizer_overlap_count": 0,
        "fit_indices_sha256": _array_sha(fit),
        "selection_indices_sha256": _array_sha(selected),
        "fit_features_sha256": _array_sha(fit_features),
        "selection_features_sha256": _array_sha(selected_features),
        # FM diagnostics use the shared semantic-array digest rather than the
        # source-inventory digest above.  Persist both so a pruned cell can
        # still bind its diagnostic query to these exact held-out rows.
        "selection_features_fm_sha256": _array_sha256(selected_features),
        **(
            {"selection_target_sha256": _array_sha(selected_target)}
            if selected_target is not None
            else {}
        ),
    }
    record["partition_sha256"] = sha256_bytes(canonical_json(record).encode("utf-8"))
    return HoldoutPartition(
        fit_indices=fit,
        selection_indices=selected,
        fit_features=fit_features,
        selection_features=selected_features,
        selection_target=selected_target,
        record=record,
    )


def _resolve_path(root: Path, configured: str) -> Path:
    path = Path(configured)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _flatten(split: LoadedSplit) -> npt.NDArray[Any]:
    values = np.asarray(split.features)
    return np.ascontiguousarray(values.reshape(values.shape[0], -1))


def _source_record(splits: Mapping[str, LoadedSplit], root: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for split_name, split in splits.items():
        for artifact, path in split.source_paths.items():
            resolved = path.resolve()
            try:
                relative = resolved.relative_to(root).as_posix()
            except ValueError as exc:
                raise GlobalCampaignError(
                    f"dataset input escapes root: {path}"
                ) from exc
            files[f"{split_name}/{artifact}.npy"] = {
                "path": relative,
                "size_bytes": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
    return dict(sorted(files.items()))


def load_campaign_cell_data(
    cell: CampaignCell, config: Mapping[str, Any], project_root: Path
) -> CellData:
    registry_path = _resolve_path(project_root, cell.registry)
    registry = load_registry(registry_path, validate_official_coverage=False)
    if cell.registry_overlay is not None:
        registry = apply_registry_overlay(
            registry, _resolve_path(project_root, cell.registry_overlay)
        )
    try:
        spec = registry[cell.dataset]
    except KeyError as exc:
        raise GlobalCampaignError(
            f"registry has no campaign dataset {cell.dataset!r}"
        ) from exc
    configured_root = (
        config["data"]["root"]
        if cell.source_kind == "exact_archive"
        else config["data"]["generated_root"]
    )
    data_root = _resolve_path(project_root, str(configured_root))
    splits = load_dataset(
        data_root,
        spec,
        representation=cell.representation,
        mmap_mode=config["data"].get("mmap_mode"),
    )
    if tuple(splits) != ("train", "val", "test"):
        raise GlobalCampaignError(f"{cell.key} must expose train/val/test")
    train_target = splits["train"].lid
    validation_target = splits["val"].lid
    test_target = splits["test"].lid
    has_targets = train_target is not None
    if has_targets != (cell.target_policy == "known_lid"):
        raise GlobalCampaignError(
            f"inventory target policy disagrees with registry for {cell.key}"
        )
    if (validation_target is None) != (not has_targets) or (test_target is None) != (
        not has_targets
    ):
        raise GlobalCampaignError(f"partial LID targets are forbidden for {cell.key}")
    source_files = _source_record(splits, data_root)
    input_record = {
        "schema_version": 1,
        "inventory_id": cell.inventory_id,
        "source_kind": cell.source_kind,
        "source_contract": {
            "exact_archive": cell.exact_archive,
            "dataset_config": cell.dataset_config,
            "data_root": cell.data_root,
            "upstream_revision": cell.upstream_revision,
            "archive_sha256": cell.archive_sha256,
            "provenance_label": cell.provenance_label,
        },
        "registry": {
            "path": cell.registry,
            "sha256": sha256_file(registry_path),
            **(
                {
                    "overlay_path": cell.registry_overlay,
                    "overlay_sha256": sha256_file(
                        _resolve_path(project_root, cell.registry_overlay)
                    ),
                }
                if cell.registry_overlay is not None
                else {}
            ),
        },
        "suite_id": cell.suite_id,
        "dataset": cell.dataset,
        "representation": cell.representation,
        "feature_shape": list(splits["train"].feature_shape),
        "source_files": source_files,
        "applied_overrides": {
            name: _plain(split.applied_overrides) for name, split in splits.items()
        },
    }
    input_sha = sha256_bytes(canonical_json(input_record).encode("utf-8"))
    return CellData(
        train=_flatten(splits["train"]),
        validation=_flatten(splits["val"]),
        test=_flatten(splits["test"]),
        train_target=(
            None
            if train_target is None
            else np.ascontiguousarray(np.asarray(train_target, dtype=np.float64))
        ),
        validation_target=(
            None
            if validation_target is None
            else np.ascontiguousarray(np.asarray(validation_target, dtype=np.float64))
        ),
        test_target=(
            None
            if test_target is None
            else np.ascontiguousarray(np.asarray(test_target, dtype=np.float64))
        ),
        validation_labels=(
            None
            if splits["val"].labels is None
            else np.ascontiguousarray(np.asarray(splits["val"].labels))
        ),
        test_labels=(
            None
            if splits["test"].labels is None
            else np.ascontiguousarray(np.asarray(splits["test"].labels))
        ),
        input_record=input_record,
        input_sha256=input_sha,
    )


def load_campaign_inventory(
    config: Mapping[str, Any], project_root: Path
) -> tuple[CampaignCell, ...]:
    """Load and merge the canonical and explicit generated E3/E4 inventories."""

    try:
        from experiments.global_inventory import load_global_inventory
    except ImportError as exc:
        raise GlobalCampaignError("global inventory module is unavailable") from exc
    inventory_config = config["campaign"]["inventory"]
    inventories = []
    for field in ("canonical", "generated_e3_e4"):
        path = _resolve_path(project_root, str(inventory_config[field]))
        if not path.is_file():
            raise GlobalCampaignError(f"required global inventory is missing: {path}")
        try:
            inventories.append(load_global_inventory(path, project_root=project_root))
        except Exception as exc:
            raise GlobalCampaignError(f"cannot load global inventory {path}") from exc
    cells: list[CampaignCell] = []
    available_suites: set[str] = set()
    for inventory in inventories:
        source = inventory.source
        registry = str(source.registry)
        for suite in inventory.suites:
            if suite.available:
                available_suites.add(str(suite.suite_id))
            for raw in suite.cells:
                cells.append(
                    CampaignCell(
                        inventory_id=str(inventory.inventory_id),
                        source_kind=str(source.kind),
                        exact_archive=bool(source.exact_archive),
                        dataset_config=str(source.dataset_config),
                        data_root=str(source.data_root),
                        registry=registry,
                        registry_overlay=(
                            None
                            if source.registry_overlay is None
                            else str(source.registry_overlay)
                        ),
                        upstream_revision=str(source.upstream_revision),
                        archive_sha256=(
                            None
                            if source.archive_sha256 is None
                            else str(source.archive_sha256)
                        ),
                        provenance_label=str(source.provenance_label),
                        suite_id=str(raw.suite_id),
                        dataset=str(raw.dataset),
                        representation=str(raw.representation),
                        target_policy=str(raw.target_policy),
                        selection_protocol=str(raw.selection_protocol),
                        comparison_group=str(raw.comparison_group),
                        reference_dataset=(
                            None
                            if raw.reference_dataset is None
                            else str(raw.reference_dataset)
                        ),
                        expected_lid_delta=(
                            None
                            if raw.expected_lid_delta is None
                            else float(raw.expected_lid_delta)
                        ),
                    )
                )
    if available_suites != set(REQUIRED_SUITE_IDS):
        raise GlobalCampaignError(
            "merged inventories do not provide all required suites: "
            f"available={sorted(available_suites)}"
        )
    keys = [cell.key for cell in cells]
    if not cells or len(keys) != len(set(keys)):
        raise GlobalCampaignError("global inventory is empty or repeats cells")
    order = {suite: index for index, suite in enumerate(REQUIRED_SUITE_IDS)}
    cells.sort(key=lambda cell: order[cell.suite_id])
    # Every reference must already precede its dependants in the authored
    # inventory.  Preserve that exact order because it is part of the campaign
    # allowlist and the unified report ordering.
    result: list[CampaignCell] = []
    for suite_id in REQUIRED_SUITE_IDS:
        suite_cells = [cell for cell in cells if cell.suite_id == suite_id]
        available: set[tuple[str, str]] = set()
        for cell in suite_cells:
            if (
                cell.reference_dataset not in {None, cell.dataset}
                and (cell.reference_dataset, cell.representation) not in available
            ):
                raise GlobalCampaignError(
                    f"inventory reference does not precede dependent cell {cell.key}"
                )
            available.add((cell.dataset, cell.representation))
        result.extend(suite_cells)
    return tuple(result)


def validate_campaign_sources(
    config: Mapping[str, Any],
    project_root: Path,
    cells: Sequence[CampaignCell],
) -> Mapping[str, Mapping[str, Any]]:
    """Validate source-level seals once, before Comet or model work starts."""

    from datasets.archive import (
        EXACT_ARCHIVE_SHA256,
        verify_exact_archive,
        verify_extracted_tree,
    )
    from datasets.generated_e3_e4 import (
        MANIFEST_FILENAME,
        validate_generated_e3_e4,
    )
    from utils.provenance import EXPECTED_LID_BENCHMARKS_SHA

    records: dict[str, Mapping[str, Any]] = {}
    by_inventory: dict[str, list[CampaignCell]] = {}
    for cell in cells:
        by_inventory.setdefault(cell.inventory_id, []).append(cell)
    for inventory_id, related in by_inventory.items():
        first = related[0]
        source_identity = {
            "source_kind": first.source_kind,
            "exact_archive": first.exact_archive,
            "dataset_config": first.dataset_config,
            "data_root": first.data_root,
            "registry": first.registry,
            "registry_overlay": first.registry_overlay,
            "upstream_revision": first.upstream_revision,
            "archive_sha256": first.archive_sha256,
            "provenance_label": first.provenance_label,
        }
        for cell in related[1:]:
            candidate = {
                "source_kind": cell.source_kind,
                "exact_archive": cell.exact_archive,
                "dataset_config": cell.dataset_config,
                "data_root": cell.data_root,
                "registry": cell.registry,
                "registry_overlay": cell.registry_overlay,
                "upstream_revision": cell.upstream_revision,
                "archive_sha256": cell.archive_sha256,
                "provenance_label": cell.provenance_label,
            }
            if canonical_json(candidate) != canonical_json(source_identity):
                raise GlobalCampaignError(
                    f"inventory {inventory_id!r} has inconsistent source identity"
                )
        if first.upstream_revision != EXPECTED_LID_BENCHMARKS_SHA:
            raise GlobalCampaignError(
                f"inventory {inventory_id!r} has an unapproved upstream revision"
            )
        if first.source_kind == "exact_archive":
            if not first.exact_archive or first.archive_sha256 != EXACT_ARCHIVE_SHA256:
                raise GlobalCampaignError(
                    f"inventory {inventory_id!r} is not bound to the exact archive"
                )
            configured_root = _resolve_path(project_root, str(config["data"]["root"]))
            inventory_root = _resolve_path(project_root, first.data_root)
            if configured_root != inventory_root:
                raise GlobalCampaignError(
                    "canonical config root differs from its inventory data_root"
                )
            archive_path = _resolve_path(
                project_root, str(config["data"]["canonical_archive"])
            )
            extracted_root = _resolve_path(
                project_root, str(config["data"]["canonical_extracted_root"])
            )
            if configured_root != extracted_root / "benchmarks":
                raise GlobalCampaignError(
                    "canonical extracted root does not contain configured data root"
                )
            try:
                archive_manifest = verify_exact_archive(archive_path)
                verify_extracted_tree(extracted_root, archive_manifest)
            except Exception as exc:
                raise GlobalCampaignError(
                    "canonical archive/extracted tree failed exact verification"
                ) from exc
            records[inventory_id] = {
                "schema_version": 1,
                "kind": "canonical_exact_archive_contract",
                "archive_sha256": EXACT_ARCHIVE_SHA256,
                "archive_size_bytes": archive_manifest.archive_size_bytes,
                "archive_file_count": archive_manifest.file_count,
                "archive_uncompressed_size_bytes": (
                    archive_manifest.uncompressed_size_bytes
                ),
                "archive_path": str(config["data"]["canonical_archive"]),
                "extracted_root": str(config["data"]["canonical_extracted_root"]),
                "upstream_revision": first.upstream_revision,
                "configured_extracted_root": str(config["data"]["root"]),
            }
            continue
        if first.source_kind != "generated_at_pinned_revision" or first.exact_archive:
            raise GlobalCampaignError(
                f"inventory {inventory_id!r} has an unsupported source kind"
            )
        generated_root = _resolve_path(
            project_root, str(config["data"]["generated_root"])
        )
        inventory_root = _resolve_path(project_root, first.data_root)
        if generated_root != inventory_root:
            raise GlobalCampaignError(
                "generated config root differs from its inventory data_root"
            )
        pca_path = _resolve_path(project_root, str(config["data"]["canonical_pca"]))
        configured_manifest = _resolve_path(
            project_root, str(config["data"]["generated_manifest"])
        )
        expected_manifest = generated_root / MANIFEST_FILENAME
        if configured_manifest != expected_manifest:
            raise GlobalCampaignError(
                "configured generated manifest differs from generated_root seal"
            )
        try:
            result = validate_generated_e3_e4(
                generated_root,
                pca_path,
                checkout=project_root / "lid_benchmarks",
            )
        except Exception as exc:
            raise GlobalCampaignError(
                "generated E3/E4 source failed sealed preflight"
            ) from exc
        if (
            result.output_root != generated_root
            or result.manifest_path != expected_manifest
        ):
            raise GlobalCampaignError("generated E3/E4 validator returned wrong paths")
        manifest = _load_json(expected_manifest)
        if result.upstream_revision != first.upstream_revision:
            raise GlobalCampaignError("generated E3/E4 upstream identity differs")
        if manifest.get("upstream", {}).get("revision") != first.upstream_revision:
            raise GlobalCampaignError("generated E3/E4 manifest revision differs")
        if manifest.get("pca", {}).get("sha256") != result.pca_sha256:
            raise GlobalCampaignError("generated E3/E4 PCA identity differs")
        records[inventory_id] = {
            "schema_version": 1,
            "kind": "generated_e3_e4_sealed_extension",
            "manifest_path": str(config["data"]["generated_manifest"]),
            "manifest_file_sha256": sha256_path(expected_manifest),
            "content_tree_sha256": manifest.get("content_tree_sha256"),
            "seal_sha256": manifest.get("seal_sha256"),
            "pca_path": str(config["data"]["canonical_pca"]),
            "pca_sha256": result.pca_sha256,
            "upstream_revision": result.upstream_revision,
        }
    if set(records) != set(by_inventory):
        raise GlobalCampaignError("source preflight did not cover every inventory")
    return records


def _bind_source_preflight(
    data: CellData,
    cell: CampaignCell,
    source_records: Mapping[str, Mapping[str, Any]],
) -> CellData:
    try:
        source_record = source_records[cell.inventory_id]
    except KeyError as exc:
        raise GlobalCampaignError(
            f"source preflight has no record for {cell.inventory_id!r}"
        ) from exc
    input_record = _plain(data.input_record)
    if not isinstance(input_record, dict):
        raise GlobalCampaignError("cell input_record must be a mapping")
    if "source_preflight" in input_record:
        raise GlobalCampaignError("cell loader may not predeclare source_preflight")
    input_record["source_preflight"] = _plain(source_record)
    input_sha = sha256_bytes(canonical_json(input_record).encode("utf-8"))
    return replace(data, input_record=input_record, input_sha256=input_sha)


def _write_json(path: Path, value: Any) -> None:
    payload = _plain(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(_plain(value), sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _save_npy(path: Path, value: npt.ArrayLike) -> None:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise GlobalCampaignError(f"refusing to save invalid numeric array {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _checkpoint_retention_policy(evaluation: Mapping[str, Any]) -> str:
    policy = evaluation.get("checkpoint_retention", CHECKPOINT_RETENTION_RETAIN)
    if policy not in SUPPORTED_CHECKPOINT_RETENTION_POLICIES:
        raise GlobalCampaignError("cell checkpoint retention policy is invalid")
    return str(policy)


def _output_inventory(
    directory: Path,
    *,
    excluded_relative_paths: frozenset[str] = frozenset(),
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise GlobalCampaignError(f"symlink is forbidden in cell output: {path}")
        if not path.is_file() or path.name == "manifest.json":
            continue
        relative = path.relative_to(directory).as_posix()
        if relative in excluded_relative_paths:
            continue
        records[relative] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_path(path),
        }
    return records


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GlobalCampaignError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GlobalCampaignError(f"{path} must contain a JSON object")
    return value


def _load_numeric_array(path: Path, *, ndim: int) -> npt.NDArray[Any]:
    try:
        value = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise GlobalCampaignError(
            f"cannot load numeric artifact {path}: {exc}"
        ) from exc
    array = np.asarray(value)
    if (
        array.ndim != ndim
        or not np.issubdtype(array.dtype, np.number)
        or not np.isfinite(array).all()
    ):
        raise GlobalCampaignError(
            f"numeric artifact {path} must be a finite {ndim}-D array"
        )
    return np.ascontiguousarray(array)


def _same_json(left: Any, right: Any) -> bool:
    try:
        return canonical_json(_plain(left)) == canonical_json(_plain(right))
    except (TypeError, ValueError):
        return False


def _source_evidence(data: CellData, config: Mapping[str, Any]) -> dict[str, Any]:
    partition = partition_source_train(
        data.train,
        data.train_target,
        selection=config["campaign"]["selection"],
        seed=int(config["seed"]),
    )

    def digest(value: npt.ArrayLike | None) -> str | None:
        return None if value is None else _array_sha(value)

    return {
        "partition": _plain(partition.record),
        "fit_indices_sha256": _array_sha(partition.fit_indices),
        "selection_indices_sha256": _array_sha(partition.selection_indices),
        "train_selection_target_sha256": digest(partition.selection_target),
        "validation_target_sha256": digest(data.validation_target),
        "test_target_sha256": digest(data.test_target),
        "validation_labels_sha256": digest(data.validation_labels),
        "test_labels_sha256": digest(data.test_labels),
        "validation_n": int(data.validation.shape[0]),
        "test_n": int(data.test.shape[0]),
    }


def _validate_training_attestation(
    attestation: Mapping[str, Any],
    *,
    model: Mapping[str, Any],
    checkpoint_retention: str,
    checkpoint_sha256: Any,
) -> list[str]:
    errors: list[str] = []
    training_config: dict[str, Any] | None = None
    required = {
        "schema_version",
        "model_family",
        "training_config_sha256",
        "checkpoint_sha256",
        "checkpoint_size_bytes",
        "checkpoint_retention",
        "history",
    }
    if set(attestation) != required:
        errors.append("training attestation fields differ from contract")
    if attestation.get("schema_version") != TRAINING_ATTESTATION_SCHEMA_VERSION or (
        isinstance(attestation.get("schema_version"), bool)
    ):
        errors.append("training attestation schema_version is invalid")
    if attestation.get("model_family") != model.get("family"):
        errors.append("training attestation model family differs")
    try:
        training_config = _canonical_training_config_record(
            model["training"], field="attested Hydra model training config"
        )
    except (GlobalCampaignError, KeyError) as exc:
        errors.append(f"cannot attest training config: {exc}")
    else:
        expected_training_sha = sha256_bytes(
            canonical_json(training_config).encode("utf-8")
        )
        if attestation.get("training_config_sha256") != expected_training_sha:
            errors.append("training attestation config SHA differs")
    attested_sha = attestation.get("checkpoint_sha256")
    if not isinstance(attested_sha, str) or not _SHA256.fullmatch(attested_sha):
        errors.append("training attestation checkpoint SHA is invalid")
    if attested_sha != checkpoint_sha256:
        errors.append("training attestation checkpoint SHA differs from summary")
    size = attestation.get("checkpoint_size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        errors.append("training attestation checkpoint size is invalid")
    if attestation.get("checkpoint_retention") != checkpoint_retention:
        errors.append("training attestation checkpoint retention differs")

    history = attestation.get("history")
    if not isinstance(history, Mapping) or set(history) != {
        "status",
        "best_epoch",
        "best_validation_loss",
        "epochs",
    }:
        errors.append("training attestation history schema differs")
        return errors
    status = history.get("status")
    epochs = history.get("epochs")
    if status == "unavailable_from_trainer":
        if (
            history.get("best_epoch") is not None
            or history.get("best_validation_loss") is not None
            or epochs != []
        ):
            errors.append("unavailable training history contains measurements")
        if checkpoint_retention == CHECKPOINT_RETENTION_PRUNE_AFTER_EVALUATION:
            errors.append("pruned checkpoint requires complete training history")
        return errors
    if status != "complete" or not isinstance(epochs, list) or not epochs:
        errors.append("training attestation history is invalid")
        return errors
    previous_epoch = 0
    validation_by_epoch: dict[int, float] = {}
    history_rows_valid = True
    for row in epochs:
        if not isinstance(row, Mapping) or set(row) != {
            "epoch",
            "train_loss",
            "validation_loss",
            "learning_rate",
        }:
            errors.append("training attestation epoch fields differ")
            history_rows_valid = False
            continue
        epoch = row.get("epoch")
        if (
            isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or epoch <= previous_epoch
        ):
            errors.append("training attestation epochs do not strictly increase")
            history_rows_valid = False
            continue
        previous_epoch = epoch
        numeric_valid = True
        for field in ("train_loss", "validation_loss", "learning_rate"):
            value = row.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                errors.append(f"training attestation {field} is not finite")
                numeric_valid = False
                history_rows_valid = False
        if numeric_valid:
            validation_by_epoch[epoch] = float(row["validation_loss"])
    best_epoch = history.get("best_epoch")
    best_loss = history.get("best_validation_loss")
    if (
        isinstance(best_epoch, bool)
        or not isinstance(best_epoch, int)
        or best_epoch not in validation_by_epoch
        or isinstance(best_loss, bool)
        or not isinstance(best_loss, (int, float))
        or not math.isfinite(float(best_loss))
    ):
        errors.append("training attestation best history fields are invalid")
    elif float(best_loss) != validation_by_epoch[best_epoch]:
        errors.append("training attestation best loss differs from history")
    elif history_rows_valid and len(validation_by_epoch) == len(epochs):
        ordered_epochs = list(validation_by_epoch)
        expected_best_epoch = min(
            ordered_epochs, key=lambda epoch: validation_by_epoch[epoch]
        )
        expected_best_loss = validation_by_epoch[expected_best_epoch]
        if best_epoch != expected_best_epoch or float(best_loss) != expected_best_loss:
            errors.append(
                "training attestation best epoch/loss is not the first strict minimum"
            )

        if training_config is not None:
            epoch_budget = int(training_config["epochs"])
            validation_interval = int(training_config["validation_interval"])
            patience = training_config["early_stopping_patience"]
            last_epoch = ordered_epochs[-1]
            full_schedule = list(
                range(validation_interval, epoch_budget + 1, validation_interval)
            )
            if not full_schedule or full_schedule[-1] != epoch_budget:
                full_schedule.append(epoch_budget)
            if last_epoch == epoch_budget:
                if ordered_epochs != full_schedule:
                    errors.append(
                        "training attestation validation schedule differs from config"
                    )
            elif last_epoch > epoch_budget or patience is None:
                errors.append(
                    "training attestation termination is not justified by config"
                )
            else:
                early_schedule = list(
                    range(validation_interval, last_epoch + 1, validation_interval)
                )
                if ordered_epochs != early_schedule:
                    errors.append(
                        "training attestation early-stop schedule differs from config"
                    )
                stale_validations = (
                    len(ordered_epochs) - ordered_epochs.index(expected_best_epoch) - 1
                )
                if stale_validations != int(patience):
                    errors.append(
                        "training attestation early stop does not match patience"
                    )
    return errors


def _validate_affine_diagnostic_binding(
    root: Path,
    *,
    fm: Mapping[str, Any],
    model: Mapping[str, Any],
    checkpoint_sha256: Any,
    scales: npt.NDArray[np.float64],
    selection_curve: npt.NDArray[np.float64],
    partition: Mapping[str, Any],
) -> list[str]:
    """Bind a self-valid FM diagnostic subtree to this exact outer cell."""

    errors: list[str] = []
    directory = root / "fm_diagnostics"
    try:
        metadata = _load_json(directory / "metadata.json")
        diagnostic_scales = np.ravel(
            np.asarray(
                _load_numeric_array(directory / "arrays" / "scales.npy", ndim=1),
                dtype=np.float64,
            )
        )
        diagnostic_target = np.asarray(
            _load_numeric_array(directory / "arrays" / "target.npy", ndim=1),
            dtype=np.float64,
        )
        diagnostic_full = np.asarray(
            _load_numeric_array(directory / "arrays" / "full.npy", ndim=2),
            dtype=np.float64,
        )
        diagnostic_response = np.asarray(
            _load_numeric_array(directory / "arrays" / "response.npy", ndim=2),
            dtype=np.float64,
        )
        diagnostic_correction = np.asarray(
            _load_numeric_array(directory / "arrays" / "correction.npy", ndim=2),
            dtype=np.float64,
        )
        outer_target = np.asarray(
            _load_numeric_array(root / "train_selection_target.npy", ndim=1),
            dtype=np.float64,
        )
        actual_manifest_sha = sha256_path(directory / "manifest.json")
        actual_metadata_sha = sha256_path(directory / "metadata.json")
        actual_summary_sha = sha256_path(directory / "summary.json")
    except (GlobalCampaignError, OSError, ValueError) as exc:
        return [f"cannot bind FM diagnostics to outer cell: {exc}"]

    outer_curve_sha = _array_sha256(selection_curve)
    expected_fm = {
        "status": "completed_strict_v2",
        "path": "fm_diagnostics",
        "manifest_sha256": actual_manifest_sha,
        "metadata_sha256": actual_metadata_sha,
        "summary_sha256": actual_summary_sha,
        "outer_selection_curve_sha256": outer_curve_sha,
    }
    if not _same_json(fm, expected_fm):
        errors.append("FM diagnostic summary attestation does not recompute")
    if metadata.get("checkpoint_sha256") != checkpoint_sha256:
        errors.append("FM diagnostics are not bound to the outer checkpoint")
    if (
        metadata.get("outer_selection_curve_sha256") != outer_curve_sha
        or fm.get("outer_selection_curve_sha256") != outer_curve_sha
    ):
        errors.append("FM diagnostics are not bound to the outer selection curve")
    training = model.get("training")
    diagnostics = model.get("diagnostics")
    expected_variant = (
        training.get("flow_variant_id") if isinstance(training, Mapping) else None
    )
    if metadata.get("variant_id") != expected_variant:
        errors.append("FM diagnostic variant differs from the outer model")
    if not isinstance(diagnostics, Mapping) or not _same_json(
        metadata.get("config"), diagnostics
    ):
        errors.append("FM diagnostic config differs from the outer model")
    if metadata.get("raw_query_sha256") != partition.get(
        "selection_features_fm_sha256"
    ):
        errors.append("FM diagnostics are not bound to held-out selection rows")
    if not np.array_equal(diagnostic_scales, scales):
        errors.append("FM diagnostic scales differ from outer selection scales")
    if not np.array_equal(diagnostic_target, outer_target):
        errors.append("FM diagnostic target differs from outer selection target")
    curve_error = _affine_outer_curve_consistency_error(
        diagnostic_full=diagnostic_full,
        diagnostic_response=diagnostic_response,
        diagnostic_correction=diagnostic_correction,
        outer_selection_curve=selection_curve,
    )
    if curve_error is not None:
        errors.append(f"FM diagnostics differ from outer selection: {curve_error}")
    return errors


def validate_global_cell(
    directory: Path,
    *,
    expected_identity: Mapping[str, Any] | None = None,
    reference_summary: Mapping[str, Any] | None = None,
    expected_source_evidence: Mapping[str, Any] | None = None,
    allow_transient_prunable_checkpoint: bool = False,
) -> list[str]:
    """Recompute selection and split metrics from the sealed pointwise arrays."""

    root = Path(directory)
    errors: list[str] = []
    try:
        manifest = _load_json(root / "manifest.json")
    except GlobalCampaignError as exc:
        return [str(exc)]
    required = {
        "schema_version",
        "identity",
        "selection_protocol",
        "evaluation_protocol",
        "validation_candidate_count",
        "test_candidate_count",
        "outputs",
    }
    if set(manifest) != required:
        errors.append("cell manifest fields differ from contract")
    if (
        type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != GLOBAL_CELL_MANIFEST_SCHEMA_VERSION
    ):
        errors.append("unsupported cell manifest schema_version")
    identity = manifest.get("identity")
    if not isinstance(identity, Mapping):
        return errors + ["cell identity must be a mapping"]
    raw_evaluation_contract = identity.get("evaluation_contract")
    if not isinstance(raw_evaluation_contract, Mapping):
        checkpoint_retention = CHECKPOINT_RETENTION_RETAIN
        errors.append("cell evaluation contract is invalid")
    else:
        try:
            checkpoint_retention = _checkpoint_retention_policy(raw_evaluation_contract)
        except GlobalCampaignError as exc:
            checkpoint_retention = CHECKPOINT_RETENTION_RETAIN
            errors.append(str(exc))
    if expected_identity is not None and not _same_json(identity, expected_identity):
        errors.append("cell identity differs from the expected campaign cell")
    if manifest.get("evaluation_protocol") != FROZEN_EVALUATION_PROTOCOL:
        errors.append("cell evaluation protocol is not frozen single-scale")
    if (
        type(manifest.get("validation_candidate_count")) is not int
        or manifest.get("validation_candidate_count") != 1
    ):
        errors.append("cell validation candidate count is not one")
    if (
        type(manifest.get("test_candidate_count")) is not int
        or manifest.get("test_candidate_count") != 1
    ):
        errors.append("cell test candidate count is not one")
    recorded = manifest.get("outputs")
    if not isinstance(recorded, dict):
        errors.append("cell outputs must be a mapping")
    else:
        try:
            actual = _output_inventory(
                root,
                excluded_relative_paths=(
                    frozenset({"checkpoint.pt"})
                    if checkpoint_retention
                    == CHECKPOINT_RETENTION_PRUNE_AFTER_EVALUATION
                    else frozenset()
                ),
            )
        except GlobalCampaignError as exc:
            errors.append(str(exc))
        else:
            if actual != recorded:
                errors.append("cell output inventory differs from manifest")
    forbidden = (
        list(root.glob("validation_curve*.npy"))
        + list(root.glob("test_curve*.npy"))
        + list(root.glob("*training_progress*.pt"))
    )
    if forbidden:
        errors.append("retrospective curves or training progress remain in sealed cell")

    try:
        summary = _load_json(root / "summary.json")
        input_record = _load_json(root / "input_record.json")
        resolved_model_raw = yaml.safe_load(
            (root / "resolved_model.yaml").read_text(encoding="utf-8")
        )
        scales = np.ravel(
            np.asarray(
                _load_numeric_array(root / "scales.npy", ndim=1), dtype=np.float64
            )
        )
        curve = np.asarray(
            _load_numeric_array(root / "train_selection_curve.npy", ndim=2),
            dtype=np.float64,
        )
        fit_indices = np.asarray(
            _load_numeric_array(root / "train_fit_indices.npy", ndim=1),
            dtype=np.int64,
        )
        selection_indices = np.asarray(
            _load_numeric_array(root / "train_selection_indices.npy", ndim=1),
            dtype=np.int64,
        )
    except (GlobalCampaignError, OSError, UnicodeError, yaml.YAMLError) as exc:
        return errors + [f"cannot validate core cell artifacts: {exc}"]

    model_identity = identity.get("model")
    cell_identity = identity.get("cell")
    selection_contract = identity.get("selection_contract")
    evaluation_contract = identity.get("evaluation_contract")
    if not isinstance(model_identity, Mapping) or not isinstance(
        model_identity.get("model"), Mapping
    ):
        return errors + ["cell model identity is invalid"]
    if not isinstance(cell_identity, Mapping):
        return errors + ["cell inventory identity is invalid"]
    if not isinstance(selection_contract, Mapping):
        return errors + ["cell selection contract is invalid"]
    if not isinstance(evaluation_contract, Mapping):
        return errors + ["cell evaluation contract is invalid"]
    model = model_identity["model"]
    if not _same_json(resolved_model_raw, model):
        errors.append("resolved model YAML differs from cell identity")
    if not _same_json(scales, model.get("scales")):
        errors.append("scales.npy differs from the resolved model scales")
    recomputed_input_sha = sha256_bytes(canonical_json(input_record).encode("utf-8"))
    if identity.get("input_sha256") != recomputed_input_sha:
        errors.append("input_record.json does not recompute identity input SHA")
    if summary.get("input_sha256") != identity.get("input_sha256"):
        errors.append("summary input SHA differs from cell identity")
    summary_checkpoint_sha = summary.get("checkpoint_sha256")
    if not isinstance(summary_checkpoint_sha, str) or not _SHA256.fullmatch(
        summary_checkpoint_sha
    ):
        errors.append("summary checkpoint SHA is invalid")
    checkpoint_path = root / "checkpoint.pt"
    attestation_path = root / "training_attestation.json"
    attestation: dict[str, Any] | None = None
    if attestation_path.exists():
        try:
            attestation = _load_json(attestation_path)
        except GlobalCampaignError as exc:
            errors.append(f"cannot read training attestation: {exc}")
        else:
            errors.extend(
                _validate_training_attestation(
                    attestation,
                    model=model,
                    checkpoint_retention=checkpoint_retention,
                    checkpoint_sha256=summary_checkpoint_sha,
                )
            )
            if summary.get("training_attestation_sha256") != sha256_path(
                attestation_path
            ):
                errors.append("summary training-attestation SHA differs")
    elif checkpoint_retention == CHECKPOINT_RETENTION_PRUNE_AFTER_EVALUATION:
        errors.append("pruned cell lacks training attestation")

    if checkpoint_retention == CHECKPOINT_RETENTION_RETAIN:
        if not checkpoint_path.is_file():
            errors.append("retained checkpoint is missing")
        elif sha256_path(checkpoint_path) != summary_checkpoint_sha:
            errors.append("checkpoint SHA differs from summary")
        if summary.get("checkpoint_retention", CHECKPOINT_RETENTION_RETAIN) != (
            CHECKPOINT_RETENTION_RETAIN
        ):
            errors.append("summary checkpoint retention differs")
    else:
        if summary.get("checkpoint_retention") != checkpoint_retention:
            errors.append("summary checkpoint retention differs")
        if checkpoint_path.exists():
            if not allow_transient_prunable_checkpoint:
                errors.append("pruned sealed cell retains checkpoint")
            elif (
                not checkpoint_path.is_file()
                or sha256_path(checkpoint_path) != summary_checkpoint_sha
            ):
                errors.append("transient checkpoint differs from summary")
            elif attestation is not None and checkpoint_path.stat().st_size != (
                attestation.get("checkpoint_size_bytes")
            ):
                errors.append("transient checkpoint size differs from attestation")

    if scales.size == 0 or np.any(scales <= 0) or np.unique(scales).size != scales.size:
        errors.append("scales must be finite, positive, and unique")
    if curve.shape != (selection_indices.size, scales.size):
        errors.append("train-selection curve shape is invalid")
    if (
        fit_indices.size == 0
        or selection_indices.size == 0
        or np.unique(fit_indices).size != fit_indices.size
        or np.unique(selection_indices).size != selection_indices.size
        or np.intersect1d(fit_indices, selection_indices).size
    ):
        errors.append("train fit/selection indices are empty, repeated, or overlap")
    partition = summary.get("partition")
    if not isinstance(partition, Mapping):
        errors.append("summary partition is invalid")
    else:
        n_source = partition.get("n_source_train")
        if (
            isinstance(n_source, bool)
            or not isinstance(n_source, int)
            or n_source != fit_indices.size + selection_indices.size
            or not np.array_equal(
                np.sort(np.concatenate((fit_indices, selection_indices))),
                np.arange(fit_indices.size + selection_indices.size),
            )
        ):
            errors.append("saved indices do not exactly partition source train")
        if partition.get("fit_indices_sha256") != _array_sha(fit_indices):
            errors.append("fit-index SHA differs from summary partition")
        if partition.get("selection_indices_sha256") != _array_sha(selection_indices):
            errors.append("selection-index SHA differs from summary partition")
        partition_without_sha = dict(partition)
        declared_partition_sha = partition_without_sha.pop("partition_sha256", None)
        if declared_partition_sha != sha256_bytes(
            canonical_json(partition_without_sha).encode("utf-8")
        ):
            errors.append("partition SHA does not recompute")
    if expected_source_evidence is not None:
        if not _same_json(partition, expected_source_evidence.get("partition")):
            errors.append("saved partition differs from freshly loaded source train")
        if _array_sha(fit_indices) != expected_source_evidence.get(
            "fit_indices_sha256"
        ):
            errors.append("fit indices differ from freshly loaded source train")
        if _array_sha(selection_indices) != expected_source_evidence.get(
            "selection_indices_sha256"
        ):
            errors.append("selection indices differ from freshly loaded source train")

    index = summary.get("selected_index")
    valid_index = (
        not isinstance(index, bool)
        and isinstance(index, int)
        and 0 <= index < scales.size
    )
    if not valid_index:
        errors.append("selected_index is invalid")
    elif summary.get("selected_scale") != float(scales[index]):
        errors.append("selected_scale differs from scales.npy")
    if summary.get("target_policy") != cell_identity.get("target_policy"):
        errors.append("summary target policy differs from inventory identity")
    if summary.get("selection_protocol") != manifest.get("selection_protocol"):
        errors.append("summary/manifest selection protocols differ")
    if summary.get("evaluation_protocol") != FROZEN_EVALUATION_PROTOCOL:
        errors.append("summary evaluation protocol differs from frozen contract")
    for field in ("validation_candidate_count", "test_candidate_count"):
        if type(summary.get(field)) is not int or summary.get(field) != 1:
            errors.append(f"summary {field} is not one")
    if summary.get("selection_uses_validation_targets") is not False:
        errors.append("summary claims validation-target selection")
    if summary.get("selection_uses_test_targets") is not False:
        errors.append("summary claims test-target selection")

    target_policy = cell_identity.get("target_policy")
    recomputed_index: int | None = None
    recomputed_selection: Mapping[str, Any] | None = None
    if target_policy == "known_lid":
        try:
            train_target = np.asarray(
                _load_numeric_array(root / "train_selection_target.npy", ndim=1),
                dtype=np.float64,
            )
            if train_target.shape != (curve.shape[0],):
                raise GlobalCampaignError("train selection target shape differs")
            recomputed_index, recomputed_selection = _select_supervised(
                scales,
                curve,
                train_target,
                prefer=str(model.get("selection_prefer")),
                tolerance=float(selection_contract["tie_tolerance"]),
            )
            if expected_source_evidence is not None and _array_sha(
                train_target
            ) != expected_source_evidence.get("train_selection_target_sha256"):
                errors.append(
                    "train selection target differs from freshly loaded source"
                )
        except (GlobalCampaignError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"cannot recompute supervised train selection: {exc}")
        if summary.get("selection_uses_lid_targets") is not True:
            errors.append("known-LID selection is not marked target-supervised")
        if manifest.get("selection_protocol") != KNOWN_SELECTION_PROTOCOL:
            errors.append("known-LID cell has the wrong selection protocol")
        if summary.get("reference_binding") is not None:
            errors.append("known-LID cell unexpectedly binds a reference cell")
    elif target_policy in {"sample_size", "paired_delta"}:
        if (root / "train_selection_target.npy").exists():
            errors.append("target-free cell stores a train selection LID target")
        if summary.get("selection_uses_lid_targets") is not False:
            errors.append("target-free selection is marked target-supervised")
        if manifest.get("selection_protocol") != UNKNOWN_SELECTION_PROTOCOL:
            errors.append("target-free cell has the wrong selection protocol")
        reference_dataset = cell_identity.get("reference_dataset")
        if reference_dataset in {None, cell_identity.get("dataset")}:
            try:
                recomputed_index, recomputed_selection = select_stable_scale(
                    scales,
                    curve,
                    window=int(selection_contract["stability_window"]),
                    min_valid_fraction=float(
                        selection_contract["stability_min_valid_fraction"]
                    ),
                    prefer=str(model.get("selection_prefer")),
                )
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"cannot recompute target-free train selection: {exc}")
            if summary.get("reference_binding") is not None:
                errors.append("reference cell unexpectedly binds another cell")
        else:
            expected_key = (
                f"{cell_identity.get('suite_id')}/{reference_dataset}/"
                f"{cell_identity.get('representation')}"
            )
            binding = summary.get("reference_binding")
            if not isinstance(binding, Mapping):
                errors.append("dependent target-free cell lacks reference binding")
            else:
                if binding.get("cell_key") != expected_key:
                    errors.append("reference binding has the wrong exact cell key")
                if binding.get("representation") != cell_identity.get("representation"):
                    errors.append("reference binding representation differs")
                if valid_index and binding.get("selected_index") != index:
                    errors.append("reference binding selected index differs")
            if reference_summary is not None:
                recomputed_index = reference_summary.get("selected_index")
                recomputed_selection = {
                    "criterion": "reference_cell_train_stability",
                    "reference_dataset": reference_dataset,
                    "reference_selected_index": recomputed_index,
                    "reference_selected_scale": (
                        float(scales[recomputed_index])
                        if isinstance(recomputed_index, int)
                        and not isinstance(recomputed_index, bool)
                        and 0 <= recomputed_index < scales.size
                        else None
                    ),
                }
                expected_binding = {
                    "cell_key": expected_key,
                    "dataset": reference_dataset,
                    "representation": cell_identity.get("representation"),
                    "selected_index": recomputed_index,
                    "summary_sha256": reference_summary.get("summary_sha256"),
                }
                if not _same_json(binding, expected_binding):
                    errors.append("reference binding differs from sealed reference")
    else:
        errors.append("cell has an unsupported target policy")
    if recomputed_index is not None and index != recomputed_index:
        errors.append("selected index does not recompute from train evidence")
    if recomputed_selection is not None and not _same_json(
        summary.get("selection"), recomputed_selection
    ):
        errors.append("selection diagnostics do not recompute")

    metrics = summary.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != {"validation", "test"}:
        errors.append("summary metrics must contain exact validation/test mappings")
    expected_readouts = set(_model_readouts(model))
    for split_name in ("validation", "test"):
        target_path = root / f"{split_name}_target.npy"
        target: npt.NDArray[np.float64] | None = None
        if target_policy == "known_lid":
            try:
                target = np.asarray(
                    _load_numeric_array(target_path, ndim=1), dtype=np.float64
                )
            except GlobalCampaignError as exc:
                errors.append(f"cannot load {split_name} target: {exc}")
        elif target_path.exists():
            errors.append(f"target-free cell stores {split_name} LID targets")
        prediction_paths = {
            path.name[len(f"{split_name}_prediction__") : -4]: path
            for path in root.glob(f"{split_name}_prediction__*.npy")
        }
        if set(prediction_paths) != expected_readouts:
            errors.append(
                f"{split_name} prediction readouts differ from model contract"
            )
        declared_split = (
            metrics.get(split_name) if isinstance(metrics, Mapping) else None
        )
        if (
            not isinstance(declared_split, Mapping)
            or set(declared_split) != expected_readouts
        ):
            errors.append(f"{split_name} metrics readouts differ from model contract")
            declared_split = {}
        for readout, path in prediction_paths.items():
            try:
                prediction = np.asarray(
                    _load_numeric_array(path, ndim=1), dtype=np.float64
                )
                recomputed_metrics = (
                    prediction_summary(prediction)
                    if target is None
                    else known_lid_metrics(prediction, target)
                )
            except (GlobalCampaignError, ValueError) as exc:
                errors.append(f"cannot recompute {split_name}/{readout} metrics: {exc}")
                continue
            if not _same_json(declared_split.get(readout), recomputed_metrics):
                errors.append(f"{split_name}/{readout} metrics do not recompute")
        if expected_source_evidence is not None:
            target_digest = None if target is None else _array_sha(target)
            if target_digest != expected_source_evidence.get(
                f"{split_name}_target_sha256"
            ):
                errors.append(f"{split_name} target differs from freshly loaded source")
            label_path = root / f"{split_name}_labels.npy"
            if label_path.exists():
                try:
                    saved_labels = _load_numeric_array(label_path, ndim=1)
                except GlobalCampaignError as exc:
                    errors.append(f"cannot load {split_name} labels: {exc}")
                    label_digest = "invalid"
                else:
                    label_digest = _array_sha(saved_labels)
            else:
                label_digest = None
            if label_digest != expected_source_evidence.get(
                f"{split_name}_labels_sha256"
            ):
                errors.append(f"{split_name} labels differ from freshly loaded source")

    family = model.get("family")
    fm = summary.get("fm_diagnostics")
    if family == "independent_affine_flow" and target_policy == "known_lid":
        if not isinstance(fm, Mapping) or fm.get("status") != "completed_strict_v2":
            errors.append("known-LID affine cell lacks strict v2 FM diagnostics")
        elif fm.get("path") != "fm_diagnostics":
            errors.append("FM diagnostic path differs from contract")
        else:
            try:
                from experiments.fm_diagnostics import validate_fm_diagnostics

                diagnostic_errors = validate_fm_diagnostics(root / "fm_diagnostics")
            except Exception as exc:  # noqa: BLE001 - return integrity errors
                errors.append(f"cannot validate FM diagnostics: {type(exc).__name__}")
            else:
                errors.extend(f"FM diagnostics: {error}" for error in diagnostic_errors)
                errors.extend(
                    _validate_affine_diagnostic_binding(
                        root,
                        fm=fm,
                        model=model,
                        checkpoint_sha256=summary_checkpoint_sha,
                        scales=scales,
                        selection_curve=curve,
                        partition=(partition if isinstance(partition, Mapping) else {}),
                    )
                )
    elif family == "independent_affine_flow":
        if not isinstance(fm, Mapping) or fm.get("status") != (
            "not_applicable_no_lid_targets"
        ):
            errors.append("target-free affine FM diagnostic status differs")
    elif not isinstance(fm, Mapping) or fm.get("status") != (
        "not_applicable_non_affine_model"
    ):
        errors.append("non-affine FM diagnostic status differs")
    return errors


def _select_supervised(
    scales: npt.NDArray[np.float64],
    curve: npt.NDArray[np.float64],
    target: npt.NDArray[np.float64],
    *,
    prefer: str,
    tolerance: float,
) -> tuple[int, dict[str, Any]]:
    error = np.mean(np.abs(curve - target[:, None]), axis=0)
    minimum = float(error.min())
    candidates = np.flatnonzero(error <= minimum + tolerance)
    selected = int(candidates[0] if prefer == "smaller" else candidates[-1])
    return selected, {
        "criterion": "mae",
        "candidate_mae": [float(value) for value in error],
        "minimum_mae": minimum,
        "tie_candidates": [int(value) for value in candidates],
        "tie_break": prefer,
    }


def _predict(
    predict_fn: PredictFunction,
    trained: Any,
    query: npt.ArrayLike,
    scale: float,
    *,
    model: Mapping[str, Any],
    seed: int,
    batch_size: int,
    readout: str | None = None,
) -> npt.NDArray[np.float64]:
    values = np.ravel(
        np.asarray(
            predict_fn(
                trained,
                query,
                float(scale),
                family=str(model["family"]),
                readout=str(model["readout"] if readout is None else readout),
                divergence_backend=str(model["derivative_backend"]),
                trace_probes=int(model["trace_probes"]),
                trace_seed=seed,
                batch_size=batch_size,
            ),
            dtype=np.float64,
        )
    )
    expected = int(np.asarray(query).shape[0])
    if values.shape != (expected,) or not np.isfinite(values).all():
        raise GlobalCampaignError(
            f"prediction has shape {values.shape}, expected {(expected,)} finite"
        )
    return np.ascontiguousarray(values)


def _prediction_curve(
    predict_fn: PredictFunction,
    trained: Any,
    query: npt.ArrayLike,
    scales: npt.NDArray[np.float64],
    *,
    model: Mapping[str, Any],
    seed: int,
    batch_size: int,
) -> npt.NDArray[np.float64]:
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
                )
                for scale in scales
            ]
        ),
        dtype=np.float64,
    )


def _training_call(
    train_fn: TrainFunction,
    *,
    family: str,
    partition: HoldoutPartition,
    training: Mapping[str, Any],
    checkpoint: Path,
    progress: Path,
    callback: Callable[[Mapping[str, Any]], None] | None,
) -> Any:
    parameters = inspect.signature(train_fn).parameters
    if "progress_checkpoint_path" not in parameters:
        raise GlobalCampaignError(
            "global trainer must support the progress_checkpoint_path contract"
        )
    return train_fn(
        family,
        partition.fit_features,
        partition.selection_features,
        training,
        checkpoint,
        callback,
        progress_checkpoint_path=progress,
    )


def _canonical_training_config_record(value: Any, *, field: str) -> dict[str, Any]:
    """Materialize one training config through the checkpoint schema.

    Pilot Hydra files intentionally omit family-irrelevant optional settings,
    while ``TrainingConfig.to_dict()`` stores those defaults explicitly as
    ``null`` in checkpoints.  Canonicalizing both representations through the
    same validated dataclass makes only that sparse-vs-materialized difference
    equivalent; unknown fields and invalid values still fail closed.
    """

    from models.training import TrainingConfig

    if callable(getattr(value, "to_dict", None)):
        value = value.to_dict()
    try:
        record = _mapping(value, field=field)
        return TrainingConfig.from_mapping(record).to_dict()
    except (GlobalCampaignError, TypeError, ValueError) as exc:
        raise GlobalCampaignError(f"{field} is not a valid TrainingConfig") from exc


def _require_matching_training_configs(
    trained_config: Any, hydra_training: Mapping[str, Any]
) -> None:
    trained_record = _canonical_training_config_record(
        trained_config, field="trained checkpoint config"
    )
    hydra_record = _canonical_training_config_record(
        hydra_training, field="Hydra model training config"
    )
    if canonical_json(trained_record) != canonical_json(hydra_record):
        raise GlobalCampaignError(
            "trained checkpoint config differs from the Hydra model config"
        )


def _training_history_attestation(trained: Any) -> dict[str, Any]:
    """Preserve portable optimizer history before a sealed checkpoint is pruned."""

    raw_history = getattr(trained, "history", None)
    if raw_history is None:
        return {
            "status": "unavailable_from_trainer",
            "best_epoch": None,
            "best_validation_loss": None,
            "epochs": [],
        }
    if isinstance(raw_history, (str, bytes)) or not isinstance(raw_history, Sequence):
        raise GlobalCampaignError("trained history must be a sequence")
    history: list[dict[str, Any]] = []
    for index, raw_metric in enumerate(raw_history):
        if callable(getattr(raw_metric, "to_dict", None)):
            raw_metric = raw_metric.to_dict()
        metric = _mapping(raw_metric, field=f"trained history[{index}]")
        if set(metric) != {
            "epoch",
            "train_loss",
            "validation_loss",
            "learning_rate",
        }:
            raise GlobalCampaignError("trained history entry fields differ")
        epoch = metric["epoch"]
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
            raise GlobalCampaignError("trained history epoch must be positive")
        numeric: dict[str, float] = {}
        for field in ("train_loss", "validation_loss", "learning_rate"):
            value = metric[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise GlobalCampaignError(f"trained history {field} must be finite")
            numeric[field] = float(value)
        history.append({"epoch": epoch, **numeric})
    if not history or any(
        left["epoch"] >= right["epoch"] for left, right in pairwise(history)
    ):
        raise GlobalCampaignError("trained history epochs must strictly increase")
    best_epoch = getattr(trained, "best_epoch", None)
    best_validation_loss = getattr(trained, "best_validation_loss", None)
    if (
        isinstance(best_epoch, bool)
        or not isinstance(best_epoch, int)
        or best_epoch not in {row["epoch"] for row in history}
        or isinstance(best_validation_loss, bool)
        or not isinstance(best_validation_loss, (int, float))
        or not math.isfinite(float(best_validation_loss))
    ):
        raise GlobalCampaignError("trained best-history fields are invalid")
    matching = next(row for row in history if row["epoch"] == best_epoch)
    if float(best_validation_loss) != matching["validation_loss"]:
        raise GlobalCampaignError("trained best loss differs from its history epoch")
    return {
        "status": "complete",
        "best_epoch": best_epoch,
        "best_validation_loss": float(best_validation_loss),
        "epochs": history,
    }


def _training_attestation(
    trained: Any,
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
    retention_policy: str,
    model: Mapping[str, Any],
) -> dict[str, Any]:
    training_config = _canonical_training_config_record(
        model["training"], field="Hydra model training config"
    )
    attestation = {
        "schema_version": TRAINING_ATTESTATION_SCHEMA_VERSION,
        "model_family": str(model["family"]),
        "training_config_sha256": sha256_bytes(
            canonical_json(training_config).encode("utf-8")
        ),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint_retention": retention_policy,
        "history": _training_history_attestation(trained),
    }
    errors = _validate_training_attestation(
        attestation,
        model=model,
        checkpoint_retention=retention_policy,
        checkpoint_sha256=checkpoint_sha256,
    )
    if errors:
        raise GlobalCampaignError(
            f"refusing to attest inconsistent training evidence: {errors}"
        )
    return attestation


def _model_readouts(model: Mapping[str, Any]) -> tuple[str, ...]:
    if model["family"] == "independent_affine_flow":
        return ("response", "full", "fm_to_score")
    if model["family"] in {"rectified_flow", "schrodinger_bridge"}:
        return ("response", "full")
    return (str(model["readout"]),)


def run_known_affine_diagnostics(
    output_dir: Path,
    *,
    trained: Any,
    partition: HoldoutPartition,
    scales: npt.NDArray[np.float64],
    selection_curve: npt.NDArray[np.float64],
    model: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Materialize and validate the existing exhaustive v2 FM diagnostics."""

    if partition.selection_target is None:
        raise GlobalCampaignError("known-LID FM diagnostics require train targets")
    from experiments.fm_diagnostics import (
        run_fm_diagnostics,
        validate_fm_diagnostics,
    )
    from models.training import predict_affine_primitives

    diagnostic_config = model.get("diagnostics")
    if not isinstance(diagnostic_config, Mapping):
        raise GlobalCampaignError("affine model lacks its Hydra diagnostic config")
    if partition.fit_features.shape[0] < int(
        diagnostic_config["oracle_reference_size"]
    ):
        raise GlobalCampaignError(
            "FM diagnostic oracle reference exceeds optimizer-fit rows"
        )
    outer_sha = _array_sha256(selection_curve)
    if output_dir.exists():
        errors = validate_fm_diagnostics(output_dir)
    else:
        staging = output_dir.with_name(f".{output_dir.name}.incomplete")
        if staging.exists():
            # This is the one campaign-owned, identity-scoped diagnostic staging
            # directory.  A killed SDK/write may leave an arbitrary prefix, so
            # it cannot be trusted or incrementally repaired.
            if staging.is_symlink() or not staging.is_dir():
                raise GlobalCampaignError(
                    "FM diagnostic staging path is not a safe directory"
                )
            shutil.rmtree(staging)
        returned = run_fm_diagnostics(
            staging,
            variant_id=str(model["training"]["flow_variant_id"]),
            outer_selection_curve_sha256=outer_sha,
            trained=trained,
            query=partition.selection_features,
            query_model_space=_features_in_model_space(
                trained, partition.selection_features, label="global train-selection"
            ),
            target=partition.selection_target,
            oracle_reference_model_space=_features_in_model_space(
                trained,
                partition.fit_features,
                label="global optimizer-fit oracle reference",
            ),
            scales=scales,
            config=dict(diagnostic_config),
            primitive_fn=predict_affine_primitives,
        )
        if Path(returned).resolve() != staging.resolve():
            raise GlobalCampaignError("FM diagnostics returned an unexpected path")
        errors = validate_fm_diagnostics(staging)
        if errors:
            raise GlobalCampaignError(f"FM diagnostics failed validation: {errors}")
        os.replace(staging, output_dir)
        errors = validate_fm_diagnostics(output_dir)
    if errors:
        raise GlobalCampaignError(f"FM diagnostics failed validation: {errors}")
    metadata = _load_json(output_dir / "metadata.json")
    if metadata.get("outer_selection_curve_sha256") != outer_sha:
        raise GlobalCampaignError("FM diagnostics differ from the selection curve")
    if metadata.get("checkpoint_sha256") != sha256_path(Path(trained.checkpoint_path)):
        raise GlobalCampaignError("FM diagnostics differ from the trained checkpoint")
    return {
        "status": "completed_strict_v2",
        "path": "fm_diagnostics",
        "manifest_sha256": sha256_path(output_dir / "manifest.json"),
        "metadata_sha256": sha256_path(output_dir / "metadata.json"),
        "summary_sha256": sha256_path(output_dir / "summary.json"),
        "outer_selection_curve_sha256": outer_sha,
    }


def _cell_identity(
    *,
    campaign_id: str,
    campaign_config_sha: str,
    source_sha: str,
    model_plan: ModelPlan,
    cell: CampaignCell,
    cell_data: CellData,
    selection_contract: Mapping[str, Any],
    evaluation_contract: Mapping[str, Any],
) -> dict[str, Any]:
    model_record = {
        "variant_id": model_plan.variant_id,
        "experiment_name": model_plan.experiment_name,
        "model": _plain(model_plan.model),
    }
    return {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "campaign_config_sha256": campaign_config_sha,
        "source_tree_sha256": source_sha,
        "model": model_record,
        "cell": _plain(cell),
        "input_sha256": cell_data.input_sha256,
        "selection_contract": _plain(selection_contract),
        "evaluation_contract": _plain(evaluation_contract),
    }


def _safe_component(value: str) -> str:
    normalized = value.replace("_", "-").replace(".", "-")
    if not _EXPERIMENT_NAME.fullmatch(normalized):
        raise GlobalCampaignError(f"unsafe artifact path component: {value!r}")
    return normalized


def _emit(
    callback: Callable[[str, Mapping[str, Any]], None] | None,
    event: str,
    *,
    model_plan: ModelPlan,
    cell: CampaignCell | None = None,
    **payload: Any,
) -> None:
    if callback is None:
        return
    record: dict[str, Any] = {
        "project": PROJECT_NAME,
        "experiment_name": model_plan.experiment_name,
        "model_variant": model_plan.variant_id,
    }
    if cell is not None:
        record.update(
            suite=cell.suite_id,
            dataset=cell.dataset,
            representation=cell.representation,
        )
    record.update(_plain(payload))
    callback(event, record)


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
    affine_diagnostics_fn: AffineDiagnosticsFunction,
    callback: Callable[[str, Mapping[str, Any]], None] | None,
) -> tuple[Path, dict[str, Any]]:
    checkpoint_retention = _checkpoint_retention_policy(
        config["campaign"]["evaluation"]
    )
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
    cell_id = sha256_bytes(canonical_json(identity).encode("utf-8"))[:20]
    model_dir = campaign_root / "runs" / _safe_component(model_plan.variant_id)
    suite_dir = model_dir / _safe_component(cell.suite_id)
    label = (
        f"{_safe_component(cell.dataset)}__{_safe_component(cell.representation)}"
        f"__{cell_id}"
    )
    final_dir = suite_dir / label
    if final_dir.exists():
        errors = validate_global_cell(
            final_dir,
            expected_identity=identity,
            reference_summary=reference_summary,
        )
        if errors:
            raise GlobalCampaignError(
                f"refusing to reuse invalid global cell {final_dir}: {errors}"
            )
        summary = _load_json(final_dir / "summary.json")
        summary["summary_sha256"] = sha256_path(final_dir / "summary.json")
        _emit(
            callback,
            "cell.reused",
            model_plan=model_plan,
            cell=cell,
            cell_id=cell_id,
            selected_scale=summary["selected_scale"],
        )
        return final_dir, summary

    suite_dir.mkdir(parents=True, exist_ok=True)
    work_dir = suite_dir / f".{label}.incomplete"
    identity_path = work_dir / "identity.json"
    if work_dir.exists():
        if not identity_path.is_file() or canonical_json(_load_json(identity_path)) != (
            canonical_json(identity)
        ):
            raise GlobalCampaignError(
                f"stable incomplete cell has a different identity: {work_dir}"
            )
    else:
        work_dir.mkdir()
        _write_json(identity_path, identity)
        _write_yaml(work_dir / "resolved_model.yaml", model_plan.model)
    _write_json(work_dir / "input_record.json", data.input_record)

    selection_config = config["campaign"]["selection"]
    seed = int(config["seed"])
    partition = partition_source_train(
        data.train,
        data.train_target,
        selection=selection_config,
        seed=seed,
    )
    _save_npy(work_dir / "train_fit_indices.npy", partition.fit_indices)
    _save_npy(work_dir / "train_selection_indices.npy", partition.selection_indices)
    if partition.selection_target is not None:
        _save_npy(work_dir / "train_selection_target.npy", partition.selection_target)

    checkpoint = work_dir / "checkpoint.pt"
    progress = work_dir / str(
        config["campaign"]["resume"]["training_progress_filename"]
    )

    # The post-evaluation manifest is written before pruning.  If the process
    # dies after the atomic unlink but before the directory rename, all science
    # outputs are already immutable and can be sealed without retraining.
    if (
        checkpoint_retention == CHECKPOINT_RETENTION_PRUNE_AFTER_EVALUATION
        and (work_dir / "manifest.json").is_file()
        and not checkpoint.exists()
        and not progress.exists()
    ):
        errors = validate_global_cell(
            work_dir,
            expected_identity=identity,
            reference_summary=reference_summary,
        )
        if errors:
            raise GlobalCampaignError(
                f"pruned incomplete global cell failed recovery: {errors}"
            )
        os.replace(work_dir, final_dir)
        summary = _load_json(final_dir / "summary.json")
        summary["summary_sha256"] = sha256_path(final_dir / "summary.json")
        _emit(
            callback,
            "cell.completed",
            model_plan=model_plan,
            cell=cell,
            cell_id=cell_id,
            selected_scale=summary["selected_scale"],
            metrics=summary["metrics"],
            recovered_after_checkpoint_prune=True,
            shared_filesystem_cell_dir=str(final_dir),
        )
        return final_dir, summary

    def training_log(payload: Mapping[str, Any]) -> None:
        epoch = payload.get("epoch")
        _emit(
            callback,
            "cell.training.epoch",
            model_plan=model_plan,
            cell=cell,
            step=epoch,
            training=dict(payload),
        )

    if not (checkpoint.is_file() and not progress.exists()):
        _emit(
            callback,
            "cell.started",
            model_plan=model_plan,
            cell=cell,
            cell_id=cell_id,
            resumed=progress.is_file(),
        )
        _training_call(
            train_fn,
            family=str(model_plan.model["family"]),
            partition=partition,
            training=model_plan.model["training"],
            checkpoint=checkpoint,
            progress=progress,
            callback=training_log if callback is not None else None,
        )
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise GlobalCampaignError(f"trainer did not write checkpoint: {checkpoint}")
    # Evaluation and the surviving attestation must come from a round-tripped
    # checkpoint, never only from the trainer's in-memory return value.  Once
    # pruning unlinks the file this loaded payload is the last opportunity to
    # prove that optimizer history/config and evaluated weights are identical.
    trained = load_checkpoint_fn(
        checkpoint, device=str(model_plan.model["training"]["device"])
    )
    trained_config = getattr(trained, "config", None)
    if trained_config is None:
        if checkpoint_retention == CHECKPOINT_RETENTION_PRUNE_AFTER_EVALUATION:
            raise GlobalCampaignError(
                "pruned checkpoint round-trip lacks its training config"
            )
    else:
        _require_matching_training_configs(trained_config, model_plan.model["training"])
    declared_checkpoint_sha = getattr(trained, "checkpoint_sha256", None)
    checkpoint_sha = sha256_path(checkpoint)
    if declared_checkpoint_sha is None:
        if checkpoint_retention == CHECKPOINT_RETENTION_PRUNE_AFTER_EVALUATION:
            raise GlobalCampaignError(
                "pruned checkpoint round-trip lacks its checkpoint SHA"
            )
    elif declared_checkpoint_sha != checkpoint_sha:
        raise GlobalCampaignError("trained checkpoint SHA differs from checkpoint file")
    declared_checkpoint_path = getattr(trained, "checkpoint_path", None)
    if checkpoint_retention == CHECKPOINT_RETENTION_PRUNE_AFTER_EVALUATION and (
        declared_checkpoint_path is None
        or Path(declared_checkpoint_path).resolve() != checkpoint.resolve()
    ):
        raise GlobalCampaignError(
            "pruned checkpoint round-trip lacks its exact checkpoint path"
        )
    training_attestation = _training_attestation(
        trained,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha,
        retention_policy=checkpoint_retention,
        model=model_plan.model,
    )
    training_attestation_path = work_dir / "training_attestation.json"
    _write_json(training_attestation_path, training_attestation)

    scales = np.ascontiguousarray(
        np.asarray(model_plan.model["scales"], dtype=np.float64)
    )
    execution = config.get("execution", {})
    evaluation_batch_override = (
        execution.get("evaluation_batch_size_override")
        if isinstance(execution, Mapping)
        else None
    )
    batch_size = int(
        config["campaign"]["evaluation"]["batch_size"]
        if evaluation_batch_override is None
        else evaluation_batch_override
    )
    curve = _prediction_curve(
        predict_fn,
        trained,
        partition.selection_features,
        scales,
        model=model_plan.model,
        seed=seed,
        batch_size=batch_size,
    )
    _save_npy(work_dir / "scales.npy", scales)
    _save_npy(work_dir / "train_selection_curve.npy", curve)

    if cell.target_policy == "known_lid":
        if partition.selection_target is None:
            raise GlobalCampaignError(f"known-LID cell lacks train target: {cell.key}")
        selected_index, selection_diagnostics = _select_supervised(
            scales,
            curve,
            partition.selection_target,
            prefer=str(model_plan.model["selection_prefer"]),
            tolerance=float(selection_config["tie_tolerance"]),
        )
        selection_protocol = KNOWN_SELECTION_PROTOCOL
        reference_binding = None
    elif cell.selection_protocol == "target_free_train_stability":
        if cell.reference_dataset not in {None, cell.dataset}:
            if reference_summary is None:
                raise GlobalCampaignError(
                    f"reference selection is unavailable for {cell.key}"
                )
            selected_index = int(reference_summary["selected_index"])
            if not 0 <= selected_index < scales.size:
                raise GlobalCampaignError("reference selected index is invalid")
            selection_diagnostics = {
                "criterion": "reference_cell_train_stability",
                "reference_dataset": cell.reference_dataset,
                "reference_selected_index": selected_index,
                "reference_selected_scale": float(scales[selected_index]),
            }
            reference_binding = {
                "cell_key": (
                    f"{cell.suite_id}/{cell.reference_dataset}/{cell.representation}"
                ),
                "dataset": cell.reference_dataset,
                "representation": cell.representation,
                "selected_index": selected_index,
                "summary_sha256": str(reference_summary["summary_sha256"]),
            }
        else:
            selected_index, selection_diagnostics = select_stable_scale(
                scales,
                curve,
                window=int(selection_config["stability_window"]),
                min_valid_fraction=float(
                    selection_config["stability_min_valid_fraction"]
                ),
                prefer=str(model_plan.model["selection_prefer"]),
            )
            reference_binding = None
        selection_protocol = UNKNOWN_SELECTION_PROTOCOL
    else:
        raise GlobalCampaignError(
            f"unsupported inventory selection protocol for {cell.key}"
        )
    selected_scale = float(scales[selected_index])

    readout_predictions: dict[str, dict[str, npt.NDArray[np.float64]]] = {}
    readout_metrics: dict[str, dict[str, Any]] = {}
    for split_name, query, target in (
        ("validation", data.validation, data.validation_target),
        ("test", data.test, data.test_target),
    ):
        readout_predictions[split_name] = {}
        readout_metrics[split_name] = {}
        for readout in _model_readouts(model_plan.model):
            prediction = _predict(
                predict_fn,
                trained,
                query,
                selected_scale,
                model=model_plan.model,
                seed=seed,
                batch_size=batch_size,
                readout=readout,
            )
            readout_predictions[split_name][readout] = prediction
            readout_metrics[split_name][readout] = (
                prediction_summary(prediction)
                if target is None
                else known_lid_metrics(prediction, target)
            )
            _save_npy(work_dir / f"{split_name}_prediction__{readout}.npy", prediction)
    if data.validation_target is not None:
        _save_npy(work_dir / "validation_target.npy", data.validation_target)
    if data.test_target is not None:
        _save_npy(work_dir / "test_target.npy", data.test_target)
    if data.validation_labels is not None:
        _save_npy(work_dir / "validation_labels.npy", data.validation_labels)
    if data.test_labels is not None:
        _save_npy(work_dir / "test_labels.npy", data.test_labels)

    if (
        model_plan.model["family"] == "independent_affine_flow"
        and cell.target_policy == "known_lid"
    ):
        fm_diagnostics = dict(
            affine_diagnostics_fn(
                work_dir / "fm_diagnostics",
                trained=trained,
                partition=partition,
                scales=scales,
                selection_curve=curve,
                model=model_plan.model,
            )
        )
    elif model_plan.model["family"] == "independent_affine_flow":
        fm_diagnostics = {
            "status": config["campaign"]["fm_diagnostics"]["unknown_lid_status"]
        }
    else:
        fm_diagnostics = {"status": "not_applicable_non_affine_model"}
    summary = {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "cell_id": cell_id,
        "model_variant": model_plan.variant_id,
        "suite_id": cell.suite_id,
        "dataset": cell.dataset,
        "representation": cell.representation,
        "target_policy": cell.target_policy,
        "selection_protocol": selection_protocol,
        "selection_uses_lid_targets": cell.target_policy == "known_lid",
        "selection_uses_validation_targets": False,
        "selection_uses_test_targets": False,
        "selected_index": selected_index,
        "selected_scale": selected_scale,
        "selection": _plain(selection_diagnostics),
        "reference_binding": reference_binding,
        "partition": partition.record,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_retention": checkpoint_retention,
        "training_attestation_sha256": sha256_path(training_attestation_path),
        "input_sha256": data.input_sha256,
        "evaluation_protocol": FROZEN_EVALUATION_PROTOCOL,
        "validation_candidate_count": 1,
        "test_candidate_count": 1,
        "retrospective_validation_curves_saved": False,
        "retrospective_test_curves_saved": False,
        "metrics": readout_metrics,
        "fm_diagnostics": fm_diagnostics,
    }
    _write_json(work_dir / "summary.json", summary)
    summary["summary_sha256"] = sha256_path(work_dir / "summary.json")
    # Store the self-hash only in the returned/aggregate record.  Embedding a
    # file's hash inside itself would create a circular identity.
    outputs = _output_inventory(
        work_dir,
        excluded_relative_paths=(
            frozenset({"checkpoint.pt"})
            if checkpoint_retention == CHECKPOINT_RETENTION_PRUNE_AFTER_EVALUATION
            else frozenset()
        ),
    )
    manifest = {
        "schema_version": GLOBAL_CELL_MANIFEST_SCHEMA_VERSION,
        "identity": identity,
        "selection_protocol": selection_protocol,
        "evaluation_protocol": FROZEN_EVALUATION_PROTOCOL,
        "validation_candidate_count": 1,
        "test_candidate_count": 1,
        "outputs": outputs,
    }
    _write_json(work_dir / "manifest.json", manifest)
    errors = validate_global_cell(
        work_dir,
        expected_identity=identity,
        reference_summary=reference_summary,
        allow_transient_prunable_checkpoint=(
            checkpoint_retention == CHECKPOINT_RETENTION_PRUNE_AFTER_EVALUATION
        ),
    )
    if errors:
        raise GlobalCampaignError(f"new global cell failed validation: {errors}")
    if checkpoint_retention == CHECKPOINT_RETENTION_PRUNE_AFTER_EVALUATION:
        checkpoint.unlink()
        errors = validate_global_cell(
            work_dir,
            expected_identity=identity,
            reference_summary=reference_summary,
        )
        if errors:
            raise GlobalCampaignError(
                f"pruned global cell failed strict validation: {errors}"
            )
    os.replace(work_dir, final_dir)
    summary["summary_sha256"] = sha256_path(final_dir / "summary.json")
    _emit(
        callback,
        "cell.completed",
        model_plan=model_plan,
        cell=cell,
        cell_id=cell_id,
        selected_scale=selected_scale,
        metrics=readout_metrics,
        shared_filesystem_cell_dir=str(final_dir),
    )
    return final_dir, summary


def _deterministic_experiment_key(campaign_identity: str, variant_id: str) -> str:
    return hashlib.sha256(f"{campaign_identity}\0{variant_id}".encode()).hexdigest()[
        :32
    ]


def _reference_cell(cells: Sequence[CampaignCell], cell: CampaignCell) -> CampaignCell:
    reference_dataset = cell.reference_dataset
    if reference_dataset is None:
        raise GlobalCampaignError(f"aggregate cell has no reference: {cell.key}")
    matches = [
        candidate
        for candidate in cells
        if candidate.suite_id == cell.suite_id
        and candidate.dataset == reference_dataset
        and candidate.representation == cell.representation
    ]
    if len(matches) != 1:
        raise GlobalCampaignError(
            f"aggregate expected one exact reference for {cell.key}, found {len(matches)}"
        )
    return matches[0]


def _readout_vectors(directory: Path, split: str) -> dict[str, npt.NDArray[np.float64]]:
    prefix = f"{split}_prediction__"
    values: dict[str, npt.NDArray[np.float64]] = {}
    for path in sorted(directory.glob(f"{prefix}*.npy")):
        readout = path.name[len(prefix) : -len(".npy")]
        if not readout or readout in values:
            raise GlobalCampaignError(f"invalid aggregate readout artifact {path}")
        values[readout] = np.asarray(
            _load_numeric_array(path, ndim=1), dtype=np.float64
        )
    if not values:
        raise GlobalCampaignError(
            f"aggregate cell {directory} has no {split} predictions"
        )
    return values


def recompute_model_aggregate(
    model_variant: str,
    cells: Sequence[CampaignCell],
    cell_directories: Mapping[str, Path],
) -> dict[str, Any]:
    """Rebuild all model-level tables from sealed pointwise cell artifacts."""

    expected_keys = [cell.key for cell in cells]
    if set(cell_directories) != set(expected_keys) or len(cell_directories) != len(
        expected_keys
    ):
        raise GlobalCampaignError(
            f"model {model_variant} aggregate does not cover the exact inventory"
        )
    first_manifest = _load_json(
        Path(cell_directories[expected_keys[0]]) / "manifest.json"
    )
    try:
        primary_readout = str(first_manifest["identity"]["model"]["model"]["readout"])
    except (KeyError, TypeError) as exc:
        raise GlobalCampaignError(
            "model aggregate lacks primary readout identity"
        ) from exc
    summaries: dict[str, dict[str, Any]] = {}
    predictions: dict[tuple[str, str], dict[str, npt.NDArray[np.float64]]] = {}
    labels: dict[tuple[str, str], npt.NDArray[Any] | None] = {}
    known_lid: list[dict[str, Any]] = []
    for cell in cells:
        directory = Path(cell_directories[cell.key])
        summary = _load_json(directory / "summary.json")
        if summary.get("model_variant") != model_variant:
            raise GlobalCampaignError(f"aggregate model identity differs in {cell.key}")
        summaries[cell.key] = summary
        for split in ("validation", "test"):
            split_predictions = _readout_vectors(directory, split)
            predictions[(cell.key, split)] = split_predictions
            label_path = directory / f"{split}_labels.npy"
            split_labels: npt.NDArray[Any] | None = None
            if label_path.exists():
                split_labels = _load_numeric_array(label_path, ndim=1)
                if not np.issubdtype(split_labels.dtype, np.integer):
                    raise GlobalCampaignError(
                        f"aggregate labels must be integral: {label_path}"
                    )
            labels[(cell.key, split)] = split_labels
            if cell.target_policy != "known_lid":
                continue
            target = np.asarray(
                _load_numeric_array(directory / f"{split}_target.npy", ndim=1),
                dtype=np.float64,
            )
            for readout, prediction in split_predictions.items():
                known_lid.append(
                    {
                        "suite_id": cell.suite_id,
                        "dataset": cell.dataset,
                        "representation": cell.representation,
                        "split": split,
                        "readout": readout,
                        "is_primary_readout": readout == primary_readout,
                        "selected_index": summary["selected_index"],
                        "selected_scale": summary["selected_scale"],
                        "metrics": known_lid_metrics(prediction, target),
                    }
                )

    sample_size: list[dict[str, Any]] = []
    paired_delta: list[dict[str, Any]] = []
    for cell in cells:
        if cell.target_policy not in {"sample_size", "paired_delta"}:
            continue
        if cell.expected_lid_delta is None:
            raise GlobalCampaignError(
                f"aggregate cell {cell.key} lacks expected_lid_delta"
            )
        reference = _reference_cell(cells, cell)
        current_summary = summaries[cell.key]
        reference_summary = summaries[reference.key]
        if current_summary["selected_index"] != reference_summary["selected_index"]:
            raise GlobalCampaignError(
                f"aggregate cell {cell.key} did not reuse frozen reference index"
            )
        if cell.dataset != reference.dataset:
            binding = current_summary.get("reference_binding")
            if not isinstance(binding, Mapping) or binding.get("cell_key") != (
                reference.key
            ):
                raise GlobalCampaignError(
                    f"aggregate cell {cell.key} lacks exact reference binding"
                )
            reference_sha = sha256_path(
                Path(cell_directories[reference.key]) / "summary.json"
            )
            if binding.get("summary_sha256") != reference_sha:
                raise GlobalCampaignError(
                    f"aggregate cell {cell.key} binds the wrong reference summary"
                )
        for split in ("validation", "test"):
            current_predictions = predictions[(cell.key, split)]
            reference_predictions = predictions[(reference.key, split)]
            if set(current_predictions) != set(reference_predictions):
                raise GlobalCampaignError(
                    f"aggregate readouts differ for {cell.key}/{split}"
                )
            for readout in sorted(current_predictions):
                current = current_predictions[readout]
                base = reference_predictions[readout]
                if cell.target_policy == "sample_size":
                    current_distribution = prediction_summary(current)
                    reference_distribution = prediction_summary(base)
                    mean_delta = float(
                        current_distribution["mean"] - reference_distribution["mean"]
                    )
                    median_delta = float(
                        current_distribution["median"]
                        - reference_distribution["median"]
                    )
                    sample_size.append(
                        {
                            "suite_id": cell.suite_id,
                            "comparison_group": cell.comparison_group,
                            "dataset": cell.dataset,
                            "reference_dataset": reference.dataset,
                            "representation": cell.representation,
                            "split": split,
                            "readout": readout,
                            "is_primary_readout": readout == primary_readout,
                            "n_source_train": current_summary["partition"][
                                "n_source_train"
                            ],
                            "frozen_selected_index": current_summary["selected_index"],
                            "frozen_selected_scale": current_summary["selected_scale"],
                            "expected_lid_delta": cell.expected_lid_delta,
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
                if current_labels is None or reference_labels is None:
                    raise GlobalCampaignError(
                        f"paired-delta cell {cell.key}/{split} lacks saved labels"
                    )
                if (
                    current.shape != base.shape
                    or current_labels.shape != reference_labels.shape
                    or current_labels.shape != current.shape
                    or not np.array_equal(current_labels, reference_labels)
                ):
                    raise GlobalCampaignError(
                        f"paired-delta cell {cell.key}/{split} is not row-aligned"
                    )
                paired_delta.append(
                    {
                        "suite_id": cell.suite_id,
                        "comparison_group": cell.comparison_group,
                        "dataset": cell.dataset,
                        "reference_dataset": reference.dataset,
                        "representation": cell.representation,
                        "split": split,
                        "readout": readout,
                        "is_primary_readout": readout == primary_readout,
                        "n": int(current.size),
                        "labels_sha256": _array_sha(current_labels),
                        "frozen_selected_index": current_summary["selected_index"],
                        "frozen_selected_scale": current_summary["selected_scale"],
                        "expected_lid_delta": cell.expected_lid_delta,
                        "metrics": paired_delta_metrics(
                            base,
                            current,
                            expected_delta=cell.expected_lid_delta,
                        ),
                    }
                )
    return {
        "schema_version": 1,
        "model_variant": model_variant,
        "primary_readout": primary_readout,
        "coverage": {
            "cells": len(cells),
            "known_lid_records": len(known_lid),
            "e1_sample_size_records": len(sample_size),
            "e5_paired_delta_records": len(paired_delta),
        },
        "known_lid": known_lid,
        "e1_sample_size_stability": sample_size,
        "e5_paired_delta": paired_delta,
    }


def _campaign_aggregate(
    campaign_identity: str,
    model_aggregates: Sequence[Mapping[str, Any]],
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
        "schema_version": 1,
        "campaign_identity": campaign_identity,
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


def _selected_coordinate_name(model_variant: str) -> str:
    if model_variant in {"rectified_flow", "schrodinger_bridge"}:
        return "time"
    if model_variant in APPROVED_MODEL_VARIANTS[4:]:
        return "lambda"
    return "scale"


def render_unified_results_csv(aggregate: Mapping[str, Any]) -> str:
    """Render the exact flat table consumed after the multi-day campaign."""

    rows: list[dict[str, Any]] = []
    for analysis, source, protocol in (
        ("known_lid", aggregate["known_lid"], KNOWN_SELECTION_PROTOCOL),
        (
            "e1_sample_size_stability",
            aggregate["e1_sample_size_stability"],
            UNKNOWN_SELECTION_PROTOCOL,
        ),
        ("e5_paired_delta", aggregate["e5_paired_delta"], UNKNOWN_SELECTION_PROTOCOL),
    ):
        for record in source:
            variant = str(record["model_variant"])
            selected_index = record.get(
                "selected_index", record.get("frozen_selected_index")
            )
            selected_coordinate = record.get(
                "selected_scale", record.get("frozen_selected_scale")
            )
            metrics = record.get("metrics")
            if not isinstance(metrics, Mapping):
                metrics = record.get("prediction_summary", {})
            reference_metrics = record.get("reference_prediction_summary", {})
            row = {
                "analysis": analysis,
                "model_variant": variant,
                "suite_id": record["suite_id"],
                "dataset": record["dataset"],
                "reference_dataset": record.get("reference_dataset"),
                "representation": record["representation"],
                "split": record["split"],
                "readout": record["readout"],
                "primary_readout": (
                    record["readout"]
                    if record.get("is_primary_readout") is True
                    else next(
                        (
                            candidate["readout"]
                            for candidate in source
                            if candidate["model_variant"] == variant
                            and candidate.get("is_primary_readout") is True
                        ),
                        "",
                    )
                ),
                "is_primary_readout": record.get("is_primary_readout"),
                "selection_protocol": protocol,
                "selected_coordinate_name": _selected_coordinate_name(variant),
                "selected_index": selected_index,
                "selected_coordinate": selected_coordinate,
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
        stream,
        fieldnames=list(_UNIFIED_TABLE_FIELDS),
        extrasaction="raise",
        lineterminator="\n",
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

    primary_readout = str(aggregate["primary_readout"])
    known_test = [
        float(record["metrics"]["mae"])
        for record in aggregate["known_lid"]
        if record["split"] == "test"
        and record.get("is_primary_readout") is True
        and "mae" in record["metrics"]
    ]
    e5_test = [
        float(record["metrics"]["mae"])
        for record in aggregate["e5_paired_delta"]
        if record["split"] == "test"
        and record.get("is_primary_readout") is True
        and "mae" in record["metrics"]
    ]
    e1_test = [
        abs(float(record["mean_delta_error"]))
        for record in aggregate["e1_sample_size_stability"]
        if record["split"] == "test" and record.get("is_primary_readout") is True
    ]
    per_readout: dict[str, dict[str, float | None]] = {}
    readouts = sorted(
        {record["readout"] for record in aggregate["known_lid"]}
        | {record["readout"] for record in aggregate["e5_paired_delta"]}
        | {record["readout"] for record in aggregate["e1_sample_size_stability"]}
    )
    for readout in readouts:
        per_readout[readout] = {
            "known_lid_test_mean_mae": mean(
                [
                    float(record["metrics"]["mae"])
                    for record in aggregate["known_lid"]
                    if record["split"] == "test"
                    and record["readout"] == readout
                    and "mae" in record["metrics"]
                ]
            ),
            "e1_test_mean_absolute_mean_delta_error": mean(
                [
                    abs(float(record["mean_delta_error"]))
                    for record in aggregate["e1_sample_size_stability"]
                    if record["split"] == "test" and record["readout"] == readout
                ]
            ),
            "e5_test_mean_paired_delta_mae": mean(
                [
                    float(record["metrics"]["mae"])
                    for record in aggregate["e5_paired_delta"]
                    if record["split"] == "test"
                    and record["readout"] == readout
                    and "mae" in record["metrics"]
                ]
            ),
        }
    return {
        "coverage": _plain(aggregate["coverage"]),
        "primary_readout": primary_readout,
        "known_lid_test_mean_mae": mean(known_test),
        "e1_test_mean_absolute_mean_delta_error": mean(e1_test),
        "e5_test_mean_paired_delta_mae": mean(e5_test),
        "per_readout": per_readout,
    }


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json(_plain(value)) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise GlobalCampaignError(
                        f"blank line in durable journal {path}:{line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise GlobalCampaignError(
                        f"non-object durable journal record {path}:{line_number}"
                    )
                records.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GlobalCampaignError(f"cannot read durable journal {path}") from exc
    return records


class _DurableModelLogger:
    """Write-ahead telemetry spool; remote SDK failures never stop science."""

    def __init__(
        self,
        model_plan: ModelPlan,
        state_dir: Path,
        handle: ModelLoggerHandle,
    ) -> None:
        component = _safe_component(model_plan.variant_id)
        spool_dir = state_dir / "comet" / "spool"
        self._events_path = spool_dir / f"{component}.events.jsonl"
        self._acks_path = spool_dir / f"{component}.acks.jsonl"
        self._failures_path = spool_dir / f"{component}.failures.jsonl"
        self._campaign_root = state_dir.parent
        self._handle = handle
        self._events = _load_jsonl(self._events_path)
        self._acked = {
            str(record.get("event_sha256")) for record in _load_jsonl(self._acks_path)
        }
        for sequence, record in enumerate(self._events):
            if set(record) != {
                "schema_version",
                "sequence",
                "kind",
                "event",
                "payload",
                "event_sha256",
            }:
                raise GlobalCampaignError("durable Comet event schema differs")
            without_sha = dict(record)
            declared_sha = without_sha.pop("event_sha256")
            if (
                type(record.get("sequence")) is not int
                or record["sequence"] != sequence
                or declared_sha
                != sha256_bytes(canonical_json(without_sha).encode("utf-8"))
            ):
                raise GlobalCampaignError("durable Comet event journal is invalid")
        event_shas = {str(record["event_sha256"]) for record in self._events}
        if not self._acked.issubset(event_shas):
            raise GlobalCampaignError("durable Comet ack references an unknown event")
        self._delivery_blocked = False
        self._close_status = "not_closed"
        self._deliver_pending()

    @property
    def callback(self) -> Callable[[str, Mapping[str, Any]], None]:
        return self.__call__

    @property
    def experiment_key(self) -> str | None:
        return self._handle.experiment_key

    def _failure(
        self, stage: str, exc: BaseException, *, event_sha256: str | None = None
    ) -> None:
        _append_jsonl(
            self._failures_path,
            {
                "schema_version": 1,
                "stage": stage,
                "event_sha256": event_sha256,
                "exception_type": type(exc).__name__,
                "at_utc": datetime.now(UTC).isoformat(),
            },
        )

    def _ack(self, event_sha256: str, *, delivery: str) -> None:
        if event_sha256 in self._acked:
            return
        _append_jsonl(
            self._acks_path,
            {
                "schema_version": 1,
                "event_sha256": event_sha256,
                "delivery": delivery,
                "at_utc": datetime.now(UTC).isoformat(),
            },
        )
        self._acked.add(event_sha256)

    def _deliver_pending(self) -> None:
        if self._delivery_blocked:
            return
        for record in self._events:
            event_sha = str(record["event_sha256"])
            if event_sha in self._acked:
                continue
            if self._handle.connection_status == "disabled":
                self._ack(event_sha, delivery="disabled")
                continue
            kind = record["kind"]
            remote_available = (
                self._handle.callback is not None
                if kind == "event"
                else self._handle.log_asset is not None
            )
            if not remote_available:
                self._delivery_blocked = True
                return
            if kind == "asset":
                relative = Path(str(record["payload"]["path"]))
                asset_path = (self._campaign_root / relative).resolve()
                try:
                    asset_path.relative_to(self._campaign_root.resolve())
                except ValueError as exc:
                    raise GlobalCampaignError(
                        "durable asset escapes campaign root"
                    ) from exc
                if (
                    not asset_path.is_file()
                    or sha256_path(asset_path) != record["payload"]["sha256"]
                ):
                    raise GlobalCampaignError(
                        "durable asset identity changed before upload"
                    )
            try:
                if kind == "event":
                    assert self._handle.callback is not None
                    self._handle.callback(str(record["event"]), record["payload"])
                elif kind == "asset":
                    assert self._handle.log_asset is not None
                    self._handle.log_asset(
                        asset_path, str(record["payload"]["asset_name"])
                    )
                else:
                    raise GlobalCampaignError("durable telemetry kind is invalid")
            except Exception as exc:  # noqa: BLE001 - remote telemetry is best-effort
                self._failure("event_delivery", exc, event_sha256=event_sha)
                self._delivery_blocked = True
                return
            self._ack(event_sha, delivery="comet")

    def __call__(self, event: str, payload: Mapping[str, Any]) -> None:
        if not isinstance(event, str) or not event.strip():
            raise GlobalCampaignError("durable telemetry event must be non-empty")
        if not isinstance(payload, Mapping) or _contains_secret(payload):
            raise GlobalCampaignError(
                "durable telemetry payload is invalid or secret-like"
            )
        without_sha = {
            "schema_version": 1,
            "sequence": len(self._events),
            "kind": "event",
            "event": event,
            "payload": _plain(payload),
        }
        record = {
            **without_sha,
            "event_sha256": sha256_bytes(canonical_json(without_sha).encode("utf-8")),
        }
        _append_jsonl(self._events_path, record)
        self._events.append(record)
        self._deliver_pending()

    def asset(self, path: Path, *, name: str) -> None:
        asset_path = Path(path).resolve()
        try:
            relative = asset_path.relative_to(self._campaign_root.resolve())
        except ValueError as exc:
            raise GlobalCampaignError("telemetry asset escapes campaign root") from exc
        if not asset_path.is_file() or not name or Path(name).name != name:
            raise GlobalCampaignError("telemetry asset path/name is invalid")
        payload = {
            "path": relative.as_posix(),
            "asset_name": name,
            "sha256": sha256_path(asset_path),
        }
        without_sha = {
            "schema_version": 1,
            "sequence": len(self._events),
            "kind": "asset",
            "event": "asset.upload",
            "payload": payload,
        }
        record = {
            **without_sha,
            "event_sha256": sha256_bytes(canonical_json(without_sha).encode("utf-8")),
        }
        _append_jsonl(self._events_path, record)
        self._events.append(record)
        self._deliver_pending()

    def close(self) -> None:
        self._delivery_blocked = False
        self._deliver_pending()
        try:
            self._handle.close()
        except Exception as exc:  # noqa: BLE001 - remote telemetry is best-effort
            self._failure("experiment_close", exc)
            self._close_status = "pending_remote_close"
        else:
            self._close_status = "closed"

    def telemetry_record(self) -> dict[str, Any]:
        pending = len(self._events) - len(self._acked)
        return {
            "schema_version": 1,
            "connection_status": self._handle.connection_status,
            "delivery_status": (
                "delivered" if pending == 0 else "pending_remote_delivery"
            ),
            "close_status": self._close_status,
            "experiment_key": self._handle.experiment_key,
            "events_recorded": len(self._events),
            "events_delivered_or_disabled": len(self._acked),
            "events_pending": pending,
            "events_path": self._events_path.relative_to(
                self._campaign_root
            ).as_posix(),
            "events_sha256": (
                sha256_path(self._events_path) if self._events_path.exists() else None
            ),
        }


def _safe_model_logger(
    factory: LoggerFactory,
    model_plan: ModelPlan,
    campaign_identity: str,
    state_dir: Path,
    logging: Mapping[str, Any],
) -> _DurableModelLogger:
    expected_key = (
        _deterministic_experiment_key(campaign_identity, model_plan.variant_id)
        if logging["backend"] == "comet"
        else None
    )
    try:
        handle = factory(model_plan, campaign_identity, state_dir, logging)
    except GlobalCampaignError:
        raise
    except Exception as exc:
        from experiments.comet_logging import CometConfigurationError

        if isinstance(exc, (CometConfigurationError, ImportError)):
            raise GlobalCampaignError(
                "Comet configuration/dependency preflight failed"
            ) from exc
        _append_jsonl(
            state_dir
            / "comet"
            / "spool"
            / f"{_safe_component(model_plan.variant_id)}.failures.jsonl",
            {
                "schema_version": 1,
                "stage": "experiment_open",
                "event_sha256": None,
                "exception_type": type(exc).__name__,
                "at_utc": datetime.now(UTC).isoformat(),
            },
        )
        handle = ModelLoggerHandle(
            None,
            lambda: None,
            expected_key,
            connection_status="pending_remote_open",
        )
    if logging["backend"] == "comet" and handle.experiment_key != expected_key:
        raise GlobalCampaignError(
            "logger factory returned a non-deterministic Comet key"
        )
    return _DurableModelLogger(model_plan, state_dir, handle)


def open_model_logger(
    model_plan: ModelPlan,
    campaign_identity: str,
    state_dir: Path,
    logging: Mapping[str, Any],
) -> ModelLoggerHandle:
    """Open or resume exactly one deterministic Comet experiment per model."""

    if logging["backend"] == "none":
        return ModelLoggerHandle(None, lambda: None, None, connection_status="disabled")
    from experiments.comet_logging import (
        CometEventLogger,
        require_comet_environment,
    )

    require_comet_environment()
    import comet_ml

    key = _deterministic_experiment_key(campaign_identity, model_plan.variant_id)
    state_path = state_dir / "comet" / f"{_safe_component(model_plan.variant_id)}.json"
    expected_core = {
        "schema_version": 1,
        "campaign_identity": campaign_identity,
        "model_variant": model_plan.variant_id,
        "experiment_name": model_plan.experiment_name,
        "experiment_key": key,
        "project": PROJECT_NAME,
        "workspace": WORKSPACE_NAME,
    }
    if state_path.exists():
        state = _load_json(state_path)
        if set(state) != {*expected_core, "registration_status"} or not _same_json(
            {key: state[key] for key in expected_core}, expected_core
        ):
            raise GlobalCampaignError(f"Comet resume state is invalid: {state_path}")
        status = state.get("registration_status")
        if status not in {"intent", "ready"}:
            raise GlobalCampaignError(
                f"Comet registration state is invalid: {state_path}"
            )
        if status == "ready":
            experiment = comet_ml.ExistingExperiment(
                experiment_key=key,
                project_name=PROJECT_NAME,
                workspace=WORKSPACE_NAME,
            )
        else:
            try:
                experiment = comet_ml.ExistingExperiment(
                    experiment_key=key,
                    project_name=PROJECT_NAME,
                    workspace=WORKSPACE_NAME,
                )
            except Exception:  # noqa: BLE001 - retry alternate SDK attach/create
                experiment = comet_ml.Experiment(
                    experiment_key=key,
                    project_name=PROJECT_NAME,
                    workspace=WORKSPACE_NAME,
                )
    else:
        _write_json(state_path, {**expected_core, "registration_status": "intent"})
        experiment = comet_ml.Experiment(
            experiment_key=key,
            project_name=PROJECT_NAME,
            workspace=WORKSPACE_NAME,
        )
        if experiment.get_key() != key:
            experiment.end()
            raise GlobalCampaignError(
                "Comet did not honor deterministic experiment key"
            )
    state = _load_json(state_path)
    if state["registration_status"] == "intent":
        if experiment.get_key() != key:
            try:
                experiment.end()
            finally:
                raise GlobalCampaignError(
                    "Comet did not honor deterministic experiment key"
                )
        experiment.set_name(model_plan.experiment_name)
        experiment.add_tag(model_plan.experiment_name)
        experiment.add_tag(model_plan.variant_id)
        experiment.add_tag("global-e1-e8-canonical-plus-generated-e3-e4")
        _write_json(state_path, {**expected_core, "registration_status": "ready"})
    logger = CometEventLogger(experiment, experiment_name=model_plan.experiment_name)
    return ModelLoggerHandle(
        logger,
        logger.end,
        key,
        connection_status="online_or_resumed",
        log_asset=lambda path, name: logger.log_asset(path, name=name),
    )


@contextmanager
def _exclusive_campaign_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise GlobalCampaignError(
                "another global campaign process holds the lock"
            ) from exc
        stream.seek(0)
        stream.truncate()
        stream.write(f"pid={os.getpid()}\n")
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _clear_accelerator_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def validate_global_campaign(
    campaign_root: Path,
    *,
    expected_campaign_identity: str | None = None,
    project_root: Path | None = None,
    verify_inputs: bool = True,
    source_preflight_fn: SourcePreflight | None = None,
    cell_loader: CellLoader | None = None,
) -> list[str]:
    root = Path(campaign_root).resolve()
    checkout = (
        repository_root() if project_root is None else Path(project_root).resolve()
    )
    errors: list[str] = []
    fresh_source_evidence: dict[str, Mapping[str, Any]] = {}
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
        "input_inventory_path",
        "input_inventory_file_sha256",
        "inventory_cells",
        "approved_model_variants",
        "model_contracts",
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
    }
    if set(manifest) != required:
        errors.append("final campaign manifest fields differ from contract")
    if (
        type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != GLOBAL_FINAL_MANIFEST_SCHEMA_VERSION
    ):
        errors.append("unsupported final campaign schema_version")
    campaign_identity = manifest.get("campaign_identity")
    if (
        expected_campaign_identity is not None
        and campaign_identity != expected_campaign_identity
    ):
        errors.append("final campaign identity differs from current campaign")
    if manifest.get("complete") is not True:
        errors.append("final campaign is not marked complete")
    if manifest.get("expected_models") != len(APPROVED_MODEL_VARIANTS):
        errors.append("final campaign expected_models is not hard-pinned to ten")
    if manifest.get("expected_cells_per_model") != EXPECTED_GLOBAL_CELL_COUNT:
        errors.append("final campaign is not hard-pinned to 39 cells per model")
    if tuple(manifest.get("approved_model_variants") or ()) != (
        APPROVED_MODEL_VARIANTS
    ):
        errors.append("final campaign model allowlist differs from the approved list")

    try:
        resolved_raw = yaml.safe_load(
            (root / "resolved_config.yaml").read_text(encoding="utf-8")
        )
        resolved_config = validate_global_campaign_config(resolved_raw)
    except (OSError, UnicodeError, yaml.YAMLError, GlobalCampaignError) as exc:
        errors.append(f"cannot validate resolved campaign config: {exc}")
        resolved_config = None
    if resolved_config is not None:
        recomputed_config_sha = sha256_bytes(
            canonical_json(resolved_config).encode("utf-8")
        )
        if manifest.get("config_sha256") != recomputed_config_sha:
            errors.append("resolved campaign config SHA does not recompute")
        if manifest.get("campaign_id") != resolved_config["campaign"]["campaign_id"]:
            errors.append("campaign id differs from resolved Hydra config")
        try:
            current_source_sha = hash_declared_sources(checkout)
        except Exception as exc:  # noqa: BLE001 - standalone validator returns errors
            errors.append(f"cannot hash current source tree: {type(exc).__name__}")
        else:
            if manifest.get("source_tree_sha256") != current_source_sha:
                errors.append("campaign source tree differs from current checkout")

    inventory_rows = manifest.get("inventory_cells")
    inventory_cells: list[CampaignCell] = []
    expected_input_inventory: list[dict[str, Any]] = []
    if not isinstance(inventory_rows, list) or len(inventory_rows) != (
        EXPECTED_GLOBAL_CELL_COUNT
    ):
        errors.append("final campaign inventory must contain exactly 39 cells")
    else:
        for ordinal, row in enumerate(inventory_rows):
            if not isinstance(row, Mapping):
                errors.append(f"inventory row {ordinal} is not a mapping")
                continue
            cell_record = {
                key: value
                for key, value in row.items()
                if key not in {"input_sha256", "input_record"}
            }
            try:
                cell = CampaignCell(**cell_record)
            except (TypeError, ValueError) as exc:
                errors.append(f"inventory row {ordinal} is invalid: {exc}")
                continue
            input_record = row.get("input_record")
            input_sha = row.get("input_sha256")
            if not isinstance(input_record, Mapping) or input_sha != sha256_bytes(
                canonical_json(_plain(input_record)).encode("utf-8")
            ):
                errors.append(f"inventory row {ordinal} input identity is invalid")
            inventory_cells.append(cell)
            expected_input_inventory.append(
                {
                    "cell": _plain(cell),
                    "input_sha256": input_sha,
                    "input_record": _plain(input_record),
                }
            )
        keys = [cell.key for cell in inventory_cells]
        if len(keys) != len(set(keys)):
            errors.append("final campaign inventory repeats cells")
        if tuple(keys) != APPROVED_GLOBAL_CELL_KEYS:
            errors.append(
                "final campaign inventory differs from approved 39-cell order"
            )
        if {cell.suite_id for cell in inventory_cells} != set(REQUIRED_SUITE_IDS):
            errors.append("final campaign inventory does not cover E1 through E8")
        if len(inventory_cells) != EXPECTED_GLOBAL_CELL_COUNT:
            errors.append("final campaign has malformed inventory rows")
        if resolved_config is not None:
            try:
                approved_cells = load_campaign_inventory(resolved_config, checkout)
            except GlobalCampaignError as exc:
                errors.append(f"cannot reload approved inventory: {exc}")
            else:
                if not _same_json(inventory_cells, approved_cells):
                    errors.append(
                        "final campaign inventory identities differ from pinned YAMLs"
                    )
                if verify_inputs:
                    selected_source_preflight = (
                        validate_campaign_sources
                        if source_preflight_fn is None
                        else source_preflight_fn
                    )
                    selected_cell_loader = (
                        load_campaign_cell_data if cell_loader is None else cell_loader
                    )
                    try:
                        fresh_source_records = dict(
                            selected_source_preflight(
                                resolved_config, checkout, approved_cells
                            )
                        )
                    except Exception as exc:  # noqa: BLE001 - return audit errors
                        errors.append(
                            "cannot revalidate campaign source seals: "
                            f"{type(exc).__name__}"
                        )
                    else:
                        for ordinal, cell in enumerate(approved_cells):
                            try:
                                fresh_data = _bind_source_preflight(
                                    selected_cell_loader(
                                        cell, resolved_config, checkout
                                    ),
                                    cell,
                                    fresh_source_records,
                                )
                            except Exception as exc:  # noqa: BLE001 - audit all cells
                                errors.append(
                                    f"cannot rehash input {cell.key}: "
                                    f"{type(exc).__name__}"
                                )
                                continue
                            persisted = inventory_rows[ordinal]
                            if fresh_data.input_sha256 != persisted.get(
                                "input_sha256"
                            ) or not _same_json(
                                fresh_data.input_record,
                                persisted.get("input_record"),
                            ):
                                errors.append(
                                    f"fresh input identity differs for {cell.key}"
                                )
                            try:
                                fresh_source_evidence[cell.key] = _source_evidence(
                                    fresh_data, resolved_config
                                )
                            except (GlobalCampaignError, ValueError) as exc:
                                errors.append(
                                    f"cannot derive fresh source evidence for "
                                    f"{cell.key}: {exc}"
                                )
    recomputed_input_sha = sha256_bytes(
        canonical_json(expected_input_inventory).encode("utf-8")
    )
    if manifest.get("input_inventory_sha256") != recomputed_input_sha:
        errors.append("input inventory SHA does not recompute")
    if manifest.get("input_inventory_path") != "input_inventory.json":
        errors.append("input inventory path differs from contract")
    input_inventory_path = root / "input_inventory.json"
    try:
        persisted_input_inventory = json.loads(
            input_inventory_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read persisted input inventory: {exc}")
    else:
        if not _same_json(persisted_input_inventory, expected_input_inventory):
            errors.append("persisted input inventory differs from final manifest")
        if sha256_path(input_inventory_path) != manifest.get(
            "input_inventory_file_sha256"
        ):
            errors.append("persisted input inventory file SHA differs")
    identity_record = {
        "schema_version": 1,
        "campaign_id": manifest.get("campaign_id"),
        "config_sha256": manifest.get("config_sha256"),
        "source_tree_sha256": manifest.get("source_tree_sha256"),
        "input_inventory_sha256": manifest.get("input_inventory_sha256"),
    }
    if campaign_identity != sha256_bytes(
        canonical_json(identity_record).encode("utf-8")
    ):
        errors.append("campaign identity does not recompute")

    contracts = manifest.get("model_contracts")
    if (
        not isinstance(contracts, list)
        or tuple(
            contract.get("variant_id")
            for contract in contracts
            if isinstance(contract, Mapping)
        )
        != APPROVED_MODEL_VARIANTS
        or len(contracts) != len(APPROVED_MODEL_VARIANTS)
    ):
        errors.append("model contracts differ from exact approved order")
        model_contracts: dict[str, Mapping[str, Any]] = {}
    else:
        model_contracts = {
            str(contract["variant_id"]): contract for contract in contracts
        }
        if resolved_config is not None:
            expected_contracts = [
                {
                    "variant_id": plan.variant_id,
                    "experiment_name": plan.experiment_name,
                    "model": _plain(plan.model),
                }
                for plan in model_plans(resolved_config)
            ]
            if not _same_json(contracts, expected_contracts):
                errors.append("model contracts differ from resolved Hydra config")

    cell_records = manifest.get("cells")
    record_by_pair: dict[tuple[str, str], Mapping[str, Any]] = {}
    if not isinstance(cell_records, list) or len(cell_records) != (
        len(APPROVED_MODEL_VARIANTS) * EXPECTED_GLOBAL_CELL_COUNT
    ):
        errors.append("final campaign must contain exactly 390 cell records")
    else:
        for record in cell_records:
            if not isinstance(record, Mapping):
                errors.append("final campaign contains a non-mapping cell record")
                continue
            if set(record) != {
                "model_variant",
                "cell_key",
                "ordinal",
                "path",
                "manifest_sha256",
                "summary_sha256",
            }:
                errors.append("final campaign cell record fields differ from contract")
            pair = (str(record.get("model_variant")), str(record.get("cell_key")))
            if pair in record_by_pair:
                errors.append("final campaign repeats a model/cell record")
            record_by_pair[pair] = record

    recomputed_model_aggregates: list[dict[str, Any]] = []
    model_rows = manifest.get("models")
    if (
        not isinstance(model_rows, list)
        or len(model_rows) != len(APPROVED_MODEL_VARIANTS)
        or tuple(
            row.get("model_variant") for row in model_rows if isinstance(row, Mapping)
        )
        != APPROVED_MODEL_VARIANTS
    ):
        errors.append("model records differ from exact approved coverage")
        model_rows = []
    if len(inventory_cells) == EXPECTED_GLOBAL_CELL_COUNT:
        for model_index, variant in enumerate(APPROVED_MODEL_VARIANTS):
            contract = model_contracts.get(variant)
            if contract is None:
                continue
            summaries: dict[str, dict[str, Any]] = {}
            directories: dict[str, Path] = {}
            for ordinal, cell in enumerate(inventory_cells):
                pair = (variant, cell.key)
                record = record_by_pair.get(pair)
                if record is None:
                    errors.append(f"missing exact cell record {pair}")
                    continue
                if (
                    type(record.get("ordinal")) is not int
                    or record.get("ordinal") != ordinal
                ):
                    errors.append(f"{pair}: cell ordinal differs from inventory")
                row = inventory_rows[ordinal]
                expected_identity = {
                    "schema_version": 1,
                    "campaign_id": manifest.get("campaign_id"),
                    "campaign_config_sha256": manifest.get("config_sha256"),
                    "source_tree_sha256": manifest.get("source_tree_sha256"),
                    "model": contract,
                    "cell": _plain(cell),
                    "input_sha256": row.get("input_sha256"),
                    "selection_contract": (
                        resolved_config["campaign"]["selection"]
                        if resolved_config is not None
                        else {}
                    ),
                    "evaluation_contract": (
                        resolved_config["campaign"]["evaluation"]
                        if resolved_config is not None
                        else {}
                    ),
                }
                cell_id = sha256_bytes(
                    canonical_json(expected_identity).encode("utf-8")
                )[:20]
                expected_relative = (
                    Path("runs")
                    / _safe_component(variant)
                    / _safe_component(cell.suite_id)
                    / (
                        f"{_safe_component(cell.dataset)}__"
                        f"{_safe_component(cell.representation)}__{cell_id}"
                    )
                ).as_posix()
                if record.get("path") != expected_relative:
                    errors.append(f"{pair}: cell path differs from exact identity")
                directory = (root / expected_relative).resolve()
                try:
                    directory.relative_to(root)
                except ValueError:
                    errors.append(f"{pair}: cell path escapes campaign root")
                    continue
                reference_summary = None
                if cell.reference_dataset not in {None, cell.dataset}:
                    reference_key = (
                        f"{cell.suite_id}/{cell.reference_dataset}/"
                        f"{cell.representation}"
                    )
                    reference_summary = summaries.get(reference_key)
                    if reference_summary is None:
                        errors.append(f"{pair}: exact reference summary is unavailable")
                cell_errors = validate_global_cell(
                    directory,
                    expected_identity=expected_identity,
                    reference_summary=reference_summary,
                    expected_source_evidence=fresh_source_evidence.get(cell.key),
                )
                errors.extend(f"{pair}: {error}" for error in cell_errors)
                manifest_path = directory / "manifest.json"
                summary_path = directory / "summary.json"
                if manifest_path.is_file() and sha256_path(manifest_path) != record.get(
                    "manifest_sha256"
                ):
                    errors.append(f"{pair}: manifest SHA differs")
                if summary_path.is_file() and sha256_path(summary_path) != record.get(
                    "summary_sha256"
                ):
                    errors.append(f"{pair}: summary SHA differs")
                try:
                    summary = _load_json(summary_path)
                except GlobalCampaignError:
                    continue
                summary["summary_sha256"] = sha256_path(summary_path)
                summaries[cell.key] = summary
                directories[cell.key] = directory
            if len(directories) != EXPECTED_GLOBAL_CELL_COUNT:
                continue
            try:
                recomputed = recompute_model_aggregate(
                    variant, inventory_cells, directories
                )
            except GlobalCampaignError as exc:
                errors.append(f"{variant}: cannot recompute model aggregate: {exc}")
                continue
            recomputed_model_aggregates.append(recomputed)
            model_row = model_rows[model_index] if model_index < len(model_rows) else {}
            expected_experiment_key = (
                _deterministic_experiment_key(str(campaign_identity), variant)
                if resolved_config is not None
                and resolved_config["logging"]["backend"] == "comet"
                else None
            )
            if (
                type(model_row.get("schema_version")) is not int
                or model_row.get("schema_version") != 1
                or model_row.get("campaign_identity") != campaign_identity
                or model_row.get("model_variant") != variant
                or model_row.get("experiment_name") != contract.get("experiment_name")
                or model_row.get("experiment_key") != expected_experiment_key
                or not _same_json(model_row.get("model_contract"), contract)
                or model_row.get("expected_cells") != EXPECTED_GLOBAL_CELL_COUNT
                or model_row.get("complete_cells") != EXPECTED_GLOBAL_CELL_COUNT
                or model_row.get("complete") is not True
            ):
                errors.append(f"{variant}: model manifest core differs from contract")
            telemetry = model_row.get("comet_telemetry")
            telemetry_fields = {
                "schema_version",
                "connection_status",
                "delivery_status",
                "close_status",
                "experiment_key",
                "events_recorded",
                "events_delivered_or_disabled",
                "events_pending",
                "events_path",
                "events_sha256",
            }
            if not isinstance(telemetry, Mapping) or set(telemetry) != telemetry_fields:
                errors.append(f"{variant}: Comet telemetry record is invalid")
            else:
                recorded_events = telemetry.get("events_recorded")
                delivered_events = telemetry.get("events_delivered_or_disabled")
                pending_events = telemetry.get("events_pending")
                expected_events_path = (
                    Path("state")
                    / "comet"
                    / "spool"
                    / f"{_safe_component(variant)}.events.jsonl"
                ).as_posix()
                if (
                    type(telemetry.get("schema_version")) is not int
                    or telemetry.get("schema_version") != 1
                    or telemetry.get("experiment_key") != expected_experiment_key
                    or type(recorded_events) is not int
                    or type(delivered_events) is not int
                    or type(pending_events) is not int
                    or min(recorded_events, delivered_events, pending_events) < 0
                    or delivered_events + pending_events != recorded_events
                    or telemetry.get("events_path") != expected_events_path
                ):
                    errors.append(f"{variant}: Comet telemetry counts/key differ")
                event_path = root / expected_events_path
                if not event_path.is_file() or sha256_path(event_path) != telemetry.get(
                    "events_sha256"
                ):
                    errors.append(f"{variant}: Comet event spool SHA differs")
            expected_model_path = (
                Path("models") / _safe_component(variant) / "manifest.json"
            ).as_posix()
            expected_aggregate_path = (
                Path("models") / _safe_component(variant) / "aggregate.json"
            ).as_posix()
            if model_row.get("manifest_path") != expected_model_path:
                errors.append(f"{variant}: model manifest path differs")
            if model_row.get("aggregate_path") != expected_aggregate_path:
                errors.append(f"{variant}: model aggregate path differs")
            model_manifest_path = root / expected_model_path
            model_aggregate_path = root / expected_aggregate_path
            try:
                stored_model_manifest = _load_json(model_manifest_path)
                stored_model_aggregate = _load_json(model_aggregate_path)
            except GlobalCampaignError as exc:
                errors.append(f"{variant}: cannot load model outputs: {exc}")
                continue
            row_without_manifest = {
                key: value
                for key, value in model_row.items()
                if key not in {"manifest_path", "manifest_sha256"}
            }
            if not _same_json(stored_model_manifest, row_without_manifest):
                errors.append(f"{variant}: model row differs from model manifest")
            if not _same_json(stored_model_aggregate, recomputed):
                errors.append(f"{variant}: model aggregate does not recompute")
            if sha256_path(model_manifest_path) != model_row.get("manifest_sha256"):
                errors.append(f"{variant}: model manifest SHA differs")
            if sha256_path(model_aggregate_path) != model_row.get("aggregate_sha256"):
                errors.append(f"{variant}: model aggregate SHA differs")

    expected_pairs = {
        (variant, cell.key)
        for variant in APPROVED_MODEL_VARIANTS
        for cell in inventory_cells
    }
    if set(record_by_pair) != expected_pairs:
        errors.append("final campaign does not cover the exact 10x39 product")
    if len(recomputed_model_aggregates) == len(APPROVED_MODEL_VARIANTS):
        recomputed_campaign_aggregate = _campaign_aggregate(
            str(campaign_identity), recomputed_model_aggregates
        )
        if manifest.get("aggregate_path") != "aggregate.json":
            errors.append("campaign aggregate path differs from contract")
        aggregate_path = root / "aggregate.json"
        try:
            stored_campaign_aggregate = _load_json(aggregate_path)
        except GlobalCampaignError as exc:
            errors.append(f"cannot load campaign aggregate: {exc}")
        else:
            if not _same_json(stored_campaign_aggregate, recomputed_campaign_aggregate):
                errors.append("campaign aggregate does not recompute from raw cells")
            if sha256_path(aggregate_path) != manifest.get("aggregate_sha256"):
                errors.append("campaign aggregate SHA differs")
            expected_csv = render_unified_results_csv(recomputed_campaign_aggregate)
            if manifest.get("unified_results_path") != "unified_results.csv":
                errors.append("unified results path differs from contract")
            unified_path = root / "unified_results.csv"
            try:
                stored_csv = unified_path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                errors.append(f"cannot read unified results CSV: {exc}")
            else:
                if stored_csv != expected_csv:
                    errors.append("unified results CSV does not recompute")
                if sha256_path(unified_path) != manifest.get("unified_results_sha256"):
                    errors.append("unified results CSV SHA differs")
    return errors


def run_global_campaign(
    hydra_config: DictConfig | Mapping[str, Any],
    *,
    root: Path | None = None,
    output_root: Path | None = None,
    inventory_loader: InventoryLoader | None = None,
    source_preflight_fn: SourcePreflight | None = None,
    cell_loader: CellLoader | None = None,
    train_fn: TrainFunction | None = None,
    predict_fn: PredictFunction | None = None,
    load_checkpoint_fn: Callable[..., Any] | None = None,
    logger_factory: LoggerFactory | None = None,
    affine_diagnostics_fn: AffineDiagnosticsFunction | None = None,
) -> Path:
    """Execute all cells sequentially in this process and seal one campaign."""

    config = validate_global_campaign_config(hydra_config)
    if config["execution"]["strategy"] != EXECUTION_STRATEGY_SEQUENTIAL:
        raise GlobalCampaignError(
            "cell_dag_pool campaigns must use experiments.global_parallel"
        )
    project_root = (repository_root() if root is None else Path(root)).resolve()
    configured_output = _safe_path(config["output_root"], field="output_root")
    selected_output = configured_output if output_root is None else Path(output_root)
    if not selected_output.is_absolute():
        selected_output = project_root / selected_output
    selected_output = selected_output.resolve()
    source_sha = hash_declared_sources(project_root)
    config_sha = sha256_bytes(canonical_json(config).encode("utf-8"))
    campaign_id = str(config["campaign"]["campaign_id"])

    inventory_loader = (
        load_campaign_inventory if inventory_loader is None else inventory_loader
    )
    source_preflight_fn = (
        validate_campaign_sources
        if source_preflight_fn is None
        else source_preflight_fn
    )
    cell_loader = load_campaign_cell_data if cell_loader is None else cell_loader
    cells = tuple(inventory_loader(config, project_root))
    if not cells:
        raise GlobalCampaignError("campaign inventory contains no cells")
    if {cell.suite_id for cell in cells} != set(REQUIRED_SUITE_IDS):
        raise GlobalCampaignError("campaign cells do not cover suites e1 through e8")
    expected_keys = [cell.key for cell in cells]
    if tuple(expected_keys) != APPROVED_GLOBAL_CELL_KEYS:
        raise GlobalCampaignError(
            "campaign inventory differs from the exact approved ordered 39 cells"
        )
    source_records = dict(source_preflight_fn(config, project_root, cells))
    if set(source_records) != {cell.inventory_id for cell in cells}:
        raise GlobalCampaignError("source preflight does not cover exact inventories")

    # Bind the campaign root and deterministic Comet keys to every input before
    # opening a logger or starting GPU work.  Data are released after each
    # preflight cell so the 39-cell matrix is never resident in memory at once.
    preflight_inputs: dict[str, dict[str, Any]] = {}
    for cell in cells:
        data = _bind_source_preflight(
            cell_loader(cell, config, project_root), cell, source_records
        )
        if cell.key in preflight_inputs:
            raise GlobalCampaignError(f"preflight repeated campaign cell {cell.key}")
        preflight_inputs[cell.key] = {
            "input_sha256": data.input_sha256,
            "input_record": _plain(data.input_record),
            "source_evidence": _source_evidence(data, config),
        }
        del data
    for cell in cells:
        if cell.target_policy != "paired_delta":
            continue
        reference = _reference_cell(cells, cell)
        current_evidence = preflight_inputs[cell.key]["source_evidence"]
        reference_evidence = preflight_inputs[reference.key]["source_evidence"]
        for split in ("validation", "test"):
            current_label_sha = current_evidence[f"{split}_labels_sha256"]
            reference_label_sha = reference_evidence[f"{split}_labels_sha256"]
            if (
                current_label_sha is None
                or reference_label_sha is None
                or current_label_sha != reference_label_sha
                or current_evidence[f"{split}_n"] != reference_evidence[f"{split}_n"]
            ):
                raise GlobalCampaignError(
                    f"paired-delta source rows are not aligned: {cell.key}/{split}"
                )
    input_inventory = [
        {
            "cell": _plain(cell),
            "input_sha256": preflight_inputs[cell.key]["input_sha256"],
            "input_record": preflight_inputs[cell.key]["input_record"],
        }
        for cell in cells
    ]
    input_inventory_sha = sha256_bytes(canonical_json(input_inventory).encode("utf-8"))
    campaign_identity_record = {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "config_sha256": config_sha,
        "source_tree_sha256": source_sha,
        "input_inventory_sha256": input_inventory_sha,
    }
    campaign_identity = sha256_bytes(
        canonical_json(campaign_identity_record).encode("utf-8")
    )
    campaign_root = selected_output / f"{campaign_id}__{campaign_identity[:20]}"
    campaign_root.mkdir(parents=True, exist_ok=True)
    state_dir = campaign_root / "state"
    state_dir.mkdir(exist_ok=True)
    input_inventory_artifact = [
        {
            "cell": _plain(cell),
            "input_sha256": preflight_inputs[cell.key]["input_sha256"],
            "input_record": preflight_inputs[cell.key]["input_record"],
        }
        for cell in cells
    ]
    input_inventory_path = campaign_root / "input_inventory.json"
    if input_inventory_path.exists():
        try:
            existing_input_inventory = json.loads(
                input_inventory_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise GlobalCampaignError(
                "existing campaign input inventory is unreadable"
            ) from exc
        if not _same_json(existing_input_inventory, input_inventory_artifact):
            raise GlobalCampaignError(
                "existing campaign input inventory differs from current preflight"
            )
    else:
        _write_json(input_inventory_path, input_inventory_artifact)
    plans = model_plans(config)
    logger_factory = open_model_logger if logger_factory is None else logger_factory
    lock_path = state_dir / str(config["campaign"]["resume"]["lock_filename"])
    final_path = campaign_root / "campaign.json"
    if final_path.exists():
        errors = validate_global_campaign(
            campaign_root,
            expected_campaign_identity=campaign_identity,
            project_root=project_root,
            source_preflight_fn=source_preflight_fn,
            cell_loader=cell_loader,
        )
        if errors:
            raise GlobalCampaignError(
                f"existing final global campaign is invalid: {errors}"
            )
        # Scientific artifacts are immutable, but a later invocation is an
        # opportunity to replay telemetry left pending by a transient outage.
        with _exclusive_campaign_lock(lock_path):
            for model_plan in plans:
                durable = _safe_model_logger(
                    logger_factory,
                    model_plan,
                    campaign_identity,
                    state_dir,
                    config["logging"],
                )
                durable.close()
        return campaign_root

    if train_fn is None or predict_fn is None or load_checkpoint_fn is None:
        from models.training import load_checkpoint, predict_lid, train_model

        train_fn = train_model if train_fn is None else train_fn
        predict_fn = predict_lid if predict_fn is None else predict_fn
        load_checkpoint_fn = (
            load_checkpoint if load_checkpoint_fn is None else load_checkpoint_fn
        )
    affine_diagnostics_fn = (
        run_known_affine_diagnostics
        if affine_diagnostics_fn is None
        else affine_diagnostics_fn
    )

    cell_records: list[dict[str, Any]] = []
    model_records: list[dict[str, Any]] = []
    model_aggregates: list[dict[str, Any]] = []
    with _exclusive_campaign_lock(lock_path):
        _write_yaml(campaign_root / "resolved_config.yaml", config)
        for model_plan in plans:
            if hash_declared_sources(project_root) != source_sha:
                raise GlobalCampaignError("source tree changed during global campaign")
            handle = _safe_model_logger(
                logger_factory,
                model_plan,
                campaign_identity,
                state_dir,
                config["logging"],
            )
            per_model: dict[str, dict[str, Any]] = {}
            per_model_directories: dict[str, Path] = {}
            handle_closed = False
            try:
                _emit(
                    handle.callback,
                    "model.started",
                    model_plan=model_plan,
                    campaign_identity=campaign_identity,
                    expected_cells=len(cells),
                )
                for ordinal, cell in enumerate(cells):
                    if (
                        config["campaign"]["resume"][
                            "source_tree_check_before_every_cell"
                        ]
                        and hash_declared_sources(project_root) != source_sha
                    ):
                        raise GlobalCampaignError(
                            "source tree changed between campaign cells"
                        )
                    data = _bind_source_preflight(
                        cell_loader(cell, config, project_root), cell, source_records
                    )
                    if data.input_sha256 != preflight_inputs[cell.key]["input_sha256"]:
                        raise GlobalCampaignError(
                            f"input changed after campaign preflight: {cell.key}"
                        )
                    reference_summary: Mapping[str, Any] | None = None
                    if cell.reference_dataset not in {None, cell.dataset}:
                        reference_key = (
                            f"{cell.suite_id}/{cell.reference_dataset}/"
                            f"{cell.representation}"
                        )
                        if reference_key not in per_model:
                            raise GlobalCampaignError(
                                f"expected one completed reference cell for {cell.key}"
                            )
                        reference_summary = per_model[reference_key]
                    final_dir, summary = _run_cell(
                        campaign_root=campaign_root,
                        campaign_id=campaign_id,
                        campaign_config_sha=config_sha,
                        source_sha=source_sha,
                        config=config,
                        model_plan=model_plan,
                        cell=cell,
                        data=data,
                        reference_summary=reference_summary,
                        train_fn=train_fn,
                        predict_fn=predict_fn,
                        load_checkpoint_fn=load_checkpoint_fn,
                        affine_diagnostics_fn=affine_diagnostics_fn,
                        callback=handle.callback,
                    )
                    per_model[cell.key] = summary
                    per_model_directories[cell.key] = final_dir
                    record = {
                        "model_variant": model_plan.variant_id,
                        "cell_key": cell.key,
                        "ordinal": ordinal,
                        "path": final_dir.relative_to(campaign_root).as_posix(),
                        "manifest_sha256": sha256_path(final_dir / "manifest.json"),
                        "summary_sha256": sha256_path(final_dir / "summary.json"),
                    }
                    cell_records.append(record)
                    _write_json(
                        state_dir / "ledger.json",
                        {
                            "schema_version": 1,
                            "campaign_identity": campaign_identity,
                            "completed_cells": cell_records,
                        },
                    )
                    del data
                    _clear_accelerator_cache()
                model_aggregate = recompute_model_aggregate(
                    model_plan.variant_id,
                    cells,
                    per_model_directories,
                )
                model_aggregate_path = (
                    campaign_root
                    / "models"
                    / _safe_component(model_plan.variant_id)
                    / "aggregate.json"
                )
                _write_json(model_aggregate_path, model_aggregate)
                model_aggregates.append(model_aggregate)
                handle.asset(
                    model_aggregate_path,
                    name=f"{_safe_component(model_plan.variant_id)}-aggregate.json",
                )
                model_manifest_path = (
                    campaign_root
                    / "models"
                    / _safe_component(model_plan.variant_id)
                    / "manifest.json"
                )
                _emit(
                    handle.callback,
                    "model.completed",
                    model_plan=model_plan,
                    complete_cells=len(per_model),
                    model_manifest_path=str(model_manifest_path),
                    aggregate=_aggregate_macros(model_aggregate),
                )
                handle.close()
                handle_closed = True
                model_manifest = {
                    "schema_version": 1,
                    "campaign_identity": campaign_identity,
                    "model_variant": model_plan.variant_id,
                    "experiment_name": model_plan.experiment_name,
                    "experiment_key": handle.experiment_key,
                    "comet_telemetry": handle.telemetry_record(),
                    "model_contract": {
                        "variant_id": model_plan.variant_id,
                        "experiment_name": model_plan.experiment_name,
                        "model": _plain(model_plan.model),
                    },
                    "expected_cells": len(cells),
                    "complete_cells": len(per_model),
                    "aggregate_path": model_aggregate_path.relative_to(
                        campaign_root
                    ).as_posix(),
                    "aggregate_sha256": sha256_path(model_aggregate_path),
                    "complete": len(per_model) == len(cells),
                }
                _write_json(model_manifest_path, model_manifest)
                model_records.append(
                    {
                        **model_manifest,
                        "manifest_path": model_manifest_path.relative_to(
                            campaign_root
                        ).as_posix(),
                        "manifest_sha256": sha256_path(model_manifest_path),
                    }
                )
            finally:
                if not handle_closed:
                    handle.close()
                _clear_accelerator_cache()

        expected_total = len(plans) * len(cells)
        if len(cell_records) != expected_total:
            raise GlobalCampaignError("campaign did not complete its exact matrix")
        campaign_aggregate = _campaign_aggregate(campaign_identity, model_aggregates)
        aggregate_path = campaign_root / "aggregate.json"
        _write_json(aggregate_path, campaign_aggregate)
        unified_table_path = campaign_root / "unified_results.csv"
        _write_text(
            unified_table_path,
            render_unified_results_csv(campaign_aggregate),
        )
        # Upload intents are themselves durable.  Reopening uses the same
        # deterministic experiment keys and does not create additional runs.
        for model_index, model_plan in enumerate(plans):
            aggregate_handle = _safe_model_logger(
                logger_factory,
                model_plan,
                campaign_identity,
                state_dir,
                config["logging"],
            )
            aggregate_handle.asset(aggregate_path, name="global-aggregate.json")
            aggregate_handle.asset(
                unified_table_path, name="global-unified-results.csv"
            )
            aggregate_handle.close()
            model_record = model_records[model_index]
            model_record["comet_telemetry"] = aggregate_handle.telemetry_record()
            model_manifest_path = campaign_root / str(model_record["manifest_path"])
            _write_json(
                model_manifest_path,
                {
                    key: value
                    for key, value in model_record.items()
                    if key not in {"manifest_path", "manifest_sha256"}
                },
            )
            model_record["manifest_sha256"] = sha256_path(model_manifest_path)
        final_manifest = {
            "schema_version": GLOBAL_FINAL_MANIFEST_SCHEMA_VERSION,
            "campaign_identity": campaign_identity,
            "campaign_id": campaign_id,
            "config_sha256": config_sha,
            "source_tree_sha256": source_sha,
            "input_inventory_sha256": input_inventory_sha,
            "inventory_cells": [
                {
                    **_plain(cell),
                    "input_sha256": preflight_inputs[cell.key]["input_sha256"],
                    "input_record": preflight_inputs[cell.key]["input_record"],
                }
                for cell in cells
            ],
            "approved_model_variants": list(APPROVED_MODEL_VARIANTS),
            "model_contracts": [
                {
                    "variant_id": plan.variant_id,
                    "experiment_name": plan.experiment_name,
                    "model": _plain(plan.model),
                }
                for plan in plans
            ],
            "input_inventory_path": "input_inventory.json",
            "input_inventory_file_sha256": sha256_path(
                campaign_root / "input_inventory.json"
            ),
            "aggregate_path": "aggregate.json",
            "aggregate_sha256": sha256_path(aggregate_path),
            "unified_results_path": "unified_results.csv",
            "unified_results_sha256": sha256_path(unified_table_path),
            "created_at_utc": datetime.now(UTC).isoformat(),
            "models": model_records,
            "cells": cell_records,
            "expected_models": len(plans),
            "expected_cells_per_model": len(cells),
            "complete": True,
        }
        _write_json(final_path, final_manifest)
        errors = validate_global_campaign(
            campaign_root,
            expected_campaign_identity=campaign_identity,
            project_root=project_root,
            source_preflight_fn=source_preflight_fn,
            cell_loader=cell_loader,
        )
        if errors:
            final_path.unlink(missing_ok=True)
            raise GlobalCampaignError(
                f"new global campaign failed validation: {errors}"
            )
    return campaign_root


@hydra.main(version_base="1.3", config_path=None, config_name="global_campaign")
def _hydra_main(config: DictConfig) -> None:
    output = run_global_campaign(config)
    print(output)


def main() -> None:
    has_config_dir = any(
        argument == "--config-dir" or argument.startswith("--config-dir=")
        for argument in sys.argv[1:]
    )
    if not has_config_dir:
        sys.argv[1:1] = ["--config-dir", str(_config_dir())]
    _hydra_main()


if __name__ == "__main__":
    main()


__all__ = [
    "APPROVED_MODEL_VARIANTS",
    "CampaignCell",
    "CellData",
    "GlobalCampaignError",
    "HoldoutPartition",
    "ModelLoggerHandle",
    "ModelPlan",
    "adaptive_holdout_size",
    "compose_global_campaign_config",
    "load_campaign_cell_data",
    "load_campaign_inventory",
    "model_plans",
    "partition_source_train",
    "run_global_campaign",
    "validate_global_campaign",
    "validate_global_campaign_config",
    "validate_global_cell",
]
