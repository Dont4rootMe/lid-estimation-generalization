"""Train and evaluate one generative-model family on the three E8 datasets.

One pilot run owns three independently trained models: one for Gaussian4, one
for Spaghetti and one for Sphere4.  A deterministic subset of the upstream
training split is held out from optimizer batches.  After training, its LID
targets select the best prediction scale/time.  That index is frozen before
the benchmark validation and test splits are evaluated, so neither validation
nor test labels can influence model-scale selection.

The public :func:`run_pilot` function accepts injectable training, prediction
and logging callables.  Production uses :mod:`models.training`; tests can use a
small deterministic implementation without weakening the artifact contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import hydra
import numpy as np
import numpy.typing as npt
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from datasets.registry import DatasetRegistry, LoadedSplit, load_dataset, load_registry
from experiments.metrics import known_lid_metrics
from experiments.run_manifest import (
    canonical_json,
    environment_state,
    hash_declared_sources,
    sha256_bytes,
    sha256_path,
)
from utils.provenance import sha256_file

PROJECT_NAME = "lid-generalization"
WORKSPACE_NAME = "dont4rootme"
PILOT_DATASETS = (
    "e8_gaussian4_pca",
    "e8_spaghetti_pca",
    "e8_sphere4_pca",
)
PILOT_MANIFEST_SCHEMA_VERSION = 3
TRAIN_SELECTION_PROTOCOL = "held_out_source_train_supervised_v1"
FROZEN_EVALUATION_PROTOCOL = "single_train_selected_scale_v1"
_FAMILY_FOR_ARTIFACTS = {
    "diffusion": "gaussian_diffusion",
    "gaussian_diffusion": "gaussian_diffusion",
    "rectified_flow": "rectified_flow",
    "independent_affine_flow": "independent_affine_flow",
    "scale_conditioned_nf": "scale_conditioned_normalizing_flow",
    "schrodinger_bridge": "brownian_schrodinger_bridge",
}
_EXPERIMENT_NAME_STEM = {
    "diffusion": ("lid-generalization-e8-suite-diffusion-train-mae-scale-selection"),
    "gaussian_diffusion": (
        "lid-generalization-e8-suite-diffusion-train-mae-scale-selection"
    ),
    "rectified_flow": (
        "lid-generalization-e8-suite-rectified-flow-matching-train-mae-time-selection"
    ),
    "scale_conditioned_nf": (
        "lid-generalization-e8-suite-scale-conditioned-normalizing-flow-"
        "train-mae-scale-selection"
    ),
    "schrodinger_bridge": (
        "lid-generalization-e8-suite-brownian-schrodinger-bridge-"
        "train-mae-time-selection"
    ),
}
_AFFINE_VARIANTS: dict[str, dict[str, str]] = {
    "direct_rectified_flow": {
        "schedule": "rectified_linear",
        "parameterization": "direct_velocity",
        "experiment_stem": (
            "lid-generalization-e8-suite-fm-rectified-direct-velocity-"
            "all-readouts-debug-train-mae-lambda-selection"
        ),
    },
    "posterior_rectified_flow": {
        "schedule": "rectified_linear",
        "parameterization": "posterior_mean",
        "experiment_stem": (
            "lid-generalization-e8-suite-fm-rectified-posterior-mean-"
            "all-readouts-debug-train-mae-lambda-selection"
        ),
    },
    "direct_log_noise_affine_flow": {
        "schedule": "log_noise",
        "parameterization": "direct_velocity",
        "experiment_stem": (
            "lid-generalization-e8-suite-fm-log-noise-direct-velocity-"
            "all-readouts-debug-train-mae-lambda-selection"
        ),
    },
    "posterior_log_noise_affine_flow": {
        "schedule": "log_noise",
        "parameterization": "posterior_mean",
        "experiment_stem": (
            "lid-generalization-e8-suite-fm-log-noise-posterior-mean-"
            "all-readouts-debug-train-mae-lambda-selection"
        ),
    },
    "direct_vp_trigonometric_flow": {
        "schedule": "vp_trigonometric",
        "parameterization": "direct_velocity",
        "experiment_stem": (
            "lid-generalization-e8-suite-fm-vp-trigonometric-direct-velocity-"
            "all-readouts-debug-train-mae-lambda-selection"
        ),
    },
    "posterior_vp_trigonometric_flow": {
        "schedule": "vp_trigonometric",
        "parameterization": "posterior_mean",
        "experiment_stem": (
            "lid-generalization-e8-suite-fm-vp-trigonometric-posterior-mean-"
            "all-readouts-debug-train-mae-lambda-selection"
        ),
    },
}
_AFFINE_READOUTS = ("response", "full", "fm_to_score")
_AFFINE_OUTER_CURVE_ROUNDOFF_UNITS = 64.0
_AFFINE_SCALE_GRID = np.asarray(
    (0.01, 0.0178, 0.0316, 0.0562, 0.1, 0.1778, 0.3162, 0.5623, 1.0),
    dtype=np.float64,
)
_AFFINE_DIAGNOSTIC_FIELDS = {
    "schema_version",
    "enabled",
    "source_split",
    "primary_divergence_backend",
    "probe_kind",
    "trace_probes",
    "trace_seed",
    "exact_subset_size",
    "exact_subset_seed",
    "oracle_reference_size",
    "oracle_reference_seed",
    "oracle_chunk_size",
    "batch_size",
}
_COMMON_TRAINING_FIELDS = {
    "seed",
    "device",
    "epochs",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "hidden_dim",
    "time_embedding_dim",
    "validation_interval",
    "early_stopping_patience",
    "gradient_clip_norm",
    "num_workers",
    "deterministic",
    "fourier_features",
    "max_condition_frequency",
    "dropout",
    "normalize",
    "normalization_epsilon",
}
_FAMILY_TRAINING_FIELDS = {
    "diffusion": {"depth", "sigma_min", "sigma_max"},
    "gaussian_diffusion": {"depth", "sigma_min", "sigma_max"},
    "rectified_flow": {"depth", "time_min", "time_max"},
    "independent_affine_flow": {
        "depth",
        "flow_variant_id",
        "flow_schedule",
        "flow_parameterization",
        "flow_conditioning",
        "flow_scale_sampling",
        "flow_loss_weighting",
        "flow_noise_ratio_min",
        "flow_noise_ratio_max",
    },
    # NF integration owns these exact likelihood-path and architecture fields.
    "scale_conditioned_nf": {
        "num_coupling_layers",
        "conditioner_depth",
        "log_scale_limit",
        "epsilon_min",
        "epsilon_max",
    },
    "schrodinger_bridge": {
        "depth",
        "bridge_construction",
        "bridge_reference_process",
        "bridge_initial_marginal",
        "bridge_terminal_marginal",
        "bridge_factor_f",
        "bridge_factor_g",
        "bridge_conditioning",
        "bridge_diffusivity",
        "bridge_terminal_time",
        "bridge_tau_min",
        "bridge_tau_max",
    },
}

FloatArray = npt.NDArray[np.float64]
LogCallback = Callable[[str, Mapping[str, Any]], None]
PrimitiveFunction = Callable[..., Any]
DiagnosticsFunction = Callable[..., Path]
DiagnosticsValidator = Callable[[Path], list[str]]


@dataclass(frozen=True)
class TrainSelectionPartition:
    """Disjoint optimizer/selector views of the canonical source train split."""

    fit_indices: npt.NDArray[np.int64]
    selection_indices: npt.NDArray[np.int64]
    fit_features: npt.NDArray[Any]
    selection_features: npt.NDArray[Any]
    selection_target: FloatArray
    record: Mapping[str, Any]


class TrainFunction(Protocol):
    def __call__(
        self,
        family: str,
        train: npt.ArrayLike,
        validation: npt.ArrayLike,
        config: Mapping[str, Any],
        checkpoint_path: Path,
        log_callback: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> Any: ...


class PredictFunction(Protocol):
    def __call__(
        self,
        model_or_result: Any,
        query: npt.ArrayLike,
        scale: float,
        *,
        family: str | None = None,
        readout: str = "full",
        divergence_backend: str = "hutchinson",
        trace_probes: int = 16,
        trace_seed: int = 0,
        batch_size: int = 128,
    ) -> npt.ArrayLike: ...


class PilotConfigError(ValueError):
    """Raised before training when the standalone Hydra config is unsafe."""


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def compose_pilot_config(
    overrides: Sequence[str] = (), *, root: Path | None = None
) -> DictConfig:
    """Compose the pilot exclusively from Hydra YAML files."""

    if root is None:
        from experiments.cli import _default_config_dir

        config_dir = _default_config_dir()
    else:
        config_dir = Path(root) / "configs"
    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(config_dir.resolve()),
    ):
        config = compose(config_name="pilot", overrides=list(overrides))
    OmegaConf.set_struct(config, True)
    return config


def _resolved_mapping(config: DictConfig | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(config, DictConfig):
        value = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    else:
        value = dict(config)
    if not isinstance(value, dict):
        raise PilotConfigError("pilot config must resolve to a mapping")
    return value


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], *, field: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise PilotConfigError(f"unknown {field} fields: {sorted(unknown)}")


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PilotConfigError(f"{field} must be a positive integer")
    return value


def _safe_output_root(value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise PilotConfigError("output_root must be a non-empty path string")
    path = Path(value)
    if any(part == ".." for part in path.parts):
        raise PilotConfigError("output_root must not contain '..'")
    return path


def _contains_secret_field(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            if any(token in normalized for token in ("api_key", "secret", "token")):
                return True
            if _contains_secret_field(child):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_secret_field(child) for child in value)
    return False


def _expected_experiment_name(model: Mapping[str, Any], *, seed: int) -> str:
    family = str(model.get("family"))
    if family == "independent_affine_flow":
        training = model.get("training")
        if not isinstance(training, Mapping):
            raise PilotConfigError("pilot_model.training must be a mapping")
        variant_id = training.get("flow_variant_id")
        try:
            stem = _AFFINE_VARIANTS[str(variant_id)]["experiment_stem"]
        except KeyError as exc:
            raise PilotConfigError(
                "pilot_model.training.flow_variant_id must be one of "
                f"{sorted(_AFFINE_VARIANTS)!r}"
            ) from exc
    else:
        stem = _EXPERIMENT_NAME_STEM[family]
    return f"{stem}-seed-{seed}"


def _validate_affine_diagnostics(
    diagnostics: Any,
    *,
    evaluation: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> None:
    if not isinstance(diagnostics, dict):
        raise PilotConfigError(
            "independent affine flow requires pilot_model.diagnostics mapping"
        )
    _reject_unknown(
        diagnostics,
        _AFFINE_DIAGNOSTIC_FIELDS,
        field="pilot_model.diagnostics",
    )
    missing = _AFFINE_DIAGNOSTIC_FIELDS - set(diagnostics)
    if missing:
        raise PilotConfigError(
            f"missing pilot_model.diagnostics fields: {sorted(missing)}"
        )
    exact_values = {
        "schema_version": 1,
        "enabled": True,
        "source_split": "train_selection",
        "primary_divergence_backend": "hutchinson",
        "probe_kind": "rademacher",
    }
    for field, expected in exact_values.items():
        if diagnostics[field] != expected or (
            field == "schema_version" and isinstance(diagnostics[field], bool)
        ):
            raise PilotConfigError(
                f"pilot_model.diagnostics.{field} must be exactly {expected!r}"
            )
    for field in (
        "trace_probes",
        "exact_subset_size",
        "oracle_reference_size",
        "oracle_chunk_size",
        "batch_size",
    ):
        _positive_int(diagnostics[field], field=f"pilot_model.diagnostics.{field}")
    exact_sizes = {
        "trace_probes": 16,
        "exact_subset_size": 32,
        "oracle_reference_size": 4096,
        "oracle_chunk_size": 1024,
        "batch_size": 128,
    }
    for field, expected in exact_sizes.items():
        if diagnostics[field] != expected:
            raise PilotConfigError(
                f"pilot_model.diagnostics.{field} must be exactly {expected}"
            )
    for field in ("trace_seed", "exact_subset_seed", "oracle_reference_seed"):
        raw = diagnostics[field]
        if isinstance(raw, bool) or not isinstance(raw, int) or not 0 <= raw < 2**64:
            raise PilotConfigError(
                f"pilot_model.diagnostics.{field} must be an integer in [0, 2**64)"
            )
    if diagnostics["trace_probes"] != evaluation["trace_probes"]:
        raise PilotConfigError(
            "diagnostics.trace_probes must equal evaluation.trace_probes"
        )
    if diagnostics["trace_seed"] != evaluation["trace_seed"]:
        raise PilotConfigError(
            "diagnostics.trace_seed must equal evaluation.trace_seed"
        )
    for field in ("exact_subset_seed", "oracle_reference_seed"):
        if diagnostics[field] != evaluation["trace_seed"]:
            raise PilotConfigError(
                f"diagnostics.{field} must equal evaluation.trace_seed"
            )
    if diagnostics["batch_size"] != evaluation["batch_size"]:
        raise PilotConfigError(
            "diagnostics.batch_size must equal evaluation.batch_size"
        )
    if diagnostics["exact_subset_size"] > selection["subset_size"]:
        raise PilotConfigError(
            "diagnostics.exact_subset_size must not exceed selection.subset_size"
        )
    if diagnostics["oracle_chunk_size"] > diagnostics["oracle_reference_size"]:
        raise PilotConfigError(
            "diagnostics.oracle_chunk_size must not exceed oracle_reference_size"
        )


def validate_pilot_config(
    config: DictConfig | Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve and strictly validate the three-dataset pilot contract."""

    value = _resolved_mapping(config)
    _reject_unknown(
        value,
        {
            "schema_version",
            "project",
            "experiment_name",
            "seed",
            "output_root",
            "data",
            "evaluation",
            "logging",
            "pilot_model",
        },
        field="top-level",
    )
    if value.get("schema_version") != 1 or isinstance(
        value.get("schema_version"), bool
    ):
        raise PilotConfigError("pilot schema_version must be 1")
    if value.get("project") != PROJECT_NAME:
        raise PilotConfigError(f"project must be exactly {PROJECT_NAME!r}")
    if (
        not isinstance(value.get("experiment_name"), str)
        or not value["experiment_name"]
    ):
        raise PilotConfigError("experiment_name must be a non-empty string")
    if _contains_secret_field(value):
        raise PilotConfigError(
            "credentials are forbidden in Hydra config; use environment variables"
        )
    seed = value.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise PilotConfigError("seed must be a non-negative integer")
    _safe_output_root(value.get("output_root"))

    data = value.get("data")
    if not isinstance(data, dict):
        raise PilotConfigError("data must be a mapping")
    _reject_unknown(
        data,
        {"root", "registry", "representation", "mmap_mode", "names"},
        field="data",
    )
    if not isinstance(data.get("root"), str) or not data["root"]:
        raise PilotConfigError("data.root must be a non-empty path string")
    registry = data.get("registry")
    if not isinstance(registry, str) or not registry.endswith(".yaml") or not registry:
        raise PilotConfigError("data.registry must name a .yaml file")
    if data.get("representation") != "dataset":
        raise PilotConfigError("pilot representation must be exactly 'dataset'")
    if data.get("mmap_mode") not in {None, "r"}:
        raise PilotConfigError("data.mmap_mode must be null or 'r'")
    if tuple(data.get("names", ())) != PILOT_DATASETS:
        raise PilotConfigError(
            f"pilot datasets must be exactly {list(PILOT_DATASETS)!r}"
        )

    evaluation = value.get("evaluation")
    if not isinstance(evaluation, dict):
        raise PilotConfigError("evaluation must be a mapping")
    _reject_unknown(
        evaluation,
        {
            "batch_size",
            "divergence_backend",
            "trace_probes",
            "trace_seed",
            "selection",
        },
        field="evaluation",
    )
    _positive_int(evaluation.get("batch_size"), field="evaluation.batch_size")
    if evaluation.get("divergence_backend") not in {"exact", "hutchinson"}:
        raise PilotConfigError(
            "evaluation.divergence_backend must be 'exact' or 'hutchinson'"
        )
    trace_probes = evaluation.get("trace_probes")
    if evaluation["divergence_backend"] == "hutchinson":
        _positive_int(trace_probes, field="evaluation.trace_probes")
    elif trace_probes not in {0, None}:
        raise PilotConfigError("exact divergence requires trace_probes: 0")
    trace_seed = evaluation.get("trace_seed")
    if (
        isinstance(trace_seed, bool)
        or not isinstance(trace_seed, int)
        or trace_seed < 0
    ):
        raise PilotConfigError("evaluation.trace_seed must be non-negative")
    selection = evaluation.get("selection")
    if not isinstance(selection, dict):
        raise PilotConfigError("evaluation.selection must be a mapping")
    _reject_unknown(
        selection,
        {
            "source_split",
            "subset_size",
            "seed",
            "criterion",
            "tie_tolerance",
            "tie_break",
        },
        field="evaluation.selection",
    )
    if selection.get("source_split") != "train":
        raise PilotConfigError("selection.source_split must be exactly 'train'")
    _positive_int(selection.get("subset_size"), field="selection.subset_size")
    selection_seed = selection.get("seed")
    if (
        isinstance(selection_seed, bool)
        or not isinstance(selection_seed, int)
        or not 0 <= selection_seed < 2**64
    ):
        raise PilotConfigError("selection.seed must be an integer in [0, 2**64)")
    if selection.get("criterion") != "mae":
        raise PilotConfigError("selection.criterion must be exactly 'mae'")
    tie_tolerance = selection.get("tie_tolerance")
    if (
        isinstance(tie_tolerance, bool)
        or not isinstance(tie_tolerance, (int, float))
        or not math.isfinite(float(tie_tolerance))
        or float(tie_tolerance) < 0.0
    ):
        raise PilotConfigError(
            "selection.tie_tolerance must be a finite non-negative number"
        )
    if selection.get("tie_break") not in {"smaller", "larger"}:
        raise PilotConfigError("selection.tie_break must be 'smaller' or 'larger'")
    model = value.get("pilot_model")
    if not isinstance(model, dict):
        raise PilotConfigError("pilot_model must be a mapping")
    _reject_unknown(
        model,
        {
            "name",
            "family",
            "experiment_name",
            "readout",
            "selection_prefer",
            "derivative_backend",
            "trace_probes",
            "training",
            "scales",
            "diagnostics",
        },
        field="pilot_model",
    )
    family = model.get("family")
    if family not in _FAMILY_FOR_ARTIFACTS:
        raise PilotConfigError(
            "pilot_model.family must be diffusion/gaussian_diffusion, "
            "rectified_flow, independent_affine_flow, scale_conditioned_nf, "
            "or schrodinger_bridge"
        )
    if not isinstance(model.get("name"), str) or not model["name"]:
        raise PilotConfigError("pilot_model.name must be non-empty")
    expected_experiment_name = _expected_experiment_name(model, seed=seed)
    if model.get("experiment_name") != expected_experiment_name:
        raise PilotConfigError(
            "pilot_model.experiment_name must be exactly "
            f"{expected_experiment_name!r} for family {family!r}"
        )
    if value["experiment_name"] != model["experiment_name"]:
        raise PilotConfigError("experiment_name must equal pilot_model.experiment_name")
    allowed_readouts = (
        {"fixed_likelihood"}
        if family == "scale_conditioned_nf"
        else (
            set(_AFFINE_READOUTS)
            if family == "independent_affine_flow"
            else {"full", "response"}
        )
    )
    if model.get("readout") not in allowed_readouts:
        raise PilotConfigError(
            f"pilot_model.readout must be one of {sorted(allowed_readouts)!r} "
            f"for family {family!r}"
        )
    if model.get("selection_prefer") not in {"smaller", "larger"}:
        raise PilotConfigError(
            "pilot_model.selection_prefer must be 'smaller' or 'larger'"
        )
    expected_derivative = (
        ("exact", 0) if family == "scale_conditioned_nf" else ("hutchinson", 16)
    )
    model_trace_probes = model.get("trace_probes")
    actual_derivative = (model.get("derivative_backend"), model_trace_probes)
    if isinstance(model_trace_probes, bool) or not isinstance(model_trace_probes, int):
        raise PilotConfigError("pilot_model.trace_probes must be an integer")
    if actual_derivative != expected_derivative:
        raise PilotConfigError(
            "pilot_model derivative contract must be exactly "
            f"backend={expected_derivative[0]!r}, probes={expected_derivative[1]} "
            f"for family {family!r}"
        )
    if (
        evaluation["divergence_backend"] != model["derivative_backend"]
        or evaluation["trace_probes"] != model["trace_probes"]
    ):
        raise PilotConfigError(
            "evaluation derivative settings must resolve from pilot_model"
        )
    if selection["tie_break"] != model["selection_prefer"]:
        raise PilotConfigError(
            "selection.tie_break must equal pilot_model.selection_prefer"
        )
    if not isinstance(model.get("training"), dict) or not model["training"]:
        raise PilotConfigError("pilot_model.training must be a non-empty mapping")
    expected_training_fields = (
        _COMMON_TRAINING_FIELDS | _FAMILY_TRAINING_FIELDS[str(family)]
    )
    _reject_unknown(
        model["training"], expected_training_fields, field="pilot_model.training"
    )
    missing_training_fields = expected_training_fields - set(model["training"])
    if missing_training_fields:
        raise PilotConfigError(
            f"missing pilot_model.training fields: {sorted(missing_training_fields)}"
        )
    diagnostics = model.get("diagnostics")
    if family == "independent_affine_flow":
        if model["readout"] != "full":
            raise PilotConfigError(
                "independent affine flow primary readout must be exactly 'full'"
            )
        if model["selection_prefer"] != "smaller":
            raise PilotConfigError(
                "independent affine flow must prefer smaller lambda on selection ties"
            )
        affine_training = model["training"]
        variant_id = affine_training["flow_variant_id"]
        if model["name"] != variant_id:
            raise PilotConfigError(
                "pilot_model.name must equal training.flow_variant_id for affine flow"
            )
        try:
            expected_variant = _AFFINE_VARIANTS[str(variant_id)]
        except KeyError as exc:
            raise PilotConfigError(
                f"unsupported independent affine-flow variant {variant_id!r}"
            ) from exc
        for field, expected in (
            ("flow_schedule", expected_variant["schedule"]),
            ("flow_parameterization", expected_variant["parameterization"]),
            ("flow_conditioning", "log_noise_ratio"),
            ("flow_scale_sampling", "log_uniform_noise_ratio"),
            ("flow_loss_weighting", "posterior_bias_equivalent"),
        ):
            if affine_training[field] != expected:
                raise PilotConfigError(
                    f"pilot_model.training.{field} must be exactly {expected!r} "
                    f"for {variant_id!r}"
                )
        for field, expected in (
            ("flow_noise_ratio_min", 0.01),
            ("flow_noise_ratio_max", 1.0),
        ):
            raw = affine_training[field]
            if (
                isinstance(raw, bool)
                or not isinstance(raw, (int, float))
                or not math.isfinite(float(raw))
                or float(raw) != expected
            ):
                raise PilotConfigError(
                    f"pilot_model.training.{field} must be exactly {expected}"
                )
        _validate_affine_diagnostics(
            diagnostics,
            evaluation=evaluation,
            selection=selection,
        )
    elif diagnostics is not None:
        raise PilotConfigError(
            "pilot_model.diagnostics is reserved for independent_affine_flow"
        )
    if family == "scale_conditioned_nf":
        nf_training = model["training"]
        if nf_training["dropout"] != 0.0:
            raise PilotConfigError(
                "scale-conditioned NF requires training.dropout to be exactly 0"
            )
        for field in ("num_coupling_layers", "conditioner_depth"):
            raw = nf_training[field]
            if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
                raise PilotConfigError(
                    f"pilot_model.training.{field} must be a positive integer"
                )
        for field in ("log_scale_limit", "epsilon_min", "epsilon_max"):
            raw = nf_training[field]
            if (
                isinstance(raw, bool)
                or not isinstance(raw, (int, float))
                or not math.isfinite(float(raw))
                or float(raw) <= 0.0
            ):
                raise PilotConfigError(
                    f"pilot_model.training.{field} must be finite and positive"
                )
        if not float(nf_training["epsilon_min"]) < float(nf_training["epsilon_max"]):
            raise PilotConfigError(
                "NF epsilon bounds must satisfy epsilon_min < epsilon_max"
            )
    if family == "schrodinger_bridge":
        bridge_training = model["training"]
        exact_bridge_contract = {
            "bridge_construction": "terminal-data-lebesgue-factor-v1",
            "bridge_reference_process": "brownian-motion",
            "bridge_initial_marginal": "gaussian-convolution-of-terminal-data",
            "bridge_terminal_marginal": "dataset-terminal-law",
            "bridge_factor_f": "lebesgue-measure",
            "bridge_factor_g": "dataset-terminal-law",
            "bridge_conditioning": "time-to-go-tau",
        }
        for field, expected in exact_bridge_contract.items():
            if bridge_training.get(field) != expected:
                raise PilotConfigError(
                    f"pilot_model.training.{field} must be exactly {expected!r}"
                )
        diffusivity = bridge_training["bridge_diffusivity"]
        terminal_time = bridge_training["bridge_terminal_time"]
        tau_min = bridge_training["bridge_tau_min"]
        tau_max = bridge_training["bridge_tau_max"]
        for field, raw in (
            ("bridge_diffusivity", diffusivity),
            ("bridge_terminal_time", terminal_time),
            ("bridge_tau_min", tau_min),
            ("bridge_tau_max", tau_max),
        ):
            if (
                isinstance(raw, bool)
                or not isinstance(raw, (int, float))
                or not math.isfinite(float(raw))
                or float(raw) <= 0
            ):
                raise PilotConfigError(
                    f"pilot_model.training.{field} must be finite and positive"
                )
        if not 0 < float(tau_min) < float(tau_max) <= float(terminal_time):
            raise PilotConfigError(
                "bridge tau bounds must satisfy "
                "0 < bridge_tau_min < bridge_tau_max <= bridge_terminal_time"
            )
    scales = np.asarray(model.get("scales"), dtype=np.float64)
    if (
        scales.ndim != 1
        or scales.size < 2
        or not np.isfinite(scales).all()
        or np.any(scales <= 0)
        or np.unique(scales).size != scales.size
    ):
        raise PilotConfigError(
            "pilot_model.scales must contain enough unique finite positive values"
        )
    if family == "rectified_flow" and np.any(scales >= 1):
        raise PilotConfigError("rectified-flow scales must lie strictly in (0, 1)")
    if family == "independent_affine_flow":
        if not np.array_equal(scales, _AFFINE_SCALE_GRID):
            raise PilotConfigError(
                "independent affine-flow scales must be the exact common lambda grid "
                f"{_AFFINE_SCALE_GRID.tolist()!r}"
            )
        minimum = float(model["training"]["flow_noise_ratio_min"])
        maximum = float(model["training"]["flow_noise_ratio_max"])
        if np.any(scales < minimum) or np.any(scales > maximum):
            raise PilotConfigError(
                "independent affine-flow lambda scales must lie inside training bounds"
            )
    if family == "scale_conditioned_nf":
        epsilon_min = float(model["training"]["epsilon_min"])
        epsilon_max = float(model["training"]["epsilon_max"])
        if np.any(scales < epsilon_min) or np.any(scales > epsilon_max):
            raise PilotConfigError(
                "NF epsilon scales must lie inside training epsilon bounds"
            )
    if family == "schrodinger_bridge":
        tau_min = float(model["training"]["bridge_tau_min"])
        tau_max = float(model["training"]["bridge_tau_max"])
        if np.any(scales < tau_min) or np.any(scales > tau_max):
            raise PilotConfigError(
                "Schrodinger-bridge tau scales must lie inside training tau bounds"
            )

    logging = value.get("logging")
    if not isinstance(logging, dict):
        raise PilotConfigError("logging must be a mapping")
    _reject_unknown(
        logging,
        {"backend", "project", "experiment_name", "workspace"},
        field="logging",
    )
    if logging.get("backend") not in {"none", "comet"}:
        raise PilotConfigError("logging.backend must be 'none' or 'comet'")
    if logging.get("project") != PROJECT_NAME:
        raise PilotConfigError(f"logging.project must be exactly {PROJECT_NAME!r}")
    if logging.get("experiment_name") != value["experiment_name"]:
        raise PilotConfigError("logging.experiment_name must equal experiment_name")
    if logging.get("workspace") != WORKSPACE_NAME:
        raise PilotConfigError(f"logging.workspace must be exactly {WORKSPACE_NAME!r}")
    return value


def _resolve_path(root: Path, configured: str) -> Path:
    path = Path(configured)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _flatten_features(split: LoadedSplit) -> np.ndarray:
    values = np.asarray(split.features)
    return values.reshape(values.shape[0], -1)


def _strict_json_value(value: Any) -> Any:
    """Convert trainer metadata without silently accepting NaN/Infinity."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"non-finite value in trainer metadata: {result!r}")
        return result
    if isinstance(value, np.integer):
        return int(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _strict_json_value(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _strict_json_value(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _strict_json_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_value(child) for child in value]
    if hasattr(value, "item"):
        try:
            return _strict_json_value(value.item())
        except (TypeError, ValueError):
            pass
    raise TypeError(f"trainer metadata is not JSON serializable: {type(value)!r}")


def _write_json(path: Path, value: Any) -> None:
    payload = _strict_json_value(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_yaml(path: Path, value: Mapping[str, Any]) -> None:
    payload = _strict_json_value(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _save_npy(path: Path, value: npt.ArrayLike) -> None:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"refusing to save non-numeric array {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    temporary.replace(path)


def _source_file_records(
    splits: Mapping[str, LoadedSplit], *, benchmark_root: Path
) -> dict[str, dict[str, str | int]]:
    records: dict[str, dict[str, str | int]] = {}
    for split_name, split in splits.items():
        for artifact, path in split.source_paths.items():
            resolved = path.resolve()
            try:
                relative = resolved.relative_to(benchmark_root).as_posix()
            except ValueError as exc:
                raise PilotConfigError(
                    f"dataset source resolves outside data.root: {resolved}"
                ) from exc
            records[f"{split_name}/{artifact}.npy"] = {
                "path": relative,
                "size_bytes": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
    return dict(sorted(records.items()))


def _records_sha256(records: Mapping[str, Any]) -> str:
    identity = {
        key: {
            "size_bytes": record["size_bytes"],
            "sha256": record["sha256"],
        }
        for key, record in sorted(records.items())
    }
    return sha256_bytes(canonical_json(identity).encode("utf-8"))


def _applied_override_records(
    splits: Mapping[str, LoadedSplit],
) -> dict[str, dict[str, dict[str, float | str]]]:
    """Return the in-memory corrections applied after raw-value validation."""

    return {
        split_name: {
            artifact: dict(sorted(details.items()))
            for artifact, details in sorted(split.applied_overrides.items())
        }
        for split_name, split in splits.items()
    }


def _load_inputs(
    *, root: Path, config: Mapping[str, Any]
) -> tuple[
    DatasetRegistry, Path, Path, dict[str, Mapping[str, LoadedSplit]], dict[str, Any]
]:
    data = config["data"]
    registry_path = _resolve_path(root, str(data["registry"]))
    benchmark_root = _resolve_path(root, str(data["root"]))
    registry = load_registry(registry_path)
    loaded: dict[str, Mapping[str, LoadedSplit]] = {}
    records: dict[str, Any] = {}
    for name in PILOT_DATASETS:
        try:
            spec = registry[name]
        except KeyError as exc:
            raise PilotConfigError(f"registry has no pilot dataset {name!r}") from exc
        if "lid" not in spec.required_artifacts:
            raise PilotConfigError(
                f"pilot dataset {name!r} must declare train/val/test LID targets"
            )
        splits = load_dataset(
            benchmark_root,
            spec,
            representation="dataset",
            mmap_mode=data.get("mmap_mode"),
        )
        if tuple(splits) != ("train", "val", "test"):
            raise PilotConfigError(
                f"pilot dataset {name!r} must expose train/val/test in that order"
            )
        source_files = _source_file_records(splits, benchmark_root=benchmark_root)
        training_key = "train/dataset.npy"
        if training_key not in source_files:
            raise PilotConfigError(f"{name!r} has no {training_key}")
        records[name] = {
            "representation": "dataset",
            "feature_shape": list(splits["train"].feature_shape),
            "n_train": splits["train"].n_samples,
            "n_validation": splits["val"].n_samples,
            "n_test": splits["test"].n_samples,
            "training_dataset_sha256": source_files[training_key]["sha256"],
            "source_files_sha256": _records_sha256(source_files),
            "source_files": source_files,
            "applied_overrides": _applied_override_records(splits),
        }
        loaded[name] = splits
    input_record = {
        "registry": {
            "path": str(data["registry"]),
            "size_bytes": registry_path.stat().st_size,
            "sha256": sha256_file(registry_path),
        },
        "datasets": records,
    }
    return registry, registry_path, benchmark_root, loaded, input_record


def _input_identity(input_record: Mapping[str, Any]) -> str:
    portable = json.loads(canonical_json(input_record))
    return sha256_bytes(canonical_json(portable).encode("utf-8"))


def _emit(
    callback: LogCallback | None,
    event: str,
    *,
    experiment_name: str,
    family: str,
    dataset: str | None = None,
    **payload: Any,
) -> None:
    if callback is None:
        return
    record: dict[str, Any] = {
        "project": PROJECT_NAME,
        "experiment_name": experiment_name,
        "family": family,
    }
    if dataset is not None:
        record["dataset"] = dataset
    record.update(_strict_json_value(payload))
    callback(event, record)


def _log_asset(callback: LogCallback | None, path: Path, *, name: str) -> bool:
    """Upload a final artifact when the callback exposes the Comet asset API."""

    if callback is None:
        return False
    method = getattr(callback, "log_asset", None)
    if not callable(method):
        return False
    method(path, name=name)
    return True


def _prediction_curve(
    *,
    predict_fn: PredictFunction,
    trained: Any,
    query: npt.ArrayLike,
    scales: FloatArray,
    family: str,
    model: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> FloatArray:
    columns = [
        _prediction_at_scale(
            predict_fn=predict_fn,
            trained=trained,
            query=query,
            scale=float(scale),
            family=family,
            model=model,
            evaluation=evaluation,
        )
        for scale in scales
    ]
    return np.ascontiguousarray(np.column_stack(columns), dtype=np.float64)


def _prediction_at_scale(
    *,
    predict_fn: PredictFunction,
    trained: Any,
    query: npt.ArrayLike,
    scale: float,
    family: str,
    model: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    readout: str | None = None,
) -> FloatArray:
    """Evaluate exactly one declared model scale and preserve row cardinality."""

    n_samples = int(np.asarray(query).shape[0])
    prediction = np.ravel(
        np.asarray(
            predict_fn(
                trained,
                query,
                float(scale),
                family=family,
                readout=str(model["readout"] if readout is None else readout),
                divergence_backend=str(evaluation["divergence_backend"]),
                trace_probes=int(evaluation.get("trace_probes") or 0),
                trace_seed=int(evaluation["trace_seed"]),
                batch_size=int(evaluation["batch_size"]),
            ),
            dtype=np.float64,
        )
    )
    if prediction.shape != (n_samples,):
        raise ValueError(
            f"predict_lid returned {prediction.shape}, expected {(n_samples,)}"
        )
    return np.ascontiguousarray(prediction, dtype=np.float64)


def _readouts_from_affine_primitives(prediction: Any) -> dict[str, FloatArray]:
    """Compute every declared readout from one shared field/trace evaluation."""

    posterior_mean = np.asarray(prediction.posterior_mean, dtype=np.float64)
    channel_point = np.asarray(prediction.channel_point, dtype=np.float64)
    posterior_divergence = np.ravel(
        np.asarray(prediction.posterior_divergence, dtype=np.float64)
    )
    marginal_score = np.asarray(prediction.marginal_score, dtype=np.float64)
    marginal_score_divergence = np.ravel(
        np.asarray(prediction.marginal_score_divergence, dtype=np.float64)
    )
    if (
        posterior_mean.ndim != 2
        or channel_point.shape != posterior_mean.shape
        or marginal_score.shape != posterior_mean.shape
        or posterior_divergence.shape != (posterior_mean.shape[0],)
        or marginal_score_divergence.shape != (posterior_mean.shape[0],)
    ):
        raise ValueError("affine primitive arrays have inconsistent shapes")
    alpha = float(prediction.alpha)
    beta = float(prediction.beta)
    noise_ratio = float(prediction.noise_ratio)
    if not all(math.isfinite(value) for value in (alpha, beta, noise_ratio)):
        raise FloatingPointError("affine primitive scale metadata is non-finite")
    if noise_ratio <= 0.0:
        raise ValueError("affine primitive noise_ratio must be positive")
    response = alpha * posterior_divergence
    scaled_bias = (posterior_mean - channel_point) / noise_ratio
    full = response + np.einsum("ij,ij->i", scaled_bias, scaled_bias)
    fm_to_score = posterior_mean.shape[1] + beta**2 * (
        marginal_score_divergence
        + np.einsum("ij,ij->i", marginal_score, marginal_score)
    )
    result = {
        "response": np.ascontiguousarray(response, dtype=np.float64),
        "full": np.ascontiguousarray(full, dtype=np.float64),
        "fm_to_score": np.ascontiguousarray(fm_to_score, dtype=np.float64),
    }
    for readout, values in result.items():
        _require_all_finite(values, label=f"shared-primitive {readout}")
    return result


def _frozen_affine_readouts(
    *,
    predict_fn: PredictFunction,
    trained: Any,
    selected_lambda: float,
    model: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    split_queries: Mapping[str, npt.ArrayLike],
    split_targets: Mapping[str, npt.ArrayLike],
    primary_predictions: Mapping[str, npt.ArrayLike],
    primitive_fn: PrimitiveFunction | None = None,
) -> tuple[dict[str, dict[str, FloatArray]], dict[str, Any]]:
    """Evaluate every affine readout once at the train-selected lambda."""

    predictions: dict[str, dict[str, FloatArray]] = {}
    metrics: dict[str, Any] = {}
    for split, query in split_queries.items():
        target = np.ravel(np.asarray(split_targets[split], dtype=np.float64))
        split_predictions: dict[str, FloatArray] = {}
        shared_predictions: Mapping[str, FloatArray] | None = None
        if primitive_fn is not None:
            primitives = primitive_fn(
                trained,
                query,
                selected_lambda,
                family="independent_affine_flow",
                divergence_backend=str(evaluation["divergence_backend"]),
                trace_probes=int(evaluation.get("trace_probes") or 0),
                trace_seed=int(evaluation["trace_seed"]),
                batch_size=int(evaluation["batch_size"]),
            )
            shared_predictions = _readouts_from_affine_primitives(primitives)
        for readout in _AFFINE_READOUTS:
            if shared_predictions is not None:
                prediction = shared_predictions[readout]
            elif readout == model["readout"] and split in primary_predictions:
                prediction = np.ascontiguousarray(
                    np.ravel(np.asarray(primary_predictions[split], dtype=np.float64))
                )
            else:
                prediction = _prediction_at_scale(
                    predict_fn=predict_fn,
                    trained=trained,
                    query=query,
                    scale=selected_lambda,
                    family="independent_affine_flow",
                    model=model,
                    evaluation=evaluation,
                    readout=readout,
                )
            if readout == model["readout"] and split in primary_predictions:
                primary = np.ravel(
                    np.asarray(primary_predictions[split], dtype=np.float64)
                )
                if not np.array_equal(prediction, primary):
                    raise RuntimeError(
                        f"shared affine primitives changed the primary {split} "
                        f"{readout} prediction"
                    )
                prediction = np.ascontiguousarray(primary, dtype=np.float64)
            _require_all_finite(
                prediction, label=f"{split} {readout} frozen prediction"
            )
            if prediction.shape != target.shape:
                raise ValueError(
                    f"{split} {readout} prediction has shape {prediction.shape}, "
                    f"expected {target.shape}"
                )
            split_predictions[readout] = prediction
            metrics.setdefault(readout, {})[split] = known_lid_metrics(
                prediction, target
            )
        predictions[split] = split_predictions
    summary = {
        "schema_version": 1,
        "scale_semantics": "noise_ratio_lambda=beta/alpha",
        "selected_lambda": float(selected_lambda),
        "primary_readout": "full",
        "scale_selected_with_readout": "full",
        "validation_candidate_count_per_readout": 1,
        "test_candidate_count_per_readout": 1,
        "retrospective_validation_curves_saved": False,
        "retrospective_test_curves_saved": False,
        "readouts": metrics,
    }
    return predictions, summary


def _require_all_finite(value: npt.ArrayLike, *, label: str) -> None:
    """Fail before metrics/artifact sealing when an inference result is invalid."""

    array = np.asarray(value)
    finite = np.isfinite(array)
    if finite.all():
        return
    first = tuple(int(index) for index in np.argwhere(~finite)[0])
    count = int(array.size - np.count_nonzero(finite))
    raise FloatingPointError(
        f"{label} contains {count} non-finite value(s); first index={first}"
    )


def _array_sha256(value: npt.ArrayLike) -> str:
    """Hash array semantics as well as bytes, independent of ``.npy`` headers."""

    array = np.ascontiguousarray(np.asarray(value))
    header = canonical_json(
        {"dtype": array.dtype.str, "shape": [int(size) for size in array.shape]}
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _affine_outer_curve_consistency_error(
    *,
    diagnostic_full: npt.ArrayLike,
    diagnostic_response: npt.ArrayLike,
    diagnostic_correction: npt.ArrayLike,
    outer_selection_curve: npt.ArrayLike,
) -> str | None:
    """Compare independently evaluated affine readouts by a forward-error bound.

    The outer selector and the diagnostic runner evaluate the same checkpoint,
    rows, probes, and scales through independent code paths.  Requiring their
    floating-point results to be bitwise identical is invalid: even a different
    association of ``response + correction`` can change the last few float64
    bits.  The conditioning scale for that addition is
    ``abs(response) + abs(correction)``, including cancellation cases where the
    final full readout is small.  Sixty-four float64 unit roundoffs is a tight,
    deterministic allowance for the duplicated conversion/readout arithmetic.

    Cryptographic identity is deliberately handled separately by
    ``outer_selection_curve_sha256``.  This function only establishes that the
    independently recomputed inner curve is numerically the same computation.
    """

    full = np.asarray(diagnostic_full, dtype=np.float64)
    response = np.asarray(diagnostic_response, dtype=np.float64)
    correction = np.asarray(diagnostic_correction, dtype=np.float64)
    outer = np.asarray(outer_selection_curve, dtype=np.float64)
    shapes = {
        "diagnostic_full": full.shape,
        "diagnostic_response": response.shape,
        "diagnostic_correction": correction.shape,
        "outer_selection_curve": outer.shape,
    }
    if len(set(shapes.values())) != 1 or full.ndim != 2:
        return f"FM diagnostic/selection curve shape mismatch: {shapes}"
    if full.size == 0:
        return "FM diagnostic/selection curves must be nonempty"
    for label, value in (
        ("diagnostic full", full),
        ("diagnostic response", response),
        ("diagnostic correction", correction),
        ("outer selection", outer),
    ):
        if not np.isfinite(value).all():
            return f"{label} curve contains non-finite values"

    magnitude = np.maximum.reduce(
        (
            np.ones_like(full),
            np.abs(full),
            np.abs(outer),
            np.abs(response) + np.abs(correction),
        )
    )
    allowance = (
        _AFFINE_OUTER_CURVE_ROUNDOFF_UNITS * float(np.finfo(np.float64).eps) * magnitude
    )
    error = np.abs(full - outer)
    violations = error > allowance
    if not np.any(violations):
        return None
    normalized = np.divide(
        error,
        allowance,
        out=np.full_like(error, np.inf),
        where=allowance > 0.0,
    )
    index = tuple(
        int(value) for value in np.unravel_index(np.argmax(normalized), full.shape)
    )
    return (
        "FM diagnostic full curve exceeds conditioned roundoff bound at "
        f"index={index}: abs_error={float(error[index]):.17g}, "
        f"allowance={float(allowance[index]):.17g}"
    )


def _deterministic_selection_indices(
    n_samples: int, *, subset_size: int, seed: int
) -> npt.NDArray[np.int64]:
    """Select rows by a versioned SplitMix64 rank, not library RNG behavior."""

    if subset_size <= 0 or subset_size >= n_samples:
        raise PilotConfigError(
            "selection.subset_size must be positive and strictly smaller than "
            f"the source train split ({n_samples}); got {subset_size}"
        )
    indices = np.arange(n_samples, dtype=np.uint64)
    with np.errstate(over="ignore"):
        mixed = indices + np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15)
        mixed = (mixed ^ (mixed >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        mixed = (mixed ^ (mixed >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        mixed ^= mixed >> np.uint64(31)
    # Stable ordering makes the index itself the deterministic secondary key
    # in the astronomically unlikely event of a 64-bit key collision.
    ranked = np.argsort(mixed, kind="stable")[:subset_size]
    return np.sort(ranked.astype(np.int64, copy=False))


def _partition_source_train(
    train: npt.ArrayLike,
    train_target: npt.ArrayLike,
    *,
    selection: Mapping[str, Any],
) -> TrainSelectionPartition:
    """Create a target-bearing selector subset disjoint from optimizer rows."""

    features = np.asarray(train)
    target = np.ravel(np.asarray(train_target, dtype=np.float64))
    if features.ndim != 2 or features.shape[0] != target.shape[0]:
        raise PilotConfigError(
            "source train features and LID targets must have equal sample counts"
        )
    _require_all_finite(features, label="source train features")
    _require_all_finite(target, label="source train target")
    selection_indices = _deterministic_selection_indices(
        int(features.shape[0]),
        subset_size=int(selection["subset_size"]),
        seed=int(selection["seed"]),
    )
    fit_mask = np.ones(features.shape[0], dtype=bool)
    fit_mask[selection_indices] = False
    fit_indices = np.flatnonzero(fit_mask).astype(np.int64, copy=False)
    fit_features = np.ascontiguousarray(features[fit_indices])
    selection_features = np.ascontiguousarray(features[selection_indices])
    selection_target = np.ascontiguousarray(target[selection_indices])
    if np.intersect1d(fit_indices, selection_indices).size:
        raise AssertionError("optimizer and train-selection indices overlap")
    record: dict[str, Any] = {
        "schema_version": 1,
        "protocol": TRAIN_SELECTION_PROTOCOL,
        "source_split": "train",
        "index_algorithm": "splitmix64_rank_v1",
        "seed": int(selection["seed"]),
        "n_source_train": int(features.shape[0]),
        "n_optimizer_fit": int(fit_indices.size),
        "n_train_selection": int(selection_indices.size),
        "optimizer_overlap_count": 0,
        "fit_indices_sha256": _array_sha256(fit_indices),
        "selection_indices_sha256": _array_sha256(selection_indices),
        "fit_features_sha256": _array_sha256(fit_features),
        "selection_features_sha256": _array_sha256(selection_features),
        "selection_target_sha256": _array_sha256(selection_target),
    }
    record["partition_sha256"] = sha256_bytes(canonical_json(record).encode("utf-8"))
    return TrainSelectionPartition(
        fit_indices=fit_indices,
        selection_indices=selection_indices,
        fit_features=fit_features,
        selection_features=selection_features,
        selection_target=selection_target,
        record=record,
    )


def _select_by_train_targets(
    *,
    scales: FloatArray,
    curve: FloatArray,
    target: FloatArray,
    criterion: str,
    tie_tolerance: float,
    tie_break: str,
) -> tuple[int, dict[str, Any]]:
    """Select one frozen candidate using only held-out source-train targets."""

    if criterion != "mae":
        raise ValueError("only the declared train-selection criterion 'mae' is valid")
    if tie_break not in {"smaller", "larger"}:
        raise ValueError("tie_break must be 'smaller' or 'larger'")
    scale_array = np.ravel(np.asarray(scales, dtype=np.float64))
    prediction_curve = np.asarray(curve, dtype=np.float64)
    truth = np.ravel(np.asarray(target, dtype=np.float64))
    if prediction_curve.shape != (truth.size, scale_array.size):
        raise ValueError(
            "train-selection curve shape must be "
            f"{(truth.size, scale_array.size)}, got {prediction_curve.shape}"
        )
    _require_all_finite(prediction_curve, label="train-selection prediction curve")
    _require_all_finite(truth, label="train-selection target")
    candidate_metrics = [
        known_lid_metrics(prediction_curve[:, index], truth)
        for index in range(scale_array.size)
    ]
    scores = np.asarray(
        [float(metrics[criterion]) for metrics in candidate_metrics],
        dtype=np.float64,
    )
    best_score = float(scores.min())
    effective_tolerance = max(
        float(tie_tolerance),
        32.0 * np.finfo(np.float64).eps * max(1.0, abs(best_score)),
    )
    tied = np.flatnonzero(np.abs(scores - best_score) <= effective_tolerance)
    tied_scales = scale_array[tied]
    tied_position = int(np.argmin(tied_scales))
    if tie_break == "larger":
        tied_position = int(np.argmax(tied_scales))
    selected_index = int(tied[tied_position])
    diagnostics: dict[str, Any] = {
        "schema_version": 1,
        "method": TRAIN_SELECTION_PROTOCOL,
        "criterion": criterion,
        "uses_ground_truth": True,
        "ground_truth_split": "train_selection",
        "uses_validation_ground_truth": False,
        "uses_test_ground_truth": False,
        "selected_index": selected_index,
        "selected_scale": float(scale_array[selected_index]),
        "selected_score": float(scores[selected_index]),
        "criterion_scores": [float(value) for value in scores],
        "candidate_metrics": candidate_metrics,
        "configured_tie_tolerance": float(tie_tolerance),
        "effective_tie_tolerance": effective_tolerance,
        "tied_indices": [int(value) for value in tied],
        "tie_break": tie_break,
        "prefer": tie_break,
    }
    return selected_index, diagnostics


def _selection_coordinate(
    scales: FloatArray, *, family: str
) -> tuple[FloatArray, str, str, str]:
    """Return the physical coordinate associated with each model parameter.

    Rectified-flow networks are evaluated at time ``t``, while the Gaussian
    channel scale in the endpoint identity is ``lambda = (1 - t) / t``.
    Supervised selection is performed directly over candidate model indices;
    this coordinate is retained as a diagnostic, not as an optimization input.
    """

    if family == "rectified_flow":
        coordinate = (1.0 - scales) / scales
        return (
            np.ascontiguousarray(coordinate, dtype=np.float64),
            "lambda",
            "(1 - t) / t",
            "t",
        )
    if family == "independent_affine_flow":
        return (
            scales.copy(),
            "lambda",
            "beta / alpha",
            "lambda",
        )
    if family == "scale_conditioned_nf":
        return (
            np.ascontiguousarray(np.log(scales), dtype=np.float64),
            "log_epsilon",
            "log(epsilon)",
            "epsilon",
        )
    if family == "schrodinger_bridge":
        return (
            scales.copy(),
            "tau",
            "T - t",
            "tau",
        )
    return scales.copy(), "sigma", "sigma", "sigma"


def _reported_selection_diagnostics(
    diagnostics: Mapping[str, Any],
    *,
    scales: FloatArray,
    coordinates: FloatArray,
    selected_index: int,
    coordinate_name: str,
    coordinate_formula: str,
    model_scale_name: str,
    coordinate_prefer: str,
    model_scale_prefer: str,
    model_training: Mapping[str, Any],
) -> dict[str, Any]:
    """Make the selector coordinate and original model parameter unambiguous."""

    result = dict(diagnostics)
    selected_coordinate = float(coordinates[selected_index])
    selected_model_scale = float(scales[selected_index])
    result["selection_coordinate"] = {
        "name": coordinate_name,
        "formula": coordinate_formula,
        "values": [float(value) for value in coordinates],
        "selected_value": selected_coordinate,
        "prefer": coordinate_prefer,
    }
    result["model_scale"] = {
        "name": model_scale_name,
        "selected_value": selected_model_scale,
        "prefer": model_scale_prefer,
    }
    result["selection_coordinate_prefer"] = coordinate_prefer
    result["model_scale_prefer"] = model_scale_prefer
    result["selected_scale"] = selected_model_scale
    if model_scale_name == "t":
        result["selected_t"] = selected_model_scale
        result["selected_delta_t"] = 1.0 - selected_model_scale
    elif model_scale_name == "epsilon":
        result["selected_epsilon"] = selected_model_scale
        result["selected_log_epsilon"] = selected_coordinate
    elif model_scale_name == "tau":
        terminal_time = float(model_training["bridge_terminal_time"])
        diffusivity = float(model_training["bridge_diffusivity"])
        result["selected_tau"] = selected_model_scale
        result["selected_t"] = terminal_time - selected_model_scale
        result["selected_sigma"] = math.sqrt(diffusivity * selected_model_scale)
    elif model_scale_name == "lambda":
        schedule = str(model_training["flow_schedule"])
        result["selected_lambda"] = selected_model_scale
        if schedule == "log_noise":
            native_name = "u"
            native_formula = "log(lambda)"
            native_value = math.log(selected_model_scale)
        elif schedule == "rectified_linear":
            native_name = "t"
            native_formula = "1 / (1 + lambda)"
            native_value = 1.0 / (1.0 + selected_model_scale)
        elif schedule == "vp_trigonometric":
            native_name = "t"
            native_formula = "2 * atan(1 / lambda) / pi"
            native_value = 2.0 * math.atan(1.0 / selected_model_scale) / math.pi
        else:
            raise ValueError(f"unsupported affine-flow schedule {schedule!r}")
        result["affine_flow"] = {
            "variant_id": str(model_training["flow_variant_id"]),
            "schedule": schedule,
            "parameterization": str(model_training["flow_parameterization"]),
            "scale_semantics": "noise_ratio_lambda=beta/alpha",
            "selected_native_coordinate": {
                "name": native_name,
                "formula": native_formula,
                "value": native_value,
            },
        }
    return result


def _training_result_record(result: Any) -> dict[str, Any]:
    fields = {
        "family": getattr(result, "family", None),
        "best_epoch": getattr(result, "best_epoch", None),
        "best_validation_loss": getattr(result, "best_validation_loss", None),
        "metrics": getattr(result, "metrics", {}),
        "model_contract": getattr(result, "model_contract", {}),
        "internal_preprocessing": getattr(result, "preprocessing", {}),
        "internal_preprocessing_sha256": getattr(result, "preprocessing_sha256", None),
    }
    return _strict_json_value(fields)


def _validate_affine_contract(contract: Any, *, training: Mapping[str, Any]) -> None:
    """Bind an affine model contract to the exact Hydra variant identity."""

    if not isinstance(contract, Mapping):
        raise TypeError(
            "independent affine-flow result must expose a model_contract mapping"
        )
    expected = {
        "family": "independent_affine_flow",
        "scale_semantics": "noise_ratio_lambda=beta/alpha",
        "variant_id": training["flow_variant_id"],
        "schedule": training["flow_schedule"],
        "parameterization": training["flow_parameterization"],
        "conditioning": training["flow_conditioning"],
        "scale_sampling": training["flow_scale_sampling"],
        "loss_weighting": training["flow_loss_weighting"],
        "noise_ratio_min": float(training["flow_noise_ratio_min"]),
        "noise_ratio_max": float(training["flow_noise_ratio_max"]),
        "readouts": list(_AFFINE_READOUTS),
    }
    mismatches = {
        key: {"expected": expected_value, "actual": contract.get(key)}
        for key, expected_value in expected.items()
        if contract.get(key) != expected_value
    }
    if mismatches:
        raise RuntimeError(
            "affine-flow checkpoint contract differs from resolved Hydra config: "
            f"{mismatches}"
        )


def _validate_affine_training_result(
    trained: Any, *, training: Mapping[str, Any]
) -> None:
    """Bind a trained affine checkpoint to the exact Hydra variant identity."""

    _validate_affine_contract(
        getattr(trained, "model_contract", None), training=training
    )


def _features_in_model_space(
    trained: Any, value: npt.ArrayLike, *, label: str
) -> np.ndarray:
    """Mirror the predictor's CPU-fp32 normalization exactly."""

    import torch

    raw = torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float32)
    if raw.ndim < 2 or raw.shape[0] <= 0:
        raise ValueError(f"{label} must contain a nonempty batch")
    raw = raw.reshape(raw.shape[0], -1).contiguous()
    mean_value = getattr(trained, "normalization_mean", None)
    scale_value = getattr(trained, "normalization_scale", None)
    if mean_value is None or scale_value is None:
        raise TypeError(
            "independent affine-flow result must expose normalization_mean/scale"
        )
    mean = (
        torch.as_tensor(mean_value)
        .detach()
        .to(device="cpu", dtype=torch.float32)
        .reshape(-1)
    )
    scale = float(scale_value)
    if mean.shape != (raw.shape[1],):
        raise ValueError(f"{label} normalization mean has the wrong shape")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"{label} normalization scale must be finite and positive")
    model_space = ((raw - mean.reshape(1, -1)) / scale).contiguous()
    if not torch.isfinite(model_space).all():
        raise FloatingPointError(f"{label} model-space features are non-finite")
    return model_space.numpy()


def _run_affine_diagnostics(
    *,
    name: str,
    cell_dir: Path,
    partition: TrainSelectionPartition,
    trained: Any,
    scales: FloatArray,
    selection_curve: FloatArray,
    model: Mapping[str, Any],
    experiment_name: str,
    log_callback: LogCallback | None,
    diagnostics_fn: DiagnosticsFunction | None,
    diagnostics_validate_fn: DiagnosticsValidator | None,
    primitive_fn: PrimitiveFunction | None,
) -> dict[str, Any]:
    """Run, validate, and attest train-only affine diagnostics."""

    if (
        diagnostics_fn is None
        or diagnostics_validate_fn is None
        or primitive_fn is None
    ):
        raise RuntimeError(
            "independent affine-flow diagnostics implementation is missing"
        )
    diagnostic_config = model["diagnostics"]
    if partition.fit_features.shape[0] < int(
        diagnostic_config["oracle_reference_size"]
    ):
        raise PilotConfigError(
            "diagnostics.oracle_reference_size exceeds optimizer-fit rows"
        )
    query_model_space = _features_in_model_space(
        trained,
        partition.selection_features,
        label="train-selection",
    )
    reference_model_space = _features_in_model_space(
        trained,
        partition.fit_features,
        label="optimizer-fit oracle reference",
    )
    outer_selection_curve_sha = _array_sha256(selection_curve)
    diagnostics_dir = cell_dir / "fm_diagnostics"
    returned = Path(
        diagnostics_fn(
            diagnostics_dir,
            variant_id=str(model["training"]["flow_variant_id"]),
            outer_selection_curve_sha256=outer_selection_curve_sha,
            trained=trained,
            query=partition.selection_features,
            query_model_space=query_model_space,
            target=partition.selection_target,
            oracle_reference_model_space=reference_model_space,
            scales=scales,
            config=dict(diagnostic_config),
            primitive_fn=primitive_fn,
        )
    )
    if returned.resolve() != diagnostics_dir.resolve():
        raise RuntimeError("FM diagnostics returned an unexpected output directory")
    diagnostic_errors = diagnostics_validate_fn(diagnostics_dir)
    if diagnostic_errors:
        raise RuntimeError(
            "FM diagnostics failed validation: " + "; ".join(diagnostic_errors)
        )
    try:
        metadata = json.loads((diagnostics_dir / "metadata.json").read_text("utf-8"))
        diagnostic_summary = json.loads(
            (diagnostics_dir / "summary.json").read_text("utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot load validated FM diagnostics: {exc}") from exc
    if not isinstance(metadata, dict) or not isinstance(diagnostic_summary, dict):
        raise TypeError("validated FM diagnostic JSON roots must be mappings")
    variant_id = str(model["training"]["flow_variant_id"])
    if metadata.get("variant_id") != variant_id:
        raise RuntimeError("FM diagnostic variant differs from Hydra config")
    if canonical_json(metadata.get("config")) != canonical_json(diagnostic_config):
        raise RuntimeError("FM diagnostic config differs from resolved Hydra config")
    if diagnostic_summary.get("variant_id") != variant_id:
        raise RuntimeError("FM diagnostic summary variant differs from Hydra config")
    checkpoint_sha = getattr(trained, "checkpoint_sha256", None)
    if metadata.get("checkpoint_sha256") != checkpoint_sha:
        raise RuntimeError("FM diagnostics are not bound to the trained checkpoint")
    if metadata.get("outer_selection_curve_sha256") != outer_selection_curve_sha:
        raise RuntimeError(
            "FM diagnostics are not cryptographically bound to the outer "
            "train-selection curve"
        )
    raw_query_sha = _array_sha256(partition.selection_features)
    if metadata.get("raw_query_sha256") != raw_query_sha:
        raise RuntimeError("FM diagnostics are not bound to the train-selection rows")
    try:
        curve_error = _affine_outer_curve_consistency_error(
            diagnostic_full=_load_numeric_output(
                diagnostics_dir / "arrays" / "full.npy"
            ),
            diagnostic_response=_load_numeric_output(
                diagnostics_dir / "arrays" / "response.npy"
            ),
            diagnostic_correction=_load_numeric_output(
                diagnostics_dir / "arrays" / "correction.npy"
            ),
            outer_selection_curve=selection_curve,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"cannot bind FM diagnostics to the outer selection curve: {exc}"
        ) from exc
    if curve_error is not None:
        raise RuntimeError(curve_error)
    for row in diagnostic_summary.get("per_scale", []):
        if not isinstance(row, Mapping):
            raise TypeError("FM diagnostic per_scale rows must be mappings")
        _emit(
            log_callback,
            f"dataset.{name}.fm_diagnostics.scale",
            experiment_name=experiment_name,
            family="independent_affine_flow",
            dataset=name,
            step=int(row["scale_index"]),
            **dict(row),
        )
    uploaded_assets = {
        filename: _log_asset(
            log_callback,
            diagnostics_dir / filename,
            name=f"{PROJECT_NAME}-{variant_id}-{name}-fm-diagnostics-{filename}",
        )
        for filename in ("summary.json", "metadata.json", "manifest.json")
    }
    record = {
        "schema_version": metadata.get("schema_version"),
        "path": "fm_diagnostics",
        "variant_id": variant_id,
        "source_split": "train_selection",
        "protocol": metadata.get("protocol"),
        "n_query": metadata.get("n_query"),
        "n_scales": metadata.get("n_scales"),
        "checkpoint_sha256": checkpoint_sha,
        "outer_selection_curve_sha256": outer_selection_curve_sha,
        "raw_query_sha256": raw_query_sha,
        "manifest_sha256": sha256_path(diagnostics_dir / "manifest.json"),
        "metadata_sha256": sha256_path(diagnostics_dir / "metadata.json"),
        "summary_sha256": sha256_path(diagnostics_dir / "summary.json"),
    }
    _emit(
        log_callback,
        f"dataset.{name}.fm_diagnostics.completed",
        experiment_name=experiment_name,
        family="independent_affine_flow",
        dataset=name,
        diagnostics=record,
        uploaded_assets=uploaded_assets,
    )
    return record


def _macro_metrics(
    dataset_summaries: Mapping[str, Any], split: str
) -> dict[str, float]:
    metric_names = ("mae", "rmse", "bias", "median_absolute_error")
    result: dict[str, float] = {}
    for metric in metric_names:
        values = [
            float(summary[split][metric])
            for summary in dataset_summaries.values()
            if metric in summary[split]
        ]
        if values:
            result[f"mean_{metric}"] = float(np.mean(values))
    return result


def _macro_frozen_readouts(dataset_summaries: Mapping[str, Any]) -> dict[str, Any]:
    """Aggregate the three affine readouts without accepting evaluation curves."""

    result: dict[str, Any] = {}
    for readout in _AFFINE_READOUTS:
        result[readout] = {}
        for split in ("train_selection", "validation", "test"):
            metric_names = ("mae", "rmse", "bias", "median_absolute_error")
            result[readout][split] = {
                f"mean_{metric}": float(
                    np.mean(
                        [
                            float(
                                summary["frozen_readouts"]["readouts"][readout][split][
                                    metric
                                ]
                            )
                            for summary in dataset_summaries.values()
                        ]
                    )
                )
                for metric in metric_names
            }
    return result


def _portable_output_inventory(root: Path) -> dict[str, dict[str, str | int]]:
    records: dict[str, dict[str, str | int]] = {}
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in list(directory_names):
            path = directory_path / name
            if path.is_symlink():
                raise ValueError(f"symlink is forbidden in pilot outputs: {path}")
        for name in file_names:
            path = directory_path / name
            relative = path.relative_to(root).as_posix()
            if relative == "manifest.json":
                continue
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"non-regular pilot output: {path}")
            records[relative] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_path(path),
            }
    return dict(sorted(records.items()))


def _artifact_registry(*, run_dir: Path, datasets: Mapping[str, Any]) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for name, record in datasets.items():
        cell_dir = run_dir / "datasets" / name
        checkpoint = cell_dir / "checkpoint.pt"
        training_config = cell_dir / "training.yaml"
        artifacts[f"{name}/dataset"] = {
            "checkpoint_path": checkpoint.relative_to(run_dir).as_posix(),
            "checkpoint_sha256": sha256_path(checkpoint),
            "training_config_path": training_config.relative_to(run_dir).as_posix(),
            "training_config_sha256": sha256_path(training_config),
            "training_dataset_sha256": record["training_dataset_sha256"],
            "preprocessing_sha256": record["preprocessing_sha256"],
        }
    return {"schema_version": 1, "artifacts": artifacts}


def _run_dataset(
    *,
    name: str,
    splits: Mapping[str, LoadedSplit],
    input_record: Mapping[str, Any],
    run_dir: Path,
    config: Mapping[str, Any],
    train_fn: TrainFunction,
    predict_fn: PredictFunction,
    affine_primitive_fn: PrimitiveFunction | None,
    diagnostics_fn: DiagnosticsFunction | None,
    diagnostics_validate_fn: DiagnosticsValidator | None,
    log_callback: LogCallback | None,
) -> dict[str, Any]:
    model = config["pilot_model"]
    evaluation = config["evaluation"]
    family = str(model["family"])
    experiment_name = str(config["experiment_name"])
    source_train = _flatten_features(splits["train"])
    train_target = np.ravel(np.asarray(splits["train"].lid, dtype=np.float64))
    partition = _partition_source_train(
        source_train,
        train_target,
        selection=evaluation["selection"],
    )
    cell_dir = run_dir / "datasets" / name
    cell_dir.mkdir(parents=True)
    checkpoint_path = cell_dir / "checkpoint.pt"

    def training_log(payload: Mapping[str, Any]) -> None:
        trainer_payload = dict(payload)
        # The outer event owns the stable family/dataset context.  The trainer
        # reports its canonical family too; do not pass that duplicate keyword
        # through ``_emit``.
        trainer_payload.pop("family", None)
        if "epoch" in trainer_payload:
            trainer_payload["step"] = trainer_payload["epoch"]
        _emit(
            log_callback,
            f"dataset.{name}.training.epoch",
            experiment_name=experiment_name,
            family=family,
            dataset=name,
            **trainer_payload,
        )

    _emit(
        log_callback,
        f"dataset.{name}.started",
        experiment_name=experiment_name,
        family=family,
        dataset=name,
    )
    trained = train_fn(
        family,
        partition.fit_features,
        partition.selection_features,
        dict(model["training"]),
        checkpoint_path,
        training_log if log_callback is not None else None,
    )
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size <= 0:
        raise RuntimeError(f"trainer did not write a checkpoint: {checkpoint_path}")
    declared_checkpoint_sha = getattr(trained, "checkpoint_sha256", None)
    checkpoint_sha = sha256_path(checkpoint_path)
    if (
        declared_checkpoint_sha is not None
        and declared_checkpoint_sha != checkpoint_sha
    ):
        raise RuntimeError(
            "TrainingResult checkpoint SHA does not match checkpoint file"
        )
    actual_family = getattr(trained, "family", None)
    if actual_family != _FAMILY_FOR_ARTIFACTS[family]:
        raise RuntimeError(
            "TrainingResult family mismatch: "
            f"expected {_FAMILY_FOR_ARTIFACTS[family]!r}, got {actual_family!r}"
        )
    if family == "independent_affine_flow":
        _validate_affine_training_result(trained, training=model["training"])

    scales = np.asarray(model["scales"], dtype=np.float64)
    train_selection_curve = _prediction_curve(
        predict_fn=predict_fn,
        trained=trained,
        query=partition.selection_features,
        scales=scales,
        family=family,
        model=model,
        evaluation=evaluation,
    )
    _require_all_finite(train_selection_curve, label="train-selection prediction curve")
    selection = evaluation["selection"]
    (
        selection_coordinates,
        coordinate_name,
        coordinate_formula,
        model_scale_name,
    ) = _selection_coordinate(scales, family=family)
    model_scale_prefer = str(model["selection_prefer"])
    coordinate_prefer = model_scale_prefer
    if family == "rectified_flow":
        coordinate_prefer = "smaller" if model_scale_prefer == "larger" else "larger"
    selected_index, raw_selection_diagnostics = _select_by_train_targets(
        scales=scales,
        curve=train_selection_curve,
        target=partition.selection_target,
        criterion=str(selection["criterion"]),
        tie_tolerance=float(selection["tie_tolerance"]),
        tie_break=str(selection["tie_break"]),
    )
    selection_diagnostics = _reported_selection_diagnostics(
        raw_selection_diagnostics,
        scales=scales,
        coordinates=selection_coordinates,
        selected_index=selected_index,
        coordinate_name=coordinate_name,
        coordinate_formula=coordinate_formula,
        model_scale_name=model_scale_name,
        coordinate_prefer=coordinate_prefer,
        model_scale_prefer=model_scale_prefer,
        model_training=model["training"],
    )
    selection_diagnostics["partition"] = dict(partition.record)
    train_selection_prediction = train_selection_curve[:, selected_index]
    _require_all_finite(
        train_selection_prediction, label="selected train-selection prediction"
    )
    train_selection_metrics = known_lid_metrics(
        train_selection_prediction, partition.selection_target
    )
    _emit(
        log_callback,
        f"dataset.{name}.selection.frozen",
        experiment_name=experiment_name,
        family=family,
        dataset=name,
        selected_index=selected_index,
        selected_scale=float(scales[selected_index]),
        selection_target_split="train_selection",
        selection_criterion=str(selection["criterion"]),
        scale_selection=selection_diagnostics,
        train_selection=train_selection_metrics,
        partition=partition.record,
    )
    fm_diagnostics_record: dict[str, Any] | None = None
    if family == "independent_affine_flow":
        fm_diagnostics_record = _run_affine_diagnostics(
            name=name,
            cell_dir=cell_dir,
            partition=partition,
            trained=trained,
            scales=scales,
            selection_curve=train_selection_curve,
            model=model,
            experiment_name=experiment_name,
            log_callback=log_callback,
            diagnostics_fn=diagnostics_fn,
            diagnostics_validate_fn=diagnostics_validate_fn,
            primitive_fn=affine_primitive_fn,
        )
    # The selected index is immutable before either benchmark evaluation split
    # is touched. Even their feature/target arrays are intentionally resolved
    # below this boundary. Each split is evaluated exactly once at the frozen
    # scale; retrospective validation/test curves are outside the primary run.
    selected_scale = float(scales[selected_index])
    validation = _flatten_features(splits["val"])
    validation_target = np.ravel(np.asarray(splits["val"].lid, dtype=np.float64))
    test = _flatten_features(splits["test"])
    test_target = np.ravel(np.asarray(splits["test"].lid, dtype=np.float64))

    frozen_readout_predictions: dict[str, dict[str, FloatArray]] | None = None
    frozen_readout_summary: dict[str, Any] | None = None
    if family == "independent_affine_flow":
        frozen_readout_predictions, frozen_readout_summary = _frozen_affine_readouts(
            predict_fn=predict_fn,
            trained=trained,
            selected_lambda=selected_scale,
            model=model,
            evaluation=evaluation,
            split_queries={
                "train_selection": partition.selection_features,
                "validation": validation,
                "test": test,
            },
            split_targets={
                "train_selection": partition.selection_target,
                "validation": validation_target,
                "test": test_target,
            },
            primary_predictions={
                "train_selection": train_selection_prediction,
            },
            primitive_fn=affine_primitive_fn,
        )
        validation_prediction = frozen_readout_predictions["validation"]["full"]
        test_prediction = frozen_readout_predictions["test"]["full"]
    else:
        validation_prediction = _prediction_at_scale(
            predict_fn=predict_fn,
            trained=trained,
            query=validation,
            scale=selected_scale,
            family=family,
            model=model,
            evaluation=evaluation,
        )
        test_prediction = _prediction_at_scale(
            predict_fn=predict_fn,
            trained=trained,
            query=test,
            scale=selected_scale,
            family=family,
            model=model,
            evaluation=evaluation,
        )
    _require_all_finite(validation_prediction, label="validation prediction")
    validation_metrics = known_lid_metrics(validation_prediction, validation_target)
    _require_all_finite(test_prediction, label="test prediction")
    test_metrics = known_lid_metrics(test_prediction, test_target)

    _save_npy(cell_dir / "scales.npy", scales)
    _save_npy(cell_dir / "train_fit_indices.npy", partition.fit_indices)
    _save_npy(cell_dir / "train_selection_indices.npy", partition.selection_indices)
    _save_npy(cell_dir / "train_selection_curve.npy", train_selection_curve)
    _save_npy(cell_dir / "train_selection_prediction.npy", train_selection_prediction)
    _save_npy(cell_dir / "train_selection_target.npy", partition.selection_target)
    _save_npy(cell_dir / "validation_prediction.npy", validation_prediction)
    _save_npy(cell_dir / "test_prediction.npy", test_prediction)
    _save_npy(cell_dir / "validation_target.npy", validation_target)
    _save_npy(cell_dir / "test_target.npy", test_target)
    if frozen_readout_predictions is not None:
        for split, predictions in frozen_readout_predictions.items():
            for readout, prediction in predictions.items():
                _save_npy(cell_dir / f"{split}_prediction__{readout}.npy", prediction)
    history = getattr(trained, "history", [])
    _write_json(cell_dir / "training_history.json", history)

    preprocessing_spec = {"kind": "identity"}
    preprocessing_sha = sha256_bytes(canonical_json(preprocessing_spec).encode("utf-8"))
    training_config = {
        "schema_version": 1,
        "project": PROJECT_NAME,
        "experiment_name": experiment_name,
        "model": dict(model),
        "dataset": {
            "name": name,
            "representation": "dataset",
            "feature_shape": input_record["feature_shape"],
            "train_partition": dict(partition.record),
        },
        "preprocessing": {
            "external": preprocessing_spec,
            "internal_train_only_normalization": getattr(
                trained, "preprocessing", {"storage": "checkpoint"}
            ),
            "internal_preprocessing_sha256": getattr(
                trained, "preprocessing_sha256", None
            ),
        },
        "provenance": {
            "schema_version": 1,
            "model_name": str(model["name"]),
            "model_family": _FAMILY_FOR_ARTIFACTS[family],
            "model_seed": int(config["seed"]),
            "dataset_name": name,
            "representation": "dataset",
            "training_dataset_sha256": input_record["training_dataset_sha256"],
            "preprocessing_sha256": preprocessing_sha,
        },
    }
    _write_yaml(cell_dir / "training.yaml", training_config)
    summary = {
        "schema_version": 1,
        "project": PROJECT_NAME,
        "experiment_name": experiment_name,
        "dataset": name,
        "representation": "dataset",
        "model": _training_result_record(trained),
        "checkpoint_sha256": checkpoint_sha,
        "training_dataset_sha256": input_record["training_dataset_sha256"],
        "preprocessing_sha256": preprocessing_sha,
        "selection_protocol": TRAIN_SELECTION_PROTOCOL,
        "selection_target_split": "train_selection",
        "selection_uses_lid_targets": True,
        "selection_uses_validation_targets": False,
        "selection_uses_test_targets": False,
        "evaluation_protocol": FROZEN_EVALUATION_PROTOCOL,
        "frozen_evaluation": {
            "schema_version": 1,
            "selected_index": selected_index,
            "selected_scale": selected_scale,
            "validation_candidate_count": 1,
            "test_candidate_count": 1,
            "retrospective_curves_saved": False,
        },
        "scale_selection": selection_diagnostics,
        "train_selection": train_selection_metrics,
        "validation": validation_metrics,
        "test": test_metrics,
    }
    if frozen_readout_summary is not None:
        summary["frozen_readouts"] = frozen_readout_summary
    if fm_diagnostics_record is not None:
        summary["fm_diagnostics"] = fm_diagnostics_record
    _write_json(cell_dir / "summary.json", summary)
    _emit(
        log_callback,
        f"dataset.{name}.completed",
        experiment_name=experiment_name,
        family=family,
        dataset=name,
        selected_scale=float(scales[selected_index]),
        selection_coordinate=selection_diagnostics["selection_coordinate"],
        model_scale=selection_diagnostics["model_scale"],
        validation=validation_metrics,
        test=test_metrics,
        **(
            {"frozen_readouts": frozen_readout_summary}
            if frozen_readout_summary is not None
            else {}
        ),
    )
    return summary


def run_pilot(
    hydra_config: DictConfig | Mapping[str, Any],
    *,
    root: Path | None = None,
    output_root: Path | None = None,
    train_fn: TrainFunction | None = None,
    predict_fn: PredictFunction | None = None,
    affine_primitive_fn: PrimitiveFunction | None = None,
    diagnostics_fn: DiagnosticsFunction | None = None,
    diagnostics_validate_fn: DiagnosticsValidator | None = None,
    log_callback: LogCallback | None = None,
) -> Path:
    """Train three dataset-specific models and write a sealed experiment."""

    config = validate_pilot_config(hydra_config)
    project_root = (repository_root() if root is None else Path(root)).resolve()
    if train_fn is None or predict_fn is None:
        from models.training import predict_lid, train_model

        train_fn = train_model if train_fn is None else train_fn
        predict_fn = predict_lid if predict_fn is None else predict_fn
    if (
        config["pilot_model"]["family"] == "independent_affine_flow"
        and affine_primitive_fn is None
    ):
        from models.training import predict_affine_primitives

        affine_primitive_fn = predict_affine_primitives
    if config["pilot_model"]["family"] == "independent_affine_flow" and (
        diagnostics_fn is None or diagnostics_validate_fn is None
    ):
        from experiments.fm_diagnostics import (
            run_fm_diagnostics,
            validate_fm_diagnostics,
        )

        diagnostics_fn = (
            run_fm_diagnostics if diagnostics_fn is None else diagnostics_fn
        )
        diagnostics_validate_fn = (
            validate_fm_diagnostics
            if diagnostics_validate_fn is None
            else diagnostics_validate_fn
        )
    _, _, _, loaded, input_record = _load_inputs(root=project_root, config=config)
    input_sha = _input_identity(input_record)
    config_sha = sha256_bytes(canonical_json(config).encode("utf-8"))
    source_sha = hash_declared_sources(project_root)
    experiment_name = str(config["experiment_name"])
    identity = {
        "schema_version": 1,
        "project": PROJECT_NAME,
        "experiment_name": experiment_name,
        "family": config["pilot_model"]["family"],
        "config_sha256": config_sha,
        "source_tree_sha256": source_sha,
        "input_sha256": input_sha,
    }
    run_id = sha256_bytes(canonical_json(identity).encode("utf-8"))[:20]
    configured_output = _safe_output_root(config["output_root"])
    selected_output = configured_output if output_root is None else Path(output_root)
    if not selected_output.is_absolute():
        selected_output = project_root / selected_output
    selected_output = selected_output.resolve()
    family = str(config["pilot_model"]["family"])
    run_label = (
        str(config["pilot_model"]["training"]["flow_variant_id"])
        if family == "independent_affine_flow"
        else family
    )
    final_dir = selected_output / f"{run_label}__{run_id}"
    if final_dir.exists():
        errors = validate_pilot_experiment(final_dir)
        if errors:
            raise RuntimeError(
                f"refusing to reuse invalid pilot run {final_dir}: {errors}"
            )
        return final_dir

    selected_output.mkdir(parents=True, exist_ok=True)
    work_dir = selected_output / f".{run_label}__{run_id}.incomplete-{os.getpid()}"
    if work_dir.exists():
        raise RuntimeError(f"pilot work directory already exists: {work_dir}")
    work_dir.mkdir()
    _write_yaml(work_dir / "resolved_config.yaml", config)
    _emit(
        log_callback,
        "experiment.started",
        experiment_name=experiment_name,
        family=family,
        run_id=run_id,
    )
    summaries: dict[str, Any] = {}
    for name in PILOT_DATASETS:
        summaries[name] = _run_dataset(
            name=name,
            splits=loaded[name],
            input_record=input_record["datasets"][name],
            run_dir=work_dir,
            config=config,
            train_fn=train_fn,
            predict_fn=predict_fn,
            affine_primitive_fn=affine_primitive_fn,
            diagnostics_fn=diagnostics_fn,
            diagnostics_validate_fn=diagnostics_validate_fn,
            log_callback=log_callback,
        )
    aggregate = {
        "schema_version": 1,
        "project": PROJECT_NAME,
        "experiment_name": experiment_name,
        "run_id": run_id,
        "family": family,
        "selection_protocol": TRAIN_SELECTION_PROTOCOL,
        "selection_target_split": "train_selection",
        "selection_uses_lid_targets": True,
        "selection_uses_validation_targets": False,
        "selection_uses_test_targets": False,
        "evaluation_protocol": FROZEN_EVALUATION_PROTOCOL,
        "retrospective_evaluation_curves_saved": False,
        "datasets": summaries,
        "macro_train_selection": _macro_metrics(summaries, "train_selection"),
        "macro_validation": _macro_metrics(summaries, "validation"),
        "macro_test": _macro_metrics(summaries, "test"),
    }
    if family == "independent_affine_flow":
        aggregate["macro_frozen_readouts"] = _macro_frozen_readouts(summaries)
    _write_json(work_dir / "summary.json", aggregate)
    artifact_registry = _artifact_registry(run_dir=work_dir, datasets=summaries)
    _write_yaml(work_dir / "artifact_registry.yaml", artifact_registry)
    outputs = _portable_output_inventory(work_dir)
    manifest = {
        "schema_version": PILOT_MANIFEST_SCHEMA_VERSION,
        "project": PROJECT_NAME,
        "experiment_name": experiment_name,
        "run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "family": family,
        "config_sha256": config_sha,
        "source_tree_sha256": source_sha,
        "input_sha256": input_sha,
        "inputs": input_record,
        "environment": environment_state(),
        "selection_protocol": TRAIN_SELECTION_PROTOCOL,
        "selection_target_split": "train_selection",
        "selection_uses_lid_targets": True,
        "selection_uses_validation_targets": False,
        "selection_uses_test_targets": False,
        "evaluation_protocol": FROZEN_EVALUATION_PROTOCOL,
        "retrospective_evaluation_curves_saved": False,
        "outputs": outputs,
    }
    _write_json(work_dir / "manifest.json", manifest)
    errors = validate_pilot_experiment(work_dir)
    if errors:
        raise RuntimeError(f"new pilot run failed self-validation: {errors}")
    work_dir.replace(final_dir)
    uploaded_assets = {
        filename: _log_asset(
            log_callback,
            final_dir / filename,
            name=f"{PROJECT_NAME}-{filename}",
        )
        for filename in (
            "summary.json",
            "manifest.json",
            "resolved_config.yaml",
        )
    }
    _emit(
        log_callback,
        "experiment.completed",
        experiment_name=experiment_name,
        family=family,
        run_id=run_id,
        macro_train_selection=aggregate["macro_train_selection"],
        macro_validation=aggregate["macro_validation"],
        macro_test=aggregate["macro_test"],
        **(
            {"macro_frozen_readouts": aggregate["macro_frozen_readouts"]}
            if family == "independent_affine_flow"
            else {}
        ),
        shared_filesystem_run_dir=str(final_dir),
        summary_path=str(final_dir / "summary.json"),
        manifest_path=str(final_dir / "manifest.json"),
        resolved_config_path=str(final_dir / "resolved_config.yaml"),
        uploaded_assets=uploaded_assets,
    )
    return final_dir


def _safe_relative_path(value: Any) -> str | None:
    if not isinstance(value, str) or not value or "\\" in value:
        return None
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        return None
    return value


def _load_numeric_output(path: Path) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    if not isinstance(value, np.ndarray) or not np.issubdtype(value.dtype, np.number):
        raise ValueError(f"{path.name} must contain a numeric numpy array")
    return value


def _affine_diagnostic_binding_errors(
    *,
    dataset_name: str,
    metadata: Mapping[str, Any],
    diagnostic_scales: npt.ArrayLike,
    diagnostic_target: npt.ArrayLike,
    diagnostic_full: npt.ArrayLike,
    diagnostic_response: npt.ArrayLike,
    diagnostic_correction: npt.ArrayLike,
    cell_arrays: Mapping[str, npt.ArrayLike],
    summary: Mapping[str, Any],
    checkpoint_path: Path,
) -> list[str]:
    """Bind a self-valid diagnostic tree to this exact trained cell."""

    errors: list[str] = []
    checkpoint_sha = sha256_path(checkpoint_path)
    scale_selection = summary.get("scale_selection")
    partition = (
        scale_selection.get("partition", {})
        if isinstance(scale_selection, Mapping)
        else {}
    )
    raw_query_sha = (
        partition.get("selection_features_sha256")
        if isinstance(partition, Mapping)
        else None
    )
    if (
        metadata.get("checkpoint_sha256") != checkpoint_sha
        or summary.get("checkpoint_sha256") != checkpoint_sha
    ):
        errors.append(f"{dataset_name}: FM diagnostics are not bound to checkpoint.pt")
    if metadata.get("raw_query_sha256") != raw_query_sha:
        errors.append(
            f"{dataset_name}: FM diagnostics are not bound to the train-selection rows"
        )
    outer_curve_sha = _array_sha256(cell_arrays["train_selection_curve"])
    if metadata.get("outer_selection_curve_sha256") != outer_curve_sha:
        errors.append(
            f"{dataset_name}: FM diagnostics are not cryptographically bound to "
            "the train-selection curve"
        )
    if not np.array_equal(diagnostic_scales, cell_arrays["scales"]):
        errors.append(f"{dataset_name}: FM diagnostic scales differ from selection")
    if not np.array_equal(diagnostic_target, cell_arrays["train_selection_target"]):
        errors.append(f"{dataset_name}: FM diagnostic targets differ from selection")
    curve_error = _affine_outer_curve_consistency_error(
        diagnostic_full=diagnostic_full,
        diagnostic_response=diagnostic_response,
        diagnostic_correction=diagnostic_correction,
        outer_selection_curve=cell_arrays["train_selection_curve"],
    )
    if curve_error is not None:
        errors.append(f"{dataset_name}: {curve_error}")
    return errors


def _validate_selection_cell(
    *,
    directory: Path,
    dataset_name: str,
    config: Mapping[str, Any],
    source_train: npt.ArrayLike | None = None,
    source_train_target: npt.ArrayLike | None = None,
) -> tuple[list[str], dict[str, Any] | None]:
    """Recompute a cell's selected index, predictions, metrics, and partition."""

    cell = directory / "datasets" / dataset_name
    errors: list[str] = []
    model = config["pilot_model"]
    try:
        summary_value = json.loads((cell / "summary.json").read_text("utf-8"))
        if not isinstance(summary_value, dict):
            raise TypeError("summary.json must contain a mapping")
        summary: dict[str, Any] = summary_value
        training_value = yaml.safe_load((cell / "training.yaml").read_text("utf-8"))
        if not isinstance(training_value, dict):
            raise TypeError("training.yaml must contain a mapping")
        arrays = {
            name: _load_numeric_output(cell / f"{name}.npy")
            for name in (
                "scales",
                "train_fit_indices",
                "train_selection_indices",
                "train_selection_curve",
                "train_selection_prediction",
                "train_selection_target",
                "validation_prediction",
                "validation_target",
                "test_prediction",
                "test_target",
            )
        }
        if model["family"] == "independent_affine_flow":
            for split in ("train_selection", "validation", "test"):
                for readout in _AFFINE_READOUTS:
                    key = f"{split}_prediction__{readout}"
                    arrays[key] = _load_numeric_output(cell / f"{key}.npy")
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
        yaml.YAMLError,
    ) as exc:
        return [f"{dataset_name}: invalid selection artifact: {exc}"], None

    for forbidden_path in tuple(cell.glob("validation_curve*.npy")) + tuple(
        cell.glob("test_curve*.npy")
    ):
        if forbidden_path.exists():
            errors.append(
                f"{dataset_name}: retrospective evaluation artifact is forbidden: "
                f"{forbidden_path.name}"
            )

    evaluation = config["evaluation"]
    selection = evaluation["selection"]
    if canonical_json(training_value.get("model")) != canonical_json(model):
        errors.append(f"{dataset_name}: training.yaml model differs from Hydra config")
    if model["family"] == "independent_affine_flow":
        summary_model = summary.get("model")
        model_contract = (
            summary_model.get("model_contract")
            if isinstance(summary_model, Mapping)
            else None
        )
        try:
            _validate_affine_contract(
                model_contract,
                training=model["training"],
            )
        except (RuntimeError, TypeError) as exc:
            errors.append(f"{dataset_name}: invalid affine-flow model contract: {exc}")
        diagnostics_dir = cell / "fm_diagnostics"
        try:
            from experiments.fm_diagnostics import validate_fm_diagnostics

            diagnostic_errors = validate_fm_diagnostics(diagnostics_dir)
            metadata_value = json.loads(
                (diagnostics_dir / "metadata.json").read_text(encoding="utf-8")
            )
            if not isinstance(metadata_value, dict):
                raise TypeError("FM diagnostics metadata must be a mapping")
            diagnostic_scales = _load_numeric_output(
                diagnostics_dir / "arrays" / "scales.npy"
            )
            diagnostic_target = _load_numeric_output(
                diagnostics_dir / "arrays" / "target.npy"
            )
            diagnostic_full = _load_numeric_output(
                diagnostics_dir / "arrays" / "full.npy"
            )
            diagnostic_response = _load_numeric_output(
                diagnostics_dir / "arrays" / "response.npy"
            )
            diagnostic_correction = _load_numeric_output(
                diagnostics_dir / "arrays" / "correction.npy"
            )
            checkpoint_sha = sha256_path(cell / "checkpoint.pt")
            scale_selection_record = summary.get("scale_selection")
            partition_record = (
                scale_selection_record.get("partition", {})
                if isinstance(scale_selection_record, Mapping)
                else {}
            )
            raw_query_sha = (
                partition_record.get("selection_features_sha256")
                if isinstance(partition_record, Mapping)
                else None
            )
            expected_diagnostic_record = {
                "schema_version": metadata_value.get("schema_version"),
                "path": "fm_diagnostics",
                "variant_id": model["training"]["flow_variant_id"],
                "source_split": "train_selection",
                "protocol": metadata_value.get("protocol"),
                "n_query": metadata_value.get("n_query"),
                "n_scales": metadata_value.get("n_scales"),
                "checkpoint_sha256": checkpoint_sha,
                "outer_selection_curve_sha256": _array_sha256(
                    arrays["train_selection_curve"]
                ),
                "raw_query_sha256": raw_query_sha,
                "manifest_sha256": sha256_path(diagnostics_dir / "manifest.json"),
                "metadata_sha256": sha256_path(diagnostics_dir / "metadata.json"),
                "summary_sha256": sha256_path(diagnostics_dir / "summary.json"),
            }
        except (
            OSError,
            TypeError,
            UnicodeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            errors.append(f"{dataset_name}: invalid FM diagnostics: {exc}")
        else:
            errors.extend(
                f"{dataset_name}: invalid FM diagnostics: {error}"
                for error in diagnostic_errors
            )
            if canonical_json(metadata_value.get("config")) != canonical_json(
                model["diagnostics"]
            ):
                errors.append(
                    f"{dataset_name}: FM diagnostic config differs from Hydra"
                )
            if metadata_value.get("variant_id") != model["training"]["flow_variant_id"]:
                errors.append(
                    f"{dataset_name}: FM diagnostic variant differs from Hydra"
                )
            errors.extend(
                _affine_diagnostic_binding_errors(
                    dataset_name=dataset_name,
                    metadata=metadata_value,
                    diagnostic_scales=diagnostic_scales,
                    diagnostic_target=diagnostic_target,
                    diagnostic_full=diagnostic_full,
                    diagnostic_response=diagnostic_response,
                    diagnostic_correction=diagnostic_correction,
                    cell_arrays=arrays,
                    summary=summary,
                    checkpoint_path=cell / "checkpoint.pt",
                )
            )
            if canonical_json(summary.get("fm_diagnostics")) != canonical_json(
                expected_diagnostic_record
            ):
                errors.append(
                    f"{dataset_name}: FM diagnostic attestation does not recompute"
                )
    else:
        if (cell / "fm_diagnostics").exists() or "fm_diagnostics" in summary:
            errors.append(
                f"{dataset_name}: FM diagnostics are reserved for affine flow"
            )
    scales = np.ravel(np.asarray(arrays["scales"], dtype=np.float64))
    configured_scales = np.asarray(model["scales"], dtype=np.float64)
    if not np.array_equal(scales, configured_scales):
        errors.append(f"{dataset_name}: scales.npy differs from resolved Hydra config")

    fit_indices = np.asarray(arrays["train_fit_indices"])
    selection_indices = np.asarray(arrays["train_selection_indices"])
    if fit_indices.ndim != 1 or not np.issubdtype(fit_indices.dtype, np.integer):
        errors.append(f"{dataset_name}: train_fit_indices.npy must be integer rank-1")
    if selection_indices.ndim != 1 or not np.issubdtype(
        selection_indices.dtype, np.integer
    ):
        errors.append(
            f"{dataset_name}: train_selection_indices.npy must be integer rank-1"
        )
    n_source = int(
        summary.get("scale_selection", {})
        .get("partition", {})
        .get("n_source_train", -1)
    )
    if n_source > 0 and fit_indices.ndim == selection_indices.ndim == 1:
        expected_selection = _deterministic_selection_indices(
            n_source,
            subset_size=int(selection["subset_size"]),
            seed=int(selection["seed"]),
        )
        expected_fit = np.setdiff1d(
            np.arange(n_source, dtype=np.int64),
            expected_selection,
            assume_unique=True,
        )
        if not np.array_equal(selection_indices, expected_selection):
            errors.append(
                f"{dataset_name}: train-selection indices are not reproducible"
            )
        if not np.array_equal(fit_indices, expected_fit):
            errors.append(
                f"{dataset_name}: optimizer-fit indices are not the complement"
            )

    partition = summary.get("scale_selection", {}).get("partition")
    if not isinstance(partition, dict):
        errors.append(f"{dataset_name}: missing scale_selection.partition")
        partition = {}
    else:
        recorded_partition_sha = partition.get("partition_sha256")
        unhashed_partition = {
            key: value for key, value in partition.items() if key != "partition_sha256"
        }
        expected_partition_sha = sha256_bytes(
            canonical_json(unhashed_partition).encode("utf-8")
        )
        if recorded_partition_sha != expected_partition_sha:
            errors.append(f"{dataset_name}: train partition SHA is inconsistent")
        if partition.get("fit_indices_sha256") != _array_sha256(fit_indices):
            errors.append(f"{dataset_name}: fit-indices SHA is inconsistent")
        if partition.get("selection_indices_sha256") != _array_sha256(
            selection_indices
        ):
            errors.append(f"{dataset_name}: selection-indices SHA is inconsistent")
        if partition.get("selection_target_sha256") != _array_sha256(
            arrays["train_selection_target"]
        ):
            errors.append(f"{dataset_name}: train-selection target SHA is inconsistent")
        training_partition = training_value.get("dataset", {}).get("train_partition")
        if training_partition != partition:
            errors.append(
                f"{dataset_name}: training.yaml train partition differs from summary"
            )

    if source_train is not None and source_train_target is not None:
        try:
            expected_partition = _partition_source_train(
                source_train,
                source_train_target,
                selection=selection,
            )
        except (
            AssertionError,
            FloatingPointError,
            PilotConfigError,
            ValueError,
        ) as exc:
            errors.append(f"{dataset_name}: cannot reconstruct train partition: {exc}")
        else:
            if dict(expected_partition.record) != partition:
                errors.append(
                    f"{dataset_name}: train partition differs from source data"
                )

    selection_curve = np.asarray(arrays["train_selection_curve"], dtype=np.float64)
    selection_target = np.ravel(
        np.asarray(arrays["train_selection_target"], dtype=np.float64)
    )
    try:
        selected_index, raw_diagnostics = _select_by_train_targets(
            scales=scales,
            curve=selection_curve,
            target=selection_target,
            criterion=str(selection["criterion"]),
            tie_tolerance=float(selection["tie_tolerance"]),
            tie_break=str(selection["tie_break"]),
        )
        coordinates, coordinate_name, coordinate_formula, model_scale_name = (
            _selection_coordinate(scales, family=str(model["family"]))
        )
        model_prefer = str(model["selection_prefer"])
        coordinate_prefer = model_prefer
        if model["family"] == "rectified_flow":
            coordinate_prefer = "smaller" if model_prefer == "larger" else "larger"
        expected_diagnostics = _reported_selection_diagnostics(
            raw_diagnostics,
            scales=scales,
            coordinates=coordinates,
            selected_index=selected_index,
            coordinate_name=coordinate_name,
            coordinate_formula=coordinate_formula,
            model_scale_name=model_scale_name,
            coordinate_prefer=coordinate_prefer,
            model_scale_prefer=model_prefer,
            model_training=model["training"],
        )
        expected_diagnostics["partition"] = partition
        if canonical_json(summary.get("scale_selection")) != canonical_json(
            expected_diagnostics
        ):
            errors.append(
                f"{dataset_name}: scale-selection diagnostics do not recompute"
            )
    except (FloatingPointError, KeyError, TypeError, ValueError) as exc:
        errors.append(f"{dataset_name}: cannot recompute scale selection: {exc}")
        selected_index = -1

    if 0 <= selected_index < scales.size:
        train_prediction = np.ravel(
            np.asarray(arrays["train_selection_prediction"], dtype=np.float64)
        )
        if selection_curve.ndim != 2 or selection_curve.shape != (
            selection_target.size,
            scales.size,
        ):
            errors.append(f"{dataset_name}: invalid train_selection curve shape")
        elif not np.array_equal(train_prediction, selection_curve[:, selected_index]):
            errors.append(
                f"{dataset_name}: train_selection prediction is not the frozen "
                "curve column"
            )
        else:
            metric = known_lid_metrics(train_prediction, selection_target)
            if canonical_json(summary.get("train_selection")) != canonical_json(metric):
                errors.append(
                    f"{dataset_name}: train_selection metrics do not recompute"
                )

        expected_frozen_evaluation = {
            "schema_version": 1,
            "selected_index": selected_index,
            "selected_scale": float(scales[selected_index]),
            "validation_candidate_count": 1,
            "test_candidate_count": 1,
            "retrospective_curves_saved": False,
        }
        if summary.get("evaluation_protocol") != FROZEN_EVALUATION_PROTOCOL:
            errors.append(f"{dataset_name}: invalid frozen evaluation protocol")
        if canonical_json(summary.get("frozen_evaluation")) != canonical_json(
            expected_frozen_evaluation
        ):
            errors.append(
                f"{dataset_name}: frozen evaluation attestation does not recompute"
            )

        for split in ("validation", "test"):
            prediction = np.ravel(
                np.asarray(arrays[f"{split}_prediction"], dtype=np.float64)
            )
            target = np.ravel(np.asarray(arrays[f"{split}_target"], dtype=np.float64))
            if prediction.shape != target.shape:
                errors.append(f"{dataset_name}: invalid {split} prediction shape")
                continue
            if not np.isfinite(prediction).all() or not np.isfinite(target).all():
                errors.append(f"{dataset_name}: non-finite {split} output")
                continue
            metric = known_lid_metrics(prediction, target)
            if canonical_json(summary.get(split)) != canonical_json(metric):
                errors.append(f"{dataset_name}: {split} metrics do not recompute")

        if model["family"] == "independent_affine_flow":
            expected_readout_metrics: dict[str, Any] = {}
            target_by_split = {
                "train_selection": selection_target,
                "validation": np.ravel(
                    np.asarray(arrays["validation_target"], dtype=np.float64)
                ),
                "test": np.ravel(np.asarray(arrays["test_target"], dtype=np.float64)),
            }
            primary_by_split = {
                "train_selection": np.ravel(
                    np.asarray(arrays["train_selection_prediction"], dtype=np.float64)
                ),
                "validation": np.ravel(
                    np.asarray(arrays["validation_prediction"], dtype=np.float64)
                ),
                "test": np.ravel(
                    np.asarray(arrays["test_prediction"], dtype=np.float64)
                ),
            }
            for readout in _AFFINE_READOUTS:
                expected_readout_metrics[readout] = {}
                for split, target in target_by_split.items():
                    prediction = np.ravel(
                        np.asarray(
                            arrays[f"{split}_prediction__{readout}"],
                            dtype=np.float64,
                        )
                    )
                    if prediction.shape != target.shape:
                        errors.append(
                            f"{dataset_name}: invalid {split} {readout} "
                            "frozen prediction shape"
                        )
                        continue
                    if not np.isfinite(prediction).all():
                        errors.append(
                            f"{dataset_name}: non-finite {split} {readout} output"
                        )
                        continue
                    if readout == "full" and not np.array_equal(
                        prediction, primary_by_split[split]
                    ):
                        errors.append(
                            f"{dataset_name}: frozen full {split} output differs "
                            "from primary prediction"
                        )
                    expected_readout_metrics[readout][split] = known_lid_metrics(
                        prediction, target
                    )
            expected_frozen_readouts = {
                "schema_version": 1,
                "scale_semantics": "noise_ratio_lambda=beta/alpha",
                "selected_lambda": float(scales[selected_index]),
                "primary_readout": "full",
                "scale_selected_with_readout": "full",
                "validation_candidate_count_per_readout": 1,
                "test_candidate_count_per_readout": 1,
                "retrospective_validation_curves_saved": False,
                "retrospective_test_curves_saved": False,
                "readouts": expected_readout_metrics,
            }
            if canonical_json(summary.get("frozen_readouts")) != canonical_json(
                expected_frozen_readouts
            ):
                errors.append(
                    f"{dataset_name}: frozen affine readout metrics do not recompute"
                )
        elif "frozen_readouts" in summary:
            errors.append(
                f"{dataset_name}: frozen_readouts is reserved for affine flow"
            )

    expected_attestation = {
        "selection_protocol": TRAIN_SELECTION_PROTOCOL,
        "selection_target_split": "train_selection",
        "selection_uses_lid_targets": True,
        "selection_uses_validation_targets": False,
        "selection_uses_test_targets": False,
    }
    for key, expected in expected_attestation.items():
        if summary.get(key) != expected:
            errors.append(f"{dataset_name}: invalid {key} attestation")
    return errors, summary


def validate_pilot_experiment(
    run_dir: Path,
    *,
    verify_inputs: bool = False,
    verify_source_tree: bool = False,
    root: Path | None = None,
) -> list[str]:
    """Recompute the sealed artifact inventory and optional source identities."""

    directory = Path(run_dir)
    manifest_path = directory / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"invalid manifest.json: {exc}"]
    if not isinstance(manifest, dict):
        return ["manifest.json must contain a mapping"]
    errors: list[str] = []
    required = {
        "schema_version",
        "project",
        "experiment_name",
        "run_id",
        "created_at_utc",
        "family",
        "config_sha256",
        "source_tree_sha256",
        "input_sha256",
        "inputs",
        "environment",
        "selection_protocol",
        "selection_target_split",
        "selection_uses_lid_targets",
        "selection_uses_validation_targets",
        "selection_uses_test_targets",
        "evaluation_protocol",
        "retrospective_evaluation_curves_saved",
        "outputs",
    }
    if set(manifest) != required:
        errors.append(
            "manifest fields mismatch: "
            f"missing={sorted(required - set(manifest))}, "
            f"unknown={sorted(set(manifest) - required)}"
        )
    if manifest.get("schema_version") != PILOT_MANIFEST_SCHEMA_VERSION:
        errors.append("unsupported pilot manifest schema_version")
    if manifest.get("project") != PROJECT_NAME:
        errors.append(f"manifest.project is not {PROJECT_NAME!r}")
    if manifest.get("selection_protocol") != TRAIN_SELECTION_PROTOCOL:
        errors.append("manifest has an unsupported train-selection protocol")
    if manifest.get("selection_target_split") != "train_selection":
        errors.append("manifest selection target must be train_selection")
    if manifest.get("selection_uses_lid_targets") is not True:
        errors.append("manifest must attest supervised train-target selection")
    if manifest.get("selection_uses_validation_targets") is not False:
        errors.append("manifest must attest no validation-target selection")
    if manifest.get("selection_uses_test_targets") is not False:
        errors.append("manifest must attest no test-target selection")
    if manifest.get("evaluation_protocol") != FROZEN_EVALUATION_PROTOCOL:
        errors.append("manifest has an unsupported frozen evaluation protocol")
    if manifest.get("retrospective_evaluation_curves_saved") is not False:
        errors.append("manifest must forbid retrospective evaluation curves")

    resolved_path = directory / "resolved_config.yaml"
    try:
        resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
        resolved = validate_pilot_config(resolved)
    except (
        KeyError,
        OSError,
        PilotConfigError,
        TypeError,
        UnicodeError,
        ValueError,
        yaml.YAMLError,
    ) as exc:
        errors.append(f"invalid resolved_config.yaml: {exc}")
        resolved = None
    if resolved is not None:
        experiment_name = resolved["experiment_name"]
        family = resolved["pilot_model"]["family"]
        if manifest.get("experiment_name") != experiment_name:
            errors.append(
                "manifest.experiment_name does not match resolved_config.yaml"
            )
        if manifest.get("family") != family:
            errors.append("manifest.family does not match resolved_config.yaml")
        actual_config_sha = sha256_bytes(canonical_json(resolved).encode("utf-8"))
        if manifest.get("config_sha256") != actual_config_sha:
            errors.append("config_sha256 does not match resolved_config.yaml")
        identity = {
            "schema_version": 1,
            "project": PROJECT_NAME,
            "experiment_name": experiment_name,
            "family": family,
            "config_sha256": actual_config_sha,
            "source_tree_sha256": manifest.get("source_tree_sha256"),
            "input_sha256": manifest.get("input_sha256"),
        }
        expected_run_id = sha256_bytes(canonical_json(identity).encode("utf-8"))[:20]
        if manifest.get("run_id") != expected_run_id:
            errors.append("run_id is inconsistent with scientific identity")

    verified_loaded: Mapping[str, Mapping[str, LoadedSplit]] | None = None
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        errors.append("manifest.inputs must be a mapping")
    else:
        if manifest.get("input_sha256") != _input_identity(inputs):
            errors.append("input_sha256 is inconsistent with inputs")
        if verify_inputs and resolved is not None:
            project_root = (repository_root() if root is None else Path(root)).resolve()
            benchmark_root = _resolve_path(project_root, resolved["data"]["root"])
            registry_record = inputs.get("registry")
            if isinstance(registry_record, Mapping):
                registry_path = _resolve_path(
                    project_root, resolved["data"]["registry"]
                )
                if not registry_path.is_file() or sha256_path(
                    registry_path
                ) != registry_record.get("sha256"):
                    errors.append("registry input changed or is missing")
            dataset_records = inputs.get("datasets")
            if isinstance(dataset_records, Mapping):
                for dataset_name, dataset_record in dataset_records.items():
                    if not isinstance(dataset_record, Mapping):
                        errors.append(f"invalid input record for {dataset_name}")
                        continue
                    source_files = dataset_record.get("source_files")
                    if not isinstance(source_files, Mapping):
                        errors.append(f"missing source_files for {dataset_name}")
                        continue
                    for record in source_files.values():
                        if not isinstance(record, Mapping):
                            errors.append(
                                f"invalid source file record for {dataset_name}"
                            )
                            continue
                        relative = _safe_relative_path(record.get("path"))
                        if relative is None:
                            errors.append(f"unsafe source path for {dataset_name}")
                            continue
                        path = benchmark_root / relative
                        if not path.is_file() or sha256_path(path) != record.get(
                            "sha256"
                        ):
                            errors.append(
                                f"dataset input changed or missing: {relative}"
                            )
            try:
                _, _, _, reconstructed_loaded, reconstructed_inputs = _load_inputs(
                    root=project_root, config=resolved
                )
            except (KeyError, OSError, PilotConfigError, TypeError, ValueError) as exc:
                errors.append(f"cannot reconstruct pilot inputs: {exc}")
            else:
                verified_loaded = reconstructed_loaded
                if reconstructed_inputs != inputs:
                    errors.append(
                        "manifest inputs differ from reconstructed source inputs"
                    )

    cell_summaries: dict[str, Any] = {}
    if resolved is not None:
        for dataset_name in PILOT_DATASETS:
            source_train: npt.ArrayLike | None = None
            source_train_target: npt.ArrayLike | None = None
            if verified_loaded is not None:
                source_split = verified_loaded[dataset_name]["train"]
                source_train = _flatten_features(source_split)
                source_train_target = source_split.lid
            cell_errors, cell_summary = _validate_selection_cell(
                directory=directory,
                dataset_name=dataset_name,
                config=resolved,
                source_train=source_train,
                source_train_target=source_train_target,
            )
            errors.extend(cell_errors)
            if cell_summary is not None:
                cell_summaries[dataset_name] = cell_summary
        try:
            aggregate_value = json.loads(
                (directory / "summary.json").read_text(encoding="utf-8")
            )
            if not isinstance(aggregate_value, dict):
                raise TypeError("summary.json must contain a mapping")
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ) as exc:
            errors.append(f"invalid aggregate summary.json: {exc}")
        else:
            if aggregate_value.get("datasets") != cell_summaries:
                errors.append("aggregate datasets differ from per-dataset summaries")
            for split, field in (
                ("train_selection", "macro_train_selection"),
                ("validation", "macro_validation"),
                ("test", "macro_test"),
            ):
                expected_macro = _macro_metrics(cell_summaries, split)
                if aggregate_value.get(field) != expected_macro:
                    errors.append(f"{field} does not recompute from dataset summaries")
            if resolved["pilot_model"]["family"] == "independent_affine_flow":
                try:
                    expected_readouts = _macro_frozen_readouts(cell_summaries)
                except (KeyError, TypeError, ValueError) as exc:
                    errors.append(f"macro_frozen_readouts cannot be recomputed: {exc}")
                else:
                    if (
                        aggregate_value.get("macro_frozen_readouts")
                        != expected_readouts
                    ):
                        errors.append(
                            "macro_frozen_readouts does not recompute from datasets"
                        )
            elif "macro_frozen_readouts" in aggregate_value:
                errors.append(
                    "aggregate macro_frozen_readouts is reserved for affine flow"
                )
            for key, expected in {
                "selection_protocol": TRAIN_SELECTION_PROTOCOL,
                "selection_target_split": "train_selection",
                "selection_uses_lid_targets": True,
                "selection_uses_validation_targets": False,
                "selection_uses_test_targets": False,
                "evaluation_protocol": FROZEN_EVALUATION_PROTOCOL,
                "retrospective_evaluation_curves_saved": False,
            }.items():
                if aggregate_value.get(key) != expected:
                    errors.append(f"aggregate summary has invalid {key}")

    recorded_outputs = manifest.get("outputs")
    if not isinstance(recorded_outputs, dict):
        errors.append("manifest.outputs must be a mapping")
    else:
        try:
            actual_outputs = _portable_output_inventory(directory)
        except ValueError as exc:
            errors.append(str(exc))
        else:
            if actual_outputs != recorded_outputs:
                errors.append("output inventory does not match manifest")
    if verify_source_tree:
        project_root = (repository_root() if root is None else Path(root)).resolve()
        if hash_declared_sources(project_root) != manifest.get("source_tree_sha256"):
            errors.append("source tree changed since pilot execution")
    return errors


def _logging_callback(
    config: Mapping[str, Any],
) -> tuple[LogCallback | None, Callable[[], None]]:
    """Create the optional external logger without ever accepting a key in YAML."""

    if config["logging"]["backend"] == "none":
        return None, lambda: None
    try:
        from experiments.comet_logging import create_comet_callback
    except ImportError as exc:
        raise RuntimeError(
            "Comet logging is configured but experiments.comet_logging is unavailable"
        ) from exc
    return create_comet_callback(
        experiment_name=str(config["logging"]["experiment_name"]),
        tags=tuple(
            dict.fromkeys(
                (
                    str(config["pilot_model"]["family"]),
                    str(config["pilot_model"]["name"]),
                )
            )
        ),
    )


@hydra.main(version_base="1.3", config_path=None, config_name="pilot")
def _hydra_main(config: DictConfig) -> None:
    resolved = validate_pilot_config(config)
    callback, close = _logging_callback(resolved)
    try:
        output = run_pilot(config, log_callback=callback)
    finally:
        close()
    print(output)


def main() -> None:
    from experiments.cli import _default_config_dir

    has_config_dir = any(
        argument == "--config-dir" or argument.startswith("--config-dir=")
        for argument in sys.argv[1:]
    )
    if not has_config_dir:
        sys.argv[1:1] = ["--config-dir", str(_default_config_dir())]
    _hydra_main()


if __name__ == "__main__":
    main()


__all__ = [
    "PILOT_DATASETS",
    "PROJECT_NAME",
    "PilotConfigError",
    "compose_pilot_config",
    "run_pilot",
    "validate_pilot_config",
    "validate_pilot_experiment",
]
