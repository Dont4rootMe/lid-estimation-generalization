"""Train and evaluate one generative-model family on the three E8 datasets.

One pilot run owns three independently trained models: one for Gaussian4, one
for Spaghetti and one for Sphere4.  The validation LID labels are deliberately
kept out of both training and scale selection.  They are read only after the
label-free plateau selector has chosen a scale and are then used, together with
the test labels, to report auditable metrics.

The public :func:`run_pilot` function accepts injectable training, prediction
and logging callables.  Production uses :mod:`models.training`; tests can use a
small deterministic implementation without weakening the artifact contract.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Protocol

import hydra
import numpy as np
import numpy.typing as npt
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
import yaml

from datasets.registry import DatasetRegistry, LoadedSplit, load_dataset, load_registry
from experiments.metrics import known_lid_metrics
from experiments.run_manifest import (
    canonical_json,
    environment_state,
    hash_declared_sources,
    sha256_bytes,
    sha256_path,
)
from models.oracle import select_stable_scale
from utils.provenance import sha256_file


PROJECT_NAME = "lid-generalization"
EXPERIMENT_NAME = "ent-block-diffusion-eval"
WORKSPACE_NAME = "dont4rootme"
PILOT_DATASETS = (
    "e8_gaussian4_pca",
    "e8_spaghetti_pca",
    "e8_sphere4_pca",
)
PILOT_MANIFEST_SCHEMA_VERSION = 1
_FAMILY_FOR_ARTIFACTS = {
    "diffusion": "gaussian_diffusion",
    "gaussian_diffusion": "gaussian_diffusion",
    "rectified_flow": "rectified_flow",
}
_TRAINING_FIELDS = {
    "seed",
    "device",
    "epochs",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "hidden_dim",
    "depth",
    "time_embedding_dim",
    "validation_interval",
    "early_stopping_patience",
    "gradient_clip_norm",
    "num_workers",
    "deterministic",
    "sigma_min",
    "sigma_max",
    "time_min",
    "time_max",
    "fourier_features",
    "max_condition_frequency",
    "dropout",
    "normalize",
    "normalization_epsilon",
}

FloatArray = npt.NDArray[np.float64]
LogCallback = Callable[[str, Mapping[str, Any]], None]


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

    project_root = repository_root() if root is None else Path(root)
    with initialize_config_dir(
        version_base="1.3",
        config_dir=str((project_root / "configs").resolve()),
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


def _reject_unknown(
    value: Mapping[str, Any], allowed: set[str], *, field: str
) -> None:
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
    required_identity = {
        "project": PROJECT_NAME,
        "experiment_name": EXPERIMENT_NAME,
    }
    for field, expected in required_identity.items():
        if value.get(field) != expected:
            raise PilotConfigError(f"{field} must be exactly {expected!r}")
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
    if (
        not isinstance(registry, str)
        or not registry.endswith(".yaml")
        or not registry
    ):
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
    if isinstance(trace_seed, bool) or not isinstance(trace_seed, int) or trace_seed < 0:
        raise PilotConfigError("evaluation.trace_seed must be non-negative")
    selection = evaluation.get("selection")
    if not isinstance(selection, dict):
        raise PilotConfigError("evaluation.selection must be a mapping")
    _reject_unknown(
        selection,
        {"window", "min_valid_fraction"},
        field="evaluation.selection",
    )
    window = _positive_int(selection.get("window"), field="selection.window")
    min_fraction = selection.get("min_valid_fraction")
    if isinstance(min_fraction, bool) or not isinstance(min_fraction, (int, float)):
        raise PilotConfigError("selection.min_valid_fraction must be numeric")
    if not math.isfinite(float(min_fraction)) or not 0 < float(min_fraction) <= 1:
        raise PilotConfigError("selection.min_valid_fraction must lie in (0, 1]")
    model = value.get("pilot_model")
    if not isinstance(model, dict):
        raise PilotConfigError("pilot_model must be a mapping")
    _reject_unknown(
        model,
        {
            "name",
            "family",
            "readout",
            "selection_prefer",
            "training",
            "scales",
        },
        field="pilot_model",
    )
    family = model.get("family")
    if family not in _FAMILY_FOR_ARTIFACTS:
        raise PilotConfigError(
            "pilot_model.family must be diffusion/gaussian_diffusion or "
            "rectified_flow"
        )
    if not isinstance(model.get("name"), str) or not model["name"]:
        raise PilotConfigError("pilot_model.name must be non-empty")
    if model.get("readout") not in {"full", "response"}:
        raise PilotConfigError("pilot_model.readout must be 'full' or 'response'")
    if model.get("selection_prefer") not in {"smaller", "larger"}:
        raise PilotConfigError(
            "pilot_model.selection_prefer must be 'smaller' or 'larger'"
        )
    if not isinstance(model.get("training"), dict) or not model["training"]:
        raise PilotConfigError("pilot_model.training must be a non-empty mapping")
    _reject_unknown(
        model["training"], _TRAINING_FIELDS, field="pilot_model.training"
    )
    missing_training_fields = _TRAINING_FIELDS - set(model["training"])
    if missing_training_fields:
        raise PilotConfigError(
            "missing pilot_model.training fields: "
            f"{sorted(missing_training_fields)}"
        )
    scales = np.asarray(model.get("scales"), dtype=np.float64)
    if (
        scales.ndim != 1
        or scales.size < 2 * window + 1
        or not np.isfinite(scales).all()
        or np.any(scales <= 0)
        or np.unique(scales).size != scales.size
    ):
        raise PilotConfigError(
            "pilot_model.scales must contain enough unique finite positive values"
        )
    if family == "rectified_flow" and np.any(scales >= 1):
        raise PilotConfigError("rectified-flow scales must lie strictly in (0, 1)")

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
    for field, expected in required_identity.items():
        if logging.get(field) != expected:
            raise PilotConfigError(
                f"logging.{field} must be exactly {expected!r}"
            )
    if logging.get("workspace") != WORKSPACE_NAME:
        raise PilotConfigError(
            f"logging.workspace must be exactly {WORKSPACE_NAME!r}"
        )
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


def _load_inputs(
    *, root: Path, config: Mapping[str, Any]
) -> tuple[DatasetRegistry, Path, Path, dict[str, Mapping[str, LoadedSplit]], dict[str, Any]]:
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
        if any(splits[split].lid is None for split in ("val", "test")):
            raise PilotConfigError(f"pilot dataset {name!r} needs val/test LID labels")
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
    family: str,
    dataset: str | None = None,
    **payload: Any,
) -> None:
    if callback is None:
        return
    record: dict[str, Any] = {
        "project": PROJECT_NAME,
        "experiment_name": EXPERIMENT_NAME,
        "family": family,
    }
    if dataset is not None:
        record["dataset"] = dataset
    record.update(_strict_json_value(payload))
    callback(event, record)


def _log_asset(
    callback: LogCallback | None, path: Path, *, name: str
) -> bool:
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
    columns: list[FloatArray] = []
    n_samples = int(np.asarray(query).shape[0])
    for scale in scales:
        prediction = np.ravel(
            np.asarray(
                predict_fn(
                    trained,
                    query,
                    float(scale),
                    family=family,
                    readout=str(model["readout"]),
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
        columns.append(prediction)
    return np.ascontiguousarray(np.column_stack(columns), dtype=np.float64)


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


def _selection_coordinate(
    scales: FloatArray, *, family: str
) -> tuple[FloatArray, str, str, str]:
    """Return the coordinate in which plateau stability is measured.

    Rectified-flow networks are evaluated at time ``t``, while the Gaussian
    channel scale in the endpoint identity is ``lambda = (1 - t) / t``.  The
    selector must therefore differentiate the validation curve with respect to
    log-lambda, not log-t.  Diffusion already uses its native noise scale.
    """

    if family == "rectified_flow":
        coordinate = (1.0 - scales) / scales
        return (
            np.ascontiguousarray(coordinate, dtype=np.float64),
            "lambda",
            "(1 - t) / t",
            "t",
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
) -> dict[str, Any]:
    """Make the selector coordinate and original model parameter unambiguous."""

    result = dict(diagnostics)
    selected_coordinate = float(coordinates[selected_index])
    selected_model_scale = float(scales[selected_index])
    # ``select_stable_scale`` calls its input a scale.  Preserve that value in
    # the explicit coordinate record, then expose the actual network parameter
    # under the long-standing ``selected_scale`` field.
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
    return result


def _training_result_record(result: Any) -> dict[str, Any]:
    fields = {
        "family": getattr(result, "family", None),
        "best_epoch": getattr(result, "best_epoch", None),
        "best_validation_loss": getattr(result, "best_validation_loss", None),
        "metrics": getattr(result, "metrics", {}),
        "internal_preprocessing": getattr(result, "preprocessing", {}),
        "internal_preprocessing_sha256": getattr(
            result, "preprocessing_sha256", None
        ),
    }
    return _strict_json_value(fields)


def _macro_metrics(dataset_summaries: Mapping[str, Any], split: str) -> dict[str, float]:
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


def _artifact_registry(
    *, run_dir: Path, datasets: Mapping[str, Any]
) -> dict[str, Any]:
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
    log_callback: LogCallback | None,
) -> dict[str, Any]:
    model = config["pilot_model"]
    evaluation = config["evaluation"]
    family = str(model["family"])
    train = _flatten_features(splits["train"])
    validation = _flatten_features(splits["val"])
    test = _flatten_features(splits["test"])
    validation_target = np.ravel(np.asarray(splits["val"].lid, dtype=np.float64))
    test_target = np.ravel(np.asarray(splits["test"].lid, dtype=np.float64))
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
            family=family,
            dataset=name,
            **trainer_payload,
        )

    _emit(
        log_callback,
        f"dataset.{name}.started",
        family=family,
        dataset=name,
    )
    trained = train_fn(
        family,
        train,
        validation,
        dict(model["training"]),
        checkpoint_path,
        training_log if log_callback is not None else None,
    )
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size <= 0:
        raise RuntimeError(f"trainer did not write a checkpoint: {checkpoint_path}")
    declared_checkpoint_sha = getattr(trained, "checkpoint_sha256", None)
    checkpoint_sha = sha256_path(checkpoint_path)
    if declared_checkpoint_sha is not None and declared_checkpoint_sha != checkpoint_sha:
        raise RuntimeError("TrainingResult checkpoint SHA does not match checkpoint file")
    actual_family = getattr(trained, "family", None)
    if actual_family != _FAMILY_FOR_ARTIFACTS[family]:
        raise RuntimeError(
            "TrainingResult family mismatch: "
            f"expected {_FAMILY_FOR_ARTIFACTS[family]!r}, got {actual_family!r}"
        )

    scales = np.asarray(model["scales"], dtype=np.float64)
    validation_curve = _prediction_curve(
        predict_fn=predict_fn,
        trained=trained,
        query=validation,
        scales=scales,
        family=family,
        model=model,
        evaluation=evaluation,
    )
    _require_all_finite(validation_curve, label="validation prediction curve")
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
        coordinate_prefer = (
            "smaller" if model_scale_prefer == "larger" else "larger"
        )
    selected_index, raw_selection_diagnostics = select_stable_scale(
        selection_coordinates,
        validation_curve,
        window=int(selection["window"]),
        min_valid_fraction=float(selection["min_valid_fraction"]),
        prefer=coordinate_prefer,  # type: ignore[arg-type]
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
    )
    # Test inference happens only after label-free selection has completed.
    test_curve = _prediction_curve(
        predict_fn=predict_fn,
        trained=trained,
        query=test,
        scales=scales,
        family=family,
        model=model,
        evaluation=evaluation,
    )
    _require_all_finite(test_curve, label="test prediction curve")
    validation_prediction = validation_curve[:, selected_index]
    test_prediction = test_curve[:, selected_index]
    _require_all_finite(
        validation_prediction, label="selected validation prediction"
    )
    _require_all_finite(test_prediction, label="selected test prediction")
    validation_metrics = known_lid_metrics(validation_prediction, validation_target)
    test_metrics = known_lid_metrics(test_prediction, test_target)

    _save_npy(cell_dir / "scales.npy", scales)
    _save_npy(cell_dir / "validation_curve.npy", validation_curve)
    _save_npy(cell_dir / "test_curve.npy", test_curve)
    _save_npy(cell_dir / "validation_prediction.npy", validation_prediction)
    _save_npy(cell_dir / "test_prediction.npy", test_prediction)
    _save_npy(cell_dir / "validation_target.npy", validation_target)
    _save_npy(cell_dir / "test_target.npy", test_target)
    history = getattr(trained, "history", [])
    _write_json(cell_dir / "training_history.json", history)

    preprocessing_spec = {"kind": "identity"}
    preprocessing_sha = sha256_bytes(
        canonical_json(preprocessing_spec).encode("utf-8")
    )
    training_config = {
        "schema_version": 1,
        "project": PROJECT_NAME,
        "experiment_name": EXPERIMENT_NAME,
        "model": dict(model),
        "dataset": {
            "name": name,
            "representation": "dataset",
            "feature_shape": input_record["feature_shape"],
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
            "training_dataset_sha256": input_record[
                "training_dataset_sha256"
            ],
            "preprocessing_sha256": preprocessing_sha,
        },
    }
    _write_yaml(cell_dir / "training.yaml", training_config)
    summary = {
        "schema_version": 1,
        "project": PROJECT_NAME,
        "experiment_name": EXPERIMENT_NAME,
        "dataset": name,
        "representation": "dataset",
        "model": _training_result_record(trained),
        "checkpoint_sha256": checkpoint_sha,
        "training_dataset_sha256": input_record["training_dataset_sha256"],
        "preprocessing_sha256": preprocessing_sha,
        "selection_uses_lid_targets": False,
        "scale_selection": selection_diagnostics,
        "validation": validation_metrics,
        "test": test_metrics,
    }
    _write_json(cell_dir / "summary.json", summary)
    _emit(
        log_callback,
        f"dataset.{name}.completed",
        family=family,
        dataset=name,
        selected_scale=float(scales[selected_index]),
        selection_coordinate=selection_diagnostics["selection_coordinate"],
        model_scale=selection_diagnostics["model_scale"],
        validation=validation_metrics,
        test=test_metrics,
    )
    return summary


def run_pilot(
    hydra_config: DictConfig | Mapping[str, Any],
    *,
    root: Path | None = None,
    output_root: Path | None = None,
    train_fn: TrainFunction | None = None,
    predict_fn: PredictFunction | None = None,
    log_callback: LogCallback | None = None,
) -> Path:
    """Train three dataset-specific models and write a sealed experiment."""

    config = validate_pilot_config(hydra_config)
    project_root = (repository_root() if root is None else Path(root)).resolve()
    if train_fn is None or predict_fn is None:
        from models.training import predict_lid, train_model

        train_fn = train_model if train_fn is None else train_fn
        predict_fn = predict_lid if predict_fn is None else predict_fn
    _, _, _, loaded, input_record = _load_inputs(root=project_root, config=config)
    input_sha = _input_identity(input_record)
    config_sha = sha256_bytes(canonical_json(config).encode("utf-8"))
    source_sha = hash_declared_sources(project_root)
    identity = {
        "schema_version": 1,
        "project": PROJECT_NAME,
        "experiment_name": EXPERIMENT_NAME,
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
    final_dir = selected_output / f"{family}__{run_id}"
    manifest_path = final_dir / "manifest.json"
    if final_dir.exists():
        errors = validate_pilot_experiment(final_dir)
        if errors:
            raise RuntimeError(
                f"refusing to reuse invalid pilot run {final_dir}: {errors}"
            )
        return final_dir

    selected_output.mkdir(parents=True, exist_ok=True)
    work_dir = selected_output / f".{family}__{run_id}.incomplete-{os.getpid()}"
    if work_dir.exists():
        raise RuntimeError(f"pilot work directory already exists: {work_dir}")
    work_dir.mkdir()
    _write_yaml(work_dir / "resolved_config.yaml", config)
    _emit(log_callback, "experiment.started", family=family, run_id=run_id)
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
            log_callback=log_callback,
        )
    aggregate = {
        "schema_version": 1,
        "project": PROJECT_NAME,
        "experiment_name": EXPERIMENT_NAME,
        "run_id": run_id,
        "family": family,
        "selection_uses_lid_targets": False,
        "datasets": summaries,
        "macro_validation": _macro_metrics(summaries, "validation"),
        "macro_test": _macro_metrics(summaries, "test"),
    }
    _write_json(work_dir / "summary.json", aggregate)
    artifact_registry = _artifact_registry(run_dir=work_dir, datasets=summaries)
    _write_yaml(work_dir / "artifact_registry.yaml", artifact_registry)
    outputs = _portable_output_inventory(work_dir)
    manifest = {
        "schema_version": PILOT_MANIFEST_SCHEMA_VERSION,
        "project": PROJECT_NAME,
        "experiment_name": EXPERIMENT_NAME,
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "family": family,
        "config_sha256": config_sha,
        "source_tree_sha256": source_sha,
        "input_sha256": input_sha,
        "inputs": input_record,
        "environment": environment_state(),
        "selection_uses_lid_targets": False,
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
        family=family,
        run_id=run_id,
        macro_validation=aggregate["macro_validation"],
        macro_test=aggregate["macro_test"],
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
        "selection_uses_lid_targets",
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
    required_identity = {
        "project": PROJECT_NAME,
        "experiment_name": EXPERIMENT_NAME,
    }
    for field, expected in required_identity.items():
        if manifest.get(field) != expected:
            errors.append(f"manifest.{field} is not {expected!r}")
    if manifest.get("selection_uses_lid_targets") is not False:
        errors.append("manifest must attest target-free scale selection")

    resolved_path = directory / "resolved_config.yaml"
    try:
        resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
        resolved = validate_pilot_config(resolved)
    except Exception as exc:
        errors.append(f"invalid resolved_config.yaml: {exc}")
        resolved = None
    if resolved is not None:
        actual_config_sha = sha256_bytes(canonical_json(resolved).encode("utf-8"))
        if manifest.get("config_sha256") != actual_config_sha:
            errors.append("config_sha256 does not match resolved_config.yaml")
        identity = {
            "schema_version": 1,
            "project": PROJECT_NAME,
            "experiment_name": EXPERIMENT_NAME,
            "family": resolved["pilot_model"]["family"],
            "config_sha256": actual_config_sha,
            "source_tree_sha256": manifest.get("source_tree_sha256"),
            "input_sha256": manifest.get("input_sha256"),
        }
        expected_run_id = sha256_bytes(canonical_json(identity).encode("utf-8"))[:20]
        if manifest.get("run_id") != expected_run_id:
            errors.append("run_id is inconsistent with scientific identity")

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
                registry_path = _resolve_path(project_root, resolved["data"]["registry"])
                if not registry_path.is_file() or sha256_path(registry_path) != registry_record.get("sha256"):
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
                            errors.append(f"invalid source file record for {dataset_name}")
                            continue
                        relative = _safe_relative_path(record.get("path"))
                        if relative is None:
                            errors.append(f"unsafe source path for {dataset_name}")
                            continue
                        path = benchmark_root / relative
                        if not path.is_file() or sha256_path(path) != record.get("sha256"):
                            errors.append(f"dataset input changed or missing: {relative}")

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


def _logging_callback(config: Mapping[str, Any]) -> tuple[LogCallback | None, Callable[[], None]]:
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
        tags=(str(config["pilot_model"]["family"]),)
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
    "EXPERIMENT_NAME",
    "PilotConfigError",
    "compose_pilot_config",
    "run_pilot",
    "validate_pilot_config",
    "validate_pilot_experiment",
]
