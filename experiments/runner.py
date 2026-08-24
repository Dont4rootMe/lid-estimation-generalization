"""Deterministic experiment runner for oracle and learned field bundles.

The two backends share dataset preparation, immutable manifests, target-free
scale selection and aggregation, but retain distinct evidence labels.  An
empirical-channel identity check can therefore never be presented as a trained
model result, while learned results are accepted only from provenance-bound
field bundles.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import numpy.typing as npt
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from experiments.aggregate import (
    AggregateError,
    aggregate_matrix,
    recompute_aggregate_payload,
)
from datasets.archive import (
    EXACT_ARCHIVE_SHA256,
    verify_exact_archive,
    verify_extracted_tree,
)
from datasets.registry import (
    DatasetSpec,
    Representation,
    apply_registry_overlay,
    load_dataset,
    load_registry,
)
from models.fields import MODEL_FAMILY_READOUTS
from models.learned import (
    ArtifactRegistryIdentity,
    LearnedGridResult,
    ModelArtifactIdentity,
    evaluate_field_grid,
    input_inventory_sha256,
    load_artifact_registry,
    verify_model_artifacts,
)
from experiments.metrics import known_lid_metrics, prediction_summary
from models.oracle import (
    EmpiricalGaussianChannel,
    READOUT_IDS,
    readout_branch,
    select_stable_scale,
)
from utils.provenance import (
    sha256_file,
    verify_upstream_source,
)
from experiments.run_manifest import (
    build_manifest,
    canonical_json,
    hash_declared_sources,
    sha256_bytes,
    sha256_path,
    validate_manifest,
    write_json,
    write_manifest,
)
from datasets.synthetic import flat_plane


class ExperimentConfigError(ValueError):
    """Raised before a malformed experiment can write results."""


@dataclass(frozen=True)
class PreparedDataset:
    name: str
    representation: str
    raw_reference: npt.NDArray[np.float64]
    raw_validation: npt.NDArray[np.float64]
    raw_test: npt.NDArray[np.float64]
    reference: npt.NDArray[np.float64]
    validation: npt.NDArray[np.float64]
    test: npt.NDArray[np.float64]
    validation_target: npt.NDArray[np.float64] | None
    test_target: npt.NDArray[np.float64] | None
    training_dataset_sha256: str
    preprocessing_spec: Mapping[str, Any]
    preprocessing_sha256: str
    raw_selected_dataset_sha256: str
    model_selected_dataset_sha256: str
    source: Mapping[str, Any]


@dataclass(frozen=True)
class LearnedCellPreflight:
    """Validated learned inputs required before any result directory exists."""

    scale_unit: float
    scales: npt.NDArray[np.float64]
    input_files: Mapping[str, Mapping[str, str | int]]
    input_sha256: str
    bundle_input_sha256: str
    model_artifact_input_sha256: str
    checkpoint_sha256: str
    training_config_sha256: str
    trace: Mapping[str, str | int]


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def compose_experiment_config(
    overrides: Sequence[str] = (), *, root: Path | None = None
) -> DictConfig:
    """Compose the one supported configuration source: Hydra YAML groups."""

    project_root = repository_root() if root is None else root
    with initialize_config_dir(
        version_base="1.3",
        config_dir=str((project_root / "configs").resolve()),
    ):
        return compose(config_name="config", overrides=list(overrides))


_GROUP_FIELDS = {
    "experiment": frozenset(
        {"name", "scale_unit", "scale_multipliers", "readouts", "selection"}
    ),
    "models": frozenset(
        {
            "backend",
            "evidence_level",
            "name",
            "family",
            "seed",
            "readouts",
            "bundle_root",
            "artifact_registry",
            "artifact_registry_sha256",
            "trace",
            "min_finite_fraction",
        }
    ),
    "runtime": frozenset({"limits"}),
}
_PREPROCESSING_FIELDS = {
    "identity": frozenset({"kind"}),
    "scalar_affine": frozenset({"kind", "scale", "offset"}),
}
_DATASET_FIELDS = {
    "synthetic_flat": frozenset(
        {
            "source",
            "seed",
            "ambient_dim",
            "intrinsic_dim",
            "n_train",
            "n_validation",
            "n_test",
        }
    ),
    "lid_benchmarks": frozenset(
        {
            "source",
            "source_kind",
            "archive",
            "extracted_root",
            "root",
            "registry",
            "registry_overlay",
            "names",
            "representations",
        }
    ),
}
_SELECTION_FIELDS = frozenset(
    {"window", "min_valid_fraction", "min_effective_sample_size", "prefer"}
)
_LIMIT_FIELDS = frozenset(
    {"reference", "validation", "test", "reference_chunk", "query_chunk"}
)
_TRACE_FIELDS = frozenset({"backend", "probes", "seed"})


def _reject_unknown_fields(
    table: Mapping[str, Any], allowed: frozenset[str], *, path: str
) -> None:
    unknown = set(table) - allowed
    if unknown:
        raise ExperimentConfigError(
            f"unknown Hydra fields in {path}: {sorted(unknown)}"
        )


def _validate_hydra_groups(value: Mapping[str, Any]) -> None:
    """Reject misspelled YAML fields before projecting groups to the runner API."""

    experiment = value["experiment"]
    datasets = value["datasets"]
    preprocessing = value["preprocessing"]
    models = value["models"]
    runtime = value["runtime"]
    _reject_unknown_fields(experiment, _GROUP_FIELDS["experiment"], path="experiment")
    _reject_unknown_fields(models, _GROUP_FIELDS["models"], path="models")
    _reject_unknown_fields(runtime, _GROUP_FIELDS["runtime"], path="runtime")

    preprocessing_kind = preprocessing.get("kind")
    preprocessing_fields = _PREPROCESSING_FIELDS.get(preprocessing_kind)
    if preprocessing_fields is None:
        raise ExperimentConfigError(
            "preprocessing.kind must be 'identity' or 'scalar_affine'"
        )
    _reject_unknown_fields(
        preprocessing, preprocessing_fields, path="preprocessing"
    )

    dataset_fields = _DATASET_FIELDS.get(datasets.get("source"))
    if dataset_fields is None:
        dataset_fields = frozenset().union(*_DATASET_FIELDS.values())
    _reject_unknown_fields(datasets, dataset_fields, path="datasets")

    selection = experiment.get("selection", {})
    if not isinstance(selection, dict):
        raise ExperimentConfigError(
            "Hydra group 'experiment.selection' must be a mapping"
        )
    _reject_unknown_fields(
        selection, _SELECTION_FIELDS, path="experiment.selection"
    )
    limits = runtime.get("limits", {})
    if not isinstance(limits, dict):
        raise ExperimentConfigError(
            "Hydra group 'runtime.limits' must be a mapping"
        )
    _reject_unknown_fields(limits, _LIMIT_FIELDS, path="runtime.limits")
    trace = models.get("trace")
    if trace is not None:
        if not isinstance(trace, dict):
            raise ExperimentConfigError("Hydra group 'models.trace' must be a mapping")
        _reject_unknown_fields(trace, _TRACE_FIELDS, path="models.trace")


def normalize_hydra_config(config: DictConfig | Mapping[str, Any]) -> dict[str, Any]:
    """Resolve Hydra interpolation and map config groups to runner contracts."""

    if isinstance(config, DictConfig):
        value = OmegaConf.to_container(config, resolve=True)
    else:
        value = dict(config)
    if not isinstance(value, dict):
        raise ExperimentConfigError("composed Hydra config must be a mapping")
    allowed_groups = {
        "experiment",
        "datasets",
        "preprocessing",
        "models",
        "runtime",
    }
    unknown_groups = set(value) - allowed_groups
    if unknown_groups:
        raise ExperimentConfigError(
            f"unknown Hydra config groups: {sorted(unknown_groups)}"
        )
    for group in allowed_groups:
        if not isinstance(value.get(group), dict):
            raise ExperimentConfigError(f"Hydra group {group!r} must be a mapping")
    _validate_hydra_groups(value)
    experiment = value["experiment"]
    datasets = value["datasets"]
    preprocessing = value["preprocessing"]
    models = value["models"]
    runtime = value["runtime"]
    normalized = {
        "schema_version": 1,
        "name": experiment.get("name"),
        "backend": models.get("backend"),
        "evidence_level": models.get("evidence_level"),
        "readouts": experiment.get("readouts"),
        "scale_multipliers": experiment.get("scale_multipliers"),
        "scale_unit": experiment.get("scale_unit", "median_validation_nn"),
        "dataset": datasets,
        "preprocessing": preprocessing,
        "limits": runtime.get("limits", {}),
        "selection": experiment.get("selection", {}),
        "model": models,
    }
    return _validate_normalized_config(normalized)


def _validate_normalized_config(value: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "schema_version",
        "name",
        "backend",
        "evidence_level",
        "readouts",
        "scale_multipliers",
        "scale_unit",
        "dataset",
        "preprocessing",
        "limits",
        "selection",
        "model",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ExperimentConfigError(f"unknown experiment fields: {sorted(unknown)}")
    if value.get("schema_version") != 1:
        raise ExperimentConfigError("experiment schema_version must be 1")
    backend = value.get("backend")
    if backend not in {
        "empirical_gaussian_channel_oracle",
        "learned_field_bundle",
    }:
        raise ExperimentConfigError(
            "models.backend must be 'empirical_gaussian_channel_oracle' or "
            "'learned_field_bundle'"
        )
    expected_evidence = {
        "empirical_gaussian_channel_oracle": (
            "population_empirical_channel_not_trained_model"
        ),
        "learned_field_bundle": "learned_model",
    }[backend]
    if value.get("evidence_level") != expected_evidence:
        raise ExperimentConfigError(
            f"backend {backend!r} requires evidence_level={expected_evidence!r}"
        )
    name = value.get("name")
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ExperimentConfigError("name must be a non-empty safe path component")
    readout_ids = value.get("readouts")
    if not isinstance(readout_ids, list) or not readout_ids:
        raise ExperimentConfigError("readouts must be a non-empty array")
    unknown_readouts = set(readout_ids) - set(READOUT_IDS)
    if unknown_readouts or len(set(readout_ids)) != len(readout_ids):
        raise ExperimentConfigError(
            f"readouts contain unknown/duplicate IDs: {sorted(unknown_readouts)}"
        )
    scales = np.asarray(value.get("scale_multipliers"), dtype=np.float64)
    if scales.ndim != 1 or scales.size < 3 or not np.isfinite(scales).all():
        raise ExperimentConfigError("scale_multipliers needs at least 3 finite values")
    if np.any(scales <= 0) or np.unique(scales).size != scales.size:
        raise ExperimentConfigError("scale_multipliers must be positive and unique")
    if value.get("scale_unit", "median_validation_nn") not in {
        "absolute",
        "median_validation_nn",
    }:
        raise ExperimentConfigError(
            "scale_unit must be 'absolute' or 'median_validation_nn'"
        )
    dataset = value.get("dataset")
    if not isinstance(dataset, dict) or dataset.get("source") not in {
        "synthetic_flat",
        "lid_benchmarks",
    }:
        raise ExperimentConfigError(
            "datasets.source must be 'synthetic_flat' or 'lid_benchmarks'"
        )
    preprocessing = value.get("preprocessing")
    if not isinstance(preprocessing, dict):
        raise ExperimentConfigError("preprocessing must be a mapping")
    kind = preprocessing.get("kind")
    if kind == "identity":
        if set(preprocessing) != {"kind"}:
            raise ExperimentConfigError(
                "identity preprocessing accepts only the kind field"
            )
        value["preprocessing"] = {"kind": "identity"}
    elif kind == "scalar_affine":
        if set(preprocessing) != {"kind", "scale", "offset"}:
            raise ExperimentConfigError(
                "scalar_affine preprocessing requires exactly kind, scale, offset"
            )
        scale_raw = preprocessing.get("scale")
        offset_raw = preprocessing.get("offset")
        if isinstance(scale_raw, bool) or isinstance(offset_raw, bool):
            raise ExperimentConfigError(
                "scalar_affine scale and offset must be finite real numbers"
            )
        try:
            scale = float(scale_raw)
            offset = float(offset_raw)
        except (TypeError, ValueError) as exc:
            raise ExperimentConfigError(
                "scalar_affine scale and offset must be finite real numbers"
            ) from exc
        if not math.isfinite(scale) or scale == 0.0 or not math.isfinite(offset):
            raise ExperimentConfigError(
                "scalar_affine scale must be finite and nonzero; offset must be finite"
            )
        value["preprocessing"] = {
            "kind": "scalar_affine",
            "scale": scale,
            "offset": offset,
        }
    else:
        raise ExperimentConfigError(
            "preprocessing.kind must be 'identity' or 'scalar_affine'"
        )
    model = value.get("model")
    if not isinstance(model, dict):
        raise ExperimentConfigError("model must be a mapping")
    if backend == "empirical_gaussian_channel_oracle":
        if set(model) != {"backend", "evidence_level"}:
            raise ExperimentConfigError("oracle model config contains learned-only fields")
    else:
        family = model.get("family")
        if family not in MODEL_FAMILY_READOUTS:
            raise ExperimentConfigError(f"unknown learned model family: {family!r}")
        expected_readouts = list(MODEL_FAMILY_READOUTS[str(family)])
        if model.get("readouts") != expected_readouts or readout_ids != expected_readouts:
            raise ExperimentConfigError(
                f"learned family {family!r} requires readouts {expected_readouts!r}"
            )
        for field in ("name", "bundle_root"):
            if not isinstance(model.get(field), str) or not model[field]:
                raise ExperimentConfigError(f"models.{field} must be a non-empty string")
        if int(model.get("seed", -1)) < 0:
            raise ExperimentConfigError("models.seed must be non-negative")
        registry_path = model.get("artifact_registry")
        if not isinstance(registry_path, str) or not registry_path:
            raise ExperimentConfigError(
                "models.artifact_registry must be a non-empty YAML path"
            )
        registry_digest = model.get("artifact_registry_sha256")
        if not isinstance(registry_digest, str) or len(registry_digest) != 64 or any(
            character not in "0123456789abcdef" for character in registry_digest
        ):
            raise ExperimentConfigError(
                "models.artifact_registry_sha256 must be a lowercase "
                "64-character SHA-256"
            )
        trace = model.get("trace")
        if not isinstance(trace, dict) or trace.get("backend") not in {
            "exact",
            "hutchinson",
        }:
            raise ExperimentConfigError(
                "models.trace.backend must be 'exact' or 'hutchinson'"
            )
        probes = int(trace.get("probes", -1))
        if (trace["backend"] == "exact" and probes != 0) or (
            trace["backend"] == "hutchinson" and probes <= 0
        ):
            raise ExperimentConfigError(
                "models.trace.probes is inconsistent with the trace backend"
            )
        if int(trace.get("seed", -1)) < 0:
            raise ExperimentConfigError("models.trace.seed must be non-negative")
        finite = float(model.get("min_finite_fraction", -1.0))
        if not 0.0 <= finite <= 1.0:
            raise ExperimentConfigError(
                "models.min_finite_fraction must lie in [0, 1]"
            )
    for table_name in ("limits", "selection"):
        if not isinstance(value.get(table_name, {}), dict):
            raise ExperimentConfigError(f"[{table_name}] must be a table")
    return value


def _cap(value: Any, *, name: str) -> int:
    result = int(value)
    if result < 0:
        raise ExperimentConfigError(f"{name} must be non-negative (0 means all)")
    return result


def _indices(n: int, cap: int, seed: int) -> npt.NDArray[np.int64]:
    if cap == 0 or cap >= n:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=cap, replace=False)).astype(np.int64)


def _flatten_select(
    value: npt.ArrayLike, indices: npt.NDArray[np.int64]
) -> npt.NDArray[np.float64]:
    array = np.asarray(value)[indices]
    return np.ascontiguousarray(array.reshape(array.shape[0], -1), dtype=np.float64)


def _target_select(
    value: npt.ArrayLike | None, indices: npt.NDArray[np.int64]
) -> npt.NDArray[np.float64] | None:
    if value is None:
        return None
    return np.ascontiguousarray(np.asarray(value, dtype=np.float64)[indices].reshape(-1))


def _array_digest(named_arrays: Mapping[str, npt.ArrayLike]) -> str:
    digest = hashlib.sha256()
    for name in sorted(named_arrays):
        array = np.ascontiguousarray(np.asarray(named_arrays[name]))
        header = canonical_json(
            {"name": name, "shape": list(array.shape), "dtype": array.dtype.str}
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _selected_arrays_sha256(
    *,
    reference: npt.NDArray[np.float64],
    validation: npt.NDArray[np.float64],
    test: npt.NDArray[np.float64],
    validation_target: npt.NDArray[np.float64] | None,
    test_target: npt.NDArray[np.float64] | None,
) -> str:
    return _array_digest(
        {
            "reference": reference,
            "validation": validation,
            "test": test,
            **(
                {"validation_target": validation_target}
                if validation_target is not None
                else {}
            ),
            **({"test_target": test_target} if test_target is not None else {}),
        }
    )


def _preprocessing_sha256(spec: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json(dict(spec)).encode("utf-8"))


def _transform_features(
    value: npt.NDArray[np.float64], spec: Mapping[str, Any]
) -> npt.NDArray[np.float64]:
    if spec["kind"] == "identity":
        return np.ascontiguousarray(value, dtype=np.float64)
    scale = float(spec["scale"])
    offset = float(spec["offset"])
    with np.errstate(over="ignore", invalid="ignore"):
        transformed = np.asarray(value, dtype=np.float64) * scale + offset
    result = np.ascontiguousarray(transformed, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ExperimentConfigError(
            "preprocessing produced non-finite model-space features"
        )
    return result


def _make_prepared_dataset(
    *,
    name: str,
    representation: str,
    reference: npt.NDArray[np.float64],
    validation: npt.NDArray[np.float64],
    test: npt.NDArray[np.float64],
    validation_target: npt.NDArray[np.float64] | None,
    test_target: npt.NDArray[np.float64] | None,
    training_dataset_sha256: str,
    preprocessing: Mapping[str, Any],
    source: Mapping[str, Any],
) -> PreparedDataset:
    raw_reference = np.ascontiguousarray(reference, dtype=np.float64)
    raw_validation = np.ascontiguousarray(validation, dtype=np.float64)
    raw_test = np.ascontiguousarray(test, dtype=np.float64)
    spec = dict(preprocessing)
    preprocessing_sha256 = _preprocessing_sha256(spec)
    model_reference = _transform_features(raw_reference, spec)
    model_validation = _transform_features(raw_validation, spec)
    model_test = _transform_features(raw_test, spec)
    raw_selected_sha256 = _selected_arrays_sha256(
        reference=raw_reference,
        validation=raw_validation,
        test=raw_test,
        validation_target=validation_target,
        test_target=test_target,
    )
    model_selected_sha256 = _selected_arrays_sha256(
        reference=model_reference,
        validation=model_validation,
        test=model_test,
        validation_target=validation_target,
        test_target=test_target,
    )
    raw_split_sha256 = {
        "reference": _array_digest({"features": raw_reference}),
        "validation": _array_digest({"features": raw_validation}),
        "test": _array_digest({"features": raw_test}),
    }
    model_split_sha256 = {
        "reference": _array_digest({"features": model_reference}),
        "validation": _array_digest({"features": model_validation}),
        "test": _array_digest({"features": model_test}),
    }
    preprocessing_record = {
        "spec": spec,
        "sha256": preprocessing_sha256,
        "input_space": "raw_selected_rows",
        "output_space": "model",
        "scale_space": "model",
    }
    enriched_source = {
        **source,
        "training_dataset_sha256": training_dataset_sha256,
        "raw_selected_dataset_sha256": raw_selected_sha256,
        "model_selected_dataset_sha256": model_selected_sha256,
        "raw_split_sha256": raw_split_sha256,
        "model_split_sha256": model_split_sha256,
        "preprocessing": preprocessing_record,
    }
    return PreparedDataset(
        name=name,
        representation=representation,
        raw_reference=raw_reference,
        raw_validation=raw_validation,
        raw_test=raw_test,
        reference=model_reference,
        validation=model_validation,
        test=model_test,
        validation_target=validation_target,
        test_target=test_target,
        training_dataset_sha256=training_dataset_sha256,
        preprocessing_spec=spec,
        preprocessing_sha256=preprocessing_sha256,
        raw_selected_dataset_sha256=raw_selected_sha256,
        model_selected_dataset_sha256=model_selected_sha256,
        source=enriched_source,
    )


def _prepared_synthetic(config: Mapping[str, Any]) -> list[PreparedDataset]:
    dataset = config["dataset"]
    allowed = {
        "source",
        "seed",
        "ambient_dim",
        "intrinsic_dim",
        "n_train",
        "n_validation",
        "n_test",
    }
    unknown = set(dataset) - allowed
    if unknown:
        raise ExperimentConfigError(f"unknown synthetic dataset fields: {sorted(unknown)}")
    split = flat_plane(
        seed=int(dataset.get("seed", 0)),
        ambient_dim=int(dataset.get("ambient_dim", 8)),
        intrinsic_dim=int(dataset.get("intrinsic_dim", 3)),
        n_train=int(dataset.get("n_train", 2048)),
        n_validation=int(dataset.get("n_validation", 128)),
        n_test=int(dataset.get("n_test", 128)),
    )
    limits = config.get("limits", {})
    reference_idx = _indices(
        split.train.shape[0], _cap(limits.get("reference", 0), name="limits.reference"), 1101
    )
    validation_idx = _indices(
        split.validation.shape[0],
        _cap(limits.get("validation", 0), name="limits.validation"),
        1102,
    )
    test_idx = _indices(
        split.test.shape[0], _cap(limits.get("test", 0), name="limits.test"), 1103
    )
    training_dataset_sha256 = _array_digest({"training_features": split.train})
    source = {
        "kind": "synthetic_flat",
        "seed": int(dataset.get("seed", 0)),
        "generator": dict(dataset),
        "training_dataset_identity_kind": "canonical_array_sha256_v1",
        "training_dataset_sha256": training_dataset_sha256,
    }
    return [
        _make_prepared_dataset(
            name="synthetic_flat",
            representation="coordinates",
            reference=_flatten_select(split.train, reference_idx),
            validation=_flatten_select(split.validation, validation_idx),
            test=_flatten_select(split.test, test_idx),
            validation_target=_target_select(split.validation_lid, validation_idx),
            test_target=_target_select(split.test_lid, test_idx),
            training_dataset_sha256=training_dataset_sha256,
            preprocessing=config["preprocessing"],
            source=source,
        )
    ]


def _source_file_records(splits: Mapping[str, Any]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for split_name, loaded in splits.items():
        for artifact, path in loaded.source_paths.items():
            records[f"{split_name}/{artifact}.npy"] = {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return records


def _prepared_lid_benchmarks(
    config: Mapping[str, Any], root: Path
) -> list[PreparedDataset]:
    dataset = config["dataset"]
    allowed = {
        "source",
        "source_kind",
        "archive",
        "extracted_root",
        "root",
        "registry",
        "registry_overlay",
        "names",
        "representations",
    }
    unknown = set(dataset) - allowed
    if unknown:
        raise ExperimentConfigError(
            f"unknown lid_benchmarks dataset fields: {sorted(unknown)}"
        )
    source_kind = dataset.get("source_kind")
    if source_kind not in {"exact_archive", "generated_at_pinned_revision"}:
        raise ExperimentConfigError(
            "datasets.source_kind must be 'exact_archive' or "
            "'generated_at_pinned_revision'"
        )
    upstream_revision = verify_upstream_source(root / "lid_benchmarks")
    if not isinstance(dataset.get("root"), str) or not dataset["root"]:
        raise ExperimentConfigError("datasets.root must be a non-empty path string")
    data_root = root / str(dataset["root"])
    source_identity: dict[str, Any]
    if source_kind == "exact_archive":
        archive_path = root / str(dataset.get("archive", "data/benchmarks.zip"))
        extracted_root = root / str(
            dataset.get("extracted_root", "data/lid_benchmarks_exact")
        )
        archive_manifest = verify_exact_archive(archive_path)
        verify_extracted_tree(extracted_root, archive_manifest)
        expected_data_root = (extracted_root / "benchmarks").resolve()
        if data_root.resolve() != expected_data_root:
            raise ExperimentConfigError(
                "datasets.root must point to the authenticated archive's "
                f"benchmarks directory: {expected_data_root}"
            )
        source_identity = {
            "kind": "exact_archive",
            "archive_sha256": EXACT_ARCHIVE_SHA256,
            "archive_path": str(archive_path),
            "extracted_root": str(extracted_root),
            "upstream_revision": upstream_revision,
        }
    else:
        if "registry_overlay" not in dataset:
            raise ExperimentConfigError(
                "generated benchmark source requires datasets.registry_overlay"
            )
        source_identity = {
            "kind": "generated_at_pinned_revision",
            "upstream_revision": upstream_revision,
        }
    registry_path = root / str(
        dataset.get(
            "registry", "configs/datasets/registry/paper_benchmarks.yaml"
        )
    )
    registry_sha256 = sha256_file(registry_path)
    registry = load_registry(registry_path)
    registry_overlay_path: Path | None = None
    registry_overlay_sha256: str | None = None
    if source_kind == "generated_at_pinned_revision":
        registry_overlay_path = root / str(dataset["registry_overlay"])
        registry_overlay_sha256 = sha256_file(registry_overlay_path)
        registry = apply_registry_overlay(registry, registry_overlay_path)
    names = dataset.get("names")
    if names is None:
        names = list(registry)
    if not isinstance(names, list) or not names:
        raise ExperimentConfigError("dataset.names must be a non-empty array")
    absent = set(names) - set(registry)
    if absent:
        raise ExperimentConfigError(f"datasets absent from registry: {sorted(absent)}")
    representation_mode = dataset.get("representations", "all")
    if representation_mode not in {"all", "default"}:
        raise ExperimentConfigError("dataset.representations must be 'all' or 'default'")
    limits = config.get("limits", {})
    reference_cap = _cap(limits.get("reference", 0), name="limits.reference")
    validation_cap = _cap(limits.get("validation", 0), name="limits.validation")
    test_cap = _cap(limits.get("test", 0), name="limits.test")
    prepared: list[PreparedDataset] = []
    for name in names:
        spec: DatasetSpec = registry[name]
        representations: Iterable[Representation]
        if representation_mode == "all":
            representations = spec.available_representations
        else:
            representations = (spec.representation,)
        # Sample indices deliberately do not depend on dataset name.  Paired
        # transformations with aligned upstream ordering retain aligned rows.
        raw_default = load_dataset(data_root, spec, mmap_mode="r")
        reference_idx = _indices(
            raw_default["train"].n_samples, reference_cap, 2101
        )
        validation_idx = _indices(
            raw_default["val"].n_samples, validation_cap, 2102
        )
        test_idx = _indices(raw_default["test"].n_samples, test_cap, 2103)
        source_files = _source_file_records(raw_default)
        for representation in representations:
            raw = (
                raw_default
                if representation == spec.representation
                else load_dataset(
                    data_root, spec, representation=representation, mmap_mode="r"
                )
            )
            selected = {
                "reference": _flatten_select(raw["train"].features, reference_idx),
                "validation": _flatten_select(raw["val"].features, validation_idx),
                "test": _flatten_select(raw["test"].features, test_idx),
            }
            artifact = representation.value
            training_record = source_files.get(f"train/{artifact}.npy")
            validation_record = source_files.get(f"val/{artifact}.npy")
            test_record = source_files.get(f"test/{artifact}.npy")
            if training_record is None:
                raise ExperimentConfigError(
                    f"dataset source inventory has no train/{artifact}.npy: {name}"
                )
            if (
                source_kind == "exact_archive"
                and validation_record is not None
                and test_record is not None
                and validation_record["sha256"] == test_record["sha256"]
            ):
                raise ExperimentConfigError(
                    f"canonical archive contract violation: {name}/{artifact} "
                    "validation and test files are byte-identical"
                )
            paired_rows_sha256: str | None = None
            if raw["val"].labels is not None and raw["test"].labels is not None:
                paired_rows_sha256 = _array_digest(
                    {
                        "validation_labels": np.asarray(raw["val"].labels)[
                            validation_idx
                        ],
                        "test_labels": np.asarray(raw["test"].labels)[test_idx],
                    }
                )
            source = {
                **source_identity,
                "registry": str(registry_path),
                "registry_sha256": registry_sha256,
                **(
                    {
                        "registry_overlay": str(registry_overlay_path),
                        "registry_overlay_sha256": registry_overlay_sha256,
                    }
                    if registry_overlay_path is not None
                    and registry_overlay_sha256 is not None
                    else {}
                ),
                "dataset": name,
                "representation": representation.value,
                "training_dataset_identity_kind": "source_npy_file_sha256_v1",
                "training_dataset_sha256": training_record["sha256"],
                "source_files": source_files,
                "selected_indices_sha256": _array_digest(
                    {
                        "reference": reference_idx,
                        "validation": validation_idx,
                        "test": test_idx,
                    }
                ),
                **(
                    {"paired_rows_sha256": paired_rows_sha256}
                    if paired_rows_sha256 is not None
                    else {}
                ),
            }
            if not isinstance(training_record.get("sha256"), str):
                raise ExperimentConfigError(
                    f"missing training feature identity for {name}/{artifact}"
                )
            prepared.append(
                _make_prepared_dataset(
                    name=name,
                    representation=representation.value,
                    reference=selected["reference"],
                    validation=selected["validation"],
                    test=selected["test"],
                    validation_target=_target_select(
                        raw["val"].lid, validation_idx
                    ),
                    test_target=_target_select(raw["test"].lid, test_idx),
                    training_dataset_sha256=str(training_record["sha256"]),
                    preprocessing=config["preprocessing"],
                    source=source,
                )
            )
    return prepared


def prepare_datasets(config: Mapping[str, Any], root: Path) -> list[PreparedDataset]:
    source = config["dataset"]["source"]
    if source == "synthetic_flat":
        return _prepared_synthetic(config)
    return _prepared_lid_benchmarks(config, root)


def _nearest_scale(
    reference: npt.NDArray[np.float64],
    validation: npt.NDArray[np.float64],
    *,
    max_validation: int = 256,
    reference_chunk: int = 4096,
) -> float:
    probes = validation[:max_validation]
    minimum = np.full(probes.shape[0], np.inf, dtype=np.float64)
    probe_norm = np.einsum("ij,ij->i", probes, probes)
    for start in range(0, reference.shape[0], reference_chunk):
        block = reference[start : start + reference_chunk]
        block_norm = np.einsum("ij,ij->i", block, block)
        squared = probe_norm[:, None] + block_norm[None, :] - 2.0 * probes @ block.T
        np.maximum(squared, 0.0, out=squared)
        minimum = np.minimum(minimum, squared.min(axis=1))
    positive = np.sqrt(minimum[minimum > 0])
    if positive.size == 0:
        raise ValueError("cannot calibrate scale: all validation points match reference")
    result = float(np.median(positive))
    if not math.isfinite(result) or result <= 0:
        raise ValueError("median nearest-reference scale is not finite and positive")
    return result


def _save_npy(path: Path, value: npt.ArrayLike) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, np.asarray(value), allow_pickle=False)
        stream.flush()
    temporary.replace(path)


def _selected_dataset_sha256(prepared: PreparedDataset) -> str:
    """Identity of the exact selected source-space rows and targets."""

    return prepared.raw_selected_dataset_sha256


def _model_selected_dataset_sha256(prepared: PreparedDataset) -> str:
    """Identity of selected rows after the declared preprocessing transform."""

    return prepared.model_selected_dataset_sha256


def _base_dataset_record(
    prepared: PreparedDataset, *, dataset_sha256: str
) -> dict[str, Any]:
    return {
        **prepared.source,
        "name": prepared.name,
        "representation": prepared.representation,
        "sha256": dataset_sha256,
        "selected_dataset_sha256": prepared.raw_selected_dataset_sha256,
        "n_reference": int(prepared.reference.shape[0]),
        "n_validation": int(prepared.validation.shape[0]),
        "n_test": int(prepared.test.shape[0]),
        "ambient_dim": int(prepared.reference.shape[1]),
    }


def _dataset_upstream_revision(prepared: PreparedDataset) -> str:
    revision = prepared.source.get("upstream_revision", "not_applicable")
    if not isinstance(revision, str) or not revision:
        raise ExperimentConfigError("dataset upstream_revision must be a string")
    return revision


def _physical_scale_grid(
    prepared: PreparedDataset, config: Mapping[str, Any]
) -> tuple[float, npt.NDArray[np.float64]]:
    multipliers = np.asarray(config["scale_multipliers"], dtype=np.float64)
    limits = config.get("limits", {})
    unit = (
        1.0
        if config.get("scale_unit", "median_validation_nn") == "absolute"
        else _nearest_scale(
            prepared.reference,
            prepared.validation,
            reference_chunk=int(limits.get("reference_chunk", 4096)),
        )
    )
    return float(unit), np.ascontiguousarray(multipliers * unit)


def _learned_bundle_root(config: Mapping[str, Any], root: Path) -> Path:
    configured = Path(str(config["model"]["bundle_root"]))
    return (
        configured.resolve()
        if configured.is_absolute()
        else (root / configured).resolve()
    )


def _combined_learned_input_files(
    grid: LearnedGridResult, artifacts: ModelArtifactIdentity
) -> dict[str, Mapping[str, str | int]]:
    records: dict[str, Mapping[str, str | int]] = {
        name: dict(record) for name, record in artifacts.input_files.items()
    }
    for relative, record in grid.input_files.items():
        logical_name = f"field_bundles/{relative}"
        if logical_name in records:
            raise ExperimentConfigError(
                f"duplicate learned input inventory key: {logical_name}"
            )
        records[logical_name] = dict(record)
    return records


def _model_with_artifact_identity(
    model: Mapping[str, Any], artifacts: ModelArtifactIdentity
) -> dict[str, Any]:
    result = dict(model)
    result["checkpoint_sha256"] = artifacts.checkpoint_sha256
    result["training_config_sha256"] = artifacts.training_config_sha256
    return result


def _verify_cell_model_artifacts(
    *,
    registry: ArtifactRegistryIdentity,
    config: Mapping[str, Any],
    prepared: PreparedDataset,
) -> ModelArtifactIdentity:
    return verify_model_artifacts(
        registry=registry,
        model=config["model"],
        dataset_name=prepared.name,
        representation=prepared.representation,
        training_dataset_sha256=prepared.training_dataset_sha256,
        preprocessing_sha256=prepared.preprocessing_sha256,
    )


def _preflight_learned_cells(
    *,
    prepared: Sequence[PreparedDataset],
    config: Mapping[str, Any],
    root: Path,
) -> dict[tuple[str, str], LearnedCellPreflight]:
    """Validate every required bundle before creating the output matrix.

    The second evaluation performed while materializing a cell protects
    against an input changing between preflight and output.  Keeping only
    hashes here bounds peak memory by one dataset rather than the whole matrix.
    """

    bundle_root = _learned_bundle_root(config, root)
    registry = load_artifact_registry(root=root, model=config["model"])
    expected_cells = {(item.name, item.representation) for item in prepared}
    actual_cells = set(registry.artifacts)
    if actual_cells != expected_cells:
        raise ExperimentConfigError(
            "artifact registry coverage must exactly match the prepared matrix: "
            f"missing={sorted(expected_cells - actual_cells)}, "
            f"unexpected={sorted(actual_cells - expected_cells)}"
        )
    result: dict[tuple[str, str], LearnedCellPreflight] = {}
    for item in prepared:
        key = (item.name, item.representation)
        if key in result:
            raise ExperimentConfigError(
                "learned matrix repeats dataset/representation cell "
                f"{item.name!r}/{item.representation!r}"
            )
        scale_unit, scales = _physical_scale_grid(item, config)
        model_artifacts = _verify_cell_model_artifacts(
            registry=registry, config=config, prepared=item
        )
        cell_model = _model_with_artifact_identity(
            config["model"], model_artifacts
        )
        grid = evaluate_field_grid(
            bundle_root=bundle_root,
            model=cell_model,
            dataset_name=item.name,
            training_dataset_sha256=item.training_dataset_sha256,
            dataset_sha256=_selected_dataset_sha256(item),
            representation=item.representation,
            validation=item.raw_validation,
            test=item.raw_test,
            preprocessing_sha256=item.preprocessing_sha256,
            model_space_validation=item.validation,
            model_space_test=item.test,
            scales=scales,
        )
        input_files = _combined_learned_input_files(grid, model_artifacts)
        result[key] = LearnedCellPreflight(
            scale_unit=scale_unit,
            scales=scales,
            input_files=input_files,
            input_sha256=input_inventory_sha256(input_files),
            bundle_input_sha256=grid.input_sha256,
            model_artifact_input_sha256=model_artifacts.input_sha256,
            checkpoint_sha256=grid.checkpoint_sha256,
            training_config_sha256=grid.training_config_sha256,
            trace=grid.trace,
        )
    return result


def _run_oracle_cell(
    *,
    prepared: PreparedDataset,
    config: Mapping[str, Any],
    root: Path,
    matrix_dir: Path,
) -> dict[str, Any]:
    dataset_record = _base_dataset_record(
        prepared, dataset_sha256=_model_selected_dataset_sha256(prepared)
    )
    placeholder = build_manifest(
        root=root,
        config=config,
        dataset=dataset_record,
        outputs={},
        upstream_sha=_dataset_upstream_revision(prepared),
    )
    cell_name = f"{prepared.name}__{prepared.representation}__{placeholder.run_id}"
    cell_dir = matrix_dir / "cells" / cell_name
    manifest_path = cell_dir / "manifest.json"
    if manifest_path.is_file():
        errors = validate_manifest(manifest_path, expected=placeholder)
        if errors:
            raise RuntimeError(
                f"existing run cell {cell_dir} failed integrity checks: {errors}"
            )
        return {
            "cell": cell_name,
            "status": "reused",
            "manifest": str(manifest_path.relative_to(matrix_dir)),
        }
    if cell_dir.exists():
        raise RuntimeError(f"refusing to overwrite incomplete run directory: {cell_dir}")
    cell_dir.mkdir(parents=True)

    limits = config.get("limits", {})
    channel = EmpiricalGaussianChannel(
        prepared.reference,
        reference_chunk_size=int(limits.get("reference_chunk", 4096)),
    )
    multipliers = np.asarray(config["scale_multipliers"], dtype=np.float64)
    unit, scales = _physical_scale_grid(prepared, config)
    n_scales = scales.size
    validation_response = np.empty((prepared.validation.shape[0], n_scales))
    validation_full = np.empty_like(validation_response)
    validation_ess = np.empty_like(validation_response)
    test_response = np.empty((prepared.test.shape[0], n_scales))
    test_full = np.empty_like(test_response)
    test_ess = np.empty_like(test_response)
    query_chunk = int(limits.get("query_chunk", 128))
    for index, scale in enumerate(scales):
        validation_moments = channel.posterior(
            prepared.validation, float(scale), query_chunk_size=query_chunk
        )
        validation_response[:, index] = validation_moments.response
        validation_full[:, index] = validation_moments.full
        validation_ess[:, index] = validation_moments.effective_sample_size
        test_moments = channel.posterior(
            prepared.test, float(scale), query_chunk_size=query_chunk
        )
        test_response[:, index] = test_moments.response
        test_full[:, index] = test_moments.full
        test_ess[:, index] = test_moments.effective_sample_size

    selection = config.get("selection", {})
    min_ess = float(selection.get("min_effective_sample_size", 2.0))
    selectors: dict[str, tuple[int, dict[str, Any]]] = {}
    for branch, curve in (
        ("response", validation_response),
        ("full", validation_full),
    ):
        selectors[branch] = select_stable_scale(
            scales,
            curve,
            window=int(selection.get("window", 1)),
            valid_mask=validation_ess >= min_ess,
            min_valid_fraction=float(selection.get("min_valid_fraction", 0.5)),
            prefer=str(selection.get("prefer", "larger")),  # type: ignore[arg-type]
        )

    _save_npy(cell_dir / "scales.npy", scales)
    _save_npy(cell_dir / "validation_response_curve.npy", validation_response)
    _save_npy(cell_dir / "validation_full_curve.npy", validation_full)
    _save_npy(cell_dir / "validation_effective_sample_size.npy", validation_ess)
    _save_npy(cell_dir / "test_response_curve.npy", test_response)
    _save_npy(cell_dir / "test_full_curve.npy", test_full)
    _save_npy(cell_dir / "test_effective_sample_size.npy", test_ess)
    if prepared.test_target is not None:
        _save_npy(cell_dir / "test_target.npy", prepared.test_target)

    summaries: dict[str, Any] = {}
    for readout_id in config["readouts"]:
        branch = readout_branch(readout_id)
        selected_index, diagnostics = selectors[branch]
        curve = test_response if branch == "response" else test_full
        prediction = curve[:, selected_index]
        filename = f"prediction__{readout_id}.npy"
        _save_npy(cell_dir / filename, prediction)
        metric = (
            prediction_summary(prediction)
            if prepared.test_target is None
            else known_lid_metrics(prediction, prepared.test_target)
        )
        summaries[readout_id] = {
            "oracle_not_trained_model": True,
            "branch": branch,
            "scale_selection": diagnostics,
            "metrics": metric,
        }

    summary = {
        "schema_version": 1,
        "backend": "empirical_gaussian_channel_oracle",
        "evidence_level": "population_empirical_channel_not_trained_model",
        "dataset": dataset_record,
        "preprocessing": dict(dataset_record["preprocessing"]),
        "scale_space": "model",
        "scale_unit": float(unit),
        "scale_multipliers": [float(value) for value in multipliers],
        "readouts": summaries,
    }
    write_json(cell_dir / "summary.json", summary)
    output_hashes = {
        path.relative_to(cell_dir).as_posix(): sha256_path(path)
        for path in sorted(cell_dir.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = build_manifest(
        root=root,
        config=config,
        dataset=dataset_record,
        outputs=output_hashes,
        upstream_sha=_dataset_upstream_revision(prepared),
    )
    if manifest.run_id != placeholder.run_id:
        raise AssertionError("run identity changed while writing outputs")
    write_manifest(manifest_path, manifest)
    return {
        "cell": cell_name,
        "status": "completed",
        "manifest": str(manifest_path.relative_to(matrix_dir)),
    }


def _learned_dataset_record(
    prepared: PreparedDataset,
    *,
    config: Mapping[str, Any],
    preflight: LearnedCellPreflight,
    bundle_root: Path,
) -> dict[str, Any]:
    selected_sha256 = _selected_dataset_sha256(prepared)
    model_selected_sha256 = _model_selected_dataset_sha256(prepared)
    learned_identity = {
        "schema_version": 1,
        "bundle_root": str(bundle_root),
        "input_sha256": preflight.input_sha256,
        "bundle_input_sha256": preflight.bundle_input_sha256,
        "model_artifact_input_sha256": preflight.model_artifact_input_sha256,
        "input_files": preflight.input_files,
        "model_name": str(config["model"]["name"]),
        "model_family": str(config["model"]["family"]),
        "model_seed": int(config["model"]["seed"]),
        "checkpoint_sha256": preflight.checkpoint_sha256,
        "training_config_sha256": preflight.training_config_sha256,
        "training_dataset_sha256": prepared.training_dataset_sha256,
        "preprocessing_sha256": prepared.preprocessing_sha256,
        "raw_selected_dataset_sha256": selected_sha256,
        "model_selected_dataset_sha256": model_selected_sha256,
        "trace": dict(preflight.trace),
    }
    composite_sha256 = sha256_bytes(
        canonical_json(
            {
                "schema_version": 1,
                "raw_selected_dataset_sha256": selected_sha256,
                "model_selected_dataset_sha256": model_selected_sha256,
                "training_dataset_sha256": prepared.training_dataset_sha256,
                "preprocessing_sha256": prepared.preprocessing_sha256,
                "learned_input_sha256": preflight.input_sha256,
            }
        ).encode("utf-8")
    )
    return {
        **_base_dataset_record(prepared, dataset_sha256=composite_sha256),
        "selected_dataset_sha256": selected_sha256,
        "learned_input": learned_identity,
    }


def _same_learned_input(
    result: LearnedGridResult,
    artifacts: ModelArtifactIdentity,
    preflight: LearnedCellPreflight,
) -> bool:
    input_files = _combined_learned_input_files(result, artifacts)
    return (
        input_inventory_sha256(input_files) == preflight.input_sha256
        and input_files == preflight.input_files
        and result.input_sha256 == preflight.bundle_input_sha256
        and artifacts.input_sha256 == preflight.model_artifact_input_sha256
        and result.checkpoint_sha256 == preflight.checkpoint_sha256
        and artifacts.checkpoint_sha256 == preflight.checkpoint_sha256
        and result.training_config_sha256 == preflight.training_config_sha256
        and artifacts.training_config_sha256
        == preflight.training_config_sha256
        and result.trace == preflight.trace
    )


def _run_learned_cell(
    *,
    prepared: PreparedDataset,
    config: Mapping[str, Any],
    root: Path,
    matrix_dir: Path,
    preflight: LearnedCellPreflight,
) -> dict[str, Any]:
    bundle_root = _learned_bundle_root(config, root)
    dataset_record = _learned_dataset_record(
        prepared,
        config=config,
        preflight=preflight,
        bundle_root=bundle_root,
    )
    placeholder = build_manifest(
        root=root,
        config=config,
        dataset=dataset_record,
        outputs={},
        upstream_sha=_dataset_upstream_revision(prepared),
    )
    cell_name = f"{prepared.name}__{prepared.representation}__{placeholder.run_id}"
    cell_dir = matrix_dir / "cells" / cell_name
    manifest_path = cell_dir / "manifest.json"
    if manifest_path.is_file():
        errors = validate_manifest(manifest_path, expected=placeholder)
        if errors:
            raise RuntimeError(
                f"existing run cell {cell_dir} failed integrity checks: {errors}"
            )
        return {
            "cell": cell_name,
            "status": "reused",
            "manifest": str(manifest_path.relative_to(matrix_dir)),
        }
    if cell_dir.exists():
        raise RuntimeError(f"refusing to overwrite incomplete run directory: {cell_dir}")

    # Evaluate once more before creating the cell.  Besides producing curves,
    # this closes the gap where a bundle could change after matrix preflight.
    registry = load_artifact_registry(root=root, model=config["model"])
    model_artifacts = _verify_cell_model_artifacts(
        registry=registry, config=config, prepared=prepared
    )
    cell_model = _model_with_artifact_identity(config["model"], model_artifacts)
    grid = evaluate_field_grid(
        bundle_root=bundle_root,
        model=cell_model,
        dataset_name=prepared.name,
        training_dataset_sha256=prepared.training_dataset_sha256,
        dataset_sha256=str(dataset_record["selected_dataset_sha256"]),
        representation=prepared.representation,
        validation=prepared.raw_validation,
        test=prepared.raw_test,
        preprocessing_sha256=prepared.preprocessing_sha256,
        model_space_validation=prepared.validation,
        model_space_test=prepared.test,
        scales=preflight.scales,
    )
    if not _same_learned_input(grid, model_artifacts, preflight):
        raise RuntimeError(
            "learned field bundle inputs changed between preflight and cell write"
        )

    selection = config.get("selection", {})
    selectors: dict[str, tuple[int, dict[str, Any]]] = {}
    for readout_id in config["readouts"]:
        selectors[readout_id] = select_stable_scale(
            preflight.scales,
            grid.validation_curves[readout_id],
            window=int(selection.get("window", 1)),
            min_valid_fraction=float(selection.get("min_valid_fraction", 0.5)),
            prefer=str(selection.get("prefer", "larger")),  # type: ignore[arg-type]
        )

    cell_dir.mkdir(parents=True)
    _save_npy(cell_dir / "scales.npy", preflight.scales)
    if prepared.validation_target is not None:
        _save_npy(cell_dir / "validation_target.npy", prepared.validation_target)
    if prepared.test_target is not None:
        _save_npy(cell_dir / "test_target.npy", prepared.test_target)

    summaries: dict[str, Any] = {}
    for readout_id in config["readouts"]:
        validation_curve = grid.validation_curves[readout_id]
        test_curve = grid.test_curves[readout_id]
        _save_npy(
            cell_dir / f"validation_curve__{readout_id}.npy", validation_curve
        )
        _save_npy(cell_dir / f"test_curve__{readout_id}.npy", test_curve)
        selected_index, diagnostics = selectors[readout_id]
        prediction = test_curve[:, selected_index]
        _save_npy(cell_dir / f"prediction__{readout_id}.npy", prediction)
        metric = (
            prediction_summary(prediction)
            if prepared.test_target is None
            else known_lid_metrics(prediction, prepared.test_target)
        )
        summaries[readout_id] = {
            "learned_model_evidence": True,
            "selection_uses_lid_targets": False,
            "scale_selection": diagnostics,
            "metrics": metric,
        }

    summary = {
        "schema_version": 1,
        "backend": "learned_field_bundle",
        "evidence_level": "learned_model",
        "dataset": dataset_record,
        "model": {
            "name": config["model"]["name"],
            "family": config["model"]["family"],
            "seed": int(config["model"]["seed"]),
            "checkpoint_sha256": preflight.checkpoint_sha256,
            "training_config_sha256": preflight.training_config_sha256,
            "trace": dict(preflight.trace),
            "input_sha256": preflight.input_sha256,
        },
        "preprocessing": dict(dataset_record["preprocessing"]),
        "scale_space": "model",
        "scale_unit": float(preflight.scale_unit),
        "scale_multipliers": [
            float(value) for value in config["scale_multipliers"]
        ],
        "physical_scales": [float(value) for value in preflight.scales],
        "readouts": summaries,
    }
    write_json(cell_dir / "summary.json", summary)
    output_hashes = {
        path.relative_to(cell_dir).as_posix(): sha256_path(path)
        for path in sorted(cell_dir.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = build_manifest(
        root=root,
        config=config,
        dataset=dataset_record,
        outputs=output_hashes,
        upstream_sha=_dataset_upstream_revision(prepared),
    )
    if manifest.run_id != placeholder.run_id:
        raise AssertionError("run identity changed while writing learned outputs")
    write_manifest(manifest_path, manifest)
    return {
        "cell": cell_name,
        "status": "completed",
        "manifest": str(manifest_path.relative_to(matrix_dir)),
    }


def run_experiment(
    hydra_config: DictConfig,
    *,
    root: Path | None = None,
    output_root: Path | None = None,
    hydra_overrides: Sequence[str] = (),
) -> Path:
    if not isinstance(hydra_config, DictConfig) or not OmegaConf.is_struct(
        hydra_config
    ):
        raise ExperimentConfigError(
            "run_experiment requires a structured Hydra-composed DictConfig; "
            "raw mappings cannot be stamped as "
            "config_source='hydra_yaml_composition'"
        )
    project_root = (repository_root() if root is None else root).resolve()
    config = normalize_hydra_config(hydra_config)
    configured_output = Path(str(config.get("output_root", "artifacts/runs")))
    selected_output_root = (
        configured_output if output_root is None else Path(output_root)
    )
    if not selected_output_root.is_absolute():
        selected_output_root = project_root / selected_output_root
    config_hash = sha256_bytes(canonical_json(config).encode("utf-8"))
    prepared = prepare_datasets(config, project_root)
    learned_preflight: dict[tuple[str, str], LearnedCellPreflight] = {}
    if config["backend"] == "learned_field_bundle":
        learned_preflight = _preflight_learned_cells(
            prepared=prepared,
            config=config,
            root=project_root,
        )

    matrix_inputs = []
    for item in prepared:
        input_record: dict[str, Any] = {
            "dataset": item.name,
            "representation": item.representation,
            "selected_dataset_sha256": _selected_dataset_sha256(item),
            "model_selected_dataset_sha256": (
                _model_selected_dataset_sha256(item)
            ),
            "training_dataset_sha256": item.training_dataset_sha256,
            "preprocessing_sha256": item.preprocessing_sha256,
        }
        preflight = learned_preflight.get((item.name, item.representation))
        if preflight is not None:
            input_record["learned_input_sha256"] = preflight.input_sha256
        matrix_inputs.append(input_record)
    matrix_input_sha256 = sha256_bytes(
        canonical_json(matrix_inputs).encode("utf-8")
    )
    matrix_identity_sha256 = sha256_bytes(
        canonical_json(
            {
                "config_sha256": config_hash,
                "input_sha256": matrix_input_sha256,
            }
        ).encode("utf-8")
    )

    # All learned bundle paths, SHA identities and metadata are validated above;
    # malformed or missing learned evidence therefore cannot create an output.
    matrix_dir = (
        selected_output_root
        / f"{config['name']}__{matrix_identity_sha256[:12]}"
    )
    matrix_dir.mkdir(parents=True, exist_ok=True)
    if config["backend"] == "empirical_gaussian_channel_oracle":
        results = [
            _run_oracle_cell(
                prepared=item,
                config=config,
                root=project_root,
                matrix_dir=matrix_dir,
            )
            for item in prepared
        ]
    else:
        results = [
            _run_learned_cell(
                prepared=item,
                config=config,
                root=project_root,
                matrix_dir=matrix_dir,
                preflight=learned_preflight[(item.name, item.representation)],
            )
            for item in prepared
        ]
    matrix = {
        "schema_version": 1,
        "name": config["name"],
        "backend": config["backend"],
        "evidence_level": config["evidence_level"],
        "config_source": "hydra_yaml_composition",
        "hydra_overrides": list(hydra_overrides),
        "config_sha256": config_hash,
        "input_sha256": matrix_input_sha256,
        "matrix_identity_sha256": matrix_identity_sha256,
        "requested_cells": len(prepared),
        "complete_cells": len(results),
        "complete": len(results) == len(prepared),
        "aggregate": "aggregate.json",
        "cells": results,
    }
    write_json(matrix_dir / "matrix.json", matrix)
    aggregate_path = aggregate_matrix(matrix_dir, root=project_root)
    matrix["aggregate_sha256"] = sha256_path(aggregate_path)
    write_json(matrix_dir / "matrix.json", matrix)
    return matrix_dir


def run_oracle(
    hydra_config: DictConfig,
    *,
    root: Path | None = None,
    output_root: Path | None = None,
    hydra_overrides: Sequence[str] = (),
) -> Path:
    """Backward-compatible oracle-only entry point."""

    config = normalize_hydra_config(hydra_config)
    if config["backend"] != "empirical_gaussian_channel_oracle":
        raise ExperimentConfigError(
            "run_oracle accepts only models.backend="
            "'empirical_gaussian_channel_oracle'; use run_experiment for dispatch"
        )
    return run_experiment(
        hydra_config,
        root=root,
        output_root=output_root,
        hydra_overrides=hydra_overrides,
    )


def run_composed_experiment(
    overrides: Sequence[str] = (),
    *,
    root: Path | None = None,
    output_root: Path | None = None,
) -> Path:
    """Compose Hydra groups programmatically and dispatch by model backend."""

    config = compose_experiment_config(overrides, root=root)
    return run_experiment(
        config,
        root=root,
        output_root=output_root,
        hydra_overrides=overrides,
    )


_EVIDENCE_BY_BACKEND = {
    "empirical_gaussian_channel_oracle": (
        "population_empirical_channel_not_trained_model"
    ),
    "learned_field_bundle": "learned_model",
}
_AGGREGATE_MATRIX_IDENTITY_FIELDS = (
    "name",
    "backend",
    "evidence_level",
    "config_sha256",
    "input_sha256",
    "matrix_identity_sha256",
)


def _dataset_source_anchor_errors(
    *,
    dataset: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
    root: Path,
) -> list[str]:
    """Compare recorded registry identities with the current configured files."""

    dataset_config = resolved_config.get("dataset")
    if not isinstance(dataset_config, Mapping) or dataset_config.get(
        "source"
    ) != "lid_benchmarks":
        return []
    fields = [("registry", "registry_sha256")]
    if dataset_config.get("source_kind") == "generated_at_pinned_revision":
        fields.append(("registry_overlay", "registry_overlay_sha256"))
    errors: list[str] = []
    for config_field, record_field in fields:
        configured_path = dataset_config.get(config_field)
        if not isinstance(configured_path, str) or not configured_path:
            errors.append(f"resolved_config.dataset.{config_field} is invalid")
            continue
        source_path = Path(configured_path)
        if not source_path.is_absolute():
            source_path = root / source_path
        try:
            current_digest = sha256_file(source_path)
        except (OSError, ValueError) as exc:
            errors.append(f"cannot hash configured dataset.{config_field}: {exc}")
        else:
            if dataset.get(record_field) != current_digest:
                errors.append(
                    f"dataset.{record_field} does not match configured "
                    f"dataset.{config_field}"
                )
    return errors


def _expected_cell_keys_from_config(
    resolved_config: Mapping[str, Any], *, root: Path
) -> set[tuple[str, str]]:
    """Recover the complete dataset/representation grid declared by Hydra."""

    dataset = resolved_config.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ExperimentConfigError("resolved_config.dataset must be a mapping")
    source = dataset.get("source")
    if source == "synthetic_flat":
        return {("synthetic_flat", "coordinates")}
    if source != "lid_benchmarks":
        raise ExperimentConfigError(
            f"resolved_config.dataset.source is unsupported: {source!r}"
        )

    registry_value = dataset.get("registry")
    if not isinstance(registry_value, str) or not registry_value:
        raise ExperimentConfigError(
            "resolved_config.dataset.registry must be a non-empty path"
        )
    registry_path = Path(registry_value)
    if not registry_path.is_absolute():
        registry_path = root / registry_path
    registry = load_registry(registry_path)
    if dataset.get("source_kind") == "generated_at_pinned_revision":
        overlay_value = dataset.get("registry_overlay")
        if not isinstance(overlay_value, str) or not overlay_value:
            raise ExperimentConfigError(
                "generated benchmark config must declare registry_overlay"
            )
        overlay_path = Path(overlay_value)
        if not overlay_path.is_absolute():
            overlay_path = root / overlay_path
        registry = apply_registry_overlay(registry, overlay_path)

    names_value = dataset.get("names")
    if names_value is None:
        names = list(registry)
    elif isinstance(names_value, list) and all(
        isinstance(item, str) and item for item in names_value
    ):
        names = list(names_value)
    else:
        raise ExperimentConfigError(
            "resolved_config.dataset.names must be a non-empty string array"
        )
    if not names or len(set(names)) != len(names):
        raise ExperimentConfigError(
            "resolved_config.dataset.names must be non-empty and unique"
        )
    unknown = set(names) - set(registry)
    if unknown:
        raise ExperimentConfigError(
            f"resolved_config references datasets absent from registry: {sorted(unknown)}"
        )

    representation_mode = dataset.get("representations", "all")
    if representation_mode not in {"all", "default"}:
        raise ExperimentConfigError(
            "resolved_config.dataset.representations must be 'all' or 'default'"
        )
    expected: set[tuple[str, str]] = set()
    for name in names:
        spec = registry[name]
        representations = (
            spec.available_representations
            if representation_mode == "all"
            else (spec.representation,)
        )
        expected.update((name, item.value) for item in representations)
    return expected


def _is_sha256_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _preprocessing_record_errors(
    *, dataset: Mapping[str, Any], resolved_config: Mapping[str, Any]
) -> list[str]:
    errors: list[str] = []
    configured = resolved_config.get("preprocessing")
    record = dataset.get("preprocessing")
    if not isinstance(configured, Mapping):
        return ["resolved_config.preprocessing must be an object"]
    if not isinstance(record, Mapping):
        return ["dataset.preprocessing must be an object"]
    expected_record_fields = {
        "spec",
        "sha256",
        "input_space",
        "output_space",
        "scale_space",
    }
    if set(record) != expected_record_fields:
        errors.append(
            "dataset.preprocessing field inventory mismatch: "
            f"expected={sorted(expected_record_fields)}, got={sorted(record)}"
        )
    spec = record.get("spec")
    if not isinstance(spec, Mapping):
        errors.append("dataset.preprocessing.spec must be an object")
    else:
        if dict(spec) != dict(configured):
            errors.append(
                "dataset.preprocessing.spec disagrees with resolved_config.preprocessing"
            )
        try:
            expected_sha256 = _preprocessing_sha256(spec)
        except (TypeError, ValueError) as exc:
            errors.append(
                f"dataset.preprocessing.spec is not canonicalizable: {exc}"
            )
            expected_sha256 = None
        if (
            expected_sha256 is not None
            and record.get("sha256") != expected_sha256
        ):
            errors.append(
                "dataset.preprocessing.sha256 is inconsistent with canonical spec"
            )
    expected_spaces = {
        "input_space": "raw_selected_rows",
        "output_space": "model",
        "scale_space": "model",
    }
    for field, expected in expected_spaces.items():
        if record.get(field) != expected:
            errors.append(
                f"dataset.preprocessing.{field} must be {expected!r}"
            )

    for field in (
        "training_dataset_sha256",
        "raw_selected_dataset_sha256",
        "model_selected_dataset_sha256",
    ):
        if not _is_sha256_digest(dataset.get(field)):
            errors.append(f"dataset.{field} must be a lowercase SHA-256")
    if dataset.get("selected_dataset_sha256") != dataset.get(
        "raw_selected_dataset_sha256"
    ):
        errors.append(
            "dataset.selected_dataset_sha256 must equal raw selected identity"
        )

    split_records: dict[str, Mapping[str, Any]] = {}
    expected_splits = {"reference", "validation", "test"}
    for space in ("raw", "model"):
        field = f"{space}_split_sha256"
        split_record = dataset.get(field)
        if not isinstance(split_record, Mapping):
            errors.append(f"dataset.{field} must be an object")
            continue
        split_records[space] = split_record
        if set(split_record) != expected_splits:
            errors.append(
                f"dataset.{field} split inventory mismatch: "
                f"expected={sorted(expected_splits)}, got={sorted(split_record)}"
            )
        for split, digest in split_record.items():
            if not _is_sha256_digest(digest):
                errors.append(
                    f"dataset.{field}.{split} must be a lowercase SHA-256"
                )
    if configured.get("kind") == "identity":
        if dataset.get("raw_selected_dataset_sha256") != dataset.get(
            "model_selected_dataset_sha256"
        ):
            errors.append(
                "identity preprocessing must preserve selected dataset identity"
            )
        if split_records.get("raw") != split_records.get("model"):
            errors.append("identity preprocessing must preserve split identities")
    return errors


def validate_matrix(
    matrix_dir: Path, *, root: Path | None = None
) -> list[str]:
    project_root = (repository_root() if root is None else root).resolve()
    current_source_tree_sha256 = hash_declared_sources(project_root)
    matrix_path = matrix_dir / "matrix.json"
    if not matrix_path.is_file():
        return ["missing matrix.json"]
    try:
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read matrix.json: {exc}"]
    errors: list[str] = []
    if matrix.get("schema_version") != 1:
        errors.append("unsupported matrix schema_version")
    matrix_name = matrix.get("name")
    if (
        not isinstance(matrix_name, str)
        or not matrix_name
        or Path(matrix_name).name != matrix_name
    ):
        errors.append("matrix name must be a non-empty safe path component")
    matrix_backend = matrix.get("backend")
    matrix_evidence = matrix.get("evidence_level")
    expected_evidence = (
        _EVIDENCE_BY_BACKEND.get(matrix_backend)
        if isinstance(matrix_backend, str)
        else None
    )
    if expected_evidence is None:
        errors.append(f"unsupported matrix backend: {matrix_backend!r}")
    elif matrix_evidence != expected_evidence:
        errors.append(
            "matrix evidence_level is inconsistent with backend: "
            f"expected {expected_evidence!r}, got {matrix_evidence!r}"
        )
    for field in ("config_sha256", "input_sha256", "matrix_identity_sha256"):
        value = matrix.get(field)
        if not isinstance(value, str) or len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            errors.append(f"{field} must be a lowercase SHA-256")
    if all(
        isinstance(matrix.get(field), str)
        for field in ("config_sha256", "input_sha256", "matrix_identity_sha256")
    ):
        expected_matrix_identity = sha256_bytes(
            canonical_json(
                {
                    "config_sha256": matrix["config_sha256"],
                    "input_sha256": matrix["input_sha256"],
                }
            ).encode("utf-8")
        )
        if matrix.get("matrix_identity_sha256") != expected_matrix_identity:
            errors.append("matrix_identity_sha256 is inconsistent")
        if isinstance(matrix_name, str):
            expected_directory_name = (
                f"{matrix_name}__{expected_matrix_identity[:12]}"
            )
            if matrix_dir.name != expected_directory_name:
                errors.append(
                    "matrix directory name does not match matrix identity: "
                    f"expected {expected_directory_name}, got {matrix_dir.name}"
                )
    aggregate_relative = matrix.get("aggregate")
    if aggregate_relative != "aggregate.json":
        errors.append("matrix aggregate path must be aggregate.json")
    aggregate_path = matrix_dir / "aggregate.json"
    stored_aggregate: dict[str, Any] | None = None
    if not aggregate_path.is_file():
        errors.append("missing aggregate.json")
    else:
        if matrix.get("aggregate_sha256") != sha256_path(aggregate_path):
            errors.append("aggregate_sha256 mismatch")
        try:
            aggregate_value = json.loads(aggregate_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"cannot read aggregate.json: {exc}")
        else:
            if isinstance(aggregate_value, dict):
                stored_aggregate = aggregate_value
            else:
                errors.append("aggregate.json must contain an object")
    cells = matrix.get("cells")
    if not isinstance(cells, list):
        return ["matrix cells must be an array"]
    if matrix.get("requested_cells") != len(cells):
        errors.append("requested_cells does not match cell records")
    if matrix.get("complete_cells") != len(cells):
        errors.append("complete_cells does not match cell records")
    seen_cells: set[str] = set()
    actual_cell_keys: set[tuple[str, str]] = set()
    input_records: list[dict[str, Any]] = []
    representative_config: dict[str, Any] | None = None
    for cell in cells:
        if not isinstance(cell, dict) or "manifest" not in cell:
            errors.append("invalid cell record")
            continue
        cell_name = cell.get("cell")
        if not isinstance(cell_name, str) or not cell_name:
            errors.append("cell record must have a non-empty cell name")
            continue
        if cell_name in seen_cells:
            errors.append(f"duplicate cell record: {cell_name}")
            continue
        seen_cells.add(cell_name)
        if cell.get("status") not in {"completed", "reused"}:
            errors.append(f"{cell_name}: invalid completion status")
        manifest_relative = str(cell["manifest"])
        expected_manifest_relative = f"cells/{cell_name}/manifest.json"
        if manifest_relative != expected_manifest_relative:
            errors.append(
                f"{cell_name}: manifest path must be "
                f"{expected_manifest_relative}, got {manifest_relative}"
            )
            continue
        manifest_path = matrix_dir / manifest_relative
        if not manifest_path.is_file():
            errors.append(f"missing cell manifest: {cell['manifest']}")
            continue
        _, separator, expected_run_id = cell_name.rpartition("__")
        if not separator:
            errors.append(f"{cell_name}: cell name does not contain a run_id")
            expected_run_id = ""
        errors.extend(
            f"{cell_name}: {message}"
            for message in validate_manifest(
                manifest_path,
                expected_run_id=expected_run_id or None,
                expected_config_sha256=(
                    matrix.get("config_sha256")
                    if isinstance(matrix.get("config_sha256"), str)
                    else None
                ),
                expected_source_tree_sha256=current_source_tree_sha256,
            )
        )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            resolved_config = manifest["resolved_config"]
            if not isinstance(resolved_config, dict):
                raise TypeError("resolved_config is not an object")
            if representative_config is None:
                representative_config = resolved_config
            dataset = manifest["dataset"]
            if not isinstance(dataset, Mapping):
                raise TypeError("dataset is not an object")
            errors.extend(
                f"{cell_name}: {message}"
                for message in _dataset_source_anchor_errors(
                    dataset=dataset,
                    resolved_config=resolved_config,
                    root=project_root,
                )
            )
            errors.extend(
                f"{cell_name}: {message}"
                for message in _preprocessing_record_errors(
                    dataset=dataset,
                    resolved_config=resolved_config,
                )
            )
            for field in ("name", "backend", "evidence_level"):
                if resolved_config.get(field) != matrix.get(field):
                    errors.append(
                        f"{cell_name}: resolved_config.{field} disagrees with "
                        f"matrix {field}"
                    )
            try:
                resolved_config_sha256 = sha256_bytes(
                    canonical_json(resolved_config).encode("utf-8")
                )
            except (TypeError, ValueError) as exc:
                errors.append(
                    f"{cell_name}: cannot hash resolved_config canonically: {exc}"
                )
            else:
                if resolved_config_sha256 != matrix.get("config_sha256"):
                    errors.append(
                        f"{cell_name}: resolved_config hash disagrees with matrix "
                        "config_sha256"
                    )
            selected_sha = dataset.get(
                "raw_selected_dataset_sha256",
                dataset.get("selected_dataset_sha256", dataset["sha256"]),
            )
            dataset_key = (dataset["name"], dataset["representation"])
            if dataset_key in actual_cell_keys:
                errors.append(
                    f"duplicate dataset/representation cell: "
                    f"{dataset_key[0]}/{dataset_key[1]}"
                )
            actual_cell_keys.add(dataset_key)
            manifest_run_id = manifest.get("run_id")
            expected_cell_name = (
                f"{dataset_key[0]}__{dataset_key[1]}__{manifest_run_id}"
            )
            if cell_name != expected_cell_name:
                errors.append(
                    f"{cell_name}: cell name disagrees with manifest dataset, "
                    "representation, or run_id"
                )
            input_record = {
                "dataset": dataset_key[0],
                "representation": dataset_key[1],
                "selected_dataset_sha256": selected_sha,
                "model_selected_dataset_sha256": dataset[
                    "model_selected_dataset_sha256"
                ],
                "training_dataset_sha256": dataset["training_dataset_sha256"],
                "preprocessing_sha256": dataset["preprocessing"]["sha256"],
            }
            learned_input = dataset.get("learned_input")
            if isinstance(learned_input, dict):
                input_record["learned_input_sha256"] = learned_input["input_sha256"]
            input_records.append(input_record)
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            errors.append(f"{cell_name}: cannot reconstruct matrix input identity: {exc}")
    if not cells:
        errors.append("matrix must contain at least one cell")
    elif representative_config is not None:
        try:
            expected_cell_keys = _expected_cell_keys_from_config(
                representative_config, root=project_root
            )
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"cannot derive expected cell grid: {exc}")
        else:
            if actual_cell_keys != expected_cell_keys:
                missing = sorted(expected_cell_keys - actual_cell_keys)
                unexpected = sorted(actual_cell_keys - expected_cell_keys)
                errors.append(
                    "dataset/representation cell grid mismatch: "
                    f"missing={missing}, unexpected={unexpected}"
                )
            if matrix.get("requested_cells") != len(expected_cell_keys):
                errors.append(
                    "requested_cells does not match Hydra dataset/representation grid"
                )
    if isinstance(matrix.get("input_sha256"), str):
        actual_input_sha = sha256_bytes(
            canonical_json(input_records).encode("utf-8")
        )
        if matrix["input_sha256"] != actual_input_sha:
            errors.append("input_sha256 is inconsistent with cell manifests")
    cells_root = matrix_dir / "cells"
    if cells_root.is_dir():
        actual_cell_directories = {
            path.name for path in cells_root.iterdir() if path.is_dir()
        }
        non_directories = [path.name for path in cells_root.iterdir() if not path.is_dir()]
        if non_directories:
            errors.append(f"unexpected files in cells directory: {sorted(non_directories)}")
        if actual_cell_directories != seen_cells:
            errors.append(
                "cell directory inventory mismatch: "
                f"expected={sorted(seen_cells)}, actual={sorted(actual_cell_directories)}"
            )
    else:
        errors.append("missing cells directory")
    if not matrix.get("complete"):
        errors.append("matrix is not marked complete")
    if stored_aggregate is not None:
        aggregate_identity = stored_aggregate.get("matrix")
        expected_aggregate_identity = {
            field: matrix.get(field)
            for field in _AGGREGATE_MATRIX_IDENTITY_FIELDS
        }
        if aggregate_identity != expected_aggregate_identity:
            errors.append(
                "aggregate matrix identity/content disagrees with matrix.json"
            )
        try:
            expected_aggregate = recompute_aggregate_payload(
                matrix_dir, root=project_root
            )
            stored_canonical = canonical_json(stored_aggregate)
            expected_canonical = canonical_json(expected_aggregate)
        except (AggregateError, OSError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"cannot recompute aggregate from raw cell outputs: {exc}")
        else:
            if stored_canonical != expected_canonical:
                errors.append(
                    "aggregate content is inconsistent with raw cell outputs"
                )
    return errors
