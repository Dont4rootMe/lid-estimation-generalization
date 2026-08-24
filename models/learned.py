"""Strict consumer for learned-model field bundles.

Training implementations are deliberately outside this module: the local
paper specifies endpoint field interfaces, but no neural architecture or
optimizer.  A training stack earns a benchmark result only by exporting the
same versioned bundle contract, which is bound here to the exact checkpoint,
training config, selected dataset rows, scale and divergence estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt
from omegaconf import DictConfig, OmegaConf

from models.fields import (
    BundleContext,
    MODEL_FAMILY_READOUTS,
    evaluate_bundle,
    load_field_bundle,
    validate_bundle_provenance,
)
from utils.provenance import sha256_file


FloatArray = npt.NDArray[np.float64]
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class LearnedGridResult:
    """Predictions and immutable input identity for one dataset cell."""

    validation_curves: Mapping[str, FloatArray]
    test_curves: Mapping[str, FloatArray]
    input_files: Mapping[str, Mapping[str, str | int]]
    input_sha256: str
    checkpoint_sha256: str
    training_config_sha256: str
    trace: Mapping[str, str | int]


@dataclass(frozen=True)
class ModelArtifactIdentity:
    """Verified checkpoint and YAML training-config file identity."""

    input_files: Mapping[str, Mapping[str, str | int]]
    input_sha256: str
    checkpoint_sha256: str
    training_config_sha256: str


@dataclass(frozen=True)
class ArtifactRegistryIdentity:
    """Pinned YAML registry mapping every matrix cell to trained artifacts."""

    path: Path
    artifacts: Mapping[tuple[str, str], Mapping[str, str]]
    input_files: Mapping[str, Mapping[str, str | int]]
    input_sha256: str


def array_sha256(value: npt.ArrayLike) -> str:
    """Hash dtype, shape and bytes of one C-contiguous numeric array."""

    array = np.ascontiguousarray(np.asarray(value))
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _safe_component(value: Any, *, name: str) -> str:
    component = str(value)
    if (
        not component
        or component in {".", ".."}
        or Path(component).name != component
    ):
        raise ValueError(f"{name} must be one safe path component")
    return component


def _artifact_path(root: Path, value: Any, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path string")
    configured = Path(value)
    candidate = configured if configured.is_absolute() else root / configured
    if candidate.is_symlink():
        raise ValueError(f"{name} must not be a symlink: {candidate}")
    path = candidate.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing learned model artifact {name}: {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"{name} must be a non-empty regular file: {path}")
    return path


def _load_resolved_yaml_mapping(path: Path, *, name: str) -> dict[str, Any]:
    try:
        source = path.read_text(encoding="utf-8")
        if "${" in source:
            raise ValueError(f"{name} must not contain unresolved interpolation")
        value = OmegaConf.load(path)
        if not isinstance(value, DictConfig):
            raise ValueError(f"{name} must contain a YAML mapping")
        resolved = OmegaConf.to_container(value, resolve=True, throw_on_missing=True)
    except Exception as exc:
        raise ValueError(f"{name} is not resolved valid YAML: {path}") from exc
    if not isinstance(resolved, dict):
        raise ValueError(f"{name} must contain a YAML mapping")
    return resolved


def load_artifact_registry(
    *, root: Path, model: Mapping[str, Any]
) -> ArtifactRegistryIdentity:
    """Load an exactly SHA-pinned per-cell YAML artifact registry."""

    registry_path = _artifact_path(
        root, model.get("artifact_registry"), name="models.artifact_registry"
    )
    if registry_path.suffix != ".yaml":
        raise ValueError("models.artifact_registry must have the .yaml suffix")
    declared_sha256 = _sha256(
        model.get("artifact_registry_sha256", ""),
        name="models.artifact_registry_sha256",
    )
    actual_sha256 = sha256_file(registry_path)
    if actual_sha256 != declared_sha256:
        raise ValueError(
            "models.artifact_registry_sha256 does not match artifact_registry: "
            f"expected {declared_sha256}, got {actual_sha256}"
        )
    payload = _load_resolved_yaml_mapping(
        registry_path, name="models.artifact_registry"
    )
    if set(payload) != {"schema_version", "artifacts"}:
        raise ValueError(
            "artifact registry must contain exactly schema_version and artifacts"
        )
    if payload["schema_version"] != 1 or isinstance(payload["schema_version"], bool):
        raise ValueError("artifact registry schema_version must be 1")
    raw_artifacts = payload["artifacts"]
    if not isinstance(raw_artifacts, dict) or not raw_artifacts:
        raise ValueError("artifact registry artifacts must be a non-empty mapping")
    required = {
        "checkpoint_path",
        "checkpoint_sha256",
        "training_config_path",
        "training_config_sha256",
        "training_dataset_sha256",
        "preprocessing_sha256",
    }
    artifacts: dict[tuple[str, str], dict[str, str]] = {}
    for raw_key, raw_record in raw_artifacts.items():
        if not isinstance(raw_key, str) or raw_key.count("/") != 1:
            raise ValueError(
                "artifact registry keys must be '<dataset>/<representation>'"
            )
        dataset_name, representation = raw_key.split("/", 1)
        _safe_component(dataset_name, name="artifact dataset")
        _safe_component(representation, name="artifact representation")
        if not isinstance(raw_record, dict) or set(raw_record) != required:
            raise ValueError(
                f"artifact registry entry {raw_key!r} must contain exactly "
                f"{sorted(required)}"
            )
        record = {name: str(value) for name, value in raw_record.items()}
        for name in ("checkpoint_path", "training_config_path"):
            if not isinstance(raw_record[name], str) or not raw_record[name]:
                raise ValueError(
                    f"artifact registry {raw_key!r}.{name} must be a non-empty path"
                )
            relative = str(raw_record[name])
            pure = PurePosixPath(relative)
            if (
                "\\" in relative
                or pure.is_absolute()
                or pure.as_posix() != relative
                or any(part in {"", ".", ".."} for part in pure.parts)
            ):
                raise ValueError(
                    f"artifact registry {raw_key!r}.{name} must be a safe path "
                    "relative to the registry directory"
                )
        for name in (
            "checkpoint_sha256",
            "training_config_sha256",
            "training_dataset_sha256",
            "preprocessing_sha256",
        ):
            _sha256(record[name], name=f"artifact registry {raw_key!r}.{name}")
        artifacts[(dataset_name, representation)] = record
    registry_record: dict[str, dict[str, str | int]] = {
        "model/artifact_registry.yaml": {
            "path": str(registry_path),
            "size_bytes": registry_path.stat().st_size,
            "sha256": actual_sha256,
        }
    }
    return ArtifactRegistryIdentity(
        path=registry_path,
        artifacts=artifacts,
        input_files=registry_record,
        input_sha256=input_inventory_sha256(registry_record),
    )


def verify_model_artifacts(
    *,
    registry: ArtifactRegistryIdentity,
    model: Mapping[str, Any],
    dataset_name: str,
    representation: str,
    training_dataset_sha256: str,
    preprocessing_sha256: str,
) -> ModelArtifactIdentity:
    """Verify one cell's checkpoint and resolved Hydra training config."""

    key = (dataset_name, representation)
    try:
        record = registry.artifacts[key]
    except KeyError as exc:
        raise ValueError(
            "artifact registry has no entry for "
            f"{dataset_name!r}/{representation!r}"
        ) from exc
    expected_training_dataset = _sha256(
        training_dataset_sha256, name="training_dataset_sha256"
    )
    expected_preprocessing = _sha256(
        preprocessing_sha256, name="preprocessing_sha256"
    )
    if record["training_dataset_sha256"] != expected_training_dataset:
        raise ValueError(
            "artifact registry training_dataset_sha256 mismatch for "
            f"{dataset_name}/{representation}"
        )
    if record["preprocessing_sha256"] != expected_preprocessing:
        raise ValueError(
            "artifact registry preprocessing_sha256 mismatch for "
            f"{dataset_name}/{representation}"
        )

    artifact_root = registry.path.parent
    checkpoint_path = _artifact_path(
        artifact_root, record["checkpoint_path"], name="checkpoint_path"
    )
    training_config_path = _artifact_path(
        artifact_root,
        record["training_config_path"],
        name="training_config_path",
    )
    artifact_root_resolved = artifact_root.resolve()
    for name, path in (
        ("checkpoint_path", checkpoint_path),
        ("training_config_path", training_config_path),
    ):
        try:
            path.relative_to(artifact_root_resolved)
        except ValueError as exc:
            raise ValueError(
                f"{name} resolves outside the artifact registry directory"
            ) from exc
    if training_config_path.suffix != ".yaml":
        raise ValueError("training_config_path must have the .yaml suffix")
    training_config = _load_resolved_yaml_mapping(
        training_config_path, name="training_config_path"
    )
    provenance = training_config.get("provenance")
    expected_provenance = {
        "schema_version": 1,
        "model_name": str(model.get("name")),
        "model_family": str(model.get("family")),
        "model_seed": int(model.get("seed", -1)),
        "dataset_name": dataset_name,
        "representation": representation,
        "training_dataset_sha256": expected_training_dataset,
        "preprocessing_sha256": expected_preprocessing,
    }
    if provenance != expected_provenance:
        raise ValueError(
            "training config provenance mismatch: "
            f"expected {expected_provenance!r}, got {provenance!r}"
        )

    declared_checkpoint = record["checkpoint_sha256"]
    declared_training = record["training_config_sha256"]
    actual_checkpoint = sha256_file(checkpoint_path)
    actual_training = sha256_file(training_config_path)
    if actual_checkpoint != declared_checkpoint:
        raise ValueError(
            "checkpoint_sha256 does not match checkpoint_path: "
            f"expected {declared_checkpoint}, got {actual_checkpoint}"
        )
    if actual_training != declared_training:
        raise ValueError(
            "training_config_sha256 does not match training_config_path: "
            f"expected {declared_training}, got {actual_training}"
        )
    records: dict[str, dict[str, str | int]] = {
        name: dict(value) for name, value in registry.input_files.items()
    }
    records.update({
        "model/checkpoint": {
            "path": str(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
            "sha256": actual_checkpoint,
        },
        "model/training_config.yaml": {
            "path": str(training_config_path),
            "size_bytes": training_config_path.stat().st_size,
            "sha256": actual_training,
        },
    })
    return ModelArtifactIdentity(
        input_files=records,
        input_sha256=input_inventory_sha256(records),
        checkpoint_sha256=actual_checkpoint,
        training_config_sha256=actual_training,
    )


def bundle_paths(
    bundle_root: Path,
    *,
    model_name: str,
    model_seed: int,
    dataset_name: str,
    representation: str,
    scale_index: int,
    split: str,
) -> tuple[Path, Path]:
    """Return the canonical NPZ/JSON paths for one exported field bundle."""

    if split not in {"validation", "test"}:
        raise ValueError("split must be 'validation' or 'test'")
    if model_seed < 0 or scale_index < 0:
        raise ValueError("model_seed and scale_index must be non-negative")
    directory = (
        bundle_root
        / _safe_component(model_name, name="model_name")
        / f"seed-{model_seed}"
        / _safe_component(dataset_name, name="dataset_name")
        / _safe_component(representation, name="representation")
        / f"scale-{scale_index:03d}"
    )
    return directory / f"{split}.npz", directory / f"{split}.json"


def input_inventory_sha256(
    records: Mapping[str, Mapping[str, str | int]],
) -> str:
    """Hash logical name, size and content while excluding display paths."""

    identity_records: dict[str, dict[str, str | int]] = {}
    for logical_name, record in records.items():
        size = record.get("size_bytes")
        digest = record.get("sha256")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(
                f"input inventory {logical_name!r} has invalid size_bytes"
            )
        identity_records[str(logical_name)] = {
            "size_bytes": size,
            "sha256": _sha256(
                digest, name=f"input inventory {logical_name!r} sha256"
            ),
        }
    payload = json.dumps(
        identity_records,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256(value: Any, *, name: str) -> str:
    result = str(value)
    if _SHA256_RE.fullmatch(result) is None:
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256")
    return result


def _model_contract(model: Mapping[str, Any]) -> tuple[str, ...]:
    family = str(model.get("family", ""))
    if family not in MODEL_FAMILY_READOUTS:
        raise ValueError(f"unknown learned model family: {family!r}")
    raw_readouts = model.get("readouts")
    if not isinstance(raw_readouts, (list, tuple)):
        raise ValueError("learned model readouts must be an array")
    readouts = tuple(str(value) for value in raw_readouts)
    expected = MODEL_FAMILY_READOUTS[family]
    if readouts != expected:
        raise ValueError(
            f"model family {family!r} requires readouts {expected!r}; got {readouts!r}"
        )
    return readouts


def evaluate_field_grid(
    *,
    bundle_root: Path,
    model: Mapping[str, Any],
    dataset_name: str,
    training_dataset_sha256: str,
    dataset_sha256: str,
    representation: str,
    validation: npt.ArrayLike,
    test: npt.ArrayLike,
    preprocessing_sha256: str,
    model_space_validation: npt.ArrayLike,
    model_space_test: npt.ArrayLike,
    scales: Sequence[float],
) -> LearnedGridResult:
    """Load and evaluate every declared validation/test bundle in a scale grid."""

    readouts = _model_contract(model)
    model_name = _safe_component(model.get("name"), name="models.name")
    model_family = str(model["family"])
    model_seed = int(model.get("seed", -1))
    checkpoint_sha256 = _sha256(
        model.get("checkpoint_sha256", ""), name="models.checkpoint_sha256"
    )
    training_config_sha256 = _sha256(
        model.get("training_config_sha256", ""),
        name="models.training_config_sha256",
    )
    selected_dataset_sha256 = _sha256(
        dataset_sha256, name="dataset_sha256"
    )
    full_training_dataset_sha256 = _sha256(
        training_dataset_sha256, name="training_dataset_sha256"
    )
    preprocessing_digest = _sha256(
        preprocessing_sha256, name="preprocessing_sha256"
    )
    trace = model.get("trace")
    if not isinstance(trace, Mapping):
        raise ValueError("models.trace must be a mapping")
    trace_backend = str(trace.get("backend", ""))
    trace_probes = int(trace.get("probes", -1))
    trace_seed = int(trace.get("seed", -1))
    minimum_finite = float(model.get("min_finite_fraction", 1.0))
    if not 0.0 <= minimum_finite <= 1.0:
        raise ValueError("models.min_finite_fraction must lie in [0, 1]")

    query = {
        "validation": np.ascontiguousarray(np.asarray(validation)),
        "test": np.ascontiguousarray(np.asarray(test)),
    }
    for split, array in query.items():
        if array.ndim != 2 or array.shape[0] == 0:
            raise ValueError(f"{split} query must have non-empty shape (n, D)")
        if not np.isfinite(array).all():
            raise ValueError(f"{split} query contains non-finite values")
    model_query = {
        "validation": np.ascontiguousarray(np.asarray(model_space_validation)),
        "test": np.ascontiguousarray(np.asarray(model_space_test)),
    }
    for split, array in model_query.items():
        if array.ndim != 2 or array.shape[0] == 0:
            raise ValueError(
                f"{split} model-space query must have non-empty shape (n, D)"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{split} model-space query contains non-finite values")
        if array.shape[0] != query[split].shape[0]:
            raise ValueError(
                f"{split} source/model-space query row counts disagree: "
                f"{query[split].shape[0]} != {array.shape[0]}"
            )
    physical_scales = np.asarray(scales, dtype=np.float64)
    if (
        physical_scales.ndim != 1
        or physical_scales.size == 0
        or not np.isfinite(physical_scales).all()
        or np.any(physical_scales <= 0.0)
    ):
        raise ValueError("scales must be a non-empty positive finite vector")

    validation_curves = {
        readout_id: np.empty(
            (model_query["validation"].shape[0], physical_scales.size)
        )
        for readout_id in readouts
    }
    test_curves = {
        readout_id: np.empty((model_query["test"].shape[0], physical_scales.size))
        for readout_id in readouts
    }
    records: dict[str, dict[str, str | int]] = {}
    for scale_index, physical_scale in enumerate(physical_scales):
        for split in ("validation", "test"):
            npz_path, metadata_path = bundle_paths(
                bundle_root,
                model_name=model_name,
                model_seed=model_seed,
                dataset_name=dataset_name,
                representation=representation,
                scale_index=scale_index,
                split=split,
            )
            for path in (npz_path, metadata_path):
                if not path.is_file():
                    raise FileNotFoundError(f"missing learned field bundle artifact: {path}")
                relative = path.relative_to(bundle_root).as_posix()
                records[relative] = {
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            bundle = load_field_bundle(npz_path, metadata_path)
            context = BundleContext(
                model_name=model_name,
                model_family=model_family,
                model_seed=model_seed,
                checkpoint_sha256=checkpoint_sha256,
                training_config_sha256=training_config_sha256,
                dataset_name=dataset_name,
                training_dataset_sha256=full_training_dataset_sha256,
                dataset_sha256=selected_dataset_sha256,
                representation=representation,
                split=split,
                query_sha256=array_sha256(query[split]),
                preprocessing_sha256=preprocessing_digest,
                model_space_query_sha256=array_sha256(model_query[split]),
                n_samples=int(model_query[split].shape[0]),
                scale_index=scale_index,
                physical_scale=float(physical_scale),
                readout_ids=readouts,
                trace_backend=trace_backend,
                trace_probes=trace_probes,
                trace_seed=trace_seed,
            )
            validate_bundle_provenance(bundle, context)
            target = validation_curves if split == "validation" else test_curves
            for readout_id in readouts:
                prediction = evaluate_bundle(readout_id, bundle)
                if prediction.shape != (model_query[split].shape[0],):
                    raise ValueError(
                        f"{readout_id} prediction shape mismatch for {split}: "
                        f"{prediction.shape}"
                    )
                finite_fraction = float(np.isfinite(prediction).mean())
                if not math.isfinite(finite_fraction) or finite_fraction < minimum_finite:
                    raise ValueError(
                        f"{readout_id} finite fraction {finite_fraction:.6f} is below "
                        f"the declared threshold {minimum_finite:.6f}"
                    )
                target[readout_id][:, scale_index] = prediction

    return LearnedGridResult(
        validation_curves=validation_curves,
        test_curves=test_curves,
        input_files=records,
        input_sha256=input_inventory_sha256(records),
        checkpoint_sha256=checkpoint_sha256,
        training_config_sha256=training_config_sha256,
        trace={
            "backend": trace_backend,
            "probes": trace_probes,
            "seed": trace_seed,
        },
    )
