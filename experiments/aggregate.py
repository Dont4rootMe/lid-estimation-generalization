"""Deterministic aggregate reports reconstructed from immutable cell outputs.

The aggregator deliberately reads raw pointwise predictions rather than trusting
the per-cell summary JSON.  Pairwise statistics are emitted only when the
dataset registry declares row alignment (or when two representations share the
same selected-index digest).  Sample-size experiments remain collections of
independent summaries and are never converted into pointwise deltas.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt

from datasets.registry import DatasetRegistry, Representation, load_registry
from experiments.metrics import (
    known_lid_metrics,
    paired_delta_metrics,
    prediction_summary,
)
from experiments.run_manifest import validate_manifest, write_json


class AggregateError(RuntimeError):
    """Raised when a completed matrix cannot be aggregated safely."""


@dataclass(frozen=True)
class _Cell:
    dataset: str
    representation: str
    n_reference: int
    selected_indices_sha256: str | None
    paired_rows_sha256: str | None
    manifest_relative: str
    resolved_config: Mapping[str, Any]
    predictions: Mapping[str, npt.NDArray[np.float64]]
    target: npt.NDArray[np.float64] | None


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AggregateError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AggregateError(f"{label} {path} must contain a JSON object")
    return value


def _safe_child(root: Path, relative: Any, *, label: str) -> Path:
    value = Path(str(relative))
    if value.is_absolute():
        raise AggregateError(f"{label} must be relative to {root}: {value}")
    root_resolved = root.resolve()
    candidate = (root_resolved / value).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise AggregateError(f"{label} escapes {root}: {value}") from exc
    return candidate


def _load_vector(path: Path, *, label: str) -> npt.NDArray[np.float64]:
    try:
        value = np.load(path, allow_pickle=False)
        array = np.asarray(value, dtype=np.float64)
    except (OSError, ValueError, TypeError) as exc:
        raise AggregateError(f"cannot load {label} {path}: {exc}") from exc
    if array.ndim != 1:
        raise AggregateError(f"{label} {path} must have shape (n,), got {array.shape}")
    return np.ascontiguousarray(array)


def _prediction_id(relative: str) -> str | None:
    path = Path(relative)
    prefix = "prediction__"
    if path.name != relative or not relative.startswith(prefix) or not relative.endswith(
        ".npy"
    ):
        return None
    readout_id = relative[len(prefix) : -len(".npy")]
    return readout_id or None


def _load_cell(matrix_dir: Path, record: Mapping[str, Any]) -> _Cell:
    status = record.get("status")
    if status not in {"completed", "reused"}:
        raise AggregateError(f"cell is not completed or reusable: status={status!r}")
    manifest_relative = str(record.get("manifest", ""))
    manifest_path = _safe_child(
        matrix_dir, manifest_relative, label="cell manifest path"
    )
    try:
        integrity_errors = validate_manifest(manifest_path)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise AggregateError(f"cannot validate cell manifest {manifest_path}: {exc}") from exc
    if integrity_errors:
        raise AggregateError(
            f"cell manifest {manifest_path} failed integrity checks: {integrity_errors}"
        )
    manifest = _read_json_object(manifest_path, label="cell manifest")
    dataset = manifest.get("dataset")
    outputs = manifest.get("outputs")
    resolved_config = manifest.get("resolved_config")
    if not isinstance(dataset, Mapping):
        raise AggregateError(f"cell manifest {manifest_path} has no dataset object")
    if not isinstance(outputs, Mapping):
        raise AggregateError(f"cell manifest {manifest_path} has no outputs object")
    if not isinstance(resolved_config, Mapping):
        raise AggregateError(
            f"cell manifest {manifest_path} has no resolved_config object"
        )

    dataset_name = dataset.get("name")
    representation = dataset.get("representation")
    if not isinstance(dataset_name, str) or not dataset_name:
        raise AggregateError(f"cell manifest {manifest_path} has invalid dataset name")
    if not isinstance(representation, str) or not representation:
        raise AggregateError(f"cell manifest {manifest_path} has invalid representation")
    try:
        n_reference = int(dataset["n_reference"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AggregateError(
            f"cell manifest {manifest_path} has invalid n_reference"
        ) from exc
    if n_reference <= 0:
        raise AggregateError(f"cell manifest {manifest_path} has non-positive n_reference")

    cell_dir = manifest_path.parent
    predictions: dict[str, npt.NDArray[np.float64]] = {}
    for relative_raw in sorted(outputs):
        if not isinstance(relative_raw, str):
            raise AggregateError(f"cell manifest {manifest_path} has non-string output path")
        readout_id = _prediction_id(relative_raw)
        if readout_id is None:
            continue
        if readout_id in predictions:
            raise AggregateError(
                f"cell manifest {manifest_path} repeats prediction {readout_id!r}"
            )
        prediction_path = _safe_child(
            cell_dir, relative_raw, label="prediction output path"
        )
        predictions[readout_id] = _load_vector(
            prediction_path, label=f"prediction {readout_id!r}"
        )
    if not predictions:
        raise AggregateError(f"cell manifest {manifest_path} declares no predictions")

    target: npt.NDArray[np.float64] | None = None
    if "test_target.npy" in outputs:
        target_path = _safe_child(
            cell_dir, "test_target.npy", label="test target output path"
        )
        target = _load_vector(target_path, label="test target")
        prediction_sizes = {value.shape for value in predictions.values()}
        if prediction_sizes != {target.shape}:
            raise AggregateError(
                f"target/prediction shapes disagree in {manifest_path}: "
                f"target={target.shape}, predictions={sorted(prediction_sizes)}"
            )

    selected_digest_raw = dataset.get("selected_indices_sha256")
    selected_digest = (
        selected_digest_raw if isinstance(selected_digest_raw, str) else None
    )
    paired_rows_raw = dataset.get("paired_rows_sha256")
    paired_rows = paired_rows_raw if isinstance(paired_rows_raw, str) else None
    return _Cell(
        dataset=dataset_name,
        representation=representation,
        n_reference=n_reference,
        selected_indices_sha256=selected_digest,
        paired_rows_sha256=paired_rows,
        manifest_relative=manifest_relative,
        resolved_config=dict(resolved_config),
        predictions=predictions,
        target=target,
    )


def _load_completed_matrix(matrix_dir: Path) -> tuple[dict[str, Any], list[_Cell]]:
    matrix_path = matrix_dir / "matrix.json"
    matrix = _read_json_object(matrix_path, label="matrix")
    records = matrix.get("cells")
    if not isinstance(records, list):
        raise AggregateError("matrix cells must be an array")
    if matrix.get("complete") is not True:
        raise AggregateError("matrix must be complete before aggregation")
    if matrix.get("requested_cells") != len(records):
        raise AggregateError("matrix requested_cells does not match cell records")
    if matrix.get("complete_cells") != len(records):
        raise AggregateError("matrix complete_cells does not match cell records")

    cells: list[_Cell] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise AggregateError("matrix contains a non-object cell record")
        cell = _load_cell(matrix_dir, record)
        key = (cell.dataset, cell.representation)
        if key in seen:
            raise AggregateError(
                "matrix repeats dataset/representation cell "
                f"{cell.dataset!r}/{cell.representation!r}"
            )
        seen.add(key)
        cells.append(cell)
    cells.sort(key=lambda item: (item.dataset, item.representation))
    return matrix, cells


def _load_matrix_registry(
    cells: list[_Cell], *, project_root: Path
) -> DatasetRegistry | None:
    declared: set[str] = set()
    for cell in cells:
        dataset_config = cell.resolved_config.get("dataset")
        if isinstance(dataset_config, Mapping):
            registry = dataset_config.get("registry")
            if isinstance(registry, str) and registry:
                declared.add(registry)
    if not declared:
        return None
    if len(declared) != 1:
        raise AggregateError(
            f"matrix cells disagree on dataset registry: {sorted(declared)}"
        )
    registry_path = Path(next(iter(declared)))
    if not registry_path.is_absolute():
        registry_path = project_root / registry_path
    try:
        return load_registry(registry_path)
    except (OSError, ValueError) as exc:
        raise AggregateError(f"cannot load matrix dataset registry: {exc}") from exc


def _known_lid_records(cells: list[_Cell]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for cell in cells:
        if cell.target is None:
            continue
        for readout_id in sorted(cell.predictions):
            records.append(
                {
                    "dataset": cell.dataset,
                    "representation": cell.representation,
                    "readout_id": readout_id,
                    "metrics": known_lid_metrics(
                        cell.predictions[readout_id], cell.target
                    ),
                }
            )
    return records


def _alignment_error(left: _Cell, right: _Cell) -> str | None:
    if not left.selected_indices_sha256 or not right.selected_indices_sha256:
        return "missing_selected_indices_sha256"
    if left.selected_indices_sha256 != right.selected_indices_sha256:
        return "selected_indices_sha256_mismatch"
    return None


def _transformation_alignment_error(left: _Cell, right: _Cell) -> str | None:
    issue = _alignment_error(left, right)
    if issue is not None:
        return issue
    if not left.paired_rows_sha256 or not right.paired_rows_sha256:
        return "missing_paired_rows_sha256"
    if left.paired_rows_sha256 != right.paired_rows_sha256:
        return "paired_rows_sha256_mismatch"
    return None


def _paired_transformation_records(
    cells: list[_Cell], registry: DatasetRegistry | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if registry is None:
        return [], []
    by_key = {(cell.dataset, cell.representation): cell for cell in cells}
    records: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for transformed in cells:
        if transformed.dataset not in registry:
            continue
        pair = registry[transformed.dataset].transformation
        if pair is None or not pair.paired_samples:
            continue
        reference = by_key.get((pair.reference, transformed.representation))
        for readout_id in sorted(transformed.predictions):
            issue: str | None = None
            if reference is None:
                issue = "missing_reference_cell"
            elif readout_id not in reference.predictions:
                issue = "missing_reference_prediction"
            else:
                issue = _transformation_alignment_error(transformed, reference)
                if (
                    issue is None
                    and transformed.predictions[readout_id].shape
                    != reference.predictions[readout_id].shape
                ):
                    issue = "prediction_shape_mismatch"
            if issue is not None:
                missing.append(
                    {
                        "dataset": transformed.dataset,
                        "reference_dataset": pair.reference,
                        "representation": transformed.representation,
                        "readout_id": readout_id,
                        "reason": issue,
                    }
                )
                continue
            assert reference is not None
            records.append(
                {
                    "dataset": transformed.dataset,
                    "reference_dataset": pair.reference,
                    "family": pair.family,
                    "parameter": pair.parameter,
                    "parameter_value": pair.value,
                    "expected_delta": float(pair.expected_lid_delta),
                    "paired_samples": True,
                    "representation": transformed.representation,
                    "readout_id": readout_id,
                    "n": int(transformed.predictions[readout_id].size),
                    "metrics": paired_delta_metrics(
                        reference.predictions[readout_id],
                        transformed.predictions[readout_id],
                        expected_delta=pair.expected_lid_delta,
                    ),
                }
            )
    return records, missing


def _point_sort_key(point: Mapping[str, Any]) -> tuple[int, float, str]:
    if point.get("role") == "reference":
        return (0, 0.0, str(point.get("dataset", "")))
    value = point.get("parameter_value")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (1, float(value), str(point.get("dataset", "")))
    return (1, float("inf"), f"{value!s}:{point.get('dataset', '')}")


def _sample_size_curves(
    cells: list[_Cell], registry: DatasetRegistry | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if registry is None:
        return [], []
    by_key = {(cell.dataset, cell.representation): cell for cell in cells}
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    missing: list[dict[str, Any]] = []
    for cell in cells:
        if cell.dataset not in registry:
            continue
        spec = registry[cell.dataset]
        pair = spec.transformation
        if pair is None or pair.paired_samples or pair.family != "sample_size":
            continue
        for readout_id, prediction in sorted(cell.predictions.items()):
            key = (pair.reference, cell.representation, readout_id)
            group = groups.setdefault(
                key,
                {
                    "family": pair.family,
                    "reference_dataset": pair.reference,
                    "representation": cell.representation,
                    "readout_id": readout_id,
                    "paired_samples": False,
                    "points": [],
                },
            )
            group["points"].append(
                {
                    "dataset": cell.dataset,
                    "role": "sample_size_variant",
                    "parameter": pair.parameter,
                    "parameter_value": pair.value,
                    "available_train_samples": spec.expected_samples.get("train"),
                    "n_reference_used": cell.n_reference,
                    "summary": prediction_summary(prediction),
                }
            )

    curves: list[dict[str, Any]] = []
    for key in sorted(groups):
        reference_name, representation, readout_id = key
        group = groups[key]
        reference = by_key.get((reference_name, representation))
        if reference is None:
            missing.append(
                {
                    "reference_dataset": reference_name,
                    "representation": representation,
                    "readout_id": readout_id,
                    "reason": "missing_reference_cell",
                }
            )
        elif readout_id not in reference.predictions:
            missing.append(
                {
                    "reference_dataset": reference_name,
                    "representation": representation,
                    "readout_id": readout_id,
                    "reason": "missing_reference_prediction",
                }
            )
        else:
            reference_spec = registry[reference_name]
            group["points"].append(
                {
                    "dataset": reference_name,
                    "role": "reference",
                    "parameter": None,
                    "parameter_value": None,
                    "available_train_samples": reference_spec.expected_samples.get(
                        "train"
                    ),
                    "n_reference_used": reference.n_reference,
                    "summary": prediction_summary(reference.predictions[readout_id]),
                }
            )
        group["points"].sort(key=_point_sort_key)
        curves.append(group)
    return curves, missing


def _representation_records(
    cells: list[_Cell], registry: DatasetRegistry | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if registry is None:
        return [], []
    by_key = {(cell.dataset, cell.representation): cell for cell in cells}
    records: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    dataset_names = sorted({cell.dataset for cell in cells if cell.dataset in registry})
    for dataset_name in dataset_names:
        spec = registry[dataset_name]
        available = {item.value for item in spec.available_representations}
        if not {
            Representation.DATASET.value,
            Representation.COEFFICIENTS.value,
        }.issubset(available):
            continue
        related = [cell for cell in cells if cell.dataset == dataset_name]
        representation_modes = {
            dataset_config.get("representations")
            for cell in related
            if isinstance(
                dataset_config := cell.resolved_config.get("dataset"), Mapping
            )
        }
        if representation_modes and representation_modes != {"all"}:
            continue
        image = by_key.get((dataset_name, Representation.DATASET.value))
        coefficients = by_key.get(
            (dataset_name, Representation.COEFFICIENTS.value)
        )
        requested_readouts: set[str] = set()
        for cell in related:
            config_readouts = cell.resolved_config.get("readouts")
            if isinstance(config_readouts, list):
                requested_readouts.update(
                    item for item in config_readouts if isinstance(item, str)
                )
            else:
                requested_readouts.update(cell.predictions)
        for readout_id in sorted(requested_readouts):
            issue: str | None = None
            if image is None or coefficients is None:
                issue = "missing_representation_cell"
            elif readout_id not in image.predictions or readout_id not in coefficients.predictions:
                issue = "missing_representation_prediction"
            else:
                issue = _alignment_error(image, coefficients)
                if (
                    issue is None
                    and image.predictions[readout_id].shape
                    != coefficients.predictions[readout_id].shape
                ):
                    issue = "prediction_shape_mismatch"
            if issue is not None:
                missing.append(
                    {
                        "dataset": dataset_name,
                        "readout_id": readout_id,
                        "reason": issue,
                    }
                )
                continue
            assert image is not None and coefficients is not None
            records.append(
                {
                    "dataset": dataset_name,
                    "readout_id": readout_id,
                    "base_representation": Representation.COEFFICIENTS.value,
                    "transformed_representation": Representation.DATASET.value,
                    "delta_direction": "dataset_minus_coefficients",
                    "expected_delta": 0.0,
                    "n": int(image.predictions[readout_id].size),
                    "metrics": paired_delta_metrics(
                        coefficients.predictions[readout_id],
                        image.predictions[readout_id],
                        expected_delta=0.0,
                    ),
                }
            )
    return records, missing


def recompute_aggregate_payload(
    matrix_dir: str | Path,
    *,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Recompute matrix-level analyses from raw cell outputs without writing.

    This is deliberately shared by the writer and the independent matrix
    validator.  Keeping the computation side-effect free lets an auditor
    compare a stored aggregate with the raw evidence without first replacing
    the artifact that is being audited.
    """

    directory = Path(matrix_dir).resolve()
    if not directory.is_dir():
        raise AggregateError(f"matrix directory does not exist: {directory}")
    matrix, cells = _load_completed_matrix(directory)
    project_root = Path.cwd().resolve() if root is None else Path(root).resolve()
    registry = _load_matrix_registry(cells, project_root=project_root)

    known_lid = _known_lid_records(cells)
    paired, paired_missing = _paired_transformation_records(cells, registry)
    sample_curves, sample_missing = _sample_size_curves(cells, registry)
    representations, representation_missing = _representation_records(cells, registry)
    missing = {
        "paired_transformations": paired_missing,
        "sample_size_curves": sample_missing,
        "representation_discrepancies": representation_missing,
    }
    prediction_records = sum(len(cell.predictions) for cell in cells)
    aggregate = {
        "schema_version": 1,
        "matrix": {
            "name": matrix.get("name"),
            "backend": matrix.get("backend"),
            "evidence_level": matrix.get("evidence_level"),
            "config_sha256": matrix.get("config_sha256"),
            "input_sha256": matrix.get("input_sha256"),
            "matrix_identity_sha256": matrix.get("matrix_identity_sha256"),
        },
        "coverage": {
            "complete": not any(missing.values()),
            "cells_expected": int(matrix["requested_cells"]),
            "cells_loaded": len(cells),
            "prediction_records": prediction_records,
            "known_lid_records": len(known_lid),
            "paired_transformation_expected": len(paired) + len(paired_missing),
            "paired_transformation_records": len(paired),
            "sample_size_curve_records": len(sample_curves),
            "representation_discrepancy_expected": len(representations)
            + len(representation_missing),
            "representation_discrepancy_records": len(representations),
            "missing": missing,
        },
        "known_lid": known_lid,
        "paired_transformations": paired,
        "sample_size_curves": sample_curves,
        "representation_discrepancies": representations,
    }
    return aggregate


def aggregate_matrix(
    matrix_dir: str | Path,
    *,
    root: str | Path | None = None,
    output_name: str = "aggregate.json",
) -> Path:
    """Recompute and write deterministic matrix-level analyses."""

    directory = Path(matrix_dir).resolve()
    if Path(output_name).name != output_name or not output_name:
        raise AggregateError("output_name must be a single safe path component")
    aggregate = recompute_aggregate_payload(directory, root=root)
    output_path = directory / output_name
    write_json(output_path, aggregate)
    return output_path


__all__ = ["AggregateError", "aggregate_matrix", "recompute_aggregate_payload"]
