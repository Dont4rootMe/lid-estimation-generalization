"""Validated access to the official LID-Benchmarks ``.npy`` layout.

The upstream archive stores one directory per benchmark and one directory per
split::

    <root>/<dataset>/{train,val,test}/{dataset,lid,coefficients,labels}.npy

Not every benchmark has every artifact.  This module deliberately keeps the
files in their upstream layout and applies any correction in memory, after the
stored value has been validated against the registry.  In particular, this is
how the known Spaghetti target defect is handled without patching the pinned
upstream checkout or downloaded benchmark files.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import yaml


ARTIFACT_NAMES = ("dataset", "lid", "coefficients", "labels")
SPLIT_NAMES = ("train", "val", "test")

# This is the table in the README at upstream revision 2dcb8e4.  Keeping the
# list here as well as in YAML makes accidental registry omissions testable.
OFFICIAL_README_DATASETS = (
    *(f"e1_sampled_fmnist_step{step}" for step in range(1, 14)),
    "e1_spiral_pca",
    "e2_arrows",
    "e2_uniform_pca",
    "e5_padded_fmnist_adddim0",
    "e5_padded_fmnist_adddim4",
    "e5_padded_fmnist_adddim8",
    "e5_stretched_power0.25",
    "e5_stretched_power4",
    "e5_upscaled_fmnist",
    "e6_exp_pca",
    "e7_crescent_moon_radius3.0",
    "e8_gaussian4_pca",
    "e8_spaghetti_pca",
    "e8_sphere4_pca",
)


class DatasetValidationError(ValueError):
    """Raised when a registry entry or on-disk split violates its contract."""


class RegistryValidationError(DatasetValidationError):
    """Raised when the declarative dataset registry is inconsistent."""


class Representation(str, Enum):
    """Feature representation exposed to an estimator."""

    DATASET = "dataset"
    COEFFICIENTS = "coefficients"


def _as_representation(value: Representation | str) -> Representation:
    try:
        return value if isinstance(value, Representation) else Representation(value)
    except ValueError as exc:
        valid = ", ".join(item.value for item in Representation)
        raise DatasetValidationError(
            f"unknown representation {value!r}; expected one of: {valid}"
        ) from exc


def _constant_is_valid(value: float | None, field_name: str) -> None:
    if value is not None and (not math.isfinite(value) or value <= 0):
        raise DatasetValidationError(
            f"{field_name} must be a finite positive number, got {value!r}"
        )


@dataclass(frozen=True)
class TransformationPair:
    """A paired benchmark comparison with a known LID change.

    ``reference`` names the unmodified (or control) dataset, while
    ``expected_lid_delta`` is ``LID(transformed) - LID(reference)``.  The
    remaining fields are descriptive and are included in result provenance.
    """

    family: str
    reference: str
    expected_lid_delta: float
    parameter: str | None = None
    value: str | int | float | None = None
    paired_samples: bool = True

    def __post_init__(self) -> None:
        if not self.family.strip():
            raise RegistryValidationError("transformation family cannot be empty")
        if not self.reference.strip():
            raise RegistryValidationError("transformation reference cannot be empty")
        if not math.isfinite(float(self.expected_lid_delta)):
            raise RegistryValidationError(
                "transformation expected_lid_delta must be finite"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TransformationPair":
        allowed = {
            "family",
            "reference",
            "expected_lid_delta",
            "parameter",
            "value",
            "paired_samples",
        }
        unknown = set(value) - allowed
        if unknown:
            raise RegistryValidationError(
                f"unknown transformation fields: {sorted(unknown)}"
            )
        try:
            return cls(
                family=str(value["family"]),
                reference=str(value["reference"]),
                expected_lid_delta=float(value["expected_lid_delta"]),
                parameter=(
                    None if value.get("parameter") is None else str(value["parameter"])
                ),
                value=value.get("value"),
                paired_samples=bool(value.get("paired_samples", True)),
            )
        except KeyError as exc:
            raise RegistryValidationError(
                f"transformation is missing required field {exc.args[0]!r}"
            ) from exc


@dataclass(frozen=True)
class DatasetSpec:
    """Declarative validation contract for one upstream benchmark directory."""

    name: str
    section: str = ""
    representation: Representation | str = Representation.DATASET
    available_representations: tuple[Representation | str, ...] = (
        Representation.DATASET,
    )
    required_artifacts: tuple[str, ...] = ("dataset",)
    ignored_files: tuple[str, ...] = ()
    splits: tuple[str, ...] = SPLIT_NAMES
    official: bool = True
    expected_samples: Mapping[str, int] = field(default_factory=dict)
    expected_shapes: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    expected_lid: float | None = None
    stored_lid: float | None = None
    stored_lid_by_split: Mapping[str, float] = field(default_factory=dict)
    lid_override: float | None = None
    allowed_lid_values: tuple[float, ...] = ()
    transformation: TransformationPair | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if (
            not self.name
            or self.name in {".", ".."}
            or Path(self.name).name != self.name
        ):
            raise RegistryValidationError(
                f"dataset name must be a single safe path component, got {self.name!r}"
            )

        representation = _as_representation(self.representation)
        available = tuple(
            _as_representation(item) for item in self.available_representations
        )
        if not available:
            raise RegistryValidationError(
                f"dataset {self.name!r} has no available representations"
            )
        if len(set(available)) != len(available):
            raise RegistryValidationError(
                f"dataset {self.name!r} repeats an available representation"
            )
        if representation not in available:
            raise RegistryValidationError(
                f"default representation {representation.value!r} is not available "
                f"for dataset {self.name!r}"
            )

        required = tuple(self.required_artifacts)
        unknown_artifacts = set(required) - set(ARTIFACT_NAMES)
        if unknown_artifacts:
            raise RegistryValidationError(
                f"dataset {self.name!r} requires unknown artifacts "
                f"{sorted(unknown_artifacts)}"
            )
        if len(set(required)) != len(required):
            raise RegistryValidationError(
                f"dataset {self.name!r} repeats a required artifact"
            )
        for item in available:
            if item.value not in required:
                raise RegistryValidationError(
                    f"dataset {self.name!r} advertises {item.value!r}, but does not "
                    "require the corresponding artifact"
                )

        ignored_files = tuple(self.ignored_files)
        for filename in ignored_files:
            if (
                not filename
                or Path(filename).name != filename
                or not filename.endswith(".npy")
            ):
                raise RegistryValidationError(
                    f"dataset {self.name!r} has unsafe ignored file {filename!r}"
                )
            if filename.removesuffix(".npy") in ARTIFACT_NAMES:
                raise RegistryValidationError(
                    f"dataset {self.name!r} cannot ignore canonical artifact "
                    f"{filename!r}"
                )
        if len(set(ignored_files)) != len(ignored_files):
            raise RegistryValidationError(
                f"dataset {self.name!r} repeats an ignored file"
            )

        splits = tuple(self.splits)
        invalid_splits = set(splits) - set(SPLIT_NAMES)
        if invalid_splits or not splits:
            raise RegistryValidationError(
                f"dataset {self.name!r} has invalid splits: {sorted(invalid_splits)}"
            )
        if len(set(splits)) != len(splits):
            raise RegistryValidationError(f"dataset {self.name!r} repeats a split")

        expected_samples = {
            str(key): int(value) for key, value in self.expected_samples.items()
        }
        if set(expected_samples) - set(splits):
            raise RegistryValidationError(
                f"dataset {self.name!r} gives sizes for undeclared splits"
            )
        if any(value <= 0 for value in expected_samples.values()):
            raise RegistryValidationError(
                f"dataset {self.name!r} expected split sizes must be positive"
            )

        expected_shapes: dict[str, tuple[int, ...]] = {}
        for artifact, shape in self.expected_shapes.items():
            if artifact not in ARTIFACT_NAMES:
                raise RegistryValidationError(
                    f"dataset {self.name!r} gives a shape for unknown artifact "
                    f"{artifact!r}"
                )
            normalized_shape = tuple(int(dim) for dim in shape)
            if any(dim <= 0 for dim in normalized_shape):
                raise RegistryValidationError(
                    f"dataset {self.name!r} has invalid {artifact!r} shape "
                    f"{normalized_shape}"
                )
            expected_shapes[artifact] = normalized_shape

        _constant_is_valid(self.expected_lid, "expected_lid")
        _constant_is_valid(self.stored_lid, "stored_lid")
        _constant_is_valid(self.lid_override, "lid_override")
        stored_lid_by_split = {
            str(key): float(value)
            for key, value in self.stored_lid_by_split.items()
        }
        invalid_stored_splits = set(stored_lid_by_split) - set(splits)
        if invalid_stored_splits:
            raise RegistryValidationError(
                f"dataset {self.name!r} gives stored LID values for undeclared "
                f"splits: {sorted(invalid_stored_splits)}"
            )
        if stored_lid_by_split and set(stored_lid_by_split) != set(splits):
            missing_stored_splits = set(splits) - set(stored_lid_by_split)
            raise RegistryValidationError(
                f"dataset {self.name!r} split-aware stored LID policy must cover "
                f"every declared split; missing={sorted(missing_stored_splits)}"
            )
        for split_name, value in stored_lid_by_split.items():
            _constant_is_valid(value, f"stored_lid_by_split[{split_name!r}]")
        if self.stored_lid is not None and stored_lid_by_split:
            raise RegistryValidationError(
                f"dataset {self.name!r} cannot set both stored_lid and "
                "stored_lid_by_split"
            )
        allowed_lids = tuple(float(value) for value in self.allowed_lid_values)
        if any(not math.isfinite(value) or value <= 0 for value in allowed_lids):
            raise RegistryValidationError(
                f"dataset {self.name!r} has invalid allowed_lid_values"
            )
        if self.expected_lid is not None and allowed_lids:
            raise RegistryValidationError(
                f"dataset {self.name!r} cannot set both expected_lid and "
                "allowed_lid_values"
            )
        if self.lid_override is not None:
            if self.stored_lid is None and not stored_lid_by_split:
                raise RegistryValidationError(
                    f"dataset {self.name!r} must validate stored_lid or "
                    "stored_lid_by_split before applying a lid_override"
                )
            if self.expected_lid is None or not math.isclose(
                float(self.expected_lid), float(self.lid_override)
            ):
                raise RegistryValidationError(
                    f"dataset {self.name!r} lid_override must equal expected_lid"
                )
        if (
            self.expected_lid is not None
            or self.stored_lid is not None
            or stored_lid_by_split
            or allowed_lids
        ) and "lid" not in required:
            raise RegistryValidationError(
                f"dataset {self.name!r} declares a LID contract but does not require "
                "lid.npy"
            )
        if self.transformation and self.transformation.reference == self.name:
            raise RegistryValidationError(
                f"dataset {self.name!r} cannot be its own transformation reference"
            )

        object.__setattr__(self, "representation", representation)
        object.__setattr__(self, "available_representations", available)
        object.__setattr__(self, "required_artifacts", required)
        object.__setattr__(self, "ignored_files", ignored_files)
        object.__setattr__(self, "splits", splits)
        object.__setattr__(
            self, "expected_samples", MappingProxyType(expected_samples)
        )
        object.__setattr__(
            self, "expected_shapes", MappingProxyType(expected_shapes)
        )
        object.__setattr__(
            self,
            "stored_lid_by_split",
            MappingProxyType(stored_lid_by_split),
        )
        object.__setattr__(self, "allowed_lid_values", allowed_lids)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DatasetSpec":
        allowed = {
            "name",
            "section",
            "representation",
            "available_representations",
            "required_artifacts",
            "ignored_files",
            "splits",
            "official",
            "expected_samples",
            "expected_shapes",
            "expected_lid",
            "stored_lid",
            "stored_lid_by_split",
            "lid_override",
            "allowed_lid_values",
            "transformation",
            "notes",
        }
        unknown = set(value) - allowed
        if unknown:
            name = value.get("name", "<unnamed>")
            raise RegistryValidationError(
                f"dataset {name!r} has unknown fields: {sorted(unknown)}"
            )
        try:
            transformation_value = value.get("transformation")
            transformation = (
                None
                if transformation_value is None
                else TransformationPair.from_mapping(transformation_value)
            )
            expected_shapes = {
                str(key): tuple(int(dim) for dim in shape)
                for key, shape in value.get("expected_shapes", {}).items()
            }
            return cls(
                name=str(value["name"]),
                section=str(value.get("section", "")),
                representation=str(value.get("representation", "dataset")),
                available_representations=tuple(
                    value.get("available_representations", ("dataset",))
                ),
                required_artifacts=tuple(
                    value.get("required_artifacts", ("dataset",))
                ),
                ignored_files=tuple(value.get("ignored_files", ())),
                splits=tuple(value.get("splits", SPLIT_NAMES)),
                official=bool(value.get("official", True)),
                expected_samples=value.get("expected_samples", {}),
                expected_shapes=expected_shapes,
                expected_lid=(
                    None
                    if value.get("expected_lid") is None
                    else float(value["expected_lid"])
                ),
                stored_lid=(
                    None
                    if value.get("stored_lid") is None
                    else float(value["stored_lid"])
                ),
                stored_lid_by_split=value.get("stored_lid_by_split", {}),
                lid_override=(
                    None
                    if value.get("lid_override") is None
                    else float(value["lid_override"])
                ),
                allowed_lid_values=tuple(value.get("allowed_lid_values", ())),
                transformation=transformation,
                notes=str(value.get("notes", "")),
            )
        except KeyError as exc:
            raise RegistryValidationError(
                f"dataset entry is missing required field {exc.args[0]!r}"
            ) from exc

    def with_representation(
        self, representation: Representation | str
    ) -> "DatasetSpec":
        """Return a copy selecting another declared feature representation."""

        selected = _as_representation(representation)
        if selected not in self.available_representations:
            available = ", ".join(item.value for item in self.available_representations)
            raise DatasetValidationError(
                f"representation {selected.value!r} is unavailable for {self.name!r}; "
                f"available: {available}"
            )
        return replace(self, representation=selected)


@dataclass(frozen=True)
class RegistryCoverage:
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.missing and not self.unexpected

    def raise_for_errors(self) -> None:
        if not self.complete:
            raise RegistryValidationError(
                "official README coverage mismatch: "
                f"missing={list(self.missing)}, unexpected={list(self.unexpected)}"
            )


@dataclass(frozen=True)
class DatasetRegistry(Mapping[str, DatasetSpec]):
    """Immutable, name-addressable collection of dataset specifications."""

    benchmark_id: str
    schema_version: int
    specs: Mapping[str, DatasetSpec]
    official_readme_folders: tuple[str, ...] = ()
    upstream_registry: str | None = None

    def __post_init__(self) -> None:
        specs = dict(self.specs)
        if not self.benchmark_id.strip():
            raise RegistryValidationError("benchmark_id cannot be empty")
        if self.schema_version != 1:
            raise RegistryValidationError(
                f"unsupported dataset registry schema_version {self.schema_version}"
            )
        if any(name != spec.name for name, spec in specs.items()):
            raise RegistryValidationError("dataset registry keys must match spec names")
        declared = tuple(self.official_readme_folders)
        if len(set(declared)) != len(declared):
            raise RegistryValidationError("official_readme_folders contains duplicates")
        official_specs = {name for name, spec in specs.items() if spec.official}
        if declared and official_specs != set(declared):
            raise RegistryValidationError(
                "official dataset flags do not match official_readme_folders"
            )
        for spec in specs.values():
            pair = spec.transformation
            if pair and pair.reference not in specs:
                raise RegistryValidationError(
                    f"dataset {spec.name!r} references absent transformation control "
                    f"{pair.reference!r}"
                )
        object.__setattr__(self, "specs", MappingProxyType(specs))
        object.__setattr__(self, "official_readme_folders", declared)

    def __getitem__(self, key: str) -> DatasetSpec:
        return self.specs[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.specs)

    def __len__(self) -> int:
        return len(self.specs)

    @property
    def official_specs(self) -> tuple[DatasetSpec, ...]:
        return tuple(self.specs[name] for name in self.official_readme_folders)

    def coverage(
        self, expected: Sequence[str] = OFFICIAL_README_DATASETS
    ) -> RegistryCoverage:
        actual = {spec.name for spec in self.specs.values() if spec.official}
        wanted = set(expected)
        return RegistryCoverage(
            missing=tuple(sorted(wanted - actual)),
            unexpected=tuple(sorted(actual - wanted)),
        )


def load_registry(
    path: str | Path,
    *,
    validate_official_coverage: bool = True,
) -> DatasetRegistry:
    """Read and validate a YAML dataset registry."""

    registry_path = Path(path)
    try:
        with registry_path.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise RegistryValidationError(
            f"cannot read dataset registry {registry_path}: {exc}"
        ) from exc
    if not isinstance(raw, Mapping):
        raise RegistryValidationError(
            f"dataset registry {registry_path} must contain a YAML mapping"
        )

    allowed_top_level = {
        "schema_version",
        "benchmark_id",
        "description",
        "upstream_registry",
        "official_readme_folders",
        "datasets",
    }
    unknown = set(raw) - allowed_top_level
    if unknown:
        raise RegistryValidationError(
            f"unknown top-level dataset registry fields: {sorted(unknown)}"
        )
    rows = raw.get("datasets")
    if not isinstance(rows, list) or not rows:
        raise RegistryValidationError("dataset registry must contain [[datasets]]")
    specs: dict[str, DatasetSpec] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise RegistryValidationError("each dataset entry must be a YAML mapping")
        spec = DatasetSpec.from_mapping(row)
        if spec.name in specs:
            raise RegistryValidationError(
                f"duplicate dataset registry entry {spec.name!r}"
            )
        specs[spec.name] = spec

    registry = DatasetRegistry(
        benchmark_id=str(raw.get("benchmark_id", registry_path.stem)),
        schema_version=int(raw.get("schema_version", 1)),
        specs=specs,
        official_readme_folders=tuple(raw.get("official_readme_folders", ())),
        upstream_registry=(
            None
            if raw.get("upstream_registry") is None
            else str(raw["upstream_registry"])
        ),
    )
    if (
        validate_official_coverage
        and registry.benchmark_id == "lid-benchmarks-iclr-2026"
    ):
        registry.coverage().raise_for_errors()
    return registry


def apply_registry_overlay(
    registry: DatasetRegistry,
    path: str | Path,
) -> DatasetRegistry:
    """Apply a narrowly scoped source-policy overlay to a base registry.

    Overlays are intended for artifacts produced by the pinned generator when
    those differ from the canonical paper archive.  They may alter only the
    LID policy and explanatory notes; benchmark identity, shapes, splits, and
    transformation contracts remain inherited from the canonical registry.
    """

    overlay_path = Path(path)
    try:
        with overlay_path.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise RegistryValidationError(
            f"cannot read dataset registry overlay {overlay_path}: {exc}"
        ) from exc
    if not isinstance(raw, Mapping):
        raise RegistryValidationError(
            f"dataset registry overlay {overlay_path} must contain a YAML mapping"
        )
    allowed_top_level = {
        "schema_version",
        "overlay_id",
        "base_benchmark_id",
        "source_kind",
        "upstream_revision",
        "overrides",
    }
    unknown = set(raw) - allowed_top_level
    if unknown:
        raise RegistryValidationError(
            f"unknown top-level overlay fields: {sorted(unknown)}"
        )
    if int(raw.get("schema_version", 1)) != 1:
        raise RegistryValidationError("unsupported dataset overlay schema_version")
    if raw.get("base_benchmark_id") != registry.benchmark_id:
        raise RegistryValidationError(
            "dataset overlay base_benchmark_id does not match loaded registry"
        )
    rows = raw.get("overrides")
    if not isinstance(rows, list) or not rows:
        raise RegistryValidationError("dataset overlay must contain [[overrides]]")

    specs = dict(registry.specs)
    seen: set[str] = set()
    allowed_override_fields = {
        "name",
        "expected_lid",
        "stored_lid",
        "stored_lid_by_split",
        "lid_override",
        "notes",
    }
    for row in rows:
        if not isinstance(row, Mapping):
            raise RegistryValidationError("each overlay entry must be a YAML mapping")
        unknown_fields = set(row) - allowed_override_fields
        if unknown_fields:
            raise RegistryValidationError(
                f"unknown dataset overlay fields: {sorted(unknown_fields)}"
            )
        if "name" not in row:
            raise RegistryValidationError("dataset overlay entry is missing 'name'")
        name = str(row["name"])
        if name in seen:
            raise RegistryValidationError(f"duplicate dataset overlay {name!r}")
        seen.add(name)
        if name not in specs:
            raise RegistryValidationError(
                f"dataset overlay refers to absent dataset {name!r}"
            )
        updates: dict[str, Any] = {}
        for field_name in ("expected_lid", "stored_lid", "lid_override"):
            if field_name in row:
                updates[field_name] = (
                    None if row[field_name] is None else float(row[field_name])
                )
        if "stored_lid_by_split" in row:
            raw_split_values = row["stored_lid_by_split"]
            if not isinstance(raw_split_values, Mapping):
                raise RegistryValidationError(
                    "dataset overlay stored_lid_by_split must be a YAML mapping"
                )
            updates["stored_lid_by_split"] = {
                str(split): float(raw_lid)
                for split, raw_lid in raw_split_values.items()
            }
        if "notes" in row:
            updates["notes"] = str(row["notes"])
        specs[name] = replace(specs[name], **updates)

    return DatasetRegistry(
        benchmark_id=registry.benchmark_id,
        schema_version=registry.schema_version,
        specs=specs,
        official_readme_folders=registry.official_readme_folders,
        upstream_registry=registry.upstream_registry,
    )


def _is_real_numeric(array: np.ndarray) -> bool:
    return np.issubdtype(array.dtype, np.integer) or np.issubdtype(
        array.dtype, np.floating
    )


def _all_finite(array: np.ndarray, chunk_elements: int = 1_000_000) -> bool:
    if np.issubdtype(array.dtype, np.integer):
        return True
    flat = array.reshape(-1)
    return all(
        bool(np.isfinite(flat[start : start + chunk_elements]).all())
        for start in range(0, flat.size, chunk_elements)
    )


def _all_close_to(array: np.ndarray, value: float) -> bool:
    flat = array.reshape(-1)
    return all(
        bool(np.allclose(flat[start : start + 1_000_000], value, rtol=0, atol=1e-8))
        for start in range(0, flat.size, 1_000_000)
    )


@dataclass(frozen=True)
class LoadedSplit:
    """A validated split and the exact metadata used to interpret it."""

    spec: DatasetSpec
    split: str
    arrays: Mapping[str, np.ndarray]
    representation: Representation | str | None = None
    source_paths: Mapping[str, Path] = field(default_factory=dict)
    applied_overrides: Mapping[str, Mapping[str, float | str]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.split not in self.spec.splits:
            raise DatasetValidationError(
                f"split {self.split!r} is not declared for {self.spec.name!r}"
            )
        representation_value = (
            self.spec.representation
            if self.representation is None
            else self.representation
        )
        representation = _as_representation(representation_value)
        if representation not in self.spec.available_representations:
            raise DatasetValidationError(
                f"representation {representation.value!r} is not declared for "
                f"{self.spec.name!r}"
            )

        arrays = dict(self.arrays)
        unknown = set(arrays) - set(ARTIFACT_NAMES)
        if unknown:
            raise DatasetValidationError(
                f"split {self.spec.name}/{self.split} has unknown arrays "
                f"{sorted(unknown)}"
            )
        missing = set(self.spec.required_artifacts) - set(arrays)
        if missing:
            raise DatasetValidationError(
                f"split {self.spec.name}/{self.split} is missing required arrays "
                f"{sorted(missing)}"
            )
        if representation.value not in arrays:
            raise DatasetValidationError(
                f"split {self.spec.name}/{self.split} has no "
                f"{representation.value}.npy"
            )

        n_samples: int | None = None
        for name, array in arrays.items():
            if not isinstance(array, np.ndarray):
                raise DatasetValidationError(
                    f"artifact {name!r} is not a numpy array"
                )
            if not _is_real_numeric(array):
                raise DatasetValidationError(
                    f"artifact {name!r} must have a real numeric dtype, got "
                    f"{array.dtype}"
                )
            if name == "labels" and not np.issubdtype(array.dtype, np.integer):
                raise DatasetValidationError(
                    f"labels must have an integer dtype, got {array.dtype}"
                )
            if name == "coefficients" and array.ndim != 2:
                raise DatasetValidationError(
                    f"coefficients must be rank 2, got shape {array.shape}"
                )
            if name == "dataset" and array.ndim < 2:
                raise DatasetValidationError(
                    f"dataset must include a sample and feature axis, got "
                    f"shape {array.shape}"
                )
            if name in {"lid", "labels"} and array.ndim != 1:
                raise DatasetValidationError(
                    f"{name} must be rank 1, got shape {array.shape}"
                )
            if array.shape[0] == 0 or any(dim == 0 for dim in array.shape[1:]):
                raise DatasetValidationError(
                    f"artifact {name!r} cannot have an empty dimension: {array.shape}"
                )
            if n_samples is None:
                n_samples = int(array.shape[0])
            elif array.shape[0] != n_samples:
                raise DatasetValidationError(
                    f"artifact {name!r} has {array.shape[0]} samples; expected "
                    f"{n_samples}"
                )
            if not _all_finite(array):
                raise DatasetValidationError(
                    f"artifact {name!r} contains NaN or infinity"
                )
            expected_shape = self.spec.expected_shapes.get(name)
            if expected_shape is not None and array.shape[1:] != expected_shape:
                raise DatasetValidationError(
                    f"artifact {name!r} has trailing shape {array.shape[1:]}; "
                    f"expected {expected_shape}"
                )

        assert n_samples is not None  # a selected feature artifact is required
        expected_count = self.spec.expected_samples.get(self.split)
        if expected_count is not None and n_samples != expected_count:
            raise DatasetValidationError(
                f"split {self.spec.name}/{self.split} has {n_samples} samples; "
                f"expected {expected_count}"
            )

        lid = arrays.get("lid")
        if self.spec.expected_lid is not None:
            if lid is None or not _all_close_to(lid, self.spec.expected_lid):
                raise DatasetValidationError(
                    f"effective lid for {self.spec.name}/{self.split} does not match "
                    f"expected value {self.spec.expected_lid}"
                )
        if self.spec.allowed_lid_values:
            if lid is None:
                raise DatasetValidationError(
                    f"{self.spec.name}/{self.split} is missing lid.npy"
                )
            allowed = np.asarray(self.spec.allowed_lid_values, dtype=np.float64)
            observed = np.asarray(lid, dtype=np.float64)
            valid = np.any(np.isclose(observed[:, None], allowed[None, :]), axis=1)
            if not bool(valid.all()):
                invalid = np.unique(observed[~valid])
                raise DatasetValidationError(
                    f"{self.spec.name}/{self.split} contains unexpected LID values "
                    f"{invalid.tolist()}"
                )

        source_paths = {
            str(key): Path(value) for key, value in self.source_paths.items()
        }
        if set(source_paths) - set(arrays):
            raise DatasetValidationError("source_paths names must match loaded arrays")
        overrides = {
            str(key): MappingProxyType(dict(value))
            for key, value in self.applied_overrides.items()
        }
        object.__setattr__(self, "arrays", MappingProxyType(arrays))
        object.__setattr__(self, "source_paths", MappingProxyType(source_paths))
        object.__setattr__(self, "applied_overrides", MappingProxyType(overrides))
        object.__setattr__(self, "representation", representation)

    @property
    def features(self) -> np.ndarray:
        return self.arrays[self.representation.value]

    @property
    def x(self) -> np.ndarray:
        """Alias used by estimator adapters."""

        return self.features

    @property
    def lid(self) -> np.ndarray | None:
        return self.arrays.get("lid")

    @property
    def labels(self) -> np.ndarray | None:
        return self.arrays.get("labels")

    @property
    def dataset(self) -> np.ndarray | None:
        return self.arrays.get("dataset")

    @property
    def coefficients(self) -> np.ndarray | None:
        return self.arrays.get("coefficients")

    @property
    def n_samples(self) -> int:
        return int(self.features.shape[0])

    @property
    def feature_shape(self) -> tuple[int, ...]:
        return tuple(int(dim) for dim in self.features.shape[1:])

    @property
    def flat_feature_dim(self) -> int:
        return int(np.prod(self.feature_shape, dtype=np.int64))


def load_split(
    benchmark_root: str | Path,
    spec: DatasetSpec,
    split: str,
    *,
    representation: Representation | str | None = None,
    mmap_mode: str | None = None,
) -> LoadedSplit:
    """Load one split, validate it, and apply registry-declared corrections.

    The source arrays are always loaded with ``allow_pickle=False``.  Unknown
    ``.npy`` files are rejected so that a misspelled artifact cannot silently
    disappear from provenance.
    """

    if split not in spec.splits:
        raise DatasetValidationError(
            f"split {split!r} is not declared for dataset {spec.name!r}"
        )
    selected = _as_representation(
        spec.representation if representation is None else representation
    )
    if selected not in spec.available_representations:
        available = ", ".join(item.value for item in spec.available_representations)
        raise DatasetValidationError(
            f"representation {selected.value!r} is unavailable for {spec.name!r}; "
            f"available: {available}"
        )

    split_dir = Path(benchmark_root) / spec.name / split
    if not split_dir.is_dir():
        raise DatasetValidationError(
            f"dataset split directory does not exist: {split_dir}"
        )
    paths: dict[str, Path] = {}
    for path in sorted(split_dir.glob("*.npy")):
        if path.name in spec.ignored_files:
            continue
        if path.stem not in ARTIFACT_NAMES:
            raise DatasetValidationError(
                f"unknown artifact file in {split_dir}: {path.name}"
            )
        paths[path.stem] = path
    missing = set(spec.required_artifacts) - set(paths)
    if missing:
        raise DatasetValidationError(
            f"split {spec.name}/{split} is missing files "
            f"{[name + '.npy' for name in sorted(missing)]}"
        )
    if selected.value not in paths:
        raise DatasetValidationError(
            f"split {spec.name}/{split} is missing {selected.value}.npy"
        )

    arrays: dict[str, np.ndarray] = {}
    for name, path in paths.items():
        try:
            arrays[name] = np.load(
                path,
                allow_pickle=False,
                mmap_mode=mmap_mode,
            )
        except (OSError, ValueError) as exc:
            raise DatasetValidationError(f"cannot load {path}: {exc}") from exc

    lid = arrays.get("lid")
    stored_lid = spec.stored_lid_by_split.get(split, spec.stored_lid)
    if stored_lid is not None and (
        lid is None
        or not _is_real_numeric(lid)
        or not _all_close_to(lid, stored_lid)
    ):
        raise DatasetValidationError(
            f"stored lid for {spec.name}/{split} does not match registry value "
            f"{stored_lid}; refusing to apply any override"
        )

    applied_overrides: dict[str, Mapping[str, float | str]] = {}
    if spec.lid_override is not None:
        assert lid is not None  # guaranteed by DatasetSpec and stored check above
        arrays["lid"] = np.full(lid.shape, spec.lid_override, dtype=lid.dtype)
        applied_overrides["lid"] = {
            "kind": "constant_after_stored_value_validation",
            "stored": float(stored_lid),  # type: ignore[arg-type]
            "effective": float(spec.lid_override),
        }

    return LoadedSplit(
        spec=spec,
        split=split,
        arrays=arrays,
        representation=selected,
        source_paths=paths,
        applied_overrides=applied_overrides,
    )


def load_dataset(
    benchmark_root: str | Path,
    spec: DatasetSpec,
    *,
    representation: Representation | str | None = None,
    mmap_mode: str | None = None,
) -> Mapping[str, LoadedSplit]:
    """Load every declared split for ``spec``."""

    return MappingProxyType(
        {
            split: load_split(
                benchmark_root,
                spec,
                split,
                representation=representation,
                mmap_mode=mmap_mode,
            )
            for split in spec.splits
        }
    )
