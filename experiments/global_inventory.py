"""Fail-closed inventory for the global learned-model benchmark campaign.

The canonical paper archive and the generated E3/E4 extension are deliberately
separate inventories.  In particular, generator classes that happen to exist
upstream do not make their outputs canonical paper data.  This module validates
the YAML inventory against the corresponding dataset registry and expands each
dataset into explicit dataset x representation cells without touching data.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from datasets.archive import EXACT_ARCHIVE_SHA256
from datasets.registry import (
    DatasetRegistry,
    RegistryValidationError,
    apply_registry_overlay,
    load_registry,
)
from utils.provenance import EXPECTED_LID_BENCHMARKS_SHA, sha256_file

SCHEMA_VERSION = 1
CANONICAL_INVENTORY_CONFIG = Path("configs/global_suite/canonical_exact.yaml")
GENERATED_E3_E4_INVENTORY_CONFIG = Path("configs/global_suite/generated_e3_e4.yaml")

_SOURCE_KINDS = frozenset({"exact_archive", "generated_at_pinned_revision"})
_PROVENANCE_LABELS = {
    "exact_archive": "canonical-exact-paper-archive",
    "generated_at_pinned_revision": "generated-extension-not-canonical-exact",
}
_AVAILABILITY = frozenset(
    {"canonical_exact", "generated_extension", "absent_from_source"}
)
_TARGET_POLICIES = frozenset({"known_lid", "sample_size", "paired_delta"})
_SELECTION_PROTOCOLS = frozenset(
    {"supervised_train_mae", "target_free_train_stability"}
)
_SUITE_ID = re.compile(r"e[1-8]\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_SHA = re.compile(r"[0-9a-f]{40}\Z")

_APPROVED_INVENTORY_FILES = {
    "exact_archive": {
        "inventory_id": "lid-benchmarks-canonical-exact-all-representations-v1",
        "inventory_path": CANONICAL_INVENTORY_CONFIG.as_posix(),
        "inventory_sha256": "d9b0f002edf317a4aace773e214bdd533b012ba5bcfe66da78b4b5495239c867",
        "dataset_config": "configs/datasets/lid_benchmarks.yaml",
        "dataset_config_sha256": "16cb930a273927dcb26d69a0bf44f016898f5de717adfafe18415e53def9ec07",
        "data_root": "data/lid_benchmarks_exact/benchmarks",
        "registry": "configs/datasets/registry/paper_benchmarks.yaml",
        "registry_sha256": "08ac45cf701446dc6185ca2f5b83ddf05c4ce6d15c0dec0b499d4633ca397ab0",
        "registry_overlay": None,
        "registry_overlay_sha256": None,
        "suite_order": ("e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8"),
        "counts": {
            "suite_count": 8,
            "available_suite_count": 6,
            "dataset_count": 28,
            "cell_count": 35,
            "known_lid_dataset_count": 8,
            "known_lid_cell_count": 15,
            "target_free_dataset_count": 20,
            "target_free_cell_count": 20,
        },
    },
    "generated_at_pinned_revision": {
        "inventory_id": "lid-benchmarks-generated-e3-e4-all-representations-v1",
        "inventory_path": GENERATED_E3_E4_INVENTORY_CONFIG.as_posix(),
        "inventory_sha256": "c671a94e413c3c81948c8d86b7dfcd54722e3e1d15f9cd097f206708d1f0e757",
        "dataset_config": "configs/datasets/lid_benchmarks_generated_e3_e4.yaml",
        "dataset_config_sha256": "2dc097ca47ff557f7726a2a6a610b41c54c3400324c1fe5ddfa3223f46cc1a03",
        "data_root": "data/generated_benchmarks",
        "registry": "configs/datasets/registry/generated_e3_e4_extension.yaml",
        "registry_sha256": "07ff2c27900ca128917ea5c42986c9fb2c744eb6696fdb28bf3e1185be966454",
        "registry_overlay": "configs/datasets/registry/generated_e3_e4_policy.yaml",
        "registry_overlay_sha256": "8aae1698acbc602def57328e181d6dead69ddcd824f24d84e70f686ec5dd3c8e",
        "suite_order": ("e3", "e4"),
        "counts": {
            "suite_count": 2,
            "available_suite_count": 2,
            "dataset_count": 2,
            "cell_count": 4,
            "known_lid_dataset_count": 2,
            "known_lid_cell_count": 4,
            "target_free_dataset_count": 0,
            "target_free_cell_count": 0,
        },
    },
}

_TOP_LEVEL_FIELDS = frozenset(
    {"schema_version", "inventory_id", "source", "suite_order", "expected", "suites"}
)
_SOURCE_FIELDS = frozenset(
    {
        "kind",
        "exact_archive",
        "dataset_config",
        "data_root",
        "registry",
        "registry_overlay",
        "upstream_revision",
        "archive_sha256",
        "provenance_label",
    }
)
_GLOBAL_COUNT_FIELDS = frozenset(
    {
        "suite_count",
        "available_suite_count",
        "dataset_count",
        "cell_count",
        "known_lid_dataset_count",
        "known_lid_cell_count",
        "target_free_dataset_count",
        "target_free_cell_count",
    }
)
_SUITE_FIELDS = frozenset({"id", "availability", "expected", "datasets"})
_SUITE_COUNT_FIELDS = _GLOBAL_COUNT_FIELDS - {
    "suite_count",
    "available_suite_count",
}
_DATASET_FIELDS = frozenset(
    {
        "name",
        "representations",
        "target_policy",
        "selection_protocol",
        "comparison_group",
        "reference_dataset",
        "expected_lid_delta",
    }
)
_DATASET_CONFIG_FIELDS = frozenset(
    {
        "source",
        "source_kind",
        "archive",
        "extracted_root",
        "root",
        "registry",
        "registry_overlay",
        "representations",
        "names",
    }
)


class InventoryError(ValueError):
    """Raised when a global-suite inventory is incomplete or mislabelled."""


@dataclass(frozen=True)
class InventorySource:
    """Portable provenance declaration shared by every inventory cell."""

    kind: str
    exact_archive: bool
    dataset_config: str
    data_root: str
    registry: str
    registry_overlay: str | None
    upstream_revision: str
    archive_sha256: str | None
    provenance_label: str


@dataclass(frozen=True)
class InventoryCell:
    """One independently trained dataset x representation campaign cell."""

    suite_id: str
    dataset: str
    representation: str
    has_lid_targets: bool
    target_policy: str
    selection_protocol: str
    comparison_group: str
    reference_dataset: str | None
    expected_lid_delta: float | None

    @property
    def key(self) -> tuple[str, str]:
        return self.dataset, self.representation


@dataclass(frozen=True)
class SuiteDefinition:
    """Ordered cells and sealed counts for one upstream ``eN`` bucket."""

    suite_id: str
    availability: str
    cells: tuple[InventoryCell, ...]
    expected_dataset_count: int
    expected_cell_count: int
    expected_known_lid_dataset_count: int
    expected_known_lid_cell_count: int
    expected_target_free_dataset_count: int
    expected_target_free_cell_count: int

    @property
    def dataset_names(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(cell.dataset for cell in self.cells))

    @property
    def available(self) -> bool:
        return self.availability != "absent_from_source"


@dataclass(frozen=True)
class GlobalInventory:
    """Validated, deterministic inventory consumed by a campaign orchestrator."""

    inventory_id: str
    source: InventorySource
    suites: tuple[SuiteDefinition, ...]
    expected_suite_count: int
    expected_available_suite_count: int
    expected_dataset_count: int
    expected_cell_count: int
    expected_known_lid_dataset_count: int
    expected_known_lid_cell_count: int
    expected_target_free_dataset_count: int
    expected_target_free_cell_count: int

    @property
    def cells(self) -> tuple[InventoryCell, ...]:
        return tuple(cell for suite in self.suites for cell in suite.cells)

    @property
    def dataset_names(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(cell.dataset for cell in self.cells))

    @property
    def known_lid_cells(self) -> tuple[InventoryCell, ...]:
        return tuple(cell for cell in self.cells if cell.has_lid_targets)

    @property
    def target_free_cells(self) -> tuple[InventoryCell, ...]:
        return tuple(cell for cell in self.cells if not cell.has_lid_targets)

    def suite(self, suite_id: str) -> SuiteDefinition:
        for suite in self.suites:
            if suite.suite_id == suite_id:
                return suite
        raise KeyError(suite_id)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _mapping(value: object, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InventoryError(f"{path} must be a mapping")
    return value


def _sequence(value: object, *, path: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise InventoryError(f"{path} must be a YAML array")
    return value


def _reject_unknown(
    value: Mapping[str, Any], allowed: frozenset[str], *, path: str
) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise InventoryError(f"unknown fields in {path}: {sorted(unknown)}")


def _nonempty_string(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise InventoryError(f"{path} must be a non-empty trimmed string")
    return value


def _optional_string(value: object, *, path: str) -> str | None:
    if value is None:
        return None
    return _nonempty_string(value, path=path)


def _count(value: object, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InventoryError(f"{path} must be a non-negative integer")
    return value


def _portable_path(value: object, *, path: str) -> str:
    raw = _nonempty_string(value, path=path)
    parsed = PurePosixPath(raw)
    if parsed.is_absolute() or ".." in parsed.parts or str(parsed) != raw:
        raise InventoryError(f"{path} must be a normalized project-relative path")
    return raw


def _read_yaml(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise InventoryError(f"cannot read {label} {path}: {exc}") from exc
    return _mapping(raw, path=label)


def _parse_source(raw: object) -> InventorySource:
    source = _mapping(raw, path="source")
    _reject_unknown(source, _SOURCE_FIELDS, path="source")
    kind = _nonempty_string(source.get("kind"), path="source.kind")
    if kind not in _SOURCE_KINDS:
        raise InventoryError(f"source.kind must be one of {sorted(_SOURCE_KINDS)}")
    exact_archive = source.get("exact_archive")
    if not isinstance(exact_archive, bool):
        raise InventoryError("source.exact_archive must be a boolean")
    if exact_archive != (kind == "exact_archive"):
        raise InventoryError("source.exact_archive disagrees with source.kind")

    upstream_revision = _nonempty_string(
        source.get("upstream_revision"), path="source.upstream_revision"
    )
    if not _GIT_SHA.fullmatch(upstream_revision):
        raise InventoryError("source.upstream_revision must be a lowercase Git SHA")
    if upstream_revision != EXPECTED_LID_BENCHMARKS_SHA:
        raise InventoryError(
            "source.upstream_revision differs from the pinned LID-Benchmarks revision"
        )
    archive_sha256 = _optional_string(
        source.get("archive_sha256"), path="source.archive_sha256"
    )
    if exact_archive:
        if archive_sha256 is None or not _SHA256.fullmatch(archive_sha256):
            raise InventoryError("exact archive inventory requires archive_sha256")
        if archive_sha256 != EXACT_ARCHIVE_SHA256:
            raise InventoryError(
                "source.archive_sha256 differs from the canonical exact archive"
            )
        overlay = _optional_string(
            source.get("registry_overlay"), path="source.registry_overlay"
        )
        if overlay is not None:
            raise InventoryError("exact archive inventory forbids registry_overlay")
    else:
        if archive_sha256 is not None:
            raise InventoryError("generated inventory must not declare archive_sha256")
        overlay = _portable_path(
            source.get("registry_overlay"), path="source.registry_overlay"
        )

    provenance_label = _nonempty_string(
        source.get("provenance_label"), path="source.provenance_label"
    )
    if provenance_label != _PROVENANCE_LABELS[kind]:
        raise InventoryError(
            f"source.provenance_label must be exactly {_PROVENANCE_LABELS[kind]!r}"
        )

    return InventorySource(
        kind=kind,
        exact_archive=exact_archive,
        dataset_config=_portable_path(
            source.get("dataset_config"), path="source.dataset_config"
        ),
        data_root=_portable_path(source.get("data_root"), path="source.data_root"),
        registry=_portable_path(source.get("registry"), path="source.registry"),
        registry_overlay=overlay,
        upstream_revision=upstream_revision,
        archive_sha256=archive_sha256,
        provenance_label=provenance_label,
    )


def _parse_counts(raw: object, *, path: str, global_counts: bool) -> dict[str, int]:
    value = _mapping(raw, path=path)
    expected_fields = _GLOBAL_COUNT_FIELDS if global_counts else _SUITE_COUNT_FIELDS
    _reject_unknown(value, expected_fields, path=path)
    missing = expected_fields - set(value)
    if missing:
        raise InventoryError(f"missing fields in {path}: {sorted(missing)}")
    return {key: _count(value[key], path=f"{path}.{key}") for key in expected_fields}


def _load_source_registry(
    source: InventorySource, *, project_root: Path
) -> DatasetRegistry:
    try:
        registry = load_registry(project_root / source.registry)
        if source.registry_overlay is not None:
            registry = apply_registry_overlay(
                registry, project_root / source.registry_overlay
            )
    except RegistryValidationError as exc:
        raise InventoryError(f"invalid inventory dataset registry: {exc}") from exc
    return registry


def _validate_dataset_config(
    source: InventorySource,
    *,
    project_root: Path,
    dataset_names: tuple[str, ...],
) -> None:
    raw = _read_yaml(project_root / source.dataset_config, label="dataset config")
    _reject_unknown(raw, _DATASET_CONFIG_FIELDS, path="dataset config")
    if raw.get("source") != "lid_benchmarks":
        raise InventoryError("dataset config source must be 'lid_benchmarks'")
    if raw.get("source_kind") != source.kind:
        raise InventoryError("dataset config source_kind disagrees with inventory")
    if raw.get("root") != source.data_root:
        raise InventoryError("dataset config root disagrees with inventory")
    if raw.get("registry") != source.registry:
        raise InventoryError("dataset config registry disagrees with inventory")
    if raw.get("registry_overlay") != source.registry_overlay:
        raise InventoryError("dataset config registry_overlay disagrees with inventory")
    if raw.get("representations") != "all":
        raise InventoryError(
            "global inventory dataset config must select all representations"
        )
    names = _sequence(raw.get("names"), path="dataset config.names")
    normalized = tuple(
        _nonempty_string(item, path=f"dataset config.names[{index}]")
        for index, item in enumerate(names)
    )
    if normalized != dataset_names:
        raise InventoryError(
            "dataset config names/order disagree with the global inventory"
        )


def _parse_dataset_cells(
    raw: object,
    *,
    path: str,
    suite_id: str,
    registry: DatasetRegistry,
) -> tuple[InventoryCell, ...]:
    row = _mapping(raw, path=path)
    _reject_unknown(row, _DATASET_FIELDS, path=path)
    missing = _DATASET_FIELDS - set(row)
    if missing:
        raise InventoryError(f"missing fields in {path}: {sorted(missing)}")
    name = _nonempty_string(row["name"], path=f"{path}.name")
    if not name.startswith(f"{suite_id}_"):
        raise InventoryError(f"{path}.name does not belong to suite {suite_id}")
    if name not in registry:
        raise InventoryError(f"{path}.name is absent from the declared registry")
    spec = registry[name]

    representation_rows = _sequence(
        row["representations"], path=f"{path}.representations"
    )
    representations = tuple(
        _nonempty_string(value, path=f"{path}.representations[{index}]")
        for index, value in enumerate(representation_rows)
    )
    if not representations or len(representations) != len(set(representations)):
        raise InventoryError(f"{path}.representations must be unique and non-empty")
    available = tuple(value.value for value in spec.available_representations)
    if representations != available:
        raise InventoryError(
            f"{path}.representations must exactly match registry order {available!r}"
        )

    target_policy = _nonempty_string(row["target_policy"], path=f"{path}.target_policy")
    if target_policy not in _TARGET_POLICIES:
        raise InventoryError(
            f"{path}.target_policy must be one of {sorted(_TARGET_POLICIES)}"
        )
    selection_protocol = _nonempty_string(
        row["selection_protocol"], path=f"{path}.selection_protocol"
    )
    if selection_protocol not in _SELECTION_PROTOCOLS:
        raise InventoryError(
            f"{path}.selection_protocol must be one of {sorted(_SELECTION_PROTOCOLS)}"
        )
    comparison_group = _nonempty_string(
        row["comparison_group"], path=f"{path}.comparison_group"
    )
    reference_dataset = _optional_string(
        row["reference_dataset"], path=f"{path}.reference_dataset"
    )
    raw_delta = row["expected_lid_delta"]
    if raw_delta is None:
        expected_lid_delta = None
    elif isinstance(raw_delta, bool) or not isinstance(raw_delta, (int, float)):
        raise InventoryError(f"{path}.expected_lid_delta must be numeric or null")
    else:
        expected_lid_delta = float(raw_delta)
        if not math.isfinite(expected_lid_delta):
            raise InventoryError(f"{path}.expected_lid_delta must be finite")

    has_lid_targets = "lid" in spec.required_artifacts
    if has_lid_targets:
        if target_policy != "known_lid":
            raise InventoryError(
                f"{path} has LID artifacts and requires known_lid policy"
            )
        if selection_protocol != "supervised_train_mae":
            raise InventoryError(f"{path} known-LID cell requires supervised_train_mae")
        if reference_dataset is not None or expected_lid_delta is not None:
            raise InventoryError(f"{path} known-LID cell forbids a delta reference")
    else:
        if target_policy == "known_lid":
            raise InventoryError(f"{path} has no LID artifact and cannot be known_lid")
        if selection_protocol != "target_free_train_stability":
            raise InventoryError(
                f"{path} target-free cell requires target_free_train_stability"
            )
        if reference_dataset is None or expected_lid_delta is None:
            raise InventoryError(
                f"{path} target-free cell requires reference and delta"
            )

    transformation = spec.transformation
    if transformation is not None:
        expected_policy = (
            "sample_size" if transformation.family == "sample_size" else "paired_delta"
        )
        if target_policy != expected_policy:
            raise InventoryError(
                f"{path}.target_policy disagrees with registry transformation"
            )
        if reference_dataset != transformation.reference:
            raise InventoryError(f"{path}.reference_dataset disagrees with registry")
        assert expected_lid_delta is not None
        if not math.isclose(
            expected_lid_delta,
            transformation.expected_lid_delta,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise InventoryError(f"{path}.expected_lid_delta disagrees with registry")

    return tuple(
        InventoryCell(
            suite_id=suite_id,
            dataset=name,
            representation=representation,
            has_lid_targets=has_lid_targets,
            target_policy=target_policy,
            selection_protocol=selection_protocol,
            comparison_group=comparison_group,
            reference_dataset=reference_dataset,
            expected_lid_delta=expected_lid_delta,
        )
        for representation in representations
    )


def _computed_counts(cells: tuple[InventoryCell, ...]) -> dict[str, int]:
    datasets = tuple(dict.fromkeys(cell.dataset for cell in cells))
    known_datasets = {cell.dataset for cell in cells if cell.has_lid_targets}
    target_free_datasets = set(datasets) - known_datasets
    return {
        "dataset_count": len(datasets),
        "cell_count": len(cells),
        "known_lid_dataset_count": len(known_datasets),
        "known_lid_cell_count": sum(cell.has_lid_targets for cell in cells),
        "target_free_dataset_count": len(target_free_datasets),
        "target_free_cell_count": sum(not cell.has_lid_targets for cell in cells),
    }


def _require_expected_counts(
    declared: Mapping[str, int], actual: Mapping[str, int], *, path: str
) -> None:
    mismatches = {
        key: {"declared": declared[key], "actual": actual[key]}
        for key in actual
        if declared[key] != actual[key]
    }
    if mismatches:
        raise InventoryError(f"{path} count mismatch: {mismatches}")


def _require_approved_file(
    project_root: Path, relative: str, expected_sha256: str, *, label: str
) -> None:
    path = project_root / relative
    if path.is_symlink() or not path.is_file():
        raise InventoryError(f"approved {label} is missing or not a regular file")
    if sha256_file(path) != expected_sha256:
        raise InventoryError(f"approved {label} content differs from its pinned SHA")


def _validate_approved_inventory_contract(
    *,
    config_path: Path,
    project_root: Path,
    inventory_id: str,
    source: InventorySource,
    suite_order: tuple[str, ...],
    global_counts: Mapping[str, int],
) -> None:
    contract = _APPROVED_INVENTORY_FILES[source.kind]
    expected_inventory = (project_root / str(contract["inventory_path"])).resolve()
    if config_path.resolve() != expected_inventory:
        raise InventoryError(
            "global inventory path differs from the approved source contract"
        )
    expected_source = {
        "dataset_config": contract["dataset_config"],
        "data_root": contract["data_root"],
        "registry": contract["registry"],
        "registry_overlay": contract["registry_overlay"],
    }
    actual_source = {
        "dataset_config": source.dataset_config,
        "data_root": source.data_root,
        "registry": source.registry,
        "registry_overlay": source.registry_overlay,
    }
    if inventory_id != contract["inventory_id"] or actual_source != expected_source:
        raise InventoryError("global inventory source identity differs from contract")
    if (
        suite_order != contract["suite_order"]
        or dict(global_counts) != contract["counts"]
    ):
        raise InventoryError("global inventory suite/count contract differs")
    for path_field, sha_field, label in (
        ("inventory_path", "inventory_sha256", "inventory YAML"),
        ("dataset_config", "dataset_config_sha256", "dataset config"),
        ("registry", "registry_sha256", "dataset registry"),
    ):
        _require_approved_file(
            project_root,
            str(contract[path_field]),
            str(contract[sha_field]),
            label=label,
        )
    overlay = contract["registry_overlay"]
    overlay_sha = contract["registry_overlay_sha256"]
    if overlay is not None and overlay_sha is not None:
        _require_approved_file(
            project_root,
            str(overlay),
            str(overlay_sha),
            label="registry overlay",
        )


def load_global_inventory(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> GlobalInventory:
    """Load one suite inventory and verify its registry/provenance/count anchors."""

    root = _repository_root() if project_root is None else Path(project_root).resolve()
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = root / config_path
    raw = _read_yaml(config_path, label="global inventory")
    _reject_unknown(raw, _TOP_LEVEL_FIELDS, path="global inventory")
    missing_top = _TOP_LEVEL_FIELDS - set(raw)
    if missing_top:
        raise InventoryError(f"missing global inventory fields: {sorted(missing_top)}")
    schema_version = raw["schema_version"]
    if isinstance(schema_version, bool) or schema_version != SCHEMA_VERSION:
        raise InventoryError(
            f"unsupported global inventory schema_version {schema_version!r}"
        )
    inventory_id = _nonempty_string(raw["inventory_id"], path="inventory_id")
    source = _parse_source(raw["source"])
    registry = _load_source_registry(source, project_root=root)

    suite_order_rows = _sequence(raw["suite_order"], path="suite_order")
    suite_order = tuple(
        _nonempty_string(value, path=f"suite_order[{index}]")
        for index, value in enumerate(suite_order_rows)
    )
    if not suite_order or len(suite_order) != len(set(suite_order)):
        raise InventoryError("suite_order must contain unique suite ids")
    if any(not _SUITE_ID.fullmatch(value) for value in suite_order):
        raise InventoryError("suite_order ids must lie in e1..e8")

    suite_rows = _sequence(raw["suites"], path="suites")
    suites: list[SuiteDefinition] = []
    seen_datasets: set[str] = set()
    seen_cells: set[tuple[str, str]] = set()
    for suite_index, raw_suite in enumerate(suite_rows):
        suite_path = f"suites[{suite_index}]"
        suite_row = _mapping(raw_suite, path=suite_path)
        _reject_unknown(suite_row, _SUITE_FIELDS, path=suite_path)
        missing_suite = _SUITE_FIELDS - set(suite_row)
        if missing_suite:
            raise InventoryError(
                f"missing fields in {suite_path}: {sorted(missing_suite)}"
            )
        suite_id = _nonempty_string(suite_row["id"], path=f"{suite_path}.id")
        availability = _nonempty_string(
            suite_row["availability"], path=f"{suite_path}.availability"
        )
        if availability not in _AVAILABILITY:
            raise InventoryError(
                f"{suite_path}.availability must be one of {sorted(_AVAILABILITY)}"
            )
        if source.exact_archive and availability == "generated_extension":
            raise InventoryError(
                "exact inventory cannot contain generated-extension suites"
            )
        if not source.exact_archive and availability == "canonical_exact":
            raise InventoryError(
                "generated inventory cannot claim canonical-exact suites"
            )

        dataset_rows = _sequence(suite_row["datasets"], path=f"{suite_path}.datasets")
        cells: list[InventoryCell] = []
        suite_datasets: set[str] = set()
        for dataset_index, raw_dataset in enumerate(dataset_rows):
            dataset_path = f"{suite_path}.datasets[{dataset_index}]"
            expanded = _parse_dataset_cells(
                raw_dataset,
                path=dataset_path,
                suite_id=suite_id,
                registry=registry,
            )
            dataset_name = expanded[0].dataset
            if dataset_name in suite_datasets or dataset_name in seen_datasets:
                raise InventoryError(f"duplicate inventory dataset {dataset_name!r}")
            suite_datasets.add(dataset_name)
            seen_datasets.add(dataset_name)
            for cell in expanded:
                if cell.key in seen_cells:
                    raise InventoryError(f"duplicate inventory cell {cell.key!r}")
                seen_cells.add(cell.key)
                cells.append(cell)
        if availability == "absent_from_source" and cells:
            raise InventoryError(f"{suite_path} absent suite must contain no datasets")
        if availability != "absent_from_source" and not cells:
            raise InventoryError(f"{suite_path} available suite must contain datasets")

        suite_counts = _parse_counts(
            suite_row["expected"], path=f"{suite_path}.expected", global_counts=False
        )
        actual_suite_counts = _computed_counts(tuple(cells))
        _require_expected_counts(
            suite_counts, actual_suite_counts, path=f"{suite_path}.expected"
        )
        suites.append(
            SuiteDefinition(
                suite_id=suite_id,
                availability=availability,
                cells=tuple(cells),
                expected_dataset_count=suite_counts["dataset_count"],
                expected_cell_count=suite_counts["cell_count"],
                expected_known_lid_dataset_count=suite_counts[
                    "known_lid_dataset_count"
                ],
                expected_known_lid_cell_count=suite_counts["known_lid_cell_count"],
                expected_target_free_dataset_count=suite_counts[
                    "target_free_dataset_count"
                ],
                expected_target_free_cell_count=suite_counts["target_free_cell_count"],
            )
        )

    if tuple(suite.suite_id for suite in suites) != suite_order:
        raise InventoryError("suites must exactly follow suite_order")
    global_counts = _parse_counts(raw["expected"], path="expected", global_counts=True)
    all_cells = tuple(cell for suite in suites for cell in suite.cells)
    actual_global_counts = {
        "suite_count": len(suites),
        "available_suite_count": sum(suite.available for suite in suites),
        **_computed_counts(all_cells),
    }
    _require_expected_counts(global_counts, actual_global_counts, path="expected")

    dataset_names = tuple(dict.fromkeys(cell.dataset for cell in all_cells))
    if set(dataset_names) != set(registry):
        raise InventoryError(
            "inventory datasets must exactly cover the declared source registry"
        )
    cells_by_key = {cell.key: cell for cell in all_cells}
    for cell in all_cells:
        if cell.reference_dataset is None:
            continue
        reference_key = (cell.reference_dataset, cell.representation)
        reference = cells_by_key.get(reference_key)
        if reference is None:
            raise InventoryError(
                f"cell {cell.key!r} has absent same-representation reference "
                f"{reference_key!r}"
            )
        if (
            reference.suite_id != cell.suite_id
            or reference.has_lid_targets
            or reference.selection_protocol != "target_free_train_stability"
            or reference.target_policy != cell.target_policy
            or reference.comparison_group != cell.comparison_group
        ):
            raise InventoryError(
                f"cell {cell.key!r} must reference a target-free root in the same "
                "suite, representation, target policy, and comparison group"
            )
        if (
            reference.reference_dataset != reference.dataset
            or reference.expected_lid_delta is None
            or not math.isclose(
                reference.expected_lid_delta,
                0.0,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise InventoryError(
                f"cell {cell.key!r} target-free reference is not a zero-delta "
                "self-referencing root"
            )

    _validate_dataset_config(source, project_root=root, dataset_names=dataset_names)
    _validate_approved_inventory_contract(
        config_path=config_path,
        project_root=root,
        inventory_id=inventory_id,
        source=source,
        suite_order=suite_order,
        global_counts=global_counts,
    )
    return GlobalInventory(
        inventory_id=inventory_id,
        source=source,
        suites=tuple(suites),
        expected_suite_count=global_counts["suite_count"],
        expected_available_suite_count=global_counts["available_suite_count"],
        expected_dataset_count=global_counts["dataset_count"],
        expected_cell_count=global_counts["cell_count"],
        expected_known_lid_dataset_count=global_counts["known_lid_dataset_count"],
        expected_known_lid_cell_count=global_counts["known_lid_cell_count"],
        expected_target_free_dataset_count=global_counts["target_free_dataset_count"],
        expected_target_free_cell_count=global_counts["target_free_cell_count"],
    )


def load_canonical_inventory(
    project_root: str | Path | None = None,
) -> GlobalInventory:
    """Return the canonical exact-archive inventory (E3/E4 explicitly absent)."""

    return load_global_inventory(CANONICAL_INVENTORY_CONFIG, project_root=project_root)


def load_generated_e3_e4_inventory(
    project_root: str | Path | None = None,
) -> GlobalInventory:
    """Return the separate generated-only E3/E4 extension inventory."""

    return load_global_inventory(
        GENERATED_E3_E4_INVENTORY_CONFIG, project_root=project_root
    )


__all__ = [
    "CANONICAL_INVENTORY_CONFIG",
    "GENERATED_E3_E4_INVENTORY_CONFIG",
    "GlobalInventory",
    "InventoryCell",
    "InventoryError",
    "InventorySource",
    "SuiteDefinition",
    "load_canonical_inventory",
    "load_generated_e3_e4_inventory",
    "load_global_inventory",
]
