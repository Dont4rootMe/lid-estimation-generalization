"""Deterministic, fail-closed preparation of the generated E3/E4 extension.

The E3 and E4 generator classes exist in the pinned LID-Benchmarks source but
are not called by upstream ``prepare_all`` and are absent from the canonical
paper archive.  This module therefore gives them a separate provenance
identity and output root.  It never fits a PCA or downloads a base dataset: a
pre-existing PCA artifact is a required input and is copied into the sealed
output before the vendored classes are invoked in their frozen subprocess.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from utils.provenance import (
    EXPECTED_LID_BENCHMARKS_SHA,
    sha256_file,
    verify_upstream_source,
)

MANIFEST_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "generated_e3_e4_manifest.json"
STORED_PCA_PATH = Path("inputs/pca.joblib")
DEFAULT_RUNTIME_COMMAND = ("uv", "run", "--frozen", "python")
EXPECTED_CANONICAL_PCA_SIZE_BYTES = 98_413
EXPECTED_CANONICAL_PCA_SHA256 = (
    "532c840a040f6398911248df26f26c84ed09976f5c346446e0b964e3a582a97b"
)
EXPECTED_CANONICAL_PCA_ARCHIVE_PATH = "benchmarks/pca.joblib"

_KIND = "lid-benchmarks-generated-e3-e4-extension"
_PROVENANCE_LABEL = "generated-extension-not-canonical-exact"
_HASH_CHUNK_SIZE = 8 * 1024 * 1024
_MAX_MANIFEST_SIZE_BYTES = 1024 * 1024
_UPSTREAM_CONTRACT_FILES = (
    "UPSTREAM.yaml",
    "pyproject.toml",
    "uv.lock",
    "generators/__init__.py",
    "generators/benchmarks.py",
    "generators/utils/arrows.py",
    "generators/utils/padded_and_downscaled.py",
    "generators/utils/pca.py",
)
_RUNTIME_SHADOW_NAMES = (
    "joblib.py",
    "numpy.py",
    "requests.py",
    "sklearn.py",
    "torch.py",
    "torchvision.py",
    "tqdm.py",
)


@dataclass(frozen=True, slots=True)
class ExtensionGenerationSpec:
    """Complete data-generation contract (the CLI always uses the standard one)."""

    train_samples: int = 100_000
    val_samples: int = 1_000
    test_samples: int = 1_000
    pca_components: int = 30
    flattened_image_dim: int = 28 * 28
    intrinsic_dim: int = 20
    sphere_radius: int = 1
    seed: int = 0

    def __post_init__(self) -> None:
        if min(self.train_samples, self.val_samples, self.test_samples) <= 0:
            raise ValueError("all generated split sizes must be positive")
        if self.pca_components != 30:
            raise ValueError("the generated E3/E4 extension requires 30 PCA components")
        if self.flattened_image_dim != 28 * 28:
            raise ValueError("the generated E3/E4 extension requires 28x28 images")
        if self.intrinsic_dim != 20:
            raise ValueError("the generated E3/E4 extension requires dim=20")
        if self.sphere_radius != 1:
            raise ValueError("the generated E4 extension requires radius=1")
        if self.seed != 0:
            raise ValueError("the generated E3/E4 extension requires seed=0")

    @property
    def split_sizes(self) -> Mapping[str, int]:
        return MappingProxyType(
            {
                "train": self.train_samples,
                "val": self.val_samples,
                "test": self.test_samples,
            }
        )


STANDARD_EXTENSION_SPEC = ExtensionGenerationSpec()


@dataclass(frozen=True, slots=True)
class _DatasetContract:
    name: str
    generator_class: str
    expected_lid: int
    radius: int | None = None


_DATASETS = (
    _DatasetContract(
        name="e3_gaussian_pca",
        generator_class="GaussianPCADatasetGenerator",
        expected_lid=20,
    ),
    _DatasetContract(
        name="e4_sphere_pca_radius1",
        generator_class="SpherePCADatasetGenerator",
        expected_lid=19,
        radius=1,
    ),
)
_SPLITS = ("train", "val", "test")
_ARTIFACTS = ("dataset", "lid", "coefficients")


_GENERATION_BOOTSTRAP = r"""
import json
from pathlib import Path
import platform
import sys

import joblib
import numpy as np

from generators.benchmarks import (
    GaussianPCADatasetGenerator,
    SpherePCADatasetGenerator,
)

output = Path(sys.argv[1])
pca_path = Path(sys.argv[2])
n_train, n_val, n_test = (int(value) for value in sys.argv[3:6])
pca = joblib.load(pca_path)

components = np.asarray(getattr(pca, "components_", None))
mean = np.asarray(getattr(pca, "mean_", None))
n_components = getattr(pca, "n_components", None)
if n_components != 30:
    raise RuntimeError(f"PCA n_components must equal 30, found {n_components!r}")
if components.shape != (30, 784):
    raise RuntimeError(
        f"PCA components_ must have shape (30, 784), found {components.shape}"
    )
if mean.shape != (784,):
    raise RuntimeError(f"PCA mean_ must have shape (784,), found {mean.shape}")
if not np.issubdtype(components.dtype, np.number) or not np.isfinite(components).all():
    raise RuntimeError("PCA components_ must be finite and numeric")
if not np.issubdtype(mean.dtype, np.number) or not np.isfinite(mean).all():
    raise RuntimeError("PCA mean_ must be finite and numeric")

GaussianPCADatasetGenerator(
    dataset_root_dir=str(output),
    N_train=n_train,
    N_val=n_val,
    N_test=n_test,
    dim=20,
    seed=0,
    pca=pca,
).generate()
SpherePCADatasetGenerator(
    dataset_root_dir=str(output),
    N_train=n_train,
    N_val=n_val,
    N_test=n_test,
    dim=20,
    radius=1,
    seed=0,
    pca=pca,
).generate()

receipt = {
    "python_version": platform.python_version(),
    "numpy_version": str(getattr(np, "__version__", "unknown")),
    "joblib_version": str(getattr(joblib, "__version__", "unknown")),
    "pca_class": f"{type(pca).__module__}.{type(pca).__qualname__}",
    "pca_components_shape": list(components.shape),
    "pca_mean_shape": list(mean.shape),
}
(output / ".generation_receipt.json").write_text(
    json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
    encoding="utf-8",
)
""".strip()


class GeneratedExtensionError(RuntimeError):
    """Base error for generated E3/E4 preparation."""


class GeneratedExtensionValidationError(GeneratedExtensionError):
    """Raised when an existing or newly generated extension is invalid."""


class GeneratedExtensionConflictError(GeneratedExtensionError):
    """Raised rather than overwriting an existing incompatible output root."""


@dataclass(frozen=True, slots=True)
class GeneratedExtensionResult:
    """Result of a successful creation or idempotent validation."""

    output_root: Path
    manifest_path: Path
    upstream_revision: str
    pca_sha256: str
    created: bool


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _generator_config(spec: ExtensionGenerationSpec) -> dict[str, Any]:
    return {
        "seed": spec.seed,
        "split_sizes": dict(spec.split_sizes),
        "pca_contract": {
            "n_components": spec.pca_components,
            "components_shape": [
                spec.pca_components,
                spec.flattened_image_dim,
            ],
            "mean_shape": [spec.flattened_image_dim],
        },
        "generators": [
            {
                "class": contract.generator_class,
                "dataset_name": contract.name,
                "dim": spec.intrinsic_dim,
                "radius": contract.radius,
                "expected_lid": contract.expected_lid,
            }
            for contract in _DATASETS
        ],
        "bootstrap_sha256": hashlib.sha256(
            _GENERATION_BOOTSTRAP.encode("utf-8")
        ).hexdigest(),
    }


def _source_records(checkout: Path) -> dict[str, dict[str, int | str]]:
    records: dict[str, dict[str, int | str]] = {}
    for relative in _UPSTREAM_CONTRACT_FILES:
        path = checkout / relative
        if not path.is_file() or path.is_symlink():
            raise GeneratedExtensionValidationError(
                f"pinned upstream contract file is missing or not regular: {path}"
            )
        records[relative] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return records


def _reject_runtime_shadows(checkout: Path) -> None:
    shadows = [name for name in _RUNTIME_SHADOW_NAMES if (checkout / name).exists()]
    if shadows:
        raise GeneratedExtensionValidationError(
            f"pinned upstream checkout contains dependency-shadowing paths: {shadows}"
        )


def _stable_regular_file_hash(path: Path) -> tuple[int, str]:
    try:
        before_lstat = path.lstat()
    except OSError as exc:
        raise GeneratedExtensionValidationError(
            f"cannot inspect required PCA artifact {path}: {exc}"
        ) from exc
    if stat.S_ISLNK(before_lstat.st_mode) or not stat.S_ISREG(before_lstat.st_mode):
        raise GeneratedExtensionValidationError(
            f"required PCA artifact must be a non-symlink regular file: {path}"
        )

    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            while chunk := stream.read(_HASH_CHUNK_SIZE):
                digest.update(chunk)
            after = os.fstat(stream.fileno())
    except OSError as exc:
        raise GeneratedExtensionValidationError(
            f"cannot hash required PCA artifact {path}: {exc}"
        ) from exc
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after:
        raise GeneratedExtensionValidationError(
            f"required PCA artifact changed while it was being hashed: {path}"
        )
    return before.st_size, digest.hexdigest()


def _require_canonical_pca_identity(size_bytes: int, sha256: str) -> None:
    if (
        size_bytes != EXPECTED_CANONICAL_PCA_SIZE_BYTES
        or sha256 != EXPECTED_CANONICAL_PCA_SHA256
    ):
        raise GeneratedExtensionValidationError(
            "PCA artifact does not match the canonical exact-archive "
            f"{EXPECTED_CANONICAL_PCA_ARCHIVE_PATH}: expected "
            f"size={EXPECTED_CANONICAL_PCA_SIZE_BYTES}, "
            f"sha256={EXPECTED_CANONICAL_PCA_SHA256}; found "
            f"size={size_bytes}, sha256={sha256}"
        )


def _copy_pca(source: Path, destination: Path) -> tuple[int, str]:
    source_size, source_hash = _stable_regular_file_hash(source)
    destination.parent.mkdir(mode=0o755)
    digest = hashlib.sha256()
    try:
        with source.open("rb") as reader, destination.open("xb") as writer:
            before = os.fstat(reader.fileno())
            while chunk := reader.read(_HASH_CHUNK_SIZE):
                writer.write(chunk)
                digest.update(chunk)
            writer.flush()
            os.fsync(writer.fileno())
            after = os.fstat(reader.fileno())
    except OSError as exc:
        raise GeneratedExtensionValidationError(
            f"cannot stage required PCA artifact {source}: {exc}"
        ) from exc
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise GeneratedExtensionValidationError(
            f"required PCA artifact changed while it was being copied: {source}"
        )
    copied_hash = digest.hexdigest()
    if source_size != destination.stat().st_size or source_hash != copied_hash:
        raise GeneratedExtensionValidationError(
            "staged PCA artifact does not match the explicitly supplied input"
        )
    return source_size, source_hash


def _all_finite(array: np.ndarray, *, target_elements: int = 1_000_000) -> bool:
    if np.issubdtype(array.dtype, np.integer):
        return True
    if array.ndim == 1:
        step = target_elements
    else:
        elements_per_row = max(1, int(np.prod(array.shape[1:], dtype=np.int64)))
        step = max(1, target_elements // elements_per_row)
    return all(
        bool(np.isfinite(array[start : start + step]).all())
        for start in range(0, array.shape[0], step)
    )


def _expected_paths(spec: ExtensionGenerationSpec) -> tuple[set[str], set[str]]:
    files = {STORED_PCA_PATH.as_posix()}
    directories = {"inputs"}
    for dataset in _DATASETS:
        directories.add(dataset.name)
        for split in _SPLITS:
            prefix = f"{dataset.name}/{split}"
            directories.add(prefix)
            files.update(f"{prefix}/{artifact}.npy" for artifact in _ARTIFACTS)
    return files, directories


def _reject_unexpected_tree_entries(
    root: Path,
    spec: ExtensionGenerationSpec,
    *,
    manifest_expected: bool,
) -> None:
    expected_files, expected_directories = _expected_paths(spec)
    if manifest_expected:
        expected_files.add(MANIFEST_FILENAME)
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise GeneratedExtensionValidationError(
                f"symbolic links are forbidden in generated output: {relative}"
            )
        if path.is_dir():
            observed_directories.add(relative)
        elif path.is_file():
            observed_files.add(relative)
        else:
            raise GeneratedExtensionValidationError(
                f"special filesystem entry in generated output: {relative}"
            )
    if observed_files != expected_files:
        raise GeneratedExtensionValidationError(
            "generated output file set mismatch: "
            f"missing={sorted(expected_files - observed_files)}, "
            f"unexpected={sorted(observed_files - expected_files)}"
        )
    if observed_directories != expected_directories:
        raise GeneratedExtensionValidationError(
            "generated output directory set mismatch: "
            f"missing={sorted(expected_directories - observed_directories)}, "
            f"unexpected={sorted(observed_directories - expected_directories)}"
        )


def _artifact_shape(
    contract: _DatasetContract,
    artifact: str,
    sample_count: int,
    spec: ExtensionGenerationSpec,
) -> tuple[int, ...]:
    if artifact == "dataset":
        return (sample_count, 1, 28, 28)
    if artifact == "lid":
        return (sample_count,)
    return (sample_count, spec.pca_components)


def _inspect_artifacts(
    root: Path,
    spec: ExtensionGenerationSpec,
    *,
    manifest_expected: bool,
) -> dict[str, Any]:
    _reject_unexpected_tree_entries(
        root,
        spec,
        manifest_expected=manifest_expected,
    )
    datasets: dict[str, Any] = {}
    for contract in _DATASETS:
        splits: dict[str, Any] = {}
        for split in _SPLITS:
            sample_count = spec.split_sizes[split]
            artifacts: dict[str, Any] = {}
            for artifact in _ARTIFACTS:
                relative = f"{contract.name}/{split}/{artifact}.npy"
                path = root / relative
                try:
                    array = np.load(path, allow_pickle=False, mmap_mode="r")
                except (OSError, ValueError) as exc:
                    raise GeneratedExtensionValidationError(
                        f"cannot load generated artifact {relative}: {exc}"
                    ) from exc
                expected_shape = _artifact_shape(
                    contract,
                    artifact,
                    sample_count,
                    spec,
                )
                if tuple(array.shape) != expected_shape:
                    raise GeneratedExtensionValidationError(
                        f"generated artifact {relative} has shape {array.shape}; "
                        f"expected {expected_shape}"
                    )
                if not (
                    np.issubdtype(array.dtype, np.integer)
                    or np.issubdtype(array.dtype, np.floating)
                ):
                    raise GeneratedExtensionValidationError(
                        f"generated artifact {relative} is not real numeric: "
                        f"{array.dtype}"
                    )
                if not _all_finite(array):
                    raise GeneratedExtensionValidationError(
                        f"generated artifact {relative} contains NaN or infinity"
                    )
                record: dict[str, Any] = {
                    "path": relative,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                }
                if artifact == "lid":
                    unique = np.unique(np.asarray(array)).tolist()
                    normalized = [float(value) for value in unique]
                    if normalized != [float(contract.expected_lid)]:
                        raise GeneratedExtensionValidationError(
                            f"generated artifact {relative} has LID values "
                            f"{normalized}; expected {[float(contract.expected_lid)]}"
                        )
                    record["unique_values"] = normalized
                artifacts[artifact] = record
            splits[split] = {
                "sample_count": sample_count,
                "artifacts": artifacts,
            }
        datasets[contract.name] = {
            "generator_class": contract.generator_class,
            "expected_lid": contract.expected_lid,
            "splits": splits,
        }
    return datasets


def _read_json_object(path: Path, *, maximum_size: int) -> dict[str, Any]:
    try:
        if path.stat().st_size > maximum_size:
            raise GeneratedExtensionValidationError(
                f"JSON file exceeds {maximum_size} bytes: {path}"
            )
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise GeneratedExtensionValidationError(
            f"cannot read JSON file {path}: {exc}"
        ) from exc

    def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise GeneratedExtensionValidationError(
                    f"JSON file contains duplicate key {key!r}: {path}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(text, object_pairs_hook=reject_duplicate_pairs)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise GeneratedExtensionValidationError(
            f"cannot parse JSON file {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise GeneratedExtensionValidationError(
            f"JSON file must contain an object: {path}"
        )
    return value


def _read_receipt(root: Path, spec: ExtensionGenerationSpec) -> dict[str, Any]:
    path = root / ".generation_receipt.json"
    receipt = _read_json_object(path, maximum_size=64 * 1024)
    expected_shapes = {
        "pca_components_shape": [
            spec.pca_components,
            spec.flattened_image_dim,
        ],
        "pca_mean_shape": [spec.flattened_image_dim],
    }
    for key, expected in expected_shapes.items():
        if receipt.get(key) != expected:
            raise GeneratedExtensionValidationError(
                f"generation receipt {key} mismatch: expected {expected}, "
                f"found {receipt.get(key)!r}"
            )
    for key in ("python_version", "numpy_version", "joblib_version", "pca_class"):
        if not isinstance(receipt.get(key), str) or not receipt[key]:
            raise GeneratedExtensionValidationError(
                f"generation receipt field {key!r} must be a non-empty string"
            )
    unknown = set(receipt) - {
        "python_version",
        "numpy_version",
        "joblib_version",
        "pca_class",
        "pca_components_shape",
        "pca_mean_shape",
    }
    if unknown:
        raise GeneratedExtensionValidationError(
            f"generation receipt has unexpected fields: {sorted(unknown)}"
        )
    try:
        path.unlink()
    except OSError as exc:
        raise GeneratedExtensionValidationError(
            f"cannot remove staging-only generation receipt {path}: {exc}"
        ) from exc
    return receipt


def _content_records(
    pca_record: Mapping[str, Any], datasets: Mapping[str, Any]
) -> dict[str, Any]:
    records = {
        str(pca_record["stored_path"]): {
            "size_bytes": pca_record["size_bytes"],
            "sha256": pca_record["sha256"],
        }
    }
    for dataset in datasets.values():
        for split in dataset["splits"].values():
            for artifact in split["artifacts"].values():
                records[artifact["path"]] = dict(artifact)
    return records


def _build_manifest(
    *,
    revision: str,
    source_records: Mapping[str, Any],
    spec: ExtensionGenerationSpec,
    pca_size: int,
    pca_sha256: str,
    runtime_command: Sequence[str],
    runtime_receipt: Mapping[str, Any],
    datasets: Mapping[str, Any],
) -> dict[str, Any]:
    pca_record = {
        "stored_path": STORED_PCA_PATH.as_posix(),
        "size_bytes": pca_size,
        "sha256": pca_sha256,
        "source_contract": {
            "kind": "canonical-exact-archive-pca",
            "archive_relative_path": EXPECTED_CANONICAL_PCA_ARCHIVE_PATH,
            "expected_size_bytes": EXPECTED_CANONICAL_PCA_SIZE_BYTES,
            "expected_sha256": EXPECTED_CANONICAL_PCA_SHA256,
        },
    }
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": _KIND,
        "provenance_label": _PROVENANCE_LABEL,
        "canonical_exact_archive": False,
        "upstream": {
            "revision": revision,
            "source_files": dict(source_records),
        },
        "generator_config": _generator_config(spec),
        "pca": pca_record,
        "runtime": {
            "command": list(runtime_command),
            "environment": {
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONHASHSEED": "0",
                "PYTHONNOUSERSITE": "1",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
            },
            "receipt": dict(runtime_receipt),
        },
        "datasets": dict(datasets),
    }
    manifest["content_tree_sha256"] = _sha256_json(
        _content_records(pca_record, datasets)
    )
    manifest["seal_sha256"] = _sha256_json(manifest)
    return manifest


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _validate_manifest_header(
    manifest: Mapping[str, Any],
    *,
    revision: str,
    source_records: Mapping[str, Any],
    spec: ExtensionGenerationSpec,
) -> None:
    expected_keys = {
        "schema_version",
        "kind",
        "provenance_label",
        "canonical_exact_archive",
        "upstream",
        "generator_config",
        "pca",
        "runtime",
        "datasets",
        "content_tree_sha256",
        "seal_sha256",
    }
    if set(manifest) != expected_keys:
        raise GeneratedExtensionValidationError(
            "generated manifest fields mismatch: "
            f"missing={sorted(expected_keys - set(manifest))}, "
            f"unexpected={sorted(set(manifest) - expected_keys)}"
        )
    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise GeneratedExtensionValidationError(
            f"unsupported generated manifest schema {manifest['schema_version']!r}"
        )
    if manifest["kind"] != _KIND or manifest["provenance_label"] != _PROVENANCE_LABEL:
        raise GeneratedExtensionValidationError(
            "generated manifest provenance mismatch"
        )
    if manifest["canonical_exact_archive"] is not False:
        raise GeneratedExtensionValidationError(
            "generated E3/E4 data must never claim canonical exact-archive identity"
        )
    if manifest["upstream"] != {
        "revision": revision,
        "source_files": dict(source_records),
    }:
        raise GeneratedExtensionValidationError(
            "generated manifest does not match the current pinned upstream source"
        )
    if manifest["generator_config"] != _generator_config(spec):
        raise GeneratedExtensionValidationError(
            "generated manifest generator configuration mismatch"
        )
    seal = manifest.get("seal_sha256")
    if not _is_sha256(seal):
        raise GeneratedExtensionValidationError(
            "generated manifest has invalid seal_sha256"
        )
    without_seal = dict(manifest)
    without_seal.pop("seal_sha256")
    if seal != _sha256_json(without_seal):
        raise GeneratedExtensionValidationError(
            "generated manifest seal_sha256 mismatch"
        )


def _validate_runtime_manifest(runtime: object) -> None:
    if not isinstance(runtime, Mapping) or set(runtime) != {
        "command",
        "environment",
        "receipt",
    }:
        raise GeneratedExtensionValidationError("generated manifest runtime is invalid")
    command = runtime["command"]
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(token, str) and token for token in command)
    ):
        raise GeneratedExtensionValidationError(
            "generated manifest runtime command is invalid"
        )
    expected_environment = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }
    if runtime["environment"] != expected_environment:
        raise GeneratedExtensionValidationError(
            "generated manifest deterministic runtime environment mismatch"
        )
    receipt = runtime["receipt"]
    if not isinstance(receipt, Mapping):
        raise GeneratedExtensionValidationError("generated manifest receipt is invalid")
    expected_receipt_keys = {
        "python_version",
        "numpy_version",
        "joblib_version",
        "pca_class",
        "pca_components_shape",
        "pca_mean_shape",
    }
    if set(receipt) != expected_receipt_keys:
        raise GeneratedExtensionValidationError(
            "generated manifest receipt fields mismatch"
        )
    if receipt["pca_components_shape"] != [30, 784] or receipt["pca_mean_shape"] != [
        784
    ]:
        raise GeneratedExtensionValidationError(
            "generated manifest receipt PCA shape mismatch"
        )
    if any(
        not isinstance(receipt[key], str) or not receipt[key]
        for key in ("python_version", "numpy_version", "joblib_version", "pca_class")
    ):
        raise GeneratedExtensionValidationError(
            "generated manifest receipt contains an invalid version/class field"
        )


def validate_generated_e3_e4(
    output_root: str | Path,
    pca_path: str | Path,
    *,
    checkout: str | Path | None = None,
    _spec: ExtensionGenerationSpec | None = None,
) -> GeneratedExtensionResult:
    """Validate the exact output tree and its binding to source and PCA input."""

    spec = STANDARD_EXTENSION_SPEC if _spec is None else _spec
    root = Path(output_root).expanduser().resolve()
    source = (
        _repository_root() / "lid_benchmarks"
        if checkout is None
        else Path(checkout).expanduser().resolve()
    )
    pca = Path(pca_path).expanduser().absolute()
    if not root.is_dir() or root.is_symlink():
        raise GeneratedExtensionValidationError(
            f"generated E3/E4 output is not a non-symlink directory: {root}"
        )
    revision = verify_upstream_source(source)
    if revision != EXPECTED_LID_BENCHMARKS_SHA:
        raise GeneratedExtensionValidationError(
            f"unexpected upstream revision {revision!r}"
        )
    _reject_runtime_shadows(source)
    source_records = _source_records(source)
    supplied_pca_size, supplied_pca_sha = _stable_regular_file_hash(pca)
    _require_canonical_pca_identity(supplied_pca_size, supplied_pca_sha)
    manifest = _read_json_object(
        root / MANIFEST_FILENAME,
        maximum_size=_MAX_MANIFEST_SIZE_BYTES,
    )
    _validate_manifest_header(
        manifest,
        revision=revision,
        source_records=source_records,
        spec=spec,
    )
    _validate_runtime_manifest(manifest["runtime"])

    expected_pca = {
        "stored_path": STORED_PCA_PATH.as_posix(),
        "size_bytes": supplied_pca_size,
        "sha256": supplied_pca_sha,
        "source_contract": {
            "kind": "canonical-exact-archive-pca",
            "archive_relative_path": EXPECTED_CANONICAL_PCA_ARCHIVE_PATH,
            "expected_size_bytes": EXPECTED_CANONICAL_PCA_SIZE_BYTES,
            "expected_sha256": EXPECTED_CANONICAL_PCA_SHA256,
        },
    }
    if manifest["pca"] != expected_pca:
        raise GeneratedExtensionValidationError(
            "generated output PCA identity does not match the supplied PCA artifact"
        )
    stored_pca = root / STORED_PCA_PATH
    stored_size, stored_sha = _stable_regular_file_hash(stored_pca)
    if (stored_size, stored_sha) != (supplied_pca_size, supplied_pca_sha):
        raise GeneratedExtensionValidationError(
            "stored PCA copy does not match the supplied PCA artifact"
        )

    actual_datasets = _inspect_artifacts(root, spec, manifest_expected=True)
    if manifest["datasets"] != actual_datasets:
        raise GeneratedExtensionValidationError(
            "generated artifact records do not match the manifest"
        )
    expected_content_hash = _sha256_json(
        _content_records(expected_pca, actual_datasets)
    )
    if manifest["content_tree_sha256"] != expected_content_hash:
        raise GeneratedExtensionValidationError(
            "generated manifest content_tree_sha256 mismatch"
        )
    return GeneratedExtensionResult(
        output_root=root,
        manifest_path=root / MANIFEST_FILENAME,
        upstream_revision=revision,
        pca_sha256=supplied_pca_sha,
        created=False,
    )


@contextmanager
def _output_lock(output_root: Path) -> Iterator[None]:
    lock_path = output_root.with_name(f".{output_root.name}.generation.lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _run_generation(
    checkout: Path,
    staging: Path,
    staged_pca: Path,
    spec: ExtensionGenerationSpec,
    runtime_command: Sequence[str],
) -> None:
    if not runtime_command or any(
        not isinstance(token, str) or not token for token in runtime_command
    ):
        raise ValueError("runtime_command must contain non-empty string tokens")
    command = tuple(runtime_command) + (
        "-c",
        _GENERATION_BOOTSTRAP,
        str(staging),
        str(staged_pca),
        str(spec.train_samples),
        str(spec.val_samples),
        str(spec.test_samples),
    )
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
    )
    try:
        subprocess.run(command, cwd=checkout, env=environment, check=True)
    except FileNotFoundError as exc:
        raise GeneratedExtensionError(
            f"generated E3/E4 runtime is unavailable: {runtime_command[0]}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise GeneratedExtensionError(
            f"pinned E3/E4 generator failed with exit code {exc.returncode}"
        ) from exc


def _ensure_separate_paths(output_root: Path, checkout: Path, pca: Path) -> None:
    try:
        output_root.relative_to(checkout)
    except ValueError:
        pass
    else:
        raise GeneratedExtensionError(
            "generated output root must be separate from the pinned source checkout"
        )
    try:
        pca.relative_to(output_root)
    except ValueError:
        pass
    else:
        raise GeneratedExtensionError(
            "the explicitly supplied PCA artifact must be outside the generated root"
        )
    if output_root == pca.parent:
        raise GeneratedExtensionError(
            "generated output root must be separate from the PCA source directory"
        )


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def prepare_generated_e3_e4(
    output_root: str | Path,
    pca_path: str | Path,
    *,
    checkout: str | Path | None = None,
    runtime_command: Sequence[str] = DEFAULT_RUNTIME_COMMAND,
    _spec: ExtensionGenerationSpec | None = None,
) -> GeneratedExtensionResult:
    """Create the standard extension atomically, or validate an identical root.

    ``_spec`` exists only for small repository tests.  The public CLI never
    exposes it and always uses :data:`STANDARD_EXTENSION_SPEC`.
    """

    spec = STANDARD_EXTENSION_SPEC if _spec is None else _spec
    source = (
        _repository_root() / "lid_benchmarks"
        if checkout is None
        else Path(checkout).expanduser().resolve()
    )
    pca = Path(pca_path).expanduser().absolute()
    output = Path(output_root).expanduser()
    output_parent = output.parent.resolve()
    output = output_parent / output.name
    if not output.name or output.name in {".", ".."}:
        raise GeneratedExtensionError("generated output root must name a directory")
    _ensure_separate_paths(output, source, pca)
    pca_size, pca_sha = _stable_regular_file_hash(pca)
    _require_canonical_pca_identity(pca_size, pca_sha)
    output_parent.mkdir(parents=True, exist_ok=True)

    with _output_lock(output):
        if output.is_symlink() or (output.exists() and not output.is_dir()):
            raise GeneratedExtensionConflictError(
                f"generated output path already exists and is not a directory: {output}"
            )
        if output.exists():
            try:
                return validate_generated_e3_e4(
                    output,
                    pca,
                    checkout=source,
                    _spec=spec,
                )
            except GeneratedExtensionValidationError as exc:
                raise GeneratedExtensionConflictError(
                    f"refusing to overwrite conflicting generated output {output}: {exc}"
                ) from exc

        revision = verify_upstream_source(source)
        if revision != EXPECTED_LID_BENCHMARKS_SHA:
            raise GeneratedExtensionValidationError(
                f"unexpected upstream revision {revision!r}"
            )
        _reject_runtime_shadows(source)
        source_records = _source_records(source)
        staging = Path(
            tempfile.mkdtemp(
                dir=output_parent,
                prefix=f".{output.name}.staging-",
            )
        )
        published = False
        try:
            copied_size, copied_sha = _copy_pca(pca, staging / STORED_PCA_PATH)
            if (copied_size, copied_sha) != (pca_size, pca_sha):
                raise GeneratedExtensionValidationError(
                    "PCA source changed between preflight and staging"
                )
            _run_generation(
                source, staging, staging / STORED_PCA_PATH, spec, runtime_command
            )
            runtime_receipt = _read_receipt(staging, spec)

            revision_after = verify_upstream_source(source)
            _reject_runtime_shadows(source)
            source_records_after = _source_records(source)
            if revision_after != revision or source_records_after != source_records:
                raise GeneratedExtensionValidationError(
                    "pinned upstream source changed during E3/E4 generation"
                )
            stored_size, stored_sha = _stable_regular_file_hash(
                staging / STORED_PCA_PATH
            )
            if (stored_size, stored_sha) != (pca_size, pca_sha):
                raise GeneratedExtensionValidationError(
                    "staged PCA artifact changed during E3/E4 generation"
                )
            datasets = _inspect_artifacts(staging, spec, manifest_expected=False)
            manifest = _build_manifest(
                revision=revision,
                source_records=source_records,
                spec=spec,
                pca_size=pca_size,
                pca_sha256=pca_sha,
                runtime_command=runtime_command,
                runtime_receipt=runtime_receipt,
                datasets=datasets,
            )
            _write_manifest(staging / MANIFEST_FILENAME, manifest)
            _reject_unexpected_tree_entries(staging, spec, manifest_expected=True)
            if output.exists() or output.is_symlink():
                raise GeneratedExtensionConflictError(
                    f"generated output appeared during preparation: {output}"
                )
            os.rename(staging, output)
            published = True
            _fsync_directory(output_parent)
        finally:
            if not published and staging.exists():
                shutil.rmtree(staging)

    return GeneratedExtensionResult(
        output_root=output,
        manifest_path=output / MANIFEST_FILENAME,
        upstream_revision=revision,
        pca_sha256=pca_sha,
        created=True,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """CLI for standard E3/E4 generation or strict validation."""

    parser = argparse.ArgumentParser(
        description=(
            "Prepare the non-canonical generated E3/E4 extension from an "
            "explicit existing PCA artifact. Production geometry, seed, and "
            "split sizes are fixed and cannot be overridden."
        )
    )
    parser.add_argument(
        "--pca",
        type=Path,
        required=True,
        help="existing canonical pca.joblib; this command never fits or downloads PCA",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="separate generated output root (must not be the exact-archive root)",
    )
    parser.add_argument(
        "--checkout",
        type=Path,
        default=_repository_root() / "lid_benchmarks",
        help="pinned vendored LID-Benchmarks checkout",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate an existing output without running either generator",
    )
    args = parser.parse_args(argv)
    if args.validate_only:
        result = validate_generated_e3_e4(
            args.output,
            args.pca,
            checkout=args.checkout,
        )
        action = "validated"
    else:
        result = prepare_generated_e3_e4(
            args.output,
            args.pca,
            checkout=args.checkout,
        )
        action = "created" if result.created else "validated existing"
    print(
        f"{action} generated E3/E4 extension at {result.output_root}; "
        f"upstream={result.upstream_revision}; pca_sha256={result.pca_sha256}"
    )


if __name__ == "__main__":
    main()
