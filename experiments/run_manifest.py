"""Immutable run metadata and integrity checks.

The repository currently has no initial Git commit, so a Git SHA alone cannot
identify the code.  We therefore hash the executable source/config tree in
addition to recording Git state.  Once the project is committed, both fields
remain useful: the tree hash catches dirty-run changes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import socket
import subprocess
import sys
from typing import Any, Iterable, Mapping

from datasets.archive import EXACT_ARCHIVE_SHA256
from utils.provenance import EXPECTED_LID_BENCHMARKS_SHA


MANIFEST_SCHEMA_VERSION = 2
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "created_at_utc",
        "config_sha256",
        "source_tree_sha256",
        "dataset_record_sha256",
        "environment_sha256",
        "upstream_sha",
        "git",
        "environment",
        "resolved_config",
        "dataset",
        "outputs",
    }
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_declared_sources(root: Path, patterns: Iterable[str] | None = None) -> str:
    selected_patterns = tuple(
        patterns
        or (
            "pyproject.toml",
            "uv.lock",
            "models/**/*.py",
            "experiments/**/*.py",
            "datasets/**/*.py",
            "utils/**/*.py",
            "configs/**/*.yaml",
            "experiments/configs/**/*.yaml",
            "lid_benchmarks/UPSTREAM.yaml",
            "lid_benchmarks/DATA.yaml",
        )
    )
    files: set[Path] = set()
    for pattern in selected_patterns:
        files.update(path for path in root.glob(pattern) if path.is_file())
    digest = hashlib.sha256()
    for path in sorted(files):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        file_digest = bytes.fromhex(sha256_path(path))
        digest.update(file_digest)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str | None:
    process = subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if process.returncode != 0:
        return None
    return process.stdout.strip()


def git_state(root: Path) -> dict[str, Any]:
    head = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    return {
        "head": head,
        "branch": _git(root, "branch", "--show-current"),
        "dirty": bool(status),
        "status_sha256": sha256_bytes((status or "").encode("utf-8")),
    }


def environment_state() -> dict[str, Any]:
    distributions: dict[str, str] = {}
    for package in (
        "lid-estimation",
        "hydra-core",
        "numpy",
        "omegaconf",
        "torch",
        "torchvision",
    ):
        try:
            distributions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "packages": distributions,
    }


@dataclass(frozen=True)
class RunManifest:
    schema_version: int
    run_id: str
    created_at_utc: str
    config_sha256: str
    source_tree_sha256: str
    dataset_record_sha256: str
    environment_sha256: str
    upstream_sha: str
    git: Mapping[str, Any]
    environment: Mapping[str, Any]
    resolved_config: Mapping[str, Any]
    dataset: Mapping[str, Any]
    outputs: Mapping[str, str]


def hash_dataset_record(dataset: Mapping[str, Any]) -> str:
    """Hash canonical scientific identity, excluding machine-local paths.

    File locations remain in the display record, while logical keys, sizes and
    content hashes carry scientific identity. Registry/archive content has its
    own digest, so checkout prefixes cannot perturb an otherwise identical run.
    """

    identity = dict(dataset)
    for field in ("archive_path", "extracted_root", "registry", "registry_overlay"):
        identity.pop(field, None)
    source_files = identity.get("source_files")
    if isinstance(source_files, Mapping):
        identity["source_files"] = {
            str(relative): {
                field: record.get(field)
                for field in ("size_bytes", "sha256")
            }
            for relative, record in source_files.items()
            if isinstance(record, Mapping)
        }
    learned_input = identity.get("learned_input")
    if isinstance(learned_input, Mapping):
        canonical_learned_input = dict(learned_input)
        canonical_learned_input.pop("bundle_root", None)
        identity["learned_input"] = canonical_learned_input
    payload = {"identity_schema_version": 1, "dataset": identity}
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def hash_environment_record(environment: Mapping[str, Any]) -> str:
    """Hash stable runtime fields while excluding host/process metadata."""

    identity = {
        field: environment.get(field)
        for field in ("python", "platform", "machine", "packages")
    }
    return sha256_bytes(canonical_json(identity).encode("utf-8"))


def make_run_id(
    *,
    config: Mapping[str, Any],
    source_tree_sha256: str,
    dataset_record_sha256: str,
    environment_sha256: str,
) -> str:
    identity = {
        "config": config,
        "source_tree_sha256": source_tree_sha256,
        "dataset_record_sha256": dataset_record_sha256,
        "environment_sha256": environment_sha256,
    }
    return sha256_bytes(canonical_json(identity).encode("utf-8"))[:20]


def build_manifest(
    *,
    root: Path,
    config: Mapping[str, Any],
    dataset: Mapping[str, Any],
    outputs: Mapping[str, str],
    upstream_sha: str,
) -> RunManifest:
    source_hash = hash_declared_sources(root)
    dataset_record_hash = hash_dataset_record(dataset)
    environment = environment_state()
    environment_hash = hash_environment_record(environment)
    return RunManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        run_id=make_run_id(
            config=config,
            source_tree_sha256=source_hash,
            dataset_record_sha256=dataset_record_hash,
            environment_sha256=environment_hash,
        ),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        config_sha256=sha256_bytes(canonical_json(config).encode("utf-8")),
        source_tree_sha256=source_hash,
        dataset_record_sha256=dataset_record_hash,
        environment_sha256=environment_hash,
        upstream_sha=upstream_sha,
        git=git_state(root),
        environment=environment,
        resolved_config=config,
        dataset=dataset,
        outputs=outputs,
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_manifest(path: Path, manifest: RunManifest) -> None:
    write_json(path, asdict(manifest))


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_run_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 20
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_utc_timestamp(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.utcoffset() == timezone.utc.utcoffset(parsed)


def _load_json_object(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"non-finite JSON number: {value}")

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        return None, [f"invalid manifest JSON: {error}"]
    if not isinstance(payload, dict):
        return None, ["manifest must be a JSON object"]
    return payload, []


def _safe_output_path(relative: object, *, manifest_name: str) -> str | None:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        return None
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or pure.as_posix() != relative
        or relative == manifest_name
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        return None
    return pure.as_posix()


def _actual_output_inventory(
    run_dir: Path, manifest_path: Path
) -> tuple[set[str], list[str]]:
    files: set[str] = set()
    errors: list[str] = []
    if manifest_path.is_symlink():
        errors.append("manifest.json must not be a symlink")
    elif not manifest_path.is_file():
        errors.append("manifest.json must be a regular file")

    # Do not follow directory symlinks. Reject them explicitly rather than
    # silently treating them as empty directories.
    for directory, directory_names, file_names in os.walk(
        run_dir, topdown=True, followlinks=False
    ):
        directory_path = Path(directory)
        retained_directories: list[str] = []
        for name in directory_names:
            candidate = directory_path / name
            relative = candidate.relative_to(run_dir).as_posix()
            if candidate.is_symlink():
                errors.append(f"symlink is not allowed in run directory: {relative}")
            else:
                retained_directories.append(name)
        directory_names[:] = retained_directories

        for name in file_names:
            candidate = directory_path / name
            if candidate == manifest_path:
                continue
            relative = candidate.relative_to(run_dir).as_posix()
            if candidate.is_symlink():
                errors.append(f"symlink is not allowed in run directory: {relative}")
            elif not candidate.is_file():
                errors.append(f"non-regular output is not allowed: {relative}")
            else:
                files.add(relative)
    return files, errors


def _expected_identity(
    expected: RunManifest | Mapping[str, Any] | None,
) -> dict[str, str]:
    if expected is None:
        return {}
    payload: Mapping[str, Any]
    if isinstance(expected, RunManifest):
        payload = asdict(expected)
    else:
        payload = expected
    identity: dict[str, str] = {}
    for field in (
        "run_id",
        "config_sha256",
        "source_tree_sha256",
        "dataset_record_sha256",
        "environment_sha256",
        "upstream_sha",
    ):
        value = payload.get(field)
        if value is not None:
            identity[field] = str(value)
    dataset = payload.get("dataset")
    if isinstance(dataset, Mapping) and dataset.get("sha256") is not None:
        identity["dataset_sha256"] = str(dataset["sha256"])
    elif payload.get("dataset_sha256") is not None:
        identity["dataset_sha256"] = str(payload["dataset_sha256"])
    return identity


def _recorded_path_matches(recorded: object, configured: object) -> bool:
    if not isinstance(recorded, str) or not recorded:
        return False
    if not isinstance(configured, str) or not configured or "\\" in configured:
        return False
    recorded_path = PurePosixPath(recorded)
    configured_path = PurePosixPath(configured)
    if configured_path.is_absolute():
        return recorded_path == configured_path
    if any(part in {"", ".", ".."} for part in configured_path.parts):
        return False
    return recorded_path.parts[-len(configured_path.parts) :] == configured_path.parts


def _dataset_provenance_errors(
    *,
    dataset: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
    upstream_sha: object,
) -> list[str]:
    """Bind the dataset record to the normalized Hydra dataset selection."""

    errors: list[str] = []
    dataset_config = resolved_config.get("dataset")
    if not isinstance(dataset_config, Mapping):
        return ["resolved_config.dataset must be an object"]
    source = dataset_config.get("source")
    kind = dataset.get("kind")
    if not _is_sha256(dataset.get("training_dataset_sha256")):
        errors.append("dataset.training_dataset_sha256 is invalid")
    if source == "synthetic_flat":
        if kind != "synthetic_flat":
            errors.append(
                "dataset.kind is inconsistent with resolved_config.dataset.source"
            )
        if upstream_sha != "not_applicable":
            errors.append("synthetic dataset must use upstream_sha='not_applicable'")
        generator = dataset.get("generator")
        if generator != dataset_config:
            errors.append(
                "dataset.generator is inconsistent with resolved_config.dataset"
            )
        configured_seed = dataset_config.get("seed", 0)
        if (
            isinstance(configured_seed, bool)
            or not isinstance(configured_seed, int)
            or dataset.get("seed") != configured_seed
        ):
            errors.append(
                "dataset.seed is inconsistent with resolved_config.dataset.seed"
            )
        if "upstream_revision" in dataset:
            errors.append("synthetic dataset must not declare upstream_revision")
        if dataset.get("training_dataset_identity_kind") != (
            "canonical_array_sha256_v1"
        ):
            errors.append("synthetic training dataset identity kind is invalid")
        return errors

    if source != "lid_benchmarks":
        return [
            "resolved_config.dataset.source must be 'synthetic_flat' or "
            "'lid_benchmarks'"
        ]

    source_kind = dataset_config.get("source_kind")
    if source_kind not in {"exact_archive", "generated_at_pinned_revision"}:
        errors.append("resolved_config.dataset.source_kind is invalid")
    if kind != source_kind:
        errors.append(
            "dataset.kind is inconsistent with resolved_config.dataset.source_kind"
        )
    revision = dataset.get("upstream_revision")
    if not isinstance(revision, str) or not revision:
        errors.append("dataset.upstream_revision must be a non-empty string")
    elif revision != upstream_sha:
        errors.append("dataset.upstream_revision does not match upstream_sha")
    if revision != EXPECTED_LID_BENCHMARKS_SHA:
        errors.append("dataset.upstream_revision does not match pinned upstream")
    if upstream_sha != EXPECTED_LID_BENCHMARKS_SHA:
        errors.append("upstream_sha does not match pinned upstream")

    if dataset.get("dataset") != dataset.get("name"):
        errors.append("dataset.dataset must match dataset.name")
    registry = dataset.get("registry")
    configured_registry = dataset_config.get(
        "registry", "configs/datasets/registry/paper_benchmarks.yaml"
    )
    if not _recorded_path_matches(registry, configured_registry):
        errors.append("dataset.registry is inconsistent with resolved_config.dataset")
    if not _is_sha256(dataset.get("registry_sha256")):
        errors.append("dataset.registry_sha256 is invalid")

    source_files = dataset.get("source_files")
    if not isinstance(source_files, Mapping) or not source_files:
        errors.append("dataset.source_files must be a non-empty object")
    else:
        for relative, record in source_files.items():
            safe_relative = _safe_output_path(relative, manifest_name="")
            if safe_relative is None:
                errors.append(f"unsafe dataset.source_files path: {relative!r}")
                continue
            if not isinstance(record, Mapping):
                errors.append(
                    f"dataset.source_files[{safe_relative!r}] must be an object"
                )
                continue
            if not isinstance(record.get("path"), str) or not record.get("path"):
                errors.append(
                    f"dataset.source_files[{safe_relative!r}].path is invalid"
                )
            size = record.get("size_bytes")
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                errors.append(
                    f"dataset.source_files[{safe_relative!r}].size_bytes is invalid"
                )
            if not _is_sha256(record.get("sha256")):
                errors.append(
                    f"dataset.source_files[{safe_relative!r}].sha256 is invalid"
                )
        training_key = f"train/{dataset.get('representation')}.npy"
        training_record = source_files.get(training_key)
        if not isinstance(training_record, Mapping) or training_record.get(
            "sha256"
        ) != dataset.get("training_dataset_sha256"):
            errors.append(
                "dataset.training_dataset_sha256 does not match full training "
                f"source file {training_key}"
            )
    if dataset.get("training_dataset_identity_kind") != (
        "source_npy_file_sha256_v1"
    ):
        errors.append("benchmark training dataset identity kind is invalid")

    if source_kind == "exact_archive":
        if dataset.get("archive_sha256") != EXACT_ARCHIVE_SHA256:
            errors.append(
                "exact archive dataset.archive_sha256 does not match canonical archive"
            )
        configured_archive = dataset_config.get("archive", "data/benchmarks.zip")
        if not _recorded_path_matches(dataset.get("archive_path"), configured_archive):
            errors.append(
                "dataset.archive_path is inconsistent with resolved_config.dataset"
            )
        configured_extracted = dataset_config.get(
            "extracted_root", "data/lid_benchmarks_exact"
        )
        if not _recorded_path_matches(
            dataset.get("extracted_root"), configured_extracted
        ):
            errors.append(
                "dataset.extracted_root is inconsistent with resolved_config.dataset"
            )
    else:
        if any(
            field in dataset
            for field in ("archive_sha256", "archive_path", "extracted_root")
        ):
            errors.append("generated dataset must not declare exact archive provenance")
        configured_overlay = dataset_config.get("registry_overlay")
        if not _recorded_path_matches(
            dataset.get("registry_overlay"), configured_overlay
        ):
            errors.append(
                "dataset.registry_overlay is inconsistent with "
                "resolved_config.dataset"
            )
        if not _is_sha256(dataset.get("registry_overlay_sha256")):
            errors.append("dataset.registry_overlay_sha256 is invalid")
    return errors


def validate_manifest(
    path: Path,
    *,
    expected: RunManifest | Mapping[str, Any] | None = None,
    expected_run_id: str | None = None,
    expected_config_sha256: str | None = None,
    expected_source_tree_sha256: str | None = None,
    expected_dataset_sha256: str | None = None,
    expected_dataset_record_sha256: str | None = None,
    expected_environment_sha256: str | None = None,
    expected_upstream_sha: str | None = None,
) -> list[str]:
    """Validate a run manifest, its identity, and exact file inventory.

    ``expected`` compares only immutable identity fields of a precomputed
    :class:`RunManifest` (or equivalent mapping), not timestamps, runtime
    metadata, or outputs. Individual ``expected_*`` arguments override the
    corresponding value from ``expected``. A runner should pass its placeholder
    manifest when deciding whether an existing cell is reusable.
    """

    payload, errors = _load_json_object(path)
    if payload is None:
        return errors

    missing = sorted(_MANIFEST_FIELDS - payload.keys())
    unexpected = sorted(payload.keys() - _MANIFEST_FIELDS)
    if missing:
        errors.append(f"missing manifest fields: {', '.join(missing)}")
    if unexpected:
        errors.append(f"unexpected manifest fields: {', '.join(unexpected)}")

    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION or isinstance(
        payload.get("schema_version"), bool
    ):
        errors.append("unsupported manifest schema_version")
    if not _is_run_id(payload.get("run_id")):
        errors.append("run_id must be 20 lowercase hexadecimal characters")
    if not _is_utc_timestamp(payload.get("created_at_utc")):
        errors.append("created_at_utc must be a timezone-aware UTC timestamp")
    for field in (
        "config_sha256",
        "source_tree_sha256",
        "dataset_record_sha256",
        "environment_sha256",
    ):
        if not _is_sha256(payload.get(field)):
            errors.append(f"{field} must be a lowercase SHA-256 digest")
    if not isinstance(payload.get("upstream_sha"), str) or not payload.get(
        "upstream_sha"
    ):
        errors.append("upstream_sha must be a non-empty string")
    for field in ("git", "environment", "resolved_config", "dataset"):
        if not isinstance(payload.get(field), dict):
            errors.append(f"{field} must be an object")

    environment = payload.get("environment")
    if isinstance(environment, dict):
        for field in ("python", "platform", "machine"):
            if not isinstance(environment.get(field), str) or not environment.get(
                field
            ):
                errors.append(f"environment.{field} must be a non-empty string")
        packages = environment.get("packages")
        if not isinstance(packages, dict) or any(
            not isinstance(name, str)
            or not name
            or not isinstance(version, str)
            or not version
            for name, version in (
                packages.items() if isinstance(packages, dict) else ()
            )
        ):
            errors.append("environment.packages must map names to versions")
        try:
            actual_environment_hash = hash_environment_record(environment)
        except (TypeError, ValueError) as error:
            errors.append(f"environment is not canonicalizable: {error}")
        else:
            if payload.get("environment_sha256") != actual_environment_hash:
                errors.append(
                    "environment_sha256 is inconsistent with environment: "
                    f"expected {actual_environment_hash}, "
                    f"got {payload.get('environment_sha256')}"
                )

    resolved_config = payload.get("resolved_config")
    if isinstance(resolved_config, dict):
        try:
            actual_config_hash = sha256_bytes(
                canonical_json(resolved_config).encode("utf-8")
            )
        except (TypeError, ValueError) as error:
            errors.append(f"resolved_config is not canonicalizable: {error}")
        else:
            if payload.get("config_sha256") != actual_config_hash:
                errors.append(
                    "config_sha256 is inconsistent with resolved_config: "
                    f"expected {actual_config_hash}, got {payload.get('config_sha256')}"
                )

    dataset = payload.get("dataset")
    dataset_hash: object = dataset.get("sha256") if isinstance(dataset, dict) else None
    if not _is_sha256(dataset_hash):
        errors.append("dataset.sha256 must be a lowercase SHA-256 digest")
    actual_dataset_record_hash: str | None = None
    if isinstance(dataset, dict):
        try:
            actual_dataset_record_hash = hash_dataset_record(dataset)
        except (TypeError, ValueError) as error:
            errors.append(f"dataset record is not canonicalizable: {error}")
        else:
            if payload.get("dataset_record_sha256") != actual_dataset_record_hash:
                errors.append(
                    "dataset_record_sha256 is inconsistent with dataset: "
                    f"expected {actual_dataset_record_hash}, "
                    f"got {payload.get('dataset_record_sha256')}"
                )
        if isinstance(resolved_config, dict):
            errors.extend(
                _dataset_provenance_errors(
                    dataset=dataset,
                    resolved_config=resolved_config,
                    upstream_sha=payload.get("upstream_sha"),
                )
            )
    if (
        isinstance(resolved_config, dict)
        and _is_sha256(payload.get("source_tree_sha256"))
        and actual_dataset_record_hash is not None
        and _is_sha256(payload.get("environment_sha256"))
    ):
        expected_self_run_id = make_run_id(
            config=resolved_config,
            source_tree_sha256=str(payload["source_tree_sha256"]),
            dataset_record_sha256=actual_dataset_record_hash,
            environment_sha256=str(payload["environment_sha256"]),
        )
        if payload.get("run_id") != expected_self_run_id:
            errors.append(
                "run_id is inconsistent with resolved_config/source_tree/dataset: "
                f"expected {expected_self_run_id}, got {payload.get('run_id')}"
            )

    expected_identity = _expected_identity(expected)
    explicit_expected = {
        "run_id": expected_run_id,
        "config_sha256": expected_config_sha256,
        "source_tree_sha256": expected_source_tree_sha256,
        "dataset_sha256": expected_dataset_sha256,
        "dataset_record_sha256": expected_dataset_record_sha256,
        "environment_sha256": expected_environment_sha256,
        "upstream_sha": expected_upstream_sha,
    }
    expected_identity.update(
        {
            field: value
            for field, value in explicit_expected.items()
            if value is not None
        }
    )
    actual_identity = {
        "run_id": payload.get("run_id"),
        "config_sha256": payload.get("config_sha256"),
        "source_tree_sha256": payload.get("source_tree_sha256"),
        "dataset_sha256": dataset_hash,
        "dataset_record_sha256": payload.get("dataset_record_sha256"),
        "environment_sha256": payload.get("environment_sha256"),
        "upstream_sha": payload.get("upstream_sha"),
    }
    for field, expected_value in expected_identity.items():
        actual_value = actual_identity[field]
        if actual_value != expected_value:
            errors.append(
                f"manifest identity mismatch for {field}: "
                f"expected {expected_value}, got {actual_value}"
            )

    outputs = payload.get("outputs")
    if not isinstance(outputs, dict):
        return errors + ["outputs must be an object"]
    if not outputs:
        errors.append("outputs must be a non-empty object")

    run_dir = path.parent
    actual_files, inventory_errors = _actual_output_inventory(run_dir, path)
    errors.extend(inventory_errors)
    declared_files: set[str] = set()
    for relative, expected_hash in outputs.items():
        safe_relative = _safe_output_path(relative, manifest_name=path.name)
        if safe_relative is None:
            errors.append(f"unsafe output path: {relative!r}")
            continue
        declared_files.add(safe_relative)
        if not _is_sha256(expected_hash):
            errors.append(
                f"output checksum for {safe_relative} must be a lowercase "
                "SHA-256 digest"
            )
            continue
        output = run_dir / safe_relative
        if safe_relative not in actual_files:
            errors.append(f"missing output: {safe_relative}")
            continue
        actual = sha256_path(output)
        if actual != expected_hash:
            errors.append(
                f"checksum mismatch for {safe_relative}: "
                f"expected {expected_hash}, got {actual}"
            )

    for relative in sorted(actual_files - declared_files):
        errors.append(f"unlisted output: {relative}")
    return errors
