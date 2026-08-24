"""Deterministic SHA-256 manifests and pinned-upstream verification."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

import yaml


EXPECTED_LID_BENCHMARKS_SHA = "2dcb8e41015f53413ff1ddd049bb006c81a5df52"
MANIFEST_SCHEMA_VERSION = 1


class ProvenanceError(RuntimeError):
    """Base class for provenance failures."""


class UpstreamRevisionError(ProvenanceError):
    """Raised when the vendored upstream checkout is not the pinned source."""


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the lowercase SHA-256 digest of a regular file."""

    source = Path(path)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not source.is_file():
        raise FileNotFoundError(f"not a regular file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_excluded_paths(
    root: Path, excluded_paths: Sequence[str | Path]
) -> tuple[str, ...]:
    normalized: set[str] = set()
    for value in excluded_paths:
        path = Path(value)
        if path.is_absolute():
            try:
                path = path.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"excluded path is outside manifest root: {path}") from exc
        relative = path.as_posix()
        if relative in {"", "."} or relative.startswith("../"):
            raise ValueError(f"invalid excluded manifest path: {value}")
        normalized.add(relative)
    return tuple(sorted(normalized))


def _tree_digest(files: Mapping[str, Mapping[str, Any]]) -> str:
    """Hash canonical file records, independent of the tree's absolute path."""

    digest = hashlib.sha256()
    for relative_path in sorted(files):
        record = files[relative_path]
        canonical_record = {
            "path": relative_path,
            "sha256": str(record["sha256"]),
            "size_bytes": int(record["size_bytes"]),
        }
        digest.update(
            json.dumps(
                canonical_record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _tree_manifest(
    root: str | Path,
    *,
    excluded_paths: Sequence[str | Path] = (),
) -> dict[str, Any]:
    base = Path(root)
    if not base.is_dir():
        raise FileNotFoundError(f"manifest root is not a directory: {base}")
    base = base.resolve()
    excluded = set(_normalize_excluded_paths(base, excluded_paths))

    files: dict[str, dict[str, int | str]] = {}
    for path in sorted(base.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(base).as_posix()
        if relative in excluded:
            continue
        if path.is_symlink():
            raise ProvenanceError(
                f"symbolic links are not allowed in a reproducibility manifest: {path}"
            )
        if not path.is_file():
            continue
        files[relative] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    tree_hash = _tree_digest(files)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "algorithm": "sha256",
        # Informational only.  Digests and verification use relative paths, so
        # an archive remains verifiable after being moved to another machine.
        "root_name": base.name,
        "excluded_paths": sorted(excluded),
        "file_count": len(files),
        # ``sha256`` is the compact run-manifest interface; the longer alias
        # makes its tree-wide meaning explicit to humans reading the file.
        "sha256": tree_hash,
        "tree_sha256": tree_hash,
        "files": files,
    }


def tree_manifest(root: str | Path) -> dict[str, Any]:
    """Build a deterministic SHA-256 manifest for every file below ``root``."""

    return _tree_manifest(root)


def write_tree_manifest(root: str | Path, output: str | Path) -> dict[str, Any]:
    """Create a manifest and atomically write it as canonical JSON.

    If ``output`` is inside ``root``, the manifest records that path as an
    explicit exclusion.  This makes the common ``root/manifest.json`` workflow
    immediately self-verifiable rather than making the manifest hash itself.
    """

    base = Path(root).resolve()
    destination = Path(output)
    destination_absolute = destination.resolve()
    excluded: tuple[Path, ...] = ()
    try:
        excluded = (destination_absolute.relative_to(base),)
    except ValueError:
        pass

    manifest = _tree_manifest(base, excluded_paths=excluded)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    finally:
        if temporary_name is not None:
            temporary_path = Path(temporary_name)
            if temporary_path.exists():
                temporary_path.unlink()
    return manifest


def read_tree_manifest(path: str | Path) -> dict[str, Any]:
    """Read a JSON tree manifest."""

    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"cannot read manifest {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProvenanceError(f"manifest {source} must contain a JSON object")
    return value


def verify_tree_manifest(
    root: str | Path,
    manifest: Mapping[str, Any] | str | Path,
) -> list[str]:
    """Return human-readable mismatches; an empty list means exact agreement."""

    if isinstance(manifest, (str, Path)):
        try:
            expected: Mapping[str, Any] = read_tree_manifest(manifest)
        except ProvenanceError as exc:
            return [str(exc)]
    else:
        expected = manifest

    errors: list[str] = []
    if expected.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        errors.append(
            "unsupported manifest schema_version: "
            f"{expected.get('schema_version')!r}"
        )
    if expected.get("algorithm") != "sha256":
        errors.append(f"unsupported manifest algorithm: {expected.get('algorithm')!r}")
    expected_files_raw = expected.get("files")
    if not isinstance(expected_files_raw, Mapping):
        return errors + ["manifest field 'files' must be an object"]
    if not all(isinstance(path, str) for path in expected_files_raw):
        return errors + ["manifest file paths must be strings"]
    invalid_records = [
        path
        for path, record in expected_files_raw.items()
        if not isinstance(record, Mapping)
        or not isinstance(record.get("size_bytes"), int)
        or isinstance(record.get("size_bytes"), bool)
        or int(record["size_bytes"]) < 0
        or not isinstance(record.get("sha256"), str)
        or len(record["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in record["sha256"])
    ]
    if invalid_records:
        return errors + [
            f"manifest contains invalid file records: {sorted(invalid_records)}"
        ]

    excluded_raw = expected.get("excluded_paths", ())
    if not isinstance(excluded_raw, list) or not all(
        isinstance(item, str) for item in excluded_raw
    ):
        return errors + ["manifest field 'excluded_paths' must be a list of paths"]
    try:
        actual = _tree_manifest(root, excluded_paths=excluded_raw)
    except (OSError, ValueError, ProvenanceError) as exc:
        return errors + [f"cannot build current tree manifest: {exc}"]

    actual_files = actual["files"]
    expected_paths = set(expected_files_raw)
    actual_paths = set(actual_files)
    for path in sorted(expected_paths - actual_paths):
        errors.append(f"missing file: {path}")
    for path in sorted(actual_paths - expected_paths):
        errors.append(f"unexpected file: {path}")
    for path in sorted(expected_paths & actual_paths):
        record = expected_files_raw[path]
        assert isinstance(record, Mapping)  # validated above
        expected_size = record.get("size_bytes")
        expected_hash = record.get("sha256")
        actual_record = actual_files[path]
        if expected_size != actual_record["size_bytes"]:
            errors.append(
                f"size mismatch: {path} (expected {expected_size}, "
                f"actual {actual_record['size_bytes']})"
            )
        if expected_hash != actual_record["sha256"]:
            errors.append(f"sha256 mismatch: {path}")

    expected_count = expected.get("file_count")
    if expected_count != len(expected_files_raw):
        errors.append(
            "manifest file_count does not match its file records: "
            f"{expected_count!r} != {len(expected_files_raw)}"
        )
    expected_tree_hash = expected.get("tree_sha256")
    if expected.get("sha256") != expected_tree_hash:
        errors.append("manifest sha256 alias does not match tree_sha256")
    recomputed_expected_hash: str | None = None
    try:
        recomputed_expected_hash = _tree_digest(expected_files_raw)
    except (KeyError, TypeError, ValueError):
        errors.append("manifest contains an invalid file digest record")
    if (
        recomputed_expected_hash is not None
        and expected_tree_hash != recomputed_expected_hash
    ):
        errors.append("manifest tree_sha256 does not match its file records")
    if expected_tree_hash != actual["tree_sha256"] and not any(
        message.startswith(
            (
                "missing file:",
                "unexpected file:",
                "size mismatch:",
                "sha256 mismatch:",
            )
        )
        for message in errors
    ):
        errors.append("tree_sha256 mismatch")
    return errors


def _git(
    checkout: Path,
    *arguments: str,
) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(checkout), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise UpstreamRevisionError("git executable is not available") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip() or f"exit {exc.returncode}"
        raise UpstreamRevisionError(
            f"cannot inspect upstream checkout {checkout}: {detail}"
        ) from exc
    return completed.stdout.strip()


def verify_upstream_source(
    path: str | Path,
    expected_revision: str = EXPECTED_LID_BENCHMARKS_SHA,
    *,
    require_clean: bool = True,
) -> str:
    """Verify the imported LID-Benchmarks source and return its upstream SHA.

    A developer clone is checked through Git.  The normal repository layout is
    a top-level, editable import without nested Git metadata; in that case
    ``UPSTREAM.yaml`` pins hashes of every original file while allowing new
    benchmark files to be added beside them.
    """

    checkout = Path(path)
    if not checkout.is_dir():
        raise UpstreamRevisionError(f"upstream checkout does not exist: {checkout}")
    if len(expected_revision) != 40 or any(
        character not in "0123456789abcdef" for character in expected_revision.lower()
    ):
        raise ValueError(f"expected_revision is not a full Git SHA: {expected_revision!r}")
    if not (checkout / ".git").exists():
        metadata_path = checkout / "UPSTREAM.yaml"
        try:
            metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise UpstreamRevisionError(
                f"cannot read imported-source metadata {metadata_path}: {exc}"
            ) from exc
        if not isinstance(metadata, Mapping):
            raise UpstreamRevisionError(
                f"imported-source metadata must be a YAML mapping: {metadata_path}"
            )
        actual = str(metadata.get("revision", "")).lower()
        if actual != expected_revision.lower():
            raise UpstreamRevisionError(
                f"unexpected LID-Benchmarks revision at {checkout}: expected "
                f"{expected_revision}, found {actual or '<missing>'}"
            )
        source_files = metadata.get("source_files")
        if not isinstance(source_files, Mapping) or not source_files:
            raise UpstreamRevisionError("UPSTREAM.yaml has no source_files mapping")
        mismatches: list[str] = []
        for relative, expected_hash in source_files.items():
            source = checkout / str(relative)
            if not source.is_file():
                mismatches.append(f"missing {relative}")
                continue
            actual_hash = sha256_file(source)
            if actual_hash != str(expected_hash):
                mismatches.append(f"modified {relative}")
        if mismatches and require_clean:
            raise UpstreamRevisionError(
                "original imported files do not match pinned upstream: "
                + ", ".join(mismatches)
            )
        return actual

    actual = _git(checkout, "rev-parse", "HEAD").lower()
    if actual != expected_revision.lower():
        raise UpstreamRevisionError(
            f"unexpected LID-Benchmarks revision at {checkout}: expected "
            f"{expected_revision}, found {actual}"
        )
    if require_clean:
        status = _git(checkout, "status", "--porcelain", "--untracked-files=no")
        if status:
            changed = ", ".join(
                line[3:] if len(line) > 3 else line for line in status.splitlines()
            )
            raise UpstreamRevisionError(
                f"tracked files in upstream checkout are modified: {changed}"
            )
    return actual


# Compatibility alias for callers created before the top-level import layout.
verify_upstream_checkout = verify_upstream_source
