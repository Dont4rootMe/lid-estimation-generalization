from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from datasets.archive import EXACT_ARCHIVE_SHA256
from utils.provenance import EXPECTED_LID_BENCHMARKS_SHA
from experiments.run_manifest import (
    MANIFEST_SCHEMA_VERSION,
    RunManifest,
    canonical_json,
    hash_dataset_record,
    hash_declared_sources,
    hash_environment_record,
    make_run_id,
    sha256_bytes,
    sha256_path,
    validate_manifest,
    write_manifest,
)


def test_source_tree_hash_covers_source_and_wheel_hydra_layouts(
    tmp_path: Path,
) -> None:
    source_config = tmp_path / "configs" / "config.yaml"
    wheel_config = tmp_path / "experiments" / "configs" / "config.yaml"
    source_config.parent.mkdir(parents=True)
    wheel_config.parent.mkdir(parents=True)
    source_config.write_text("value: source\n", encoding="utf-8")
    wheel_config.write_text("value: wheel\n", encoding="utf-8")

    baseline = hash_declared_sources(tmp_path)
    source_config.write_text("value: changed-source\n", encoding="utf-8")
    assert hash_declared_sources(tmp_path) != baseline

    source_changed = hash_declared_sources(tmp_path)
    wheel_config.write_text("value: changed-wheel\n", encoding="utf-8")
    assert hash_declared_sources(tmp_path) != source_changed


def _synthetic_config(seed: int = 7) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "dataset": {
            "source": "synthetic_flat",
            "seed": seed,
            "ambient_dim": 8,
            "intrinsic_dim": 3,
            "n_train": 32,
            "n_validation": 8,
            "n_test": 8,
        },
        "model": {"name": "test"},
    }


def _synthetic_dataset(config: Mapping[str, Any]) -> dict[str, Any]:
    dataset_config = dict(config["dataset"])
    return {
        "kind": "synthetic_flat",
        "seed": dataset_config["seed"],
        "generator": dataset_config,
        "training_dataset_identity_kind": "canonical_array_sha256_v1",
        "training_dataset_sha256": "1" * 64,
        "name": "synthetic_flat",
        "representation": "coordinates",
        "sha256": "b" * 64,
        "n_reference": 32,
        "n_validation": 8,
        "n_test": 8,
        "ambient_dim": 8,
    }


def _exact_config() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "dataset": {
            "source": "lid_benchmarks",
            "source_kind": "exact_archive",
            "archive": "data/benchmarks.zip",
            "extracted_root": "data/lid_benchmarks_exact",
            "root": "data/lid_benchmarks_exact/benchmarks",
            "registry": "configs/datasets/registry/paper_benchmarks.yaml",
            "names": ["fixture"],
            "representations": "all",
        },
        "model": {"name": "test"},
    }


def _exact_dataset(config: Mapping[str, Any]) -> dict[str, Any]:
    revision = EXPECTED_LID_BENCHMARKS_SHA
    return {
        "kind": "exact_archive",
        "archive_sha256": EXACT_ARCHIVE_SHA256,
        "archive_path": "/checkout/data/benchmarks.zip",
        "extracted_root": "/checkout/data/lid_benchmarks_exact",
        "upstream_revision": revision,
        "registry": "/checkout/configs/datasets/registry/paper_benchmarks.yaml",
        "registry_sha256": "d" * 64,
        "dataset": "fixture",
        "name": "fixture",
        "representation": "dataset",
        "training_dataset_identity_kind": "source_npy_file_sha256_v1",
        "training_dataset_sha256": "e" * 64,
        "source_files": {
            "train/dataset.npy": {
                "path": "/checkout/data/lid_benchmarks_exact/benchmarks/fixture/"
                "train/dataset.npy",
                "size_bytes": 128,
                "sha256": "e" * 64,
            }
        },
        "sha256": "f" * 64,
        "n_reference": 32,
        "n_validation": 8,
        "n_test": 8,
        "ambient_dim": 4,
    }


def _environment() -> dict[str, Any]:
    return {
        "python": "3.11.fixture",
        "executable": "/python",
        "platform": "fixture-os",
        "machine": "fixture-machine",
        "hostname": "ignored-host",
        "pid": 123,
        "packages": {"numpy": "2.0"},
    }


def _manifest(
    tmp_path: Path,
    *,
    output_name: str = "predictions.npy",
    config: Mapping[str, Any] | None = None,
    dataset: Mapping[str, Any] | None = None,
    upstream_sha: str = "not_applicable",
) -> tuple[Path, RunManifest]:
    output = tmp_path / output_name
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"prediction")
    resolved_config = dict(config or _synthetic_config())
    dataset_record = dict(dataset or _synthetic_dataset(resolved_config))
    source_hash = "a" * 64
    environment = _environment()
    dataset_record_hash = hash_dataset_record(dataset_record)
    environment_hash = hash_environment_record(environment)
    manifest = RunManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        run_id=make_run_id(
            config=resolved_config,
            source_tree_sha256=source_hash,
            dataset_record_sha256=dataset_record_hash,
            environment_sha256=environment_hash,
        ),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        config_sha256=sha256_bytes(
            canonical_json(resolved_config).encode("utf-8")
        ),
        source_tree_sha256=source_hash,
        dataset_record_sha256=dataset_record_hash,
        environment_sha256=environment_hash,
        upstream_sha=upstream_sha,
        git={},
        environment=environment,
        resolved_config=resolved_config,
        dataset=dataset_record,
        outputs={output_name: sha256_path(output)},
    )
    path = tmp_path / "manifest.json"
    write_manifest(path, manifest)
    return path, manifest


def _rewrite(path: Path, manifest: RunManifest) -> None:
    write_manifest(path, manifest)


def _reidentify(
    manifest: RunManifest,
    *,
    config: Mapping[str, Any] | None = None,
    dataset: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
    source_tree_sha256: str | None = None,
    upstream_sha: str | None = None,
) -> RunManifest:
    resolved_config = dict(config or manifest.resolved_config)
    dataset_record = dict(dataset or manifest.dataset)
    environment_record = dict(environment or manifest.environment)
    source_hash = source_tree_sha256 or manifest.source_tree_sha256
    dataset_record_hash = hash_dataset_record(dataset_record)
    environment_hash = hash_environment_record(environment_record)
    return replace(
        manifest,
        run_id=make_run_id(
            config=resolved_config,
            source_tree_sha256=source_hash,
            dataset_record_sha256=dataset_record_hash,
            environment_sha256=environment_hash,
        ),
        config_sha256=sha256_bytes(
            canonical_json(resolved_config).encode("utf-8")
        ),
        source_tree_sha256=source_hash,
        dataset_record_sha256=dataset_record_hash,
        environment_sha256=environment_hash,
        upstream_sha=upstream_sha or manifest.upstream_sha,
        environment=environment_record,
        resolved_config=resolved_config,
        dataset=dataset_record,
    )


def test_run_id_is_deterministic_and_dataset_and_environment_sensitive() -> None:
    arguments = {
        "config": {"seed": 7},
        "source_tree_sha256": "a" * 64,
        "dataset_record_sha256": "b" * 64,
        "environment_sha256": "c" * 64,
    }
    first = make_run_id(**arguments)
    assert first == make_run_id(**arguments)
    assert first != make_run_id(**{**arguments, "dataset_record_sha256": "d" * 64})
    assert first != make_run_id(**{**arguments, "environment_sha256": "e" * 64})


def test_dataset_record_identity_ignores_checkout_paths_not_content() -> None:
    config = _exact_config()
    first = _exact_dataset(config)
    relocated = dict(first)
    relocated.update(
        {
            "archive_path": "/other-machine/data/benchmarks.zip",
            "extracted_root": "/other-machine/data/lid_benchmarks_exact",
            "registry": (
                "/other-machine/configs/datasets/registry/paper_benchmarks.yaml"
            ),
        }
    )
    relocated_files = {
        key: {**value, "path": f"/other-machine/{key}"}
        for key, value in dict(first["source_files"]).items()
    }
    relocated["source_files"] = relocated_files
    assert hash_dataset_record(first) == hash_dataset_record(relocated)

    changed_content = dict(relocated)
    changed_files = {
        key: {**value, "sha256": "0" * 64}
        for key, value in relocated_files.items()
    }
    changed_content["source_files"] = changed_files
    assert hash_dataset_record(first) != hash_dataset_record(changed_content)


def test_manifest_validates_output_checksum(tmp_path: Path) -> None:
    path, manifest = _manifest(tmp_path)
    assert validate_manifest(path) == []
    (tmp_path / "predictions.npy").write_bytes(b"tampered")
    assert any("checksum mismatch" in error for error in validate_manifest(path))
    assert json.loads(path.read_text())["run_id"] == manifest.run_id


def test_manifest_requires_nonempty_exact_output_inventory(tmp_path: Path) -> None:
    path, manifest = _manifest(tmp_path)
    _rewrite(path, replace(manifest, outputs={}))
    errors = validate_manifest(path)
    assert "outputs must be a non-empty object" in errors
    assert "unlisted output: predictions.npy" in errors


def test_manifest_rejects_unlisted_extra_output(tmp_path: Path) -> None:
    path, _ = _manifest(tmp_path)
    (tmp_path / "extra.txt").write_text("not declared", encoding="utf-8")
    assert "unlisted output: extra.txt" in validate_manifest(path)


@pytest.mark.parametrize(
    "unsafe_path",
    ["../escape.npy", "/absolute.npy", "nested/../prediction.npy", "a\\b.npy"],
)
def test_manifest_rejects_unsafe_output_paths(
    tmp_path: Path, unsafe_path: str
) -> None:
    path, manifest = _manifest(tmp_path)
    _rewrite(
        path,
        replace(manifest, outputs={unsafe_path: next(iter(manifest.outputs.values()))}),
    )
    assert any("unsafe output path" in error for error in validate_manifest(path))


def test_manifest_rejects_output_symlinks(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.bin"
    outside.write_bytes(b"prediction")
    link = tmp_path / "linked.bin"
    link.symlink_to(outside)
    path, manifest = _manifest(tmp_path)
    _rewrite(
        path,
        replace(
            manifest,
            outputs={
                "predictions.npy": manifest.outputs["predictions.npy"],
                "linked.bin": sha256_path(outside),
            },
        ),
    )
    errors = validate_manifest(path)
    assert "symlink is not allowed in run directory: linked.bin" in errors
    assert "missing output: linked.bin" in errors


@pytest.mark.parametrize(
    "field",
    [
        "run_id",
        "config_sha256",
        "dataset_record_sha256",
        "environment_sha256",
    ],
)
def test_manifest_rejects_internally_inconsistent_identity(
    tmp_path: Path, field: str
) -> None:
    path, manifest = _manifest(tmp_path)
    forged = replace(manifest, **{field: "0" * len(getattr(manifest, field))})
    _rewrite(path, forged)
    errors = validate_manifest(path)
    assert any(f"{field} is inconsistent" in error for error in errors)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("source_tree_sha256", "d" * 64),
        ("dataset_sha256", "e" * 64),
        ("environment", "changed-platform"),
    ],
)
def test_expected_manifest_rejects_self_consistent_forged_identity(
    tmp_path: Path, field: str, replacement: str
) -> None:
    path, expected = _manifest(tmp_path)
    dataset = dict(expected.dataset)
    environment = dict(expected.environment)
    source_hash = expected.source_tree_sha256
    if field == "source_tree_sha256":
        source_hash = replacement
    elif field == "dataset_sha256":
        dataset["sha256"] = replacement
    else:
        environment["platform"] = replacement
    forged = _reidentify(
        expected,
        dataset=dataset,
        environment=environment,
        source_tree_sha256=source_hash,
    )
    _rewrite(path, forged)
    assert validate_manifest(path) == []
    errors = validate_manifest(path, expected=expected)
    expected_field = {
        "source_tree_sha256": "source_tree_sha256",
        "dataset_sha256": "dataset_sha256",
        "environment": "environment_sha256",
    }[field]
    assert any(
        f"manifest identity mismatch for {expected_field}" in error for error in errors
    )


def test_synthetic_kind_and_seed_cannot_be_relabelled(tmp_path: Path) -> None:
    path, manifest = _manifest(tmp_path)
    for field, replacement, expected_error in (
        ("kind", "exact_archive", "dataset.kind is inconsistent"),
        ("seed", 99, "dataset.seed is inconsistent"),
    ):
        dataset = dict(manifest.dataset)
        dataset[field] = replacement
        forged = _reidentify(manifest, dataset=dataset)
        _rewrite(path, forged)
        assert any(expected_error in error for error in validate_manifest(path))


@pytest.mark.parametrize(
    "field",
    ["archive_sha256", "source_files", "registry"],
)
def test_expected_identity_rejects_exact_provenance_relabelling(
    tmp_path: Path, field: str
) -> None:
    config = _exact_config()
    dataset = _exact_dataset(config)
    path, expected = _manifest(
        tmp_path,
        config=config,
        dataset=dataset,
        upstream_sha=str(dataset["upstream_revision"]),
    )
    forged_dataset = dict(expected.dataset)
    if field == "archive_sha256":
        forged_dataset[field] = "0" * 64
    elif field == "registry":
        forged_dataset[field] = "/checkout/configs/datasets/registry/other.yaml"
    else:
        source_files = {
            key: dict(value)
            for key, value in dict(forged_dataset[field]).items()
        }
        source_files["train/dataset.npy"]["sha256"] = "0" * 64
        forged_dataset[field] = source_files
    forged = _reidentify(expected, dataset=forged_dataset)
    _rewrite(path, forged)
    errors = validate_manifest(path, expected=expected)
    if field == "registry":
        assert any("dataset.registry is inconsistent" in error for error in errors)
    else:
        assert any(
            "manifest identity mismatch for dataset_record_sha256" in error
            for error in errors
        )


def test_upstream_revision_is_bound_to_manifest_upstream_sha(tmp_path: Path) -> None:
    config = _exact_config()
    dataset = _exact_dataset(config)
    path, manifest = _manifest(
        tmp_path,
        config=config,
        dataset=dataset,
        upstream_sha=str(dataset["upstream_revision"]),
    )
    forged_dataset = dict(manifest.dataset)
    forged_dataset["upstream_revision"] = "0" * 40
    _rewrite(path, _reidentify(manifest, dataset=forged_dataset))
    assert "dataset.upstream_revision does not match upstream_sha" in validate_manifest(
        path
    )

    forged_revision = "0" * 40
    _rewrite(
        path,
        _reidentify(
            manifest,
            dataset={**manifest.dataset, "upstream_revision": forged_revision},
            upstream_sha=forged_revision,
        ),
    )
    errors = validate_manifest(path)
    assert "dataset.upstream_revision does not match pinned upstream" in errors
    assert "upstream_sha does not match pinned upstream" in errors


def test_explicit_expected_values_override_expected_manifest(tmp_path: Path) -> None:
    path, manifest = _manifest(tmp_path)
    assert (
        validate_manifest(
            path,
            expected=replace(manifest, upstream_sha="forged"),
            expected_upstream_sha=manifest.upstream_sha,
        )
        == []
    )


def test_manifest_validates_nested_exact_inventory(tmp_path: Path) -> None:
    path, _ = _manifest(tmp_path, output_name="nested/predictions.npy")
    assert validate_manifest(path) == []


def test_manifest_validates_required_schema_fields(tmp_path: Path) -> None:
    path, manifest = _manifest(tmp_path)
    payload = asdict(manifest)
    del payload["environment"]
    payload["surprise"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    errors = validate_manifest(path)
    assert "missing manifest fields: environment" in errors
    assert "unexpected manifest fields: surprise" in errors
