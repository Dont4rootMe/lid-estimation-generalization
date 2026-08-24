from __future__ import annotations

from pathlib import Path

from utils.provenance import sha256_file
from experiments.runner import _dataset_source_anchor_errors


def test_registry_anchor_detects_semantically_equivalent_file_tamper(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "registry.yaml"
    registry.write_text("schema_version: 1\n", encoding="utf-8")
    config = {
        "dataset": {
            "source": "lid_benchmarks",
            "source_kind": "exact_archive",
            "registry": "registry.yaml",
        }
    }
    dataset = {"registry_sha256": sha256_file(registry)}
    assert (
        _dataset_source_anchor_errors(
            dataset=dataset,
            resolved_config=config,
            root=tmp_path,
        )
        == []
    )

    registry.write_text("schema_version: 1\n# tampered\n", encoding="utf-8")
    assert _dataset_source_anchor_errors(
        dataset=dataset,
        resolved_config=config,
        root=tmp_path,
    ) == [
        "dataset.registry_sha256 does not match configured dataset.registry"
    ]


def test_generated_overlay_is_also_anchored(tmp_path: Path) -> None:
    registry = tmp_path / "registry.yaml"
    overlay = tmp_path / "overlay.yaml"
    registry.write_text("schema_version: 1\n", encoding="utf-8")
    overlay.write_text("schema_version: 1\n", encoding="utf-8")
    config = {
        "dataset": {
            "source": "lid_benchmarks",
            "source_kind": "generated_at_pinned_revision",
            "registry": "registry.yaml",
            "registry_overlay": "overlay.yaml",
        }
    }
    dataset = {
        "registry_sha256": sha256_file(registry),
        "registry_overlay_sha256": sha256_file(overlay),
    }
    assert (
        _dataset_source_anchor_errors(
            dataset=dataset,
            resolved_config=config,
            root=tmp_path,
        )
        == []
    )

    overlay.write_text("schema_version: 1\n# tampered\n", encoding="utf-8")
    assert _dataset_source_anchor_errors(
        dataset=dataset,
        resolved_config=config,
        root=tmp_path,
    ) == [
        "dataset.registry_overlay_sha256 does not match configured "
        "dataset.registry_overlay"
    ]
