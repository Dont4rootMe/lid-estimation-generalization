from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from datasets.registry import load_registry
from experiments.global_inventory import (
    CANONICAL_INVENTORY_CONFIG,
    GENERATED_E3_E4_INVENTORY_CONFIG,
    InventoryError,
    load_canonical_inventory,
    load_generated_e3_e4_inventory,
    load_global_inventory,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_global_suite_yaml_files_are_hydra_groups() -> None:
    with initialize_config_dir(
        config_dir=str(REPOSITORY_ROOT / "configs"), version_base=None
    ):
        canonical = compose(
            config_name="config", overrides=["+global_suite=canonical_exact"]
        )
        generated = compose(
            config_name="config", overrides=["+global_suite=generated_e3_e4"]
        )

    canonical_raw = OmegaConf.to_container(canonical.global_suite, resolve=True)
    generated_raw = OmegaConf.to_container(generated.global_suite, resolve=True)
    assert canonical_raw["expected"]["cell_count"] == 35
    assert generated_raw["expected"]["cell_count"] == 4


def test_canonical_inventory_covers_every_exact_representation_cell() -> None:
    inventory = load_canonical_inventory(REPOSITORY_ROOT)

    assert inventory.source.kind == "exact_archive"
    assert inventory.source.exact_archive is True
    assert inventory.expected_suite_count == 8
    assert inventory.expected_available_suite_count == 6
    assert inventory.expected_dataset_count == len(inventory.dataset_names) == 28
    assert inventory.expected_cell_count == len(inventory.cells) == 35
    assert len(inventory.known_lid_cells) == 15
    assert len(inventory.target_free_cells) == 20
    assert tuple(suite.suite_id for suite in inventory.suites) == tuple(
        f"e{index}" for index in range(1, 9)
    )

    per_suite = {
        suite.suite_id: (len(suite.dataset_names), len(suite.cells))
        for suite in inventory.suites
    }
    assert per_suite == {
        "e1": (14, 15),
        "e2": (2, 3),
        "e3": (0, 0),
        "e4": (0, 0),
        "e5": (7, 7),
        "e6": (1, 2),
        "e7": (1, 2),
        "e8": (3, 6),
    }
    assert inventory.suite("e3").availability == "absent_from_source"
    assert inventory.suite("e4").availability == "absent_from_source"

    registry = load_registry(
        REPOSITORY_ROOT / "configs/datasets/registry/paper_benchmarks.yaml"
    )
    expected_cells = {
        (name, representation.value)
        for name, spec in registry.items()
        for representation in spec.available_representations
    }
    assert {cell.key for cell in inventory.cells} == expected_cells


def test_canonical_target_policies_separate_supervised_and_target_free() -> None:
    inventory = load_canonical_inventory(REPOSITORY_ROOT)

    assert {cell.selection_protocol for cell in inventory.known_lid_cells} == {
        "supervised_train_mae"
    }
    assert {cell.selection_protocol for cell in inventory.target_free_cells} == {
        "target_free_train_stability"
    }
    assert {cell.target_policy for cell in inventory.target_free_cells} == {
        "sample_size",
        "paired_delta",
    }

    adddim4 = next(
        cell
        for cell in inventory.cells
        if cell.key == ("e5_padded_fmnist_adddim4", "dataset")
    )
    assert adddim4.reference_dataset == "e5_downscaled_fmnist"
    assert adddim4.expected_lid_delta == 4.0
    sampled = next(
        cell
        for cell in inventory.cells
        if cell.key == ("e1_sampled_fmnist_step13", "dataset")
    )
    assert sampled.reference_dataset == "e1_sampled_fmnist_step1"
    assert sampled.expected_lid_delta == 0.0


def test_generated_e3_e4_is_a_separate_non_exact_inventory() -> None:
    canonical = load_canonical_inventory(REPOSITORY_ROOT)
    generated = load_generated_e3_e4_inventory(REPOSITORY_ROOT)

    assert generated.source.kind == "generated_at_pinned_revision"
    assert generated.source.exact_archive is False
    assert generated.source.archive_sha256 is None
    assert "not-canonical-exact" in generated.source.provenance_label
    assert tuple(suite.suite_id for suite in generated.suites) == ("e3", "e4")
    assert all(
        suite.availability == "generated_extension" for suite in generated.suites
    )
    assert generated.expected_dataset_count == len(generated.dataset_names) == 2
    assert generated.expected_cell_count == len(generated.cells) == 4
    assert {cell.key for cell in generated.cells} == {
        ("e3_gaussian_pca", "dataset"),
        ("e3_gaussian_pca", "coefficients"),
        ("e4_sphere_pca_radius1", "dataset"),
        ("e4_sphere_pca_radius1", "coefficients"),
    }
    assert set(canonical.dataset_names).isdisjoint(generated.dataset_names)

    registry = load_registry(
        REPOSITORY_ROOT / "configs/datasets/registry/generated_e3_e4_extension.yaml"
    )
    assert registry["e3_gaussian_pca"].expected_lid == 20
    assert registry["e4_sphere_pca_radius1"].expected_lid == 19
    assert not any(spec.official for spec in registry.values())


def _mutated_inventory(tmp_path: Path, relative_path: Path) -> tuple[dict, Path]:
    source_path = REPOSITORY_ROOT / relative_path
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    output_path = tmp_path / source_path.name
    return raw, output_path


def test_generated_inventory_cannot_be_mislabeled_as_exact(tmp_path: Path) -> None:
    raw, path = _mutated_inventory(tmp_path, GENERATED_E3_E4_INVENTORY_CONFIG)
    raw["source"]["exact_archive"] = True
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(InventoryError, match="disagrees with source.kind"):
        load_global_inventory(path, project_root=REPOSITORY_ROOT)


def test_generated_inventory_cannot_claim_canonical_provenance_label(
    tmp_path: Path,
) -> None:
    raw, path = _mutated_inventory(tmp_path, GENERATED_E3_E4_INVENTORY_CONFIG)
    raw["source"]["provenance_label"] = "canonical-exact-paper-archive"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(InventoryError, match="generated-extension-not-canonical-exact"):
        load_global_inventory(path, project_root=REPOSITORY_ROOT)


def test_target_free_reference_cannot_point_to_known_lid_cell(tmp_path: Path) -> None:
    raw, path = _mutated_inventory(tmp_path, CANONICAL_INVENTORY_CONFIG)
    e1 = next(suite for suite in raw["suites"] if suite["id"] == "e1")
    root = next(
        dataset
        for dataset in e1["datasets"]
        if dataset["name"] == "e1_sampled_fmnist_step1"
    )
    root["reference_dataset"] = "e1_spiral_pca"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(InventoryError, match="target-free root"):
        load_global_inventory(path, project_root=REPOSITORY_ROOT)


def test_approved_registry_content_is_sha_pinned(tmp_path: Path) -> None:
    required = (
        CANONICAL_INVENTORY_CONFIG,
        Path("configs/datasets/lid_benchmarks.yaml"),
        Path("configs/datasets/registry/paper_benchmarks.yaml"),
    )
    for relative in required:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPOSITORY_ROOT / relative, destination)

    registry_path = tmp_path / "configs/datasets/registry/paper_benchmarks.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    registry["benchmark_id"] = "scientifically-different-policy-with-same-cells"
    registry_path.write_text(
        yaml.safe_dump(registry, sort_keys=False), encoding="utf-8"
    )

    with pytest.raises(InventoryError, match="dataset registry content"):
        load_global_inventory(CANONICAL_INVENTORY_CONFIG, project_root=tmp_path)


@pytest.mark.parametrize(
    ("relative_path", "field", "replacement", "message"),
    [
        (
            CANONICAL_INVENTORY_CONFIG,
            "upstream_revision",
            "0" * 40,
            "pinned LID-Benchmarks revision",
        ),
        (
            GENERATED_E3_E4_INVENTORY_CONFIG,
            "upstream_revision",
            "0" * 40,
            "pinned LID-Benchmarks revision",
        ),
        (
            CANONICAL_INVENTORY_CONFIG,
            "archive_sha256",
            "1" * 64,
            "canonical exact archive",
        ),
    ],
)
def test_inventory_provenance_hashes_are_pinned_to_authoritative_constants(
    tmp_path: Path,
    relative_path: Path,
    field: str,
    replacement: str,
    message: str,
) -> None:
    raw, path = _mutated_inventory(tmp_path, relative_path)
    raw["source"][field] = replacement
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(InventoryError, match=message):
        load_global_inventory(path, project_root=REPOSITORY_ROOT)


def test_inventory_rejects_boolean_schema_version(tmp_path: Path) -> None:
    raw, path = _mutated_inventory(tmp_path, CANONICAL_INVENTORY_CONFIG)
    raw["schema_version"] = True
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(InventoryError, match="schema_version"):
        load_global_inventory(path, project_root=REPOSITORY_ROOT)


def test_inventory_rejects_omitted_representation_even_if_counts_are_edited(
    tmp_path: Path,
) -> None:
    raw, path = _mutated_inventory(tmp_path, CANONICAL_INVENTORY_CONFIG)
    e8 = next(suite for suite in raw["suites"] if suite["id"] == "e8")
    gaussian = next(
        dataset for dataset in e8["datasets"] if dataset["name"] == "e8_gaussian4_pca"
    )
    gaussian["representations"] = ["dataset"]
    e8["expected"]["cell_count"] -= 1
    e8["expected"]["known_lid_cell_count"] -= 1
    raw["expected"]["cell_count"] -= 1
    raw["expected"]["known_lid_cell_count"] -= 1
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(InventoryError, match="must exactly match registry order"):
        load_global_inventory(path, project_root=REPOSITORY_ROOT)
