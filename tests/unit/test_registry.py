from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from datasets.registry import (
    apply_registry_overlay,
    DatasetSpec,
    DatasetValidationError,
    OFFICIAL_README_DATASETS,
    RegistryValidationError,
    Representation,
    load_registry,
    load_split,
)
from utils.provenance import (
    EXPECTED_LID_BENCHMARKS_SHA,
    sha256_file,
    tree_manifest,
    verify_tree_manifest,
    verify_upstream_source,
    write_tree_manifest,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def tiny_pca_split(tmp_path: Path) -> tuple[Path, DatasetSpec]:
    split_dir = tmp_path / "tiny_pca" / "test"
    split_dir.mkdir(parents=True)
    np.save(split_dir / "dataset.npy", np.arange(24, dtype=np.float32).reshape(3, 2, 4))
    np.save(split_dir / "coefficients.npy", np.arange(15, dtype=np.float64).reshape(3, 5))
    np.save(split_dir / "lid.npy", np.full(3, 2, dtype=np.int64))
    spec = DatasetSpec(
        name="tiny_pca",
        splits=("test",),
        representation="dataset",
        available_representations=("dataset", "coefficients"),
        required_artifacts=("dataset", "coefficients", "lid"),
        expected_samples={"test": 3},
        expected_shapes={"dataset": (2, 4), "coefficients": (5,)},
        expected_lid=2,
    )
    return tmp_path, spec


def test_loads_official_layout_and_selects_representation(
    tiny_pca_split: tuple[Path, DatasetSpec],
) -> None:
    root, spec = tiny_pca_split

    image_split = load_split(root, spec, "test")
    coefficient_split = load_split(
        root,
        spec,
        "test",
        representation=Representation.COEFFICIENTS,
        mmap_mode="r",
    )

    assert image_split.features.shape == (3, 2, 4)
    assert image_split.flat_feature_dim == 8
    assert coefficient_split.features.shape == (3, 5)
    assert coefficient_split.representation is Representation.COEFFICIENTS
    assert coefficient_split.n_samples == 3
    np.testing.assert_array_equal(coefficient_split.lid, [2, 2, 2])


@pytest.mark.parametrize(
    ("artifact", "replacement", "error"),
    [
        ("dataset", np.zeros((2, 2, 4), dtype=np.float32), "samples"),
        (
            "dataset",
            np.array(
                [
                    [[np.nan, 0, 0, 0], [0, 0, 0, 0]],
                    [[0, 0, 0, 0], [0, 0, 0, 0]],
                    [[0, 0, 0, 0], [0, 0, 0, 0]],
                ],
                dtype=np.float32,
            ),
            "NaN or infinity",
        ),
        ("coefficients", np.zeros((3, 1, 5), dtype=np.float32), "rank 2"),
        ("lid", np.array(["2", "2", "2"]), "real numeric dtype"),
    ],
)
def test_rejects_invalid_shapes_finiteness_and_dtypes(
    tiny_pca_split: tuple[Path, DatasetSpec],
    artifact: str,
    replacement: np.ndarray,
    error: str,
) -> None:
    root, spec = tiny_pca_split
    np.save(root / spec.name / "test" / f"{artifact}.npy", replacement)

    with pytest.raises(DatasetValidationError, match=error):
        load_split(root, spec, "test")


def test_generated_spaghetti_override_validates_stored_target_before_correction(
    tmp_path: Path,
) -> None:
    split_dir = tmp_path / "spaghetti" / "test"
    split_dir.mkdir(parents=True)
    np.save(split_dir / "dataset.npy", np.ones((4, 2), dtype=np.float32))
    np.save(split_dir / "lid.npy", np.full(4, 20, dtype=np.int64))
    spec = DatasetSpec(
        name="spaghetti",
        splits=("test",),
        required_artifacts=("dataset", "lid"),
        stored_lid=20,
        lid_override=1,
        expected_lid=1,
    )

    loaded = load_split(tmp_path, spec, "test")

    np.testing.assert_array_equal(loaded.lid, np.ones(4, dtype=np.int64))
    assert loaded.applied_overrides["lid"]["stored"] == 20
    # np.load proves that the source artifact was not patched in place.
    np.testing.assert_array_equal(
        np.load(split_dir / "lid.npy", allow_pickle=False),
        np.full(4, 20, dtype=np.int64),
    )

    np.save(split_dir / "lid.npy", np.full(4, 19, dtype=np.int64))
    with pytest.raises(DatasetValidationError, match="refusing to apply any override"):
        load_split(tmp_path, spec, "test")


def test_split_aware_lid_policy_validates_each_raw_split_before_correction(
    tmp_path: Path,
) -> None:
    raw_lid_by_split = {"train": 20, "val": 20, "test": 1}
    spec = DatasetSpec(
        name="spaghetti",
        required_artifacts=("dataset", "lid"),
        expected_samples={split: 3 for split in raw_lid_by_split},
        stored_lid_by_split=raw_lid_by_split,
        lid_override=1,
        expected_lid=1,
    )
    for split, raw_lid in raw_lid_by_split.items():
        split_dir = tmp_path / spec.name / split
        split_dir.mkdir(parents=True)
        np.save(split_dir / "dataset.npy", np.ones((3, 2), dtype=np.float32))
        np.save(split_dir / "lid.npy", np.full(3, raw_lid, dtype=np.int64))

    for split, raw_lid in raw_lid_by_split.items():
        loaded = load_split(tmp_path, spec, split)
        np.testing.assert_array_equal(loaded.lid, np.ones(3, dtype=np.int64))
        assert loaded.applied_overrides["lid"] == {
            "kind": "constant_after_stored_value_validation",
            "stored": float(raw_lid),
            "effective": 1.0,
        }
        np.testing.assert_array_equal(
            np.load(
                tmp_path / spec.name / split / "lid.npy",
                allow_pickle=False,
            ),
            np.full(3, raw_lid, dtype=np.int64),
        )

    np.save(
        tmp_path / spec.name / "val" / "lid.npy",
        np.ones(3, dtype=np.int64),
    )
    with pytest.raises(
        DatasetValidationError,
        match=r"stored lid for spaghetti/val.*20\.0.*refusing",
    ):
        load_split(tmp_path, spec, "val")


def test_split_aware_lid_policy_requires_complete_unambiguous_split_coverage() -> None:
    with pytest.raises(RegistryValidationError, match="must cover every declared split"):
        DatasetSpec(
            name="spaghetti",
            required_artifacts=("dataset", "lid"),
            stored_lid_by_split={"train": 20, "val": 20},
            lid_override=1,
            expected_lid=1,
        )

    with pytest.raises(RegistryValidationError, match="cannot set both"):
        DatasetSpec(
            name="spaghetti",
            required_artifacts=("dataset", "lid"),
            stored_lid=20,
            stored_lid_by_split={"train": 20, "val": 20, "test": 1},
            lid_override=1,
            expected_lid=1,
        )


def test_registry_covers_every_folder_in_pinned_readme() -> None:
    registry = load_registry(
        REPOSITORY_ROOT
        / "configs"
        / "datasets"
        / "registry"
        / "paper_benchmarks.yaml"
    )

    assert registry.coverage().complete
    assert tuple(spec.name for spec in registry.official_specs) == OFFICIAL_README_DATASETS
    assert len(registry.official_specs) == 27
    assert not registry["e5_downscaled_fmnist"].official

    spaghetti = registry["e8_spaghetti_pca"]
    assert spaghetti.expected_lid == 1
    assert spaghetti.stored_lid is None
    assert dict(spaghetti.stored_lid_by_split) == {
        "train": 20,
        "val": 20,
        "test": 1,
    }
    assert spaghetti.lid_override == 1
    assert spaghetti.ignored_files == ("lid-OLD_WRONG.npy",)
    assert registry["e6_exp_pca"].ignored_files == ("lid-OLD_WRONG.npy",)
    assert registry["e7_crescent_moon_radius3.0"].ignored_files == (
        "lid-OLD_WRONG.npy",
    )
    assert set(spaghetti.available_representations) == {
        Representation.DATASET,
        Representation.COEFFICIENTS,
    }

    assert (
        registry["e5_padded_fmnist_adddim4"].transformation.expected_lid_delta
        == 4
    )
    assert registry["e5_upscaled_fmnist"].transformation.expected_lid_delta == 0


def test_canonical_spaghetti_and_generated_overlay_are_distinct(tmp_path: Path) -> None:
    registry = load_registry(
        REPOSITORY_ROOT
        / "configs"
        / "datasets"
        / "registry"
        / "paper_benchmarks.yaml"
    )
    canonical = registry["e8_spaghetti_pca"]
    # Use a tiny copy of the canonical contract while preserving its LID policy.
    canonical = DatasetSpec(
        name=canonical.name,
        splits=("test",),
        required_artifacts=("dataset", "lid"),
        ignored_files=canonical.ignored_files,
        expected_lid=canonical.expected_lid,
        stored_lid_by_split={"test": canonical.stored_lid_by_split["test"]},
        lid_override=canonical.lid_override,
    )
    split_dir = tmp_path / canonical.name / "test"
    split_dir.mkdir(parents=True)
    np.save(split_dir / "dataset.npy", np.ones((2, 3), dtype=np.float32))
    np.save(split_dir / "lid.npy", np.ones(2, dtype=np.int64))
    np.save(split_dir / "lid-OLD_WRONG.npy", np.full(2, 20, dtype=np.int64))

    loaded = load_split(tmp_path, canonical, "test")
    np.testing.assert_array_equal(loaded.lid, [1, 1])
    assert loaded.applied_overrides["lid"] == {
        "kind": "constant_after_stored_value_validation",
        "stored": 1.0,
        "effective": 1.0,
    }

    generated_registry = apply_registry_overlay(
        registry,
        REPOSITORY_ROOT
        / "configs"
        / "datasets"
        / "registry"
        / "generated_fallback.yaml",
    )
    generated = generated_registry["e8_spaghetti_pca"]
    assert generated.stored_lid is None
    assert dict(generated.stored_lid_by_split) == {
        "train": 20,
        "val": 20,
        "test": 20,
    }
    assert generated.lid_override == generated.expected_lid == 1


def test_sha256_tree_manifest_detects_tampering(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    artifact = data_dir / "dataset.npy"
    np.save(artifact, np.arange(8, dtype=np.float32))

    first = tree_manifest(data_dir)
    second = tree_manifest(data_dir)
    assert first == second
    assert first["sha256"] == first["tree_sha256"]
    assert first["files"]["dataset.npy"]["sha256"] == sha256_file(artifact)

    output = data_dir / "manifest.json"
    written = write_tree_manifest(data_dir, output)
    assert written["excluded_paths"] == ["manifest.json"]
    assert verify_tree_manifest(data_dir, output) == []

    np.save(artifact, np.arange(9, dtype=np.float32))
    errors = verify_tree_manifest(data_dir, output)
    assert any("dataset.npy" in error for error in errors)


def test_pinned_upstream_import_preserves_original_files() -> None:
    checkout = REPOSITORY_ROOT / "lid_benchmarks"
    assert EXPECTED_LID_BENCHMARKS_SHA == "2dcb8e41015f53413ff1ddd049bb006c81a5df52"
    assert verify_upstream_source(checkout) == EXPECTED_LID_BENCHMARKS_SHA
