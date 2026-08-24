from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from models.learned import (
    ArtifactRegistryIdentity,
    array_sha256,
    bundle_paths,
    evaluate_field_grid,
    load_artifact_registry,
    verify_model_artifacts,
)
from utils.provenance import sha256_file


CHECKPOINT_SHA = "a" * 64
CONFIG_SHA = "b" * 64
DATASET_SHA = "c" * 64
TRAINING_DATASET_SHA = "d" * 64
PREPROCESSING_SHA = "e" * 64


def _model() -> dict[str, object]:
    return {
        "name": "diffusion-reference",
        "family": "gaussian_diffusion",
        "seed": 5,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "training_config_sha256": CONFIG_SHA,
        "readouts": ["diffusion_flipd_full"],
        "trace": {"backend": "exact", "probes": 0, "seed": 19},
        "min_finite_fraction": 1.0,
    }


def _write_grid(root: Path, validation: np.ndarray, test: np.ndarray) -> None:
    for scale_index, sigma in enumerate((0.1, 0.2)):
        for split, query in (("validation", validation), ("test", test)):
            npz_path, metadata_path = bundle_paths(
                root,
                model_name="diffusion-reference",
                model_seed=5,
                dataset_name="fixture",
                representation="coordinates",
                scale_index=scale_index,
                split=split,
            )
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                npz_path,
                score=np.zeros_like(query),
                score_divergence=np.full(query.shape[0], -100.0),
            )
            metadata = {
                "scalars": {"sigma": sigma, "ambient_dim": query.shape[1]},
                "metadata": {
                    "schema_version": 1,
                    "model_name": "diffusion-reference",
                    "model_family": "gaussian_diffusion",
                    "model_seed": 5,
                    "checkpoint_sha256": CHECKPOINT_SHA,
                    "training_config_sha256": CONFIG_SHA,
                    "dataset_name": "fixture",
                    "training_dataset_sha256": TRAINING_DATASET_SHA,
                    "dataset_sha256": DATASET_SHA,
                    "representation": "coordinates",
                    "split": split,
                    "query_sha256": array_sha256(query),
                    "preprocessing_sha256": PREPROCESSING_SHA,
                    "model_space_query_sha256": array_sha256(query),
                    "n_samples": query.shape[0],
                    "scale_index": scale_index,
                    "physical_scale": sigma,
                    "readout_ids": ["diffusion_flipd_full"],
                    "trace": {"backend": "exact", "probes": 0, "seed": 19},
                },
            }
            metadata_path.write_text(
                json.dumps(metadata, sort_keys=True), encoding="utf-8"
            )


def test_learned_grid_is_bound_to_exact_rows_and_model_inputs(tmp_path: Path) -> None:
    validation = np.arange(6, dtype=np.float64).reshape(2, 3)
    test = np.arange(9, dtype=np.float64).reshape(3, 3)
    _write_grid(tmp_path, validation, test)

    result = evaluate_field_grid(
        bundle_root=tmp_path,
        model=_model(),
        dataset_name="fixture",
        training_dataset_sha256=TRAINING_DATASET_SHA,
        dataset_sha256=DATASET_SHA,
        representation="coordinates",
        validation=validation,
        test=test,
        preprocessing_sha256=PREPROCESSING_SHA,
        model_space_validation=validation,
        model_space_test=test,
        scales=(0.1, 0.2),
    )
    np.testing.assert_allclose(
        result.validation_curves["diffusion_flipd_full"],
        [[2.0, -1.0], [2.0, -1.0]],
    )
    np.testing.assert_allclose(
        result.test_curves["diffusion_flipd_full"],
        [[2.0, -1.0], [2.0, -1.0], [2.0, -1.0]],
    )
    assert len(result.input_files) == 8
    assert len(result.input_sha256) == 64

    with pytest.raises(ValueError, match="query_sha256.*mismatch"):
        evaluate_field_grid(
            bundle_root=tmp_path,
            model=_model(),
            dataset_name="fixture",
            training_dataset_sha256=TRAINING_DATASET_SHA,
            dataset_sha256=DATASET_SHA,
            representation="coordinates",
            validation=validation[::-1],
            test=test,
            preprocessing_sha256=PREPROCESSING_SHA,
            model_space_validation=validation[::-1],
            model_space_test=test,
            scales=(0.1, 0.2),
        )


def test_model_family_cannot_silently_change_its_readouts(tmp_path: Path) -> None:
    model = _model()
    model["readouts"] = ["fm_affine_response"]
    with pytest.raises(ValueError, match="requires readouts"):
        evaluate_field_grid(
            bundle_root=tmp_path,
            model=model,
            dataset_name="fixture",
            training_dataset_sha256=TRAINING_DATASET_SHA,
            dataset_sha256=DATASET_SHA,
            representation="coordinates",
            validation=np.zeros((1, 3)),
            test=np.zeros((1, 3)),
            preprocessing_sha256=PREPROCESSING_SHA,
            model_space_validation=np.zeros((1, 3)),
            model_space_test=np.zeros((1, 3)),
            scales=(0.1,),
        )


def test_same_selected_queries_reject_changed_training_source_or_transform(
    tmp_path: Path,
) -> None:
    validation = np.arange(6, dtype=np.float64).reshape(2, 3)
    test = np.arange(9, dtype=np.float64).reshape(3, 3)
    _write_grid(tmp_path, validation, test)

    common = {
        "bundle_root": tmp_path,
        "model": _model(),
        "dataset_name": "fixture",
        "dataset_sha256": DATASET_SHA,
        "representation": "coordinates",
        "validation": validation,
        "test": test,
        "model_space_validation": validation,
        "model_space_test": test,
        "scales": (0.1, 0.2),
    }
    with pytest.raises(ValueError, match="training_dataset_sha256.*mismatch"):
        evaluate_field_grid(
            **common,
            training_dataset_sha256="f" * 64,
            preprocessing_sha256=PREPROCESSING_SHA,
        )
    with pytest.raises(ValueError, match="preprocessing_sha256.*mismatch"):
        evaluate_field_grid(
            **common,
            training_dataset_sha256=TRAINING_DATASET_SHA,
            preprocessing_sha256="f" * 64,
        )


def test_model_space_query_hash_binds_actual_transformed_rows(tmp_path: Path) -> None:
    validation = np.arange(6, dtype=np.float64).reshape(2, 3)
    test = np.arange(9, dtype=np.float64).reshape(3, 3)
    _write_grid(tmp_path, validation, test)

    with pytest.raises(ValueError, match="model_space_query_sha256.*mismatch"):
        evaluate_field_grid(
            bundle_root=tmp_path,
            model=_model(),
            dataset_name="fixture",
            training_dataset_sha256=TRAINING_DATASET_SHA,
            dataset_sha256=DATASET_SHA,
            representation="coordinates",
            validation=validation,
            test=test,
            preprocessing_sha256=PREPROCESSING_SHA,
            model_space_validation=validation + 1.0,
            model_space_test=test,
            scales=(0.1, 0.2),
        )


def _write_artifact_registry(
    root: Path,
    *,
    checkpoint_path: str = "model.ckpt",
    training_config_path: str = "training.yaml",
    create_checkpoint: bool = True,
) -> tuple[dict[str, object], ArtifactRegistryIdentity]:
    checkpoint = root / checkpoint_path
    training_config = root / training_config_path
    if create_checkpoint:
        checkpoint.write_bytes(b"checkpoint fixture\n")
    training_payload = {
        "provenance": {
            "schema_version": 1,
            "model_name": "diffusion-reference",
            "model_family": "gaussian_diffusion",
            "model_seed": 5,
            "dataset_name": "fixture",
            "representation": "coordinates",
            "training_dataset_sha256": TRAINING_DATASET_SHA,
            "preprocessing_sha256": PREPROCESSING_SHA,
        },
        "optimizer": {"name": "adam"},
    }
    training_config.write_text(
        json.dumps(training_payload, sort_keys=True), encoding="utf-8"
    )
    checkpoint_sha = sha256_file(checkpoint) if checkpoint.is_file() else "0" * 64
    registry_payload = {
        "schema_version": 1,
        "artifacts": {
            "fixture/coordinates": {
                "checkpoint_path": checkpoint_path,
                "checkpoint_sha256": checkpoint_sha,
                "training_config_path": training_config_path,
                "training_config_sha256": sha256_file(training_config),
                "training_dataset_sha256": TRAINING_DATASET_SHA,
                "preprocessing_sha256": PREPROCESSING_SHA,
            }
        },
    }
    registry_path = root / "artifacts.yaml"
    registry_path.write_text(
        json.dumps(registry_payload, sort_keys=True), encoding="utf-8"
    )
    model: dict[str, object] = {
        **_model(),
        "artifact_registry": str(registry_path),
        "artifact_registry_sha256": sha256_file(registry_path),
    }
    registry = load_artifact_registry(root=root, model=model)
    return model, registry


def _verify_fixture_artifacts(
    model: dict[str, object], registry: ArtifactRegistryIdentity
):
    return verify_model_artifacts(
        registry=registry,
        model=model,
        dataset_name="fixture",
        representation="coordinates",
        training_dataset_sha256=TRAINING_DATASET_SHA,
        preprocessing_sha256=PREPROCESSING_SHA,
    )


def test_per_cell_model_artifacts_are_hashed_and_bound(tmp_path: Path) -> None:
    model, registry = _write_artifact_registry(tmp_path)
    identity = _verify_fixture_artifacts(model, registry)
    assert set(identity.input_files) == {
        "model/artifact_registry.yaml",
        "model/checkpoint",
        "model/training_config.yaml",
    }
    assert len(identity.input_sha256) == 64

    (tmp_path / "model.ckpt").write_bytes(b"tampered\n")
    with pytest.raises(ValueError, match="does not match checkpoint_path"):
        _verify_fixture_artifacts(model, registry)


def test_per_cell_artifacts_require_files_yaml_and_matching_provenance(
    tmp_path: Path,
) -> None:
    missing_root = tmp_path / "missing"
    missing_root.mkdir()
    model, registry = _write_artifact_registry(
        missing_root, create_checkpoint=False
    )
    with pytest.raises(FileNotFoundError, match="checkpoint_path"):
        _verify_fixture_artifacts(model, registry)

    suffix_root = tmp_path / "suffix"
    suffix_root.mkdir()
    model, registry = _write_artifact_registry(
        suffix_root, training_config_path="training.yml"
    )
    with pytest.raises(ValueError, match="must have the .yaml suffix"):
        _verify_fixture_artifacts(model, registry)

    provenance_root = tmp_path / "provenance"
    provenance_root.mkdir()
    model, registry = _write_artifact_registry(provenance_root)
    with pytest.raises(ValueError, match="training_dataset_sha256 mismatch"):
        verify_model_artifacts(
            registry=registry,
            model=model,
            dataset_name="fixture",
            representation="coordinates",
            training_dataset_sha256="f" * 64,
            preprocessing_sha256=PREPROCESSING_SHA,
        )


def test_artifact_registry_identity_is_stable_after_relocation(
    tmp_path: Path,
) -> None:
    identities = []
    for directory_name in ("machine-a", "machine-b"):
        directory = tmp_path / directory_name
        directory.mkdir()
        model, registry = _write_artifact_registry(directory)
        identities.append(_verify_fixture_artifacts(model, registry))

    assert identities[0].input_sha256 == identities[1].input_sha256
    assert (
        identities[0].input_files["model/checkpoint"]["path"]
        != identities[1].input_files["model/checkpoint"]["path"]
    )
