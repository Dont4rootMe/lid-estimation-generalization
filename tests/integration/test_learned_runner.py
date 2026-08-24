from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pytest

from models.learned import array_sha256, bundle_paths
from experiments.runner import (
    ExperimentConfigError,
    _selected_dataset_sha256,
    compose_experiment_config,
    normalize_hydra_config,
    prepare_datasets,
    run_composed_experiment,
    validate_matrix,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_BYTES = b"learned checkpoint fixture\n"
CHECKPOINT_SHA = hashlib.sha256(CHECKPOINT_BYTES).hexdigest()
SCALES = (0.1, 0.2, 0.4)


def _overrides(bundle_root: Path) -> list[str]:
    artifact_root = bundle_root.parent / "model-artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    registry_path = artifact_root / "artifacts.yaml"
    checkpoint_path = artifact_root / "model.ckpt"
    training_config_path = artifact_root / "training.yaml"
    checkpoint_path.write_bytes(CHECKPOINT_BYTES)
    overrides = [
        "models=diffusion",
        "experiment=learned_smoke",
        f"models.bundle_root={bundle_root}",
        f"models.artifact_registry={registry_path}",
        f"models.artifact_registry_sha256={'0' * 64}",
        "models.seed=7",
        "models.trace.seed=29",
        "datasets.n_train=32",
        "datasets.n_validation=4",
        "datasets.n_test=5",
        "runtime.limits.reference=16",
        "runtime.limits.validation=4",
        "runtime.limits.test=5",
    ]
    hydra_config = compose_experiment_config(overrides, root=REPOSITORY_ROOT)
    config = normalize_hydra_config(hydra_config)
    cell = prepare_datasets(config, REPOSITORY_ROOT)[0]
    training_payload = {
        "provenance": {
            "schema_version": 1,
            "model_name": "diffusion-reference",
            "model_family": "gaussian_diffusion",
            "model_seed": 7,
            "dataset_name": "synthetic_flat",
            "representation": "coordinates",
            "training_dataset_sha256": cell.training_dataset_sha256,
            "preprocessing_sha256": cell.preprocessing_sha256,
        },
        "optimizer": {"name": "adam"},
    }
    training_config_path.write_text(
        json.dumps(training_payload, sort_keys=True), encoding="utf-8"
    )
    registry_payload = {
        "schema_version": 1,
        "artifacts": {
            "synthetic_flat/coordinates": {
                "checkpoint_path": "model.ckpt",
                "checkpoint_sha256": CHECKPOINT_SHA,
                "training_config_path": "training.yaml",
                "training_config_sha256": hashlib.sha256(
                    training_config_path.read_bytes()
                ).hexdigest(),
                "training_dataset_sha256": cell.training_dataset_sha256,
                "preprocessing_sha256": cell.preprocessing_sha256,
            }
        },
    }
    registry_path.write_text(
        json.dumps(registry_payload, sort_keys=True), encoding="utf-8"
    )
    registry_sha = hashlib.sha256(registry_path.read_bytes()).hexdigest()
    overrides[4] = f"models.artifact_registry_sha256={registry_sha}"
    return overrides


def _write_diffusion_grid(bundle_root: Path, overrides: list[str]) -> None:
    hydra_config = compose_experiment_config(overrides, root=REPOSITORY_ROOT)
    config = normalize_hydra_config(hydra_config)
    prepared = prepare_datasets(config, REPOSITORY_ROOT)
    assert len(prepared) == 1
    cell = prepared[0]
    registry_path = bundle_root.parent / "model-artifacts" / "artifacts.yaml"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    artifact = registry["artifacts"]["synthetic_flat/coordinates"]
    dataset_sha256 = _selected_dataset_sha256(cell)
    for scale_index, sigma in enumerate(SCALES):
        for split, raw_query, model_query in (
            ("validation", cell.raw_validation, cell.validation),
            ("test", cell.raw_test, cell.test),
        ):
            npz_path, metadata_path = bundle_paths(
                bundle_root,
                model_name="diffusion-reference",
                model_seed=7,
                dataset_name="synthetic_flat",
                representation="coordinates",
                scale_index=scale_index,
                split=split,
            )
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            desired_lid = float(scale_index + 1)
            divergence = np.full(
                model_query.shape[0],
                (desired_lid - model_query.shape[1]) / sigma**2,
                dtype=np.float64,
            )
            np.savez(
                npz_path,
                score=np.zeros_like(model_query),
                score_divergence=divergence,
            )
            payload = {
                "scalars": {
                    "sigma": sigma,
                    "ambient_dim": int(model_query.shape[1]),
                },
                "metadata": {
                    "schema_version": 1,
                    "model_name": "diffusion-reference",
                    "model_family": "gaussian_diffusion",
                    "model_seed": 7,
                    "checkpoint_sha256": artifact["checkpoint_sha256"],
                    "training_config_sha256": artifact["training_config_sha256"],
                    "dataset_name": "synthetic_flat",
                    "training_dataset_sha256": cell.training_dataset_sha256,
                    "dataset_sha256": dataset_sha256,
                    "representation": "coordinates",
                    "split": split,
                    "query_sha256": array_sha256(raw_query),
                    "preprocessing_sha256": cell.preprocessing_sha256,
                    "model_space_query_sha256": array_sha256(model_query),
                    "n_samples": int(model_query.shape[0]),
                    "scale_index": scale_index,
                    "physical_scale": sigma,
                    "readout_ids": ["diffusion_flipd_full"],
                    "trace": {"backend": "exact", "probes": 0, "seed": 29},
                },
            }
            metadata_path.write_text(
                json.dumps(payload, sort_keys=True), encoding="utf-8"
            )


def test_learned_backend_writes_validated_curves_identity_and_aggregate(
    tmp_path: Path,
) -> None:
    bundle_root = tmp_path / "bundles"
    output_root = tmp_path / "outputs"
    overrides = _overrides(bundle_root)
    _write_diffusion_grid(bundle_root, overrides)

    matrix_dir = run_composed_experiment(
        overrides, root=REPOSITORY_ROOT, output_root=output_root
    )
    assert validate_matrix(matrix_dir) == []
    matrix = json.loads((matrix_dir / "matrix.json").read_text(encoding="utf-8"))
    assert matrix["backend"] == "learned_field_bundle"
    assert matrix["evidence_level"] == "learned_model"
    assert matrix["complete"] is True

    manifest_path = matrix_dir / matrix["cells"][0]["manifest"]
    cell_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = manifest["dataset"]
    learned_input = dataset["learned_input"]
    assert dataset["sha256"] != dataset["selected_dataset_sha256"]
    assert learned_input["checkpoint_sha256"] == CHECKPOINT_SHA
    assert learned_input["training_config_sha256"] == learned_input["input_files"][
        "model/training_config.yaml"
    ]["sha256"]
    assert learned_input["trace"] == {"backend": "exact", "probes": 0, "seed": 29}
    assert len(learned_input["input_files"]) == 15
    assert len(learned_input["bundle_input_sha256"]) == 64
    assert len(learned_input["model_artifact_input_sha256"]) == 64
    assert len(learned_input["input_sha256"]) == 64
    assert manifest["upstream_sha"] == "not_applicable"

    validation_curve = np.load(
        cell_dir / "validation_curve__diffusion_flipd_full.npy",
        allow_pickle=False,
    )
    test_curve = np.load(
        cell_dir / "test_curve__diffusion_flipd_full.npy", allow_pickle=False
    )
    prediction = np.load(
        cell_dir / "prediction__diffusion_flipd_full.npy", allow_pickle=False
    )
    np.testing.assert_allclose(validation_curve, [[1.0, 2.0, 3.0]] * 4)
    np.testing.assert_allclose(test_curve, [[1.0, 2.0, 3.0]] * 5)
    np.testing.assert_allclose(prediction, np.full(5, 2.0))
    assert (cell_dir / "validation_target.npy").is_file()
    assert (cell_dir / "test_target.npy").is_file()

    summary = json.loads((cell_dir / "summary.json").read_text(encoding="utf-8"))
    readout = summary["readouts"]["diffusion_flipd_full"]
    assert readout["selection_uses_lid_targets"] is False
    assert readout["scale_selection"]["uses_ground_truth"] is False
    assert readout["scale_selection"]["selected_index"] == 1
    aggregate = json.loads(
        (matrix_dir / "aggregate.json").read_text(encoding="utf-8")
    )
    assert aggregate["matrix"]["backend"] == "learned_field_bundle"
    assert aggregate["coverage"]["complete"] is True
    assert aggregate["coverage"]["known_lid_records"] == 1


def test_learned_input_hash_changes_the_cell_run_identity(tmp_path: Path) -> None:
    bundle_root = tmp_path / "bundles"
    output_root = tmp_path / "outputs"
    overrides = _overrides(bundle_root)
    _write_diffusion_grid(bundle_root, overrides)
    matrix_dir = run_composed_experiment(
        overrides, root=REPOSITORY_ROOT, output_root=output_root
    )
    first_matrix = json.loads(
        (matrix_dir / "matrix.json").read_text(encoding="utf-8")
    )
    first_cell = first_matrix["cells"][0]["cell"]

    npz_path, _ = bundle_paths(
        bundle_root,
        model_name="diffusion-reference",
        model_seed=7,
        dataset_name="synthetic_flat",
        representation="coordinates",
        scale_index=2,
        split="test",
    )
    with np.load(npz_path, allow_pickle=False) as archive:
        score = np.asarray(archive["score"])
        divergence = np.asarray(archive["score_divergence"]).copy()
    divergence[0] += 1.0
    np.savez(npz_path, score=score, score_divergence=divergence)

    repeated = run_composed_experiment(
        overrides, root=REPOSITORY_ROOT, output_root=output_root
    )
    second_matrix = json.loads(
        (repeated / "matrix.json").read_text(encoding="utf-8")
    )
    assert repeated != matrix_dir
    assert second_matrix["cells"][0]["cell"] != first_cell
    assert validate_matrix(repeated) == []


def test_missing_learned_bundles_fail_before_output(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    with pytest.raises(FileNotFoundError, match="missing learned field bundle"):
        run_composed_experiment(
            _overrides(tmp_path / "missing-bundles"),
            root=REPOSITORY_ROOT,
            output_root=output_root,
        )
    assert not output_root.exists()


def test_checkpoint_mismatch_fails_before_output(tmp_path: Path) -> None:
    bundle_root = tmp_path / "bundles"
    output_root = tmp_path / "outputs"
    overrides = _overrides(bundle_root)
    checkpoint_path = tmp_path / "model-artifacts" / "model.ckpt"
    checkpoint_path.write_bytes(b"tampered checkpoint\n")

    with pytest.raises(ValueError, match="does not match checkpoint_path"):
        run_composed_experiment(
            overrides,
            root=REPOSITORY_ROOT,
            output_root=output_root,
        )
    assert not output_root.exists()


def test_artifact_registry_coverage_must_equal_prepared_matrix(
    tmp_path: Path,
) -> None:
    bundle_root = tmp_path / "bundles"
    output_root = tmp_path / "outputs"
    overrides = _overrides(bundle_root)
    registry_path = tmp_path / "model-artifacts" / "artifacts.yaml"
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    payload["artifacts"]["unexpected/coordinates"] = dict(
        payload["artifacts"]["synthetic_flat/coordinates"]
    )
    registry_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    registry_sha = hashlib.sha256(registry_path.read_bytes()).hexdigest()
    overrides = [
        (
            f"models.artifact_registry_sha256={registry_sha}"
            if item.startswith("models.artifact_registry_sha256=")
            else item
        )
        for item in overrides
    ]

    with pytest.raises(ExperimentConfigError, match="coverage must exactly match"):
        run_composed_experiment(
            overrides,
            root=REPOSITORY_ROOT,
            output_root=output_root,
        )
    assert not output_root.exists()


def test_null_learned_sha_fails_before_output(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    with pytest.raises(ExperimentConfigError, match="artifact_registry_sha256"):
        run_composed_experiment(
            [
                "models=diffusion",
                "experiment=learned_smoke",
                f"models.bundle_root={tmp_path / 'bundles'}",
                f"models.artifact_registry={tmp_path / 'artifacts.yaml'}",
            ],
            root=REPOSITORY_ROOT,
            output_root=output_root,
        )
    assert not output_root.exists()
