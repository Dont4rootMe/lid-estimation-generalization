from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from experiments.aggregate import aggregate_matrix
from experiments.run_manifest import (
    canonical_json,
    hash_dataset_record,
    make_run_id,
    sha256_bytes,
    sha256_path,
    write_json,
)
from experiments.runner import (
    ExperimentConfigError,
    compose_experiment_config,
    normalize_hydra_config,
    prepare_datasets,
    run_composed_experiment,
    validate_matrix,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_hydra_composes_experiment_dataset_model_and_runtime_groups() -> None:
    smoke = compose_experiment_config(root=REPOSITORY_ROOT)
    assert smoke.experiment.name == "oracle-smoke"
    assert smoke.datasets.source == "synthetic_flat"
    assert smoke.preprocessing.kind == "identity"
    assert smoke.models.backend == "empirical_gaussian_channel_oracle"

    paper = compose_experiment_config(
        ["experiment=paper_oracle_matrix"], root=REPOSITORY_ROOT
    )
    assert paper.datasets.source == "lid_benchmarks"
    assert paper.runtime.limits.reference == 4096
    assert len(paper.datasets.names) == 28

    affine = compose_experiment_config(
        ["preprocessing=scalar_affine"], root=REPOSITORY_ROOT
    )
    assert affine.preprocessing == {
        "kind": "scalar_affine",
        "scale": 1.0,
        "offset": 0.0,
    }


def test_hydra_oracle_smoke_is_complete_and_integrity_checked(tmp_path: Path) -> None:
    overrides = [
        "datasets.n_train=256",
        "datasets.n_validation=32",
        "datasets.n_test=32",
        "runtime.limits.reference=256",
        "runtime.limits.validation=32",
        "runtime.limits.test=32",
        "runtime.limits.reference_chunk=64",
        "runtime.limits.query_chunk=16",
        "experiment.scale_multipliers=[1.0,1.5,2.0]",
        "experiment.selection.min_valid_fraction=0.5",
    ]
    matrix_dir = run_composed_experiment(
        overrides, root=REPOSITORY_ROOT, output_root=tmp_path
    )
    assert validate_matrix(matrix_dir) == []

    matrix = json.loads((matrix_dir / "matrix.json").read_text(encoding="utf-8"))
    assert matrix["config_source"] == "hydra_yaml_composition"
    assert matrix["hydra_overrides"] == overrides
    assert matrix["complete"] is True
    assert matrix["aggregate"] == "aggregate.json"
    aggregate = json.loads(
        (matrix_dir / matrix["aggregate"]).read_text(encoding="utf-8")
    )
    assert aggregate["coverage"]["complete"] is True
    assert aggregate["coverage"]["known_lid_records"] == len(
        config_readouts := compose_experiment_config(
            overrides, root=REPOSITORY_ROOT
        ).experiment.readouts
    )

    cell_dir = next((matrix_dir / "cells").iterdir())
    diffusion = np.load(
        cell_dir / "prediction__diffusion_flipd_full.npy", allow_pickle=False
    )
    fm_full = np.load(
        cell_dir / "prediction__fm_affine_full.npy", allow_pickle=False
    )
    np.testing.assert_array_equal(diffusion, fm_full)
    summary = json.loads((cell_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["scale_space"] == "model"
    assert summary["preprocessing"]["spec"] == {"kind": "identity"}
    assert summary["dataset"]["raw_selected_dataset_sha256"] == (
        summary["dataset"]["model_selected_dataset_sha256"]
    )


def test_scalar_affine_preprocessing_changes_model_identity_and_matrix(
    tmp_path: Path,
) -> None:
    common = [
        "datasets.n_train=32",
        "datasets.n_validation=8",
        "datasets.n_test=8",
        "runtime.limits.reference=32",
        "runtime.limits.validation=8",
        "runtime.limits.test=8",
        "runtime.limits.reference_chunk=16",
        "runtime.limits.query_chunk=8",
        "experiment.scale_multipliers=[1.0,1.5,2.0]",
        "experiment.selection.min_valid_fraction=0.5",
    ]
    identity_config = normalize_hydra_config(
        compose_experiment_config(common, root=REPOSITORY_ROOT)
    )
    affine_overrides = [
        *common,
        "preprocessing=scalar_affine",
        "preprocessing.scale=2.5",
        "preprocessing.offset=-3.0",
    ]
    affine_config = normalize_hydra_config(
        compose_experiment_config(affine_overrides, root=REPOSITORY_ROOT)
    )
    identity = prepare_datasets(identity_config, REPOSITORY_ROOT)[0]
    affine = prepare_datasets(affine_config, REPOSITORY_ROOT)[0]

    np.testing.assert_array_equal(identity.raw_reference, identity.reference)
    np.testing.assert_array_equal(identity.raw_validation, identity.validation)
    np.testing.assert_array_equal(identity.raw_test, identity.test)
    np.testing.assert_array_equal(affine.raw_reference, identity.raw_reference)
    np.testing.assert_allclose(
        affine.reference, identity.raw_reference * 2.5 - 3.0
    )
    assert affine.raw_selected_dataset_sha256 == (
        identity.raw_selected_dataset_sha256
    )
    assert affine.model_selected_dataset_sha256 != (
        identity.model_selected_dataset_sha256
    )
    assert affine.preprocessing_sha256 != identity.preprocessing_sha256

    identity_dir = run_composed_experiment(
        common, root=REPOSITORY_ROOT, output_root=tmp_path
    )
    affine_dir = run_composed_experiment(
        affine_overrides, root=REPOSITORY_ROOT, output_root=tmp_path
    )
    assert affine_dir != identity_dir
    identity_matrix = json.loads(
        (identity_dir / "matrix.json").read_text(encoding="utf-8")
    )
    affine_matrix = json.loads(
        (affine_dir / "matrix.json").read_text(encoding="utf-8")
    )
    assert affine_matrix["config_sha256"] != identity_matrix["config_sha256"]
    assert affine_matrix["input_sha256"] != identity_matrix["input_sha256"]
    assert validate_matrix(identity_dir) == []
    assert validate_matrix(affine_dir) == []


@pytest.mark.parametrize(
    "override",
    [
        "preprocessing.scale=0.0",
        "preprocessing.scale=nan",
        "preprocessing.offset=inf",
    ],
)
def test_scalar_affine_preprocessing_rejects_noninvertible_or_nonfinite_values(
    override: str,
) -> None:
    config = compose_experiment_config(
        ["preprocessing=scalar_affine", override], root=REPOSITORY_ROOT
    )
    with pytest.raises(ExperimentConfigError, match="finite and nonzero"):
        normalize_hydra_config(config)


def test_oracle_reuse_rejects_forged_manifest_identity(tmp_path: Path) -> None:
    overrides = [
        "datasets.n_train=32",
        "datasets.n_validation=8",
        "datasets.n_test=8",
        "runtime.limits.reference=32",
        "runtime.limits.validation=8",
        "runtime.limits.test=8",
        "runtime.limits.reference_chunk=16",
        "runtime.limits.query_chunk=8",
        "experiment.scale_multipliers=[1.0,1.5,2.0]",
        "experiment.selection.min_valid_fraction=0.5",
    ]
    matrix_dir = run_composed_experiment(
        overrides, root=REPOSITORY_ROOT, output_root=tmp_path
    )
    manifest_path = next((matrix_dir / "cells").glob("*/manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["upstream_sha"] = "0" * 40
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="identity mismatch for upstream_sha"):
        run_composed_experiment(
            overrides, root=REPOSITORY_ROOT, output_root=tmp_path
        )


def test_validate_matrix_rejects_manifest_path_not_bound_to_cell(
    tmp_path: Path,
) -> None:
    overrides = [
        "datasets.n_train=32",
        "datasets.n_validation=8",
        "datasets.n_test=8",
        "runtime.limits.reference=32",
        "runtime.limits.validation=8",
        "runtime.limits.test=8",
        "runtime.limits.reference_chunk=16",
        "runtime.limits.query_chunk=8",
        "experiment.scale_multipliers=[1.0,1.5,2.0]",
        "experiment.selection.min_valid_fraction=0.5",
    ]
    matrix_dir = run_composed_experiment(
        overrides, root=REPOSITORY_ROOT, output_root=tmp_path
    )
    matrix_path = matrix_dir / "matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    matrix["cells"][0]["manifest"] = "../foreign/manifest.json"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")

    assert any(
        "manifest path must be cells/" in error
        for error in validate_matrix(matrix_dir)
    )


def test_validate_matrix_recomputes_aggregate_after_hash_is_rewritten(
    tmp_path: Path,
) -> None:
    overrides = [
        "datasets.n_train=32",
        "datasets.n_validation=8",
        "datasets.n_test=8",
        "runtime.limits.reference=32",
        "runtime.limits.validation=8",
        "runtime.limits.test=8",
        "runtime.limits.reference_chunk=16",
        "runtime.limits.query_chunk=8",
        "experiment.scale_multipliers=[1.0,1.5,2.0]",
        "experiment.selection.min_valid_fraction=0.5",
    ]
    matrix_dir = run_composed_experiment(
        overrides, root=REPOSITORY_ROOT, output_root=tmp_path
    )
    matrix_path = matrix_dir / "matrix.json"
    aggregate_path = matrix_dir / "aggregate.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))

    aggregate["coverage"]["known_lid_records"] += 1
    write_json(aggregate_path, aggregate)
    matrix["aggregate_sha256"] = sha256_path(aggregate_path)
    write_json(matrix_path, matrix)

    assert "aggregate content is inconsistent with raw cell outputs" in (
        validate_matrix(matrix_dir)
    )


def test_validate_matrix_rejects_posthoc_oracle_to_learned_relabel(
    tmp_path: Path,
) -> None:
    overrides = [
        "datasets.n_train=32",
        "datasets.n_validation=8",
        "datasets.n_test=8",
        "runtime.limits.reference=32",
        "runtime.limits.validation=8",
        "runtime.limits.test=8",
        "runtime.limits.reference_chunk=16",
        "runtime.limits.query_chunk=8",
        "experiment.scale_multipliers=[1.0,1.5,2.0]",
        "experiment.selection.min_valid_fraction=0.5",
    ]
    matrix_dir = run_composed_experiment(
        overrides, root=REPOSITORY_ROOT, output_root=tmp_path
    )
    matrix_path = matrix_dir / "matrix.json"
    aggregate_path = matrix_dir / "aggregate.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))

    # Auditor scenario: forge every visible backend/evidence label and then
    # recompute the stored aggregate file hash.  The immutable cell config is
    # still oracle evidence and must make the matrix invalid.
    matrix["backend"] = "learned_field_bundle"
    matrix["evidence_level"] = "learned_model"
    aggregate["matrix"]["backend"] = "learned_field_bundle"
    aggregate["matrix"]["evidence_level"] = "learned_model"
    write_json(aggregate_path, aggregate)
    matrix["aggregate_sha256"] = sha256_path(aggregate_path)
    write_json(matrix_path, matrix)

    errors = validate_matrix(matrix_dir)
    assert any("resolved_config.backend disagrees" in error for error in errors)
    assert any(
        "resolved_config.evidence_level disagrees" in error for error in errors
    )


def test_validate_matrix_binds_preprocessing_record_to_resolved_hydra(
    tmp_path: Path,
) -> None:
    overrides = [
        "datasets.n_train=32",
        "datasets.n_validation=8",
        "datasets.n_test=8",
        "runtime.limits.reference=32",
        "runtime.limits.validation=8",
        "runtime.limits.test=8",
        "runtime.limits.reference_chunk=16",
        "runtime.limits.query_chunk=8",
        "experiment.scale_multipliers=[1.0,1.5,2.0]",
        "experiment.selection.min_valid_fraction=0.5",
    ]
    matrix_dir = run_composed_experiment(
        overrides, root=REPOSITORY_ROOT, output_root=tmp_path
    )
    matrix_path = matrix_dir / "matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    cell = matrix["cells"][0]
    manifest_path = matrix_dir / cell["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Keep all generic manifest identities self-consistent while forging the
    # scientific meaning of the preprocessing record only.
    manifest["dataset"]["preprocessing"]["spec"] = {
        "kind": "scalar_affine",
        "scale": 2.0,
        "offset": 0.0,
    }
    manifest["dataset_record_sha256"] = hash_dataset_record(manifest["dataset"])
    manifest["run_id"] = make_run_id(
        config=manifest["resolved_config"],
        source_tree_sha256=manifest["source_tree_sha256"],
        dataset_record_sha256=manifest["dataset_record_sha256"],
        environment_sha256=manifest["environment_sha256"],
    )
    old_cell_dir = manifest_path.parent
    new_cell_name = (
        f"{manifest['dataset']['name']}__"
        f"{manifest['dataset']['representation']}__{manifest['run_id']}"
    )
    new_cell_dir = old_cell_dir.with_name(new_cell_name)
    old_cell_dir.rename(new_cell_dir)
    write_json(new_cell_dir / "manifest.json", manifest)
    cell["cell"] = new_cell_name
    cell["manifest"] = f"cells/{new_cell_name}/manifest.json"
    write_json(matrix_path, matrix)

    errors = validate_matrix(matrix_dir)
    assert any(
        "preprocessing.spec disagrees with resolved_config" in error
        for error in errors
    )
    assert any(
        "preprocessing.sha256 is inconsistent with canonical spec" in error
        for error in errors
    )


def test_validate_matrix_rejects_omitted_declared_dataset_cell(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "benchmarks"
    for dataset_index, dataset_name in enumerate(("alpha", "beta")):
        for split_index, (split, n_samples) in enumerate(
            (("train", 12), ("val", 4), ("test", 4))
        ):
            split_dir = data_root / dataset_name / split
            split_dir.mkdir(parents=True)
            rng = np.random.default_rng(100 * dataset_index + split_index)
            np.save(split_dir / "dataset.npy", rng.normal(size=(n_samples, 2)))
            np.save(split_dir / "lid.npy", np.ones(n_samples, dtype=np.float64))

    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        """
schema_version: 1
benchmark_id: matrix-omission-fixture
datasets:
  - name: alpha
    official: false
    required_artifacts: [dataset, lid]
    expected_lid: 1
    expected_samples: {train: 12, val: 4, test: 4}
    expected_shapes: {dataset: [2]}
  - name: beta
    official: false
    required_artifacts: [dataset, lid]
    expected_lid: 1
    expected_samples: {train: 12, val: 4, test: 4}
    expected_shapes: {dataset: [2]}
""".lstrip(),
        encoding="utf-8",
    )
    overlay_path = tmp_path / "overlay.yaml"
    overlay_path.write_text(
        """
schema_version: 1
overlay_id: matrix-omission-fixture
base_benchmark_id: matrix-omission-fixture
overrides:
  - name: alpha
    notes: generated fixture
""".lstrip(),
        encoding="utf-8",
    )
    overrides = [
        "datasets=lid_benchmarks_generated",
        f"datasets.root={data_root}",
        f"datasets.registry={registry_path}",
        f"datasets.registry_overlay={overlay_path}",
        "+datasets.names=[alpha,beta]",
        "datasets.representations=default",
        "runtime.limits.reference=12",
        "runtime.limits.validation=4",
        "runtime.limits.test=4",
        "runtime.limits.reference_chunk=6",
        "runtime.limits.query_chunk=4",
        "experiment.scale_multipliers=[0.5,1.0,2.0]",
        "experiment.selection.min_valid_fraction=0.5",
        "experiment.selection.min_effective_sample_size=0.0",
    ]
    matrix_dir = run_composed_experiment(
        overrides, root=REPOSITORY_ROOT, output_root=tmp_path / "outputs"
    )
    matrix_path = matrix_dir / "matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    kept = next(
        cell for cell in matrix["cells"] if cell["cell"].startswith("alpha__")
    )
    omitted = next(
        cell for cell in matrix["cells"] if cell["cell"].startswith("beta__")
    )
    shutil.rmtree(matrix_dir / "cells" / omitted["cell"])
    matrix["cells"] = [kept]
    matrix["requested_cells"] = 1
    matrix["complete_cells"] = 1

    manifest = json.loads(
        (matrix_dir / kept["manifest"]).read_text(encoding="utf-8")
    )
    dataset = manifest["dataset"]
    input_record = {
        "dataset": dataset["name"],
        "representation": dataset["representation"],
        "selected_dataset_sha256": dataset.get(
            "selected_dataset_sha256", dataset["sha256"]
        ),
        "model_selected_dataset_sha256": dataset[
            "model_selected_dataset_sha256"
        ],
        "training_dataset_sha256": dataset["training_dataset_sha256"],
        "preprocessing_sha256": dataset["preprocessing"]["sha256"],
    }
    matrix["input_sha256"] = sha256_bytes(
        canonical_json([input_record]).encode("utf-8")
    )
    matrix["matrix_identity_sha256"] = sha256_bytes(
        canonical_json(
            {
                "config_sha256": matrix["config_sha256"],
                "input_sha256": matrix["input_sha256"],
            }
        ).encode("utf-8")
    )
    forged_dir = matrix_dir.with_name(
        f"{matrix['name']}__{matrix['matrix_identity_sha256'][:12]}"
    )
    matrix_dir.rename(forged_dir)
    matrix_path = forged_dir / "matrix.json"
    write_json(matrix_path, matrix)
    aggregate_path = aggregate_matrix(forged_dir, root=REPOSITORY_ROOT)
    matrix["aggregate_sha256"] = sha256_path(aggregate_path)
    write_json(matrix_path, matrix)

    errors = validate_matrix(forged_dir, root=REPOSITORY_ROOT)
    assert any(
        "dataset/representation cell grid mismatch" in error for error in errors
    )
    assert any(
        "requested_cells does not match Hydra dataset/representation grid" in error
        for error in errors
    )


def test_configuration_tree_contains_yaml_only() -> None:
    forbidden = [
        path
        for path in (REPOSITORY_ROOT / "configs").rglob("*")
        if path.is_file() and path.suffix not in {".yaml", ".md"}
    ]
    assert forbidden == []


@pytest.mark.parametrize(
    ("override", "path"),
    [
        ("+experiment.typo=1", "experiment"),
        ("+datasets.typo=1", "datasets"),
        ("+preprocessing.typo=1", "preprocessing"),
        ("+models.typo=1", "models"),
        ("+runtime.typo=1", "runtime"),
        ("+experiment.selection.typo=1", "experiment.selection"),
        ("+runtime.limits.typo=1", "runtime.limits"),
    ],
)
def test_unknown_hydra_fields_are_rejected_before_normalization(
    override: str, path: str
) -> None:
    config = compose_experiment_config([override], root=REPOSITORY_ROOT)
    with pytest.raises(ExperimentConfigError, match=rf"fields in {path}"):
        normalize_hydra_config(config)
