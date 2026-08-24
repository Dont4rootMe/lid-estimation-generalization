from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from experiments.aggregate import aggregate_matrix
from experiments.run_manifest import (
    MANIFEST_SCHEMA_VERSION,
    RunManifest,
    canonical_json,
    hash_dataset_record,
    hash_environment_record,
    make_run_id,
    sha256_bytes,
    sha256_path,
    write_json,
    write_manifest,
)


READOUT_ID = "diffusion_flipd_full"


def _write_cell(
    matrix_dir: Path,
    *,
    dataset: str,
    representation: str,
    prediction: list[float],
    selected_indices_sha256: str,
    paired_rows_sha256: str | None = None,
    n_reference: int,
    resolved_config: dict[str, object],
    target: list[float] | None = None,
) -> dict[str, str]:
    cell_name = f"{dataset}__{representation}"
    cell_dir = matrix_dir / "cells" / cell_name
    cell_dir.mkdir(parents=True)
    prediction_path = cell_dir / f"prediction__{READOUT_ID}.npy"
    np.save(prediction_path, np.asarray(prediction, dtype=np.float64))
    outputs = {prediction_path.name: sha256_path(prediction_path)}
    if target is not None:
        target_path = cell_dir / "test_target.npy"
        np.save(target_path, np.asarray(target, dtype=np.float64))
        outputs[target_path.name] = sha256_path(target_path)
    source_hash = "a" * 64
    dataset_config = dict(resolved_config["dataset"])
    dataset_record = {
        "kind": "synthetic_flat",
        "seed": dataset_config["seed"],
        "generator": dataset_config,
        "training_dataset_identity_kind": "canonical_array_sha256_v1",
        "training_dataset_sha256": "1" * 64,
        "name": dataset,
        "representation": representation,
        "n_reference": n_reference,
        "selected_indices_sha256": selected_indices_sha256,
        **(
            {"paired_rows_sha256": paired_rows_sha256}
            if paired_rows_sha256 is not None
            else {}
        ),
        "sha256": sha256_bytes(
            canonical_json(
                {
                    "dataset": dataset,
                    "representation": representation,
                    "prediction": prediction,
                    "target": target,
                }
            ).encode("utf-8")
        ),
    }
    environment = {
        "python": "fixture",
        "platform": "fixture",
        "machine": "fixture",
        "packages": {},
    }
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
        upstream_sha="not_applicable",
        git={},
        environment=environment,
        resolved_config=resolved_config,
        dataset=dataset_record,
        outputs=outputs,
    )
    manifest_path = cell_dir / "manifest.json"
    write_manifest(manifest_path, manifest)
    return {
        "cell": cell_name,
        "status": "completed",
        "manifest": manifest_path.relative_to(matrix_dir).as_posix(),
    }


def test_aggregate_recomputes_all_analysis_types_without_false_pairing(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "configs" / "datasets" / "registry.yaml"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(
        """
schema_version: 1
benchmark_id: aggregate-fixture
datasets:
  - name: base
    representation: dataset
    available_representations: [dataset, coefficients]
    required_artifacts: [dataset, coefficients]
    expected_samples: {train: 100}
  - name: transformed
    required_artifacts: [dataset]
    expected_samples: {train: 100}
    transformation:
      family: ADI
      reference: base
      parameter: additional_dimensions
      value: 2
      expected_lid_delta: 2.0
      paired_samples: true
  - name: sample2
    required_artifacts: [dataset]
    expected_samples: {train: 50}
    transformation:
      family: sample_size
      reference: base
      parameter: sampling_step
      value: 2
      expected_lid_delta: 0.0
      paired_samples: false
""".lstrip(),
        encoding="utf-8",
    )
    matrix_dir = tmp_path / "matrix"
    matrix_dir.mkdir()
    resolved_config: dict[str, object] = {
        "dataset": {
            "source": "synthetic_flat",
            "seed": 0,
            "registry": registry_path.relative_to(tmp_path).as_posix(),
            "representations": "all",
        },
        "readouts": [READOUT_ID],
    }
    cells = [
        _write_cell(
            matrix_dir,
            dataset="base",
            representation="dataset",
            prediction=[1.0, 2.0],
            target=[1.0, 1.0],
            selected_indices_sha256="aligned",
            paired_rows_sha256="same-upstream-rows",
            n_reference=100,
            resolved_config=resolved_config,
        ),
        _write_cell(
            matrix_dir,
            dataset="base",
            representation="coefficients",
            prediction=[1.5, 1.5],
            target=[1.0, 1.0],
            selected_indices_sha256="aligned",
            paired_rows_sha256="same-upstream-rows",
            n_reference=100,
            resolved_config=resolved_config,
        ),
        _write_cell(
            matrix_dir,
            dataset="transformed",
            representation="dataset",
            prediction=[3.0, 4.0],
            selected_indices_sha256="aligned",
            paired_rows_sha256="same-upstream-rows",
            n_reference=100,
            resolved_config=resolved_config,
        ),
        # The sample-size variant intentionally has a different prediction
        # length.  Aggregation must summarize it independently, not invent a
        # pointwise pairing with the reference dataset.
        _write_cell(
            matrix_dir,
            dataset="sample2",
            representation="dataset",
            prediction=[10.0, 20.0, 30.0],
            selected_indices_sha256="not-paired",
            n_reference=50,
            resolved_config=resolved_config,
        ),
    ]
    write_json(
        matrix_dir / "matrix.json",
        {
            "schema_version": 1,
            "name": "aggregate-fixture",
            "backend": "fixture",
            "config_sha256": "config",
            "requested_cells": len(cells),
            "complete_cells": len(cells),
            "complete": True,
            "cells": cells,
        },
    )

    output = aggregate_matrix(matrix_dir, root=tmp_path)
    first_bytes = output.read_bytes()
    aggregate_matrix(matrix_dir, root=tmp_path)
    assert output.read_bytes() == first_bytes
    report = json.loads(first_bytes)

    assert report["coverage"]["complete"] is True
    assert report["coverage"]["known_lid_records"] == 2
    assert report["coverage"]["paired_transformation_records"] == 1
    assert report["coverage"]["representation_discrepancy_records"] == 1

    paired = report["paired_transformations"][0]
    assert paired["dataset"] == "transformed"
    assert paired["metrics"]["mae"] == 0.0

    curve = report["sample_size_curves"][0]
    assert curve["paired_samples"] is False
    assert [point["dataset"] for point in curve["points"]] == ["base", "sample2"]
    assert curve["points"][1]["summary"]["n"] == 3
    assert curve["points"][1]["summary"]["mean"] == 20.0

    discrepancy = report["representation_discrepancies"][0]
    assert discrepancy["delta_direction"] == "dataset_minus_coefficients"
    assert discrepancy["metrics"]["mae"] == 0.5
