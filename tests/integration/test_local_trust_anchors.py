from __future__ import annotations

from pathlib import Path

import experiments.runner as runner


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_matrix_validation_is_anchored_to_current_source_tree(
    tmp_path: Path, monkeypatch
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
    matrix_dir = runner.run_composed_experiment(
        overrides,
        root=REPOSITORY_ROOT,
        output_root=tmp_path,
    )
    assert runner.validate_matrix(matrix_dir) == []

    monkeypatch.setattr(
        runner,
        "hash_declared_sources",
        lambda root: "0" * 64,
    )
    assert any(
        "manifest identity mismatch for source_tree_sha256" in error
        for error in runner.validate_matrix(matrix_dir)
    )
