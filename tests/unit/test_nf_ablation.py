from __future__ import annotations

import csv
import hashlib
import io
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from experiments import global_campaign, nf_ablation

EXPECTED_SENTINELS = (
    "e4/e4_sphere_pca_radius1/dataset",
    "e4/e4_sphere_pca_radius1/coefficients",
    "e7/e7_crescent_moon_radius3.0/dataset",
    "e7/e7_crescent_moon_radius3.0/coefficients",
    "e1/e1_sampled_fmnist_step1/dataset",
    "e1/e1_sampled_fmnist_step7/dataset",
    "e5/e5_downscaled_fmnist/dataset",
    "e5/e5_padded_fmnist_adddim8/dataset",
)
REFERENCE_SENTINELS = {
    "e1/e1_sampled_fmnist_step1/dataset",
    "e5/e5_downscaled_fmnist/dataset",
}


def _stage1_stratum(cell_key: str) -> tuple[str, str, bool]:
    if cell_key.startswith("e1/e1_sampled"):
        return "e1_sample_size", "dataset", cell_key not in REFERENCE_SENTINELS
    if cell_key.startswith("e5/"):
        return "e5_paired_delta", "dataset", cell_key not in REFERENCE_SENTINELS
    representation = cell_key.rsplit("/", 1)[1]
    return f"known_{representation}", representation, True


def _stage1_records(
    *,
    p0_runtime_ratio: float = 12.01,
) -> list[dict[str, Any]]:
    loss_ratio = {
        "C0": 1.0,
        "C1": 0.80,
        "C2": 0.95,
        "C3": 1.05,
        "C4": 1.05,
        "C5": 1.05,
        "P0": 0.60,
    }
    records: list[dict[str, Any]] = []
    for candidate in nf_ablation.STAGE1_CANDIDATES:
        readouts = (
            (nf_ablation.PAPER_PARITY_READOUT,)
            if candidate.independent_fixed_epsilon
            else nf_ablation.READOUTS
        )
        for readout in readouts:
            for cell_key in EXPECTED_SENTINELS:
                stratum, representation, include = _stage1_stratum(cell_key)
                runtime_ratio = (
                    p0_runtime_ratio if candidate.candidate_id == "P0" else 1.5
                )
                if candidate.candidate_id == "C0":
                    runtime_ratio = 1.0
                records.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "readout": readout,
                        "seed": 0,
                        "cell_key": cell_key,
                        "suite_id": cell_key.split("/", 1)[0],
                        "representation": representation,
                        "target_policy": (
                            "sample_size"
                            if stratum == "e1_sample_size"
                            else "paired_delta"
                            if stratum == "e5_paired_delta"
                            else "known_lid"
                        ),
                        "split": "validation",
                        "loss": 100.0 * loss_ratio[candidate.candidate_id],
                        "finite_fraction": 1.0,
                        "stratum": stratum,
                        "include_in_macro": include,
                        "runtime_seconds": 10.0 * runtime_ratio,
                    }
                )
    return records


def _promotion(
    candidate_id: str = "C1", readout: str = "ols5"
) -> nf_ablation.Promotion:
    return nf_ablation.Promotion(
        candidate_id=candidate_id,
        readout=readout,
        median_log_ratio=math.log(0.8),
        win_rate=0.8,
        stratum_median_ratios={"known_dataset": 0.8},
        dataset_to_coefficients_ratio=1.0,
    )


def _known_cells() -> tuple[str, ...]:
    return tuple(
        key
        for key in global_campaign.APPROVED_GLOBAL_CELL_KEYS
        if not key.startswith("e1/e1_sampled") and not key.startswith("e5/")
    )


def _stage2_records(
    *,
    candidate_readout: str = "ols5",
    canonical_ratios: dict[str, float] | None = None,
    generated_ratio: float = 2.0,
) -> list[dict[str, Any]]:
    cells = _known_cells()
    canonical = tuple(key for key in cells if not key.startswith(("e3/", "e4/")))
    if canonical_ratios is None:
        canonical_ratios = {
            key: (0.50 if index < 10 else 0.90) for index, key in enumerate(canonical)
        }
    records: list[dict[str, Any]] = []
    for seed in (0, 1):
        for index, cell_key in enumerate(cells):
            # Vary the control strongly by cell and seed.  A correct gate must
            # compare paired observations, not aggregate absolute MAEs.
            control = 10.0 + float(index) + 100.0 * float(seed)
            ratio = (
                generated_ratio
                if cell_key.startswith(("e3/", "e4/"))
                else canonical_ratios[cell_key]
            )
            common = {
                "seed": seed,
                "cell_key": cell_key,
                "split": "validation",
                "target_policy": "known_lid",
                "finite_fraction": 1.0,
            }
            records.append(
                {
                    **common,
                    "candidate_id": "C0",
                    "readout": "autograd",
                    "loss": control,
                }
            )
            records.append(
                {
                    **common,
                    "candidate_id": "C1",
                    "readout": candidate_readout,
                    "loss": control * ratio,
                }
            )
    return records


def _unified_row(**updates: Any) -> dict[str, Any]:
    row: dict[str, Any] = {field: "" for field in global_campaign._UNIFIED_TABLE_FIELDS}
    row.update(
        {
            "analysis": "known_lid",
            "model_variant": "scale_conditioned_nf",
            "suite_id": "e4",
            "dataset": "e4_sphere_pca_radius1",
            "representation": "dataset",
            "split": "validation",
            "readout": "fixed_likelihood",
        }
    )
    row.update(updates)
    return row


def _csv_bytes(rows: list[dict[str, Any]], *, line_ending: str = "\r\n") -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=global_campaign._UNIFIED_TABLE_FIELDS,
        lineterminator=line_ending,
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def test_candidate_matrix_and_sentinel_dependency_closure_are_exact() -> None:
    nf_ablation.validate_candidate_matrix()

    assert tuple(candidate.candidate_id for candidate in nf_ablation.CANDIDATES) == (
        "C0",
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
    )
    assert tuple(
        candidate.candidate_id for candidate in nf_ablation.STAGE1_CANDIDATES
    ) == (
        "C0",
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "P0",
    )
    assert nf_ablation.PAPER_PARITY_CANDIDATE.independent_fixed_epsilon is True
    assert (
        nf_ablation.PAPER_PARITY_CANDIDATE.selection_scales
        == nf_ablation.SELECTION_SCALES
    )
    assert (
        nf_ablation.CONDITIONAL_SELECTION_SCALES == nf_ablation.SELECTION_SCALES[1:-1]
    )
    assert nf_ablation.STAGE1_SENTINEL_KEYS == EXPECTED_SENTINELS
    # The non-known-LID probes are usable only when their exact reference is
    # in the same candidate/seed DAG.
    assert REFERENCE_SENTINELS <= set(nf_ablation.STAGE1_SENTINEL_KEYS)

    overrides = {
        candidate.candidate_id: dict(candidate.training_overrides)
        for candidate in nf_ablation.STAGE1_CANDIDATES
    }
    assert overrides["C0"] == {"batch_size": 4096, "epochs": 200}
    assert overrides["C1"] == {"batch_size": 1024, "epochs": 200}
    assert overrides["C2"] == {
        "batch_size": 1024,
        "early_stopping_patience": 40,
        "epochs": 400,
    }
    assert overrides["C3"]["hidden_dim"] == 768
    assert overrides["C3"]["num_coupling_layers"] == 12
    assert overrides["C4"]["max_condition_frequency"] == 32.0
    assert overrides["C5"]["conditioner_depth"] == 3
    assert overrides["P0"] == overrides["C2"]


def test_stage1_promotes_only_complete_candidates_and_applies_p0_runtime_gate() -> None:
    # P0 trains nine independent components.  It is scientifically strongest
    # here, but costs just over its declared 12x ceiling and cannot be promoted.
    promotions = nf_ablation.rank_stage1(_stage1_records(p0_runtime_ratio=12.01))
    assert [(row.candidate_id, row.readout) for row in promotions] == [
        ("C1", "autograd")
    ]

    # The threshold is inclusive; at exactly 12x P0 is globally best.
    promotions = nf_ablation.rank_stage1(_stage1_records(p0_runtime_ratio=12.0))
    assert [(row.candidate_id, row.readout) for row in promotions] == [
        ("P0", nf_ablation.PAPER_PARITY_READOUT),
        ("C1", "autograd"),
    ]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong_p0_readout"])
def test_stage1_rejects_non_rectangular_candidate_cell_readout_evidence(
    mutation: str,
) -> None:
    records = _stage1_records()
    target = next(
        row
        for row in records
        if row["candidate_id"] == "C2"
        and row["readout"] == "ols5"
        and row["cell_key"] == EXPECTED_SENTINELS[-1]
    )
    if mutation == "missing":
        records.remove(target)
    elif mutation == "duplicate":
        records.append(dict(target))
    else:
        target = next(row for row in records if row["candidate_id"] == "P0")
        target["readout"] = "autograd"

    with pytest.raises(
        nf_ablation.NFAblationError, match="stage-1|coverage|matrix|repeat"
    ):
        nf_ablation.rank_stage1(records)


def test_stage1_rejects_test_rows_non_sentinels_and_nonfinite_values() -> None:
    records = _stage1_records()
    records[0]["split"] = "test"
    with pytest.raises(nf_ablation.NFAblationError, match="validation-only"):
        nf_ablation.rank_stage1(records)

    records = _stage1_records()
    records[0]["cell_key"] = "e8/e8_spaghetti_pca/dataset"
    with pytest.raises(nf_ablation.NFAblationError, match="sentinel"):
        nf_ablation.rank_stage1(records)

    records = _stage1_records()
    records[0]["loss"] = float("nan")
    with pytest.raises(nf_ablation.NFAblationError, match="non-finite"):
        nf_ablation.rank_stage1(records)


def test_stage2_uses_paired_c0_ratios_and_gates_only_canonical_cells() -> None:
    records = _stage2_records(generated_ratio=25.0)
    winner = nf_ablation.pick_stage2_winner(records, [_promotion()])

    canonical = tuple(
        key for key in _known_cells() if not key.startswith(("e3/", "e4/"))
    )
    expected_ratio = math.exp(
        float(
            np.mean(
                [
                    math.log(0.50 if index < 10 else 0.90)
                    for index, _ in enumerate(canonical)
                ]
            )
        )
    )
    assert winner.candidate_id == "C1"
    assert winner.readout == "ols5"
    assert winner.canonical_geometric_mean_ratio == pytest.approx(expected_ratio)
    assert winner.canonical_wins == 15
    assert winner.canonical_regressions_over_25pct == 0
    assert winner.generated_geometric_mean_ratio == pytest.approx(25.0)


def test_stage2_rejects_more_than_two_canonical_regressions() -> None:
    canonical = tuple(
        key for key in _known_cells() if not key.startswith(("e3/", "e4/"))
    )
    ratios = {key: (1.26 if index < 3 else 0.40) for index, key in enumerate(canonical)}
    with pytest.raises(nf_ablation.NFAblationError, match="no promoted candidate"):
        nf_ablation.pick_stage2_winner(
            _stage2_records(canonical_ratios=ratios),
            [_promotion()],
        )


def test_stage2_rejects_a_readout_that_was_not_promoted() -> None:
    with pytest.raises(nf_ablation.NFAblationError, match="promoted|readout"):
        nf_ablation.pick_stage2_winner(
            _stage2_records(candidate_readout="autograd"),
            [_promotion(readout="ols5")],
        )


def test_global_ols_recovers_exact_per_sample_slopes() -> None:
    scales = np.asarray(nf_ablation.SELECTION_SCALES, dtype=np.float64)
    ambient_dim = 17
    target_lid = np.asarray([2.0, 4.5, 9.0], dtype=np.float64)
    intercept = np.asarray([100.0, -3.0, 7.5], dtype=np.float64)
    curve = intercept[:, None] + (target_lid - ambient_dim)[:, None] * np.log(scales)

    actual = nf_ablation._global_ols_lid(
        curve,
        scales=scales,
        ambient_dim=ambient_dim,
    )

    np.testing.assert_allclose(actual, target_lid, rtol=0.0, atol=1.0e-12)
    assert actual.flags.c_contiguous


@pytest.mark.parametrize(
    ("curve", "scales"),
    [
        (np.zeros((2, 8)), nf_ablation.SELECTION_SCALES[:-1]),
        (np.zeros(9), nf_ablation.SELECTION_SCALES),
        (
            np.asarray([[float("nan")] * 9]),
            nf_ablation.SELECTION_SCALES,
        ),
        (np.zeros((2, 9)), (0.0, *nf_ablation.SELECTION_SCALES[1:])),
    ],
)
def test_global_ols_fails_closed_on_invalid_curves(curve: Any, scales: Any) -> None:
    with pytest.raises(nf_ablation.NFAblationError, match="likelihood curve"):
        nf_ablation._global_ols_lid(curve, scales=scales, ambient_dim=4)


def test_merge_preserves_every_baseline_byte_and_appends_schema_rows(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline.csv"
    combined = tmp_path / "combined.csv"
    original = _csv_bytes([_unified_row()], line_ending="\r\n")
    baseline.write_bytes(original)
    extension = _unified_row(
        model_variant="nf-realnvp-b1024-e400-h512-l8-d2-f100-seed2",
        split="test",
        readout="ols5",
    )

    report = nf_ablation.merge_unified_results(
        baseline_csv=baseline,
        extension_rows=[extension],
        output_csv=combined,
    )

    assert baseline.read_bytes() == original
    assert combined.read_bytes().startswith(original)
    assert report == {
        "baseline_sha256": hashlib.sha256(original).hexdigest(),
        "baseline_rows": 1,
        "extension_rows": 1,
        "combined_rows": 2,
        "combined_sha256": hashlib.sha256(combined.read_bytes()).hexdigest(),
    }
    with combined.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert rows[1]["model_variant"] == extension["model_variant"]


def test_merge_rejects_collisions_without_publishing_output(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.csv"
    baseline.write_bytes(_csv_bytes([_unified_row()]))

    collision = _unified_row(mae=1.0)
    output = tmp_path / "collision.csv"
    with pytest.raises(
        nf_ablation.NFAblationError, match="duplicate unified result key"
    ):
        nf_ablation.merge_unified_results(
            baseline_csv=baseline,
            extension_rows=[collision],
            output_csv=output,
        )
    assert not output.exists()

    fresh = _unified_row(model_variant="fresh-nf", split="test", readout="ols9")
    with pytest.raises(
        nf_ablation.NFAblationError, match="duplicate unified result key"
    ):
        nf_ablation.merge_unified_results(
            baseline_csv=baseline,
            extension_rows=[fresh, dict(fresh)],
            output_csv=output,
        )
    assert not output.exists()


def test_merge_rejects_in_place_and_schema_drift(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.csv"
    baseline.write_bytes(_csv_bytes([_unified_row()]))
    with pytest.raises(nf_ablation.NFAblationError, match="must not overwrite"):
        nf_ablation.merge_unified_results(
            baseline_csv=baseline,
            extension_rows=[],
            output_csv=baseline,
        )

    incomplete = _unified_row(model_variant="new")
    incomplete.pop("mae")
    with pytest.raises(nf_ablation.NFAblationError, match="schema-compatible"):
        nf_ablation.merge_unified_results(
            baseline_csv=baseline,
            extension_rows=[incomplete],
            output_csv=tmp_path / "bad.csv",
        )
