from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from experiments import global_campaign, nf_ablation, nf_readout_followup


def test_checked_in_reviewed_evidence_manifest_is_exact() -> None:
    root = Path(__file__).resolve().parents[2]
    reviewed = nf_readout_followup.load_reviewed_evidence(
        root / "configs" / "campaign" / "nf_readout_followup_reviewed_evidence.yaml"
    )

    assert reviewed.source_campaign_identity == (
        "b37fc13d34fec456905cbdfdbb4245df117c21f58985c550f91c87b9d4c9813f"
    )
    assert reviewed.stage2_scores_sha256 == (
        "f4763c57efea3338bbd9cce8805e3dbeed68fdc258be60e821a5b320f4da82c0"
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _stage2_fixture(
    tmp_path: Path,
) -> tuple[
    Path,
    nf_readout_followup.ReviewedEvidence,
    str,
    str,
]:
    root = tmp_path / "source-ablation"
    baseline_identity = "a" * 64
    baseline_sha = "b" * 64
    identity = {
        "schema_version": 1,
        "ablation_id": nf_ablation.NF_ABLATION_ID,
        "baseline_campaign_identity": baseline_identity,
        "baseline_unified_sha256": baseline_sha,
    }
    source_identity = nf_ablation._canonical_sha(identity)
    _write_json(root / "ablation_identity.json", identity)
    _write_json(
        root / "state" / "stage1" / "promotions.json",
        {
            "schema_version": 1,
            "campaign_identity": source_identity,
            "promotions": [{"candidate_id": "C1", "readout": "ols5"}],
        },
    )
    stage1_completed: list[dict[str, Any]] = []
    stage1_scores: list[dict[str, Any]] = []
    for candidate in nf_ablation.STAGE1_CANDIDATES:
        readouts = (
            (nf_ablation.PAPER_PARITY_READOUT,)
            if candidate.independent_fixed_epsilon
            else nf_ablation.READOUTS
        )
        for cell_index, cell_key in enumerate(nf_ablation.STAGE1_SENTINEL_KEYS):
            directory = (
                root
                / "stage1-cells"
                / candidate.candidate_id.lower()
                / f"cell-{cell_index:02d}"
            )
            _write_json(directory / "manifest.json", {"outputs": []})
            _write_json(directory / "summary.json", {"sealed": True})
            stage1_completed.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "seed": 0,
                    "cell_key": cell_key,
                    "evaluate_test": False,
                    "test_readout": None,
                    "path": directory.relative_to(root).as_posix(),
                    "manifest_sha256": nf_ablation._sha256_path(
                        directory / "manifest.json"
                    ),
                    "summary_sha256": nf_ablation._sha256_path(
                        directory / "summary.json"
                    ),
                }
            )
            for readout in readouts:
                stage1_scores.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "readout": readout,
                        "seed": 0,
                        "cell_key": cell_key,
                        "split": "validation",
                        "finite_fraction": 1.0,
                        "loss": 1.0,
                    }
                )
    assert len(stage1_completed) == 56
    assert len(stage1_scores) == 248
    _write_json(
        root / "state" / "stage1" / "ledger.json",
        {
            "schema_version": 1,
            "campaign_identity": source_identity,
            "stage": "stage1",
            "expected_task_count": 56,
            "completed_tasks": stage1_completed,
            "complete": True,
        },
    )
    _write_json(
        root / "state" / "stage1" / "validation_scores.json",
        {
            "schema_version": 1,
            "campaign_identity": source_identity,
            "records": stage1_scores,
        },
    )
    known_cells = tuple(
        key
        for key in global_campaign.APPROVED_GLOBAL_CELL_KEYS
        if not key.startswith("e1/e1_sampled") and not key.startswith("e5/")
    )
    canonical = tuple(key for key in known_cells if not key.startswith(("e3/", "e4/")))
    completed: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    for candidate_id in ("C0", "C1"):
        for seed in (0, 1):
            for cell_index, cell_key in enumerate(known_cells):
                directory = (
                    root
                    / "cells"
                    / candidate_id.lower()
                    / f"seed-{seed}"
                    / f"cell-{cell_index:02d}"
                )
                _write_json(directory / "manifest.json", {"outputs": []})
                _write_json(
                    directory / "summary.json",
                    {
                        "candidate_id": candidate_id,
                        "seed": seed,
                        "cell_key": cell_key,
                        "evaluation_splits": ["validation"],
                    },
                )
                completed.append(
                    {
                        "candidate_id": candidate_id,
                        "seed": seed,
                        "cell_key": cell_key,
                        "evaluate_test": False,
                        "test_readout": None,
                        "path": directory.relative_to(root).as_posix(),
                        "manifest_sha256": nf_ablation._sha256_path(
                            directory / "manifest.json"
                        ),
                        "summary_sha256": nf_ablation._sha256_path(
                            directory / "summary.json"
                        ),
                    }
                )
                for readout in nf_ablation.READOUTS:
                    if candidate_id == "C0":
                        loss = 2.0 if readout == "ols5" else 10.0
                    elif readout != "ols5":
                        loss = 8.0
                    elif cell_key.startswith(("e3/", "e4/")):
                        loss = 1.2
                    else:
                        canonical_index = canonical.index(cell_key)
                        loss = 1.6 if canonical_index < 11 else 3.0
                    scores.append(
                        {
                            "candidate_id": candidate_id,
                            "readout": readout,
                            "seed": seed,
                            "cell_key": cell_key,
                            "split": "validation",
                            "target_policy": "known_lid",
                            "finite_fraction": 1.0,
                            "loss": loss,
                        }
                    )
    assert len(completed) == 76
    assert len(scores) == 380
    _write_json(
        root / "state" / "stage2" / "ledger.json",
        {
            "schema_version": 1,
            "campaign_identity": source_identity,
            "stage": "stage2",
            "expected_task_count": 76,
            "completed_tasks": completed,
            "complete": True,
        },
    )
    _write_json(
        root / "state" / "stage2" / "validation_scores.json",
        {
            "schema_version": 1,
            "campaign_identity": source_identity,
            "records": scores,
        },
    )
    reviewed = nf_readout_followup.ReviewedEvidence(
        source_campaign_identity=source_identity,
        source_identity_sha256=nf_ablation._sha256_path(
            root / "ablation_identity.json"
        ),
        source_baseline_campaign_identity=baseline_identity,
        source_baseline_unified_sha256=baseline_sha,
        stage1_ledger_sha256=nf_ablation._sha256_path(
            root / "state" / "stage1" / "ledger.json"
        ),
        stage1_scores_sha256=nf_ablation._sha256_path(
            root / "state" / "stage1" / "validation_scores.json"
        ),
        promotions_sha256=nf_ablation._sha256_path(
            root / "state" / "stage1" / "promotions.json"
        ),
        stage2_ledger_sha256=nf_ablation._sha256_path(
            root / "state" / "stage2" / "ledger.json"
        ),
        stage2_scores_sha256=nf_ablation._sha256_path(
            root / "state" / "stage2" / "validation_scores.json"
        ),
    )
    return root, reviewed, baseline_identity, baseline_sha


def test_stage2_evidence_freezes_readout_without_relaxing_training_gate(
    tmp_path: Path,
) -> None:
    root, reviewed, baseline_identity, baseline_sha = _stage2_fixture(tmp_path)

    evidence = nf_readout_followup.load_stage2_evidence(
        root,
        reviewed=reviewed,
        expected_baseline_campaign_identity=baseline_identity,
        expected_baseline_unified_sha256=baseline_sha,
    )

    assert evidence.completed_tasks == 76
    assert evidence.validation_records == 380
    assert evidence.promoted_candidate_id == "C1"
    assert evidence.promoted_readout == "ols5"
    assert evidence.failed_training_gate.passed is False
    assert evidence.failed_training_gate.canonical_wins == 11
    assert evidence.failed_training_gate.canonical_regressions_over_25pct == 4
    assert evidence.frozen_readout_gate.passed is True
    assert evidence.frozen_readout_gate.canonical_geometric_mean_ratio == pytest.approx(
        0.2
    )
    assert evidence.frozen_readout_gate.canonical_wins == 15


def test_stage2_evidence_rejects_wrong_explicit_campaign_identity(
    tmp_path: Path,
) -> None:
    root, reviewed, baseline_identity, baseline_sha = _stage2_fixture(tmp_path)
    wrong = nf_readout_followup.ReviewedEvidence(
        **{**reviewed.__dict__, "source_campaign_identity": "f" * 64}
    )

    with pytest.raises(
        nf_readout_followup.NFReadoutFollowupError,
        match="source campaign identity differs",
    ):
        nf_readout_followup.load_stage2_evidence(
            root,
            reviewed=wrong,
            expected_baseline_campaign_identity=baseline_identity,
            expected_baseline_unified_sha256=baseline_sha,
        )


@dataclass(frozen=True)
class _FakeCell:
    key: str
    inventory_id: str
    suite_id: str
    dataset: str
    representation: str
    target_policy: str
    reference_dataset: str | None
    expected_lid_delta: float | None


def _fake_cells() -> tuple[_FakeCell, ...]:
    cells: list[_FakeCell] = []
    for index, key in enumerate(global_campaign.APPROVED_GLOBAL_CELL_KEYS):
        suite_id, dataset, representation = key.split("/")
        if dataset.startswith("e1_sampled"):
            target_policy = "sample_size"
            reference_dataset = "e1_sampled_fmnist_step1"
            expected_lid_delta = 0.0
        elif suite_id == "e5":
            target_policy = "paired_delta"
            reference_dataset = "e5_downscaled_fmnist"
            expected_lid_delta = 0.0
        else:
            target_policy = "known_lid"
            reference_dataset = None
            expected_lid_delta = None
        cells.append(
            _FakeCell(
                key=key,
                inventory_id=f"inventory-{index}",
                suite_id=suite_id,
                dataset=dataset,
                representation=representation,
                target_policy=target_policy,
                reference_dataset=reference_dataset,
                expected_lid_delta=expected_lid_delta,
            )
        )
    return tuple(cells)


def _prepared(tmp_path: Path) -> nf_ablation._PreparedNFAblation:
    cells = _fake_cells()
    preflight = nf_ablation.BaselinePreflight(
        baseline_root=tmp_path / "baseline",
        baseline_campaign_identity="a" * 64,
        baseline_unified_sha256="b" * 64,
        baseline_row_count=1872,
        config={},
        cells=cells,
        input_sha256_by_key={cell.key: "c" * 64 for cell in cells},
        source_records={cell.inventory_id: {"id": cell.inventory_id} for cell in cells},
    )
    return nf_ablation._PreparedNFAblation(
        project_root=str(tmp_path),
        campaign_root=str(tmp_path / "followup"),
        campaign_identity="d" * 64,
        identity_record={},
        preflight=preflight,
        worker_count=2,
    )


def test_followup_uses_78_trainings_for_three_output_variants(tmp_path: Path) -> None:
    tasks = nf_readout_followup._followup_tasks(_prepared(tmp_path))

    assert len(tasks) == 78
    seed2 = [task for task in tasks if task.seed == 2]
    seed3 = [task for task in tasks if task.seed == 3]
    assert len(seed2) == len(seed3) == 39
    assert {task.test_readout for task in seed2} == {("autograd", "ols5")}
    assert {task.test_readout for task in seed3} == {"ols5"}
    assert nf_ablation._task_descriptor(seed2[0])["test_readout"] == [
        "autograd",
        "ols5",
    ]


def _metrics(value: float) -> dict[str, Any]:
    return {
        "n": 3,
        "finite_n": 3,
        "finite_fraction": 1.0,
        "mean": value,
        "std": 0.0,
        "median": value,
        "q05": value,
        "q95": value,
        "target_finite_n": 3,
        "mae": value,
        "rmse": value,
        "bias": value,
        "median_absolute_error": value,
    }


def _make_results(root: Path, cells: tuple[_FakeCell, ...]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for seed in (2, 3):
        for index, cell in enumerate(cells):
            directory = root / f"seed-{seed}" / f"cell-{index}"
            directory.mkdir(parents=True)
            readouts = ("autograd", "ols5") if seed == 2 else ("ols5",)
            for split in ("validation", "test"):
                if cell.target_policy == "known_lid":
                    np.save(directory / f"{split}_target.npy", np.ones(3))
                elif cell.target_policy == "paired_delta":
                    np.save(directory / f"{split}_labels.npy", np.arange(3))
            for readout in readouts:
                for split in ("validation", "test"):
                    np.save(
                        directory / f"{split}_prediction__{readout}.npy",
                        np.full(3, float(index + seed)),
                    )
            identity = {
                "candidate_id": "C0",
                "training_seed": seed,
                "cell": {
                    "suite_id": cell.suite_id,
                    "dataset": cell.dataset,
                    "representation": cell.representation,
                },
            }
            run_id = nf_ablation._canonical_sha(identity)
            summary = {
                "model_variant": nf_ablation._model_variant(
                    nf_ablation.candidate_by_id("C0"), seed=seed
                ),
                "partition": {"n_source_train": 100},
                "readouts": {
                    readout: {
                        "selected_index": 2,
                        "selected_scale": 0.1,
                        "metrics": {
                            "validation": _metrics(1.0),
                            "test": _metrics(2.0),
                        },
                    }
                    for readout in readouts
                },
                "run_id": run_id,
                "run_id_contract": "full_sha256_of_cell_identity_v1",
                "cell_id": run_id[:20],
                "training_attestation_sha256": "e" * 64,
            }
            _write_json(directory / "identity.json", identity)
            _write_json(directory / "summary.json", summary)
            _write_json(directory / "manifest.json", {"outputs": []})
            results.append(
                {
                    "candidate_id": "C0",
                    "seed": seed,
                    "cell_key": cell.key,
                    "directory": str(directory),
                    "summary": summary,
                }
            )
    return results


def test_followup_rows_are_exact_unambiguous_234_matrix(tmp_path: Path) -> None:
    cells = _fake_cells()
    results = _make_results(tmp_path / "sealed", cells)

    rows, provenance = nf_readout_followup.build_followup_unified_rows(results, cells)

    assert len(rows) == len(provenance) == 234
    assert len({row["model_variant"] for row in rows}) == 3
    assert sum(row["readout"] == "autograd" for row in rows) == 78
    assert sum(row["readout"] == "ols5" for row in rows) == 156
    assert sum(row["analysis"] == "known_lid" for row in rows) == 114
    assert sum(row["analysis"] == "e1_sample_size_stability" for row in rows) == 78
    assert sum(row["analysis"] == "e5_paired_delta" for row in rows) == 42
    assert {row["evaluation_role"] for row in provenance} == {
        "validation_selection_conditioned",
        "test_confirmatory",
    }
    assert (
        len(
            {
                (
                    row["model_variant"],
                    row["analysis"],
                    row["suite_id"],
                    row["dataset"],
                    row["representation"],
                    row["split"],
                    row["readout"],
                )
                for row in rows
            }
        )
        == 234
    )


def _baseline_row(index: int) -> dict[str, Any]:
    row = {field: "" for field in global_campaign._UNIFIED_TABLE_FIELDS}
    row.update(
        {
            "model_variant": f"baseline-{index}",
            "analysis": "known_lid",
            "suite_id": "e0",
            "dataset": "baseline",
            "representation": "dataset",
            "split": "validation",
            "readout": "baseline",
        }
    )
    return row


def test_finalize_produces_self_contained_compact_bundle(tmp_path: Path) -> None:
    source_root, _reviewed, baseline_identity, _old_sha = _stage2_fixture(tmp_path)
    baseline_root = tmp_path / "baseline"
    baseline_csv = baseline_root / "unified_results.csv"
    nf_ablation._write_csv(
        baseline_csv, [_baseline_row(index) for index in range(1872)]
    )
    baseline_sha = nf_ablation._sha256_path(baseline_csv)

    identity_path = source_root / "ablation_identity.json"
    source_identity_record = json.loads(identity_path.read_text(encoding="utf-8"))
    source_identity_record["baseline_unified_sha256"] = baseline_sha
    _write_json(identity_path, source_identity_record)
    source_identity = nf_ablation._canonical_sha(source_identity_record)
    for relative in (
        "state/stage1/ledger.json",
        "state/stage1/validation_scores.json",
        "state/stage1/promotions.json",
        "state/stage2/ledger.json",
        "state/stage2/validation_scores.json",
    ):
        path = source_root / relative
        value = json.loads(path.read_text(encoding="utf-8"))
        value["campaign_identity"] = source_identity
        _write_json(path, value)
    reviewed = nf_readout_followup.ReviewedEvidence(
        source_campaign_identity=source_identity,
        source_identity_sha256=nf_ablation._sha256_path(identity_path),
        source_baseline_campaign_identity=baseline_identity,
        source_baseline_unified_sha256=baseline_sha,
        stage1_ledger_sha256=nf_ablation._sha256_path(
            source_root / "state/stage1/ledger.json"
        ),
        stage1_scores_sha256=nf_ablation._sha256_path(
            source_root / "state/stage1/validation_scores.json"
        ),
        promotions_sha256=nf_ablation._sha256_path(
            source_root / "state/stage1/promotions.json"
        ),
        stage2_ledger_sha256=nf_ablation._sha256_path(
            source_root / "state/stage2/ledger.json"
        ),
        stage2_scores_sha256=nf_ablation._sha256_path(
            source_root / "state/stage2/validation_scores.json"
        ),
    )
    evidence = nf_readout_followup.load_stage2_evidence(
        source_root,
        reviewed=reviewed,
        expected_baseline_campaign_identity=baseline_identity,
        expected_baseline_unified_sha256=baseline_sha,
    )
    cells = _fake_cells()
    preflight = nf_ablation.BaselinePreflight(
        baseline_root=baseline_root,
        baseline_campaign_identity=baseline_identity,
        baseline_unified_sha256=baseline_sha,
        baseline_row_count=1872,
        config={},
        cells=cells,
        input_sha256_by_key={cell.key: "c" * 64 for cell in cells},
        source_records={cell.inventory_id: {"id": cell.inventory_id} for cell in cells},
    )
    root = tmp_path / "final"
    results = _make_results(root / "cells", cells)
    _write_json(root / "followup_identity.json", {"followup": "test"})
    for relative in (
        "baseline_provenance.json",
        "source_stage2_evidence.json",
        "reviewed_evidence.json",
        "preflight.json",
    ):
        _write_json(root / relative, {"test": True})
    (root / "resolved_followup.yaml").write_text("test: true\n", encoding="utf-8")
    for relative in (
        "telemetry/nvidia_smi.jsonl",
        "telemetry/worker_occupancy.jsonl",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    completed = []
    for result in results:
        directory = Path(result["directory"])
        completed.append(
            {
                "task_id": result["summary"]["run_id"],
                "candidate_id": result["candidate_id"],
                "seed": result["seed"],
                "cell_key": result["cell_key"],
                "path": directory.relative_to(root).as_posix(),
                "manifest_sha256": nf_ablation._sha256_path(
                    directory / "manifest.json"
                ),
                "summary_sha256": nf_ablation._sha256_path(directory / "summary.json"),
            }
        )
    _write_json(
        root / "state/stage3_readout_followup/ledger.json",
        {"complete": True, "completed_tasks": completed},
    )

    finalized = nf_readout_followup.finalize_followup_campaign(
        root, preflight, evidence, results
    )

    assert nf_readout_followup.validate_followup_campaign(finalized) == []
    manifest = json.loads((root / "campaign.json").read_text(encoding="utf-8"))
    assert manifest["baseline"]["byte_size"] == baseline_csv.stat().st_size
    assert len(manifest["source_stage2"]["bundled_evidence"]) == 6

    extension_path = root / manifest["extension"]["path"]
    original_extension = extension_path.read_bytes()
    _header, extension_rows = nf_ablation._read_csv(extension_path)
    extension_rows[0]["selected_index"] = "999"
    nf_ablation._write_csv(extension_path, extension_rows)
    assert any(
        "row/sealed-summary binding differs" in error
        for error in nf_readout_followup._remote_ledger_join_errors(root, manifest)
    )
    extension_path.write_bytes(original_extension)

    ledger_path = root / manifest["stage"]["ledger_path"]
    original_ledger = ledger_path.read_bytes()
    ledger = json.loads(original_ledger)
    ledger["completed_tasks"][0]["task_id"] = "0" * 64
    _write_json(ledger_path, ledger)
    assert any(
        "row/sealed-summary binding differs" in error
        for error in nf_readout_followup._remote_ledger_join_errors(root, manifest)
    )
    ledger_path.write_bytes(original_ledger)

    compact = tmp_path / "compact"
    compact.mkdir()
    shutil.copyfile(root / "campaign.json", compact / "campaign.json")
    for output in manifest["outputs"]:
        source = root / output["path"]
        destination = compact / output["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    baseline_root.rename(tmp_path / "baseline-moved")
    source_root.rename(tmp_path / "source-moved")
    shutil.rmtree(root / "cells")

    assert nf_readout_followup.validate_followup_compact_bundle(compact) == []
    combined = compact / manifest["combined"]["path"]
    original_combined = combined.read_bytes()
    payload = bytearray(original_combined)
    payload[0] ^= 1
    combined.write_bytes(payload)
    assert any(
        "prefix" in error or "output differs" in error
        for error in nf_readout_followup.validate_followup_compact_bundle(compact)
    )
    combined.write_bytes(original_combined)

    compact_manifest_path = compact / "campaign.json"
    original_manifest = compact_manifest_path.read_bytes()
    compact_manifest = json.loads(original_manifest)
    baseline_size = int(compact_manifest["baseline"]["byte_size"])
    suffix_lines = original_combined[baseline_size:].splitlines(keepends=True)
    suffix_lines[0], suffix_lines[1] = suffix_lines[1], suffix_lines[0]
    combined.write_bytes(original_combined[:baseline_size] + b"".join(suffix_lines))
    combined_sha = nf_ablation._sha256_path(combined)
    compact_manifest["combined"]["combined_sha256"] = combined_sha
    for output in compact_manifest["outputs"]:
        if output["path"] == compact_manifest["combined"]["path"]:
            output["sha256"] = combined_sha
            output["size"] = combined.stat().st_size
    _write_json(compact_manifest_path, compact_manifest)
    assert "combined CSV extension suffix differs" in (
        nf_readout_followup.validate_followup_compact_bundle(compact)
    )
    combined.write_bytes(original_combined)
    compact_manifest_path.write_bytes(original_manifest)

    provenance_path = compact / manifest["provenance"]["path"]
    original_provenance = provenance_path.read_bytes()
    provenance = json.loads(original_provenance)
    provenance["records"][0]["dataset"] = "tampered"
    _write_json(provenance_path, provenance)
    assert nf_readout_followup.validate_followup_compact_bundle(compact)
    provenance_path.write_bytes(original_provenance)

    audit_path = compact / manifest["row_audit"]["path"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["records"][0]["csv_row_sha256"] = "0" * 64
    _write_json(audit_path, audit)
    assert nf_readout_followup.validate_followup_compact_bundle(compact)
