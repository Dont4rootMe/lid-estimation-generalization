from __future__ import annotations

import csv
import io
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import global_campaign_v2 as campaign
from experiments.metrics import known_lid_metrics
from experiments.run_manifest import canonical_json, sha256_bytes, sha256_path


def _cell(dataset: str, reference_dataset: str) -> campaign.CampaignCell:
    return campaign.CampaignCell(
        inventory_id="v2-selection-failure-fixture",
        source_kind="generated_at_pinned_revision",
        exact_archive=False,
        dataset_config="fixture.yaml",
        data_root="fixture",
        registry="fixture.yaml",
        registry_overlay=None,
        upstream_revision="fixture",
        archive_sha256=None,
        provenance_label="fixture",
        suite_id="e1",
        dataset=dataset,
        representation="dataset",
        target_policy="sample_size",
        selection_protocol="fixture",
        comparison_group="fixture",
        reference_dataset=reference_dataset,
        expected_lid_delta=0.0,
    )


def _data(cell: campaign.CampaignCell) -> campaign.CellData:
    train = np.column_stack((np.linspace(0.0, 1.0, 12), np.ones(12)))
    validation = train[:4].copy()
    test = train[4:8].copy()
    record = {
        "schema_version": 1,
        "cell_key": cell.key,
        "feature_shape": [2],
    }
    return campaign.CellData(
        train=train,
        validation=validation,
        test=test,
        train_target=None,
        validation_target=None,
        test_target=None,
        validation_labels=np.arange(4, dtype=np.int64),
        test_labels=np.arange(4, dtype=np.int64),
        input_record=record,
        input_sha256=sha256_bytes(canonical_json(record).encode("utf-8")),
    )


def _train(
    family,
    train,
    validation,
    config,
    checkpoint_path,
    log_callback=None,
    *,
    progress_checkpoint_path,
):
    del family, train, validation, log_callback
    Path(checkpoint_path).write_text(
        json.dumps(
            campaign.v1._canonical_training_config_record(
                config, field="fixture training config"
            ),
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    Path(progress_checkpoint_path).unlink(missing_ok=True)


def _load(path: Path, *, device: str):
    del device
    config = json.loads(path.read_text(encoding="utf-8"))
    total = 128000 * 256
    return SimpleNamespace(
        config=config,
        checkpoint_sha256=sha256_path(path),
        history=(
            {
                "step": 128000,
                "examples_seen": total,
                "train_loss": 0.5,
                "validation_loss": 0.4,
                "learning_rate": 0.0002,
            },
        ),
        weights_metadata={
            "schema_version": 1,
            "selection": "minimum_train_selection_native_loss_v1",
            "initial": {
                "step": 0,
                "examples_seen": 0,
                "validation_loss": 1.0,
            },
            "selected": {
                "kind": "validation_best",
                "step": 128000,
                "examples_seen": total,
                "state_sha256": "1" * 64,
            },
            "final": {
                "kind": "final",
                "step": 128000,
                "examples_seen": total,
                "state_sha256": "2" * 64,
            },
        },
    )


def _constant_predict(
    trained,
    query,
    scale,
    *,
    family,
    readout,
    divergence_backend,
    trace_probes,
    trace_seed,
    batch_size,
):
    del (
        trained,
        scale,
        family,
        readout,
        divergence_backend,
        trace_probes,
        trace_seed,
        batch_size,
    )
    return np.ones(np.asarray(query).shape[0], dtype=np.float64)


def _run(
    root: Path,
    config,
    plan,
    cell,
    *,
    reference_summary=None,
):
    return campaign._run_cell(
        campaign_root=root,
        campaign_id=config["campaign"]["campaign_id"],
        campaign_config_sha="3" * 64,
        source_sha="4" * 64,
        config=config,
        model_plan=plan,
        cell=cell,
        data=_data(cell),
        reference_summary=reference_summary,
        train_fn=_train,
        predict_fn=_constant_predict,
        load_checkpoint_fn=_load,
        affine_diagnostics_fn=lambda *args, **kwargs: {},
        callback=None,
    )


@pytest.mark.parametrize("variant", campaign.APPROVED_MODEL_VARIANTS)
@pytest.mark.parametrize("target_policy", ["sample_size", "paired_delta"])
def test_no_knee_is_sealed_and_propagated_without_surrogate_predictions(
    tmp_path: Path,
    monkeypatch,
    variant,
    target_policy,
) -> None:
    from models import training

    monkeypatch.setattr(
        training,
        "predict_nf_lid_ols5",
        lambda trained, query, scale, **kwargs: np.ones(len(query)),
    )
    config = campaign.validate_global_campaign_config(
        campaign.compose_global_campaign_v2_config()
    )
    plan = next(
        item for item in campaign.model_plans(config) if item.variant_id == variant
    )
    reference = replace(
        _cell("reference", "reference"),
        target_policy=target_policy,
        suite_id="e1" if target_policy == "sample_size" else "e5",
    )
    reference_dir, reference_summary = _run(tmp_path, config, plan, reference)

    selection = reference_summary["protocols"][campaign.UNKNOWN_REFERENCE_PROTOCOL][
        "selection"
    ]
    assert selection["status"] == "selection_failed"
    assert selection["failure_reason"] == "no_knee"
    assert reference_summary["selected_index"] is None
    assert reference_summary["selected_scale"] is None
    assert not list(reference_dir.glob("*_prediction__*__reference.npy"))
    assert not (reference_dir / "checkpoint.pt").exists()
    assert campaign.validate_global_cell(reference_dir) == []

    dependent = replace(
        _cell("dependent", "reference"),
        target_policy=target_policy,
        suite_id=reference.suite_id,
    )
    dependent_dir, dependent_summary = _run(
        tmp_path,
        config,
        plan,
        dependent,
        reference_summary=reference_summary,
    )
    dependent_selection = dependent_summary["protocols"][
        campaign.UNKNOWN_REFERENCE_PROTOCOL
    ]["selection"]
    assert dependent_selection["status"] == "selection_failed"
    assert dependent_selection["failure_reason"] == "reference_selection_failed"
    assert (
        dependent_summary["reference_binding"]["summary_sha256"]
        == reference_summary["summary_sha256"]
    )
    assert not list(dependent_dir.glob("*_prediction__*__reference.npy"))
    assert (
        campaign.validate_global_cell(
            dependent_dir, reference_summary=reference_summary
        )
        == []
    )


def test_aggregate_and_unified_table_preserve_selection_failed_rows(
    tmp_path: Path,
) -> None:
    config = campaign.validate_global_campaign_config(
        campaign.compose_global_campaign_v2_config()
    )
    cells = campaign.load_campaign_inventory(
        config, Path(__file__).resolve().parents[2]
    )
    directories: dict[str, Path] = {}
    summaries: dict[str, dict] = {}
    for ordinal, cell in enumerate(cells):
        directory = tmp_path / f"cell-{ordinal}"
        directory.mkdir()
        directories[cell.key] = directory
        if cell.target_policy == "known_lid":
            target = np.asarray([2.0, 3.0])
            prediction = target + 0.25
            fallback = np.asarray([0, 1], dtype=np.uint8)
            knee_time = np.asarray([0.2, -1.0])
            np.save(directory / "validation_target.npy", target)
            np.save(directory / "test_target.npy", target)
            for split in ("validation", "test"):
                np.save(
                    directory / f"{split}_prediction__full__supervised.npy", prediction
                )
                np.save(
                    directory / f"{split}_prediction__full__kneedle.npy", prediction
                )
                np.save(directory / f"{split}_knee_fallback__full.npy", fallback)
                np.save(directory / f"{split}_knee_time__full.npy", knee_time)
            metric_a = known_lid_metrics(prediction, target)
            metric_b = {
                **metric_a,
                "no_knee_n": 1,
                "no_knee_fraction": 0.5,
            }
            summary = {
                "model_variant": "ve_diffusion",
                "cell_id": f"known-{ordinal}",
                "primary_readout": "full",
                "selected_index": 2,
                "selected_scale": 0.5,
                "partition": {"n_source_train": 12},
                "protocols": {
                    campaign.KNOWN_SUPERVISED_PROTOCOL: {
                        "selected_index": 2,
                        "selected_lambda": 0.5,
                        "selection": {"status": "selected"},
                    }
                },
                "metrics": {
                    campaign.KNOWN_SUPERVISED_PROTOCOL: {
                        "validation": {"full": metric_a},
                        "test": {"full": metric_a},
                    },
                    campaign.KNOWN_KNEEDLE_PROTOCOL: {
                        "validation": {"full": metric_b},
                        "test": {"full": metric_b},
                    },
                },
            }
        else:
            reference_key = (
                f"{cell.suite_id}/{cell.reference_dataset}/{cell.representation}"
            )
            self_reference = cell.reference_dataset in {None, cell.dataset}
            failure_reason = (
                "no_knee" if self_reference else "reference_selection_failed"
            )
            binding = None
            if not self_reference:
                binding = {
                    "cell_key": reference_key,
                    "summary_sha256": sha256_path(
                        directories[reference_key] / "summary.json"
                    ),
                }
            summary = {
                "model_variant": "ve_diffusion",
                "cell_id": f"failed-{ordinal}",
                "primary_readout": "full",
                "selected_index": None,
                "selected_scale": None,
                "reference_binding": binding,
                "partition": {"n_source_train": 12},
                "protocols": {
                    campaign.UNKNOWN_REFERENCE_PROTOCOL: {
                        "selection": {
                            "status": "selection_failed",
                            "failure_reason": failure_reason,
                        }
                    }
                },
                "metrics": {},
            }
            np.save(directory / "validation_labels.npy", np.arange(2))
            np.save(directory / "test_labels.npy", np.arange(2))
            campaign._write_json(
                directory / "manifest.json",
                {
                    "identity": {
                        "model": {
                            "model": {
                                "family": "diffusion",
                                "primary_readout": "full",
                            }
                        }
                    }
                },
            )
        campaign._write_json(directory / "summary.json", summary)
        summaries[cell.key] = summary

    aggregate = campaign.recompute_model_aggregate("ve_diffusion", cells, directories)
    assert aggregate["physical_training_count"] == 39
    assert aggregate["coverage"]["canonical_cells"] == 35
    failed = aggregate["e1_sample_size_stability"] + aggregate["e5_paired_delta"]
    assert failed
    assert all(row["selection_status"] == "selection_failed" for row in failed)
    table = campaign.render_unified_results_csv(
        campaign._campaign_aggregate("fixture", [aggregate])
    )
    assert "selection_failed" in table
    assert "reference_selection_failed" in table
    table_rows = list(csv.DictReader(io.StringIO(table)))
    failed_e1 = [
        row
        for row in table_rows
        if row["analysis"] == "e1_sample_size_stability"
        and row["selection_status"] == "selection_failed"
    ]
    assert failed_e1
    assert {row["n_source_train"] for row in failed_e1} == {"12"}
    macros = campaign._aggregate_macros(aggregate)
    assert macros["e1_test_mean_absolute_mean_delta_error"] is None
    assert macros["e5_test_mean_paired_delta_mae"] is None
