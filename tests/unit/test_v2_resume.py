from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import global_campaign_v2 as campaign
from experiments import global_parallel, v2_resume
from experiments.global_parallel_v2 import _v2_api, with_h100_v2_profile
from experiments.run_manifest import sha256_bytes, sha256_path

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def imported(tmp_path):
    config = with_h100_v2_profile(campaign.compose_global_campaign_v2_config())
    cells = campaign.load_campaign_inventory(config, ROOT)
    plans = campaign.model_plans(config)
    prepared = SimpleNamespace(
        campaign_root=str(tmp_path),
        state_dir=str(tmp_path / "state"),
        campaign_id=config["campaign"]["campaign_id"],
        campaign_identity="b" * 64,
        config_sha="c" * 64,
        source_sha="d" * 64,
        input_inventory_sha="e" * 64,
        project_root=str(ROOT),
        config=config,
        cells=cells,
        plans=plans,
        preflight_inputs={
            cell.key: {
                "input_sha256": "f" * 64,
                "input_record": {"feature_shape": [30]},
            }
            for cell in cells
        },
    )
    pairs = set(campaign.CANARY_REUSE_TASKS)
    for plan in plans:
        for cell in cells:
            if len(pairs) < 147:
                pairs.add((plan.variant_id, cell.key))
    original = SimpleNamespace(**vars(prepared))
    original.source_sha = v2_resume.PREDECESSOR_SOURCE
    rows = []
    for m, plan in enumerate(plans):
        for c, cell in enumerate(cells):
            if (plan.variant_id, cell.key) not in pairs:
                continue
            directory, identity = campaign._expected_cell_directory(
                campaign_root=tmp_path,
                prepared=original,
                model_index=m,
                cell_index=c,
                allow_reuse=False,
            )
            directory.mkdir(parents=True)
            (directory / "manifest.json").write_text(json.dumps({"identity": identity}))
            (directory / "summary.json").write_text(json.dumps({"fixture": cell.key}))
            rows.append(
                {
                    "model_variant": plan.variant_id,
                    "cell_key": cell.key,
                    "path": directory.relative_to(tmp_path).as_posix(),
                    "manifest_sha256": sha256_path(directory / "manifest.json"),
                    "summary_sha256": sha256_path(directory / "summary.json"),
                }
            )
    lineage = {
        "schema_version": 1,
        "campaign_identity": prepared.campaign_identity,
        "source_tree_sha256": prepared.source_sha,
        "config_sha256": prepared.config_sha,
        "input_inventory_sha256": prepared.input_inventory_sha,
        "predecessor_campaign_identity": v2_resume.PREDECESSOR_IDENTITY,
        "predecessor_source_sha256": v2_resume.PREDECESSOR_SOURCE,
        "predecessor_root": "/fixture",
        "compatibility": {},
        "canary_report_sha256": "0" * 64,
        "reused_cells": rows,
    }
    campaign._write_json(tmp_path / v2_resume.LINEAGE_FILENAME, lineage)
    return prepared, lineage


def test_all_429_directories_keep_original_source_only_for_147_imports(imported):
    prepared, lineage = imported
    source_counts = {}
    root = Path(prepared.campaign_root)
    for m in range(11):
        for c in range(39):
            directory, identity = campaign._expected_cell_directory(
                campaign_root=root,
                prepared=prepared,
                model_index=m,
                cell_index=c,
            )
            with _v2_api():
                assert global_parallel._expected_cell(
                    prepared,
                    model_index=m,
                    cell_index=c,
                ) == (directory, identity)
            source = identity["source_tree_sha256"]
            source_counts[source] = source_counts.get(source, 0) + 1
    assert source_counts == {
        v2_resume.PREDECESSOR_SOURCE: 147,
        prepared.source_sha: 282,
    }
    assert v2_resume.load_lineage(root) == lineage


@pytest.mark.parametrize(
    "mutation", ["duplicate", "missing", "wrong_source", "path", "hash"]
)
def test_reuse_rejects_tampered_coverage_identity_or_binding(imported, mutation):
    prepared, original = imported
    lineage = copy.deepcopy(original)
    if mutation == "duplicate":
        lineage["reused_cells"][-1] = lineage["reused_cells"][0]
    elif mutation == "missing":
        lineage["reused_cells"].pop()
    elif mutation == "wrong_source":
        lineage["predecessor_source_sha256"] = "1" * 64
    elif mutation == "path":
        lineage["reused_cells"][0]["path"] = "../../foreign"
    else:
        lineage["reused_cells"][0]["manifest_sha256"] = "2" * 64
    root = Path(prepared.campaign_root)
    campaign._write_json(root / v2_resume.LINEAGE_FILENAME, lineage)
    with pytest.raises(campaign.GlobalCampaignError):
        campaign._expected_cell_directory(
            campaign_root=root,
            prepared=prepared,
            model_index=0,
            cell_index=0,
        )


def test_canary_reuse_checks_original_commit_and_unchanged_inputs(
    imported, monkeypatch
):
    from experiments import v2_canary

    prepared, _ = imported
    captured = {}

    def validate(path, **expected):
        captured.update(expected)
        return {"status": "passed"}

    monkeypatch.setattr(v2_canary, "load_and_validate_canary_report", validate)
    assert (
        v2_resume.load_predecessor_canary(Path("fixture"), prepared)["status"]
        == "passed"
    )
    assert captured["expected_commit"] == v2_resume.PREDECESSOR_COMMIT
    assert captured["expected_campaign_identity"] == v2_resume.PREDECESSOR_IDENTITY
    assert captured["expected_declared_source_sha256"] == v2_resume.PREDECESSOR_SOURCE
    assert captured["expected_campaign_config_sha256"] == prepared.config_sha
    assert captured["expected_input_inventory_sha256"] == prepared.input_inventory_sha


@pytest.mark.parametrize("dimension", [30, 256, 784, 900, 1024])
def test_all_11_variants_accept_every_production_scale_grid(dimension):
    config = campaign.validate_global_campaign_config(
        campaign.compose_global_campaign_v2_config()
    )
    initial = campaign.initial_supervised_lambdas()
    grids = [
        initial,
        campaign.unknown_reference_lambdas(),
        campaign.vp_lambda_from_time(campaign.flipd_timesteps()),
    ]
    for sign in [-1, 1]:
        grids.append(
            campaign.select_supervised_bounded(
                initial,
                np.tile(initial**sign, (4, 1)),
                np.zeros(4),
                evaluate=lambda x, power=sign: np.tile(x**power, (4, 1)),
            )[0]
        )
    grid = np.unique(
        np.concatenate(
            grids + [[campaign.COMMON_LAMBDA_MIN, campaign.COMMON_LAMBDA_MAX]]
        )
    )
    for plan in campaign.model_plans(config):
        model = campaign.resolve_cell_model(plan, {"feature_shape": [dimension]})
        campaign._assert_lambdas_supported(model, grid, include_nf_stencil=True)
        for scale in grid:
            assert np.isfinite(campaign._native_coordinate(model, scale))


def test_maintenance_cannot_change_scientific_sources(monkeypatch):
    def git(root, *args):
        if args[0] == "status":
            return ""
        return "experiments/fm_diagnostics.py\nmodels/training.py"

    monkeypatch.setattr(v2_resume, "_git", git)
    with pytest.raises(campaign.GlobalCampaignError, match="unreviewed maintenance"):
        v2_resume.source_compatibility(ROOT)


def test_lineage_rejects_changed_config_before_accepting_canary(imported, monkeypatch):
    prepared, _ = imported
    monkeypatch.setattr(v2_resume, "source_compatibility", lambda root: {})
    # Matching superficial pins still cannot import another scientific identity.
    with pytest.raises(
        campaign.GlobalCampaignError, match="changed scientific inputs/config"
    ):
        v2_resume.validate_lineage(Path(prepared.campaign_root), prepared)


def test_lineage_hash_is_in_final_manifest(imported, monkeypatch):
    prepared, _ = imported
    monkeypatch.setattr(campaign, "_load_bound_canary_report", lambda *args: {})
    root = Path(prepared.campaign_root)
    (root / "canary_report.json").write_text("{}")
    extras = campaign.final_manifest_extras(campaign_root=root, prepared=prepared)
    assert extras["resume_lineage"] == {
        "path": v2_resume.LINEAGE_FILENAME,
        "sha256": sha256_bytes((root / v2_resume.LINEAGE_FILENAME).read_bytes()),
        "reused_cell_count": 147,
        "new_training_count": 282,
    }
