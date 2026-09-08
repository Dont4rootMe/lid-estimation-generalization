"""Copy validated task-9201 cells unchanged into a diagnostic-maintenance run."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from experiments import global_campaign_v2 as campaign
from experiments.run_manifest import canonical_json, sha256_bytes, sha256_path

PREDECESSOR_COMMIT = "03c12b04561575fbef5c89c6876138bf7d7c2e44"
PREDECESSOR_SOURCE = "87a1be064e10f91ca8a2f547217c44ce01ff0ca8c83d126eb0d78c686e8b3b4d"
PREDECESSOR_IDENTITY = (
    "92fe3436909e3bf28427c380fdaa463e7f652c9d884f8ffe28cdc02cd11ff2b7"
)
REUSED_CELL_COUNT = 147
LINEAGE_FILENAME = "state/resume_lineage.json"
_MAINTENANCE_FILES = {
    "experiments/fm_diagnostics.py",
    "experiments/global_campaign_v2.py",
    "experiments/global_parallel.py",
    "experiments/v2_resume.py",
    "scripts/run_v2_canary.sh",
}


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def source_compatibility(root: Path) -> dict[str, Any]:
    if _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise campaign.GlobalCampaignError("maintenance resume requires clean source")
    changed = _git(root, "diff", "--name-only", PREDECESSOR_COMMIT, "HEAD").splitlines()
    rejected = [
        name
        for name in changed
        if name not in _MAINTENANCE_FILES and not name.startswith(("tests/", "docs/"))
    ]
    if rejected:
        raise campaign.GlobalCampaignError(
            f"unreviewed maintenance changes: {rejected}"
        )
    if "experiments/fm_diagnostics.py" not in changed:
        raise campaign.GlobalCampaignError("diagnostic fix is absent")
    return {
        "predecessor_commit": PREDECESSOR_COMMIT,
        "execution_commit": _git(root, "rev-parse", "HEAD"),
        "changed_files": changed,
        "diff_sha256": sha256_bytes(
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "diff",
                    "--binary",
                    PREDECESSOR_COMMIT,
                    "HEAD",
                ],
                check=True,
                capture_output=True,
            ).stdout
        ),
        "scope": "diagnostic_ratios_and_immutable_cell_import_only",
    }


def load_lineage(root: Path) -> dict[str, Any] | None:
    path = Path(root) / LINEAGE_FILENAME
    if not path.exists():
        return None
    value = campaign._load_json(path)
    if (
        set(value)
        != {
            "schema_version",
            "campaign_identity",
            "source_tree_sha256",
            "config_sha256",
            "input_inventory_sha256",
            "predecessor_campaign_identity",
            "predecessor_source_sha256",
            "predecessor_root",
            "compatibility",
            "canary_report_sha256",
            "reused_cells",
        }
        or value["schema_version"] != 1
    ):
        raise campaign.GlobalCampaignError("invalid maintenance lineage fields")
    if (
        value["predecessor_source_sha256"] != PREDECESSOR_SOURCE
        or value["predecessor_campaign_identity"] != PREDECESSOR_IDENTITY
    ):
        raise campaign.GlobalCampaignError("unapproved maintenance predecessor")
    rows = value["reused_cells"]
    if not isinstance(rows, list) or len(rows) != REUSED_CELL_COUNT:
        raise campaign.GlobalCampaignError("maintenance import must contain 147 cells")
    pairs = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "model_variant",
            "cell_key",
            "path",
            "manifest_sha256",
            "summary_sha256",
        }:
            raise campaign.GlobalCampaignError("invalid maintenance cell record")
        pair = (row["model_variant"], row["cell_key"])
        if (
            pair in pairs
            or pair[0] not in campaign.APPROVED_MODEL_VARIANTS
            or pair[1] not in campaign.APPROVED_GLOBAL_CELL_KEYS
        ):
            raise campaign.GlobalCampaignError("invalid maintenance cell coverage")
        pairs.add(pair)
    if not set(campaign.CANARY_REUSE_TASKS).issubset(pairs):
        raise campaign.GlobalCampaignError("maintenance import lacks original canary")
    return value


def reused_cell(
    prepared: Any, *, campaign_root: Path, model_index: int, cell_index: int
):
    lineage = load_lineage(campaign_root)
    if lineage is None:
        return None
    if (
        lineage["source_tree_sha256"] != prepared.source_sha
        or lineage["config_sha256"] != prepared.config_sha
    ):
        raise campaign.GlobalCampaignError("maintenance target source/config differs")
    plan, cell = prepared.plans[model_index], prepared.cells[cell_index]
    record = next(
        (
            row
            for row in lineage["reused_cells"]
            if (row["model_variant"], row["cell_key"]) == (plan.variant_id, cell.key)
        ),
        None,
    )
    if record is None:
        return None
    original = SimpleNamespace(**vars(prepared))
    original.source_sha = PREDECESSOR_SOURCE
    directory, identity = campaign._expected_cell_directory(
        campaign_root=campaign_root,
        prepared=original,
        model_index=model_index,
        cell_index=cell_index,
        allow_reuse=False,
    )
    if (
        record["path"] != directory.relative_to(campaign_root).as_posix()
        or not (directory / "manifest.json").is_file()
        or sha256_path(directory / "manifest.json") != record["manifest_sha256"]
        or sha256_path(directory / "summary.json") != record["summary_sha256"]
    ):
        raise campaign.GlobalCampaignError("maintenance cell binding differs")
    return directory, identity


def validate_lineage(root: Path, prepared: Any) -> dict[str, Any]:
    lineage = load_lineage(root)
    if lineage is None:
        raise campaign.GlobalCampaignError("maintenance lineage is missing")
    for field, expected in {
        "campaign_identity": prepared.campaign_identity,
        "source_tree_sha256": prepared.source_sha,
        "config_sha256": prepared.config_sha,
        "input_inventory_sha256": prepared.input_inventory_sha,
    }.items():
        if lineage[field] != expected:
            raise campaign.GlobalCampaignError(f"maintenance {field} differs")
    if lineage["compatibility"] != source_compatibility(Path(prepared.project_root)):
        raise campaign.GlobalCampaignError("maintenance compatibility proof differs")
    predecessor_identity = sha256_bytes(
        canonical_json(
            {
                "schema_version": campaign.GLOBAL_CAMPAIGN_SCHEMA_VERSION,
                "campaign_id": prepared.campaign_id,
                "config_sha256": prepared.config_sha,
                "source_tree_sha256": PREDECESSOR_SOURCE,
                "input_inventory_sha256": prepared.input_inventory_sha,
            }
        ).encode()
    )
    if predecessor_identity != PREDECESSOR_IDENTITY:
        raise campaign.GlobalCampaignError(
            "maintenance changed scientific inputs/config"
        )
    if sha256_path(root / "canary_report.json") != lineage["canary_report_sha256"]:
        raise campaign.GlobalCampaignError("original canary report changed")
    return lineage


def load_predecessor_canary(path: Path, prepared: Any):
    from experiments.v2_canary import load_and_validate_canary_report

    return load_and_validate_canary_report(
        path,
        expected_commit=PREDECESSOR_COMMIT,
        expected_declared_source_sha256=PREDECESSOR_SOURCE,
        expected_campaign_identity=PREDECESSOR_IDENTITY,
        expected_campaign_config_sha256=prepared.config_sha,
        expected_input_inventory_sha256=prepared.input_inventory_sha,
        expected_config_sha256=sha256_path(
            Path(prepared.project_root) / "configs/v2_canary.yaml"
        ),
    )


def import_sealed_cells(prepared: Any, predecessor_root: Path) -> dict[str, Any]:
    root = Path(prepared.campaign_root)
    predecessor_root = predecessor_root.resolve()
    if predecessor_root == root or predecessor_root in root.parents:
        raise campaign.GlobalCampaignError(
            "maintenance requires a separate output root"
        )
    if load_lineage(root) is not None:
        return validate_lineage(root, prepared)
    if any(root.glob("runs/**/identity.json")):
        raise campaign.GlobalCampaignError("maintenance destination already has cells")
    compatibility = source_compatibility(Path(prepared.project_root))
    report = load_predecessor_canary(predecessor_root / "canary_report.json", prepared)
    if not campaign._same_json(
        campaign._load_json(predecessor_root / "input_inventory.json"),
        campaign._load_json(root / "input_inventory.json"),
    ):
        raise campaign.GlobalCampaignError("predecessor input inventory differs")
    original = SimpleNamespace(**vars(prepared))
    original.source_sha = PREDECESSOR_SOURCE
    original.campaign_identity = PREDECESSOR_IDENTITY
    rows = []
    sources = []
    for model_index, plan in enumerate(prepared.plans):
        summaries = {}
        for cell_index, cell in enumerate(prepared.cells):
            directory, identity = campaign._expected_cell_directory(
                campaign_root=predecessor_root,
                prepared=original,
                model_index=model_index,
                cell_index=cell_index,
                allow_reuse=False,
            )
            if not directory.exists():
                continue
            reference = None
            if cell.reference_dataset not in {None, cell.dataset}:
                reference = summaries.get(
                    campaign._reference_cell(prepared.cells, cell).key
                )
                if reference is None:
                    raise campaign.GlobalCampaignError(
                        "predecessor dependency is missing"
                    )
            errors = campaign.validate_global_cell(
                directory,
                expected_identity=identity,
                reference_summary=reference,
                expected_source_evidence=prepared.preflight_inputs[cell.key][
                    "source_evidence"
                ],
            )
            if errors:
                raise campaign.GlobalCampaignError(
                    f"invalid predecessor {directory}: {errors}"
                )
            summary = campaign._load_json(directory / "summary.json")
            summary["summary_sha256"] = sha256_path(directory / "summary.json")
            summaries[cell.key] = summary
            rows.append(
                {
                    "model_variant": plan.variant_id,
                    "cell_key": cell.key,
                    "path": directory.relative_to(predecessor_root).as_posix(),
                    "manifest_sha256": sha256_path(directory / "manifest.json"),
                    "summary_sha256": summary["summary_sha256"],
                }
            )
            sources.append(directory)
    if len(rows) != REUSED_CELL_COUNT:
        raise campaign.GlobalCampaignError(
            f"expected 147 sealed predecessors, found {len(rows)}"
        )
    for directory, row in zip(sources, rows, strict=True):
        shutil.copytree(directory, root / row["path"])
    for record in report["outputs"]:
        source = predecessor_root / record["path"]
        destination = root / record["path"]
        if destination.exists():
            if sha256_path(destination) != record["sha256"]:
                raise campaign.GlobalCampaignError("copied canary artifact differs")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    shutil.copy2(predecessor_root / "canary_report.json", root / "canary_report.json")
    lineage = {
        "schema_version": 1,
        "campaign_identity": prepared.campaign_identity,
        "source_tree_sha256": prepared.source_sha,
        "config_sha256": prepared.config_sha,
        "input_inventory_sha256": prepared.input_inventory_sha,
        "predecessor_campaign_identity": PREDECESSOR_IDENTITY,
        "predecessor_source_sha256": PREDECESSOR_SOURCE,
        "predecessor_root": str(predecessor_root),
        "compatibility": compatibility,
        "canary_report_sha256": sha256_path(root / "canary_report.json"),
        "reused_cells": rows,
    }
    campaign._write_json(root / LINEAGE_FILENAME, lineage)
    validate_lineage(root, prepared)
    campaign.validate_pre_run_gate(campaign_root=root, prepared=prepared)
    return lineage


def replay_failed_diagnostics(prepared: Any, predecessor_root: Path) -> None:
    from models.training import load_checkpoint, predict_lid

    model_index = campaign.APPROVED_MODEL_VARIANTS.index("direct_rectified_flow")
    cell_index = campaign.APPROVED_GLOBAL_CELL_KEYS.index(
        "e3/e3_gaussian_pca/coefficients"
    )
    original = SimpleNamespace(**vars(prepared))
    original.source_sha = PREDECESSOR_SOURCE
    old_final, identity = campaign._expected_cell_directory(
        campaign_root=predecessor_root,
        prepared=original,
        model_index=model_index,
        cell_index=cell_index,
        allow_reuse=False,
    )
    old_work = old_final.parent / f".{old_final.name}.incomplete"
    checkpoint = old_work / "checkpoint.pt"
    if campaign._load_json(old_work / "identity.json") != identity:
        raise campaign.GlobalCampaignError("failed checkpoint identity differs")
    checkpoint_sha = sha256_path(checkpoint)
    if (
        campaign._load_json(old_work / "training_attestation.json")["checkpoint_sha256"]
        != checkpoint_sha
    ):
        raise campaign.GlobalCampaignError("failed checkpoint SHA differs")
    cell, plan = prepared.cells[cell_index], prepared.plans[model_index]
    data = campaign._bind_source_preflight(
        campaign.load_campaign_cell_data(
            cell, prepared.config, Path(prepared.project_root)
        ),
        cell,
        prepared.source_records,
    )
    if data.input_sha256 != prepared.preflight_inputs[cell.key]["input_sha256"]:
        raise campaign.GlobalCampaignError("failed-cell replay input changed")
    partition = campaign.partition_source_train(
        data.train,
        data.train_target,
        selection=prepared.config["campaign"]["selection"],
        seed=prepared.config["seed"],
    )
    model = campaign.resolve_cell_model(plan, data.input_record)
    trained = load_checkpoint(checkpoint, device="cuda:0")
    campaign.v1._require_matching_training_configs(trained.config, model["training"])
    scales = campaign.initial_supervised_lambdas()
    curve = campaign._prediction_curve(
        predict_lid,
        trained,
        partition.selection_features,
        scales,
        model=model,
        seed=prepared.config["seed"],
        batch_size=512,
        readout=model["primary_readout"],
    )
    output = Path(prepared.state_dir) / "maintenance_replay" / "fm_diagnostics"
    evidence = campaign.run_known_affine_diagnostics(
        output,
        trained=trained,
        partition=partition,
        scales=scales,
        selection_curve=curve,
        model=model,
    )
    campaign._write_json(
        output.parent / "replay.json",
        {
            "schema_version": 1,
            "status": "passed",
            "scope": "original_failed_checkpoint_train_selection_diagnostics_only",
            "original_cell_identity": identity,
            "checkpoint_sha256": checkpoint_sha,
            "execution_source_sha256": prepared.source_sha,
            "evidence": evidence,
            "is_benchmark_result": False,
        },
    )
    print(
        json.dumps(
            {
                "status": "failed_checkpoint_replay_passed",
                "cell": cell.key,
                "checkpoint_sha256": checkpoint_sha,
            }
        ),
        flush=True,
    )


def main() -> None:
    from experiments.global_parallel_v2 import prepare_global_v2_campaign
    from experiments.v2_canary import _assert_expected_identity

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-campaign", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--replay-failed-diagnostics", action="store_true")
    arguments = parser.parse_args()
    prepared = prepare_global_v2_campaign(output_root=arguments.output_root)
    for name, value in {
        "CAMPAIGN_IDENTITY": prepared.campaign_identity,
        "CAMPAIGN_CONFIG_SHA256": prepared.config_sha,
        "INPUT_INVENTORY_SHA256": prepared.input_inventory_sha,
        "DECLARED_SOURCE_SHA256": prepared.source_sha,
    }.items():
        _assert_expected_identity("LID_V2_EXPECTED_" + name, value)
    lineage = import_sealed_cells(prepared, arguments.from_campaign)
    print(
        json.dumps(
            {
                "status": "validated_import",
                "reused_cells": len(lineage["reused_cells"]),
                "remaining_cells": campaign.EXPECTED_PHYSICAL_TRAININGS
                - len(lineage["reused_cells"]),
                "campaign_root": prepared.campaign_root,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if arguments.replay_failed_diagnostics:
        replay_failed_diagnostics(prepared, arguments.from_campaign)


if __name__ == "__main__":
    main()
