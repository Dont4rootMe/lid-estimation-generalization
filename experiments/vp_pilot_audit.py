"""Read-only validation of completed VP pilot inputs and result arrays.

Writes only a small derived report/plot. It does not train, change the selected
scales, modify run directories or inspect new test curves. The report keeps
the original producing source hashes, even when current source has changed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from datasets.vp_pilot import PILOT_LIDS, sample_vp_fixture


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_run(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    expected = set(manifest["outputs_sha256"]) | {"manifest.json"}
    if {p.name for p in directory.iterdir() if p.is_file()} != expected:
        raise ValueError("pilot output inventory mismatch")
    for name, digest in manifest["outputs_sha256"].items():
        if Path(name).name != name or sha256(directory / name) != digest:
            raise ValueError(f"pilot output hash mismatch: {name}")
    summary = json.loads((directory / "summary.json").read_text())
    config = OmegaConf.to_container(
        OmegaConf.load(directory / "resolved_config.yaml"), resolve=True
    )
    data = np.load(directory / "source_train.npz", allow_pickle=False)
    regenerated = sample_vp_fixture(
        config["dataset"], config["data"]["source_train_count"], config["seed"]
    )
    np.testing.assert_array_equal(data["source"], regenerated)
    fit, held = data["fit_indices"], data["selection_indices"]
    np.testing.assert_array_equal(
        np.sort(np.concatenate((fit, held))), np.arange(len(regenerated))
    )
    mean = regenerated[fit].mean(axis=0, dtype=np.float64)
    rms = np.sqrt(np.mean((regenerated[fit].astype(np.float64) - mean) ** 2))
    np.testing.assert_array_equal(mean, data["mean"])
    np.testing.assert_allclose(rms, data["rms"], rtol=0, atol=0)
    curve = np.load(directory / "selection_curve.npz", allow_pickle=False)
    truth = PILOT_LIDS[config["dataset"]]
    np.testing.assert_array_equal(curve["target"], np.full(len(held), truth))
    np.testing.assert_allclose(
        curve["full"], curve["response"] + curve["correction"], rtol=1e-12, atol=1e-11
    )
    maes = np.abs(curve["full"] - truth).mean(axis=0)
    selected = int(np.flatnonzero(maes <= maes.min() + 1e-12)[0])
    if selected != summary["selected"]["index"]:
        raise ValueError("frozen selection is not the declared MAE winner")
    if (
        summary["examples_seen"]
        != config["training"]["steps"] * config["training"]["batch_size"]
    ):
        raise ValueError("examples-seen accounting mismatch")
    for offset, split in ((10, "validation"), (11, "test")):
        if summary["selected"]["boundary"]:
            if (directory / f"{split}.npz").exists():
                raise ValueError("boundary-unresolved run evaluated a final split")
            continue
        arrays = np.load(directory / f"{split}.npz", allow_pickle=False)
        np.testing.assert_array_equal(
            arrays["raw"],
            sample_vp_fixture(
                config["dataset"],
                config["data"]["evaluation_count"],
                config["seed"] + offset,
            ),
        )
        np.testing.assert_array_equal(
            arrays["target"], np.full(len(arrays["raw"]), truth)
        )
        measured = float(np.abs(arrays["full"] - truth).mean())
        if not np.isclose(
            measured, summary["evaluation"][split]["mae"], rtol=0, atol=1e-12
        ):
            raise ValueError("reported LID MAE differs from pointwise predictions")
    return {
        "run": directory.as_posix(),
        "manifest_sha256": sha256(directory / "manifest.json"),
        "producing_source_sha256": manifest["source_sha256"],
        "configuration": config,
        "summary": summary,
        "verified": True,
        "current_fixture_matches_saved_inputs_bitwise": True,
    }


def plot_runs(records: list[dict], destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex="col")
    names = ("rank3_gaussian_control", "e7_crescent_moon_radius3.0")
    for column, name in enumerate(names):
        for record in records:
            if record["summary"]["dataset"] != name:
                continue
            curve = np.load(
                Path(record["run"]) / "selection_curve.npz", allow_pickle=False
            )
            values, lam = curve["full"], curve["lambdas"]
            label = f"{record['configuration']['training']['steps'] // 1000}k steps"
            axes[0, column].plot(lam, values.mean(axis=0), label=label)
            axes[1, column].plot(lam, np.abs(values - 3).mean(axis=0), label=label)
        axes[0, column].axhline(3, color="black", linestyle=":", label="true d = 3")
        axes[0, column].set_ylim(-0.5, 8)
        axes[0, column].set_title(
            "Gaussian control" if column == 0 else "Crescent Moon"
        )
        axes[0, column].set_ylabel("Mean LID (zoom near d=3)")
        axes[1, column].set_ylabel("Pointwise MAE")
        axes[1, column].set_yscale("log")
        axes[1, column].set_xlabel("Physical noise ratio lambda")
        for row in range(2):
            axes[row, column].set_xscale("log", base=2)
            axes[row, column].grid(alpha=0.2)
        axes[0, column].legend()
    fig.suptitle("Optimizer-disjoint train-selection curves; not test curves")
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


@hydra.main(version_base="1.3", config_path=None, config_name="vp_pilot_audit")
def _hydra_main(config: DictConfig) -> None:
    if set(config) != {"runs", "output_dir"}:
        raise ValueError("unexpected audit configuration fields")
    records = [validate_run(Path(path)) for path in config.runs]
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "evidence": "exploratory_regenerated_VP_pilots",
        "note": "Budget follow-up uses the same seed and pilot datasets. Cosine horizon also changes; not a pure step-count intervention or a new confirmatory test.",
        "runs": records,
    }
    (output / "vp_pilot_20260905.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    plot_runs(records, output / "vp_pilot_curves.png")
    print(
        json.dumps(
            {
                "verified_runs": len(records),
                "report": str(output / "vp_pilot_20260905.json"),
            }
        )
    )


def main() -> None:
    """Discover YAML with the same source/wheel contract as the main CLI."""
    from experiments.cli import _default_config_dir

    has_config_dir = any(
        argument == "--config-dir" or argument.startswith("--config-dir=")
        for argument in sys.argv[1:]
    )
    if not has_config_dir:
        sys.argv[1:1] = ["--config-dir", str(_default_config_dir())]
    _hydra_main()


if __name__ == "__main__":
    main()
