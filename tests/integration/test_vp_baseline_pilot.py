from pathlib import Path
import subprocess
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import numpy as np
import pytest

pytest.importorskip("torch")

from experiments.vp_baseline_pilot import file_sha, run, validate_config
from experiments.vp_pilot_audit import validate_run


def config():
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(version_base=None, config_dir=str(root / "configs")):
        return OmegaConf.to_container(
            compose(config_name="vp_baseline_pilot"), resolve=True
        )


def test_hydra_and_closed_config():
    value = config()
    validate_config(value)
    value["training"]["early_stopping"] = True
    with pytest.raises(ValueError):
        validate_config(value)
    value = config()
    value["evaluation"]["test_mae_selection"] = True
    with pytest.raises(ValueError):
        validate_config(value)


@pytest.mark.parametrize("module", ["vp_baseline_pilot", "vp_pilot_audit"])
def test_cli_discovers_yaml_without_running(module):
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-m", f"experiments.{module}", "--cfg", "job"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    config_value = OmegaConf.create(result.stdout)
    assert isinstance(config_value, type(OmegaConf.create({})))
    assert config_value


def test_small_cpu_run_preserves_split_and_artifacts(tmp_path):
    import json

    value = config()
    value.update(
        device="cpu", output_root=str(tmp_path), dataset="rank3_gaussian_control"
    )
    value["model"].update(hidden_sizes=[16, 8, 4], time_dim=8)
    value["training"].update(
        steps=4, warmup_steps=1, validation_interval=2, batch_size=8
    )
    value["data"].update(source_train_count=40, selection_count=8, evaluation_count=8)
    value["evaluation"].update(batch_size=8, grid_count=3)
    summary = run(value)
    directory = tmp_path / f"rank3_gaussian_control-seed-{value['seed']}"
    source = np.load(directory / "source_train.npz", allow_pickle=False)
    assert len(np.intersect1d(source["fit_indices"], source["selection_indices"])) == 0
    assert summary["examples_seen"] == 32
    assert summary["selected"]["test_accessed_at_freeze"] is False
    if summary["selected"]["boundary"]:
        assert not (directory / "test.npz").exists()
        assert not summary["evaluation"]
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, digest in manifest["outputs_sha256"].items():
        assert file_sha(directory / name) == digest
    assert validate_run(directory)["verified"]
    with pytest.raises(FileExistsError):
        run(value)
