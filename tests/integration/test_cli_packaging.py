from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import zipfile

from experiments.cli import _default_config_dir


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_config_discovery_supports_source_and_installed_layouts(tmp_path: Path) -> None:
    assert _default_config_dir() == REPOSITORY_ROOT / "configs"

    package = tmp_path / "site-packages" / "experiments"
    packaged_configs = package / "configs"
    packaged_configs.mkdir(parents=True)
    (package / "cli.py").touch()
    (packaged_configs / "config.yaml").touch()
    assert _default_config_dir(package / "cli.py") == packaged_configs


def test_built_wheel_contains_and_discovers_all_hydra_yaml(tmp_path: Path) -> None:
    wheel_dir = tmp_path / "wheel"
    process = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(wheel_dir)],
        cwd=REPOSITORY_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert process.returncode == 0, process.stdout
    wheel = next(wheel_dir.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        for config in (REPOSITORY_ROOT / "configs").rglob("*.yaml"):
            relative = config.relative_to(REPOSITORY_ROOT / "configs").as_posix()
            assert f"experiments/configs/{relative}" in names
        archive.extractall(tmp_path / "installed")

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(tmp_path / "installed")
    installed_cli = subprocess.run(
        [sys.executable, "-m", "experiments.cli", "--cfg", "job"],
        cwd=tmp_path,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert installed_cli.returncode == 0, installed_cli.stdout
    assert "datasets:" in installed_cli.stdout
