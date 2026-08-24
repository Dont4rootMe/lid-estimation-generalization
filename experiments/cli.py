"""Hydra-only experiment entry point."""

from __future__ import annotations

from pathlib import Path
import sys

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from experiments.runner import run_experiment


def _default_config_dir(cli_file: Path | None = None) -> Path:
    """Locate Hydra YAML in an installed wheel or a source checkout."""

    module_file = (Path(__file__) if cli_file is None else cli_file).resolve()
    candidates = (
        module_file.parent / "configs",
        module_file.parents[1] / "configs",
    )
    for candidate in candidates:
        if (candidate / "config.yaml").is_file():
            return candidate
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Hydra config directory not found; checked: {checked}")


@hydra.main(version_base="1.3", config_path=None, config_name="config")
def _hydra_main(config: DictConfig) -> None:
    hydra_output = Path(HydraConfig.get().runtime.output_dir)
    overrides = HydraConfig.get().overrides.task
    matrix_dir = run_experiment(
        config,
        output_root=hydra_output / "results",
        hydra_overrides=overrides,
    )
    print(matrix_dir)


def main() -> None:
    """Run Hydra with the bundled or source-checkout YAML config directory."""

    has_config_dir = any(
        argument == "--config-dir" or argument.startswith("--config-dir=")
        for argument in sys.argv[1:]
    )
    if not has_config_dir:
        sys.argv[1:1] = ["--config-dir", str(_default_config_dir())]
    _hydra_main()


if __name__ == "__main__":
    main()
