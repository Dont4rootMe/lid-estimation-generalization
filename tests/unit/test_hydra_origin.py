from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from experiments.runner import (
    ExperimentConfigError,
    run_experiment,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_raw_mapping_cannot_create_hydra_stamped_artifact(tmp_path: Path) -> None:
    with pytest.raises(
        ExperimentConfigError, match="structured Hydra-composed DictConfig"
    ):
        run_experiment(  # type: ignore[arg-type]
            {},
            root=REPOSITORY_ROOT,
            output_root=tmp_path,
        )

    assert list(tmp_path.iterdir()) == []


def test_unstructured_omegaconf_cannot_impersonate_hydra_origin(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ExperimentConfigError, match="structured Hydra-composed DictConfig"
    ):
        run_experiment(
            OmegaConf.create({}),
            root=REPOSITORY_ROOT,
            output_root=tmp_path,
        )

    assert list(tmp_path.iterdir()) == []
