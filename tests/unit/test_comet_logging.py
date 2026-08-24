from __future__ import annotations

import sys
import types
from typing import Any, ClassVar

import pytest

from experiments.comet_logging import (
    COMET_CONFIG_PATH,
    COMET_PROJECT_NAME,
    COMET_WORKSPACE_NAME,
    CometConfigurationError,
    create_comet_callback,
    create_comet_event_logger,
    require_comet_environment,
    safe_scheduler_environment,
)

EXPERIMENT_NAMES = (
    "lid-generalization-e8-suite-diffusion-train-mae-scale-selection-seed-0",
    (
        "lid-generalization-e8-suite-rectified-flow-matching-"
        "train-mae-time-selection-seed-0"
    ),
    (
        "lid-generalization-e8-suite-scale-conditioned-normalizing-flow-"
        "train-mae-scale-selection-seed-0"
    ),
    (
        "lid-generalization-e8-suite-brownian-schrodinger-bridge-"
        "train-mae-time-selection-seed-0"
    ),
)


class FakeExperiment:
    instances: ClassVar[list[FakeExperiment]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.names: list[str] = []
        self.tags: list[str] = []
        self.metrics: list[tuple[dict[str, float], int | None, str | None]] = []
        self.parameters: list[tuple[dict[str, Any], str | None]] = []
        self.ended = False
        self.__class__.instances.append(self)

    def set_name(self, name: str) -> None:
        self.names.append(name)

    def add_tag(self, tag: str) -> None:
        self.tags.append(tag)

    def log_metrics(
        self,
        metrics: dict[str, float],
        *,
        step: int | None = None,
        prefix: str | None = None,
    ) -> None:
        self.metrics.append((metrics, step, prefix))

    def log_parameters(
        self, parameters: dict[str, Any], *, prefix: str | None = None
    ) -> None:
        self.parameters.append((parameters, prefix))

    def log_asset(self, file_data: str, *, file_name: str | None = None) -> None:
        pass

    def end(self) -> None:
        self.ended = True


def test_public_scheduler_environment_is_fixed_and_secret_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COMET_API_KEY", "do-not-copy")
    environment = safe_scheduler_environment()
    assert "COMET_API_KEY" not in environment
    assert environment["COMET_PROJECT_NAME"] == COMET_PROJECT_NAME
    assert "COMET_EXPERIMENT_NAME" not in environment
    assert environment["COMET_WORKSPACE"] == COMET_WORKSPACE_NAME
    assert environment["COMET_CONFIG"] == COMET_CONFIG_PATH
    assert "do-not-copy" not in repr(environment)


def test_comet_environment_is_required_and_namespace_is_exact() -> None:
    with pytest.raises(CometConfigurationError, match="COMET_CONFIG"):
        require_comet_environment({})
    with pytest.raises(CometConfigurationError, match="COMET_PROJECT_NAME"):
        require_comet_environment(
            {
                **safe_scheduler_environment(),
                "COMET_PROJECT_NAME": "wrong",
            }
        )
    require_comet_environment(safe_scheduler_environment())
    with pytest.raises(CometConfigurationError, match="COMET_EXPERIMENT_NAME"):
        require_comet_environment(
            {
                **safe_scheduler_environment(),
                "COMET_EXPERIMENT_NAME": EXPERIMENT_NAMES[0],
            }
        )
    with pytest.raises(CometConfigurationError, match="COMET_WORKSPACE"):
        require_comet_environment(
            {**safe_scheduler_environment(), "COMET_WORKSPACE": "wrong"}
        )


@pytest.mark.parametrize("experiment_name", EXPERIMENT_NAMES)
def test_factory_never_passes_api_key_to_comet_sdk(
    monkeypatch: pytest.MonkeyPatch, experiment_name: str
) -> None:
    FakeExperiment.instances.clear()
    monkeypatch.setitem(
        sys.modules, "comet_ml", types.SimpleNamespace(Experiment=FakeExperiment)
    )
    logger = create_comet_event_logger(
        experiment_name=experiment_name,
        tags=("diffusion",),
        environ=safe_scheduler_environment(),
    )
    experiment = FakeExperiment.instances[-1]
    assert experiment.kwargs == {
        "project_name": COMET_PROJECT_NAME,
        "workspace": COMET_WORKSPACE_NAME,
    }
    assert experiment.names == [experiment_name]
    assert experiment.tags == [experiment_name, "diffusion"]
    logger.end()
    assert experiment.ended


def test_production_callback_factory_returns_callback_and_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeExperiment.instances.clear()
    monkeypatch.setitem(
        sys.modules, "comet_ml", types.SimpleNamespace(Experiment=FakeExperiment)
    )
    callback, close = create_comet_callback(
        experiment_name=EXPERIMENT_NAMES[0],
        tags=("diffusion",),
        environ=safe_scheduler_environment(),
    )
    callback("training", {"step": 1, "loss": 0.75})
    close()
    experiment = FakeExperiment.instances[-1]
    assert experiment.metrics == [({"loss": 0.75}, 1, "training")]
    assert experiment.ended is True


def test_event_callback_logs_nested_metrics_and_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeExperiment.instances.clear()
    monkeypatch.setitem(
        sys.modules, "comet_ml", types.SimpleNamespace(Experiment=FakeExperiment)
    )
    logger = create_comet_event_logger(
        experiment_name=EXPERIMENT_NAMES[1], environ=safe_scheduler_environment()
    )
    logger(
        "validation",
        {
            "step": 7,
            "dataset": "e8_gaussian4_pca",
            "metrics": {"mae": 0.25, "coverage": 1.0},
        },
    )
    experiment = FakeExperiment.instances[-1]
    assert experiment.metrics == [
        ({"metrics.mae": 0.25, "metrics.coverage": 1.0}, 7, "validation")
    ]
    assert experiment.parameters == [({"dataset": "e8_gaussian4_pca"}, "validation")]


def test_event_callback_rejects_secret_like_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeExperiment.instances.clear()
    monkeypatch.setitem(
        sys.modules, "comet_ml", types.SimpleNamespace(Experiment=FakeExperiment)
    )
    logger = create_comet_event_logger(
        experiment_name=EXPERIMENT_NAMES[0], environ=safe_scheduler_environment()
    )
    with pytest.raises(CometConfigurationError, match="secret-like field"):
        logger("training", {"metadata": {"access_token": "do-not-log"}})


@pytest.mark.parametrize(
    "experiment_name",
    ["", "Rectified Flow", "rectified_flow", "ent-block-diffusion-eval"],
)
def test_factory_rejects_non_convention_experiment_names(
    monkeypatch: pytest.MonkeyPatch, experiment_name: str
) -> None:
    monkeypatch.setitem(
        sys.modules, "comet_ml", types.SimpleNamespace(Experiment=FakeExperiment)
    )
    with pytest.raises(CometConfigurationError, match="kebab-case"):
        create_comet_event_logger(
            experiment_name=experiment_name,
            environ=safe_scheduler_environment(),
        )
