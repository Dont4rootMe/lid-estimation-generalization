"""Small, secret-safe Comet adapter for pilot training runs.

The API key is deliberately never accepted as a function argument or config
field.  ``comet_ml`` reads it from ``COMET_API_KEY`` in the process
environment.  This keeps the credential out of Hydra's resolved config,
scheduler payloads, manifests and dry-run output.
"""

from __future__ import annotations

from collections.abc import Mapping
import importlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Protocol


COMET_NAMESPACE = "ent-block-diffusion-eval"
COMET_API_KEY_ENV = "COMET_API_KEY"
_FORBIDDEN_KEY_PARTS = ("api_key", "apikey", "password", "secret", "token")


class CometConfigurationError(RuntimeError):
    """Raised before creating a Comet experiment with unsafe configuration."""


class _CometExperiment(Protocol):
    def set_name(self, name: str) -> Any: ...

    def add_tag(self, tag: str) -> Any: ...

    def log_metrics(
        self,
        metrics: Mapping[str, float],
        *,
        step: int | None = None,
        prefix: str | None = None,
    ) -> Any: ...

    def log_parameters(
        self, parameters: Mapping[str, Any], *, prefix: str | None = None
    ) -> Any: ...

    def log_asset(self, file_data: str, *, file_name: str | None = None) -> Any: ...

    def end(self) -> Any: ...


def require_comet_environment(
    environ: Mapping[str, str] | None = None,
) -> None:
    """Validate the process environment without returning or copying the key."""

    source = os.environ if environ is None else environ
    if not source.get(COMET_API_KEY_ENV, "").strip():
        raise CometConfigurationError(
            f"{COMET_API_KEY_ENV} must be set in the job environment"
        )
    for variable in ("COMET_PROJECT_NAME", "COMET_EXPERIMENT_NAME"):
        configured = source.get(variable)
        if configured is not None and configured != COMET_NAMESPACE:
            raise CometConfigurationError(
                f"{variable} must be exactly {COMET_NAMESPACE!r}"
            )


def safe_scheduler_environment() -> dict[str, str]:
    """Return public Comet settings suitable for ``client_lib.Job``.

    In particular, this function never reads or returns ``COMET_API_KEY``.
    The submitted script must obtain that variable from a private runtime
    source (the cluster launcher sources a mode-0600 file on shared storage).
    """

    return {
        "COMET_PROJECT_NAME": COMET_NAMESPACE,
        "COMET_EXPERIMENT_NAME": COMET_NAMESPACE,
        "COMET_MODE": "online",
        "COMET_LOGGING_CONSOLE": "true",
    }


def _is_forbidden_key(key: object) -> bool:
    normalized = str(key).lower().replace("-", "_")
    return any(part in normalized for part in _FORBIDDEN_KEY_PARTS)


def _assert_secret_free(value: object, *, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if _is_forbidden_key(key):
                raise CometConfigurationError(
                    f"secret-like field is forbidden in Comet events: {path}.{key}"
                )
            _assert_secret_free(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_secret_free(child, path=f"{path}[{index}]")


def _flatten(
    value: Mapping[str, Any], *, prefix: str = ""
) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, child in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(child, Mapping):
            flattened.update(_flatten(child, prefix=name))
        elif isinstance(child, (list, tuple)):
            flattened[name] = json.dumps(child, sort_keys=True, default=str)
        elif isinstance(child, Path):
            flattened[name] = str(child)
        else:
            flattened[name] = child
    return flattened


class CometEventLogger:
    """Callable adapter for ``run_pilot(..., log_callback=...)``."""

    def __init__(self, experiment: _CometExperiment) -> None:
        self._experiment = experiment

    def __repr__(self) -> str:
        return f"{type(self).__name__}(namespace={COMET_NAMESPACE!r})"

    def __call__(self, event_name: str, payload: Mapping[str, Any]) -> None:
        if not isinstance(event_name, str) or not event_name.strip():
            raise CometConfigurationError("Comet event name must be non-empty")
        if not isinstance(payload, Mapping):
            raise CometConfigurationError("Comet event payload must be a mapping")
        _assert_secret_free(payload)

        flattened = _flatten(payload)
        raw_step = flattened.pop("step", None)
        step: int | None = None
        if raw_step is not None:
            if isinstance(raw_step, bool) or not isinstance(raw_step, (int, float)):
                raise CometConfigurationError("event step must be numeric")
            step = int(raw_step)

        metrics: dict[str, float] = {}
        parameters: dict[str, Any] = {}
        for key, value in flattened.items():
            if isinstance(value, bool):
                parameters[key] = value
            elif isinstance(value, (int, float)) and math.isfinite(float(value)):
                metrics[key] = float(value)
            else:
                parameters[key] = value

        if metrics:
            self._experiment.log_metrics(
                metrics,
                step=step,
                prefix=event_name,
            )
        if parameters:
            self._experiment.log_parameters(parameters, prefix=event_name)

    def log_asset(self, path: str | Path, *, name: str | None = None) -> None:
        asset = Path(path)
        if not asset.is_file():
            raise FileNotFoundError(asset)
        self._experiment.log_asset(str(asset), file_name=name)

    def end(self) -> None:
        self._experiment.end()


def create_comet_event_logger(
    *,
    tags: tuple[str, ...] = (),
    environ: Mapping[str, str] | None = None,
) -> CometEventLogger:
    """Create one Comet run in the approved namespace.

    ``environ`` exists for validation/testing only.  The SDK itself reads the
    real process environment, so a caller cannot pass the credential through
    this API.
    """

    require_comet_environment(environ)
    for tag in tags:
        if not isinstance(tag, str) or not tag.strip():
            raise CometConfigurationError("Comet tags must be non-empty strings")
    comet_ml = importlib.import_module("comet_ml")
    experiment = comet_ml.Experiment(project_name=COMET_NAMESPACE)
    experiment.set_name(COMET_NAMESPACE)
    experiment.add_tag(COMET_NAMESPACE)
    for tag in tags:
        experiment.add_tag(tag)
    return CometEventLogger(experiment)


def create_comet_callback(
    *,
    tags: tuple[str, ...] = (),
    environ: Mapping[str, str] | None = None,
) -> tuple[
    Callable[[str, Mapping[str, Any]], None],
    Callable[[], None],
]:
    """Return the callback/close pair consumed by the pilot runner."""

    logger = create_comet_event_logger(tags=tags, environ=environ)
    return logger, logger.end


__all__ = [
    "COMET_API_KEY_ENV",
    "COMET_NAMESPACE",
    "CometConfigurationError",
    "CometEventLogger",
    "create_comet_callback",
    "create_comet_event_logger",
    "require_comet_environment",
    "safe_scheduler_environment",
]
