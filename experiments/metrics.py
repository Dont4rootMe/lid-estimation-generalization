"""Metrics shared by every estimator and benchmark representation."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def prediction_summary(prediction: npt.ArrayLike) -> dict[str, Any]:
    values = np.ravel(np.asarray(prediction, dtype=np.float64))
    finite = np.isfinite(values)
    result: dict[str, Any] = {
        "n": int(values.size),
        "finite_n": int(finite.sum()),
        "finite_fraction": float(finite.mean()) if values.size else 0.0,
    }
    if finite.any():
        clean = values[finite]
        result.update(
            mean=float(clean.mean()),
            std=float(clean.std()),
            median=float(np.median(clean)),
            q05=float(np.quantile(clean, 0.05)),
            q95=float(np.quantile(clean, 0.95)),
        )
    return result


def known_lid_metrics(
    prediction: npt.ArrayLike, target: npt.ArrayLike
) -> dict[str, Any]:
    values = np.ravel(np.asarray(prediction, dtype=np.float64))
    truth = np.ravel(np.asarray(target, dtype=np.float64))
    if values.shape != truth.shape:
        raise ValueError(
            f"prediction and target must have equal shapes; got {values.shape} and {truth.shape}"
        )
    finite = np.isfinite(values) & np.isfinite(truth)
    result = prediction_summary(values)
    result["target_finite_n"] = int(np.isfinite(truth).sum())
    if not finite.any():
        return result
    error = values[finite] - truth[finite]
    result.update(
        mae=float(np.abs(error).mean()),
        rmse=float(np.sqrt(np.square(error).mean())),
        bias=float(error.mean()),
        median_absolute_error=float(np.median(np.abs(error))),
    )
    return result


def paired_delta_metrics(
    base_prediction: npt.ArrayLike,
    transformed_prediction: npt.ArrayLike,
    *,
    expected_delta: float,
) -> dict[str, Any]:
    base = np.ravel(np.asarray(base_prediction, dtype=np.float64))
    transformed = np.ravel(np.asarray(transformed_prediction, dtype=np.float64))
    if base.shape != transformed.shape:
        raise ValueError("paired predictions must have identical shapes")
    delta = transformed - base
    target = np.full(delta.shape, expected_delta, dtype=np.float64)
    result = known_lid_metrics(delta, target)
    result["expected_delta"] = float(expected_delta)
    return result

