"""Deterministic analytic fixtures used by the validation suite."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class SyntheticSplit:
    train: npt.NDArray[np.float64]
    validation: npt.NDArray[np.float64]
    test: npt.NDArray[np.float64]
    validation_lid: npt.NDArray[np.float64]
    test_lid: npt.NDArray[np.float64]


def flat_plane(
    *,
    seed: int,
    ambient_dim: int = 8,
    intrinsic_dim: int = 3,
    n_train: int = 2048,
    n_validation: int = 128,
    n_test: int = 128,
) -> SyntheticSplit:
    """Gaussian samples on a coordinate-aligned flat manifold."""

    if not 0 < intrinsic_dim <= ambient_dim:
        raise ValueError("require 0 < intrinsic_dim <= ambient_dim")
    rng = np.random.default_rng(seed)

    def sample(n: int) -> npt.NDArray[np.float64]:
        result = np.zeros((n, ambient_dim), dtype=np.float64)
        result[:, :intrinsic_dim] = rng.normal(size=(n, intrinsic_dim))
        return result

    return SyntheticSplit(
        train=sample(n_train),
        validation=sample(n_validation),
        test=sample(n_test),
        validation_lid=np.full(n_validation, intrinsic_dim, dtype=np.float64),
        test_lid=np.full(n_test, intrinsic_dim, dtype=np.float64),
    )


def half_space_reference(
    *,
    seed: int,
    ambient_dim: int = 8,
    intrinsic_dim: int = 3,
    n_train: int = 100_000,
) -> npt.NDArray[np.float64]:
    """Half-normal tangent samples with a boundary at the origin."""

    if not 1 <= intrinsic_dim <= ambient_dim:
        raise ValueError("require 1 <= intrinsic_dim <= ambient_dim")
    rng = np.random.default_rng(seed)
    result = np.zeros((n_train, ambient_dim), dtype=np.float64)
    result[:, 0] = np.abs(rng.normal(size=n_train))
    if intrinsic_dim > 1:
        result[:, 1:intrinsic_dim] = rng.normal(
            size=(n_train, intrinsic_dim - 1)
        )
    return result

