"""Empirical Gaussian-channel oracle for endpoint readout validation.

This module is deliberately an *oracle*, not a neural estimator.  It replaces
the data distribution by the weighted empirical measure on ``reference`` and
computes exact posterior moments for

``R = X + scale * N(0, I)``.

Consequently, it is useful for checking algebra, data plumbing, scale
selection, and the finite-sample behavior of the population identities in the
paper.  It does **not** train or emulate a diffusion model, flow-matching
network, Schrodinger bridge, or normalizing-flow architecture.  In particular,
the NF fields below use the canonical Gaussian heat-flow gauge; they are not
evidence that an arbitrary trained NF has learned the calibrated transport
required by the theorem.

Only NumPy is required.  Both query points and the empirical reference set are
chunked.  Posterior weights are accumulated with a streaming log-sum-exp
normalization, so very small Gaussian weights never need to be represented in
their unnormalised form.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]
ReadoutBranch = Literal["response", "full"]


READOUT_IDS: tuple[str, ...] = (
    "diffusion_flipd_full",
    "fm_affine_response",
    "fm_affine_full",
    "fm_rectified_response",
    "fm_rectified_full",
    "sb_forward_response",
    "sb_forward_full",
    "sb_current_full",
    "nf_scale_conditioned_fixed",
    "nf_calibrated_native",
    "cnf_calibrated_native",
)

_READOUT_BRANCHES: dict[str, ReadoutBranch] = {
    "diffusion_flipd_full": "full",
    "fm_affine_response": "response",
    "fm_affine_full": "full",
    "fm_rectified_response": "response",
    "fm_rectified_full": "full",
    "sb_forward_response": "response",
    "sb_forward_full": "full",
    "sb_current_full": "full",
    "nf_scale_conditioned_fixed": "full",
    "nf_calibrated_native": "response",
    "cnf_calibrated_native": "response",
}


def readout_branch(readout_id: str) -> ReadoutBranch:
    """Return the common Gaussian-channel branch used by a family readout.

    ``response`` is the normalized posterior covariance trace.  ``full`` adds
    the normalized squared posterior displacement and is boundary-safe for the
    homogeneous-cone setting in the paper.
    """

    try:
        return _READOUT_BRANCHES[readout_id]
    except KeyError as exc:
        valid = ", ".join(READOUT_IDS)
        raise ValueError(
            f"unknown oracle readout_id {readout_id!r}; expected one of: {valid}"
        ) from exc


def _positive_finite_scalar(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive; got {value!r}")
    return result


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
    return int(value)


def _matrix(value: npt.ArrayLike, *, name: str, dimension: int | None = None) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (n, D); got {array.shape}")
    if dimension is not None and array.shape[1] != dimension:
        raise ValueError(
            f"{name} has ambient dimension {array.shape[1]}, expected {dimension}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(array)


@dataclass(frozen=True)
class PosteriorMoments:
    """Batched sufficient statistics of an empirical Gaussian posterior.

    The four named posterior quantities are in data units.  The ``response``
    and ``full`` properties divide them by ``scale**2`` and are the LID
    readouts appearing throughout the paper.
    """

    query: FloatArray
    scale: float
    mean: FloatArray
    trace_covariance: FloatArray
    squared_bias: FloatArray
    full_second_moment: FloatArray
    log_density: FloatArray
    effective_sample_size: FloatArray

    @property
    def ambient_dim(self) -> int:
        return int(self.query.shape[1])

    @property
    def response(self) -> FloatArray:
        """Normalized posterior covariance trace (response branch)."""

        return self.trace_covariance / self.scale**2

    @property
    def normalized_squared_bias(self) -> FloatArray:
        return self.squared_bias / self.scale**2

    @property
    def full(self) -> FloatArray:
        """Normalized posterior squared displacement (full branch)."""

        return self.full_second_moment / self.scale**2

    @property
    def score(self) -> FloatArray:
        """Exact score of the empirical Gaussian mixture at ``query``."""

        return (self.mean - self.query) / self.scale**2

    @property
    def score_divergence(self) -> FloatArray:
        """Exact trace of the score Jacobian at ``query``."""

        return self.trace_covariance / self.scale**4 - self.ambient_dim / self.scale**2

    @property
    def scale_velocity(self) -> FloatArray:
        """Canonical heat-flow scale velocity ``u = query - posterior_mean``.

        This is one gauge satisfying the continuity equation for the empirical
        Gaussian-smoothed density path.  It is not a fitted NF velocity.
        """

        return self.query - self.mean

    @property
    def scale_velocity_divergence(self) -> FloatArray:
        """Divergence of the canonical heat-flow scale velocity."""

        return self.ambient_dim - self.response

    def readout(self, readout_id: str) -> FloatArray:
        """Evaluate a family-labelled population/empirical oracle target."""

        return self.response if readout_branch(readout_id) == "response" else self.full

    def all_readouts(self) -> dict[str, FloatArray]:
        """Return every requested family ID without recomputing the posterior."""

        return {readout_id: self.readout(readout_id) for readout_id in READOUT_IDS}


class EmpiricalGaussianChannel:
    """Exact posterior moments under a weighted empirical reference measure.

    Parameters
    ----------
    reference:
        Array of shape ``(n_reference, ambient_dim)``.  A one-dimensional
        input is interpreted as one point, not as a scalar sample vector.
    sample_weight:
        Optional non-negative empirical masses.  Zero-mass points are removed;
        multiplying all masses by a constant has no effect.
    reference_chunk_size:
        Maximum number of reference points materialized in each pairwise
        distance block.
    """

    def __init__(
        self,
        reference: npt.ArrayLike,
        *,
        sample_weight: npt.ArrayLike | None = None,
        reference_chunk_size: int = 4096,
    ) -> None:
        points = _matrix(reference, name="reference")
        if points.shape[0] == 0:
            raise ValueError("reference must contain at least one point")
        if points.shape[1] == 0:
            raise ValueError("reference must have a positive ambient dimension")

        if sample_weight is None:
            weights = np.ones(points.shape[0], dtype=np.float64)
        else:
            weights = np.ravel(np.asarray(sample_weight, dtype=np.float64))
            if weights.shape != (points.shape[0],):
                raise ValueError(
                    "sample_weight must have shape "
                    f"({points.shape[0]},); got {weights.shape}"
                )
            if not np.isfinite(weights).all() or np.any(weights < 0.0):
                raise ValueError("sample_weight must be finite and non-negative")
        positive = weights > 0.0
        if not positive.any():
            raise ValueError("sample_weight must contain at least one positive mass")
        points = points[positive].copy()
        weights = weights[positive]

        # Work in coordinates relative to a data point.  This does not change
        # distances or covariance, and avoids avoidable cancellation when all
        # coordinates carry a large common offset.
        origin = points[0].copy()
        centered = points - origin
        if not np.isfinite(centered).all():
            raise ValueError("reference coordinate range overflows float64")

        log_weight = np.log(weights)
        maximum = float(np.max(log_weight))
        log_total = maximum + math.log(float(np.exp(log_weight - maximum).sum()))

        self._origin = origin
        self._reference = np.ascontiguousarray(centered)
        self._log_probability = np.ascontiguousarray(log_weight - log_total)
        self._reference_squared_norm = np.einsum(
            "ij,ij->i", self._reference, self._reference
        )
        self._reference_chunk_size = _positive_integer(
            reference_chunk_size, name="reference_chunk_size"
        )

    @property
    def ambient_dim(self) -> int:
        return int(self._reference.shape[1])

    @property
    def n_reference(self) -> int:
        return int(self._reference.shape[0])

    def posterior(
        self,
        query: npt.ArrayLike,
        scale: float,
        *,
        query_chunk_size: int = 128,
    ) -> PosteriorMoments:
        """Compute posterior moments for all query points.

        The returned arrays always retain a batch axis, including when
        ``query`` was one-dimensional.
        """

        scale = _positive_finite_scalar(scale, name="scale")
        scale_squared = scale * scale
        if scale_squared == 0.0 or not math.isfinite(scale_squared):
            raise ValueError("scale**2 must be representable as a finite positive float64")
        queries = _matrix(query, name="query", dimension=self.ambient_dim)
        query_chunk_size = _positive_integer(
            query_chunk_size, name="query_chunk_size"
        )
        n_query = queries.shape[0]
        means = np.empty_like(queries)
        trace_covariance = np.empty(n_query, dtype=np.float64)
        squared_bias = np.empty(n_query, dtype=np.float64)
        full_second_moment = np.empty(n_query, dtype=np.float64)
        log_density = np.empty(n_query, dtype=np.float64)
        effective_sample_size = np.empty(n_query, dtype=np.float64)

        for start in range(0, n_query, query_chunk_size):
            stop = min(start + query_chunk_size, n_query)
            result = self._posterior_chunk(queries[start:stop], scale)
            means[start:stop] = result[0]
            trace_covariance[start:stop] = result[1]
            squared_bias[start:stop] = result[2]
            full_second_moment[start:stop] = result[3]
            log_density[start:stop] = result[4]
            effective_sample_size[start:stop] = result[5]

        return PosteriorMoments(
            query=queries.copy(),
            scale=scale,
            mean=means,
            trace_covariance=trace_covariance,
            squared_bias=squared_bias,
            full_second_moment=full_second_moment,
            log_density=log_density,
            effective_sample_size=effective_sample_size,
        )

    def readout(
        self,
        readout_id: str,
        query: npt.ArrayLike,
        scale: float,
        *,
        query_chunk_size: int = 128,
    ) -> FloatArray:
        """Convenience wrapper returning one family-labelled oracle readout."""

        return self.posterior(
            query, scale, query_chunk_size=query_chunk_size
        ).readout(readout_id)

    def readout_curve(
        self,
        readout_id: str,
        query: npt.ArrayLike,
        scales: npt.ArrayLike,
        *,
        query_chunk_size: int = 128,
    ) -> FloatArray:
        """Evaluate a validation curve with shape ``(n_query, n_scales)``."""

        # Validate the ID before performing an expensive posterior pass.
        readout_branch(readout_id)
        scale_array = np.ravel(np.asarray(scales, dtype=np.float64))
        if scale_array.size == 0:
            raise ValueError("scales must not be empty")
        if not np.isfinite(scale_array).all() or np.any(scale_array <= 0.0):
            raise ValueError("scales must be finite and positive")
        queries = _matrix(query, name="query", dimension=self.ambient_dim)
        columns = [
            self.readout(
                readout_id,
                queries,
                float(scale),
                query_chunk_size=query_chunk_size,
            )
            for scale in scale_array
        ]
        return np.column_stack(columns)

    def _posterior_chunk(
        self, query: FloatArray, scale: float
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
        count = query.shape[0]
        if count == 0:
            empty_vector = np.empty(0, dtype=np.float64)
            return (
                np.empty((0, self.ambient_dim), dtype=np.float64),
                empty_vector.copy(),
                empty_vector.copy(),
                empty_vector.copy(),
                empty_vector.copy(),
                empty_vector.copy(),
            )

        centered_query = query - self._origin
        if not np.isfinite(centered_query).all():
            raise ValueError("query/reference coordinate range overflows float64")
        query_squared_norm = np.einsum(
            "ij,ij->i", centered_query, centered_query
        )

        running_max: FloatArray | None = None
        denominator = np.empty(count, dtype=np.float64)
        squared_weight_sum = np.empty(count, dtype=np.float64)
        first_moment = np.empty((count, self.ambient_dim), dtype=np.float64)
        squared_distance_moment = np.empty(count, dtype=np.float64)
        inverse_two_variance = 0.5 / scale**2

        for start in range(0, self.n_reference, self._reference_chunk_size):
            stop = min(start + self._reference_chunk_size, self.n_reference)
            reference = self._reference[start:stop]
            squared_distance = (
                query_squared_norm[:, None]
                + self._reference_squared_norm[None, start:stop]
                - 2.0 * centered_query @ reference.T
            )
            # The dot-product identity can produce tiny negative values.
            np.maximum(squared_distance, 0.0, out=squared_distance)
            log_kernel_weight = (
                -inverse_two_variance * squared_distance
                + self._log_probability[None, start:stop]
            )
            block_max = np.max(log_kernel_weight, axis=1)
            if not np.isfinite(block_max).all():
                raise FloatingPointError(
                    "all Gaussian log-weights became non-finite; rescale the data"
                )
            block_weight = np.exp(log_kernel_weight - block_max[:, None])
            block_denominator = block_weight.sum(axis=1)
            block_first_moment = block_weight @ reference
            block_squared_distance = np.einsum(
                "ij,ij->i", block_weight, squared_distance
            )
            block_squared_weight = np.einsum(
                "ij,ij->i", block_weight, block_weight
            )

            if running_max is None:
                running_max = block_max
                denominator[:] = block_denominator
                first_moment[:] = block_first_moment
                squared_distance_moment[:] = block_squared_distance
                squared_weight_sum[:] = block_squared_weight
                continue

            merged_max = np.maximum(running_max, block_max)
            old_factor = np.exp(running_max - merged_max)
            block_factor = np.exp(block_max - merged_max)
            denominator *= old_factor
            denominator += block_factor * block_denominator
            first_moment *= old_factor[:, None]
            first_moment += block_factor[:, None] * block_first_moment
            squared_distance_moment *= old_factor
            squared_distance_moment += block_factor * block_squared_distance
            squared_weight_sum *= old_factor**2
            squared_weight_sum += block_factor**2 * block_squared_weight
            running_max = merged_max

        if running_max is None:  # impossible after constructor validation
            raise AssertionError("empty empirical reference")

        mean_centered = first_moment / denominator[:, None]
        mean_offset = mean_centered - centered_query
        bias = np.einsum("ij,ij->i", mean_offset, mean_offset)
        posterior_distance = squared_distance_moment / denominator
        covariance_trace = posterior_distance - bias
        roundoff_tolerance = 128.0 * np.finfo(np.float64).eps * np.maximum.reduce(
            [posterior_distance, bias, np.ones_like(bias)]
        )
        if np.any(covariance_trace < -roundoff_tolerance):
            raise FloatingPointError(
                "posterior covariance became negative beyond float64 roundoff"
            )
        np.maximum(covariance_trace, 0.0, out=covariance_trace)
        # Define the full moment through the exact bias-variance decomposition,
        # so downstream identity checks are not polluted by one-ulp subtraction.
        posterior_distance = covariance_trace + bias
        mixture_log_density = (
            running_max
            + np.log(denominator)
            - self.ambient_dim * math.log(scale)
            - 0.5 * self.ambient_dim * math.log(2.0 * math.pi)
        )
        ess = denominator**2 / squared_weight_sum

        return (
            mean_centered + self._origin,
            covariance_trace,
            bias,
            posterior_distance,
            mixture_log_density,
            ess,
        )


def select_stable_scale(
    scales: npt.ArrayLike,
    validation_curves: npt.ArrayLike,
    *,
    window: int = 1,
    valid_mask: npt.ArrayLike | None = None,
    min_valid_fraction: float = 0.5,
    prefer: Literal["smaller", "larger"] = "smaller",
) -> tuple[int, dict[str, Any]]:
    """Choose a scale from validation-curve stability, without LID labels.

    For every non-edge candidate, the criterion is the median (over validation
    examples) of the median absolute first derivative over the surrounding
    ``2 * window`` log-scale intervals.  Missing curve values are allowed.
    ``valid_mask`` can exclude unreliable estimates, for example scales whose
    empirical posterior effective sample size is too small.

    This is a label-free plateau heuristic, not a consistency theorem.  In
    particular, a degenerate nearest-neighbour plateau can look stable; callers
    should pass a reliability mask when that failure mode matters.

    Returns
    -------
    index, diagnostics:
        ``index`` refers to the original order of ``scales``.  ``diagnostics``
        contains only JSON-serializable Python values.
    """

    scale_array = np.ravel(np.asarray(scales, dtype=np.float64))
    if scale_array.size == 0:
        raise ValueError("scales must not be empty")
    if not np.isfinite(scale_array).all() or np.any(scale_array <= 0.0):
        raise ValueError("scales must be finite and positive")
    if np.unique(scale_array).size != scale_array.size:
        raise ValueError("scales must be unique")
    window = _positive_integer(window, name="window")
    if scale_array.size < 2 * window + 1:
        raise ValueError(
            f"at least {2 * window + 1} scales are required for window={window}"
        )
    if not math.isfinite(float(min_valid_fraction)) or not (
        0.0 < float(min_valid_fraction) <= 1.0
    ):
        raise ValueError("min_valid_fraction must lie in (0, 1]")
    if prefer not in {"smaller", "larger"}:
        raise ValueError("prefer must be 'smaller' or 'larger'")

    curves = np.asarray(validation_curves, dtype=np.float64)
    if curves.ndim == 0 or curves.shape[-1] != scale_array.size:
        raise ValueError(
            "validation_curves must have scales on its final axis; expected "
            f"length {scale_array.size}, got {curves.shape}"
        )
    curves = curves.reshape(-1, scale_array.size)
    if curves.shape[0] == 0:
        raise ValueError("validation_curves must contain at least one curve")

    finite = np.isfinite(curves)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        try:
            mask = np.broadcast_to(mask, np.asarray(validation_curves).shape)
        except ValueError as exc:
            raise ValueError(
                "valid_mask must be broadcastable to validation_curves"
            ) from exc
        finite &= mask.reshape(curves.shape)

    order = np.argsort(scale_array)
    ordered_scales = scale_array[order]
    ordered_curves = curves[:, order]
    ordered_valid = finite[:, order]
    log_step = np.diff(np.log(ordered_scales))
    edge_rate = np.abs(np.diff(ordered_curves, axis=1)) / log_step[None, :]
    edge_valid = ordered_valid[:, :-1] & ordered_valid[:, 1:]
    edge_rate[~edge_valid] = np.nan

    ordered_scores = np.full(scale_array.size, np.inf, dtype=np.float64)
    ordered_counts = np.zeros(scale_array.size, dtype=np.int64)
    required_count = max(
        1, int(math.ceil(float(min_valid_fraction) * curves.shape[0]))
    )
    for index in range(window, scale_array.size - window):
        local = edge_rate[:, index - window : index + window]
        local_count = np.sum(np.isfinite(local), axis=1)
        per_curve = np.full(curves.shape[0], np.nan, dtype=np.float64)
        usable = local_count == 2 * window
        if usable.any():
            per_curve[usable] = np.median(local[usable], axis=1)
        count = int(np.isfinite(per_curve).sum())
        ordered_counts[index] = count
        if count >= required_count:
            ordered_scores[index] = float(np.nanmedian(per_curve))

    finite_candidate = np.flatnonzero(np.isfinite(ordered_scores))
    if finite_candidate.size == 0:
        raise ValueError(
            "no scale has enough finite validation support for the requested window"
        )
    best_score = float(np.min(ordered_scores[finite_candidate]))
    tolerance = 32.0 * np.finfo(np.float64).eps * max(1.0, abs(best_score))
    tied = finite_candidate[
        np.abs(ordered_scores[finite_candidate] - best_score) <= tolerance
    ]
    best_ordered = int(tied[0] if prefer == "smaller" else tied[-1])
    selected_index = int(order[best_ordered])

    scores_in_input_order = np.empty_like(ordered_scores)
    counts_in_input_order = np.empty_like(ordered_counts)
    scores_in_input_order[order] = ordered_scores
    counts_in_input_order[order] = ordered_counts
    serialized_scores: list[float | None] = [
        float(value) if math.isfinite(float(value)) else None
        for value in scores_in_input_order
    ]
    diagnostics: dict[str, Any] = {
        "criterion": "median_local_absolute_log_scale_slope",
        "uses_ground_truth": False,
        "selected_index": selected_index,
        "selected_scale": float(scale_array[selected_index]),
        "selected_stability": float(scores_in_input_order[selected_index]),
        "stability_scores": serialized_scores,
        "valid_curve_counts": [int(value) for value in counts_in_input_order],
        "required_valid_curve_count": required_count,
        "window": window,
        "prefer": prefer,
    }
    return selected_index, diagnostics


__all__ = [
    "EmpiricalGaussianChannel",
    "PosteriorMoments",
    "READOUT_IDS",
    "readout_branch",
    "select_stable_scale",
]
