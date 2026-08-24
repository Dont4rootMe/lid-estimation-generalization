"""Contract for learned-model outputs consumed by LID readouts.

Training stacks are intentionally replaceable.  A model implementation writes
an NPZ bundle of primitive fields plus JSON metadata; this module validates the
bundle before any benchmark metric is computed.  NPZ loading never enables
pickle.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt

from models import readouts


READOUT_REQUIREMENTS: dict[str, frozenset[str]] = {
    "diffusion_flipd_full": frozenset({"score", "score_divergence"}),
    "fm_affine_response": frozenset({"velocity_divergence"}),
    "fm_affine_full": frozenset(
        {"velocity", "velocity_divergence", "score", "evaluation_point"}
    ),
    "fm_rectified_response": frozenset({"velocity_divergence"}),
    "fm_rectified_full": frozenset(
        {"velocity", "velocity_divergence", "data_point"}
    ),
    "sb_forward_response": frozenset({"forward_drift_divergence"}),
    "sb_forward_full": frozenset(
        {"forward_drift", "forward_drift_divergence"}
    ),
    "sb_current_full": frozenset(
        {"current_velocity", "current_velocity_divergence", "score"}
    ),
    "nf_scale_conditioned_fixed": frozenset(
        {"scale_velocity", "scale_velocity_divergence", "score"}
    ),
    "nf_calibrated_native": frozenset({"scale_velocity_divergence"}),
    "cnf_calibrated_native": frozenset({"velocity_divergence"}),
}


SCALAR_REQUIREMENTS: dict[str, frozenset[str]] = {
    "diffusion_flipd_full": frozenset({"sigma", "ambient_dim"}),
    "fm_affine_response": frozenset(
        {"ambient_dim", "alpha_log_derivative", "log_noise_ratio_derivative"}
    ),
    "fm_affine_full": frozenset(
        {"ambient_dim", "alpha_log_derivative", "log_noise_ratio_derivative"}
    ),
    "fm_rectified_response": frozenset({"ambient_dim", "t"}),
    "fm_rectified_full": frozenset({"ambient_dim", "t"}),
    "sb_forward_response": frozenset({"ambient_dim", "time_to_go"}),
    "sb_forward_full": frozenset(
        {"ambient_dim", "time_to_go", "diffusivity"}
    ),
    "sb_current_full": frozenset({"ambient_dim", "time_to_go"}),
    "nf_scale_conditioned_fixed": frozenset({"ambient_dim"}),
    "nf_calibrated_native": frozenset({"ambient_dim"}),
    "cnf_calibrated_native": frozenset(
        {"ambient_dim", "log_scale_derivative"}
    ),
}


MODEL_FAMILY_READOUTS: dict[str, tuple[str, ...]] = {
    "gaussian_diffusion": ("diffusion_flipd_full",),
    "affine_flow_matching": ("fm_affine_response", "fm_affine_full"),
    "rectified_flow": ("fm_rectified_response", "fm_rectified_full"),
    "brownian_schrodinger_bridge": (
        "sb_forward_response",
        "sb_forward_full",
        "sb_current_full",
    ),
    "scale_conditioned_normalizing_flow": ("nf_scale_conditioned_fixed",),
    "calibrated_singular_normalizing_flow": ("nf_calibrated_native",),
    "calibrated_singular_cnf": ("cnf_calibrated_native",),
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class BundleContext:
    """Identity expected for one learned field-bundle evaluation."""

    model_name: str
    model_family: str
    model_seed: int
    checkpoint_sha256: str
    training_config_sha256: str
    dataset_name: str
    training_dataset_sha256: str
    dataset_sha256: str
    representation: str
    split: str
    query_sha256: str
    preprocessing_sha256: str
    model_space_query_sha256: str
    n_samples: int
    scale_index: int
    physical_scale: float
    readout_ids: tuple[str, ...]
    trace_backend: str
    trace_probes: int
    trace_seed: int


def _sha256(value: Any, *, name: str) -> str:
    result = str(value)
    if _SHA256_RE.fullmatch(result) is None:
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256")
    return result


def _exact_metadata_value(
    metadata: Mapping[str, Any], name: str, expected: Any
) -> None:
    if metadata.get(name) != expected:
        raise ValueError(
            f"field bundle metadata {name!r} mismatch: "
            f"expected {expected!r}, got {metadata.get(name)!r}"
        )


def validate_bundle_provenance(bundle: "FieldBundle", context: BundleContext) -> None:
    """Bind learned fields to a checkpoint, config, dataset rows and trace run.

    Formula unit tests may construct :class:`FieldBundle` directly and call
    :meth:`FieldBundle.validate`.  A benchmark runner must additionally call
    this function; empty or merely descriptive metadata is never accepted as
    learned-model evidence.
    """

    metadata = bundle.metadata
    required = {
        "schema_version",
        "model_name",
        "model_family",
        "model_seed",
        "checkpoint_sha256",
        "training_config_sha256",
        "dataset_name",
        "training_dataset_sha256",
        "dataset_sha256",
        "representation",
        "split",
        "query_sha256",
        "preprocessing_sha256",
        "model_space_query_sha256",
        "n_samples",
        "scale_index",
        "physical_scale",
        "readout_ids",
        "trace",
    }
    unknown = set(metadata) - required
    missing = required - set(metadata)
    if missing or unknown:
        raise ValueError(
            "field bundle metadata schema mismatch: "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    _exact_metadata_value(metadata, "schema_version", 1)
    for name in (
        "model_name",
        "model_family",
        "model_seed",
        "checkpoint_sha256",
        "training_config_sha256",
        "dataset_name",
        "training_dataset_sha256",
        "dataset_sha256",
        "representation",
        "split",
        "query_sha256",
        "preprocessing_sha256",
        "model_space_query_sha256",
        "n_samples",
        "scale_index",
        "readout_ids",
    ):
        expected = getattr(context, name)
        if name == "readout_ids":
            actual = metadata.get(name)
            if isinstance(actual, list):
                actual = tuple(actual)
            if actual != expected:
                raise ValueError(
                    f"field bundle metadata {name!r} mismatch: "
                    f"expected {expected!r}, got {metadata.get(name)!r}"
                )
        else:
            _exact_metadata_value(metadata, name, expected)
    _sha256(metadata["checkpoint_sha256"], name="metadata.checkpoint_sha256")
    _sha256(
        metadata["training_config_sha256"],
        name="metadata.training_config_sha256",
    )
    _sha256(metadata["dataset_sha256"], name="metadata.dataset_sha256")
    _sha256(
        metadata["training_dataset_sha256"],
        name="metadata.training_dataset_sha256",
    )
    _sha256(metadata["query_sha256"], name="metadata.query_sha256")
    _sha256(
        metadata["preprocessing_sha256"],
        name="metadata.preprocessing_sha256",
    )
    _sha256(
        metadata["model_space_query_sha256"],
        name="metadata.model_space_query_sha256",
    )
    actual_scale = float(metadata["physical_scale"])
    if not np.isfinite(actual_scale) or not np.isclose(
        actual_scale, context.physical_scale, rtol=1e-12, atol=0.0
    ):
        raise ValueError(
            "field bundle metadata 'physical_scale' mismatch: "
            f"expected {context.physical_scale!r}, got {metadata['physical_scale']!r}"
        )
    trace = metadata["trace"]
    if not isinstance(trace, dict):
        raise ValueError("field bundle metadata trace must be an object")
    expected_trace = {
        "backend": context.trace_backend,
        "probes": context.trace_probes,
        "seed": context.trace_seed,
    }
    if trace != expected_trace:
        raise ValueError(
            f"field bundle trace mismatch: expected {expected_trace!r}, got {trace!r}"
        )
    if context.trace_backend not in {"exact", "hutchinson"}:
        raise ValueError("trace backend must be 'exact' or 'hutchinson'")
    if context.trace_probes < 0 or (
        context.trace_backend == "exact" and context.trace_probes != 0
    ) or (
        context.trace_backend == "hutchinson" and context.trace_probes <= 0
    ):
        raise ValueError("trace probe count is inconsistent with trace backend")
    if context.n_samples <= 0:
        raise ValueError("bundle context n_samples must be positive")
    expected_arrays = frozenset().union(
        *(READOUT_REQUIREMENTS[readout_id] for readout_id in context.readout_ids)
    )
    expected_scalars = frozenset().union(
        *(SCALAR_REQUIREMENTS[readout_id] for readout_id in context.readout_ids)
    )
    if set(bundle.arrays) != expected_arrays or set(bundle.scalars) != expected_scalars:
        raise ValueError(
            "field bundle primitive inventory mismatch: "
            f"arrays expected={sorted(expected_arrays)}, got={sorted(bundle.arrays)}; "
            f"scalars expected={sorted(expected_scalars)}, got={sorted(bundle.scalars)}"
        )
    for readout_id in context.readout_ids:
        bundle.validate(readout_id)
        required_arrays = READOUT_REQUIREMENTS[readout_id]
        for name in required_arrays:
            if np.asarray(bundle.arrays[name]).shape[0] != context.n_samples:
                raise ValueError(
                    f"field {name!r} has the wrong number of rows for bundle context"
                )


@dataclass(frozen=True)
class FieldBundle:
    arrays: Mapping[str, npt.NDArray[Any]]
    scalars: Mapping[str, float | int]
    metadata: Mapping[str, Any]

    def validate(self, readout_id: str) -> None:
        if readout_id not in READOUT_REQUIREMENTS:
            raise ValueError(f"unknown readout_id: {readout_id}")
        missing_arrays = READOUT_REQUIREMENTS[readout_id] - self.arrays.keys()
        missing_scalars = SCALAR_REQUIREMENTS[readout_id] - self.scalars.keys()
        if missing_arrays or missing_scalars:
            raise ValueError(
                f"incomplete {readout_id} field bundle: "
                f"missing arrays={sorted(missing_arrays)}, "
                f"missing scalars={sorted(missing_scalars)}"
            )
        batch_sizes: set[int] = set()
        for name in READOUT_REQUIREMENTS[readout_id]:
            array = np.asarray(self.arrays[name])
            if array.ndim == 0:
                raise ValueError(f"field {name} must have a batch dimension")
            if not np.issubdtype(array.dtype, np.number):
                raise ValueError(f"field {name} is not numeric")
            if not np.isfinite(array).all():
                raise ValueError(f"field {name} contains non-finite values")
            batch_sizes.add(int(array.shape[0]))
        if len(batch_sizes) != 1:
            raise ValueError(f"field batch sizes disagree: {sorted(batch_sizes)}")


def load_field_bundle(npz_path: Path, metadata_path: Path) -> FieldBundle:
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    with metadata_path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, dict):
        raise ValueError("field metadata must be a JSON object")
    if set(raw) != {"scalars", "metadata"}:
        raise ValueError(
            "field metadata JSON must contain exactly 'scalars' and 'metadata'"
        )
    scalars = raw.get("scalars")
    metadata = raw.get("metadata", {})
    if not isinstance(scalars, dict) or not isinstance(metadata, dict):
        raise ValueError("metadata JSON requires object-valued scalars and metadata")
    return FieldBundle(arrays=arrays, scalars=scalars, metadata=metadata)


def evaluate_bundle(readout_id: str, bundle: FieldBundle) -> npt.NDArray[np.float64]:
    bundle.validate(readout_id)
    a = bundle.arrays
    s = bundle.scalars
    if readout_id == "diffusion_flipd_full":
        result = readouts.diffusion_flipd(
            a["score"],
            a["score_divergence"],
            sigma=float(s["sigma"]),
            ambient_dim=int(s["ambient_dim"]),
        )
    elif readout_id == "fm_affine_response":
        result = readouts.affine_fm_response(
            a["velocity_divergence"],
            ambient_dim=int(s["ambient_dim"]),
            alpha_log_derivative=float(s["alpha_log_derivative"]),
            log_noise_ratio_derivative=float(s["log_noise_ratio_derivative"]),
        )
    elif readout_id == "fm_affine_full":
        result = readouts.affine_fm_full(
            a["velocity"],
            a["velocity_divergence"],
            a["score"],
            a["evaluation_point"],
            ambient_dim=int(s["ambient_dim"]),
            alpha_log_derivative=float(s["alpha_log_derivative"]),
            log_noise_ratio_derivative=float(s["log_noise_ratio_derivative"]),
        )
    elif readout_id == "fm_rectified_response":
        result = readouts.rectified_flow_response(
            a["velocity_divergence"],
            t=float(s["t"]),
            ambient_dim=int(s["ambient_dim"]),
        )
    elif readout_id == "fm_rectified_full":
        result = readouts.rectified_flow_full(
            a["velocity"],
            a["velocity_divergence"],
            a["data_point"],
            t=float(s["t"]),
            ambient_dim=int(s["ambient_dim"]),
        )
    elif readout_id == "sb_forward_response":
        result = readouts.sb_forward_response(
            a["forward_drift_divergence"],
            time_to_go=float(s["time_to_go"]),
            ambient_dim=int(s["ambient_dim"]),
        )
    elif readout_id == "sb_forward_full":
        result = readouts.sb_forward_full(
            a["forward_drift"],
            a["forward_drift_divergence"],
            time_to_go=float(s["time_to_go"]),
            diffusivity=float(s["diffusivity"]),
            ambient_dim=int(s["ambient_dim"]),
        )
    elif readout_id == "sb_current_full":
        result = readouts.sb_current_full(
            a["current_velocity"],
            a["current_velocity_divergence"],
            a["score"],
            time_to_go=float(s["time_to_go"]),
            ambient_dim=int(s["ambient_dim"]),
        )
    elif readout_id == "nf_scale_conditioned_fixed":
        result = readouts.nf_fixed_density(
            a["scale_velocity"],
            a["scale_velocity_divergence"],
            a["score"],
            ambient_dim=int(s["ambient_dim"]),
        )
    elif readout_id == "nf_calibrated_native":
        result = readouts.nf_calibrated_native(
            a["scale_velocity_divergence"], ambient_dim=int(s["ambient_dim"])
        )
    elif readout_id == "cnf_calibrated_native":
        result = readouts.cnf_calibrated_native(
            a["velocity_divergence"],
            log_scale_derivative=float(s["log_scale_derivative"]),
            ambient_dim=int(s["ambient_dim"]),
        )
    else:  # guarded by validate, retained for static exhaustiveness
        raise AssertionError(readout_id)
    return np.asarray(result, dtype=np.float64)
