from __future__ import annotations

import numpy as np
import pytest

from models.fields import (
    BundleContext,
    FieldBundle,
    evaluate_bundle,
    validate_bundle_provenance,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64
SHA_0 = "0" * 64


def _context() -> BundleContext:
    return BundleContext(
        model_name="diffusion-reference",
        model_family="gaussian_diffusion",
        model_seed=7,
        checkpoint_sha256=SHA_A,
        training_config_sha256=SHA_B,
        dataset_name="fixture",
        training_dataset_sha256=SHA_C,
        dataset_sha256=SHA_D,
        representation="coordinates",
        split="validation",
        query_sha256=SHA_E,
        preprocessing_sha256=SHA_F,
        model_space_query_sha256=SHA_0,
        n_samples=2,
        scale_index=0,
        physical_scale=0.1,
        readout_ids=("diffusion_flipd_full",),
        trace_backend="exact",
        trace_probes=0,
        trace_seed=13,
    )


def _metadata(context: BundleContext) -> dict[str, object]:
    return {
        "schema_version": 1,
        "model_name": context.model_name,
        "model_family": context.model_family,
        "model_seed": context.model_seed,
        "checkpoint_sha256": context.checkpoint_sha256,
        "training_config_sha256": context.training_config_sha256,
        "dataset_name": context.dataset_name,
        "training_dataset_sha256": context.training_dataset_sha256,
        "dataset_sha256": context.dataset_sha256,
        "representation": context.representation,
        "split": context.split,
        "query_sha256": context.query_sha256,
        "preprocessing_sha256": context.preprocessing_sha256,
        "model_space_query_sha256": context.model_space_query_sha256,
        "n_samples": context.n_samples,
        "scale_index": context.scale_index,
        "physical_scale": context.physical_scale,
        "readout_ids": list(context.readout_ids),
        "trace": {
            "backend": context.trace_backend,
            "probes": context.trace_probes,
            "seed": context.trace_seed,
        },
    }


def test_bundle_requires_all_primitives() -> None:
    bundle = FieldBundle(
        arrays={"score": np.zeros((2, 3))},
        scalars={"sigma": 0.1, "ambient_dim": 3},
        metadata={},
    )
    with pytest.raises(ValueError, match="score_divergence"):
        bundle.validate("diffusion_flipd_full")


def test_bundle_evaluates_diffusion() -> None:
    bundle = FieldBundle(
        arrays={
            "score": np.zeros((2, 3)),
            "score_divergence": np.full(2, -100.0),
        },
        scalars={"sigma": 0.1, "ambient_dim": 3},
        metadata={"checkpoint_sha256": "test"},
    )
    np.testing.assert_allclose(
        evaluate_bundle("diffusion_flipd_full", bundle), np.full(2, 2.0)
    )


def test_bundle_rejects_misaligned_batches() -> None:
    bundle = FieldBundle(
        arrays={
            "forward_drift": np.zeros((2, 3)),
            "forward_drift_divergence": np.zeros(3),
        },
        scalars={"time_to_go": 0.1, "diffusivity": 1.0, "ambient_dim": 3},
        metadata={},
    )
    with pytest.raises(ValueError, match="batch sizes"):
        bundle.validate("sb_forward_full")


def test_learned_bundle_provenance_binds_every_identity() -> None:
    context = _context()
    bundle = FieldBundle(
        arrays={
            "score": np.zeros((2, 3)),
            "score_divergence": np.full(2, -100.0),
        },
        scalars={"sigma": 0.1, "ambient_dim": 3},
        metadata=_metadata(context),
    )
    validate_bundle_provenance(bundle, context)

    forged = dict(bundle.metadata)
    forged["dataset_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="dataset_sha256.*mismatch"):
        validate_bundle_provenance(
            FieldBundle(bundle.arrays, bundle.scalars, forged), context
        )


def test_learned_bundle_rejects_empty_or_unversioned_metadata() -> None:
    context = _context()
    bundle = FieldBundle(
        arrays={
            "score": np.zeros((2, 3)),
            "score_divergence": np.full(2, -100.0),
        },
        scalars={"sigma": 0.1, "ambient_dim": 3},
        metadata={},
    )
    with pytest.raises(ValueError, match="metadata schema mismatch"):
        validate_bundle_provenance(bundle, context)

    invalid = _metadata(context)
    invalid["checkpoint_sha256"] = "test"
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        validate_bundle_provenance(
            FieldBundle(bundle.arrays, bundle.scalars, invalid), context
        )
