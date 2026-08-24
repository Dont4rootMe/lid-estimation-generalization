from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

import datasets.generated_e3_e4 as generated
from datasets.generated_e3_e4 import (
    ExtensionGenerationSpec,
    GeneratedExtensionConflictError,
    GeneratedExtensionValidationError,
    prepare_generated_e3_e4,
    validate_generated_e3_e4,
)
from utils.provenance import EXPECTED_LID_BENCHMARKS_SHA

_FAKE_BENCHMARKS = r"""
from pathlib import Path

import numpy as np


class _Base:
    dataset_name = ""
    lid_value = 0

    def __init__(
        self,
        *,
        dataset_root_dir,
        N_train,
        N_val,
        N_test,
        dim,
        seed,
        pca,
        radius=None,
    ):
        assert dim == 20
        assert seed == 0
        if radius is not None:
            assert radius == 1
        self.root = Path(dataset_root_dir)
        self.sizes = (N_train, N_val, N_test)
        self.pca = pca

    def generate(self):
        total = sum(self.sizes)
        coefficients = np.zeros((total, 30), dtype=np.float64)
        coefficients[:, :20] = np.arange(total * 20, dtype=np.float64).reshape(total, 20) / 100
        dataset = (coefficients @ self.pca.components_ + self.pca.mean_).reshape(total, 1, 28, 28)
        lid = np.full(total, self.lid_value, dtype=np.int64)
        offset = 0
        for split, size in zip(("train", "val", "test"), self.sizes):
            destination = self.root / self.dataset_name / split
            destination.mkdir(parents=True)
            stop = offset + size
            np.save(destination / "dataset.npy", dataset[offset:stop])
            np.save(destination / "lid.npy", lid[offset:stop])
            np.save(destination / "coefficients.npy", coefficients[offset:stop])
            offset = stop


class GaussianPCADatasetGenerator(_Base):
    dataset_name = "e3_gaussian_pca"
    lid_value = 20


class SpherePCADatasetGenerator(_Base):
    dataset_name = "e4_sphere_pca_radius1"
    lid_value = 19
""".lstrip()


_FAKE_JOBLIB = r"""
import numpy as np

__version__ = "test-double-1"


class FakePCA:
    n_components = 30
    components_ = np.zeros((30, 784), dtype=np.float64)
    mean_ = np.zeros(784, dtype=np.float64)


def load(path):
    with open(path, "rb") as stream:
        assert stream.read() == b"canonical-pca-test-double"
    return FakePCA()
""".lstrip()


def _fake_checkout(tmp_path: Path) -> Path:
    checkout = tmp_path / "lid_benchmarks"
    (checkout / "generators" / "utils").mkdir(parents=True)
    files = {
        "UPSTREAM.yaml": "revision: fake\n",
        "pyproject.toml": "[project]\nname='fake-upstream'\n",
        "uv.lock": "version = 1\n",
        "generators/__init__.py": "",
        "generators/benchmarks.py": _FAKE_BENCHMARKS,
        "generators/utils/arrows.py": "",
        "generators/utils/padded_and_downscaled.py": "",
        "generators/utils/pca.py": "",
        "joblib.py": _FAKE_JOBLIB,
    }
    for relative, content in files.items():
        path = checkout / relative
        path.write_text(content, encoding="utf-8")
    return checkout


def _inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, ExtensionGenerationSpec]:
    checkout = _fake_checkout(tmp_path)
    pca = tmp_path / "exact" / "benchmarks" / "pca.joblib"
    pca.parent.mkdir(parents=True)
    pca.write_bytes(b"canonical-pca-test-double")
    digest = hashlib.sha256(pca.read_bytes()).hexdigest()
    monkeypatch.setattr(
        generated, "EXPECTED_CANONICAL_PCA_SIZE_BYTES", pca.stat().st_size
    )
    monkeypatch.setattr(generated, "EXPECTED_CANONICAL_PCA_SHA256", digest)
    monkeypatch.setattr(generated, "_RUNTIME_SHADOW_NAMES", ())
    monkeypatch.setattr(
        generated,
        "verify_upstream_source",
        lambda path: EXPECTED_LID_BENCHMARKS_SHA,
    )
    spec = ExtensionGenerationSpec(train_samples=7, val_samples=3, test_samples=2)
    return checkout, pca, spec


def test_prepare_is_atomic_sealed_and_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, pca, spec = _inputs(tmp_path, monkeypatch)
    output = tmp_path / "generated" / "generated_benchmarks"

    created = prepare_generated_e3_e4(
        output,
        pca,
        checkout=checkout,
        runtime_command=(sys.executable,),
        _spec=spec,
    )

    assert created.created is True
    assert created.output_root == output.resolve()
    manifest = json.loads(created.manifest_path.read_text(encoding="utf-8"))
    assert manifest["canonical_exact_archive"] is False
    assert manifest["provenance_label"] == "generated-extension-not-canonical-exact"
    assert manifest["upstream"]["revision"] == EXPECTED_LID_BENCHMARKS_SHA
    assert manifest["pca"]["sha256"] == hashlib.sha256(pca.read_bytes()).hexdigest()
    assert manifest["pca"]["source_contract"]["kind"] == ("canonical-exact-archive-pca")
    assert manifest["generator_config"]["split_sizes"] == {
        "train": 7,
        "val": 3,
        "test": 2,
    }
    assert manifest["generator_config"]["generators"] == [
        {
            "class": "GaussianPCADatasetGenerator",
            "dataset_name": "e3_gaussian_pca",
            "dim": 20,
            "expected_lid": 20,
            "radius": None,
        },
        {
            "class": "SpherePCADatasetGenerator",
            "dataset_name": "e4_sphere_pca_radius1",
            "dim": 20,
            "expected_lid": 19,
            "radius": 1,
        },
    ]
    assert manifest["datasets"]["e3_gaussian_pca"]["splits"]["val"]["artifacts"]["lid"][
        "unique_values"
    ] == [20.0]
    assert manifest["datasets"]["e4_sphere_pca_radius1"]["splits"]["test"]["artifacts"][
        "dataset"
    ]["shape"] == [2, 1, 28, 28]
    assert not list(output.parent.glob(f".{output.name}.staging-*"))

    validated = validate_generated_e3_e4(
        output,
        pca,
        checkout=checkout,
        _spec=spec,
    )
    repeated = prepare_generated_e3_e4(
        output,
        pca,
        checkout=checkout,
        runtime_command=(sys.executable,),
        _spec=spec,
    )
    assert validated.created is False
    assert repeated.created is False
    assert repeated.pca_sha256 == created.pca_sha256


def test_tampered_existing_root_is_never_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, pca, spec = _inputs(tmp_path, monkeypatch)
    output = tmp_path / "generated_benchmarks"
    prepare_generated_e3_e4(
        output,
        pca,
        checkout=checkout,
        runtime_command=(sys.executable,),
        _spec=spec,
    )
    artifact = output / "e3_gaussian_pca" / "train" / "lid.npy"
    original = artifact.read_bytes()
    artifact.write_bytes(original + b"tamper")

    with pytest.raises(
        GeneratedExtensionValidationError,
        match="artifact records do not match",
    ):
        validate_generated_e3_e4(
            output,
            pca,
            checkout=checkout,
            _spec=spec,
        )
    with pytest.raises(
        GeneratedExtensionConflictError,
        match="refusing to overwrite conflicting generated output",
    ):
        prepare_generated_e3_e4(
            output,
            pca,
            checkout=checkout,
            runtime_command=(sys.executable,),
            _spec=spec,
        )
    assert artifact.read_bytes() == original + b"tamper"


def test_wrong_pca_is_rejected_before_output_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, pca, spec = _inputs(tmp_path, monkeypatch)
    output = tmp_path / "generated_benchmarks"
    pca.write_bytes(b"not-the-pinned-canonical-pca")

    with pytest.raises(
        GeneratedExtensionValidationError,
        match="does not match the canonical exact-archive",
    ):
        prepare_generated_e3_e4(
            output,
            pca,
            checkout=checkout,
            runtime_command=(sys.executable,),
            _spec=spec,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".generated_benchmarks.staging-*"))


def test_manifest_seal_detects_metadata_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, pca, spec = _inputs(tmp_path, monkeypatch)
    output = tmp_path / "generated_benchmarks"
    result = prepare_generated_e3_e4(
        output,
        pca,
        checkout=checkout,
        runtime_command=(sys.executable,),
        _spec=spec,
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    manifest["runtime"]["receipt"]["python_version"] = "tampered"
    result.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        GeneratedExtensionValidationError,
        match="seal_sha256 mismatch",
    ):
        validate_generated_e3_e4(
            output,
            pca,
            checkout=checkout,
            _spec=spec,
        )
