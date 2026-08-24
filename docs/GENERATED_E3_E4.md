# Generated E3/E4 extension

`e3_gaussian_pca` and `e4_sphere_pca_radius1` are an explicit local extension.
Their generator classes are present at pinned upstream revision
`2dcb8e41015f53413ff1ddd049bb006c81a5df52`, but upstream does not include them
in `prepare_all`, its README dataset table, or the canonical exact archive.
They must never be reported as exact-archive or paper-parity data.

The production command requires both paths explicitly:

```bash
uv run lid-benchmarks-generate-e3-e4 \
  --pca data/lid_benchmarks_exact/benchmarks/pca.joblib \
  --output data/generated_benchmarks
```

There are deliberately no CLI overrides for the scientific configuration. It
always invokes the pinned vendored classes in this order:

1. `GaussianPCADatasetGenerator(dim=20, seed=0)`;
2. `SpherePCADatasetGenerator(dim=20, radius=1, seed=0)`.

Both use `train=100000`, `val=1000`, and `test=1000`. The supplied PCA must be
the exact-archive `benchmarks/pca.joblib`, size `98,413` bytes and SHA-256
`532c840a040f6398911248df26f26c84ed09976f5c346446e0b964e3a582a97b`.
The utility does not contain a PCA-training or dataset-download fallback.

Generation occurs in a hidden sibling staging directory. The complete root is
published with one rename only after source, PCA, shapes, finite values, LID
constants, and every artifact hash pass validation. If the destination already
exists, it is accepted only when its complete manifest is valid for the same
source and PCA; otherwise the command fails without overwriting it.

The output manifest is
`data/generated_benchmarks/generated_e3_e4_manifest.json`. It seals:

- the pinned upstream revision and hashes of generator/runtime source files;
- the immutable generator configuration and bootstrap hash;
- the canonical PCA size, SHA-256, and stored copy;
- runtime versions and deterministic thread/environment settings;
- every split artifact's SHA-256, byte size, dtype, and shape;
- observed LID values (`20` for E3, `19` for E4);
- a content-tree digest and a whole-manifest seal.

Strict validation without generation uses the same explicit inputs:

```bash
uv run lid-benchmarks-generate-e3-e4 \
  --pca data/lid_benchmarks_exact/benchmarks/pca.joblib \
  --output data/generated_benchmarks \
  --validate-only
```

Python campaign preflight uses:

```python
from datasets.generated_e3_e4 import validate_generated_e3_e4

result = validate_generated_e3_e4(
    "data/generated_benchmarks",
    "data/lid_benchmarks_exact/benchmarks/pca.joblib",
    checkout="lid_benchmarks",
)
```

The campaign should additionally hash the manifest file into its input
provenance and retain `result.upstream_revision` and `result.pca_sha256`.
