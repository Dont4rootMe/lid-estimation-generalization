# Shared native backbones: upstream handoff

The common runner now applies the author-selected stock image width8 to all
twelve field interfaces, including t-Flow and PFGM++. Previously its resolver
and capacity reference still hardcoded width32 despite the new specification.
The declared widths now control both model construction and parameter counts.

Protocol: `full_ambient_shared_stock8_retained_nf_v7`.
Branch: `fix/native-backbone-parity-20260911`, based on `066c987` (the previous
native-model integration). This branch includes both added model families and
the earlier common-runner changes; the final backbone commit alone depends on
that preceding integration.

| Representation | All twelve field interfaces | Normalizing flow |
|---|---|---|
| Full30 coefficients | Same `SpectralResidualCore`, width512, four blocks, 3005214 parameters | Existing full-dimensional RealNVP; capacity matched within10% |
| Grayscale images | Same `ImageUNet`, width8, 97905 parameters | Retained eight width9 U-Net conditioners, 905008 parameters |
| RGB images | Same `ImageUNet`, width8, 98195 parameters | Retained eight width9 U-Net conditioners, 908928 parameters |

The NF exception is explicit. Shrinking its conditioners to rematch the smaller
field would change the author's retained NF baseline. Its source, settings and
parameter counts are preserved. The vector route retains every supplied
coordinate. The image route retains every pixel; neither uses a fitted support
projection. Native kernels, targets, loss weights and noise/time sampling stay
family-specific. The common128k budget, optimizer and selection rules remain.

## Verification

- All507 resolved configurations (39 representations by13 interfaces) were
  constructed on the meta device. Actual core classes, state-dict shapes and
  parameter counts agree across the twelve fields within every representation.
- Comparing against `066c987`, exactly360 image-field configurations change
  only `image_width: 32 -> 8`. The other147 configurations, including every NF,
  and their parameter counts are identical.
- The103 focused unit/regression tests pass. They cover shared native adapters,
  actual short training and reload, derivative checks, NF inverse/log determinant,
  selection/output accounting, configurable widths and stale-cache prevention.
- Twelve additional eight-update technical fits use only source-train Exp and
  Spiral data: Gaussian RF/t-Flow/PFGM++ by two full representations. All reload
  bitwise; the maximum directional finite-difference gap is3.18e-10. These are
  execution checks, not estimates of learned manifold or LID quality.
- The shared image class and image NF source are byte-identical to the retained
  `d296210` implementation. Existing checkpoints and measured128k outputs were
  not rewritten. Previous t-Flow/PFGM++ image results remain width32/v6 results.

## Commands

Use the common runner; historical Hydra campaign configurations retain their
historical recipes. The following matrix command performs no training:

```bash
python -m experiments.fair_campaign matrix --output artifacts/fair_v7_matrix.json
```

Regression checks, from the repository root with its training dependencies:

```bash
python -m pytest -q \
  tests/unit/test_native_backbone_parity.py \
  tests/unit/test_fair_comparison.py \
  tests/unit/test_non_gaussian_fields.py \
  tests/unit/test_ambient_models.py \
  tests/unit/test_fair_measurements.py \
  tests/unit/test_fair_outputs.py
```

Reproduce the507-route audit and optional technical fits in a fresh output
directory. Omit `--canonical-root` to perform only model construction. The data
root contains `e6_exp_pca/train/` and `e1_spiral_pca/train/`; no test file is read.

```bash
PYTHONPATH=. CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python diagnostics/native_backbone_parity_20260911.py \
  --output artifacts/native_backbone_check \
  --canonical-root /path/to/canonical/benchmarks --device cuda
```

The audit writes complete route/configuration records, technical checkpoint
receipts and source hashes. Numerical artifacts from this handoff are linked
from `notes/native_backbone_parity_20260911/` in the paper repository. No full
benchmark campaign or new training-quality claim accompanies this commit.
