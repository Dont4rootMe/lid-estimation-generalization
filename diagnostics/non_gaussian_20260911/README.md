# Reproduce the non-Gaussian Exp / Spiral check

HISTORICAL PROJECTED STUDY: reproduce this directory at commit fe8c94b. The
author's14:43 override excludes these results from the paper, including
ablations. The current unprojected work is in
`diagnostics/non_gaussian_ambient_20260911/`; do not run this old campaign with
the changed common v6 routing or use its scores to certify the new geometry.

The common runner supports `t_flowmatching` and `pfgmpp` on every configured
representation. This directory reproduces the bounded twelve-training study:
two new models and a Gaussian posterior rectified-flow control on Exp/Spiral,
coefficients and canonical PCA renderings. It does not launch the full507-cell
comparison. See [the frozen specification](protocol.md) before interpreting a
selected score. All computations use the canonical arrays and common128k budget.

The original study completed all twelve128k runs and passed measurement replay,
matched data/split/normalization/capacity checks and trained derivative checks.
The maximum trained trace/finite-difference discrepancy was1.46e-7. On1000 test
queries per row, with one response scale selected using1000 source-train holdout
queries, the measured response MAE was:

| Dataset / representation | True LID | t-Flow | PFGM++ | Gaussian RF response control |
|---|---:|---:|---:|---:|
| Exp / coefficients | 2 | .349282 | .325757 | .162739 |
| Spiral / coefficients | 1 | .213268 | .220503 | .255999 |
| Exp / PCA render | 2 | .243428 | .295597 | .173966 |
| Spiral / PCA render | 1 | .280319 | .268318 | .286097 |

All eight new-model rows pass the declared MAE<.5 practical check. This does not
certify small-noise accuracy: the learned response stays near the fitted
linear-span rank on the smallest grid scales. The t-Flow Spiral rendering
selects the upper boundary64. Its active conditional kernel width in raw units
is.01413, comparable to.01607 at the coefficient model's lambda2.828; this unit
conversion does not remove the boundary flag. Its denoising-risk improvement
also has a paired query interval crossing zero. Common-grid Kneedle remains
weaker, with new-model Spiral MAE.778--.989. No test-based changes to training,
tail parameters or selection were made. The two methods use different native
losses/samplers and belong to the same radial Student kernel class; the study
is not a kernel-only causal comparison or a retraining uncertainty estimate.

These runs use the existing fit-derived affine-span projection for both
representations, identically across methods. This explicitly supplies global
support structure and constrains the posterior; it is more than normalization.
The author has subsequently objected to this aided representation protocol.
Retain the numbers as a labelled study of that frozen recipe, not evidence of
unassisted ambient-image learning. Quality under a replacement ambient-space
recipe must be measured anew; the tested U-Net integration alone does not prove it.

Pre-training checks write their fresh numerical receipts under
`artifacts/non_gaussian_20260911/`, so they preserve the clean checkout required
by the launcher. The committed `reference_checks.json` is retained evidence
from the original validation run, not a file overwritten during reproduction.

Run these commands from a clean repository checkout with its training and test
dependencies installed. `PYTHONPATH=.` makes the standalone diagnostic scripts
use this checkout's code. The fixed output directory must be fresh. The original
producing model source is commit d1f978f; later commits add tests/diagnostics only.
The scripts here replace the original workspace-specific root by the checkout
root. The scientific operations and selection rules are the same.

```bash
export PYTHONPATH=.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2

python diagnostics/non_gaussian_20260911/prepare_data.py \
  --archive data/benchmarks.zip \
  --inventory diagnostics/non_gaussian_20260911/input_inventory.json \
  --output artifacts/non_gaussian_20260911/data/benchmarks

python -m pytest -q tests/unit/test_non_gaussian_fields.py \
  tests/unit/test_fair_comparison.py tests/unit/test_fair_measurements.py \
  tests/unit/test_fair_outputs.py

python diagnostics/non_gaussian_20260911/check_reference.py
python diagnostics/non_gaussian_20260911/run_preflights.py
python diagnostics/non_gaussian_20260911/population_curves.py
python diagnostics/non_gaussian_20260911/check_query_precision.py
python diagnostics/non_gaussian_20260911/run_campaign.py
python diagnostics/non_gaussian_20260911/evaluate_quality.py --watch
python diagnostics/non_gaussian_20260911/make_results.py
cp diagnostics/non_gaussian_20260911/report_ru.tex artifacts/non_gaussian_20260911/report/
cd artifacts/non_gaussian_20260911/report
pdflatex -interaction=nonstopmode -halt-on-error report_ru.tex
pdflatex -interaction=nonstopmode -halt-on-error report_ru.tex
```

The archive is authenticated against SHA256
`ce0d153a1a78a3a752b29ec2e60167134b6b20c3249db2fe92f9fc1b8b8a9181`.
The small inventory is the unchanged Exp/Spiral subset of the original producing
archive inventory; each extracted array is checked separately. The extractor
does not alter labels, coordinates or renders. Exp's train-label correction is
implemented by the existing registry at load time, as in all other methods.

The campaign launcher uses two model processes per GPU on devices0/1. Once the
launcher has written `campaign.json`, the quality watcher can also run from a
second terminal; it consumes only completed, verified models. Completed quality
summaries are named `quality/summary.json`, so they cannot be mistaken for cell
completion receipts by benchmark aggregation.

`make_results.py` requires all12 cells and verifies the common data, indices,
normalization, budget, capacity, producing source, immutable primary selections
and saved predictions. Its Gaussian response-selected row is an explicitly
predeclared additional diagnostic; the benchmark's full-selected dependent
response remains unchanged. It writes complete metrics, fixed-checkpoint query
uncertainty, holdout selector bootstraps and scientific PDF/PNG curves, including
a linear-scale figure for one fixed point.

Report outputs are written to `artifacts/non_gaussian_20260911/report/`, keeping
the producing checkout clean. The PDF build additionally needs pdfLaTeX with
Russian Babel and T2A fonts. `report_ru.tex` is the explanatory report source; measured table fragments and
the final interpretation are populated only after the complete study is
verified. `results/verification.json` and each producing run's manifests are the
authoritative numerical records. A short preflight proves execution only and
is rejected from benchmark aggregation.
