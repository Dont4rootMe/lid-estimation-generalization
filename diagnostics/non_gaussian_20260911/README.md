# Reproduce the non-Gaussian Exp / Spiral check

The common runner supports `t_flowmatching` and `pfgmpp` on every configured
representation. This directory reproduces the bounded twelve-training study:
two new models and a Gaussian posterior rectified-flow control on Exp/Spiral,
coefficients and canonical PCA renderings. It does not launch the full507-cell
comparison. See [the frozen specification](protocol.md) before interpreting a
selected score. All computations use the canonical arrays and common128k budget.

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

`report_ru.tex` is the explanatory report source; measured table fragments and
the final interpretation are populated only after the complete study is
verified. `results/verification.json` and each producing run's manifests are the
authoritative numerical records. A short preflight proves execution only and
is rejected from benchmark aggregation.
