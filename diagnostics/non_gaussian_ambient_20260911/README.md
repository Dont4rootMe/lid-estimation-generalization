# Full-ambient t-Flow / PFGM++ integration and evaluation

This is the author's requested eight-run study: two methods, Exp/Spiral, and
two independent representations (all30 coefficients or all784 pixels).
Read [the frozen protocol](protocol.md). Model/config source is b90a2b7;
subsequent diagnostic commits do not change producing model source.
No fitted low-rank projection or analytic normal-coordinate completion is used.
The historical projected study is excluded from the paper, including ablations.

Both methods use the common unrestricted architecture in each representation,
128000 updates, batch256, fit99000/holdout1000/test1000 and identical optimization.
Their native kernels, samplers and losses remain distinct. t-Flow uses radial
Student-t5 and native noise MSE; PFGM++ uses augmented dimension128 and the
native EDM denoising objective. LID is posterior response, not a Gaussian score
substitution. All13 common interfaces support the unrestricted geometry, but
only the requested eight new-model cells are trained in this study.

## Reproduce from a fresh output directory

Use a clean checkout with the repository's training/test dependencies. The
canonical archive is supplied separately; no private download URL is embedded.
The extractor authenticates archive and per-array hashes. The old diagnostic
directory below supplies only the data extractor and continuous-law integrator;
its projected model checkpoints and learned predictions are never reused.

```bash
export PYTHONPATH=.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2

python diagnostics/non_gaussian_20260911/prepare_data.py \
  --archive data/benchmarks.zip \
  --inventory diagnostics/non_gaussian_20260911/input_inventory.json \
  --output artifacts/non_gaussian_ambient_20260911/data/benchmarks

python -m pytest -q tests/unit/test_ambient_models.py \
  tests/unit/test_non_gaussian_fields.py tests/unit/test_fair_comparison.py \
  tests/unit/test_fair_measurements.py tests/unit/test_fair_outputs.py

python diagnostics/non_gaussian_20260911/check_reference.py
python diagnostics/non_gaussian_20260911/population_curves.py \
  --data artifacts/non_gaussian_ambient_20260911/data/benchmarks \
  --output artifacts/non_gaussian_ambient_20260911/population
python diagnostics/non_gaussian_20260911/check_query_precision.py \
  --population artifacts/non_gaussian_ambient_20260911/population
python diagnostics/non_gaussian_ambient_20260911/run.py --preflight
python diagnostics/non_gaussian_ambient_20260911/run.py
python diagnostics/non_gaussian_ambient_20260911/evaluate_quality.py --watch
python diagnostics/non_gaussian_ambient_20260911/check_sampler.py
python diagnostics/non_gaussian_ambient_20260911/generation_quality.py --watch
python diagnostics/non_gaussian_ambient_20260911/make_results.py
```

The launcher uses two workers on each of GPUs0/1. The quality watcher may run
in a second terminal after `full_budget/campaign.json` exists; it waits for
verified completion receipts. Existing output directories are not overwritten.
Full model/config source and protocol hashes are frozen and verified at exit.
Short preflights are rejected as benchmark results. No fit or test data are
projected; exact continuous-law references are evaluation-only integrals of
the known data generator.

The quality study preserves the primary holdout choice and all1000 test
queries. It adds exact full-coordinate traces on32 fixed holdout points,
independent64-probe repeats on image test, full-coordinate finite differences,
continuous-law posterior comparisons, and paired denoising on128 fresh-noise
test points with eight observations each. Neither tail parameters nor training
recipe are tuned using these diagnostics.

The [supplemental generation check](generation_control.md) uses512 native ODE
samples per final model, checks Heun step refinement and measures full-ambient
distribution/normal errors. It can run as a separate watcher on GPU1. A good
denoising loss does not replace this distribution check.

Read [the supplemental selection control](selection_control.md) before treating
MAE<.5 as evidence of manifold learning: the zero residual head can pass that
score through supervised scale selection. The separate
`flat_scaling_control.py` is a paired synthetic parameterization diagnostic,
not one of the eight benchmark trainings and not permission to change their
frozen source. Its output is isolated under `flat_scaling_control/`.
`flat_linear_control.py` tests one additional unrestricted learned-linear
shortcut on the same synthetic draw sequence. Neither short intervention was
adopted: changing the scalar coefficient worsened the minimum-scale normal
error, and the linear shortcut still left a response25.77 on a clean plane
of dimension2 after4000 steps. These are limited negative diagnostics, not
claims that a larger budget could never improve either alternative.

## Report

`make_results.py` verifies all eight receipts and creates CSV/tables/plots under
`artifacts/non_gaussian_ambient_20260911/report/results/`. It includes holdout
selector bootstraps and both mean and single-query LID curves, with logarithmic
and linear scale axes. The report's `assessment.tex` is a measured interpretation
written only after results exist; it must not be fabricated from preflights.

After adding that interpretation, compile the standalone Russian report:

```bash
cp diagnostics/non_gaussian_ambient_20260911/report_ru.tex \
  artifacts/non_gaussian_ambient_20260911/report/
cd artifacts/non_gaussian_ambient_20260911/report
pdflatex -interaction=nonstopmode -halt-on-error report_ru.tex
pdflatex -interaction=nonstopmode -halt-on-error report_ru.tex
```

The output needs pdfLaTeX, Russian Babel and T2A fonts. Numerical provenance is
in `results/verification.json` and each run's model/selection/quality receipts.
Query bootstrap does not measure retraining uncertainty. A complete measurement
receipt is not by itself a claim that the learned small-noise posterior is good.
