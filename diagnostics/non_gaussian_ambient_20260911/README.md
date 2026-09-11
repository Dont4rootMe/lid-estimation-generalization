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

## Measured outcome of the frozen128k study

All eight runs and numerical/replay gates completed. Primary test response MAE
uses1000 queries and a scale fixed on1000 source-train holdout queries:

| Dataset / full representation | True LID | t-Flow MAE (lambda) | PFGM++ MAE (lambda) |
| --- | ---: | ---: | ---: |
| Exp /30 coefficients | 2 | .502301 (.353553) | .417934 (.0625) |
| Spiral /30 coefficients | 1 | .216052 (2) | .236663 (1.414214) |
| Exp /784 pixels | 2 | .478396 (.5) | .721105 (.353553) |
| Spiral /784 pixels | 1 | .294483 (8) | .324088 (2) |

This is **not a learned endpoint-quality pass**. Six point estimates pass the
descriptive MAE<.5 threshold; all eight are worse than the separately selected
zero-head response control. Image minimum-scale exact traces range65.2–352.4,
far from their continuous posteriors. Kneedle MAE exceeds.5 in all eight.
Fine generation geometry also fails to reproduce the thin Exp radius and the
Spiral radius/phase relation. Full-coordinate finite differences, independent
image probes and sampler refinement distinguish these from a trace arithmetic
bug. The supplementary Student density readout is retained even where it fails.

The measured interpretation is versioned in `assessment_128k_results.tex`.
It describes these exact saved runs; write a new assessment for new trainings.
The producing source remains b90a2b7. Later commits add diagnostics/reports only.

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
python diagnostics/non_gaussian_ambient_20260911/check_student_full.py \
  --output artifacts/non_gaussian_ambient_20260911/student_full_math_gate.json
python diagnostics/non_gaussian_ambient_20260911/evaluate_student_full.py --watch
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

`generation_geometry.py` adds an explicitly posthoc check of the saved samples'
Exp relative radius and Spiral wrapped phase, alongside full ambient normal
error, a float32 round-trip control, and paired512/1024-step results. No sample
or trajectory is projected or changed by this diagnostic. Display coordinates
are obtained only for evaluation from the known renderer.

The [supplemental generation check](generation_control.md) uses512 native ODE
samples per final model, checks Heun step refinement and measures full-ambient
distribution/normal errors. It can run as a separate watcher on GPU1. A good
denoising loss does not replace this distribution check.

The [supplementary native density readout](student_full_protocol.md) reconstructs
Student density dilation from the same posterior and one extra scale derivative.
It is verified against exact mixture kernels and a homogeneous boundary, not a
Gaussian correction substitution. Its separate holdout selection is persisted
before its own test inference. The original response campaign is unchanged.
This extension was derived after training-prefix diagnostics; its timing and
the distinction between exact identities and learned fit are explicit.

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
The additional `flat_no_skip_control.py` keeps the same core/draw sequence and
removes the analytic noisy-input copy. At4000 steps its response at lambda1/32
is1.983, but denoising risk35.977 exceeds the retained baseline's15.928; at
lambda1/256 its risk is3986.17 versus32.272. It therefore also fails to justify
adoption, despite a better trace. All three checks are isolated from the frozen
eight-run architecture.

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

For replay of the existing recorded checkpoints, `assessment_128k_results.tex`
may be copied to the report's `results/assessment.tex` before `make_results.py`.
`verification.json` certifies numerical/provenance replay only; it does not
declare learned quality sufficient. The author-facing PDF and CSV are also
delivered under the paper repository's `notes/non_gaussian_ambient_20260911/`.

The output needs pdfLaTeX, Russian Babel and T2A fonts. Numerical provenance is
in `results/verification.json` and each run's model/selection/quality receipts.
Query bootstrap does not measure retraining uncertainty. A complete measurement
receipt is not by itself a claim that the learned small-noise posterior is good.
