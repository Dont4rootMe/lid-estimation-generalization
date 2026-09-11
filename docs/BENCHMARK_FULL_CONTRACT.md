# Full benchmark contract, v8

This replaces the response-primary Student evaluation and the v7 image-NF
capacity exemption. Entry point: `python -m experiments.fair_campaign`.
Historical Hydra runners and historical result receipts retain their original
meaning; they are not accepted as current-protocol benchmark results.

## What is selected and reported

The primary quantity for every family is **Full density LID**,
`F = N + d log rho_lambda(y) / d log lambda`, at a fixed canonical query `y`.
`N` is the supplied ambient dimension; `rho_lambda` is the data distribution
smoothed by the family's kernel. The field methods estimate this quantity
through posterior identities; an inexact learned field need not itself be
an exact posterior. NF supplies a learned density and approximates its
derivative with the existing five-point OLS
readout (`ols5`), with exact float64 scale differentiation reported separately.
The other twelve interfaces expose `full` and the dependent `response` readout.

One shared selector receives only the primary Full curve:

| Track | Selection data and criterion | Candidates | Response |
|---|---|---|---|
| Known LID, supervised | Source-train holdout Full MAE against known LID | The same29 half-octaves, lambda1/256 through64 | Same checkpoint and Full-selected lambda |
| Known LID, automatic | Kneedle on each query's Full curve; no LID labels | Exactly the same29 candidates | Same Full-selected index for each query |
| E1/E5, unknown absolute LID | Kneedle on the source-train reference mean Full curve | Existing50 reference-VP-time candidates over the same physical support | Same reference lambda, transferred unchanged to each dependent cell |

The old22-point Kneedle grid remains a separately named historical diagnostic.
It cannot replace the wide-grid result. A missing pointwise Full knee gives
the original ambient-dimension fallback for **both** Full and Response; both
MAEs include that query. Its failure flag is saved. A missing E1/E5 reference
knee produces missing measurement coverage, with no invented default lambda.

Checkpoint weights still use native holdout loss and the same raw/EMA candidate
policy. They are not selected on test MAE. `selection.json` is frozen before
the test split is opened. Saved curves, Full-selected indices, secondary
predictions and all metrics are replayed before completion and aggregation.
Changing the Response curve cannot change either primary selector.

## Student Full is not the Gaussian correction

Both t-Flow and PFGM++ use the radial multivariate Student kernel. Let `nu` be
its degrees of freedom and `lambda` its per-coordinate RMS noise. Let
`b(y,lambda)` denote the learned approximation to the posterior mean, and let
`R = div_y b` be Response. Define the displacement `e=y-b`,
`a=(nu-2)*lambda^2` and `s=||e||^2/a`. The Student density identity is

```
F = [R + (N+nu)*s + e dot (d b / d log lambda)/a] / (1+s).
```

The scale derivative holds `y` fixed. In code its displacement multiplier is
detached, so differentiation applies only to `b`. The native formula is in
`models/student_density.py` and is called by the common `training.predict_lid`
API. It does not use Gaussian `R+||e||^2/lambda^2`.

Independent validity controls differentiate a finite Student-mixture density
directly for dimensions2/6/30, nu5/128 and lambda0.02/0.7/3. Another control
uses the exact posterior at the endpoint of a half-line: Full recovers true
LID1 for ambient dimensions2/30/784, whereas Response alone is deficient.
These establish the readout identity, not learned-model accuracy.

Existing Student checkpoints can be loaded through a narrow metadata migration:
only the readout-schema fields change; all stored kernel, training and
architecture settings must still match. Their old Response-selected receipts
are not valid Full benchmark receipts. No historical metrics are relabelled.

## Why eight interfaces currently share a learning problem

“Canonical posterior” means predicting the clean vector `X` from a noisy
observation expressed as `y=X+lambda*Z`. It is a coordinate choice, not a claim
that the trained network equals the exact posterior.

An affine Gaussian scheduler presents `Y_s=alpha_s*X+beta_s*Z`, with
`lambda=beta_s/alpha_s`. The current shared wrapper divides its native input
by `alpha_s` before the neural core sees it. The core therefore receives the
same standardized `X+lambda*Z`, with the same log-lambda conditioning. Its
learned output is the same function `b_theta(y,lambda)`.

For the six independent affine-FM variants, “direct velocity” and “posterior”
are different exposed outputs but the learnable core is parameterized as this
posterior in both cases. If `kappa=beta'/beta-alpha'/alpha`, the direct output
is recovered as

```
v_theta(Y_s,s) = (beta'/beta)*Y_s - alpha_s*kappa*b_theta(Y_s/alpha_s,lambda).
```

Consequently its error against the conditional velocity target is
`-alpha_s*kappa*(b_theta-X)`. The configured velocity-loss weight
`1/(alpha_s*kappa*lambda)^2` cancels that conversion. The posterior interface
uses weight `1/lambda^2` directly. Both reduce to the same coordinate-mean loss:

```
mean_coordinates ( (b_theta(X+lambda*Z,lambda)-X) / lambda )^2.
```

The rectified, log-noise and VP-trigonometric affine schedules also draw the
same log-uniform physical lambda, with the same corruption and architecture.
VE uses this same canonical input and loss. The implemented Gaussian-smoothed
Brownian bridge has `lambda=sqrt(gamma*tau)` and `alpha=1`; its log-uniform tau
sampling and declared drift-loss weighting reduce to the same law and loss.
Its public output is bridge drift; its neural core again predicts `b_theta`.
This statement concerns this concrete bridge, not arbitrary Schrödinger bridges.

Thus **VE + this bridge + six affine-FM interfaces form one mathematically
equivalent learning problem under the current recipe**. A paired control copies
one nonzero core into all eight models and compares losses and full parameter
gradients using identical random draws. The test rejects a relative gradient
discrepancy above1e-10. Floating-point evaluation order can still perturb long
training trajectories; it does not make these eight distinct objectives.

The native VP diffusion baseline and the separate Gaussian rectified-flow
baseline keep their own time distributions/objectives. They are not part of
this equivalence statement. Student kernels and the NF likelihood objective
also remain distinct. Identical backbone alone would not make them equivalent.

**Implication:** this canonicalized track tests compatible interfaces to a
common learned function. It cannot establish scheduler-dependent optimization
advantages. To study those advantages, the intervention would retain native
`Y_s` input amplitudes instead of dividing by `alpha_s`, with other choices
controlled. That is a separate experiment, not a silent change in this fix.
No training objective, scheduler or clipping policy was changed here.

## Pinned data and E5 pairs

E5 compares each transformed image with its corresponding reference image:
`mean_i |(F_transformed_i-F_reference_i)-expected_delta|`. Pairing different
images changes that metric even if their class labels match. Merely checking
row numbers0,1,2,... and repeated labels would miss a within-class permutation.

`configs/fair_comparison/input_manifest.json` pins file hashes and sizes for
all39 representations. It is distilled from the input inventory accompanying
rerun64ae8a405a42c248d14c and retains that inventory's hash and source-preflight
identities. The35 canonical representations remain separate from the four
generated E3/E4 representations and their fixed generated snapshot. This
change does not claim a fresh download or fresh authentication of the archive.

The runner verifies each loaded split against those expected bytes before
using it. Test verification happens only after the scale receipt is frozen.
Replay and E5 accounting require the pinned manifest and expected file hashes,
as well as the correct reference dataset, representation, row IDs and labels.
A regression permutes features within classes while keeping labels, shapes,
file sizes and row numbers unchanged; both input loading and a forged receipt
are rejected. This closes an acceptance gap; it is not evidence that earlier
canonical arrays were actually mispaired. Internal correctness of the original
canonical transformation remains the benchmark's data-generation contract.

## Image NF capacity

The NF remains an exactly normalized, invertible RealNVP with spatial U-Net
conditioners. The current rule searches the declared widths and even coupling
counts solely for the closest parameter count to the shared field. It does
not use task-specific validation quality. The selected architecture is two
couplings with width4 conditioners and opposite checkerboard masks.

| Image channels | Field U-Net8 parameters | NF parameters | Relative excess |
|---|---:|---:|---:|
| 1 | 97,905 | 106,012 | 8.28% |
| 3 | 98,195 | 106,452 | 8.41% |

Both satisfy the same10% capacity tolerance used for vectors. The U-Net and
coupling implementation are unchanged; configuration controls their width and
count. Tests use nonzero conditioners to check inversion, analytic logdet
against a full Jacobian determinant, and log-scale differentiation against
finite differences. Equal parameters and update counts do not imply equal
FLOPs or equal learned density quality.

## Reproduce verification

```
python -m pytest -q tests/unit/test_full_benchmark_contract.py \
  tests/unit/test_fair_comparison.py tests/unit/test_native_backbone_parity.py \
  tests/unit/test_non_gaussian_fields.py tests/unit/test_fair_measurements.py \
  tests/unit/test_fair_outputs.py tests/unit/test_training.py \
  tests/unit/test_vp_training.py tests/unit/test_affine_flow_training.py \
  tests/unit/test_ambient_models.py
python diagnostics/native_backbone_parity_20260911.py --output artifacts/full_route_audit
python -m experiments.fair_campaign verify --receipt /path/to/run/complete.json
```

Technical runs made with `--preflight --steps 8` exercise training, checkpoint
reload, frozen Full selection, both readouts, Kneedle and replay on real input
files. They do not establish full-budget LID quality and the aggregate command
explicitly rejects them as benchmark results. The final full-budget matrix
still requires new matching-protocol runs.

## Verification of this change

- The combined targeted regression suite passes223 tests. The existing kneed
  constant-curve warning is exercised by the missing-knee negative control.
- All507 model configurations construct with the declared shared field and
  NF parameter counts. Source-train/test pin metadata covers all39 cells.
- Ten real eight-update runner checks complete: NF/Spiral images;
  t-Flow/Exp coefficients and Spiral images; PFGM++/Spiral coefficients and Exp
  images; VP/Exp images; and VE/PFGM++ reference-plus-adddim4 E5 pairs.
  All arrays/receipts replay and export42 records. In these short runs, both
  E5 references have a boundary knee rejected by the existing reference rule;
  all four E5 cells correctly preserve missing coverage. No successful E5
  quality measurement is claimed. Two known-LID cells retain their supervised
  boundary choices. The passing gate is execution/accounting, not LID quality.
- Eight-interface paired loss/gradient checks on30-coordinate vectors and
  small full images give maximum relative gradient gap4.05e-13.
- Four historical128k Student vector checkpoints load without file changes.
  The new Full API and the previous independently checked native Student
  formula agree exactly on three queries at each of lambda0.5 and32: eight
  comparisons, maximum observed difference0. These are API checks, not a new
  scale-selection or quality campaign.

Run receipts, source snapshots, checkpoints and test logs are under
`artifacts/full_contract_checks/`; the507-route audit is under
`artifacts/full_route_audit/`. The reproducible replay/export/equivalence
command is:

```
python diagnostics/full_contract_validation_20260911.py \
  --runs artifacts/full_contract_checks \
  --output artifacts/full_contract_checks/verification.json
```

The optional `--legacy-root` and `--canonical-root` arguments repeat the
historical-weight identity checks; the handoff preserves the separately run
results in `legacy_validation.json`.
