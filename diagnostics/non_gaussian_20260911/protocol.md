# Non-Gaussian model integration and Exp/Spiral check

Frozen before full-budget training, 2026-09-11. This is an implementation and
quality check, not a manuscript result or a full benchmark rerun.

## Question and comparator

Can the requested t-Flow and PFGM++ models train and estimate LID under the
existing common comparison, without applying a Gaussian readout to their
non-Gaussian corruption? The earlier 12k Student-t pilot achieved a low selected
Spiral MAE while missing its small-scale response. Therefore a selected score
alone will not be interpreted as successful manifold learning.

Train each new model and the existing posterior rectified-flow Gaussian control
on Exp and Spiral, for both canonical coefficients and their canonical PCA image
rendering. These PCA datasets use the same fit-covariance spectral backbone by
the existing dataset routing rule, including the rendered representation. The
new image U-Net route is independently tested, but these four cells are not a
U-Net training study. No previous pilot checkpoint is reused.

## Common quantities

- Unmodified canonical archive arrays, checked against the producing inventory.
  Exp has true LID 2; Spiral has true LID 1. Registry correction exposes Exp's
  known target without changing its original train label file.
- Identical source-train fit/holdout indices: 99000/1000; all 1000 test queries.
  Fit-only centering, scalar RMS normalization and numerical covariance span.
- One seed, 128000 optimizer updates, batch256, AdamW 2e-4, weight decay1e-6,
  gradient clip1, common antithetic pairing, last8000-step cosine decay/EMA.
  Checkpoint selection by native holdout loss using the common raw/EMA policy.
- Same spectral core and parameter count on each representation. Native neural
  input/output scaling can differ as required by the corruption law.
- Same 29 half-octave RMS noise scales, 1/256 through64, for source-train
  holdout-MAE selection and pointwise Kneedle; no grid truncation after seeing
  results. Exact active trace for every vector field, no stochastic probes.
- The test split is loaded for inference only after the checkpoint and primary
  scale receipt are frozen. Save every pointwise scale curve and selector result.
- Equal update/example budgets do not imply equal FLOPs or native objectives.

## Declared native differences

t-Flow uses a radial multivariate Student-t source with nu=5 and one chi-square
denominator per sample. Native time is uniform on the common support and the
objective is native noise prediction MSE, implemented equivalently through the
posterior. Reference: https://arxiv.org/html/2410.14171v2 Appendix B, Eq.164-166.

PFGM++ uses augmented dimension128. Its Poisson kernel equals the multivariate
Student-t kernel with nu=128. The native EDM denoising loss and lognormal native
noise sampling are used, truncated by inverse CDF to the shared support. In
fit-RMS units sigma_data=1; the original EDM lognormal mean -1.2 at sigma_data=.5
becomes -1.2+log(2), with std1.2 unchanged. No clipping of noise radii/tails.
References: https://proceedings.mlr.press/v202/xu23m/xu23m.pdf and
https://github.com/Newbeeer/pfgmpp/blob/main/training/loss.py.

The two methods have different native fields and training conventions, but
belong to the same radial Student-t kernel class at different tail parameters.
They are not two unrelated non-Gaussian kernel classes. Tail parameters are
fixed across datasets and are not selected by LID.

Evaluation lambda is per-coordinate RMS corruption in normalized input units.
Native Student scale sigma=lambda*sqrt((nu-2)/nu). t-Flow time is
t=sqrt(nu/(nu-2))/(sqrt(nu/(nu-2))+lambda). PFGM radius is
r=lambda*sqrt(nu-2). Gaussian-control lambda is unchanged.

For a data span of rank k in ambient dimension N, observed normal component n
cannot be discarded under Student-t noise. Its exact conditional active kernel
has df=nu+N-k and RMS scale squared
((nu-2)*lambda^2+||n||^2)/(nu+N-k-2). Supplying this effective scale to the common
k-dimensional core retains all posterior-relevant information without extra
learned features. All coordinates are retained by the image U-Net route.

The new models use R=trace of the posterior-mean Jacobian. This equals the native
t-Flow expression N*t+t*(1-t)*div(v) and the native PFGM expression N-r*div(u).
Gaussian R plus squared posterior displacement is rejected for these kernels.

## Evaluation and acceptance

First pass the independent radial-law, loss, conditional-kernel, native-field,
finite-difference, checkpoint and common-runner validity gates. A failed gate is
an implementation defect to repair before a quality claim.

Primary benchmark controls retain their original readout/selection convention.
For an additional direct response comparison, select the Gaussian control's R
scale from its source-train holdout R curve, freeze a separate receipt, then
evaluate test R. Label this diagnostic separately from the benchmark's primary
full-selected dependent response. Do not select or tune from test curves.

Report test mean absolute error, selected RMS scale, mean response, and common
Kneedle MAE/coverage. Query-bootstrap uncertainty is conditional on the fixed
training and holdout selection; it is not retraining uncertainty. A half LID
unit (MAE<0.5 on each task/representation) is a prospective practical score check,
not a theorem or a guarantee for other benchmark cells.

Inspect the native validation-loss trajectory and the complete R(lambda)
curves. Independently compare denoising/model geometry with the known continuous
laws and/or generated samples, including numerical refinement checks. A low
selected MAE at a large lambda does not establish small-noise convergence. If
the continuous posterior itself lacks a useful finite-scale plateau, distinguish
that effect from a learned posterior-Jacobian discrepancy. Any further repair
must be declared, use a general rule rather than test-specific tuning, and have
its affected comparison arms rerun consistently.
