# Requested native models in the full supplied ambient spaces

The author's 2026-09-11 14:43 override excludes every fitted support projection
from both representations and excludes the earlier projected results from the
paper, including ablations. Preserve those artifacts as history only.

This continues the original request to add t-Flow and PFGM++ and measure them on
canonical Exp and Spiral: eight fresh trainings, two methods times two datasets
times two representations. It does not launch the separately proposed VP/RF
four-image pilot. The producing common runner also supports all old interfaces
under the same full-ambient architecture rule.

All30 supplied coefficient coordinates use one unrestricted spectral residual
core (width512, four blocks, scalar schedule-only input/output scaling).
All784 supplied pixels use the common full28x28 U-Net (width32). No covariance
fitting, projection, analytic normal-coordinate completion or true-LID training
target is permitted. Native corruption, learned outputs and differentiation
retain the entire ambient space. Full-dimensional RealNVP remains the common
invertible NF comparator; image NF keeps the accepted original architecture.

Retain the already declared common optimization:128000 updates, batch256,
AdamW2e-4/weight_decay1e-6, clipping1, antithetic noise, last8000-step cosine/EMA,
same native holdout-loss checkpoint policy. Scalar center/RMS normalization uses
only the99000 fit observations. Source-train holdout1000 and test1000 are the
same authenticated canonical partitions as other methods. No seed sweep or
task-dependent architecture/tail/optimizer setting.

t-Flow uses radial Student-t nu5, uniform native time and native noise MSE;
PFGM++ uses augmented dimension128, native EDM weighted denoising and truncated
native lognormal sigma sampling. Their shared radial denominator is retained
in the full ambient noise; no fitted-span conditional-kernel substitution.

Primary LID is the trace of the full posterior-mean Jacobian. Exact full30
coordinate derivatives for vectors;64 full-ambient Rademacher probes for images,
with independent-probe precision checks before a numerical quality claim.
Do not use active-span trace shortcuts. Gaussian full/score conversion remains
invalid for these native non-Gaussian kernels.

One scale is selected by response MAE on1000 source-train holdout queries using
the common29 RMS scales1/256 through64, frozen before test. Pointwise Kneedle
uses the identical grid. Save all queries/scales, boundaries and fallbacks.
The earlier test observations from the excluded projected models are known;
they are not used to tune this unrestricted recipe or any individual model.

First verify full-coordinate training, native identities, checkpoint replay,
matrix coverage, trace computation and actual GPU execution. Only then freeze
the producing source and launch the eight full-budget runs. A practical
selected-MAE check remains <.5, but numerical precision and independent
denoising/model-versus-continuous-law diagnostics must be reported too. Failure
of the practical score or small-noise fitting is not to be hidden or corrected
by projection. Any optimization repair must apply a general rule and retain
the failed models and immutable initial results.
