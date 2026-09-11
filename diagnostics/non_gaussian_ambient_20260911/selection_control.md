# Supplemental zero-head control, 2026-09-11 15:49 UTC

During the unrestricted trainings, before their selected test results existed,
an arithmetic control exposed a limitation of the practical half-unit score
check. The common zero residual head gives b0(y,lambda)=y/(1+lambda^2), hence
R0=N/(1+lambda^2), independently of the data geometry and native Student tail.
For constant known LID, supervised scale selection can calibrate this response
without learning a manifold. This does not change the frozen training recipe,
primary selector or original protocol; it adds an explicitly labelled control.

Using the same29 candidates and constant1000-query holdout targets, the control
selects lambda4/5.656854 on30-dimensional Exp/Spiral, with MAE.235294/.090909.
On784 pixels it selects22.627417/32, with MAE.471735/.235122. All four pass<.5.
Common-grid Kneedle instead has MAE1.333/2.333/85.111/86.111, respectively.

Therefore passing the declared selected-MAE check alone is not a training
quality certificate. Report the control's separately selected scale, keep
learned-model selections immutable, and assess learned posterior quality using
the full-ambient exact data-law comparison and fresh-noise denoising. The
control is not a new trained method or a requested extra VP/RF pilot.
