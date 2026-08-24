# Independent affine flow-matching campaign

The FM debug campaign is a controlled 3 × 2 factorial. It varies the affine
path and the learned field while keeping the data, network capacity,
conditioning coordinate, noise-ratio sampling, loss convention, prediction
grid, selector, derivative backend, and random seed fixed.

| Hydra `pilot_model` group | Schedule | Network output |
| --- | --- | --- |
| `direct_rectified_flow` | rectified linear | native velocity |
| `posterior_rectified_flow` | rectified linear | posterior mean |
| `direct_log_noise_affine_flow` | log noise | native velocity |
| `posterior_log_noise_affine_flow` | log noise | posterior mean |
| `direct_vp_trigonometric_flow` | VP trigonometric | native velocity |
| `posterior_vp_trigonometric_flow` | VP trigonometric | posterior mean |

Every group resolves to `family: independent_affine_flow`. Its immutable
`flow_variant_id` equals the Hydra group ID and is checked against the schedule
and parameterization before training and against the saved checkpoint contract
after training.

These six runs are a schedule × parameterization numerical factorial, not six
population-distinct LID methods. The endpoint identity applies to the broader,
infinite class of independent scalar-affine paths; the three schedules here
probe conditioning and approximation behavior under controlled choices. This
campaign deliberately excludes OT-CFM, reflow, dependent couplings, and
nonlinear interpolants because the current theorem assumes an independent
Gaussian source and scalar-affine interpolation. Adding those methods requires
a new derivation before an experiment is valid.

The older `rectified_flow` pilot remains a legacy failure/control run. It used
uniform native-time sampling, native-time conditioning, and the former loss
contract, so it is not one of the six factorial cells and must not be mixed into
the controlled comparison as if only the network output had changed.

## Common scientific coordinate

The public model scale is always the physical noise ratio
`lambda = beta / alpha`, irrespective of the schedule's native time. Training
uses `log(lambda)` conditioning and log-uniform sampling on `[0.01, 1]`. Scale
selection uses held-out source-train targets and the common lambda grid:

```text
0.01, 0.0178, 0.0316, 0.0562, 0.1, 0.1778, 0.3162, 0.5623, 1.0
```

The primary selector minimizes train-selection MAE for the `full` readout,
breaking a numerical tie toward smaller lambda. The selected index is frozen
before validation or test features and targets are resolved. Native schedule
coordinates (`t` for rectified/VP, `u` for log noise) are reported only as
deterministically recomputed diagnostics.

## Frozen readouts and debug boundary

At the one selected lambda, the pilot stores pointwise predictions and metrics
for `response`, `full`, and `fm_to_score` on train-selection, validation, and
test. It never stores validation/test scale curves. Root summaries expose
`macro_frozen_readouts.<readout>.<split>` so the eventual comparison table can
be built without trusting an external logger.

Detailed field diagnostics run only on the train-selection partition. Their
entire configuration is present under `pilot_model.diagnostics` in Hydra:
Hutchinson and exact-trace settings, deterministic subset seeds, empirical
Gaussian oracle reference size/chunking, and batch size. Diagnostic artifacts
are part of the sealed pilot output inventory.

## Comet names

Each job gets one descriptive Comet experiment. Names include the schedule,
parameterization, all-readout debug protocol, train-MAE lambda selection, and
seed, for example:

```text
lid-generalization-e8-suite-fm-vp-trigonometric-posterior-mean-all-readouts-debug-train-mae-lambda-selection-seed-137
```

The scheduler's fair-use job description is a separate cluster-level contract
and is not reused as a Comet experiment name.
