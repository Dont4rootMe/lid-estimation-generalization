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
selection uses a held-out source-train subset and the common lambda grid:

```text
0.01, 0.0178, 0.0316, 0.0562, 0.1, 0.1778, 0.3162, 0.5623, 1.0
```

For known-LID, the primary selector minimizes train-selection MAE for the
`full` readout; E1/E5 use their target-free reference-stability criteria. A
numerical tie breaks toward smaller lambda. The selected index is frozen before
validation or test features and targets are resolved. Native schedule
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

## Выполненное покрытие и результат

Все шесть factorial variants обучены на 39 cells с seed 0: всего 234
физических FM trainings внутри глобальной кампании из 390. Primary readout для
заранее объявленного сравнения — `full`; `response` и `fm_to_score` являются
secondary frozen readouts того же checkpoint. Legacy `rectified_flow` имеет
отдельные 39 trainings и не входит в факториал.

На primary `full` posterior variants оказались сильнее direct на known-LID и
E5, а direct variants — на E1 sample-size stability. Лучший primary balance
known-LID/E5 даёт posterior log-noise; лучший E1 — direct VP-trigonometric.
Универсального победителя нет. Агрегированные cell-level error summaries для
`full` и `fm_to_score` практически совпадают (maximum relative difference
`4.18e-05`), но compact CSV не доказывает pointwise equality; readouts не
считаются независимыми методами.

Exploratory выбор среди secondary readouts на validation выделил
`posterior_rectified_flow/response`; он конкурентен diffusion/SB на test, но
должен маркироваться как post-hoc validation-selected secondary result, а не
как замена predeclared primary. Числа, oracle caveat и cross-family сравнение
приведены в [`EXPERIMENT_RESULTS.md`](EXPERIMENT_RESULTS.md).

## Comet names

Each job gets one descriptive Comet experiment. Names include the schedule,
parameterization, all-readout debug protocol, train-MAE lambda selection, and
seed, for example:

```text
lid-generalization-e8-suite-fm-vp-trigonometric-posterior-mean-all-readouts-debug-train-mae-lambda-selection-seed-137
```

The scheduler's fair-use job description is a separate cluster-level contract
and is not reused as a Comet experiment name.
