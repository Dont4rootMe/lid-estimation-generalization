# Model/readout matrix

Локальная статья предлагает математические model interfaces, но не задаёт
нейросетевые архитектуры. Ниже перечислено то, что действительно входит в
основной экспериментальный scope.

Каждая строка ниже покрыта отдельным или совместным Hydra model group в
`configs/models/`. Oracle проверяет равенства формул; `learned_field_bundle`
принимает строку в empirical matrix только после привязки к реальным
checkpoint/training-config SHA и точным dataset rows.

| Family | Readout ID | Branch/scope | Learned primitive fields |
|---|---|---|---|
| Gaussian diffusion | `diffusion_flipd_full` | density/full; boundary-safe | score, score divergence, sigma |
| Affine FM | `fm_affine_response` | posterior response; flat interiors | velocity divergence, schedule |
| Affine FM | `fm_affine_full` | density/full; boundaries/cones | velocity, divergence, score, mean-path point |
| Rectified Flow | `fm_rectified_response` | affine specialization; flat interiors | velocity divergence at `t*x` |
| Rectified Flow | `fm_rectified_full` | Gaussian boundary-safe | velocity at `t*x`, divergence, original `x` |
| Brownian SB | `sb_forward_response` | terminal response; flat interiors | forward drift divergence |
| Brownian SB | `sb_forward_full` | terminal full; boundaries/cones | drift, divergence, gamma, time-to-go |
| SB/SF2M | `sb_current_full` | current-score full, factor two | current velocity, divergence, score |
| Scale-conditioned NF | `nf_scale_conditioned_fixed` | gauge-invariant fixed-point likelihood | scale velocity, divergence, score |
| Calibrated singular NF | `nf_calibrated_native` | calibrated interior projector trace | scale-velocity divergence |
| Calibrated singular CNF | `cnf_calibrated_native` | same under declared time schedule | velocity divergence, d-log-scale/dt |

Gaussian posterior covariance and conditional risk are exact diagnostics for
FM/SB, not additional trained model families. Source-reversed SB,
time-dependent scalar diffusivity and uniformly elliptic reference variants
are conditional extensions; they join the full learned matrix only when the
corresponding model exposes the required fields and assumptions.

## Negative controls

- A fixed regular normalizing flow must return ambient dimension `D` locally;
  it is **not** a native nontrivial LID estimator.
- A finite regular CNF cannot exactly collapse an absolutely continuous base
  to a lower-dimensional endpoint.
- `nf_material` is gauge/path dependent and is diagnostic unless a calibrated
  collapse normal form is declared and checked.

These cases must be labelled `negative_control`, not presented as failed
versions of the proposed valid NF interfaces.

## Реализованная completed-study matrix

Теоретическая interface matrix выше шире одной конкретной кампании. В
завершённой learned-study фактически обучены десять baseline configurations:

| Группа | Реализованные variants | Primary / secondary readouts |
|---|---|---|
| Vanilla/control | diffusion, legacy rectified flow, scale-conditioned NF, Schrödinger bridge | primary `full`/`fixed_likelihood`; RF/SB также `response` |
| Independent affine FM | 3 schedules × direct/posterior = 6 variants | primary `full`; secondary `response`, `fm_to_score` |

Каждый baseline variant покрывает 39 cells с validation и test: 390 физических
trainings. Supplemental NF follow-up добавляет 78 C0 trainings и три logical
result variants: seed2/autograd, seed2/OLS5 и seed3/OLS5. Итого consolidated
table содержит 13 logical variants, но это не 13 независимых математических
семейств и не 13 × 39 независимых checkpoint.

Точная таксономия, row accounting и результаты находятся в
[`EXPERIMENT_RESULTS.md`](EXPERIMENT_RESULTS.md); NF selection — в
[`NF_CAMPAIGN.md`](NF_CAMPAIGN.md).

## Explicitly outside the confirmatory matrix

VAE, arbitrary nonlinear interpolants, non-Brownian bridges, Stable Diffusion
3.5, SoftFlow, ID-NF, Spread Flows and related architectures are prior work or
research extensions in the paper. They can be added to an exploratory report,
but missing theory/model normal form prevents them from being silently counted
as “all proposed models”.
