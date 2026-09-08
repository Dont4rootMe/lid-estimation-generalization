# Полностью валидируемый протокол

## Уровни доказательств

1. **Formula/unit:** численно проверяются коэффициенты, знаки, mean-path
   evaluation и factor two для current velocity.
2. **Analytic/oracle:** один и тот же empirical Gaussian endpoint channel
   вычисляет posterior moments; эквивалентные diffusion/FM/SB/NF readouts
   обязаны совпасть. На half-space response стремится к `d - 2/pi`, correction
   к `2/pi`, full к `d`.
3. **Dataset contract:** exact archive проверяется по SHA-256, структуре,
   split sizes, arrays и ground-truth policy.
4. **Learned bundle smoke:** внешний model adapter экспортирует все обязательные
   поля; runner проверяет bundle и provenance до создания output, затем требует
   объявленную долю finite predictions.
5. **Per-model matrix:** один Hydra job покрывает все запрошенные
   dataset × representation × readout cells для одной model/seed пары; raw
   predictions присутствуют, aggregate report пересчитывается из них.

Oracle и learned results всегда выводятся в разные таблицы.

## Реализованная benchmark matrix

| Аспект статьи | Dataset group | Автоматически вычисляемый analysis |
|---|---|---|
| 3.1–3.5 geometry | Gaussian, Spheres, Spaghetti, Uniform, Moon, Funnel, Spiral | pointwise predictions + global known-LID metrics |
| 3.6 representation | coefficients vs IDR image | paired estimate difference after row-identity checks |
| 3.7 sample size | FMNIST step 1…13 | independent estimate/dispersion curves vs available and used train size |
| 3.8 transformations | downscaled/base, ADI, ME, ASE | paired delta MAE (`+k` for ADI, `0` otherwise) after label/order checks |
| 3.9 real-like known LID | Arrows | global metrics against the published `6 × arrows` construction label |

Known-LID aggregate metrics are mean/std, MAE, RMSE, signed bias, median absolute
error and finite fraction. Unknown-LID datasets report paired delta error,
sample-size stability and representation discrepancy. Every result retains
per-point estimates, selected-row hashes and targets so a declared downstream
analysis can be reproduced without rerunning models.

Некоторые графики из benchmark paper требуют row-level covariates, которых нет
в опубликованном archive contract: component/radius identity, расстояние до
границы, положение на Moon/Spiral/Funnel и overlap/quantization state для
Arrows. Поэтому stratified error-vs-radius/edge/position plots **не** считаются
готовым output этого runner. Их можно добавить отдельным Hydra analysis group
только вместе с версионированным covariate artifact, его SHA, правилом
выравнивания строк и regression tests; до этого подтверждаемыми остаются
перечисленные выше метрики.

## Train-only scale/time selection without validation leakage

- The canonical source train split is deterministically partitioned into
  disjoint optimizer-fit and train-selection subsets.
- The train-selection subset is never used in optimizer batches; it may monitor
  target-free training loss, then its LID targets select the minimum-MAE
  scale/time after training.
- Ties resolve by the family-specific Hydra policy. The chosen candidate index
  is frozen before validation or test features/targets are accessed.
- The full train-selection curve is stored. Validation and test each execute
  exactly one inference at the frozen train-selected scale/time; no
  retrospective validation/test curve is part of the primary run.
- Split indices, targets, the train-selection curve, frozen predictions,
  diagnostics, and their hashes are sealed in the output manifest. The
  validator recomputes the partition, winning index, selected train column, and
  all reported metrics, and rejects validation/test curve artifacts.
- Validation/test targets and their MAE are never consulted for selection.

## Randomness

Hydra фиксирует dataset seed, model seed и trace-probe seed. Train sampling,
initialization и minibatch-order seeds принадлежат внешнему versioned training
config и связываются с результатом через его SHA-256. Для confirmatory study
рекомендуются минимум три model seeds, но текущий template использует seed 0 и
не выдаёт это пожелание за автоматический coverage gate. Deterministic exporters
должны дать одинаковый input inventory и побитово совпадающие `.npz` bundles
при повторном запуске в том же окружении.

## Acceptance gates

- 100% requested matrix coverage, no silently skipped cells;
- 100% finite target inputs; finite prediction fraction reported and at least
  the declared threshold;
- scale/time chosen only from held-out source-train targets, with zero optimizer
  overlap and no validation/test target access before the index is frozen;
- output checksum verification succeeds;
- model checkpoint/config SHA present for learned cells;
- canonical exact archive and regenerated fallback never aggregated together;
- representation transformations and any normalization are part of dataset
  identity.

Эти gates относятся к одной model/seed Hydra matrix. Покрытие полного
семейства моделей и seeds нужно проверять отдельным study-level manifest до
публикации общей таблицы; текущий runner не объявляет несколько независимых
Hydra jobs одной завершённой study. Аналогично, agreement exact trace и
Hutchinson на small-D fixture является обязательным тестом внешнего exporter,
если он заявляет stochastic trace, но пока не проверяется этим репозиторием.

## Compute budget

Maintenance continuation, 2026-09-08: task 9201 stopped in an auxiliary FM
ratio, not a model-quality gate. Diagnostic schema v3 explicitly represents
undefined endpoint/posterior trace ratios. Only the two ratio arrays may
contain NaN, and the validator independently recomputes their exact masks:
zero denominator, including cancellation within 32 float64 eps times the sum
of the two endpoint terms. JSON reports the undefined count and statistics
over defined values only. Raw fields, LID predictions and their metrics remain
finite-only. The v2 validator remains available for unchanged sealed evidence.

The task-specific maintenance importer permits exactly 147 validated cells
from commit 03c12b04561575fbef5c89c6876138bf7d7c2e44 / campaign
92fe3436909e3bf28427c380fdaa463e7f652c9d884f8ffe28cdc02cd11ff2b7.
Inputs, model configurations, training budget, selectors and readouts are
unchanged. Cells are copied byte-for-byte into a new source-bound root;
original run IDs/source hashes remain intact and the 282 newly executed cells
use the new source hash. The final manifest binds the immutable import lineage
under state/ and the original canary report. It does not claim that the old
canary measured new training. Fresh GPU-worker preflight and a replay of the
previously failing checkpoint diagnostics precede the remaining DAG. The replay
is engineering evidence, not an additional benchmark result. Incomplete cells
are not imported as results or silently counted as finished.

Author decision, 2026-09-07: canary v5 separates benchmark outcomes from
execution integrity at every dimension. Native-loss improvement/plateau,
reconstruction relative to trivial predictors, pointwise MAE/boundary selection,
NF versus Gaussian bin wins, and exact/H16/H64 accuracy are non-blocking
diagnostics. Historical thresholds remain recorded for interpretation, not
acceptance. GPU utilization, memory headroom and the 96-hour runtime projection
also cannot veto the full matrix. A canary PASS means complete, structurally
valid evidence, not good models, convergence, or baseline superiority.
Non-finite outputs, incorrect source/input identities, split leakage, missing
coverage and corrupt artifacts still fail closed. Analytic/unit regression
tests remain mandatory. No-knee remains an explicit selection-failed outcome;
it is not replaced with fabricated metrics. Training budget, selectors,
inline evaluation and checkpoint retention are unchanged.

The production v2 continuation approved on 2026-09-07 uses 128,000 optimizer
steps per cell at batch size 256 (32,768,000 presented examples), increased
from the unsuccessful 32,000-step canary attempts. The minimum native-loss
checkpoint on the disjoint train-selection subset is evaluated. The v4 canary
reports loss-tail improvement and whether a plateau was observed, but continued
improvement at the budget is not a rejection criterion. The finite budget does
not establish convergence; any `still_improving_at_budget` outcome remains
explicit. No test metric chooses the training budget or checkpoint. The new
source/config/campaign identities prevent mixing the two budgets.

The benchmark paper reports roughly 4–22 GPU-hours for FLIPD/LIDL on many
single datasets, up to several days for diffusion NB, and 10 days on 8 GPUs for
Arrows NB. A complete new four-family, multi-seed matrix is therefore a cluster
job, not a laptop smoke test. Infrastructure completeness and completed GPU
results are separate milestones.
