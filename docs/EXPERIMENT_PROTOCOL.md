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

The benchmark paper reports roughly 4–22 GPU-hours for FLIPD/LIDL on many
single datasets, up to several days for diffusion NB, and 10 days on 8 GPUs for
Arrows NB. A complete new four-family, multi-seed matrix is therefore a cluster
job, not a laptop smoke test. Infrastructure completeness and completed GPU
results are separate milestones.
