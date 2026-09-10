# Выполненные learned-эксперименты и результаты

Документ является единой отслеживаемой точкой входа для фактически выполненной
learned-study. Он фиксирует состав запусков, правила чтения единой таблицы,
дизайн flow matching и normalizing flow экспериментов, подтверждённые результаты
и границы их интерпретации. Теоретические утверждения остаются в
[`agent/THEORY_CONTRACT.md`](agent/THEORY_CONTRACT.md), а команды запуска — в
[`agent/RUNBOOK.md`](agent/RUNBOOK.md).

## Актуальное дополнение: benchmark v2, 2026-09-08

CloudBot9344 завершилась с exit0. Полностью проверены429 cells (11×39),
147 immutable reused +282 новых; full remote validator со свежими inputs
PASS. Единый compact bundle:
`artifacts/results/lid_benchmark_v2/64ae8a405a42c248d14c/`.
В `unified_results.csv`2368 строк:1368 known-LID,650 E1,350 E5;
2080 canonical и288 generated E3/E4. Локальные SHA и429 cell-ID joins
проверены. README bundle описывает каждый файл, selection protocols,
11-model test сводку и ограничения. Bulk pointwise evidence/checkpoints не скачивались.
Source commit `6d041365b9bc27de5054293be6c1318766ddf911`,
CSV SHA `64d5b2ae446d48505a86337c2c8b8e173f8722447ea505ff179b4a0d8e597903`.

Это самостоятельная v2-study, а не дописанные строки к v1: два раздельных
known-LID selection protocols, явные VP/VE,128000-step budget и новые
модельные контракты. Старые NF OLS5 результаты не выдаются за v2-тренировки.
Полнота подтверждена, хорошее качество всех веток — нет. Например, canonical
primary test mean MAE VP:3.372 supervised против128.971 pointwise Kneedle.
Сравнение mean MAE между протоколами описательное, не proof of convergence.
Полные pins, validation receipt и пути — в agent status/runbook и launch passport.

Проверка2026-09-10: known-LID Kneedle использовал22 масштаба
lambda0.210656–3.229097, supervised selection имел support[1/256,64].
103/165 canonical supervised choices лежат вне узкого диапазона. Поэтому
разницу качества нельзя приписывать только правилу выбора knee. Расширенный
validation/test eval пока не выполнен: широкие кривые не сохранены, финальные
веса удалены; audit receipt — `artifacts/launches/kneedle_range_audit_20260910.json`.

## Историческая study v1 с NF extension

Все числа и выводы ниже относятся к срезу **2026-08-30**, не к v2.

## Источник и проверяемая идентичность

Локальный compact bundle находится в исключённом из Git каталоге:

```text
artifacts/results/researh_l1_pretrain_ultra/a80ddd279bb32d666649/
```

Основная таблица — `unified_results.csv`. Числа ниже были сверены с её manifest,
SHA и локальным анализом. В чистом checkout bundle отсутствует, поэтому этот
документ не заменяет исходные CSV/JSON и не превращает локальную копию в
переносимое training evidence.

| Объект | Значение |
|---|---|
| Baseline campaign identity | `a80ddd279bb32d66664994fcd1e0e9c781030064d22f9af8f45d6b96e7f584bd` |
| Consolidated bundle identity | `2ebedf14e7f06d8fb84b8dba030ce5817a41d2fc49ecec3ba1cb5f89a4110441` |
| NF source-ablation identity | `b37fc13d34fec456905cbdfdbb4245df117c21f58985c550f91c87b9d4c9813f` |
| NF follow-up identity | `40a0b4e34e6f0dc7c7ccc69a4181cb59e9e9bdf40718fa21a50f2b21ff483d7c` |
| Baseline 1872-row CSV SHA / byte prefix | `df2140318020663875a0bdd6384d18954473dede44799cf2bf4278aabfdc36ef` |
| Final 2106-row CSV SHA | `827a56ab5cd67ced10a07b3b28d06c23471db44201b87caf7c34d8f18aa7477c` |
| Aggregate SHA | `78d9ba71259aa4d300e2f93f8065c7653612cb015683a96ba8ac297a40129750` |
| Resolved config SHA | `17495350461409894a96d8994e1d163a993ce8f98ea34b3a1971662aaba9b34c` |
| Input inventory file SHA | `d358f112e465731ea52c7a9c53e723d931bfb827489448fb21cf059497965281` |
| Semantic input inventory SHA | `383ee23fe18b8c2019a15b33317c4d751678d28d54dc03682a603d3c09bf4ac7` |
| Source tree SHA | `da4b07e708ae809750faca05b042fd23ff2c043e03eb74a5e0afc5c195705033` |
| Exact archive SHA | `ce0d153a1a78a3a752b29ec2e60167134b6b20c3249db2fe92f9fc1b8b8a9181` |
| Upstream data revision | `2dcb8e41015f53413ff1ddd049bb006c81a5df52` |
| Validation status | `remote_full_and_local_compact_validators_passed` |

Validator подтверждает структуру, provenance, hashes, полноту и пересчёт
сохранённых метрик. Он не превращает descriptive leaderboard в статистический
тест и не подтверждает теоретическое утверждение автоматически.

## Что именно было запущено

Нельзя употреблять количества обучений, вариантов и строк как синонимы.

| Уровень учёта | Количество | Что считается |
|---|---:|---|
| Исходные обучения | 390 | 10 model configurations × 39 cells |
| Дополнительные NF-обучения | 78 | unchanged C0 × seeds 2/3 × 39 cells |
| Всего обучений в consolidated provenance | 468 | 390 + 78; diagnostic ablation сюда не входит |
| Логические result variants | 13 | 10 исходных + 3 NF readout/seed variants |
| Baseline CSV rows | 1872 | все validation/test и все frozen readouts |
| NF extension rows | 234 | 3 variants × 39 cells × 2 splits |
| Итоговые CSV rows | 2106 | 1872 + 234 |
| Primary-readout rows | 1014 | 13 variants × 39 cells × 2 splits |
| Alternate-readout rows | 1092 | вторичные readouts тех же checkpoint |

Десять исходных записей — это **model variants/configurations**, а не десять
различных математических семейств. Шесть из них образуют управляемый факториал
одного семейства independent affine flow matching.

Production-запуск использовал один `h100_8gpu_cell_dag`: восемь независимых
однопроцессных H100 workers, одна cell на GPU, без DDP. Для baseline
использовались seed 0, batch size 4096, max-epoch budget 200 с early stopping и
inline evaluation. После оценки cell checkpoint удалялся; координатор
единолично записывал ledgers, aggregates и manifest.

## Матрица из 39 cells

| Suite | Cells | Состав |
|---|---:|---|
| E1 | 15 | FMNIST sample-size steps 1–13; spiral PCA в `dataset` и `coefficients` |
| E2 | 3 | Arrows; uniform PCA в `dataset` и `coefficients` |
| E3 | 2 | generated Gaussian PCA в `dataset` и `coefficients` |
| E4 | 2 | generated sphere PCA в `dataset` и `coefficients` |
| E5 | 7 | downscaled reference, padded +0/+4/+8, stretched 0.25/4, upscaled |
| E6 | 2 | exponential PCA в `dataset` и `coefficients` |
| E7 | 2 | crescent moon в `dataset` и `coefficients` |
| E8 | 6 | Gaussian4, spaghetti и sphere4, каждый в двух representations |

Разбиение по анализам:

- **35 canonical exact-archive cells**: 15 known-LID, 13 E1 sample-size и
  7 E5 paired-delta;
- **4 generated E3/E4 cells**: только отдельное known-LID extension;
- E1 step 1 является reference anchor, а E5 downscaled — self-reference;
  как самостоятельные scoring cells они не считаются, поэтому остаётся 33
  nontrivial cells; E1 step 1 сохраняется только как нулевая граница AUC;
- всего используются 30 `dataset` и 9 `coefficients` representations.

Generated E3/E4 нельзя смешивать с canonical headline. Их происхождение и
валидация описаны в [`GENERATED_E3_E4.md`](GENERATED_E3_E4.md).

## Модели, checkpoint и readouts

### Vanilla и control

| Model variant | Роль | Primary readout | Secondary readout | CSV rows |
|---|---|---|---|---:|
| `diffusion` | Gaussian diffusion | `full` | — | 78 |
| `rectified_flow` | legacy RF failure/control | `full` | `response` | 156 |
| `scale_conditioned_nf` | conditional RealNVP baseline | `fixed_likelihood` | — | 78 |
| `schrodinger_bridge` | Brownian bridge | `full` | `response` | 156 |

Legacy `rectified_flow` обучался по старому native-time контракту. Он полезен
как vanilla/control, но не является седьмой cell управляемого FM-факториала.

Для diffusion, RF, SB и шести FM configurations общий learned-field профиль
использовал hidden size 1024, depth 6, learning rate `2e-4`, weight decay
`1e-6`, max-epoch budget 200 и Hutchinson divergence с 16 probes. Diffusion обучал score-поле по
noise scale; SB — forward drift обобщённого Brownian bridge по time-to-go;
RF/FM — velocity либо posterior-mean primitive по своему schedule. Baseline NF
использовал conditional RealNVP с 8 coupling layers, hidden 512, conditioner
depth 2, epsilon 0.01…1 и exact derivative backend.

### Управляемый flow-matching факториал

| Schedule | Direct velocity | Posterior mean |
|---|---|---|
| Rectified linear | `direct_rectified_flow` | `posterior_rectified_flow` |
| Log noise | `direct_log_noise_affine_flow` | `posterior_log_noise_affine_flow` |
| VP trigonometric | `direct_vp_trigonometric_flow` | `posterior_vp_trigonometric_flow` |

Каждый из шести дизайнов обучен на всех 39 cells: **234 FM trainings** внутри
исходных 390. Один checkpoint даёт три frozen readouts — primary `full` и
secondary `response`, `fm_to_score` — поэтому 234 строки одного FM variant не
означают три независимых обучения. Полный контракт находится в
[`FM_CAMPAIGN.md`](FM_CAMPAIGN.md).

### NF extension

| Logical variant | Physical checkpoint | Readout | CSV rows |
|---|---|---|---:|
| C0 seed 2 / autograd | seed 2 | `autograd` | 78 |
| C0 seed 2 / OLS5 | тот же seed-2 checkpoint | `ols5` | 78 |
| C0 seed 3 / OLS5 | seed 3 | `ols5` | 78 |

Таким образом, три логических NF-варианта получены из 78, а не из 117
обучений. Дизайн ablation и правила заморозки OLS5 описаны в
[`NF_CAMPAIGN.md`](NF_CAMPAIGN.md).

## Как читать единую таблицу

### Selection protocol

Канонический scale/time/lambda внутри каждой cell выбирается на отделённом от
optimizer-fit подмножестве source-train. Для known-LID используется supervised
MAE; для E1/E5 — target-free reference-stability criterion. Выбранный индекс
замораживается до чтения validation/test features и targets. Validation и test
вычисляются только в одной замороженной координате; test используется как
confirmatory split.

Это не то же самое, что выбор model/readout между вариантами. Если кандидат
сравнивался и выбирался по validation после кампании, он явно маркируется как
`validation-selected`, `selection-conditioned` и/или `post-hoc`.

### Три разных типа задач

Пустые поля — часть schema-by-task, а не признак умершего eval.

| `analysis` | Вопрос | Заполненные метрики | Ожидаемо пусто |
|---|---|---|---|
| `known_lid` | Ошибка относительно известной LID | `mae`, `rmse`, `bias`, `median_absolute_error` | `reference_mean`, `reference_median` |
| `e1_sample_size_stability` | Дрейф оценки при уменьшении train size | `reference_mean`, `reference_median`, delta-поля, `mean_delta_error` | абсолютные `mae`, `rmse` |
| `e5_paired_delta` | Ошибка парной ожидаемой дельты LID | `mae`, `rmse`, `bias`, `expected_lid_delta` | reference mean/median |

Один столбец нельзя усреднять сразу по трём analysis types. В анализе
используются:

- known-LID: `relative_mae = mae / abs(mean - bias)` и
  `normalized_mse = (rmse / abs(mean - bias))^2`; на 15 canonical known-LID
  cells, для cells `i` внутри suites `s`,
  `Known score = expm1(mean_s(mean_i(log1p(relative_mae_i))))`, то есть suites
  имеют равный вес независимо от числа cells;
- E1: `x = log2(n_ref / n_train)`,
  `y = log1p(abs(mean_delta_error / reference_mean))`,
  `E1 score = expm1(trapezoid(y, x) / (x_max - x_min))`; step-1 anchor
  `(0, 0)` входит как boundary point интеграла, но не как отдельная scoring cell;
- E5: после удаления self-reference
  `E5 balanced = expm1(0.5 * (mean_invariance(log1p(mae)) +`
  `mean_equivariance(log1p(mae))))`. Поэтому этот score не равен обычному
  arithmetic macro MAE.

Базовый фильтр для reported logical variants:

```text
split == "test"
is_primary_readout == True
join input_inventory on (suite_id, dataset, representation)
matched_inventory_record.cell.source_kind == "exact_archive"
```

После этого E1/E5 anchors не считаются самостоятельными scoring cells; E1
step 1 сохраняется только как нулевая граница AUC, E5 self-reference удаляется.
В самом CSV нет столбца `canonical`: provenance join обязателен. Для текущего
замороженного inventory эквивалентный shortcut — исключить `suite_id` E3/E4,
но его нельзя переносить на будущие inventory без повторной проверки.
`is_primary_readout=True` означает primary **внутри данного logical variant**,
но не гарантирует predeclared primary всей study. У baseline/FM это заранее
объявленный primary; OLS5 в NF extension был заморожен после validation-only
follow-up и остаётся явно validation-selected, хотя его test rows confirmatory.
FM alternate readouts, generated E3/E4 и oracle варианты показываются отдельно.

## Результаты vanilla-моделей

### Known-LID, 15 canonical test cells

| Model | Raw mean MAE | Median MAE | Mean без Arrows | Geo normalized MSE |
|---|---:|---:|---:|---:|
| Diffusion | 50.511 | 1.170 | 1.803 | 0.389 |
| Schrödinger bridge | 52.779 | 1.078 | 1.776 | 0.320 |
| Rectified flow | 96.722 | 4.879 | 15.004 | 5.755 |
| Scale-conditioned NF | 22.008 | 23.515 | 16.409 | 12.805 |

Raw mean даёт ложную картину из-за `e2_arrows`: MAE приблизительно равен 100
для NF, 732 для diffusion, 767 для SB и 1241 для RF. По robust-нормированным
метрикам верхнюю vanilla-группу образуют diffusion и SB, а не NF.

### Representation, sample size и transformations

- На canonical cells `coefficients` лучше `dataset` во всех **28/28** парных
  vanilla-сравнениях; на generated E3/E4 — ещё в 8/8. Это самый стабильный
  cross-family эффект текущей study.
- E1 AUC score: SB 0.488, diffusion 0.508, RF 1.044, NF 1.370. При самом
  малом `n=12` absolute relative drift
  `abs(mean_delta_error / reference_mean)` меняет порядок: RF 1.701 лучше
  NF 2.016, SB 2.939 и diffusion 3.087. Поэтому одна усреднённая цифра скрывает
  пересечение кривых.
- E5 paired MAE без self-reference: SB 20.7, diffusion 25.5, RF 90.9,
  NF 383.9. SB выигрывает пять из шести nontrivial transformations;
  diffusion — `stretched_power4`.

## Flow matching: что варьировалось и что получилось

Все шесть designs используют одну физическую координату
`lambda = beta / alpha`, conditioning по `log(lambda)`, log-uniform train
sampling на `[0.01, 1]` и общую сетку из девяти lambda. Менялись только:

1. affine schedule: rectified, log-noise или VP-trigonometric;
2. parameterization: direct velocity или posterior mean.

Primary `full` результаты:

| FM design | Known score | E1 AUC score | E5 balanced |
|---|---:|---:|---:|
| Posterior / log-noise | 1.384 | 0.508 | 24.743 |
| Posterior / VP-trig | 1.399 | 0.489 | 29.697 |
| Posterior / rectified | 2.077 | 0.485 | 34.497 |
| Direct / VP-trig | 2.682 | 0.131 | 204.152 |
| Direct / log-noise | 3.529 | 0.196 | 210.212 |
| Direct / rectified | 4.971 | 0.223 | 253.290 |

Вывод не сводится к одному победителю:

- posterior designs заметно сильнее на known-LID и E5;
- direct designs заметно сильнее на E1 sample-size stability;
- posterior/log-noise даёт лучший primary balance known-LID/E5;
- direct/VP даёт лучший E1, но плохо переносится на E5;
- агрегированные cell-level error summaries для `full` и `fm_to_score`
  практически совпадают: максимальное относительное различие `4.18e-05`.
  Compact bundle не позволяет утверждать pointwise equality; в любом случае
  это две формы одного checkpoint, а не два независимых метода.

В отдельном exploratory сравнении на validation перебирались все 18 frozen FM
combinations: 6 designs × (`full`, `response`, `fm_to_score`). Выбранный
`posterior_rectified_flow / response` является secondary readout. На test он
дал known score 0.863, geo normalized MSE 0.300, E1 0.703 и E5 19.8. Это
конкурент diffusion/SB, но результат является **post-hoc validation-selected
secondary readout**, а не заменой заранее объявленного primary `full` и не
доказательством абсолютного превосходства.

Per-cell test oracle также является только diagnostic ceiling. Для primary FM
он выигрывает у лучшей vanilla-модели 9/15 known-LID и 11/12 E1 cells, но 0/6
E5 cells. Такой oracle выбирает разный design после просмотра test и не является
одной разворачиваемой моделью.

## Normalizing flows: ablation и итог

Сначала был выполнен validation-only staged ablation по optimizer,
architecture, conditioning и readout. Он не перезаписывал baseline. C1/OLS5
был отклонён исходным training-quality gate, поэтому улучшение архитектуры или
batch режима не объявляется. C0/OLS5 прошёл отдельный matched readout gate
против C0/autograd: validation geometric ratio 0.231, 14/15 canonical wins и
одна регрессия более 25%. OLS9 остался diagnostic-only.

После заморозки unchanged C0 + OLS5 были обучены seeds 2 и 3 на всех 39 cells.
Основная сводка:

| NF variant | Known score | E1 AUC score | E5 balanced |
|---|---:|---:|---:|
| C0 seed 3 / OLS5 | 1.027 | 0.428 | 251.132 |
| C0 seed 2 / OLS5 | 1.224 | 0.710 | 244.762 |
| Baseline scale-conditioned NF | 3.825 | 1.370 | 294.025 |
| C0 seed 2 / autograd | 4.159 | 1.224 | 216.289 |

Чистое matched test-сравнение seed2 показывает:

- OLS5 выигрывает autograd на **15/15** canonical known-LID cells;
- geometric relative-MAE ratio равен примерно **0.209**, то есть улучшение
  порядка 4.8 раза по этой агрегации;
- raw known-LID macro MAE уменьшается примерно с 27.8 до 14.5;
- raw E5 macro, наоборот, растёт примерно с 227 до 353, то есть на 55%;
- train-selected scale index меняется в 25/39 cells.

Поэтому подтверждённый эффект называется **readout + scale-selection protocol**,
а не заменой производной при фиксированном scale. Seed3/OLS5 — лучший NF по
balanced descriptive ranking и показывает высокую cell-wise согласованность с
seed2, но всё ещё не становится универсальным победителем. Его suite-balanced
known score лежит между SB и diffusion, geometric relative-MAE остаётся хуже
обоих, а E5 остаётся слабым.

## Cross-family descriptive view

Ниже `overall rank` — средний percentile-rank, сначала вычисленный отдельно по
known-LID, E1 и E5, затем усреднённый с равным весом типов задач. Lower is
better. Это компактная навигация, а не единая физическая метрика и не
статистический тест.

| Reported logical variant | Overall rank | Known | E1 | E5 |
|---|---:|---:|---:|---:|
| Schrödinger bridge | 0.274 | 0.991 | 0.488 | 19.444 |
| Diffusion | 0.300 | 1.092 | 0.508 | 23.884 |
| FM posterior / log-noise | 0.324 | 1.384 | 0.508 | 24.743 |
| FM posterior / VP-trig | 0.338 | 1.399 | 0.489 | 29.697 |
| FM direct / VP-trig | 0.417 | 2.682 | 0.131 | 204.152 |
| FM posterior / rectified | 0.458 | 2.077 | 0.485 | 34.497 |
| FM direct / log-noise | 0.514 | 3.529 | 0.196 | 210.212 |
| NF C0 seed 3 / OLS5 | 0.527 | 1.027 | 0.428 | 251.132 |
| FM direct / rectified | 0.541 | 4.971 | 0.223 | 253.290 |
| NF C0 seed 2 / OLS5 | 0.583 | 1.224 | 0.710 | 244.762 |
| Rectified flow | 0.669 | 4.232 | 1.044 | 82.462 |
| NF C0 seed 2 / autograd | 0.730 | 4.159 | 1.224 | 216.289 |
| Baseline scale-conditioned NF | 0.825 | 3.825 | 1.370 | 294.025 |

Наиболее устойчивые выводы текущей study:

1. diffusion и Schrödinger bridge образуют сильнейшую vanilla-группу;
2. representation effect (`coefficients` лучше `dataset`) устойчивее различий
   между многими архитектурами;
3. FM имеет реальный trade-off: posterior лучше known-LID/E5, direct лучше E1;
4. OLS5 резко лечит known-LID readout NF, но не лечит E5 и потому не даёт
   универсального улучшения NF;
5. универсального победителя для всех трёх типов задач нет.

## Ограничения и правила цитирования

- Baseline и FM имеют один training seed 0. Два NF OLS5 seeds также не дают
  формального confidence interval.
- Разброс по benchmark cells не равен training-seed uncertainty.
- `dataset`/`coefficients` зависимы, E1 levels вложены, E5 transforms парные;
  обычные независимые confidence intervals к ним неприменимы без отдельной
  модели зависимости.
- Если validation использовалась для выбора model/readout, соответствующие
  validation metrics selection-conditioned; test rows не использовались для
  этого выбора. Обычные baseline validation rows после train-selection сами по
  себе не selection-conditioned.
- Primary FM `full` нельзя молча заменить удачным secondary `response`.
- Test-oracle envelopes и per-cell minima не являются deployable моделями.
- Compact bundle не содержит checkpoint, pointwise `.npy` и telemetry. Исходный
  sealed remote evidence позволяет проверить pointwise/cell artifacts,
  provenance и attested checkpoint hashes, но checkpoint bytes также были
  удалены по policy; reinference без нового обучения невозможен.
- Любое число в статье должно сопровождаться bundle identity, split, row
  filter, analysis type, variant/readout и указанием canonical/generated scope.

Локальный графический отчёт с 34 figures находится в
`artifacts/analysis/researh_l1_pretrain_ultra/`; он исключён из Git и является
производным от описанного здесь CSV, а не отдельным источником истины.
