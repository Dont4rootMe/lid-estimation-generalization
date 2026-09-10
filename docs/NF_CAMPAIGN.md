# Normalizing-flow quality campaign

Документ фиксирует проведённый staged ablation и последующий evidence-bound
NF readout follow-up. Он дополняет общую сводку
[`EXPERIMENT_RESULTS.md`](EXPERIMENT_RESULTS.md) и не меняет sealed baseline.

## Исходная проблема и граница эксперимента

Baseline `scale_conditioned_nf` — conditional RealNVP с 8 coupling layers,
hidden size 512, conditioner depth 2, batch 4096 и max-epoch budget 200 с early
stopping. Его primary
`fixed_likelihood` readout оказался нестабильным на части known-LID и особенно
слабым на E5. Ablation проверял четыре разные причины:

1. шаги оптимизации и batch size;
2. capacity coupling network;
3. smoothness scale conditioning;
4. способ извлечения производной likelihood по log-scale.

Это численное исследование конкретного conditional RealNVP, а не утверждение
обо всех normalizing flows. Регулярный fixed NF по-прежнему является
теоретическим negative control, если не объявлен калиброванный singular limit.

## Кандидаты

| ID | Изменение относительно C0 |
|---|---|
| C0 | batch 4096, max epochs 200 |
| C1 | batch 1024, max epochs 200 |
| C2 | batch 1024, max epochs 400, patience 40; matched update budget |
| C3 | batch 1024, max epochs 250, patience 25; hidden 768, 12 layers, depth 3 |
| C4 | batch 1024, max epochs 400, patience 40; embedding 256, Fourier 64, max frequency 32, epsilon 0.005…1.25 |
| C5 | batch 1024, max epochs 250, patience 25; capacity и conditioning overrides C3/C4 |
| P0 | batch 1024, max epochs 400, patience 40; independent fixed-epsilon RealNVP, global OLS9 diagnostic |

Для C0–C5 оценивались `autograd`, `symmetric_fd`, `ols3`, `ols5` и `ols9`.
OLS3/5/9 аппроксимируют локальный slope likelihood по трём, пяти или девяти
log-scale points; P0 использует отдельный `global_ols9` контракт.

## Staged protocol

### Stage 1: широкий validation-only screen

- 7 candidates × 8 sentinel cells = 56 cell tasks;
- seed 0;
- sentinel matrix включает high-dimensional data, coefficients,
  sample-size reference и paired-delta reference;
- test не читается;
- promotion ограничена максимум двумя candidate/readout pairs и использует
  normalized paired ratios, win rate, strata и runtime gates.

Stage 1 породил 248 validation readout records: пять readouts для C0–C5 и один
для P0 на каждой из восьми cells. Число 56 — scheduler cell tasks, не число
model fits: 48 conditional tasks обучают по одному fit, а восемь P0 tasks — по
девять fixed-epsilon components, всего 120 Stage-1 fits.

### Stage 2: полный known-LID validation gate

- C0 и promoted C1;
- seeds 0 и 1;
- все 19 known-LID cells, включая 15 canonical и 4 generated;
- 76 cell tasks и 380 validation readout records;
- winner gate учитывает canonical geometric error ratio, wins и число
  регрессий более 25%.

Training-кандидат C1/OLS5 против C0/OLS5 **не прошёл** gate:

| Gate statistic | C1/OLS5 vs C0/OLS5 |
|---|---:|
| Canonical geometric ratio | 0.9495 |
| Canonical wins | 11/15 |
| Regressions >25% | 4 |
| Decision | rejected |

Эта остановка является корректным результатом predeclared gate. Нельзя
переопределять C1 как успешное architecture/optimizer improvement post hoc.
Diagnostic stage1/stage2 tasks не входят в поле
`total_physical_training_runs=468` консолидированного result bundle и не
добавлены в `unified_results.csv`.

### Source Stage 3: не выполнен

Исходный ablation планировал Stage 3 как C0/seed2 и winner/seeds2,3 на всех 39
cells с validation/test, то есть 117 cell tasks. Он требовал promoted
non-control winner. Поскольку C1 не прошёл Stage-2 gate, source Stage 3 не был
запущен и не имеет публикуемых rows. Compact reviewed evidence намеренно
заканчивается полными Stage 1/2; последующий C0/OLS5 follow-up ниже является
отдельным протоколом, а не переименованным Stage 3.

## Почему был разрешён отдельный follow-up

В тех же sealed Stage-2 evidence C0/OLS5 сравнили с C0/autograd при неизменном
training candidate:

| Gate statistic | C0/OLS5 vs C0/autograd |
|---|---:|
| Canonical geometric ratio | 0.2313 |
| Canonical wins | 14/15 |
| Regressions >25% | 1 |
| Decision | passed |

Это не ослабление проваленного C1 gate. Был разрешён отдельный readout-only
follow-up с замороженными unchanged C0 и OLS5. OLS9 остался diagnostic-only.

## Confirmatory follow-up

- C0 обучен заново на всех 39 cells для seeds 2 и 3;
- всего 78 физических trainings;
- seed-2 checkpoint оценён одновременно через autograd и OLS5;
- seed-3 checkpoint оценён через OLS5;
- validation/test выполнены inline, checkpoint сразу удалён;
- test не использовался для выбора readout;
- исходные 1872 baseline rows сохранены точным byte prefix;
- добавлены ровно 234 schema-compatible rows.

Получившиеся logical variants:

```text
C0 seed 2 / autograd
C0 seed 2 / OLS5
C0 seed 3 / OLS5
```

## Подтверждённый результат

Matched seed2 OLS5 против seed2 autograd на test:

- 15/15 wins на canonical known-LID;
- geometric relative-MAE ratio около 0.209;
- raw canonical known-LID macro MAE примерно 27.8 → 14.5;
- raw E5 macro примерно 227 → 353, то есть ухудшение около 55%;
- выбранный train-selection scale index меняется в 25/39 cells.

Следовательно, OLS5 существенно улучшает **known-LID readout + selection**,
но не является универсальным улучшением NF. Он не исправляет E5, а различие
selected scale не позволяет приписать весь эффект одной формуле производной
при фиксированном scale.

Seed3/OLS5 даёт лучший balanced NF result и согласованный cell-wise порядок с
seed2, но двух seeds недостаточно для формального confidence interval. Даже
лучший NF остаётся ниже сильнейших diffusion/SB и posterior-FM по общему
descriptive ranking и особенно слаб на E5.

## Что можно и нельзя утверждать

Можно:

- утверждать, что локальный OLS5 readout резко улучшил canonical known-LID в
  matched C0 seed2 сравнении;
- показывать E1 улучшения и E5 регрессии одновременно;
- использовать seed3 как ограниченную confirmatory replication.

Нельзя:

- объявлять C1 успешным;
- включать OLS9 в consolidated result table как выбранный метод;
- называть эффект OLS5 универсальным улучшением normalizing flows;
- использовать test для повторного выбора candidate/readout;
- считать три logical variants тремя независимыми наборами checkpoint.

Исполняемый контракт находится в `experiments/nf_ablation.py` и
`experiments/nf_readout_followup.py`; команды — в
[`agent/RUNBOOK.md`](agent/RUNBOOK.md).
