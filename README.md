# LID estimation experiments

Воспроизводимый экспериментальный контур для статьи **Endpoint Channels Reveal
Local Dimension: Diffusions, Flow Matching, Bridges, and Calibrated Normalizing
Flows** и набора **Why We Need New Benchmarks for Local Intrinsic Dimension
Estimation** (ICLR 2026).

## Запуск через Hydra

Все конфигурации экспериментов — YAML и компонуются только Hydra. TOML остаётся
только там, где это формат Python tooling (`pyproject.toml` и `uv.lock`), а не
экспериментальная конфигурация.

```bash
uv sync --frozen --group dev
uv run pytest

# Быстрый population/oracle smoke по всем readout families
uv run lid-estimation

# Проверить и безопасно распаковать exact archive
uv run lid-benchmarks-data verify --archive data/benchmarks.zip
uv run lid-benchmarks-data extract \
  --archive data/benchmarks.zip \
  --destination data/lid_benchmarks_exact

# Три независимые learned-модели на E8 Gaussian/Spaghetti/Sphere в одном
# Comet experiment на family; SDK читает credential из private config.
uv run python -m experiments.pilot pilot_model=diffusion
uv run python -m experiments.pilot pilot_model=rectified_flow

# Population/oracle matrix на всех данных статьи
uv run lid-estimation experiment=paper_oracle_matrix

# Learned jobs по всем model interfaces и трём seeds. Каждый реальный model
# config должен закреплять YAML artifact registry: отдельные checkpoint и
# resolved training config для каждой dataset × representation cell. Шаблоны
# fail-closed и эта команда не является готовой paper table без registries/bundles.
uv run lid-estimation -m experiment=paper_learned_matrix \
  models=diffusion,affine_fm,rectified_flow,schrodinger_bridge,scale_conditioned_nf,calibrated_nf,calibrated_cnf \
  models.seed=0,1,2

# Обычные Hydra overrides и multirun
uv run lid-estimation datasets.seed=7 runtime.limits.reference=1024
uv run lid-estimation -m datasets.seed=1,2,3

# Явный scalar-affine preprocessing; identity используется по умолчанию
uv run lid-estimation preprocessing=scalar_affine \
  preprocessing.scale=0.5 preprocessing.offset=-1.0
```

## GPU pilot: четыре обучаемых model family

Первый обучаемый pilot закреплён на трёх full-data representations:
`e8_gaussian4_pca`, `e8_spaghetti_pca`, `e8_sphere4_pca`. Один запуск семейства
последовательно обучает три независимых checkpoint. Исходный train
детерминированно делится на optimizer-fit и непересекающийся train-selection
holdout. После обучения scale/time выбирается по минимальному train-selection
MAE; индекс замораживается до первого обращения к benchmark validation/test.
Сохраняется полная train-selection curve, а validation и test вычисляются ровно
один раз в замороженной точке. Также сохраняются индексы разбиения, pointwise
predictions, targets, метрики, training history и sealed manifest.

```bash
# Локальная проверка Hydra-конфига без GPU
python -m experiments.pilot --cfg job pilot_model=diffusion
python -m experiments.pilot --cfg job pilot_model=rectified_flow
python -m experiments.pilot --cfg job pilot_model=scale_conditioned_nf
python -m experiments.pilot --cfg job pilot_model=schrodinger_bridge

# Secret-free dry run четырёх scheduler payloads
python -m experiments.cluster_submit \
  --config configs/cluster/shared_a100.yaml

# Реальная отправка с сервера из окружения block-diff
python -m experiments.cluster_submit \
  --config configs/cluster/shared_a100.yaml --submit

# Отправка только rectified-flow job (например, для безопасного resubmit)
python -m experiments.cluster_submit \
  --config configs/cluster/shared_a100.yaml \
  --family rectified_flow --submit
```

Launcher fail-closed по умолчанию фиксирует ровно четыре jobs по одной GPU (или одну
явно выбранную через `--family`), `queue_name=shared`,
`priority_class=shared-medium` и literal job description
`echimbulatov | ent-block-diffusion-eval #ID0137 #rnd`. Это неизменяемый
scheduler-only идентификатор, а не имя Comet experiment. Comet использует project
`lid-generalization` в workspace `dont4rootme` и создаёт полностью описательные
имена:
`lid-generalization-e8-suite-diffusion-train-mae-scale-selection-seed-0` и
`lid-generalization-e8-suite-rectified-flow-matching-train-mae-time-selection-seed-0`,
`lid-generalization-e8-suite-scale-conditioned-normalizing-flow-train-mae-scale-selection-seed-0`
и
`lid-generalization-e8-suite-brownian-schrodinger-bridge-train-mae-time-selection-seed-0`.
Источник этих имён
— соответствующие Hydra YAML в `configs/pilot_model/`; resolved config, summary
и manifest закрепляют то же имя. Training и validation series разделены по
dataset. API key читает сам Comet SDK из
mode-0600 `/home/jovyan/.comet.config`, выбранного публичной переменной
`COMET_CONFIG`; credential не попадает в Hydra YAML, scheduler payload, dry-run
или manifests.

Каждый запуск получает каталог вида
`artifacts/hydra/<experiment>/<date>/<time-with-microseconds>/`. Hydra сохраняет
в нём полностью
resolved `config.yaml`, `overrides.yaml`, `hydra.yaml` и лог; проверяемые
prediction/manifest artifacts лежат рядом в `results/`.

Standalone pilot сохраняет отдельный checkpoint для каждого из трёх datasets,
полную pointwise train-selection curve, single-scale validation/test predictions
и targets, training history/config, per-dataset и macro metrics. Корневые
`resolved_config.yaml`, `summary.json`, `artifact_registry.yaml` и
`manifest.json` хэшируют полный output; первые три итоговых файла также
прикрепляются к Comet. Scale/time выбирается только по LID targets
детерминированного train-selection holdout. При равном MAE tie-break идёт к
меньшему `sigma` для diffusion и к endpoint `t -> 1` для rectified flow.
Validation и test targets не участвуют в выборе.

## Один scheduler job для глобальной кампании

Полный Hydra campaign запускает **10 trainable configs × 39 cells = 390
обучений** одним процессом на одной A100. Из 39 cells 35 принадлежат exact
canonical archive (E1, E2, E5–E8), а четыре — явно отделённому generated
extension E3/E4. В десять моделей входят diffusion, legacy rectified FM,
scale-conditioned NF, Brownian Schrödinger bridge и шесть прямых/posterior
affine-FM вариантов. Calibrated NF/CNF исключены: для них нет trainer contract;
population oracle тоже исключён, потому что он не является обучаемой моделью.

Перед submit generated extension нужно один раз материализовать и строго
проверить (команды idempotent и никогда не дообучают PCA):

```bash
uv run python -m datasets.generated_e3_e4 \
  --pca data/lid_benchmarks_exact/benchmarks/pca.joblib \
  --output data/generated_benchmarks

uv run python -m datasets.generated_e3_e4 \
  --pca data/lid_benchmarks_exact/benchmarks/pca.joblib \
  --output data/generated_benchmarks --validate-only
```

Runner до Comet/GPU заново проверяет exact ZIP и распакованное дерево,
generated manifest/seal/PCA/upstream revision, а затем хэширует входы всех 39
cells. Внутри job модели и suites идут строго последовательно. Каждая cell имеет
стабильный identity-каталог, атомарный final publish и epoch-progress checkpoint;
повтор команды переиспользует валидные cells и продолжает оборванную cell, но
падает на повреждённом или изменившемся evidence. Это один scheduler job, но
ровно 10 отдельных, model-specific Comet experiments с durable local
spool/replay для временных сетевых сбоев.

Оценка чистого compute — примерно 50–90 GPU-hours; для планирования очереди,
валидации и диагностик разумный консервативный бюджет 100–140 часов. Scheduler
не задаёт walltime/retry, поэтому resumability является частью обязательного
контракта, а не оптимизацией.

```bash
# По умолчанию только secret-free план ровно одного Job
python -m experiments.global_cluster_submit \
  --config configs/cluster/shared_a100_global.yaml

# Единственная команда, которая разрешает реальную отправку
python -m experiments.global_cluster_submit \
  --config configs/cluster/shared_a100_global.yaml --submit
```

Global launcher сохраняет те же fair-use параметры, что и pilot:
`queue_name=shared`, `priority_class=shared-medium`, `region=A100-MT`,
`instance_type=a100.1gpu` и literal scheduler description
`echimbulatov | ent-block-diffusion-eval #ID0137 #rnd`. Model и dataset
identity отсутствуют в scheduler metadata. Единственная безопасная source-команда
выбирает Hydra group `campaign=all_suites_all_models`; отдельные описательные
Comet experiment names принадлежат campaign config, а не scheduler payload.
Credential по-прежнему читает только Comet SDK из mode-0600
`/home/jovyan/.comet.config`; dry run и payload не содержат API key.
После полного прохода `aggregate.json` и `unified_results.csv` пересчитываются
из sealed pointwise arrays. CSV содержит model/suite/dataset/representation,
split/readout, train-selected coordinate и known-LID, E1 stability, E5 paired
delta metrics; его SHA закреплён в final manifest и upload intent доставляется
во все десять Comet experiments.

## Структура

```text
configs/
  config.yaml                 Hydra defaults и run/sweep directories
  experiment/                 experiment matrices
  datasets/                   dataset choice + immutable registries
  preprocessing/              source-space -> model-space transform
  models/                     model/backend choice
  runtime/                    local/cluster limits
  pilot_model/                production pilot family/training groups
models/                       readouts, field contracts, model adapters
experiments/                  Hydra runner, metrics, aggregation, manifests
datasets/                     dataset loader, synthetic fixtures, archives
utils/                        shared provenance and integrity helpers
lid_benchmarks/               расширяемый benchmark-код в корне
paper/{eng,ru}/               приватные LaTeX-версии статьи (исключены из Git)
docs/                         protocol, model matrix, upstream audit
tests/                        unit, oracle, data-contract, Hydra integration
```

`lid_benchmarks/` импортирован из официального GitHub commit
`2dcb8e41015f53413ff1ddd049bb006c81a5df52` как обычная директория: новые
бенчмарки можно добавлять в основной репозиторий без отдельного submodule
workflow. Хэши исходных upstream-файлов закреплены в
`lid_benchmarks/UPSTREAM.yaml`; дополнительные файлы разрешены, незаметное
изменение оригиналов обнаруживается тестами.

Upstream generator не импортируется в основной Python process: его абсолютные
`generators.*` imports запускаются адаптером в отдельном frozen subprocess с
`cwd=lid_benchmarks`. Локально сгенерированный (не paper-parity) fallback:

```bash
uv run lid-benchmarks-generate
```

Это data-preparation utility с фиксированными provenance semantics, а не второй
способ конфигурировать эксперимент: все experiment runs по-прежнему идут только
через Hydra/YAML.

У upstream на зафиксированном commit нет `LICENSE`. Для внутренней работы код
импортирован по явному запросу, но перед публичным распространением нужно
получить разрешение авторов или лицензию.

## Что валидируется

Контур разделяет два уровня доказательств:

1. `empirical_gaussian_channel_oracle` проверяет population identities для
   Gaussian diffusion, affine/rectified flow matching, Brownian Schrödinger
   bridge и двух допустимых normalizing-flow endpoint interfaces;
2. learned-модель экспортирует стандартный безопасный bundle примитивов
   (`score`, `velocity`, `divergence`, checkpoint/config/dataset/query hashes,
   model seed и trace metadata), после чего независимый readout считает LID и
   метрики.

Learned model config закрепляет SHA отдельного YAML artifact registry. Registry
покрывает ровно запрошенную matrix и для каждой dataset × representation cell
указывает относительные пути и SHA checkpoint/resolved training YAML, а также
full training-dataset и preprocessing identities. Runner повторно хэширует все
эти файлы до записи результата.

Preprocessing — отдельная Hydra-группа, а не скрытый код model adapter.
`identity` численно сохраняет выбранные строки; `scalar_affine` требует finite
ненулевой scale и finite offset. Manifest связывает canonical transform SHA,
raw/model-space hashes и полный training-dataset hash. Все `physical_scales` и
scale selection определены в model space.

Oracle-результат всегда помечается как
`population_empirical_channel_not_trained_model`. Это не замена neural runs:
текущая версия статьи не задаёт единственную обязательную архитектуру и
optimizer. Pilot поэтому явно версионирует выбранную reference-архитектуру и
optimizer в Hydra YAML и не выдаёт их за «официальную реализацию» статьи.
Общий learned-run по-прежнему падает до создания output, если SHA или bundle
хотя бы одной matrix cell отсутствует.

## Данные

Для paper-parity используется опубликованный авторами password-protected
`benchmarks.zip`, а не повторная генерация: upstream прямо предупреждает, что
generator может дать другие данные. Архив весит `4,685,463,657` bytes и требует
около `6.9 GB` после распаковки. Canonical archive и locally generated fallback
имеют разные provenance identities и никогда не агрегируются вместе.

Exact archive закреплён SHA-256
`ce0d153a1a78a3a752b29ec2e60167134b6b20c3249db2fe92f9fc1b8b8a9181`.
Runner повторно проверяет архив, полное распакованное дерево и pinned upstream
source до чтения первой dataset cell.

Registry находится в
`configs/datasets/registry/paper_benchmarks.yaml`. Он проверяет все 28 каталогов,
split sizes, shapes, finite numeric arrays, known LID и transformation deltas.
Known upstream defects (включая старый Spaghetti target) обрабатываются явно,
не скрытыми правками данных. В частности, canonical Spaghetti проверяется по
raw split-контракту `train=20`, `val=20`, `test=1`, после чего effective LID 1
выставляется только в памяти для всех split; per-split correction сохраняется
в input provenance/manifest.

Подробности: `docs/ARCHITECTURE.md`, `docs/EXPERIMENT_PROTOCOL.md`,
`docs/MODEL_MATRIX.md`, `docs/LEARNED_BUNDLES.md`,
`docs/UPSTREAM_AUDIT.md`.
