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
# Comet experiment на family. COMET_API_KEY читается только из environment.
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

## GPU pilot: diffusion и rectified flow

Первый обучаемый pilot закреплён на трёх full-data representations:
`e8_gaussian4_pca`, `e8_spaghetti_pca`, `e8_sphere4_pca`. Один запуск семейства
последовательно обучает три независимых checkpoint, выбирает scale по
target-free validation stability, затем сохраняет полные validation/test curves,
pointwise predictions, targets, метрики, training history и sealed manifest.

```bash
# Локальная проверка Hydra-конфига без GPU
python -m experiments.pilot --cfg job pilot_model=diffusion
python -m experiments.pilot --cfg job pilot_model=rectified_flow

# Secret-free dry run двух scheduler payloads
python -m experiments.cluster_submit \
  --config configs/cluster/shared_a100.yaml

# Реальная отправка с сервера из окружения block-diff
python -m experiments.cluster_submit \
  --config configs/cluster/shared_a100.yaml --submit
```

Launcher fail-closed фиксирует ровно две jobs по одной GPU, `queue_name=shared`,
`priority_class=shared-medium` и literal job description
`echimbulatov | ent-block-diffusion-eval #ID0137 #rnd`. Comet создаёт по одному
experiment на model family; training и validation series разделены по dataset.
API key читается только во время исполнения из mode-0600 Comet config и не
попадает в Hydra YAML, scheduler environment, dry-run или manifests.

Каждый запуск получает каталог вида
`artifacts/hydra/<experiment>/<date>/<time-with-microseconds>/`. Hydra сохраняет
в нём полностью
resolved `config.yaml`, `overrides.yaml`, `hydra.yaml` и лог; проверяемые
prediction/manifest artifacts лежат рядом в `results/`.

Standalone pilot сохраняет отдельный checkpoint для каждого из трёх datasets,
полные pointwise validation/test curves, выбранные predictions и targets,
training history/config, per-dataset и macro metrics. Корневые
`resolved_config.yaml`, `summary.json`, `artifact_registry.yaml` и
`manifest.json` хэшируют полный output; первые три итоговых файла также
прикрепляются к Comet. Scale выбирается по validation curves без доступа к LID
labels: к меньшему `sigma` для diffusion и к endpoint `t -> 1` для rectified
flow при равной stability.

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
не скрытыми правками данных.

Подробности: `docs/ARCHITECTURE.md`, `docs/EXPERIMENT_PROTOCOL.md`,
`docs/MODEL_MATRIX.md`, `docs/LEARNED_BUNDLES.md`,
`docs/UPSTREAM_AUDIT.md`.
