# Архитектура экспериментального контура

## Основная граница

Один experiment cell проходит пять независимых слоёв:

```text
dataset + split
      ↓
declared preprocessing (raw -> model space)
      ↓
learned field / population oracle
      ↓
mathematical LID readout
      ↓
trace backend (exact or stochastic)
      ↓
metrics + immutable run manifest
```

Это разделение обязательно. Например, Rectified Flow model должен предоставить
`v_t(t x)` и `div v_t(t x)`, но не имеет права сам возвращать «готовую LID»:
коэффициент `t(1-t)` и boundary correction принадлежат readout layer и
проверяются независимо.

## Слои

### Dataset

`datasets.registry` читает только публичный upstream layout (`train/`, `val/`,
`test/`, файлы `dataset.npy`, `lid.npy`, `coefficients.npy`, `labels.npy`).
Loader проверяет формы, числовые dtype, finite values, длины sidecars и known-LID
policy. Registry описывает representation и научный анализ; модель не знает
имён upstream-папок.

### Preprocessing

`configs/preprocessing/` задаёт единственную source-to-model-space границу.
По умолчанию `identity` сохраняет массивы побитово; опциональный
`scalar_affine` применяет один finite nonzero scale и finite offset ко всем
reference/validation/test features. Runner сохраняет canonical spec и SHA,
отдельные raw/model split identities, selected-dataset identities и identity
полного raw training set. Nearest-neighbour calibration, field queries и все
physical scales после этой границы всегда находятся в model space.

### Field/model

Есть два backend class:

- `EmpiricalGaussianChannel` — deterministic finite-sample population oracle;
- learned adapter — внешний training stack экспортирует безопасный NPZ (без
  pickle) и JSON metadata с checkpoint/config hashes.

Они не сравниваются как равноправные empirical algorithms. Oracle валидирует
тождество и finite-sample target; learned backend измеряет approximation,
derivative и scale-selection error.

### Readout

`models.readouts` — NumPy-only реализация ровно тех формул, которые
заявлены в статье. Capability contract в `models.fields` отклоняет
неполный bundle до подсчёта метрик.

### Trace

Для small-D validation предпочтителен exact Jacobian trace. Для high-D learned
runs bundle может объявить Hutchinson; строгая metadata schema сохраняет
backend, число probes и seed. Текущий runner не пересчитывает Jacobian и не
проверяет empirical standard error: agreement с exact trace на small-D fixture
остаётся обязательным versioned тестом внешнего exporter. Выбор trace backend
не должен неявно менять model checkpoint.

### Metrics/provenance

Raw pointwise predictions сохраняются вместе с aggregate metrics. Manifest
содержит resolved config, source-tree hash, Git state, upstream SHA, dataset
hash, environment и hashes каждого output. Cell считается complete только
после повторной проверки этих hashes.

## Benchmark source boundary

Официальный `LID-Benchmarks` импортирован в top-level `lid_benchmarks/`, чтобы
новые генераторы и бенчмарки коммитились вместе с основным проектом. Исходные
файлы upstream закреплены по SHA-256 в `lid_benchmarks/UPSTREAM.yaml`; новые
файлы разрешены, изменение исходных обнаруживается provenance-тестом.
`datasets.upstream_generator` — отдельная data-preparation boundary: после
проверки source hashes он запускает upstream через `uv run --frozen` в
subprocess с `cwd=lid_benchmarks`, не меняя `sys.path` основного процесса.
Модели не импортируют внутренности генератора напрямую, поэтому exact archive
остаётся canonical, а regenerated data — отдельным provenance class.

## Hydra composition

Экспериментальный configuration graph существует только в YAML:

```text
configs/config.yaml
  ├── experiment/<matrix>.yaml
  ├── datasets/<source>.yaml
  ├── preprocessing/<transform>.yaml
  ├── models/<backend>.yaml
  └── runtime/<environment>.yaml
```

Hydra сохраняет resolved config и overrides для каждого run/multirun. Python
runner получает уже скомпонованный `DictConfig` и не имеет второго TOML/JSON
пути конфигурации.
