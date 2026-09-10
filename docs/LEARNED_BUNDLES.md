# Learned model bundle contract

## Почему это отдельный backend

Статья задаёт endpoint readout interfaces, но не фиксирует neural architecture,
optimizer, preprocessing policy или checkpoint. Репозиторий поэтому не приписывает
авторам несуществующую reference implementation. Любой training stack может
участвовать в benchmark, только если экспортирует одинаковый строгий контракт;
oracle и learned results имеют разные `evidence_level`.

## Hydra model groups

В `configs/models/` есть семь семейств:

| Hydra config | Family | Readouts |
|---|---|---|
| `diffusion` | Gaussian diffusion | `diffusion_flipd_full` |
| `affine_fm` | independent affine FM | response, full |
| `rectified_flow` | rectified-flow specialization | response, full |
| `schrodinger_bridge` | Brownian SB / current-score | forward response/full, current full |
| `scale_conditioned_nf` | fixed-point scale-conditioned NF | fixed density |
| `calibrated_nf` | calibrated singular NF | native scale velocity |
| `calibrated_cnf` | calibrated singular CNF | native time velocity |

`artifact_registry` и `artifact_registry_sha256` намеренно равны `null` в
шаблонах. Первый указывает на Hydra/YAML registry, второй закрепляет его точный
lowercase SHA-256. Registry обязан покрывать ровно все запрошенные
dataset × representation cells — без пропусков и лишних entries.

```yaml
schema_version: 1
artifacts:
  e8_gaussian4_pca/dataset:
    checkpoint_path: e8_gaussian4_pca/model.ckpt
    checkpoint_sha256: <64 hex>
    training_config_path: e8_gaussian4_pca/training.yaml
    training_config_sha256: <64 hex>
    training_dataset_sha256: <64 hex>
    preprocessing_sha256: <64 hex>
```

Artifact paths всегда относительны директории registry. Runner реально хэширует
registry, checkpoint и training config; абсолютные display paths не входят в
scientific input SHA. Один глобальный checkpoint для paper matrix запрещён:
каждая cell получает собственную запись.

## Layout

Для model `M`, seed `S`, dataset `D`, representation `R` и scale index `K`:

```text
<bundle_root>/
  M/
    seed-S/
      D/
        R/
          scale-KKK/
            validation.npz
            validation.json
            test.npz
            test.json
```

NPZ хранит только numeric arrays и загружается с `allow_pickle=False`. JSON —
не experiment config, а immutable output metadata: scalars формулы и provenance.
Он обязан содержать schema version, model/family/seed, checkpoint и training
config SHA, full training-dataset SHA, selected raw dataset/query SHA,
preprocessing SHA, фактический model-space query SHA, representation, split,
число строк, physical scale, полный список readouts и trace
backend/probes/seed. Лишние и недостающие ключи/примитивы отклоняются.

Каждый `training.yaml` должен быть полностью resolved Hydra YAML без
интерполяций и содержать точный `provenance` block: schema version,
model name/family/seed, dataset, representation, full training-dataset SHA и
preprocessing SHA. Эти значения независимо сверяются с registry и prepared
dataset, поэтому registry не может служить единственным self-claim.

Preprocessing выбирается только Hydra-группой `configs/preprocessing/`.
Bundle metadata одновременно привязан к raw query и к transformed query;
совпадение числа строк без совпадения обоих hashes недостаточно. Physical scale
интерпретируется только в model space.

## Two-phase validation

1. До создания matrix output runner проверяет SHA registry, его точное покрытие,
   реальные checkpoint/training-config files и все bundle files во всех
   dataset × representation × scale × split cells.
2. При materialization каждая bundle читается и проверяется повторно. Hash
   полного input inventory входит в run identity вместе с raw/model selected
   dataset SHA и preprocessing SHA.

Затем task-specific curve на отделённом от optimizer-fit подмножестве
source-train выбирает scale: target-based MAE для known-LID либо target-free
reference stability для E1/E5. Индекс замораживается до чтения
validation/test; validation и test вычисляются только в одной выбранной точке,
без сохранения их scale curves. Test prediction сохраняется pointwise, а
aggregate report пересчитывается из raw `.npy`.
Checkpoint, training config, dataset rows или trace seed нельзя поменять с
повторным использованием старой cell: изменится run ID либо provenance check
упадёт.

## Статус completed evidence и переносимость

Production learned campaign была выполнена с versioned configs, checkpoint SHA,
per-cell provenance и sealed pointwise evidence; её состав и результаты
зафиксированы в [`EXPERIMENT_RESULTS.md`](EXPERIMENT_RESULTS.md). Локальная
consolidated копия намеренно компактна: она содержит analysis-ready CSV,
aggregates, manifests и hashes, но не checkpoint и pointwise `.npy`.

Поэтому существуют два разных утверждения:

- completed remote evidence подтверждает фактически выполненную study;
- чистый checkout с шаблонными `artifact_registry: null` не может сам
  воспроизвести learned table без нового обучения или отдельно сохранённых
  versioned training artifacts.

Для нового confirmatory прогона registry, training configs и checkpoint снова
обязательны. Compact bundle достаточен для воспроизводимого table-level анализа
и проверки hashes, но не для переоценки pointwise predictions. Исходный remote
evidence хранит sealed pointwise/cell outputs и checkpoint-hash attestation, но
не checkpoint bytes: они удалены по `prune_after_cell_evaluation`.
