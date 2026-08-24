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

Затем validation curve выбирает scale без target labels, test prediction
сохраняется pointwise, а aggregate report пересчитывается из raw `.npy`.
Checkpoint, training config, dataset rows или trace seed нельзя поменять с
повторным использованием старой cell: изменится run ID либо provenance check
упадёт.

## Что ещё нужно для настоящей paper table

Нужны versioned training configs и checkpoints конкретных реализаций. Это
научный выбор, отсутствующий в текущем тексте статьи, а не инфраструктурная
деталь. После их фиксации каждый model YAML получает путь и SHA per-cell artifact
registry, и тот же Hydra multirun становится confirmatory learned matrix. До
этого oracle matrix можно использовать только для проверки формул, scale/data
plumbing и benchmark aggregation.
