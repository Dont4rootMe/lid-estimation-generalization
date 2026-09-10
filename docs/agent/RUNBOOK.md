# Операционный runbook

Документ перечисляет актуальные точки запуска и предотвращает смешение
исторического последовательного runner с production DAG. Перед реальным
запуском нужно прочитать предметный протокол и проверить `--help` текущего
модуля: команды ниже не заменяют scientific config.

## Локальная проверка

```bash
uv sync --frozen --group dev
uv run pytest
```

Population/oracle smoke:

```bash
uv run lid-estimation
```

## Глобальная learned campaign v1 (immutable)

Два пути существуют одновременно:

- `python -m experiments.global_campaign` — legacy sequential execution из
  `configs/global_campaign.yaml`;
- `python -m experiments.global_parallel` — production cell-DAG. Модуль до
  вычисления campaign identity заменяет execution block на неизменяемый
  профиль `h100_8gpu_cell_dag`: 8 workers, по одному видимому H100 на worker,
  без DDP и межмодельной синхронизации.

Безопасный H100 preflight:

```bash
python -m experiments.global_parallel --preflight-only
```

Полный запуск использует ту же команду без `--preflight-only`. Он требует
канонических данных, generated E3/E4 extension, CUDA/H100 preflight и Comet
credential согласно существующим контрактам. Координатор один записывает
ledgers, telemetry, aggregates и manifest; workers публикуют только
непересекающиеся cell-директории.

Корневой README пока подробно описывает legacy single-A100 orchestration. При
расхождении с production поведением каноничны `experiments/global_parallel.py`,
`configs/global_campaign.yaml`, тесты и этот явно зафиксированный выбор профиля.

## Benchmark rerun v2

Завершённая job — **9344**, `lerobot-research-lid-v2-r09-resume`, normal 8gpu,
старт 2026-09-08 12:51:57 UTC, `bot-8h100-12-0`. Commit
`6d041365b9bc27de5054293be6c1318766ddf911`, identity
`64ae8a405a42c248d14cab5bb1e2fd2b19e2e4c6ca5fb774c4acff50a5732c62`.
Root `.../runs/lerobot_research_lid_v2/resume-20260908__6d04136/campaign/lid-global-e1-e8-canonical-plus-generated-e3-e4-vp-ve-all-models-v2__64ae8a405a42c248d14c`.
Completed2026-09-08 16:50:03 UTC, exit0. Полная матрица429/429:
147 reused +282 новых cells,11×39. Full remote
`validate_global_campaign(root, expected_campaign_identity=identity,
project_root=repo, verify_inputs=True)` PASS в16:57:02 UTC.
Независимый обход подтвердил отсутствие checkpoints/progress и incomplete
директорий. Новых запусков или изменений sealed roots не требуется.
Единственный local compact bundle:
`artifacts/results/lid_benchmark_v2/64ae8a405a42c248d14c/` —2368 CSV rows,
7 исходных файлов (5.30MB), README с описанием каждого файла и сводкой.
SHA копии,429 cell-ID joins, coverage и finite numeric fields проверены
локально. Полная pointwise/source проверка требует серверного evidence;
compact bundle не содержит cells/NPY/checkpoints или отдельных follow-ups.
Монитор удалён после доставки bundle. Исторический v1 bundle не изменён.
Полный паспорт — `artifacts/launches/lid_v2_resume_20260908.json`.

Далее — история superseded attempts, не команды для нового запуска.

Предыдущая job — **9326**, `lerobot-research-lid-v2-r08-resume`, normal 8gpu,
`bot-8h100-12-0`, старт 2026-09-08 12:29:10 UTC. Commit
`03eb31acaa2a9dd24fc97fb87404d4e5eace6405`, identity
`1274e2bc716494a39ace56d8d89a80ab60eb855abd523c911ce15e615f7de74d`.
Root `.../runs/lerobot_research_lid_v2/resume-20260908__03eb31a/campaign/lid-global-e1-e8-canonical-plus-generated-e3-e4-vp-ve-all-models-v2__1274e2bc716494a39ace`.
Полные пути и pins в `artifacts/launches/lid_v2_resume_20260908.json`.
644 local tests, prelaunch full input/147-cell validation PASS, 147 immutable
copies plus copied24canary PASS. GPU replay PASS; затем task FAILED в
12:31:56 UTC до обучения из-за race в preflight (exit 0 уже отчитавшегося
worker ошибочно считался аварией). Monitor удалён, исправление готовится.

Обновление 2026-09-08: task **9201 FAILED** 2026-09-07 15:37:30 UTC, после
147/429 cells, в FM endpoint ratio с нулевым знаменателем. Следующие сведения
о её running state и monitor исторические; monitor сейчас отсутствует.
Diagnostic-maintenance continuation готовится с новым source-bound root.
Только для закреплённой task 9201 `LID_V2_RESUME_FROM=<полный campaign root>`
включает importer в том же `scripts/run_v2_full_job.sh`: проверка неизменных
научных config/input, 147 cell validators, копирование bytes без изменения
идентичностей, проверка старого canary, replay на failed-cell checkpoint,
свежий 24-worker GPU preflight, затем оставшиеся 282 cells. Исходный root
не модифицируется. `state/resume_lineage.json` и `campaign.json` связывают
старый source каждого imported result с новым execution source. Это не
разрешение переиспользовать результаты после изменений моделей или данных.
Ниже запрет reuse других superseded roots остаётся в силе.

Актуальная task: **9201**, `lerobot-research-lid-v2-r07-outcomes`, normal 8gpu;
commit `03c12b04561575fbef5c89c6876138bf7d7c2e44`, canary v5, 587 tests passed.
Стартовала 2026-09-07 12:43:25 UTC на `bot-8h100-10-0`; в 12:46 UTC все24
workers реально обучаются (2000–8000 шагов), GPU0–7: 99–100% utilization.
Дальнейший статус проверять через `bot info/logs 9201`; активен read-only
`lid-benchmark-results-monitor`, каждые15 минут, уведомления только о переходах.
Root `/mnt/virtual_ai0001071-01239_SR006-nfs2/afedorov/runs/lerobot_research_lid_v2/outcomes-20260907__03c12b0`.
Campaign suffix `campaign/lid-global-e1-e8-canonical-plus-generated-e3-e4-vp-ve-all-models-v2__92fe3436909e3bf28427`.
Identity `92fe3436909e3bf28427c380fdaa463e7f652c9d884f8ffe28cdc02cd11ff2b7`.
Это полный 429-cell запуск без veto по качеству, не результат уже завершённого
бенчмарка. Старые task/root ниже исторические; не использовать их для resume.

Обновление 2026-09-07: task 9196 уже FAILED (NF/Gaussian quality threshold),
а не pending. Для следующей попытки нужен canary v5: все empirical quality,
utilization/headroom и ETA thresholds только диагностические; finite/data/
provenance/coverage проверки блокируют по-прежнему. PASS подтверждает integrity,
не хорошее качество и не сходимость. Бюджет остаётся 128000 × batch 256,
inline eval/prune и одна 8-H100 job с 24 logical workers сохранены.
Нельзя запускать старый commit v4 или переиспользовать его canary attestation.

Текущая попытка: **9196**, `lerobot-research-lid-v2-r06-128k`, normal `8gpu`.
При отправке 2026-09-07 стоит первой в очереди, старт ещё не подтверждён.
Commit `6ebc9845a4dba22ce5a2f7bc8d9fa41fa8a9379d`, identity
`07c79a6448974971efafbfa2afe5e694814ba1bc5e0703fe223b6cd19ee45861`.
Root `/mnt/virtual_ai0001071-01239_SR006-nfs2/afedorov/runs/lerobot_research_lid_v2/extended128k-20260907__6ebc984`.
Campaign suffix `campaign/lid-global-e1-e8-canonical-plus-generated-e3-e4-vp-ve-all-models-v2__07c79a6448974971efaf`.
Canary v4 и full DAG: 128000 шагов, batch 256, 24 logical workers / 8 H100.
Plateau является diagnostic; оценка лучшего native-loss checkpoint и inline
eval/prune сохранены. Полный suite: 579 passed. Все следующие параметры task
9138 относятся к старой 32000-step попытке, не использовать их для resume новой.

Актуальный исход task `9138`: **failed**, `2026-09-06T22:49:43Z`, exit 1.
VE/Arrows отклонён native-loss plateau gate (tail improvement 0.2558695 > 0.10).
Сохранены 13 final cell summaries и 11 incomplete directories; state ledger
пуст, canary report, campaign manifest, aggregate и unified CSV отсутствуют.
Следующие launch/start сведения являются историей попытки. Новый полный bundle
не создан; монитор этой terminal-задачи удалён.

- Source branch: `afedorov/vp-benchmark-rerun-v2`.
- Launch commit: `6e03ed8c41909b0009dac879747a8819c08bb474`.
- CloudBot task: `9138` (`lerobot-research-lid-v2-r05`), normal `8gpu`;
  запущен `2026-09-06T22:42:54Z` на `bot-8h100-6-0`. Стартовый лог содержит
  все worker 0--23 и training progress; live snapshot всех восьми H100
  показал 99--100% utilization.
- Task root:
  `/mnt/virtual_ai0001071-01239_SR006-nfs2/afedorov/runs/lerobot_research_lid_v2/overnight-20260907__6e03ed8c4190`.
- Campaign root:
  `/mnt/virtual_ai0001071-01239_SR006-nfs2/afedorov/runs/lerobot_research_lid_v2/overnight-20260907__6e03ed8c4190/campaign/lid-global-e1-e8-canonical-plus-generated-e3-e4-vp-ve-all-models-v2__c259d9cd484beb01d280`.
- Campaign identity:
  `c259d9cd484beb01d280cf867db7d61fc692f9d8014c03ee2d3b4d3062401158`.

Tasks `9093`, `9098` и `9113` являются superseded failures. Первый остановился до
обучения на вложенной Hydra composition. Второй запустил восемь quality cells,
но canary неверно применил score-to-denoiser формулу к VE, которая обучается
непосредственно как `x0`-denoiser. В обоих root нет sealed cells; partial
outputs и checkpoints не переиспользуются и не являются результатами. Task
`9113` поднял все 24 worker, но упал до PASS report на one-ULP выходе верхней
reference-lambda за строгий affine-FM support; его 12 final и 12 incomplete
directories также не переиспользуются. Fix канонизирует только roundoff на
генерируемых границах и требует новый source-bound root.

Task `9131` также superseded: он прошёл структурные и reconstruction checks,
но старый canary ошибочно превращал слабый high-D VP/Arrows benchmark outcome
в ошибку целостности запуска. Его pointwise MAE 23.43 и H16/H64 disagreement
2.54 не скрываются и не объявляются хорошим результатом. В canary v3 high-D
quality и boundary/no-knee outcomes являются обязательными diagnostics, тогда
как nonfinite значения, структурные ошибки, low-D exact checks и нарушения
provenance остаются блокирующими. Report validator самостоятельно открывает и
проверяет захешированные retained summary/quality artifacts.

Одна CloudBot job запускает:

```bash
bash scripts/run_v2_full_job.sh
```

Job сначала выполняет `experiments.v2_canary`, а после валидного PASS
переходит к `experiments.global_parallel_v2`. Canary и full используют один
resolved config, campaign identity и output root. Canary запускает 24 logical
workers на восьми H100 — три независимые batch-256 cell на физическое
устройство. Восемь quality cells и 16 companion cells являются частью матрицы
11 × 39 = 429, после PASS проходят точную повторную валидацию и
переиспользуются при full run; остаётся 405 cells.

Каждая production cell выполняет `train → eval → seal → checkpoint prune`.
Компактные точки входа находятся непосредственно в campaign root:
`canary_report.json`, `campaign.json`, `aggregate.json` и
`unified_results.csv`; отдельные follow-up/NF result roots не создаются.

Launcher требует точные значения переменных `LID_V2_JOB_ROOT`,
`LID_V2_EXPECTED_COMMIT`, `LID_V2_EXPECTED_GIT_TREE_SHA256`,
`LID_V2_EXPECTED_ARCHIVE_SHA256`, `LID_V2_EXPECTED_CAMPAIGN_IDENTITY`,
`LID_V2_EXPECTED_CAMPAIGN_CONFIG_SHA256`,
`LID_V2_EXPECTED_INPUT_INVENTORY_SHA256`,
`LID_V2_EXPECTED_DECLARED_SOURCE_SHA256` и
`LID_V2_ARROWS_REVIEWED_GATE`. Последняя указывает только на визуально
проверенный `arrows_gate_reviewed.json`; credentials в документации не
сохраняются.

Canary fail-closed требует полного измерения 0→32000 шагов для каждой из 24
cells. Пока PASS report отсутствует, наличие любой exact final или stable
incomplete cell требует нового task-specific root: иначе reuse занизил бы ETA
или исказил telemetry. После PASS все 24 sealed cells переиспользуются. Full
DAG сохраняет deterministic resume и фиксирует mapping
`logical_worker_slot % 8` вместе с lane/device identity.

## NF ablation и readout follow-up

Staged NF ablation запускается отдельным модулем и требует sealed baseline:

```bash
python -m experiments.nf_ablation \
  --baseline-root <sealed-global-campaign> \
  --output-root <nf-ablation-output> \
  --preflight-only
```

Production требует восемь H100 workers; `--allow-cpu` предназначен только для
локальных integration tests. Полный запуск выполняется без `--preflight-only`.

Evidence-bound readout follow-up:

```bash
python -m experiments.nf_readout_followup \
  --baseline-root <sealed-global-campaign> \
  --source-ablation-root <sealed-nf-ablation> \
  --reviewed-evidence-manifest \
    configs/campaign/nf_readout_followup_reviewed_evidence.yaml \
  --output-root <nf-followup-output> \
  --preflight-only
```

Follow-up не переобучает и не перезаписывает baseline; он добавляет
schema-compatible строки только после проверки закреплённых identity и SHA.

## Локальный консолидированный bundle

На текущем host хранится:

```text
artifacts/results/researh_l1_pretrain_ultra/a80ddd279bb32d666649/
```

Первым читается его `README.md`, затем `campaign.json`, `resolved_config.yaml`
и `input_inventory.json`. Analysis-ready таблица — `unified_results.csv`
(2106 строк); связанный отчёт находится в
`artifacts/analysis/researh_l1_pretrain_ultra/ANALYSIS.md`.
Tracked-описание study и результатов —
[`../EXPERIMENT_RESULTS.md`](../EXPERIMENT_RESULTS.md).

Минимальная read-only проверка текущей объединённой локальной копии:

```bash
uv run python - <<'PY'
import csv
import hashlib
import json
from pathlib import Path

root = Path("artifacts/results/researh_l1_pretrain_ultra/a80ddd279bb32d666649")
manifest = json.loads((root / "campaign.json").read_text())
payload = (root / "unified_results.csv").read_bytes()
aggregate = (root / "aggregate.json").read_bytes()
rows = list(csv.DictReader(payload.decode().splitlines()))
expected_identity = "2ebedf14e7f06d8fb84b8dba030ce5817a41d2fc49ecec3ba1cb5f89a4110441"
expected_csv_sha = "827a56ab5cd67ced10a07b3b28d06c23471db44201b87caf7c34d8f18aa7477c"
expected_aggregate_sha = "78d9ba71259aa4d300e2f93f8065c7653612cb015683a96ba8ac297a40129750"
assert manifest["complete"] is True
assert manifest["results_bundle_identity"] == expected_identity
assert manifest["unified_results_sha256"] == expected_csv_sha
assert hashlib.sha256(payload).hexdigest() == expected_csv_sha
assert manifest["aggregate_sha256"] == expected_aggregate_sha
assert hashlib.sha256(aggregate).hexdigest() == expected_aggregate_sha
assert manifest["total_physical_training_runs"] == 468
assert len(rows) == 2106
assert len({row["model_variant"] for row in rows}) == 13
assert sum(row["is_primary_readout"] == "True" for row in rows) == 1014
print(expected_identity, "OK")
PY
```

Эта проверка подтверждает локальный consolidated CSV/manifest. Исходный sealed
remote evidence позволяет полнее проверить cell/pointwise artifacts, provenance
и attested checkpoint hashes. Сами checkpoint bytes были удалены по
`prune_after_cell_evaluation`, поэтому повторная проверка их содержимого и
reinference невозможны без нового обучения. Manifest текущего bundle фиксирует,
что remote full и local compact validators ранее прошли.

Ограничения:

- каталог исключён из Git и отсутствует в чистом checkout;
- локальная копия не содержит checkpoint и pointwise `.npy`;
- descriptive leaderboard не является статистическим тестом;
- baseline/FM представлены одним seed, а два NF follow-up seeds не дают
  формального confidence interval;
- generated E3/E4 не смешиваются с canonical headline;
- scale/time/lambda выбирается только на disjoint train-selection; validation
  используется лишь для явно помеченного выбора model/readout, test — как
  confirmatory split.

Нельзя переносить отдельные числа в статью без ссылки на конкретный row set,
метрику, split, variant, bundle identity и ограничения анализа.
