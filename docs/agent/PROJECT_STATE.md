# Текущее состояние проекта

Срез состояния на **2026-09-10**.

## Цель

Статья исследует, при каких вероятностных свойствах калиброванного
генеративного семейства локальная внутренняя размерность (LID) целевого
распределения восстанавливается из доступного выхода модели. Центральный тезис:
LID определяется геометрией предельного семейства и его физическим масштабом,
а не названием архитектуры.

## Рукопись

- Рабочая версия текущей редакционной задачи:
  `/Users/artemon/projects/lid_generalization_paper/paper/ru/`.
  Это отдельный checkout рукописи; одноимённый каталог в репозитории
  исследований не подменяет указанную автором версию.
- Английская версия: `paper/eng/`; сейчас это исторический исходник, а не
  синхронный перевод.
- По прямому запросу автора от 2026-09-05 текущая подключённая рукопись
  переведена на научный английский в
  `/Users/artemon/projects/lid_generalization_paper/result/`:
  аннотация, разделы 1--4, литература и приложения A--H. Это перевод,
  а не новая содержательная редакция; русский источник и `paper/eng/`
  сохранены без изменений. Порядок подключений, формальные утверждения,
  формулы и ключи ссылок сохранены. Сборка: `result/build.sh`;
  пользовательский PDF: `result/output/pdf/endpoint_channels_lid_en_theory.pdf`.
- Последующей адресной правкой по запросу автора в английском `result/`
  расширены только два вводных абзаца раздела 3: переход от вопроса об
  общей методологии к каналу зашумления, плотности наблюдений и условной
  реконструкции теперь объясняется до формальной постановки. Определения,
  формулы, теоремы и доказательства не менялись; русский текст не
  синхронизировался. Английский PDF после правки содержит 28 страниц,
  основной текст заканчивается на странице 7, приложения начинаются на 10.
- По комментариям автора в `endpoint_channels_lid_en_theory (2).pdf`
  переработаны английские аннотация и введение: мотивация от выборочных
  ограничений к повторному использованию генеративной модели, общий
  критерий для ядра возмущения, связь с реконструкцией и два теоретических
  вклада. Экспериментальный вклад и заявления о сравнительном качестве
  убраны из этих двух частей; результаты study этим не переоцениваются.
  В аннотации и введении больше нет необъяснённых `channel`, `endpoint`
  и `physical scale`; терминология глав 2--4 и приложений пока сохранена.
  Существующая предобученная модель подходит только при наличии нужного
  семейства распределений, масштаба и выходов; область теорем не расширена.
  Главы 2--4, приложения, русский источник и шаблон не изменены.
  PDF по-прежнему содержит 28 страниц, основной текст заканчивается на 7,
  приложения начинаются на 10. Сохранены 140 блоков выключной математики,
  172 метки и 18 формальных утверждений; теперь используются 25 ключей
  литературы вместо 26 после редакционной правки цитат во введении.
- `paper/ru/main.tex` подключает аннотацию, `sections/01_introduction.tex`,
  `sections/02_related_work.tex`, теорию `sections/03_unified_setup.tex`,
  модельные следствия `sections/04_model_consequences.tex`, библиографию
  и адресные доказательства/расширения.
- Исторические `04_master_criterion.tex`, `05_model_adapters.tex`, разделы
  06--08 и старые приложения не подключены и не считаются заново одобренными
  по факту наличия. Они не удалены.
- Мотивация и главный вопрос остаются во введении. По прямому уточнению
  автора раздел 2 — исключительно related work: четыре связанных абзаца
  без формул, постановки и новых обозначений. Выборочные и генеративные
  оценки объединены в первом абзаце; далее идут реконструкция, модельные
  выходы и границы известных обобщений. Приоритет общей теоремы каналов
  сформулирован с оговоркой «насколько известно авторам», без заявления
  о первом архитектурно-независимом методе или охвате любых моделей.
  Определения LID, канала,
  масштаба и статистик возвращены в раздел 3. Введение и блок «Основные
  результаты» сохранены. Сравнение композиции пяти опубликованных статей
  ICLR 2025 и принятые редакционные правила зафиксированы в README рукописи.
- Последний авторский лимит — семь страниц на весь текущий основной текст
  до литературы, включая аннотацию и введение. Глава 3 сохраняет цепочку
  «канал — две статистики — разложение — касательная геометрия и
  однородность — класс K — мастер-теорема — смысл геометрических пределов».
  В ней три формальных блока: разложение 3.1, мастер-теорема 3.2 и
  касательное следствие 3.3. Определение класса K, лемма Эйлера и расчёт
  границы перенесены в адресные приложения; их роль объяснена в основном
  тексте. Глава 4 содержит только особенности семейств и четыре итоговых
  следствия 4.1--4.4, без промежуточных выкладок. Принцип адаптера и сводная
  таблица находятся в приложении C, модельные доказательства — в E--G.
  NF по калиброванному пути плотностей остаётся следствием мастер-теоремы;
  транспортная теорема класса T — отдельным результатом приложения G.
- `sections/appendix/app_0_full_theory.tex` исключён из сборки как дублирующая
  версия теории, но сохранён на диске. Основные формулировки каноничны
  в разделах 3--4; перенесённые технические утверждения и доказательства организованы в восемь
  приложений A--H. Каждое оставшееся формальное утверждение приложения
  непосредственно упомянуто в основном тексте, а доказательства ссылаются
  на свои основные утверждения.
- Раздел 2 размещается на странице 3, раздел 3 — 4--5, раздел 4 — 5--7.
  Основной текст заканчивается примерно в середине страницы 7, затем
  начинается литература. Приложения начинаются на странице 10; полный PDF
  содержит 29 страниц. В девятистраничном бюджете остаётся более двух
  страниц для экспериментов и заключительной части. После сокращения глав
  3--4 отдельно переработан обзор на странице 3; остальные страницы PDF
  сохранили текст. Введение, аннотация, главы 3--4 и параметры шаблона
  этой правкой обзора не затронуты. Сохранены параллельные
  уточнения доминирования масштабной производной и обозначений точных полей.
- `paper/` и `output/` исключены из Git намеренно.

## Теория

- Основной вероятностный механизм: класс K калиброванных предельных каналов.
- Отдельный транспортный механизм: класс T калиброванных вырождающихся
  транспортов. Он не является автоматическим следствием теоремы для класса K.
- Модельные интерфейсы: диффузия, независимое аффинное FM, обобщённый
  броуновский мост Шрёдингера, масштабно-обусловленное семейство NF и отдельный
  калиброванный сингулярный NF/CNF.
- Подробные допустимые и недопустимые утверждения перечислены в
  `THEORY_CONTRACT.md`.

## Эксперименты

- 2026-09-10: запрошено расширение known-LID pointwise Kneedle без retraining.
  Live audit всех209 known-LID cells подтвердил22 масштаба lambda0.210656–3.229097;
  103/165 canonical supervised choices находятся вне этого диапазона.
  Validation/test содержат только узкие кривые (418 массивов), широкие
  supervised curves сохранены лишь для train-selection. Финальные checkpoints
  удалены после eval; среди файлов прежних v2 попыток совпадающих SHA не найдено.
  Полный eval-only rerun заблокирован до предоставления backup весов либо
  отдельного разрешения восстановить165 canonical +44 generated known-LID
  моделей обучением. E1/E5 уже используют wide reference Kneedle, не затронуты.
  Новых jobs/изменений remote artifacts нет. Audit receipt:
  `artifacts/launches/kneedle_range_audit_20260910.json`.

- Актуальная continuation — **CloudBot 9344**, `lerobot-research-lid-v2-r09-resume`,
  normal 8gpu, старт 2026-09-08 12:51:57 UTC на `bot-8h100-12-0`.
  Commit `6d041365b9bc27de5054293be6c1318766ddf911`, **648 tests passed**.
  Identity `64ae8a405a42c248d14cab5bb1e2fd2b19e2e4c6ca5fb774c4acff50a5732c62`,
  source `c366332d77ec710f6746725bdc18cc9722db57b7a5c7322e0276236302a96289`.
  Job root `.../runs/lerobot_research_lid_v2/resume-20260908__6d04136`,
  campaign suffix `campaign/lid-global-e1-e8-canonical-plus-generated-e3-e4-vp-ve-all-models-v2__64ae8a405a42c248d14c`.
  **Completed 2026-09-08 16:50:03 UTC, exit 0. Все 429/429 cells**:
  147 immutable reused +282 новых; каждая из11 моделей имеет39/39.
  Full remote `validate_global_campaign(..., verify_inputs=True)` PASS
  в16:57:02 UTC: свежие source/input bindings, pointwise metrics,
  cell/reference/FM validators, canary/lineage, ledger и все aggregates/CSV.
  Дополнительный обход:0 checkpoints/progress files и0 incomplete directories.
  Скопирован и побитово проверен один compact bundle (5.30MB +README):
  `artifacts/results/lid_benchmark_v2/64ae8a405a42c248d14c/`.
  В нём2368 строк:1368 known-LID,650 E1,350 E5;2080 canonical и288 generated.
  CSV/manifest joins дают429 уникальных cell IDs,11×39; все заполненные
  числовые поля конечны. 94 строки scale_unresolved относятся к20 cells:
  метрики сохранены. Selection_failed rows=0; pointwise Kneedle содержит
  ambient fallback, его no-knee counts не подменены нулями.
  Полнота не означает хорошее качество: canonical primary test raw mean MAE
  VP равна3.372 при supervised scale selection и128.971 при pointwise Kneedle;
  E1 и часть E5 outcomes остаются слабыми. Протоколы не смешиваются.
  Все11 Comet connections disabled, experiment keys отсутствуют; provenance
  доступна через cell IDs и manifests, не через выдуманные Comet run IDs.
  README bundle содержит описание каждого файла, краткую11-model сводку
  и границы проверки. Исторический390-cell v1/NF bundle не изменён.
  Полные pins/пути/предыдущая попытка — `artifacts/launches/lid_v2_resume_20260908.json`.
  Монитор удалён после доставки проверенного bundle; новых запусков нет.

- Предыдущая continuation — **CloudBot 9326**, `lerobot-research-lid-v2-r08-resume`,
  normal 8gpu, стартовала 2026-09-08 12:29:10 UTC на `bot-8h100-12-0`.
  Commit `03eb31acaa2a9dd24fc97fb87404d4e5eace6405`, **644 tests passed**.
  Campaign identity `1274e2bc716494a39ace56d8d89a80ab60eb855abd523c911ce15e615f7de74d`,
  source `4fc0d630de4da953e1ae0b76cdc1d379e9ccc3409d3edfccf498645bd58ea694`.
  Root: `/mnt/virtual_ai0001071-01239_SR006-nfs2/afedorov/runs/lerobot_research_lid_v2/resume-20260908__03eb31a/campaign/lid-global-e1-e8-canonical-plus-generated-e3-e4-vp-ve-all-models-v2__1274e2bc716494a39ace`.
  До submit свежие inputs и все 147 исходных sealed cells проверены, затем
  выполнено побитовое копирование и повторная проверка 24 canary cells.
  GPU replay исходного failed checkpoint прошёл. Task FAILED в 12:31:56 UTC
  до обучения: preflight ошибочно считал exit 0 уже отчитавшегося worker
  аварией при пустой очереди отчётов. Monitor удалён; готовится исправленная
  continuation. 147 copied cells сохранены, новых обучений в task 9326 нет.
  Локальный паспорт: `artifacts/launches/lid_v2_resume_20260908.json`.
  До terminal status следующей job нельзя изменять её pinned checkout.

- Актуальный исход task **9201**: failed 2026-09-07 15:37:30 UTC (18:37 МСК),
  147/429 sealed cells, 24 interrupted incomplete directories. Canary v5 прошёл,
  но direct_rectified_flow / e3_gaussian_pca / coefficients остановил DAG в
  дополнительном endpoint ratio: математически нулевой знаменатель при
  D=30, d=20, lambda=0.5. Это не quality gate и не свидетельство NaN обучения.
  Полного campaign.json / aggregate / unified CSV нет; прежний monitor отсутствует.
  По запросу автора готовится diagnostic-maintenance continuation: новая source
  identity, неизменные копии 147 проверенных cells и 282 новых выполнения.
  Старые run IDs/source hashes сохраняются; неполные 24 cells не считаются
  результатами и не импортируются. FM diagnostic v3 допускает undefined только
  в двух ratio arrays, с пересчитываемой маской и явным количеством в JSON;
  predictions/metrics остаются finite-only. Ниже запись о старте 9201 — история.

- Новая job **9201**, `lerobot-research-lid-v2-r07-outcomes`, принята normal
  8gpu, стартовала 2026-09-07 12:43:25 UTC на `bot-8h100-10-0`, commit
  `03c12b04561575fbef5c89c6876138bf7d7c2e44`. Canary v5; **587 tests passed**.
  Campaign identity `92fe3436909e3bf28427c380fdaa463e7f652c9d884f8ffe28cdc02cd11ff2b7`.
  Task root `/mnt/virtual_ai0001071-01239_SR006-nfs2/afedorov/runs/lerobot_research_lid_v2/outcomes-20260907__03c12b0`.
  Input inventory SHA совпадает с прошлой попыткой; изменены policy/source/config,
  не данные. Arrows contact sheet повторно просмотрен агентом. В 12:46 UTC
  подтверждены все 24 training workers, 2000–8000 шагов и 99–100% utilization
  на каждой из восьми H100. Ошибок в просмотренном логе нет; canary/full
  completion ещё не подтверждены. Активен `lid-benchmark-results-monitor`.

- Решение автора 2026-09-07 после task 9196: снять все блокирующие требования
  к качеству learned-моделей. Canary v5 сохраняет loss, reconstruction, MAE,
  NF/Gaussian и trace agreement как diagnostics на любой размерности, включая
  coefficients. Utilization и ETA также не являются veto. Source/input SHA,
  finite outputs, split isolation, coverage и artifact integrity остаются
  обязательными. Task 9196 завершился failed в 11:58:34 UTC: NF выиграла у
  Gaussian только 1/5 bins против старого порога 50%, хотя достигла 128000
  шагов. Ниже запись о pending 9196 — история отправки, не текущий статус.
  Новая попытка готовится в launch worktree `/tmp/lid-vp-rerun.vx9crj`;
  бюджет 128000 × 256 и матрица 429 cells не меняются.

- Актуальная попытка после разрешения увеличить бюджет: CloudBot `9196`
  (`lerobot-research-lid-v2-r06-128k`), normal `8gpu`, создана 2026-09-07;
  при отправке pending, позиция 1. Source commit
  `6ebc9845a4dba22ce5a2f7bc8d9fa41fa8a9379d`, campaign identity
  `07c79a6448974971efafbfa2afe5e694814ba1bc5e0703fe223b6cd19ee45861`.
  Canary v4 и весь production используют 128000 шагов × batch 256; plateau
  теперь diagnostic и не блокирует продолжающееся улучшение. Оценивается
  лучший native-loss checkpoint на train-selection; конечный бюджет не
  доказывает сходимость. Полный test suite: 579 passed. Root:
  `/mnt/virtual_ai0001071-01239_SR006-nfs2/afedorov/runs/lerobot_research_lid_v2/extended128k-20260907__6ebc984`.
  Ниже сохранена история предшествующих 32000-step попыток, а не их текущий
  running status. Новый полный bundle пока отсутствует.
- Реализован Hydra-ориентированный контур данных, oracle/readout-слоёв,
  learned-bundle контрактов, provenance и тестов.
- Историческая v1 campaign остаётся неизменяемым результатом: runner
  `experiments.global_parallel` выполнил 10 model variants × 39 cells = 390
  физических обучений. Её sealed artifacts и локальный consolidated bundle не
  перезаписываются и не считаются результатом v2. Описание последовательного
  запуска на одной A100 в корневом README является legacy-путём.
- Новый benchmark rerun v2 запускается отдельно из ветки
  `afedorov/vp-benchmark-rerun-v2` на commit
  `6e03ed8c41909b0009dac879747a8819c08bb474`. Runner
  `experiments.global_parallel_v2` содержит 11 model variants × 39 cells = 429
  физических обучений. Предшествующий CloudBot task `9138` называется
  `lerobot-research-lid-v2-r05`; campaign identity —
  `c259d9cd484beb01d280cf867db7d61fc692f9d8014c03ee2d3b4d3062401158`.
  Task стартовал `2026-09-06T22:42:54Z` на `bot-8h100-6-0`, но завершился
  `failed / exited_nonzero` в `2026-09-06T22:49:43Z`. VE/Arrows не прошёл
  native-loss plateau gate: tail improvement 0.2558695 при максимуме 0.10,
  хотя loss снизился с 4487.822 до 11.625. Сохранены 13 final cell directories
  с summary и 11 incomplete directories; ledger пуст, canary PASS, campaign
  manifest, aggregate и unified CSV отсутствуют. Full DAG не начался;
  новых валидированных benchmark-результатов пока нет. При проверке активных
  задач пользователя не было; приостановленный monitor удалён как завершивший
  назначение. Task `9093` упал до
  обучения из-за вложенной Hydra composition. Task `9098` начал canary, но
  остановился на ошибочной VE reconstruction-проверке, которая трактовала
  выход обученного `x0`-denoiser как score. Task `9113` запустил 24-worker
  canary, но после 12 final и 12 incomplete directories упал из-за
  floating-point endpoint `64.00000000000001`, вышедшего на один ULP за
  строгий affine-FM support. Commit `4572b003` канонизирует только созданные
  runner'ом roundoff-отклонения границ и не меняет научную сетку или бюджет.
  Task `9131` затем прошёл reconstruction checks, но выявил слабое качество
  VP/Arrows: pointwise MAE 23.43 и H16/H64 disagreement 2.54. Это реальный
  benchmark outcome, а не нарушение целостности запуска. Commit `6e03ed8c`
  оставляет эти high-D показатели обязательными diagnostics, но отделяет их
  от блокирующих low-D integrity checks; одновременно исправлены истинный
  per-sample shared prefix, выбор trace на production lambda и независимая
  проверка retained summary/quality artifacts.
  Ни один superseded task root не является источником результатов.
- V2 использует 24 logical workers на восьми физических H100: три независимых
  cell-процесса на устройство, каждый со своим неизменным batch 256 и бюджетом
  32000 шагов. В canary входят восемь quality cells —
  `{VP, VE, posterior log-noise FM, scale-conditioned NF}` ×
  `{Uniform-PCA/coefficients, Arrows/dataset}` — и 16 полноценных companion
  cells на D=30/256/784/1024. Все 24 входят в матрицу 429 и после PASS
  валидируются и переиспользуются full DAG, оставляя 405 обучений. Known-LID
  публикует supervised bounded
  selection на disjoint train-selection и pointwise FLIPD Kneedle без
  LID-targets. E1/E5 используют reference-mean Kneedle; отсутствие допустимого
  knee запечатывается как `selection_failed` без surrogate-метрик. FM
  exact+oracle diagnostics ограничены D ≤ 64; для D > 64 используется
  H16/H64 shared-prefix diagnostic без exact/oracle claim.
- Remote task root:
  `/mnt/virtual_ai0001071-01239_SR006-nfs2/afedorov/runs/lerobot_research_lid_v2/overnight-20260907__6e03ed8c4190`.
  Статус `completed` допустим только после строгой проверки всех 429 sealed
  cells, campaign manifest, input inventory, aggregates и единой CSV-таблицы.
- Population/oracle результаты и learned-model результаты являются разными
  уровнями доказательств и не объединяются в одну таблицу.
- Локально присутствует игнорируемый Git консолидированный bundle
  `artifacts/results/researh_l1_pretrain_ultra/a80ddd279bb32d666649/`:
  390 исходных обучений (10 model variants × 39 cells), 78 NF-follow-up
  обучений, 13 логических result variants и 2106 строк;
  заявленная identity bundle —
  `2ebedf14e7f06d8fb84b8dba030ce5817a41d2fc49ecec3ba1cb5f89a4110441`.
  Он analysis-ready, но не переносим в чистый checkout и не содержит тяжёлых
  checkpoint/pointwise артефактов. Перед цитированием чисел агент обязан
  проверить `campaign.json`, закреплённые SHA и ограничения локального
  `README.md`, а не полагаться на пересказ.
- Подтверждённая картина: diffusion и Schrödinger bridge образуют сильнейшую
  vanilla-группу; FM показывает trade-off posterior для known-LID/E5 против
  direct для E1; NF OLS5 резко улучшает known-LID, но ухудшает E5 и не является
  универсальным winner. Полный состав, числа и ограничения зафиксированы в
  `docs/EXPERIMENT_RESULTS.md`; NF-протокол — в `docs/NF_CAMPAIGN.md`.
- Headline использует 35 canonical cells отдельно от generated E3/E4 extension.
  `390`, `78`, `13` и `2106` обозначают разные уровни учёта и не должны
  употребляться как взаимозаменяемые количества экспериментов.
- Конфигурация или adapter сами по себе не доказывают наличие обученного
  результата. Шаблонные registries остаются fail-closed; локальный bundle
  нельзя подменять отсутствующим переносимым evidence.
- Канонические требования находятся в `docs/EXPERIMENT_PROTOCOL.md` и
  `docs/MODEL_MATRIX.md`; выполненная study — в
  `docs/EXPERIMENT_RESULTS.md`; команды и различие legacy/production путей —
  в `docs/agent/RUNBOOK.md`.

## Ближайший редакционный порядок

1. Согласовать таблицы и формулировки экспериментальной секции с
   `docs/EXPERIMENT_RESULTS.md`, не смешивая population/oracle и learned
   evidence.
2. Подключить экспериментальную главу только после проверки каждого числа по
   campaign manifest и bundle identity.
3. Затем переработать ограничения и заключение, сохранив границы
   `THEORY_CONTRACT.md`.

После изменения фазы основной агент обновляет этот файл, а не оставляет новое
состояние только в истории чата.
