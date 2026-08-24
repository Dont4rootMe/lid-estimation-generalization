# Audit of `DominikFilipiak/LID-Benchmarks`

## Pin and licensing

- URL: `https://github.com/DominikFilipiak/LID-Benchmarks`
- pinned commit: `2dcb8e41015f53413ff1ddd049bb006c81a5df52`
- upstream history at integration time: one commit
- contents: dataset generators and lock file only; no estimator runner, tests,
  CI, trained checkpoints or result reproduction scripts
- no `LICENSE` file at the pinned commit

По требованию структуры проекта source импортирован как обычная top-level
директория `lid_benchmarks/`, чтобы следующие benchmark generators можно было
добавлять в основном Git history. Оригинальные файлы зафиксированы в
`lid_benchmarks/UPSTREAM.yaml` по SHA-256. Поскольку software licence не
предоставлена, перед публичным распространением этой копии нужно получить
разрешение авторов.

## Canonical archive

The upstream README publishes a password-protected `benchmarks.zip` and states
that regeneration is not guaranteed to reproduce the paper datasets. Thus:

- `exact_archive` is canonical for paper-parity;
- `generated@2dcb8e4` is an audited fallback and a distinct dataset identity;
- results from the two provenance classes are never pooled.

Archive metadata recorded during audit:

- Google Drive file id: `1mGzGUVa37AjUREHx_vFoPzl1OCLjPJ1Q`;
- compressed size: `4,685,463,657` bytes;
- uncompressed payload: `6,918,816,685` bytes;
- ZIP64, 306 entries, 193 files, traditional ZipCrypto encryption;
- SHA-256: `ce0d153a1a78a3a752b29ec2e60167134b6b20c3249db2fe92f9fc1b8b8a9181`;
- password is published in upstream README.

The independently verified SHA-256 and archive layout are pinned in
`lid_benchmarks/DATA.yaml`.

`lid-benchmarks-data` validates the outer digest, every central-directory path,
encryption/type metadata and the complete extracted tree (size + CRC32). It
never overwrites an existing destination and removes only a destination it
created itself if extraction fails.

## Confirmed generator/parity defects

1. `spaghetti_pca` constructs a one-dimensional curve but writes `lid=dim`
   (default 20). In the exact archive, canonical `e8_spaghetti_pca/lid.npy`
   contains 20 in `train` and `val`, but 1 in `test`; an
   `lid-OLD_WRONG.npy` marker is also retained. The YAML registry validates the
   raw value independently in every split (`20/20/1`) before exposing the
   theoretical effective LID 1 in memory for all three splits. The generated
   fallback has its own split policy (`20/20/20 -> 1`). Neither path modifies
   an archive or generated source artifact.
2. The exact archive also retains partially corrected LID targets for Funnel
   and Crescent Moon: `e6_exp_pca` stores `1/1/2` across train/validation/test
   although the pinned generator defines LID 2, and
   `e7_crescent_moon_radius3.0` stores `2/2/3` although its generator defines
   LID 3. The registry validates these raw split contracts before exposing
   effective LID 2 and 3 in memory; the archive and its `lid-OLD_WRONG.npy`
   sidecars remain untouched. The separately identified generated fallback
   overrides the raw contracts to `2/2/2` and `3/3/3`, matching what the pinned
   generators actually write while preserving the same effective targets.
3. The published paper describes the E8 sphere benchmark as four copies of
   $S^4$ embedded through five active coordinates, which would have LID 4,
   with radii $1, 1/3, 1/9, 1/27$. The official supplemental generator and
   exact archive instead call `sphere4_pca(dim=6)`: the coefficients use six
   active coordinates and lie on $S^5$, every raw split stores LID 5, and the
   direction radii are $3, 1, 1/3, 1/9$. The canonical campaign keeps the
   scientifically correct target 5 for the actual sealed artifact and records
   the full paper-versus-artifact construction mismatch. A paper-spec variant
   must be generated and versioned as a separate dataset identity.
4. `PaddedDatasetGenerator(additional_dimension=0)` reaches
   `add_random_numbers_to_borders` where `areas_to_modify` is undefined.
5. Several e1/e5 generators set `copy_val_to_test=True`; generated validation
   and test arrays are identical, while the exact archive contains distinct
   splits. A strict split-overlap gate rejects this in confirmatory runs.
6. Arrows generation samples integer RGB values, rasterizes with clipping and
   combines overlaps using pixelwise maximum. This does not exactly implement
   a smooth six-continuous-variable-per-arrow chart everywhere. Reported
   `6 × arrows` ground truth is retained for parity but labelled
   `construction_assumption`. Overlap/quantization sensitivity requires a
   separately versioned row-level covariate artifact and is not claimed by the
   current global aggregate.
7. The top-level generator has no tests or output checksums and globally seeds
   NumPy in several functions; Torch randomness in border augmentation is not
   explicitly seeded.

Эти дефекты не исправляются скрыто в оригинальных импортированных файлах.
Adapter сначала проверяет pinned source hashes, затем запускает оригинальный
entry point в отдельном `uv run --frozen` subprocess с рабочей директорией
`lid_benchmarks/`. Он не импортирует абсолютный namespace `generators` в
основной process и не меняет `sys.path`. Дефекты отклоняются или маркируются
на dataset/readout boundary, а расширения добавляются отдельными файлами,
поэтому upstream snapshot остаётся проверяемым.
