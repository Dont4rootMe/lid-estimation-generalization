# Applied native-model repair

This patch targets upstream `6d041365b9bc27de5054293be6c1318766ddf911`.
`shared_recipe.json` is the common rerun recipe. The original eleven model
contracts are retained verbatim in `native_contracts.json`; model families keep
their own targets, time/noise sampling, optimizer, batch size and normalization.
The normalizing flow remains an invertible conditional RealNVP with a complete
likelihood determinant. Legacy entry points retain their original defaults.
Run these commands from the source checkout, with its `train` and `upstream`
dependencies installed. The proof package records the exact measured runtime.

Prepare the pinned benchmark archive using its original source split:

```bash
python -m experiments.lambda_repair_data \
  --archive /path/to/benchmarks.zip --output artifacts/repair_data
```

Run one explicitly named cell (same syntax for all eleven variants):

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.lambda_repair_run \
  --variant ve_diffusion --dataset e6_exp_pca --steps 128000 \
  --data artifacts/repair_data --output artifacts/ve_exp_fixed
```

Use `--dry-run` to print the complete command list. For NF use
`--variant scale_conditioned_nf`; for coefficient data add
`--representation coefficients`. Launch separate named cells with different
`CUDA_VISIBLE_DEVICES` values for parallel execution. The runner refuses to
overwrite an existing output. It performs real training, checkpoint reload,
source-P16 and exact-span readouts (NF: OLS5 and exact log-scale derivative),
source-train holdout scale selection, then test and fresh-noise denoising.

The fitted covariance uses only the99k optimizer-fit observations. Its rank is
the global affine rank, not a supplied intrinsic dimension. A single relative
eigenvalue rule and weak-coordinate guard apply to every cell. Original ambient
RMS units define lambda throughout; rendering or projection never silently
rescales the grid. Covariance rank, omitted variance, parameter count and source,
data and checkpoint hashes are saved. Check the omitted variance before using
the analytic normal-coordinate factorization on another dataset.

The vector correction uses Gaussian input/output scaling, a shared spectral
residual network without the unconditional raw-coordinate output skip,
antithetic noise with the original256 network examples per update, and a last8k
cosine/EMA suffix. Native VP retains its existing full-budget cosine. This is an
architecture and optimization change; it is not a parameter-count-matched
backbone comparison. Exact span traces differentiate the whole nonlinear learned
posterior and remove stochastic trace error without substituting a LID label.

NF uses smoother log-scale conditioning (`max_condition_frequency=1`) to reduce
the rapid oscillation masked by the original OLS5 stencil. OLS5 is a smoothed
readout and is reported separately from the instantaneous derivative. The heat
identity diagnostic measures whether the learned conditional density resembles
one Gaussian convolution path; exact normalization of each conditional density
alone does not ensure that property.

Weights are selected by the native holdout objective, independently of LID.
The original supervised bounded-v2 scale rule is used on1000 source-train
holdout queries before test inference. This is a benchmark-tuned track, not a
label-free deployment method. A boundary winner is explicitly unresolved.
`full_support_v1` is available as a diagnostic: globally minimizing scalar LID
MAE can favor a large-noise crossing even for a substantially improved field.
Always inspect full curves and denoising; small lambda alone proves nothing.

`empirical_posterior_v1`, `data_v1`, `gaussian_tail_v1` and width736 are isolated ablations, not hidden
defaults. The empirical target averages over the full optimizer-fit training
bank. It preserves the expected native squared-loss parameter gradient for
that finite prior, reduces conditional sampling variance, and adds substantial
float64 work. It does not supply a continuous-manifold oracle at small noise.

Focused checks:

```bash
python -m pytest -q tests/unit/test_field_preconditioning.py \
  tests/unit/test_nf_preconditioning.py tests/unit/test_noise_pairing.py \
  tests/unit/test_empirical_target.py tests/unit/test_lambda_repair_resume.py \
  tests/unit/test_lambda_repair_selection.py
```

The accompanying research artifacts retain the source-faithful paired runs,
pointwise predictions, posterior/Jacobian controls and successful and unsuccessful
ablations. Representative outlier improvements are verified; this is not a
claim of uniform small-noise convergence or a completed full benchmark rerun.
