# Arrows repair

This directory preserves the native VE and rectified-flow upstream contracts.
Use the dedicated runner below: the frozen global-v2 campaign has a different
capacity policy and must not silently relabel these image models as its MLPs.

The common image field uses a width32 U-Net (893507 parameters), normalized
native inputs, log physical-noise conditioning and an analytic Gaussian part.
`native_tail_image` additionally makes the learned posterior residual decay as
`lambda/(1+lambda^2)` at large noise. It has no unconditional dense input/output
skip. VE and RF are trained separately with their original native targets and
noise/time samplers. No LID labels enter training.

Prepare the original extracted uint8 data once, from the repository root:

```bash
PYTHONPATH=. python -m experiments.prepare_arrows_repair \
  --arrows-root data/lid_benchmarks_exact/benchmarks/e2_arrows \
  --output artifacts/arrows_repair_data
```

This reproduces the source-v2 99000/1000 fit/holdout split, records all six
data-file hashes and fit-only normalization, and links to the unchanged images.
Each output directory must be new. Python dependencies are the upstream ones,
including `kneed==0.8.6` for the campaign selector.

The completed 32000-step Gaussian-tail recipe is reproduced with:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m experiments.arrows_preconditioning_pair \
  --arm native_tail_image --steps 32000 \
  --data artifacts/arrows_repair_data \
  --contract configs/arrows_repair/ve_diffusion_upstream.yaml \
  --output artifacts/arrows_ve_tail

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m experiments.arrows_preconditioning_pair \
  --arm native_tail_image --steps 32000 \
  --data artifacts/arrows_repair_data \
  --contract configs/arrows_repair/rectified_flow_upstream.yaml \
  --output artifacts/arrows_rf_tail
```

The two commands may run sequentially on one GPU. Each uses seed0, batch256,
constant AdamW2e-4, decay1e-6 and clipping1, without EMA or warmup. BF16 is used
only inside the spatial residual during training; validation, field derivatives
and inference use float32. The lowest native validation loss (every500 steps)
chooses `model.pt`; the final weights and complete resumable progress are also
retained. `complete.json` is written after exact checkpoint reload verification.

Load raw NHWC images through the standard adapter. The noise argument is
native: VE uses `lambda`, RF uses `t=1/(1+lambda)`. For example:

```python
from models.training import load_checkpoint, predict_lid

result = load_checkpoint("artifacts/arrows_rf_tail/model.pt", device="cuda")
physical_lambda = 14.357672600024857  # heldout-selected, not a universal constant
native_t = 1 / (1 + physical_lambda)
lid = predict_lid(result, raw_images, native_t, readout="full",
                  divergence_backend="hutchinson", trace_probes=64,
                  trace_seed=0, batch_size=128)
```

Select a new run's scale on its source-train holdout before opening its test.
The repair does not cap lambda, clip estimates or fit a scalar correction.
The measured claim concerns these two trained generative families on Arrows;
it does not establish small-noise LID consistency for quantized raster images.

Additional declared ablations in the runner are `native_no_input_skip`
(Gaussian MLP), `native_image` (ordinary Gaussian U-Net) and
`native_pixel_image` (fit-only spike/Gaussian pixel prior). `--image-fp32`
switches residual training precision. `--antithetic` pairs Z and -Z while
keeping256 network examples and independent validation. `--resume-progress`
can continue an image run with an explicitly recorded constant-schedule budget,
precision or noise-pairing change, retaining parent model, optimizer and RNG.
