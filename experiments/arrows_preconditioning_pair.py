"""Native v2 Arrows repair with explicitly recorded architecture/loss ablations.

Uses the upstream training loop with opt-in field and equivalent-loss branches.
Instrumentation observes gradients and states and retains resumable snapshots
every 8000 steps; it consumes no RNG.
"""
from __future__ import annotations

import yaml
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from models import training


def digest_tensor(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('original', 'preconditioned', 'preconditioned_no_input_skip', 'native_no_input_skip', 'native_image', 'native_pixel_image', 'native_tail_image'), required=True)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--image-width', type=int, default=32)
    parser.add_argument('--image-fp32', action='store_true')
    parser.add_argument('--antithetic', action='store_true')
    parser.add_argument('--resume-progress', type=Path)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    contract = yaml.safe_load(args.contract.read_text())
    family = contract['model']['family']
    config = training.TrainingConfig.from_mapping(contract['model']['training'])
    if args.arm == 'preconditioned':
        config = replace(config, posterior_preconditioning='unit_rms_gaussian_v1')
    if args.arm == 'preconditioned_no_input_skip':
        config = replace(config, posterior_preconditioning='unit_rms_gaussian_no_input_skip_v1')
    if args.arm == 'native_no_input_skip':
        config = replace(config, native_preconditioning='unit_rms_gaussian_no_input_skip_v1')
    if args.arm in {'native_image','native_pixel_image','native_tail_image'}:
        config = replace(config,native_preconditioning={
                         'native_image':'unit_rms_gaussian_image_v1',
                         'native_pixel_image':'unit_rms_pixel_mixture_image_v1',
                         'native_tail_image':'unit_rms_gaussian_tail_image_v1'}[args.arm],
                         field_hidden_sizes=None,image_shape=(32,32,3),image_width=args.image_width,
                         image_training_bf16=True)
    if args.image_fp32:
        if config.native_preconditioning not in training.IMAGE_PRECONDITIONING_MODES:
            raise ValueError('--image-fp32 requires an image field')
        config = replace(config,image_training_bf16=False)
    if args.steps is not None:
        config = replace(config,steps=args.steps)
    if args.antithetic:
        config = replace(config,native_noise_pairing='antithetic_v1')
    if args.smoke:
        config = replace(config, steps=20, validation_interval_steps=10)
    start = time.time()
    source = Path(__file__).resolve().parents[1]
    source_names = ['models/training.py', 'models/vp_baseline.py', 'models/affine_flow.py',
                    'models/neural_fields.py', 'models/gaussian_fields.py', 'models/image_gaussian_fields.py', 'models/pixel_mixture.py',
                    'experiments/arrows_preconditioning_pair.py']
    for name in source_names:
        target = args.output/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((source/name).read_bytes())
    provenance = {
        'arm': args.arm, 'variant':contract.get('variant_id'), 'family':family,
        'start_unix': start, 'config': config.to_dict(),
        'source_base_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=source, text=True).strip(),
        'source_sha256': {n: hashlib.sha256((source/n).read_bytes()).hexdigest() for n in source_names},
        'torch': str(torch.__version__), 'numpy': np.__version__,
        'gpu': torch.cuda.get_device_name(), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'tf32_matmul': torch.backends.cuda.matmul.allow_tf32, 'tf32_cudnn': torch.backends.cudnn.allow_tf32,
        'precision': 'bf16 residual during training; fp32 validation and inference' if config.image_training_bf16 else 'float32',
        'labels_used_for_loss': False,
        'checkpoint_rule': f'minimum native validation loss, every {config.validation_interval_steps} steps',
        'data_manifest': json.loads((args.data/'manifest.json').read_text()),
    }
    raw = np.load(args.data/'benchmarks/e2_arrows/train/dataset.npy', mmap_mode='r')
    fit_indices = np.load(args.data/'fit_indices.npy')
    holdout_indices = np.load(args.data/'holdout_indices.npy')
    fit, holdout = raw[fit_indices], raw[holdout_indices]
    if args.smoke:
        fit, holdout = fit[:512], holdout[:64]
    write_json(args.output/'manifest.json', provenance)
    if args.resume_progress:
        # A declared compute-precision intervention from an immutable snapshot.
        # Keep weights, AdamW state and every RNG byte. Only the remaining
        # constant-schedule budget, residual precision and noise pairing may change.
        from models.image_gaussian_fields import ImageFieldConfig
        parent = torch.load(args.resume_progress,map_location='cpu',weights_only=True)
        parent_cfg = training.TrainingConfig.from_mapping(parent['training_config'])
        if config.native_preconditioning not in training.IMAGE_PRECONDITIONING_MODES:
            raise ValueError('precision continuation is restricted to image fields')
        expected = replace(parent_cfg,steps=config.steps,image_training_bf16=config.image_training_bf16,
                           native_noise_pairing=config.native_noise_pairing)
        if expected != config or parent['scheduler_state']['schedule'] != 'constant_v1':
            raise ValueError('continuation may change only precision, noise pairing and constant-schedule total steps')
        if parent['global_step'] >= config.steps:
            raise ValueError('continuation requires additional optimizer steps')
        provenance['parent_progress_sha256'] = training._checkpoint_sha256(args.resume_progress)
        provenance['parent_global_step'] = parent['global_step']
        provenance['parent_state_sha256'] = training._state_dict_sha256(parent['model_state'])
        provenance['continuation_changes'] = {'steps':[parent_cfg.steps,config.steps],
            'image_training_bf16':[parent_cfg.image_training_bf16,config.image_training_bf16],
            'native_noise_pairing':[parent_cfg.native_noise_pairing,config.native_noise_pairing]}
        provenance['optimizer_and_all_rng_retained'] = True
        parent['training_config'] = config.to_dict()
        parent['architecture'] = replace(ImageFieldConfig.from_mapping(parent['architecture']),
            training_bf16=config.image_training_bf16).to_dict()
        parent['scheduler_state']['total_steps'] = config.steps
        training._atomic_torch_save(args.output/'progress.pt',parent)
        write_json(args.output/'manifest.json',provenance)

    # Observation only: call the original operations with the original arguments.
    original_objective = training._objective
    original_clip = torch.nn.utils.clip_grad_norm_
    original_save = training._atomic_torch_save
    observed = {'model': None, 'norms': []}

    def objective(family, model, batch, cfg, generator):
        if observed['model'] is None:
            observed['model'] = model
            provenance['initial_state_sha256'] = {k:digest_tensor(v) for k,v in model.state_dict().items()}
            provenance['parameter_count'] = sum(p.numel() for p in model.parameters())
            write_json(args.output/'manifest.json', provenance)
        return original_objective(family, model, batch, cfg, generator)

    def clip(*values, **kwargs):
        norm = original_clip(*values, **kwargs)
        observed['norms'].append(float(norm))
        return norm

    def save(path, payload):
        if path.name == 'progress.pt':
            step = int(payload['global_step'])
            if step % 8000 != 0 and step != config.steps:
                return
            original_save(path, payload)
            snapshot = args.output/f'progress_{step:06d}.pt'
            os.link(path, snapshot)
            return
        original_save(path, payload)

    def callback(metric):
        norms = np.asarray(observed['norms'], dtype=np.float64)
        model = observed['model']
        trace_a = float(model.decoder[-1].weight[:, :3072].diagonal().sum()) if hasattr(model,'decoder') else None
        record = {**metric, 'elapsed_seconds': time.time()-start,
                  'direct_matrix_trace': trace_a,
                  'gradient_norm_mean': float(norms.mean()),
                  'gradient_norm_max': float(norms.max()),
                  'gradient_clipped_fraction': float((norms>config.gradient_clip_norm).mean())}
        observed['norms'].clear()
        with (args.output/'history.jsonl').open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False)+'\n')
        print(json.dumps(record, allow_nan=False), flush=True)

    training._objective = objective
    torch.nn.utils.clip_grad_norm_ = clip
    training._atomic_torch_save = save
    try:
        result = training.train_model(
            family, fit, holdout, config, args.output/'model.pt',
            log_callback=callback, progress_checkpoint_path=args.output/'progress.pt')
    finally:
        training._objective = original_objective
        torch.nn.utils.clip_grad_norm_ = original_clip
        training._atomic_torch_save = original_save
    loaded = training.load_checkpoint(result.checkpoint_path, device='cuda')
    for name, value in result.model.state_dict().items():
        torch.testing.assert_close(value, loaded.model.state_dict()[name], rtol=0, atol=0)
    if not args.smoke:
        assert result.normalization_scale == provenance['data_manifest']['normalization_rms']
        np.testing.assert_array_equal(result.normalization_mean.numpy().reshape(32,32,3),
                                      np.load(args.data/'mean.npy'))
    write_json(args.output/'complete.json', {
        'status': 'complete', 'metrics': result.metrics, 'weights_metadata': dict(result.weights_metadata),
        'checkpoint_sha256': result.checkpoint_sha256,
        'normalization_rms': result.normalization_scale,
        'normalization_sha256': result.preprocessing_sha256,
        'duration_seconds': time.time()-start, 'reload_state_exact': True,
    })


if __name__ == '__main__':
    main()
