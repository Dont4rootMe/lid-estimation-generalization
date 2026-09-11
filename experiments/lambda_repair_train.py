"""Run a source-native fixed-budget pair and preserve its training state."""
import argparse
from dataclasses import replace
import hashlib
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import time

import numpy as np
import torch
from models import training

CONTRACTS=Path(__file__).resolve().parents[1]/'configs/lambda_repair/native_contracts.json'


def extend_constant_progress(payload,config,family):
    """Extend only a constant-LR budget; preserve parameters, Adam and all RNGs."""
    previous=training.TrainingConfig.from_mapping(payload['training_config'])
    resolved=replace(config,field_projection_rank=previous.field_projection_rank)
    if payload['family']!=training._canonical_family(family):raise ValueError('resume family differs')
    if replace(previous,steps=resolved.steps)!=resolved:raise ValueError('resume changes settings other than total steps')
    if not payload['global_step']<resolved.steps or resolved.steps<previous.steps:
        raise ValueError('resume budget must cover and extend the completed prefix')
    if resolved.steps!=previous.steps and payload['scheduler_state']['schedule']!='constant_v1':
        raise ValueError('budget extension would change the earlier learning-rate schedule')
    payload['training_config']=resolved.to_dict()
    payload['scheduler_state']['total_steps']=resolved.steps
    return payload


def replan_constant_progress(payload, config, family):
    """Close a constant-LR prefix at an explicit shorter budget, optionally
    adding a future terminal suffix. No already executed update may change.
    """
    previous=training.TrainingConfig.from_mapping(payload['training_config'])
    resolved=replace(config,field_projection_rank=previous.field_projection_rank)
    if payload['family']!=training._canonical_family(family):raise ValueError('replan family differs')
    allowed=replace(previous,steps=resolved.steps,terminal_decay_steps=resolved.terminal_decay_steps,
        terminal_learning_rate_ratio=resolved.terminal_learning_rate_ratio,
        ema_decay=resolved.ema_decay,ema_start_step=resolved.ema_start_step)
    if allowed!=resolved:raise ValueError('replan alters the completed training prefix')
    completed=payload['global_step']
    old_tail_start=previous.steps-previous.terminal_decay_steps
    schedule=payload['scheduler_state']['schedule']
    if schedule not in {'constant_v1','terminal_cosine_v1'} or (previous.terminal_decay_steps and completed>old_tail_start):
        raise ValueError('replan requires an unchanged constant-LR prefix')
    if payload.get('ema_state') is not None or (previous.ema_decay is not None and completed>previous.ema_start_step):
        raise ValueError('replan cannot discard an executed EMA trajectory')
    if completed>=resolved.steps:raise ValueError('replan must leave a nonempty training suffix')
    if resolved.terminal_decay_steps and completed>resolved.steps-resolved.terminal_decay_steps:
        raise ValueError('terminal decay would alter an already executed update')
    if resolved.ema_decay is not None and completed>resolved.ema_start_step:
        raise ValueError('EMA would need an unavailable past trajectory')
    payload['training_config']=resolved.to_dict()
    payload['scheduler_state']['total_steps']=resolved.steps
    payload['scheduler_state'].update(schedule='terminal_cosine_v1' if resolved.terminal_decay_steps else 'constant_v1',
        terminal_decay_steps=resolved.terminal_decay_steps,terminal_learning_rate_ratio=resolved.terminal_learning_rate_ratio)
    if resolved.ema_decay is not None:payload['ema_state']=None
    return payload


def prepare_terminal_progress(payload, config, family):
    """Replay a common terminal suffix within the original total update budget.

    Parameters, Adam moments, data draws and noise draws at the branching point
    are untouched. EMA is initialized there, never reconstructed from future
    checkpoints. No LID labels are involved in weight selection.
    """
    previous = training.TrainingConfig.from_mapping(payload['training_config'])
    resolved = replace(config, field_projection_rank=previous.field_projection_rank)
    if payload['family'] != training._canonical_family(family):
        raise ValueError('terminal replay family differs')
    if replace(previous, terminal_decay_steps=resolved.terminal_decay_steps,
               terminal_learning_rate_ratio=resolved.terminal_learning_rate_ratio,
               ema_decay=resolved.ema_decay, ema_start_step=resolved.ema_start_step) != resolved:
        raise ValueError('terminal replay changes settings outside its declared intervention')
    start = resolved.steps - resolved.terminal_decay_steps
    if not resolved.terminal_decay_steps or payload['global_step'] != start:
        raise ValueError('terminal replay must branch exactly before the decay suffix')
    if previous.terminal_decay_steps or previous.ema_decay is not None or payload['scheduler_state']['schedule'] != 'constant_v1':
        raise ValueError('terminal replay requires the unchanged constant-LR prefix')
    if resolved.ema_decay is not None and resolved.ema_start_step != start:
        raise ValueError('EMA must start at the replay boundary')
    payload['training_config'] = resolved.to_dict()
    payload['scheduler_state'].update(schedule='terminal_cosine_v1',
        terminal_decay_steps=resolved.terminal_decay_steps,
        terminal_learning_rate_ratio=resolved.terminal_learning_rate_ratio)
    if resolved.ema_decay is not None:
        payload['ema_state'] = None
    return payload


def prepare_empirical_target_progress(payload,config,family):
    previous=training.TrainingConfig.from_mapping(payload['training_config'])
    if previous.training_target!='sample_v1' or config.training_target!='empirical_posterior_v1':
        raise ValueError('empirical target replay must branch from sampled targets')
    if config.training_target_start_step!=payload['global_step']:
        raise ValueError('new target estimator must begin exactly at the branch')
    ordinary=replace(config,training_target=previous.training_target,
                     training_target_start_step=previous.training_target_start_step)
    ordinary=replace(ordinary,field_projection_rank=previous.field_projection_rank)
    if ordinary==previous and payload['global_step']==ordinary.steps-ordinary.terminal_decay_steps:
        # A retained state exactly before an already declared terminal suffix
        # has executed the same constant-LR prefix and has no EMA trajectory.
        if payload.get('ema_state') is not None:raise ValueError('EMA already started before target branch')
    else:
        payload=replan_constant_progress(payload,ordinary,family)
    resolved=replace(training.TrainingConfig.from_mapping(payload['training_config']),
        training_target=config.training_target,training_target_start_step=config.training_target_start_step)
    payload['training_config']=resolved.to_dict()
    payload['model_contract']=training._model_contract(training._canonical_family(family),resolved)
    return payload


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',required=True)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--representation',default='dataset',choices=['dataset','coefficients'])
    parser.add_argument('--arm',default='original')
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=32000)
    parser.add_argument('--backbone',default='bottleneck_v1')
    parser.add_argument('--residual-scaling',default='noise_v1')
    parser.add_argument('--width',type=int,default=512)
    parser.add_argument('--condition-frequency',type=float)
    parser.add_argument('--noise-pairing',default='iid')
    parser.add_argument('--training-target',default='sample_v1')
    parser.add_argument('--target-start-step',type=int,default=0)
    parser.add_argument('--empirical-target-from',type=Path)
    parser.add_argument('--resume-from',type=Path)
    parser.add_argument('--terminal-from',type=Path)
    parser.add_argument('--replan-from',type=Path)
    parser.add_argument('--terminal-steps',type=int,default=0)
    parser.add_argument('--ema-decay',type=float)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    campaign=json.loads(CONTRACTS.read_text())
    contract=next(x for x in campaign['model_contracts'] if x['variant_id']==args.variant)
    config=training.TrainingConfig.from_mapping(contract['model']['training'])
    config=replace(config,steps=args.steps,validation_interval_steps=min(500,args.steps),num_workers=0)
    config=replace(config,noise_pairing=args.noise_pairing)
    config=replace(config,training_target=args.training_target,training_target_start_step=args.target_start_step)
    config=replace(config,terminal_decay_steps=args.terminal_steps,ema_decay=args.ema_decay,
                   ema_start_step=args.steps-args.terminal_steps if args.ema_decay is not None else 0)
    if args.arm!='original': config=replace(config,field_preconditioning=args.arm,field_backbone=args.backbone,
        field_residual_scaling=args.residual_scaling,field_residual_width=args.width)
    folder=args.data/args.dataset
    raw=np.load(folder/f'train_{args.representation}.npy',mmap_mode='r')
    if contract['model']['family']=='scale_conditioned_nf':
        from experiments.global_campaign_v2 import select_nf_width
        config=replace(config,hidden_dim=select_nf_width(raw.shape[1])[0])
        if args.condition_frequency is not None:config=replace(config,max_condition_frequency=args.condition_frequency)
    elif args.condition_frequency is not None:
        raise ValueError('condition-frequency intervention is NF-only')
    fit=raw[np.load(folder/'fit_indices.npy')]
    holdout=raw[np.load(folder/'holdout_indices.npy')]
    start=time.time()
    root=Path(__file__).resolve().parents[1]
    sources=['models/training.py','models/vp_baseline.py','models/affine_flow.py','models/schrodinger_bridge.py','experiments/lambda_repair_train.py']
    for name in ['models/preconditioned_field.py','models/preconditioned_nf.py','models/normalizing_flow.py','models/spectral_residual_field.py','models/noise_pairing.py']:
        if (root/name).exists():sources.append(name)
    sources.append('models/empirical_target.py')
    for name in sources:
        dest=args.output/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((root/name).read_bytes())
    manifest=dict(variant=args.variant,dataset=args.dataset,representation=args.representation,arm=args.arm,
                  config=config.to_dict(),base_revision='6d041365b9bc27de5054293be6c1318766ddf911',start_unix=start,
                  source_sha256={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in sources},
                  data_manifest=json.loads((folder/'manifest.json').read_text()),tf32=False,
                  gpu=torch.cuda.get_device_name(),labels_in_training_loss=False)
    if sum(x is not None for x in [args.resume_from,args.terminal_from,args.replan_from,args.empirical_target_from])>1:
        raise ValueError('choose exactly one resume/replay/replan mode')
    parent = args.empirical_target_from or args.terminal_from or args.resume_from or args.replan_from
    if parent is not None:
        payload=torch.load(parent,map_location='cpu',weights_only=False)
        manifest['resume']=dict(parent_progress_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
            completed_prefix=payload['global_step'],previous_budget=payload['training_config']['steps'],
            contract=('same native expected gradient, optimizer-fit empirical posterior target; common terminal cosine/EMA'
                if args.empirical_target_from else 'explicit revised total budget; completed constant-LR prefix unchanged; optional future cosine/EMA'
                if args.replan_from else 'same total update budget, weights, Adam and RNG at branch; terminal cosine and optional EMA'
                if args.terminal_from else 'same complete optimizer/sampler/RNG state; only a constant-LR total budget may extend'))
        manifest['resume']['parent_run'] = parent.parent.name
        prepare=(prepare_empirical_target_progress if args.empirical_target_from else
                 replan_constant_progress if args.replan_from else prepare_terminal_progress if args.terminal_from else extend_constant_progress)
        payload=prepare(payload,config,contract['model']['family'])
        torch.save(payload,args.output/'progress.pt')
        os.link(args.output/'progress.pt',args.output/f"progress_{payload['global_step']:06d}.pt")
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    original_save=training._atomic_torch_save
    def save(path,payload):
        if path.name=='progress.pt':
            step=int(payload['global_step'])
            if step%8000!=0 and step!=args.steps:return
            original_save(path,payload)
            os.link(path,args.output/f'progress_{step:06d}.pt')
        else: original_save(path,payload)
    def callback(metric):
        record={**metric,'elapsed_seconds':time.time()-start}
        with (args.output/'history.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    training._atomic_torch_save=save
    result=training.train_model(contract['model']['family'],fit,holdout,config,args.output/'model.pt',log_callback=callback,progress_checkpoint_path=args.output/'progress.pt')
    training._atomic_torch_save=original_save
    loaded=training.load_checkpoint(result.checkpoint_path,device='cuda')
    for name,value in result.model.state_dict().items():torch.testing.assert_close(value,loaded.model.state_dict()[name],rtol=0,atol=0)
    (args.output/'complete.json').write_text(json.dumps(dict(status='complete',metrics=result.metrics,
        resolved_config=result.config.to_dict(),
        parameter_count=sum(p.numel() for p in result.model.parameters()),
        omitted_variance_fraction=(float(result.model.omitted_variance_fraction.item())
            if hasattr(result.model,'omitted_variance_fraction') else None),
        checkpoint_sha256=result.checkpoint_sha256,normalization_rms=result.normalization_scale,
        normalization_sha256=result.preprocessing_sha256,duration_seconds=time.time()-start,reload_exact=True),indent=2)+'\n')


if __name__=='__main__':main()
