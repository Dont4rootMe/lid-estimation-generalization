"""Execute the shared repair recipe for one explicitly named benchmark cell.

This is a separate, reproducible entry point. Legacy campaign defaults and
historical results remain source faithful. It never chooses a model using test.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def commands(args):
    root=Path(__file__).resolve().parents[1]
    contracts=json.loads((root/'configs/lambda_repair/native_contracts.json').read_text())
    variants={c['variant_id'] for c in contracts['model_contracts']}
    if args.variant not in variants:
        raise ValueError('unknown native variant: '+args.variant)
    if args.steps<8000 or args.steps%500:
        raise ValueError('use a budget >=8000 divisible by the500-step validation interval')
    recipe=json.loads((root/'configs/lambda_repair/shared_recipe.json').read_text())
    nf=args.variant=='scale_conditioned_nf';vp=args.variant=='vp_diffusion'
    settings=recipe['nf' if nf else 'vector']
    train=[sys.executable,'-m','experiments.lambda_repair_train','--variant',args.variant,
        '--dataset',args.dataset,'--representation',args.representation,'--steps',str(args.steps),
        '--data',str(args.data),'--output',str(args.output),'--arm',settings['field_preconditioning']]
    if nf:
        train+=['--condition-frequency',str(settings['max_condition_frequency'])]
    else:
        train+=['--backbone',settings['field_backbone'],'--noise-pairing',settings['noise_pairing'],
                '--width',str(settings['field_residual_width']),'--residual-scaling',settings['field_residual_scaling']]
    if not vp:
        train+=['--terminal-steps',str(settings['terminal_decay_steps']),'--ema-decay',str(settings['ema_decay'])]
    common=['--run',str(args.output),'--data',str(args.data)]
    if nf:
        evaluation=[[sys.executable,'-m','experiments.lambda_repair_nf_eval']+common,
                    [sys.executable,'-m','experiments.lambda_repair_nf_eval']+common+['--dtype','float64','--readout','fixed_likelihood'],
                    [sys.executable,'-m','experiments.lambda_repair_nf_heat']+common]
    else:
        evaluation=[[sys.executable,'-m','experiments.lambda_repair_eval']+common+
                    ['--batch-size','512','--probes',str(p),'--test'] for p in [16,0]]
    evaluation += [[sys.executable,'-m','experiments.lambda_repair_denoising']+common]
    return [train]+evaluation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',required=True);p.add_argument('--dataset',required=True)
    p.add_argument('--representation',choices=['dataset','coefficients'],default='dataset')
    p.add_argument('--steps',type=int,default=128000)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dry-run',action='store_true');args=p.parse_args()
    pipeline=commands(args)
    if args.dry_run:
        print(json.dumps(pipeline,indent=2));return
    if args.output.exists():raise FileExistsError('choose a fresh output directory: '+str(args.output))
    env=dict(os.environ);env.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    for command in pipeline:subprocess.run(command,env=env,check=True)
    (args.output/'shared_recipe_complete.json').write_text(json.dumps(dict(
        status='complete',commands=pipeline,
        scope='one named native cell; inspect denoising and scale status before scientific claims'),indent=2)+'\n')


if __name__=='__main__':main()
