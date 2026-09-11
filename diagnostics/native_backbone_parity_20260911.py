"""Audit actual shared backbones; optional eight-update train/reload checks.

No benchmark quality claim is produced. No canonical test array is loaded.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import torch

from experiments.fair_campaign import inventory,dataset_spec,write_json
from experiments.fair_protocol import (ROOT,geometry,native_contracts,resolve,
    build_model,parameter_count,audit_group,protocol,source_identity)
from experiments.lambda_repair_eval import CanonicalPosterior
from models import training


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signature(core):
    shapes={k:list(v.shape) for k,v in core.state_dict().items()}
    return hashlib.sha256(json.dumps(shapes,sort_keys=True).encode()).hexdigest()


def audit_routes():
    _,cells=inventory();contracts=native_contracts();all_rows=[]
    for cell in cells:
        spec=dataset_spec(cell)
        geo=geometry(cell.dataset,cell.representation,spec.expected_shapes[cell.representation])
        rows=[];field_signature=None
        for variant in contracts:
            cfg,resolved=resolve(variant,geo,device='cpu')
            with torch.device('meta'):
                model=build_model(variant,cfg,geo['ambient_dim'])
            assert parameter_count(model)==resolved['actual_parameters']
            assert cfg.field_projection_rank is None
            nf=variant=='scale_conditioned_nf'
            core=model.couplings[0].conditioner if nf else model.core
            core_signature=signature(core)
            if not nf:
                if field_signature is None:field_signature=core_signature
                assert core_signature==field_signature,(cell.key,variant)
                assert cfg.image_width==8 if geo['kind']=='image' else cfg.field_residual_width==512
            elif geo['kind']=='image':
                assert cfg.image_width==9 and len(model.couplings)==8
            rows.append(dict(cell_key=cell.key,variant=variant,resolved=resolved,
                core_class=type(core).__module__+'.'+type(core).__name__,
                core_state_shapes_sha256=core_signature,configuration=cfg.to_dict()))
        assert audit_group(rows)
        all_rows.extend(rows)
    assert len(cells)==39 and len(contracts)==13 and len(all_rows)==507
    # These shared files affect image NF even when its wrapper is untouched.
    retained={}
    for name in ['models/image_gaussian_fields.py','models/image_normalizing_flow.py']:
        original=subprocess.check_output(['git','show','d296210:'+name],cwd=ROOT)
        assert original==(ROOT/name).read_bytes(),name
        retained[name]=sha(ROOT/name)
    return dict(status='passed',cells=39,interfaces=13,configurations=507,
        scope='Model construction and routing; not507 completed trainings.',
        retained_image_nf_source_sha256=retained,protocol=protocol(),rows=all_rows)


def preflights(canonical_root,output,device):
    rows=[]
    for task in ['e6_exp_pca','e1_spiral_pca']:
        for representation in ['coefficients','dataset']:
            path=canonical_root/task/'train'/f'{representation}.npy'
            data=np.load(path,mmap_mode='r')
            indices=np.random.default_rng(817).choice(len(data),144,replace=False)
            raw=np.asarray(data[indices]).reshape(144,-1).astype(np.float32)
            geo=geometry(task,representation,data.shape[1:])
            for variant in ['rectified_flow','t_flowmatching','pfgmpp']:
                cfg,resolved=resolve(variant,geo,device=device,steps=8,preflight=True)
                family=native_contracts()[variant]['family']
                run=output/f'{variant}__{task}__{representation}';run.mkdir()
                result=training.train_model(family,raw[:128],raw[128:],cfg,run/'model.pt')
                loaded=training.load_checkpoint(run/'model.pt',device=device)
                assert result.metrics['steps_completed']==8
                assert result.config.to_dict()==loaded.config.to_dict()
                assert parameter_count(loaded.model)==resolved['actual_parameters']
                assert signature(loaded.model.core)==signature(result.model.core)
                for name,value in result.model.state_dict().items():
                    torch.testing.assert_close(value,loaded.model.state_dict()[name],rtol=0,atol=0)
                assert not hasattr(loaded.model,'basis')
                assert torch.count_nonzero(loaded.model.core.output.weight)>0
                model=loaded.model.double().eval()
                q=torch.as_tensor(raw[128:130],device=device,dtype=torch.float64).requires_grad_(True)
                adapter=CanonicalPosterior(SimpleNamespace(model=model,config=cfg,
                    family=training._canonical_family(family)),.7)
                generator=torch.Generator(device=device).manual_seed(818)
                direction=torch.randn(q.shape,device=device,dtype=q.dtype,generator=generator)
                probe=torch.randn(q.shape,device=device,dtype=q.dtype,generator=generator)
                value=adapter(q);gradient=torch.autograd.grad((value*probe).sum(),q)[0]
                analytic=(gradient*direction).sum(1)
                with torch.no_grad():
                    fd=((adapter(q+1e-5*direction)-adapter(q-1e-5*direction))*probe).sum(1)/2e-5
                gap=float((analytic-fd).abs().max())
                assert torch.isfinite(value).all() and torch.isfinite(gradient).all()
                torch.testing.assert_close(analytic,fd,atol=2e-5,rtol=2e-6)
                row=dict(status='passed',kind='technical_preflight_not_quality',variant=variant,
                    dataset=task,representation=representation,ambient_dim=raw.shape[1],
                    steps=8,fit_n=128,holdout_n=16,parameters=parameter_count(model),
                    image_width=cfg.image_width if geo['kind']=='image' else None,
                    field_width=cfg.field_residual_width if geo['kind']=='vector' else None,
                    best_step=result.metrics['best_step'],native_holdout_loss=result.metrics['best_validation_loss'],
                    configuration=cfg.to_dict(),checkpoint_sha256=sha(run/'model.pt'),
                    selected_source_train_indices=indices.tolist(),
                    selected_training_data_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                    derivative_directional_max_gap=gap,reload_bitwise=True,test_loaded=False)
                write_json(run/'verification.json',row);rows.append(row)
                print(json.dumps({k:row[k] for k in ['variant','dataset','representation','parameters','derivative_directional_max_gap']}),flush=True)
                del result,loaded,model,adapter,q,value,gradient
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--canonical-root',type=Path);p.add_argument('--device',default='cpu')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    routes=audit_routes();write_json(args.output/'routes.json',routes)
    results=[]
    if args.canonical_root:
        results=preflights(args.canonical_root,args.output,args.device)
    write_json(args.output/'verification.json',dict(status='passed',routing_configurations=507,
        technical_trainings=len(results),source_sha256=source_identity(),
        diagnostic_sha256=sha(Path(__file__)),results=results,
        benchmark_quality_assessed=False,canonical_test_loaded=False))


if __name__=='__main__':main()
