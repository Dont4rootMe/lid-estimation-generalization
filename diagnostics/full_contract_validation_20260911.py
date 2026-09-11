"""Replay current Full receipts and check legacy Student weights without edits."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from experiments.fair_campaign import verify_measurements,file_sha,write_json,inventory,dataset_spec
from experiments import fair_data,fair_outputs
from experiments.fair_protocol import ROOT,source_identity,native_contracts,resolve,geometry,build_model
from models import training


def equivalent_interfaces():
    variants=['ve_diffusion','schrodinger_bridge']+[v for v,c in native_contracts().items()
        if c['family']=='independent_affine_flow']
    rows=[]
    for shape,representation in [([30],'coefficients'),([4,4,1],'dataset')]:
        geo=geometry('fixture',representation,shape)
        x=torch.randn(6,geo['ambient_dim'],dtype=torch.float64,generator=torch.Generator().manual_seed(195))
        reference=None;ref_loss=None;ref_gradient=None
        for variant in variants:
            torch.manual_seed(191)
            config,_=resolve(variant,geo,device='cpu',steps=4,preflight=True)
            model=build_model(variant,config,geo['ambient_dim']).double().train()
            model._lid_noise_pairing='antithetic_v1'
            if reference is None:
                with torch.no_grad():
                    model.core.output.weight.normal_(0,.01);model.core.output.bias.normal_(0,.01)
                reference=copy.deepcopy(model.core.state_dict())
            else:model.core.load_state_dict(reference)
            loss=training._objective(training._canonical_family(native_contracts()[variant]['family']),
                model,x,config,torch.Generator().manual_seed(199))
            gradient=torch.cat([v.flatten() for v in torch.autograd.grad(loss,tuple(model.parameters()))])
            if ref_loss is None:ref_loss=loss.detach();ref_gradient=gradient
            gap=float((gradient-ref_gradient).norm()/ref_gradient.norm())
            assert gap<1e-10
            rows.append(dict(variant=variant,representation=representation,ambient_dim=geo['ambient_dim'],
                loss=float(loss.detach()),loss_gap=float(abs(loss.detach()-ref_loss)),gradient_relative_gap=gap))
    return rows


def legacy_weights(root,canonical_root,device):
    spec=importlib.util.spec_from_file_location('historical_student_full',
        ROOT/'diagnostics/non_gaussian_ambient_20260911/student_full.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=[]
    for variant in ('t_flowmatching','pfgmpp'):
        for short,dataset in [('exp','e6_exp_pca'),('spiral','e1_spiral_pca')]:
            path=root/f'{variant}__{short}__coefficients'/'model.pt'
            original_sha=file_sha(path)
            result=training.load_checkpoint(path,device=device);result.model.double()
            raw=np.array(np.load(canonical_root/dataset/'test/coefficients.npy',mmap_mode='r')[:3])
            _,_,mean,normal_scale=training._model_and_context(result,None)
            normalized=(training._flat_finite_data(raw,name='query')-mean.reshape(1,-1))/normal_scale
            batch=normalized.to(device=device,dtype=torch.float64)
            for scale in (.5,32.):
                response=training.predict_lid(result,raw,scale,readout='response',divergence_backend='exact',trace_probes=0)
                full=training.predict_lid(result,raw,scale,readout='full',divergence_backend='exact',trace_probes=0)
                historical=module.density_dilation(result.model,batch,scale,
                    torch.as_tensor(response,device=device),result.config.kernel_df)[0].cpu().numpy()
                np.testing.assert_allclose(full,historical,atol=1e-10,rtol=1e-10)
                rows.append(dict(variant=variant,dataset=dataset,representation='coefficients',
                    queries=3,lambda_value=scale,full_historical_max_gap=float(abs(full-historical).max()),
                    checkpoint_sha256=original_sha,kind='fixed_checkpoint_API_identity_not_new_quality'))
            assert file_sha(path)==original_sha
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--legacy-root',type=Path);p.add_argument('--canonical-root',type=Path)
    p.add_argument('--device',default='cpu');args=p.parse_args()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    # Registry coverage and pinned origin metadata, without opening any split.
    _,cells=inventory()
    assert len(fair_data.manifest()['cells'])==len(cells)==39
    for cell in cells:
        for split in ('train','test'):
            expected=fair_data.expected_files(cell,split)
            assert cell.representation+'.npy' in expected
            assert {a+'.npy' for a in dataset_spec(cell).required_artifacts}<=set(expected)
    entries=[];runs=[]
    for path in sorted(args.runs.resolve().glob('*/complete.json')):
        row=verify_measurements(path)
        assert row['kind']=='preflight' and row['source_sha256']==source_identity()
        entries.append((row,path.parent))
        runs.append(dict(directory=str(path.parent.relative_to(ROOT)),variant=row['variant'],
            cell_key=row['cell_key'],steps=row['steps_completed'],primary_readout=row['primary_readout'],
            test_n=row['test_n'],selected_lambda=row['selected_lambda'],measurement_status=row['measurement_status'],
            metrics=row['metrics'],automatic_metrics=row['automatic_metrics'],reference_metrics=row['reference_metrics'],
            complete_sha256=file_sha(path),checkpoint_sha256=row['checkpoint_sha256'],
            parameters=row['resolved']['actual_parameters'],test_loaded_after_selection=True))
    records=fair_outputs.result_records(entries)
    fair_outputs.write_csv(args.output.with_suffix('.results.csv'),records)
    result=dict(status='passed',benchmark_quality_assessed=False,source_sha256=source_identity(),
        input_manifest_sha256=fair_data.contract_sha(),pinned_inventory_cells=len(cells),runs=runs,
        equivalent_interfaces=equivalent_interfaces(),result_records=len(records),legacy_weight_checks=[])
    if args.legacy_root:
        if args.canonical_root is None:raise ValueError('--canonical-root required for legacy checks')
        result['legacy_weight_checks']=legacy_weights(args.legacy_root,args.canonical_root,args.device)
    write_json(args.output,result)
    print(json.dumps(dict(status='passed',technical_runs=len(runs),exported_records=len(records),
        legacy_checks=len(result['legacy_weight_checks']),max_equivalent_gradient_gap=
        max(r['gradient_relative_gap'] for r in result['equivalent_interfaces']))))


if __name__=='__main__':main()
