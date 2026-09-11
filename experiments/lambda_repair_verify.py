"""Audit retained source, training lineage and all evaluated pointwise scores."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from models.training import load_checkpoint
from datasets.registry import load_registry
from experiments.lambda_repair_eval import selected


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def equal(a,b):
    if isinstance(a,torch.Tensor):
        torch.testing.assert_close(a,b,rtol=0,atol=0);return
    if isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b);return
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b,(a,b)


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(2)
    registry=load_registry(Path(__file__).resolve().parents[1]/'configs/datasets/registry/paper_benchmarks.yaml',validate_official_coverage=False)
    completed=[];evaluations=[];lineages=[];data_checks=[];retired=[];incomplete=[]
    for folder in sorted((args.runs/'data').iterdir()):
        m=json.loads((folder/'manifest.json').read_text())
        for name,digest in m['output_sha256'].items():assert sha(folder/name)==digest
        fit=np.load(folder/'fit_indices.npy');holdout=np.load(folder/'holdout_indices.npy')
        assert len(fit)==99000 and len(holdout)==1000 and not np.intersect1d(fit,holdout).size
        assert np.array_equal(np.sort(np.r_[fit,holdout]),np.arange(100000))
        data_checks.append(dict(dataset=folder.name,files=len(m['output_sha256']),split_exact=True))
    for run in sorted(args.runs.iterdir()):
        if not run.is_dir() or run.name=='data':continue
        if not (run/'complete.json').exists():
            if (run/'manifest.json').exists():incomplete.append(run.name)
            continue
        manifest=json.loads((run/'manifest.json').read_text());completion=json.loads((run/'complete.json').read_text())
        for name,digest in manifest['source_sha256'].items():assert sha(run/'source'/name)==digest,(run,name)
        result=load_checkpoint(run/'model.pt',device='cpu')
        assert result.checkpoint_sha256==completion['checkpoint_sha256']==sha(run/'model.pt')
        assert result.config.steps==completion['metrics']['steps_completed']
        assert result.best_epoch==completion['metrics']['best_step']
        assert result.normalization_scale==completion['normalization_rms']
        assert result.preprocessing_sha256==completion['normalization_sha256']
        assert completion['reload_exact']
        completed.append(dict(run=run.name,variant=manifest['variant'],dataset=manifest['dataset'],
            representation=manifest['representation'],steps=result.config.steps,best_step=result.best_epoch,
            checkpoint_sha256=result.checkpoint_sha256,parameters=sum(x.numel() for x in result.model.parameters()),
            native_validation_loss=result.best_validation_loss))
        resume=manifest.get('resume')
        if resume:
            prefix=int(resume['completed_prefix'])
            if 'parent_run' in resume:
                parent=args.runs/resume['parent_run']/f'progress_{prefix:06d}.pt'
            else:
                # Early extension manifests retained the authoritative SHA but
                # not the optional display name. Resolve it by bytes, not guess.
                candidates=[]
                for candidate in args.runs.glob(f'*/progress_{prefix:06d}.pt'):
                    sibling=candidate.parent/'manifest.json'
                    if not sibling.exists():continue
                    source=json.loads(sibling.read_text())
                    if (source['variant'],source['dataset'])!=(manifest['variant'],manifest['dataset']):continue
                    if sha(candidate)==resume['parent_progress_sha256']:candidates.append(candidate)
                assert candidates,'parent progress SHA not found'
                parent=sorted(candidates)[0]
            child=run/f'progress_{prefix:06d}.pt'
            assert sha(parent)==resume['parent_progress_sha256']
            a=torch.load(parent,map_location='cpu',weights_only=True);b=torch.load(child,map_location='cpu',weights_only=True)
            keys=['model_state','optimizer_state','rng','sampler','counters','data_identity',
                  'normalization','best_state','best_step','best_validation_loss','history']
            for key in keys:equal(a[key],b[key])
            lineages.append(dict(run=run.name,parent=parent.parent.name,prefix=prefix,unchanged=keys,
                parent_resolved_by_sha='parent_run' not in resume))
            del a,b
        target=float(registry[manifest['dataset']].expected_lid)
        for directory in sorted(run.glob('evaluation*')):
            if 'stored_label_debug_error' in directory.name:
                retired.append(str(directory.relative_to(args.runs)));continue
            if not (directory/'selection.json').exists():continue
            receipt=json.loads((directory/'selection.json').read_text())
            arrays=np.load(directory/'holdout.npz');n=len(arrays['target'])
            assert n==1000 and np.all(arrays['target']==target)
            assert np.isfinite(arrays['full']).all()
            if 'response' in arrays:np.testing.assert_array_equal(arrays['response']+arrays['correction'],arrays['full'])
            scale,col,diagnostics=selected(arrays['full'],arrays['target'],receipt.get('selector','bounded_v2'))
            assert scale==receipt['selected_lambda'] and diagnostics['status']==receipt['status']
            np.testing.assert_allclose(np.abs(arrays['full'][:,col]-target).mean(),receipt['holdout_mae'],rtol=0,atol=1e-10)
            assert receipt['checkpoint_sha256']==result.checkpoint_sha256
            assert sum(x['n'] for x in receipt['selection_bootstrap'])==500
            if (directory/'source_parity.json').exists():
                parity=json.loads((directory/'source_parity.json').read_text())
                assert parity['float64_exact_max_difference']<1e-7
            test_mae=None
            if (directory/'test_metrics.json').exists():
                test=json.loads((directory/'test_metrics.json').read_text());values=np.load(directory/'test_selected.npz')
                assert test['checkpoint_sha256']==result.checkpoint_sha256 and test['selected_lambda']==scale
                assert len(values['target'])==1000 and np.all(values['target']==target) and values['scales'][0]==scale
                test_mae=float(abs(values['full'][:,0]-target).mean())
                np.testing.assert_allclose(test_mae,test['test_mae'],rtol=0,atol=1e-10)
            evaluations.append(dict(run=run.name,evaluation=directory.name,selected_lambda=scale,holdout_n=n,test_mae=test_mae))
        del result
    record=dict(status='verified',data=data_checks,completed_runs=completed,lineages=lineages,
        evaluations=evaluations,retired_wrong_raw_label_diagnostics=retired,
        incomplete_or_interrupted_runs=incomplete,
        scope='completed native checkpoints and saved predictions; this is not a universal convergence claim')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(status='verified',completed_runs=len(completed),evaluations=len(evaluations),
        lineages=len(lineages),data_sets=len(data_checks),retired=len(retired),incomplete=len(incomplete))))


if __name__=='__main__':main()
