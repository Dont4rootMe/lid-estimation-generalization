"""Blind continuous-law curves on fixed source-train holdout queries."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from datasets.registry import load_split
from experiments.fair_campaign import inventory,dataset_spec,write_json
from experiments.fair_measurements import common_scales
from experiments import global_campaign as v1
from models import training
from continuous_reference import reference

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'artifacts/non_gaussian_20260911/data/benchmarks'
OUT=ROOT/'artifacts/non_gaussian_20260911/population'


def prepare():
    torch.set_num_threads(2)
    cfg,cells=inventory();items=[]
    for cell in cells:
        if cell.dataset not in ('e6_exp_pca','e1_spiral_pca'):continue
        active=3 if cell.dataset=='e6_exp_pca' else 2
        name=f'{cell.dataset}__{cell.representation}'
        dest=OUT/name;dest.mkdir(parents=True,exist_ok=True)
        loaded=load_split(DATA,dataset_spec(cell),'train',representation=cell.representation,mmap_mode='r')
        raw=loaded.features.reshape(len(loaded.features),-1)
        part=v1.partition_source_train(raw,loaded.lid,selection=cfg['campaign']['selection'],seed=0)
        mean,rms,_,normalization_hash=training._normalization(training._flat_finite_data(part.fit_features,name='fit'),
            enabled=True,epsilon=1e-8)
        coefficients=np.load(DATA/cell.dataset/'train/coefficients.npy',mmap_mode='r')
        ids=part.selection_indices[:32]
        fit_ids=part.fit_indices[:4096]
        design=np.c_[coefficients[fit_ids,:active],np.ones(len(fit_ids))]
        fitted=np.linalg.lstsq(design,raw[fit_ids],rcond=None)[0]
        matrix,offset=fitted[:-1],fitted[-1];metric=matrix@matrix.T
        q=coefficients[ids,:active];raw_query=raw[ids]
        meta=dict(cell_key=cell.key,dataset=cell.dataset,representation=cell.representation,
            query_partition='first32 source-train holdout indices',test_queries=False,
            true_lid=2 if active==3 else 1,ambient=raw.shape[1],active=active,
            normalization_rms=rms,normalization_sha256=normalization_hash,
            fit_indices_sha256=v1._array_sha(part.fit_indices),
            holdout_indices_sha256=v1._array_sha(part.selection_indices),
            maximum_renderer_fit_residual=float(abs(design@fitted-raw[fit_ids]).max()),
            maximum_renderer_query_residual=float(abs(q@matrix+offset-raw_query).max()),
            maximum_renderer_gram_gap=float(abs(metric-np.eye(active)).max()),
            source_sha256=hashlib.sha256(Path(__file__).with_name('continuous_reference.py').read_bytes()).hexdigest())
        assert meta['maximum_renderer_query_residual']<1e-10
        assert meta['maximum_renderer_gram_gap']<1e-6
        np.savez_compressed(dest/'queries.npz',query_ids=ids,coefficients=q,raw=raw_query,
            renderer=matrix,offset=offset,metric=metric,mean=mean.numpy(),scales=common_scales())
        write_json(dest/'geometry.json',meta)
        for family,df in [('t_flowmatching',5),('pfgmpp',128),('gaussian',None)]:
            items.append((str(dest),family,df))
        del part,raw,loaded
    return items


def compute(item):
    folder,family,df=item;folder=Path(folder);start=time.time()
    meta=json.loads((folder/'geometry.json').read_text())
    with np.load(folder/'queries.npz') as z:
        queries=z['coefficients'];scales=z['scales'];metric=z['metric'];ids=z['query_ids']
    means=[];responses=[];checks=[]
    for index,scale in enumerate(scales):
        raw_scale=scale*meta['normalization_rms'];row_mean=[];row_response=[]
        for i,q in enumerate(queries):
            value=reference(meta['dataset'],q,raw_scale,meta['ambient'],df,metric=metric)
            if i<4:
                high=reference(meta['dataset'],q,raw_scale,meta['ambient'],df,metric=metric,
                    gamma_order=32,spatial_order=128,cutoff=12)
                response_gap=abs(value.response-high.response)
                posterior_gap=float(np.linalg.norm(value.mean-high.mean)/raw_scale)
                checks.append(dict(scale=float(scale),query_id=int(ids[i]),
                    response_order_cutoff_gap=response_gap,posterior_gap_over_raw_rms=posterior_gap))
                if max(response_gap,posterior_gap)>2e-4:
                    raise ValueError(('quadrature convergence',meta['cell_key'],family,float(scale),i,response_gap,posterior_gap))
            row_mean.append(value.mean);row_response.append(value.response)
        means.append(row_mean);responses.append(row_response)
        print(json.dumps(dict(event='oracle_scale',cell=meta['cell_key'],family=family,
            index=index,scale=float(scale),mean_response=float(np.mean(row_response)))),flush=True)
    np.savez_compressed(folder/(family+'.npz'),scales=scales,query_ids=ids,
        mean=np.asarray(means),response=np.asarray(responses))
    report=dict(status='complete',family=family,kernel_df=df,**meta,seconds=time.time()-start,
        gamma_order=16,spatial_order=64,cutoff=10,refinement=dict(gamma_order=32,
            spatial_order=128,cutoff=12,queries_per_scale=4),checks=checks)
    write_json(folder/(family+'.json'),report)
    return dict(cell=meta['cell_key'],family=family,seconds=report['seconds'])


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    jobs=prepare()
    with ProcessPoolExecutor(max_workers=6) as executor:
        rows=list(executor.map(compute,jobs))
    write_json(OUT/'complete.json',dict(status='complete',rows=rows))


if __name__=='__main__':main()
