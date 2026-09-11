"""Prepare the exact source-train partition used by the Arrows repair.

Input is an extracted e2_arrows directory, containing train/val/test.
Images stay unchanged; the output links to that directory and records hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.global_campaign import _array_sha, partition_source_train
from models.training import _flat_finite_data, _normalization


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arrows-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    source=args.arrows_root.resolve()
    if not source.is_dir():
        raise ValueError('expected extracted e2_arrows directory')
    raw=np.load(source/'train/dataset.npy',mmap_mode='r')
    if raw.dtype!=np.uint8 or raw.shape!=(100000,32,32,3):
        raise ValueError('this repair protocol expects original uint8 Arrows100000x32x32x3')
    files={}
    for split,count in [('train',100000),('val',10000),('test',10000)]:
        for name in ('dataset','lid'):
            path=source/split/f'{name}.npy'
            if np.load(path,mmap_mode='r').shape[0]!=count:
                raise ValueError(f'incorrect {split}/{name} sample count')
            files[f'{split}/{name}.npy']=sha(path)
    partition=partition_source_train(raw.reshape(len(raw),-1),None,
        selection={'fraction':.2,'minimum_selection':2,'maximum_selection':1000,'minimum_fit':8},seed=0)
    torch.set_num_threads(4)
    mean,rms,_,_= _normalization(_flat_finite_data(partition.fit_features,name='fit'),
                                enabled=True,epsilon=1e-8)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'benchmarks').mkdir()
    (args.output/'benchmarks/e2_arrows').symlink_to(source,target_is_directory=True)
    np.save(args.output/'fit_indices.npy',partition.fit_indices)
    np.save(args.output/'holdout_indices.npy',partition.selection_indices)
    np.save(args.output/'mean.npy',mean.numpy().reshape(32,32,3))
    manifest={'dataset':'e2_arrows','data_relative_root':'benchmarks/e2_arrows',
        'files_sha256':files,'n_fit':99000,'n_holdout':1000,'n_test':10000,
        'partition_matches_v2':True,'partition':partition.record,
        'fit_indices_sha256':_array_sha(partition.fit_indices),
        'holdout_indices_sha256':_array_sha(partition.selection_indices),
        'normalization_rms':rms,'mean_file_sha256':sha(args.output/'mean.npy'),
        'training_labels_used_for_loss':False,'test_labels_used_for_selection':False}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'output':str(args.output),'n_fit':99000,'n_holdout':1000,'normalization_rms':rms}))


if __name__=='__main__':
    main()
