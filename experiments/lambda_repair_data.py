"""Prepare the original strict benchmark data, without using LID for training."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess

import numpy as np

from datasets.archive import EXACT_ARCHIVE_PASSWORD, EXACT_ARCHIVE_SHA256
from experiments.global_campaign import partition_source_train
from experiments.global_campaign_v2 import compose_global_campaign_v2_config

TASKS = ['e6_exp_pca', 'e8_spaghetti_pca', 'e1_spiral_pca', 'e8_sphere4_pca',
         'e7_crescent_moon_radius3.0', 'e2_uniform_pca', 'e8_gaussian4_pca']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--inventory', type=Path)
    parser.add_argument('--datasets', nargs='+', default=TASKS)
    args = parser.parse_args()
    archive = args.archive
    digest = hashlib.sha256()
    with archive.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    assert digest.hexdigest() == EXACT_ARCHIVE_SHA256
    inventory = json.loads(args.inventory.read_text()) if args.inventory else []
    config = compose_global_campaign_v2_config()
    args.output.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        dest = args.output/dataset
        dest.mkdir(exist_ok=True)
        hashes = {}
        item = next((x for x in inventory if x['cell']['dataset']==dataset and x['cell']['representation']=='dataset'), None)
        arrays = {}
        for split in ['train', 'test']:
            for field in ['coefficients', 'dataset', 'lid']:
                member = f'benchmarks/{dataset}/{split}/{field}.npy'
                proc = subprocess.run(['unzip','-p','-P',EXACT_ARCHIVE_PASSWORD,str(archive),member], stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                if proc.returncode: raise RuntimeError(f'Extraction failed: {dataset}/{split}/{field}')
                hashes[f'{split}/{field}.npy'] = hashlib.sha256(proc.stdout).hexdigest()
                expected = item['input_record']['source_files'].get(f'{split}/{field}.npy') if item else None
                if expected: assert hashes[f'{split}/{field}.npy'] == expected['sha256']
                value = np.load(io.BytesIO(proc.stdout), allow_pickle=False)
                # Upstream _flat_finite_data performs exactly this float32 cast.
                if field != 'lid': value = value.reshape(len(value), -1).astype(np.float32)
                arrays[split,field] = value
                np.save(dest/f'{split}_{field}.npy', value)
                del proc
        part = partition_source_train(arrays['train','dataset'], arrays['train','lid'], selection=config['campaign']['selection'], seed=0)
        np.save(dest/'fit_indices.npy', part.fit_indices)
        np.save(dest/'holdout_indices.npy', part.selection_indices)
        assert len(part.fit_indices)==99000 and len(part.selection_indices)==1000
        receipt={'dataset':dataset,'archive_sha256':EXACT_ARCHIVE_SHA256,'source_array_sha256':hashes,
                 'fit_n':99000,'holdout_n':1000,'test_n':len(arrays['test','dataset']),
                 'output_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.glob('*.npy')}}
        (dest/'manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps({'dataset':dataset,'status':'ready'}),flush=True)


if __name__=='__main__':main()
