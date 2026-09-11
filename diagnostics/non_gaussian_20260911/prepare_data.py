"""Extract unchanged Exp/Spiral train/test arrays from the authenticated archive."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from datasets.archive import EXACT_ARCHIVE_PASSWORD, EXACT_ARCHIVE_SHA256


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    assert sha(args.archive) == EXACT_ARCHIVE_SHA256
    inventory = json.loads(args.inventory.read_text())
    args.output.mkdir(parents=True)
    hashes = {}
    for dataset in ('e6_exp_pca', 'e1_spiral_pca'):
        records = [r for r in inventory if r['cell']['dataset'] == dataset]
        files = {k:v for r in records for k,v in r['input_record']['source_files'].items()
            if k.startswith(('train/', 'test/'))}
        for name, metadata in sorted(files.items()):
            dest = args.output / dataset / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open('wb') as stream:
                subprocess.run(['unzip', '-p', '-P', EXACT_ARCHIVE_PASSWORD, str(args.archive),
                    f'benchmarks/{dataset}/{name}'], stdout=stream, stderr=subprocess.PIPE, check=True)
            hashes[f'{dataset}/{name}'] = sha(dest)
            assert hashes[f'{dataset}/{name}'] == metadata['sha256']
        print(json.dumps(dict(dataset=dataset, status='canonical_hashes_match')), flush=True)
    (args.output.parent/'input_verification.json').write_text(json.dumps(dict(
        archive_sha256=EXACT_ARCHIVE_SHA256, files_sha256=hashes,
        operation='unchanged canonical arrays; no representation conversion'), indent=2)+'\n')


if __name__ == '__main__': main()
