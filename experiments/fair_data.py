"""Pinned benchmark bytes establish dataset identity and canonical row order.

The manifest is distilled from the archived rerun's input inventory, whose
canonical archive and generated-extension verification receipts are retained.
Checking a local file against it does not require re-downloading the archive.
No class label is treated as an individual object's identity.
"""
import hashlib
import json
from pathlib import Path

MANIFEST_PATH=Path(__file__).resolve().parents[1]/'configs/fair_comparison/input_manifest.json'


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def manifest():
    return json.loads(MANIFEST_PATH.read_text())


def contract_sha():
    return file_sha(MANIFEST_PATH)


def expected_files(cell,split):
    cell=cell if isinstance(cell,dict) else vars(cell)
    key=f"{cell['suite_id']}/{cell['dataset']}/{cell['representation']}"
    pinned=manifest()
    expected=pinned['cells'].get(key)
    if expected is None or any(cell.get(k)!=v for k,v in expected.items()):
        raise ValueError('cell identity differs from the pinned input manifest')
    prefix=cell['dataset']+'/'+split+'/'
    files={Path(k).name:v for k,v in pinned['datasets'][cell['dataset']]['files'].items()
        if k.startswith(prefix)}
    if not files:
        raise ValueError('split has no pinned source files')
    return files


def validate_hashes(cell,split,hashes):
    expected={k:v['sha256'] for k,v in expected_files(cell,split).items()}
    if hashes!=expected:
        raise ValueError('input files/order differ from the pinned '+split+' snapshot')


def validate_loaded(cell,loaded,split):
    expected=expected_files(cell,split)
    paths=list(loaded.source_paths.values())
    if len(paths)!=len(expected) or {p.name for p in paths}!=set(expected):
        raise ValueError('loaded artifact set differs from pinned inputs')
    if any(p.stat().st_size!=expected[p.name]['size_bytes'] for p in paths):
        raise ValueError('input file size differs from pinned inputs')
    hashes={p.name:file_sha(p) for p in paths}
    validate_hashes(cell,split,hashes)
    return hashes


def validate_receipt(row,*,splits=('train','test')):
    if (row.get('input_manifest_sha256')!=contract_sha()
            or row.get('query_order_contract')!='pinned_input_rows_v1'):
        raise ValueError('missing or changed pinned input/row-order attestation')
    for split in splits:
        validate_hashes(row['cell'],split,row.get(split+'_files_sha256'))


def validate_pair(current,reference):
    """Canonical file order, same registered reference, then row-ID checks."""
    validate_receipt(current,splits=('test',))
    validate_receipt(reference,splits=('test',))
    c,r=current['cell'],reference['cell']
    if (c['suite_id']!=r['suite_id'] or c['representation']!=r['representation']
            or c['reference_dataset']!=r['dataset']
            or r['reference_dataset']!=r['dataset']):
        raise ValueError('paired rows do not use the registered canonical reference')
