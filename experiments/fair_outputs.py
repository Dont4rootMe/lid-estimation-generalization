"""Query-level benchmark accounting; no fitting or scale selection occurs here.

Canonical and generated inventories and the two primary selection protocols
remain separate. Secondary readouts reuse the primary frozen lambda. E5 uses
the canonical archive's row-pairing contract, with row IDs and class-label order
checked explicitly; class labels alone do not establish individual identity.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.fair_protocol import native_contracts
from experiments.metrics import known_lid_metrics, prediction_summary, paired_delta_metrics


SUPERVISED = 'held_out_source_train_supervised_mae_common_grid_v3'
POINTWISE = 'pointwise_kneedle_common_grid_v3'
LEGACY = 'historical_flipd_pointwise_kneedle_v1'
REFERENCE = 'held_out_reference_mean_kneedle_v2'
SCORES = ('strict_coefficients', 'strict_dataset', 'arrows_mae',
          'e1_relative_drift', 'e5_invariance', 'e5_equivariance')


def readouts(variant):
    if variant == 'scale_conditioned_nf':
        return ('ols5', 'fixed_likelihood')
    family = native_contracts()[variant]['family']
    if family in ('student_t_flow','pfgmpp'):
        return ('response',)
    return ('full', 'response') if family in (
        'independent_affine_flow', 'rectified_flow', 'schrodinger_bridge') else ('full',)


def finite_vector(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 1 or not len(value) or not np.isfinite(value).all():
        raise ValueError('invalid finite prediction vector: ' + name)
    return value


def check_alignment(current, reference, *, paired):
    if not np.array_equal(current['query_ids'], reference['query_ids']):
        raise ValueError('reference query IDs/order differ')
    if paired:
        labels, base_labels = current['labels'], reference['labels']
        if (labels.shape != current['query_ids'].shape or
                labels.dtype.kind not in 'iu' or
                not np.array_equal(labels, base_labels)):
            raise ValueError('paired class labels/order differ or are missing')


def relative_metrics(current, reference, expected_delta, *, paired):
    current = finite_vector(current, 'current')
    reference = finite_vector(reference, 'reference')
    if current.shape != reference.shape:
        raise ValueError('reference prediction counts differ')
    if not np.isfinite(expected_delta):
        raise ValueError('invalid expected LID delta')
    if paired:
        return paired_delta_metrics(reference, current, expected_delta=expected_delta)
    stats, base_stats = prediction_summary(current), prediction_summary(reference)
    delta = stats['mean'] - base_stats['mean']
    return dict(stats, reference_mean=base_stats['mean'],
        reference_median=base_stats['median'], mean_delta_from_reference=delta,
        median_delta_from_reference=stats['median'] - base_stats['median'],
        mean_delta_error=delta - expected_delta)


def reference_metrics(row, arrays, reference, reference_arrays):
    """Check reference binding and compute every readout's E1/E5 metric."""
    cell = row['cell']
    if cell['target_policy'] not in ('paired_delta', 'sample_size'):
        raise ValueError('unknown relative target policy')
    if row['selected_lambda'] != reference['selected_lambda']:
        raise ValueError('dependent result did not reuse frozen reference lambda')
    paired = cell['target_policy'] == 'paired_delta'
    check_alignment(arrays, reference_arrays, paired=paired)
    if not paired and row['test_files_sha256'] != reference['test_files_sha256']:
        raise ValueError('E1 must use the same reference test data')
    failed = row['measurement_status'] == 'selection_failed'
    if failed != (reference['measurement_status'] == 'selection_failed'):
        raise ValueError('dependent/reference measurement coverage differs')
    return {} if failed else {readout: relative_metrics(arrays[readout], reference_arrays[readout],
        cell['expected_lid_delta'], paired=paired) for readout in readouts(row['variant'])}


def result_records(entries):
    """Derive records from verified (completion row, artifact directory) pairs.

    This function also accepts preflight fixtures for diagnostic validation;
    only fair_campaign.aggregate may certify a complete benchmark matrix.
    """
    indexed = {}
    for row, directory in entries:
        key = (row['cell_key'], row['variant'])
        if key in indexed:
            raise ValueError('duplicate result cell')
        with np.load(Path(directory) / 'test_predictions.npz', allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        indexed[key] = row, Path(directory), arrays
    records = []
    for key, (row, directory, arrays) in sorted(indexed.items()):
        cell = row['cell']
        known = cell['target_policy'] == 'known_lid'
        common = dict(model_variant=row['variant'], cell_key=row['cell_key'],
            inventory_origin='canonical_35' if cell['exact_archive'] else 'generated_e3_e4_extension_4',
            suite_id=cell['suite_id'], dataset=cell['dataset'], representation=cell['representation'],
            reference_dataset=cell['reference_dataset'], split='test', kind=row['kind'],
            n=row['test_n'], n_source_train=row['n_source_train'],
            selected_coordinate=row['selected_lambda'], selected_coordinate_name='lambda',
            selection_status=row['measurement_status'], failure_reason=row['selection'].get('failure_reason'),
            reference_failure_reason=row['selection'].get('reference_failure_reason'),
            checkpoint_sha256=row['checkpoint_sha256'], protocol_sha256=row['protocol_sha256'])
        if known:
            target = finite_vector(arrays['target'], 'target')
            target_stats = dict(true_lid_mean=float(target.mean()),
                true_lid_min=float(target.min()), true_lid_max=float(target.max()))
            for readout in readouts(row['variant']):
                values = finite_vector(arrays[readout], readout)
                records.append(dict(common, **known_lid_metrics(values, target), **target_stats,
                    analysis='known_lid', selection_protocol=SUPERVISED, readout=readout,
                    is_primary_readout=readout == row['primary_readout'],
                    readout_scale_rule='reuse_primary_holdout_scale'))
            with np.load(directory / 'test_scale_curves.npz', allow_pickle=False) as z:
                for name, selector in (('kneedle_common_grid', POINTWISE), ('kneedle_legacy', LEGACY)):
                    values = finite_vector(z[name + '_prediction'], name)
                    diagnostic = row['diagnostic_metrics'][name]
                    records.append(dict(common, **known_lid_metrics(values, target), **target_stats,
                        analysis='known_lid', selection_protocol=selector, readout=row['primary_readout'],
                        is_primary_readout=True, readout_scale_rule='pointwise_primary_curve',
                        selected_coordinate=None, selection_status='selected', failure_reason=None,
                        fallback_n=diagnostic['fallback_n'], boundary_n=diagnostic['boundary_n'],
                        detected_n=diagnostic['detected_n'],
                        selected_lambda_counts=diagnostic['selected_lambda_counts']))
            continue
        reference_key = (f"{cell['suite_id']}/{cell['reference_dataset']}/{cell['representation']}", row['variant'])
        if reference_key not in indexed:
            raise ValueError('missing reference result: ' + reference_key[0])
        reference_row, _, reference_arrays = indexed[reference_key]
        paired = cell['target_policy'] == 'paired_delta'
        relative = reference_metrics(row, arrays, reference_row, reference_arrays)
        if row.get('reference_metrics') != relative:
            raise ValueError('reported reference metrics differ from saved arrays')
        for readout in readouts(row['variant']):
            stats = relative.get(readout, {})
            records.append(dict(common, **stats,
                analysis='e5_paired_delta' if paired else 'e1_sample_size_stability',
                selection_protocol=REFERENCE, readout=readout,
                is_primary_readout=readout == row['primary_readout'],
                readout_scale_rule='reuse_reference_primary_holdout_scale',
                expected_lid_delta=cell['expected_lid_delta'],
                pairing_status='canonical_row_order_and_labels_checked' if paired else 'identical_test_files_checked'))
    return records


def score_value(rows, score):
    """Existing six table definitions, with no omission of failed cells.

    Return (value, status); undefined denominators produce no numeric score.
    Coverage and exact inventory identities must be checked by table_scores.
    """
    if any(r['selection_status'] == 'selection_failed' for r in rows):
        return None, 'selection_failed'
    if score.startswith('strict_'):
        suites = defaultdict(list)
        for row in rows:
            truth = row['true_lid_mean']
            if truth == 0:
                return None, 'undefined_zero_true_lid'
            suites[row['suite_id']].append(np.log1p(row['mae'] / abs(truth)))
        value = np.expm1(np.mean([np.mean(v) for v in suites.values()]))
    elif score == 'arrows_mae':
        value = rows[0]['mae']
    elif score == 'e1_relative_drift':
        rows = sorted(rows, key=lambda r: -r['n_source_train'])
        sizes = np.array([r['n_source_train'] for r in rows], dtype=float)
        if np.any(sizes <= 0) or len(np.unique(sizes)) != len(sizes):
            raise ValueError('E1 sample sizes must be positive and distinct')
        means = np.array([r['reference_mean'] for r in rows])
        if np.any(means == 0):
            return None, 'undefined_zero_reference_mean'
        if not np.all(means == means[0]):
            raise ValueError('E1 reference mean differs across sample sizes')
        x = np.log2(sizes.max() / sizes)
        if x.max() == x.min():
            return None, 'insufficient_sample_sizes'
        drift = np.abs(np.array([r['mean_delta_error'] for r in rows]) / means)
        y = np.log1p(drift)
        area = np.sum(.5 * (y[1:] + y[:-1]) * np.diff(x))
        value = np.expm1(area / (x.max() - x.min()))
    elif score in ('e5_invariance', 'e5_equivariance'):
        value = np.expm1(np.mean(np.log1p([r['mae'] for r in rows])))
    else:
        raise ValueError('unknown table score')
    if not np.isfinite(value):
        raise ValueError('nonfinite table score')
    return float(value), 'available'


def score_cell(cell, score):
    if not cell['exact_archive']:
        return False
    if score.startswith('strict_'):
        return (cell['target_policy'] == 'known_lid' and cell['dataset'] != 'e2_arrows'
            and cell['suite_id'] not in ('e3', 'e4') and cell['representation'] == score[7:])
    if score == 'arrows_mae':
        return cell['dataset'] == 'e2_arrows'
    if score == 'e1_relative_drift':
        return cell['target_policy'] == 'sample_size'
    if score in ('e5_invariance', 'e5_equivariance'):
        return (cell['target_policy'] == 'paired_delta' and cell['dataset'] != cell['reference_dataset']
            and ((abs(cell['expected_lid_delta']) > 1e-12) == (score == 'e5_equivariance')))
    raise ValueError('unknown table score')


def table_scores(records, cells):
    """Long-form six-column tables; missing coverage is explicit, never averaged away."""
    cells = {f"{c['suite_id']}/{c['dataset']}/{c['representation']}": c for c in cells}
    result = []
    for variant in native_contracts():
        for readout in readouts(variant):
            primary = readout == readouts(variant)[0]
            for selector in (SUPERVISED, POINTWISE) if primary else (SUPERVISED,):
                selected = [r for r in records if r['model_variant'] == variant and r['readout'] == readout
                    and r['inventory_origin'] == 'canonical_35'
                    and r['selection_protocol'] in (selector, REFERENCE)]
                for score in SCORES:
                    expected = {k for k, c in cells.items() if score_cell(c, score)}
                    rows = [r for r in selected if r['cell_key'] in expected]
                    if len({r['cell_key'] for r in rows}) != len(rows):
                        raise ValueError('duplicate cell in a table score')
                    available = sum(r['selection_status'] != 'selection_failed' for r in rows)
                    value, status = (None, 'not_in_scope') if not expected else (
                        (None, 'incomplete_coverage') if len(rows) != len(expected) else score_value(rows, score))
                    result.append(dict(model_variant=variant, readout=readout,
                        is_primary_readout=primary, known_selection_protocol=selector,
                        unknown_selection_protocol=REFERENCE, score=score, value=value, status=status,
                        expected_cells=len(expected), recorded_cells=len(rows), available_cells=available,
                        boundary_selected_cells=sum(r['selection_status'] == 'scale_unresolved' for r in rows),
                        fallback_queries=sum(r.get('fallback_n', 0) for r in rows),
                        boundary_queries=sum(r.get('boundary_n', 0) for r in rows), lower_is_better=True))
    return result


def write_csv(path, records):
    """Deterministic flat export; structured distribution fields remain JSON cells."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for record in records for k in record})
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({k: json.dumps(v, sort_keys=True, allow_nan=False) if isinstance(v, (dict, list)) else v
                for k, v in row.items()})
    temporary.replace(path)
