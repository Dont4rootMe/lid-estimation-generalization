"""Accounting controls: paired arithmetic, frozen scales and complete coverage."""
import numpy as np
import pytest

from experiments import fair_outputs as out, fair_measurements as m
from experiments.fair_campaign import inventory, dataset_spec
from experiments import fair_data

pytestmark=pytest.mark.usefixtures('fixture_input_manifest')


def entry(tmp_path, dataset, *, policy='paired_delta', reference='base',
          values=(3., 5.), variant='rectified_flow', failed=False, labels=(1, 2)):
    cell = dict(suite_id='e5' if policy == 'paired_delta' else 'e1', dataset=dataset,
        reference_dataset=reference, representation='dataset', target_policy=policy,
        exact_archive=True, expected_lid_delta=0. if dataset == reference else 4.)
    row = dict(cell=cell, cell_key=f"{cell['suite_id']}/{dataset}/dataset", variant=variant,
        scale_protocol=m.SCALE_PROTOCOL,
        primary_readout=out.readouts(variant)[0], kind='preflight', test_n=2,
        n_source_train=50000 if dataset == reference else 25000,
        selected_lambda=None if failed else 2., measurement_status='selection_failed' if failed else 'selected',
        selection=dict(failure_reason='no_knee' if failed else None),
        checkpoint_sha256='fixture', protocol_sha256='fixture',
        test_files_sha256={'dataset.npy':'same_fixture'}, reference_metrics={})
    pinned=fair_data.manifest()
    pinned['cells'][row['cell_key']]=dict(cell)
    pinned['datasets'][dataset]={'files':{dataset+'/test/dataset.npy':
        dict(sha256='same_fixture',size_bytes=0)}}
    row.update(input_manifest_sha256=fair_data.contract_sha(),query_order_contract='pinned_input_rows_v1')
    arrays = dict(query_ids=np.arange(2, dtype=np.int64), labels=np.array(labels, dtype=np.int64), target=np.array([]))
    if not failed:
        arrays.update({r: np.array(values) for r in out.readouts(variant)})
    path = tmp_path / dataset
    path.mkdir()
    np.savez_compressed(path / 'test_predictions.npz', **arrays)
    return row, path, arrays


def bind(current, reference):
    row, _, arrays = current
    base, _, base_arrays = reference
    row['reference_metrics'] = out.reference_metrics(row, arrays, base, base_arrays)
    return row, current[1]


def test_paired_mae_cannot_be_replaced_by_error_of_means(tmp_path):
    base = entry(tmp_path, 'base', values=(3., 5.))
    transformed = entry(tmp_path, 'transformed', values=(9., 7.))
    records = out.result_records([bind(base, base), bind(transformed, base)])
    rows = [r for r in records if r['dataset'] == 'transformed']
    assert len(rows) == 2
    for row in rows:
        # Deltas 6 and 2 average to the expected 4, but each misses by 2.
        assert row['mae'] == 2. and row['bias'] == 0.
        assert row['selected_coordinate'] == 2.
        assert row['pairing_status'] == 'pinned_canonical_files_row_order_and_labels_checked'
    assert [r['is_primary_readout'] for r in rows] == [True, False]


@pytest.mark.parametrize('change', ['labels', 'ids', 'count', 'scale', 'prediction_nan'])
def test_bad_reference_pair_is_rejected(tmp_path, change):
    base = entry(tmp_path, 'base')
    transformed = entry(tmp_path, 'transformed')
    if change == 'labels': transformed[2]['labels'] = transformed[2]['labels'][::-1]
    elif change == 'ids': transformed[2]['query_ids'] = transformed[2]['query_ids'][::-1]
    elif change == 'count': transformed[2]['full'] = transformed[2]['full'][:1]
    elif change == 'scale': transformed[0]['selected_lambda'] = 1.
    else: transformed[2]['full'][0] = np.nan
    with pytest.raises(ValueError): bind(transformed, base)


def test_e1_saves_source_size_and_uses_exact_same_test_data(tmp_path):
    base = entry(tmp_path, 'base', policy='sample_size', values=(3., 5.))
    small = entry(tmp_path, 'small', policy='sample_size', values=(5., 9.))
    small[0]['cell']['expected_lid_delta'] = 0.
    records = out.result_records([bind(base, base), bind(small, base)])
    row = next(r for r in records if r['dataset'] == 'small' and r['readout'] == 'full')
    assert row['n_source_train'] == 25000 and row['reference_mean'] == 4.
    assert row['mean_delta_error'] == 3.
    small[0]['test_files_sha256']['dataset.npy'] = 'different_fixture'
    with pytest.raises(ValueError, match='same reference test data'): bind(small, base)


def test_failed_reference_retains_every_cell_and_readout(tmp_path):
    base = entry(tmp_path, 'base', failed=True)
    transformed = entry(tmp_path, 'transformed', failed=True)
    records = out.result_records([bind(base, base), bind(transformed, base)])
    assert len(records) == 4
    assert all(r['selection_status'] == 'selection_failed' and 'mae' not in r for r in records)
    assert out.score_value(records, 'e5_equivariance') == (None, 'selection_failed')


def test_modified_saved_reference_metric_is_rejected(tmp_path):
    base = entry(tmp_path, 'base')
    row, path = bind(base, base)
    row['reference_metrics']['full']['mae'] = 99.
    with pytest.raises(ValueError, match='reference metrics differ'):
        out.result_records([(row, path)])


def test_relative_drift_integrates_log_sample_size_and_handles_zero_denominator():
    rows = [dict(n_source_train=n, reference_mean=10., mean_delta_error=d, selection_status='selected')
        for n, d in ((12500, 30.), (50000, 0.), (25000, 10.))]
    # log1p(drift) = 0, log(2), log(4), equally spaced in log2(sample size).
    value, status = out.score_value(rows, 'e1_relative_drift')
    assert value == pytest.approx(1.) and status == 'available'
    for row in rows: row['reference_mean'] = 0.
    assert out.score_value(rows, 'e1_relative_drift') == (None, 'undefined_zero_reference_mean')


def test_strict_aggregation_balances_suites_instead_of_pooling_cells():
    rows = [dict(suite_id=s, mae=e, true_lid_mean=1., selection_status='selected')
        for s, e in (('a', 0.), ('a', 0.), ('a', 0.), ('b', 3.))]
    value, status = out.score_value(rows, 'strict_coefficients')
    assert value == pytest.approx(1.) and status == 'available'


def test_table_coverage_never_hides_missing_failed_or_boundary_cells():
    _, cells = inventory()
    cells = [vars(c) for c in cells]
    strict = [c for c in cells if out.score_cell(c, 'strict_coefficients')]
    assert len(strict) == 7 and len({c['suite_id'] for c in strict}) == 5
    records = [dict(model_variant='ve_diffusion', readout='full', inventory_origin='canonical_35',
        selection_protocol=out.SUPERVISED, cell_key=f"{c['suite_id']}/{c['dataset']}/{c['representation']}",
        suite_id=c['suite_id'], true_lid_mean=2., mae=1., selection_status='selected') for c in strict]
    def score(records):
        return next(r for r in out.table_scores(records, cells) if r['model_variant'] == 've_diffusion'
            and r['known_selection_protocol'] == out.SUPERVISED and r['score'] == 'strict_coefficients')
    result = score(records)
    assert result['value'] == pytest.approx(.5) and result['expected_cells'] == 7
    missing = score(records[:-1])
    assert missing['status'] == 'incomplete_coverage' and missing['value'] is None
    records[-1]['selection_status'] = 'scale_unresolved'
    assert score(records)['boundary_selected_cells'] == 1 and score(records)['value'] == pytest.approx(.5)
    records[-1]['selection_status'] = 'selection_failed'
    assert score(records)['status'] == 'selection_failed' and score(records)['available_cells'] == 6
    with pytest.raises(ValueError, match='duplicate'): score(records + records[:1])


def test_known_export_keeps_primary_automatic_legacy_and_secondary_scales_separate(tmp_path):
    e = entry(tmp_path, 'known', variant='posterior_rectified_flow', policy='known_lid', reference=None)
    row, path, arrays = e
    target = np.full(2, 2.)
    scales = m.common_scales()
    curve = target[:, None] + np.log2(scales)[None, :] ** 2
    plan = m.known_plan(curve, scales, target, expected_grid=scales)
    row['selected_lambda'] = 1.
    arrays.update(target=target, full=target.copy(), response=target + .25)
    row['diagnostic_metrics'], curves = m.known_results(curve, np.ones((2, 22)), target, plan, 30,
        response_curves=(curve+.25,np.ones((2,22))+.25))
    np.savez_compressed(path / 'test_predictions.npz', **arrays)
    np.savez_compressed(path / 'test_scale_curves.npz', **curves)
    records = out.result_records([(row, path)])
    assert len(records) == 6
    supervised = [r for r in records if r['selection_protocol'] == out.SUPERVISED]
    assert {r['readout'] for r in supervised} == {'full', 'response'}
    assert all(r['selected_coordinate'] == 1. and r['true_lid_mean'] == 2. for r in supervised)
    automatic = [r for r in records if r['selection_protocol'] == out.POINTWISE]
    assert len(automatic) == 2 and all(r['selected_coordinate'] is None for r in automatic)
    assert {r['readout'] for r in automatic}=={'full','response'}
    assert automatic[0]['selected_lambda_counts']==automatic[1]['selected_lambda_counts']
    out.write_csv(tmp_path / 'results.csv', records)
    assert (tmp_path / 'results.csv').read_text().count(out.POINTWISE) == 2


def test_full_table_tracks_do_not_pool_generated_cells_or_historical_kneedle():
    _, inventory_cells = inventory()
    records = []
    for cell in inventory_cells:
        row = dict(model_variant='ve_diffusion', readout='full', cell_key=cell.key,
            inventory_origin='canonical_35' if cell.exact_archive else 'generated_e3_e4_extension_4',
            suite_id=cell.suite_id, mae=.5 if cell.exact_archive else 1000., true_lid_mean=2.,
            selection_status='selected', reference_mean=10., mean_delta_error=0.,
            n_source_train=dataset_spec(cell).expected_samples['train'])
        selectors = (out.SUPERVISED, out.POINTWISE, out.LEGACY) if cell.target_policy == 'known_lid' else (out.REFERENCE,)
        records.extend(dict(row, selection_protocol=s, mae=999. if s == out.LEGACY else row['mae']) for s in selectors)
    scores = out.table_scores(records, [vars(c) for c in inventory_cells])
    scores = [r for r in scores if r['model_variant'] == 've_diffusion' and r['readout']=='full']
    assert len(scores) == 12 and all(r['status'] == 'available' for r in scores)
    expected = dict(zip(out.SCORES, (.25, .25, .5, 0., .5, .5)))
    counts = dict(zip(out.SCORES, (7, 7, 1, 13, 4, 2)))
    for row in scores:
        assert row['value'] == pytest.approx(expected[row['score']])
        assert row['expected_cells'] == row['available_cells'] == counts[row['score']]
