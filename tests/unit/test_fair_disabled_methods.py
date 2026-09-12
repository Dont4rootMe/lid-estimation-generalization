"""Disabled methods keep their implementation without entering the experiment."""
from types import SimpleNamespace

import pytest

from experiments import fair_campaign as campaign,fair_protocol as fair,fair_outputs as outputs
from models.pfgm_kernel import ClippedPFGMKernel


def test_disabled_pfgm_is_absent_from_matrix_and_score_coverage():
    assert 'pfgmpp' in fair.native_contracts()
    assert 'pfgmpp' not in fair.active_native_contracts()
    plan=campaign.matrix()
    assert plan['trainings']==429 and plan['native_interfaces']==11
    assert plan['implemented_interfaces']==12 and 'pfgmpp' in plan['disabled_variants']
    assert all(row['variant']!='pfgmpp' for row in plan['rows'])
    assert all(row['model_variant']!='pfgmpp' for row in outputs.table_scores([],[]))
    # The scientific implementation and exact mixed kernel have not been deleted.
    assert ClippedPFGMKernel(30).rms_factor()>0


def test_disabled_pfgm_fails_before_data_or_training(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('disabled method reached data/training')
    monkeypatch.setattr(campaign,'load_split',forbidden)
    monkeypatch.setattr(campaign.training,'train_model',forbidden)
    cell=campaign.inventory()[1][0]
    with pytest.raises(ValueError,match='pfgmpp is disabled'):
        campaign.run(SimpleNamespace(cell=cell.key,variant='pfgmpp'))
