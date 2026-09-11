import pytest


@pytest.fixture
def fixture_input_manifest(monkeypatch):
    """Synthetic manifests only in tests that explicitly request this fixture."""
    from experiments import fair_data
    value={'cells':{},'datasets':{}}
    monkeypatch.setattr(fair_data,'manifest',lambda:value)
    monkeypatch.setattr(fair_data,'contract_sha',lambda:'synthetic_input_fixture')
    return value
