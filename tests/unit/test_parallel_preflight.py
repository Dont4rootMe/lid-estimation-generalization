from collections import deque
from pathlib import Path
from queue import Empty
from types import SimpleNamespace

import pytest

from experiments import global_parallel as parallel


@pytest.mark.parametrize("reported_first", [True, False])
def test_preflight_accepts_clean_exit_and_delayed_reports(
    tmp_path, monkeypatch, reported_first
):
    records = [
        {
            "slot": slot,
            "visible_token": None,
            "cuda_required": False,
            "device_count": 0,
            "device_name": None,
        }
        for slot in range(2)
    ]
    messages = deque(
        [
            {"kind": "ok", "record": records[0]},
            Empty(),
            {"kind": "ok", "record": records[1]},
        ]
        if reported_first
        else [
            Empty(),
            {"kind": "ok", "record": records[0]},
            {"kind": "ok", "record": records[1]},
        ]
    )

    class ResultQueue:
        def get(self, timeout):
            value = messages.popleft()
            if isinstance(value, Exception):
                raise value
            return value

    class Process:
        exitcode = 0

        def __init__(self, **kwargs):
            self.name = kwargs["name"]

        def start(self):
            pass

        def join(self, timeout):
            pass

    context = SimpleNamespace(Queue=ResultQueue, Process=Process)
    monkeypatch.setattr(parallel.mp, "get_context", lambda _: context)
    layout = [
        {
            "visible_device_token": None,
            "physical_device_index": slot,
            "device_lane": 0,
            "multiplexed": False,
        }
        for slot in range(2)
    ]
    monkeypatch.setattr(
        parallel, "_worker_device_layout", lambda *args, **kwargs: layout
    )
    monkeypatch.setattr(
        parallel, "_execution_device_topology", lambda *args: (2, 2, 1, False)
    )
    monkeypatch.setattr(
        parallel, "_preflight_payload", lambda prepared, **kwargs: kwargs["records"]
    )
    campaign = SimpleNamespace(
        _write_json=lambda path, value: None, GlobalCampaignError=RuntimeError
    )
    monkeypatch.setattr(parallel, "_campaign_api", lambda: campaign)
    prepared = SimpleNamespace(
        config={"execution": {}},
        state_dir=str(tmp_path / "state"),
        campaign_root=str(tmp_path),
    )
    assert parallel._run_preflight_only(
        prepared,
        dependencies=parallel.ParallelDependencies(),
        require_cuda=False,
    ) == Path(tmp_path)
    assert not messages


@pytest.mark.parametrize("exitcode", [0, 1])
def test_preflight_still_rejects_missing_or_failed_workers(
    tmp_path, monkeypatch, exitcode
):
    class ResultQueue:
        def get(self, timeout):
            raise Empty()

    context = SimpleNamespace(
        Queue=ResultQueue,
        Process=lambda **kwargs: SimpleNamespace(
            name=kwargs["name"], exitcode=exitcode, start=lambda: None
        ),
    )
    monkeypatch.setattr(parallel.mp, "get_context", lambda _: context)
    monkeypatch.setattr(
        parallel,
        "_worker_device_layout",
        lambda *args, **kwargs: [
            {
                "visible_device_token": None,
                "physical_device_index": 0,
                "device_lane": 0,
                "multiplexed": False,
            }
        ],
    )
    monkeypatch.setattr(
        parallel, "_execution_device_topology", lambda *args: (1, 1, 1, False)
    )
    monkeypatch.setattr(parallel, "_terminate_workers", lambda processes: None)
    moments = iter([0, 1, 301])
    monkeypatch.setattr(parallel.time, "monotonic", lambda: next(moments))
    monkeypatch.setattr(
        parallel,
        "_campaign_api",
        lambda: SimpleNamespace(GlobalCampaignError=RuntimeError),
    )
    prepared = SimpleNamespace(
        config={"execution": {}}, state_dir=str(tmp_path), campaign_root=str(tmp_path)
    )
    with pytest.raises(
        RuntimeError, match="timed out" if exitcode == 0 else "unsuccessfully"
    ):
        parallel._run_preflight_only(
            prepared,
            dependencies=parallel.ParallelDependencies(),
            require_cuda=False,
        )
