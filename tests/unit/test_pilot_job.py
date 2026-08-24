from __future__ import annotations

from experiments.pilot_job import (
    BLOCK_DIFF_PYTHON,
    PILOT_MODULE,
    _exec_arguments,
)


def test_scheduler_source_shim_execs_pinned_environment() -> None:
    overrides = ["pilot_model=diffusion", "seed=0"]
    assert _exec_arguments(overrides) == [
        str(BLOCK_DIFF_PYTHON),
        "-m",
        PILOT_MODULE,
        *overrides,
    ]
