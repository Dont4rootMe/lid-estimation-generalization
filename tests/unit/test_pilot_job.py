from __future__ import annotations

from experiments.pilot_job import (
    BLOCK_DIFF_PYTHON,
    PILOT_MODULE,
    _exec_arguments,
    _without_scheduler_rank,
)
import pytest


def test_scheduler_source_shim_execs_pinned_environment() -> None:
    overrides = ["pilot_model=diffusion", "seed=0"]
    assert _exec_arguments(overrides) == [
        str(BLOCK_DIFF_PYTHON),
        "-m",
        PILOT_MODULE,
        *overrides,
    ]


@pytest.mark.parametrize(
    "rank_arguments",
    [
        ["--local-rank=0"],
        ["--local_rank=0"],
        ["--local-rank", "0"],
        ["--local_rank", "0"],
    ],
)
def test_scheduler_rank_is_not_forwarded_to_hydra(
    rank_arguments: list[str],
) -> None:
    hydra_arguments = ["pilot_model=rectified_flow", "seed=0"]
    assert _without_scheduler_rank([*rank_arguments, *hydra_arguments]) == (
        hydra_arguments
    )


@pytest.mark.parametrize(
    "arguments",
    [
        ["--local-rank"],
        ["--local-rank=not-an-integer"],
        ["--local-rank=0", "--local_rank=0"],
    ],
)
def test_invalid_scheduler_rank_fails_closed(arguments: list[str]) -> None:
    with pytest.raises(SystemExit, match="local-rank"):
        _without_scheduler_rank(arguments)
