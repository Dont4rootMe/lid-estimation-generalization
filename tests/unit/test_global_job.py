from __future__ import annotations

import sys
from pathlib import Path

import pytest

from experiments import global_job
from experiments.global_job import (
    BLOCK_DIFF_PYTHON,
    GLOBAL_MODULE,
    _exec_arguments,
    _without_scheduler_rank,
)


def test_global_scheduler_source_shim_execs_pinned_environment() -> None:
    overrides = [
        "campaign=all_suites_all_models",
        "seed=0",
    ]
    assert _exec_arguments(overrides) == [
        str(BLOCK_DIFF_PYTHON),
        "-m",
        GLOBAL_MODULE,
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
def test_global_scheduler_rank_is_not_forwarded_to_hydra(
    rank_arguments: list[str],
) -> None:
    hydra_arguments = ["campaign=all_suites_all_models", "seed=0"]
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
def test_invalid_global_scheduler_rank_fails_closed(arguments: list[str]) -> None:
    with pytest.raises(SystemExit, match="local-rank"):
        _without_scheduler_rank(arguments)


def test_main_changes_to_approved_repo_before_exec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "approved-repo"
    module = repo / "experiments" / "global_campaign.py"
    module.parent.mkdir(parents=True)
    module.write_text("# approved test module\n", encoding="utf-8")
    python = tmp_path / "block-diff-python"
    python.write_text("", encoding="utf-8")
    events: list[tuple[str, object]] = []

    def fake_chdir(path: object) -> None:
        events.append(("chdir", path))

    def fake_execv(path: str, arguments: list[str]) -> None:
        events.append(("execv", (path, arguments)))
        raise RuntimeError("exec intercepted")

    monkeypatch.setattr(global_job, "REPO_ROOT", repo)
    monkeypatch.setattr(global_job, "BLOCK_DIFF_PYTHON", python)
    monkeypatch.setattr(global_job.os, "chdir", fake_chdir)
    monkeypatch.setattr(global_job.os, "execv", fake_execv)
    monkeypatch.setattr(sys, "argv", ["global_job.py", "seed=0"])
    monkeypatch.setenv("PROJECT_ROOT", str(repo))

    with pytest.raises(RuntimeError, match="exec intercepted"):
        global_job.main()

    assert events == [
        ("chdir", repo),
        (
            "execv",
            (
                str(python),
                [str(python), "-m", GLOBAL_MODULE, "seed=0"],
            ),
        ),
    ]
