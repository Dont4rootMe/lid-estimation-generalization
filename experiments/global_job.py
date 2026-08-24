"""Scheduler-facing source entrypoint for the resumable global campaign.

The ``pytorch2`` service treats the first token of ``script`` as a Python
source file and may append one distributed rank argument.  This shim removes
that scheduler-owned argument and replaces the bootstrap interpreter with the
pinned ``block-diff`` interpreter before starting the Hydra campaign module.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(
    "/home/jovyan/echimbulatov/fork_afedorov/constant_repos/"
    "lid-estimation-generalization"
)
BLOCK_DIFF_PYTHON = Path("/home/jovyan/.mlspace/envs/block-diff/bin/python")
GLOBAL_MODULE = "experiments.global_campaign"


def _without_scheduler_rank(arguments: list[str]) -> list[str]:
    """Remove the sole rank flag injected by legacy distributed launchers."""

    cleaned: list[str] = []
    rank_seen = False
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        rank_value: str | None = None
        if argument in ("--local-rank", "--local_rank"):
            if index + 1 >= len(arguments):
                raise SystemExit("scheduler local-rank flag has no value")
            rank_value = arguments[index + 1]
            index += 2
        elif argument.startswith(("--local-rank=", "--local_rank=")):
            rank_value = argument.split("=", 1)[1]
            index += 1
        else:
            cleaned.append(argument)
            index += 1
            continue

        if rank_seen or not rank_value.isdecimal():
            raise SystemExit("scheduler local-rank flag is invalid or duplicated")
        rank_seen = True
    return cleaned


def _exec_arguments(arguments: list[str]) -> list[str]:
    """Return the exact argv used after the scheduler's Python bootstrap."""

    return [
        str(BLOCK_DIFF_PYTHON),
        "-m",
        GLOBAL_MODULE,
        *_without_scheduler_rank(arguments),
    ]


def main() -> None:
    if os.environ.get("PROJECT_ROOT") != str(REPO_ROOT):
        raise SystemExit("PROJECT_ROOT does not match the approved experiment root")
    module_path = REPO_ROOT.joinpath(*GLOBAL_MODULE.split(".")).with_suffix(".py")
    if REPO_ROOT.is_symlink() or not REPO_ROOT.is_dir():
        raise SystemExit("the approved experiment root is unavailable or unsafe")
    if module_path.is_symlink() or not module_path.is_file():
        raise SystemExit("the approved global campaign module is unavailable or unsafe")
    if not BLOCK_DIFF_PYTHON.is_file():
        raise SystemExit("the pinned block-diff Python interpreter is unavailable")
    os.chdir(REPO_ROOT)
    os.execv(str(BLOCK_DIFF_PYTHON), _exec_arguments(sys.argv[1:]))


if __name__ == "__main__":
    main()
