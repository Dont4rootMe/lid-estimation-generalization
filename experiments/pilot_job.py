"""Scheduler-facing source entrypoint for the pilot experiment.

The ``pytorch2`` job service treats the first token of ``script`` as a Python
source file and wraps it with ``torch.distributed.launch``.  This tiny source
shim then replaces that bootstrap interpreter with the pinned ``block-diff``
interpreter before Hydra composes the actual experiment.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys


REPO_ROOT = Path(
    "/home/jovyan/echimbulatov/fork_afedorov/constant_repos/"
    "lid-estimation-generalization"
)
BLOCK_DIFF_PYTHON = Path("/home/jovyan/.mlspace/envs/block-diff/bin/python")
PILOT_MODULE = "experiments.pilot"


def _exec_arguments(arguments: list[str]) -> list[str]:
    """Return the exact argv used after the scheduler's Python bootstrap."""

    return [str(BLOCK_DIFF_PYTHON), "-m", PILOT_MODULE, *arguments]


def main() -> None:
    if os.environ.get("PROJECT_ROOT") != str(REPO_ROOT):
        raise SystemExit("PROJECT_ROOT does not match the approved experiment root")
    if not BLOCK_DIFF_PYTHON.is_file():
        raise SystemExit("the pinned block-diff Python interpreter is unavailable")
    os.execv(str(BLOCK_DIFF_PYTHON), _exec_arguments(sys.argv[1:]))


if __name__ == "__main__":
    main()
