"""Isolated launcher for the vendored LID-Benchmarks generator.

The pinned upstream code uses absolute ``generators`` imports and relative
paths for its base-dataset cache.  Importing it into this package would require
mutating ``sys.path`` and would make import resolution depend on the caller.
Instead, this module validates the imported snapshot and starts its own locked
environment in a subprocess whose working directory is ``lid_benchmarks``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
from typing import Mapping, Sequence

from utils.provenance import verify_upstream_source


_PREPARE_ALL_BOOTSTRAP = """\
import sys
from generate_datasets import prepare_all

prepare_all(sys.argv[1])
"""

DEFAULT_RUNTIME_COMMAND = ("uv", "run", "--frozen", "python")


class UpstreamGeneratorError(RuntimeError):
    """Raised when an upstream generator invocation cannot be constructed."""


@dataclass(frozen=True)
class UpstreamGeneratorInvocation:
    """A fully resolved, inspectable subprocess invocation."""

    cwd: Path
    output_dir: Path
    command: tuple[str, ...]
    upstream_revision: str | None


def prepare_upstream_generator_invocation(
    checkout: str | Path,
    output_dir: str | Path,
    *,
    runtime_command: Sequence[str] = DEFAULT_RUNTIME_COMMAND,
    verify_source: bool = True,
) -> UpstreamGeneratorInvocation:
    """Build an isolated invocation without importing upstream Python modules.

    ``runtime_command`` defaults to the upstream project's frozen ``uv``
    environment.  Passing ``(sys.executable,)`` is useful for controlled tests
    or for an environment that already contains all upstream dependencies.
    """

    source = Path(checkout).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if not source.is_dir():
        raise UpstreamGeneratorError(f"upstream source directory not found: {source}")
    if not (source / "generate_datasets.py").is_file():
        raise UpstreamGeneratorError(
            f"upstream entry point not found: {source / 'generate_datasets.py'}"
        )
    if not runtime_command or any(not str(part) for part in runtime_command):
        raise ValueError("runtime_command must contain at least one non-empty token")

    revision = verify_upstream_source(source) if verify_source else None
    command = tuple(str(part) for part in runtime_command) + (
        "-c",
        _PREPARE_ALL_BOOTSTRAP,
        str(destination),
    )
    return UpstreamGeneratorInvocation(
        cwd=source,
        output_dir=destination,
        command=command,
        upstream_revision=revision,
    )


def run_upstream_generator(
    checkout: str | Path,
    output_dir: str | Path,
    *,
    runtime_command: Sequence[str] = DEFAULT_RUNTIME_COMMAND,
    verify_source: bool = True,
    environment: Mapping[str, str] | None = None,
) -> UpstreamGeneratorInvocation:
    """Validate and run upstream generation, returning the exact invocation.

    Output is streamed to the caller.  ``PYTHONPATH`` is deliberately removed
    from the child environment so imports resolve only through the locked
    upstream environment and its working directory.
    """

    invocation = prepare_upstream_generator_invocation(
        checkout,
        output_dir,
        runtime_command=runtime_command,
        verify_source=verify_source,
    )
    child_environment = dict(os.environ)
    if environment is not None:
        child_environment.update(environment)
    child_environment.pop("PYTHONPATH", None)
    child_environment["PYTHONNOUSERSITE"] = "1"

    subprocess.run(
        invocation.command,
        cwd=invocation.cwd,
        env=child_environment,
        check=True,
    )
    return invocation


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> None:
    """Generate the audited fallback dataset from a source checkout."""

    repository_root = _repository_root()
    parser = argparse.ArgumentParser(
        description=(
            "Run the pinned LID-Benchmarks generator in its own frozen "
            "subprocess environment. This prepares data; experiments remain "
            "Hydra-only."
        )
    )
    parser.add_argument(
        "--checkout",
        type=Path,
        default=repository_root / "lid_benchmarks",
        help="path to the pinned top-level LID-Benchmarks import",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repository_root / "data" / "generated_benchmarks",
        help="dataset root passed to upstream prepare_all",
    )
    args = parser.parse_args(argv)
    invocation = run_upstream_generator(args.checkout, args.output)
    print(
        "generated fallback datasets at "
        f"{invocation.output_dir} from {invocation.upstream_revision}"
    )


if __name__ == "__main__":
    main()
