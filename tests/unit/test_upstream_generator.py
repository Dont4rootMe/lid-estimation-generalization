from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from utils.provenance import EXPECTED_LID_BENCHMARKS_SHA
from datasets.upstream_generator import (
    UpstreamGeneratorError,
    prepare_upstream_generator_invocation,
    run_upstream_generator,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_pinned_upstream_invocation_is_resolved_without_importing_generators() -> None:
    before = tuple(sys.path)
    generators_before = sys.modules.get("generators")
    checkout = REPOSITORY_ROOT / "lid_benchmarks"
    output = REPOSITORY_ROOT / "data" / "generated_benchmarks"

    invocation = prepare_upstream_generator_invocation(checkout, output)

    assert invocation.cwd == checkout.resolve()
    assert invocation.output_dir == output.resolve()
    assert invocation.upstream_revision == EXPECTED_LID_BENCHMARKS_SHA
    assert invocation.command[:4] == ("uv", "run", "--frozen", "python")
    assert invocation.command[-1] == str(output.resolve())
    assert tuple(sys.path) == before
    assert sys.modules.get("generators") is generators_before


def test_launcher_uses_upstream_cwd_and_scrubs_pythonpath(tmp_path: Path) -> None:
    checkout = tmp_path / "lid_benchmarks"
    generators = checkout / "generators"
    generators.mkdir(parents=True)
    (generators / "__init__.py").write_text("TOKEN = 'upstream-local'\n")
    (checkout / "generate_datasets.py").write_text(
        """
import json
import os
from pathlib import Path
from generators import TOKEN

def prepare_all(output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True)
    (output / "probe.json").write_text(json.dumps({
        "cwd": os.getcwd(),
        "pythonpath": os.environ.get("PYTHONPATH"),
        "python_no_user_site": os.environ.get("PYTHONNOUSERSITE"),
        "token": TOKEN,
    }))
""".lstrip()
    )
    output = tmp_path / "artifacts" / "generated"
    before = tuple(sys.path)
    generators_before = sys.modules.get("generators")

    invocation = run_upstream_generator(
        checkout,
        output,
        runtime_command=(sys.executable,),
        verify_source=False,
        environment={"PYTHONPATH": "/must/not/leak"},
    )

    probe = json.loads((output / "probe.json").read_text())
    assert invocation.cwd == checkout.resolve()
    assert probe == {
        "cwd": str(checkout.resolve()),
        "pythonpath": None,
        "python_no_user_site": "1",
        "token": "upstream-local",
    }
    assert tuple(sys.path) == before
    assert sys.modules.get("generators") is generators_before


def test_launcher_rejects_a_directory_without_upstream_entrypoint(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "empty"
    checkout.mkdir()

    with pytest.raises(UpstreamGeneratorError, match="entry point not found"):
        prepare_upstream_generator_invocation(
            checkout,
            tmp_path / "output",
            verify_source=False,
        )
