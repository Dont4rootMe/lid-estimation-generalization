"""Public entry points for the versioned 429-cell VP/VE campaign."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

from experiments import global_campaign_v2 as campaign
from experiments import global_parallel as parallel

H100_PROFILE_V2: Mapping[str, Any] = {
    "profile": campaign.EXECUTION_PROFILE_H100,
    "strategy": campaign.EXECUTION_STRATEGY_CELL_DAG,
    "worker_count": campaign.H100_LOGICAL_WORKER_COUNT,
    "physical_device_count": campaign.H100_PHYSICAL_DEVICE_COUNT,
    "lanes_per_device": campaign.H100_LANES_PER_DEVICE,
    "training_batch_size_override": None,
    "evaluation_batch_size_override": 512,
}


@contextmanager
def _v2_api() -> Iterator[None]:
    key = parallel._CAMPAIGN_MODULE_ENV
    previous = os.environ.get(key)
    os.environ[key] = "experiments.global_campaign_v2"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def with_h100_v2_profile(hydra_config: Any) -> dict[str, Any]:
    value = campaign._plain(campaign._mapping(hydra_config, field="global v2 campaign"))
    value["execution"] = dict(H100_PROFILE_V2)
    return campaign.validate_global_campaign_config(value)


def prepare_global_v2_campaign(
    hydra_config: Any | None = None,
    *,
    root: Path | None = None,
    output_root: Path | None = None,
    dependencies: parallel.ParallelDependencies | None = None,
) -> parallel._PreparedCampaign:
    """Prepare exact identities and the shared root used by canary and full DAG."""

    selected = (
        campaign.compose_global_campaign_v2_config()
        if hydra_config is None
        else hydra_config
    )
    selected = with_h100_v2_profile(selected)
    deps = parallel.ParallelDependencies() if dependencies is None else dependencies
    with _v2_api():
        return parallel._prepare_campaign(
            selected, root=root, output_root=output_root, dependencies=deps
        )


def run_prepared_v2_cell(
    prepared: parallel._PreparedCampaign,
    *,
    model_variant: str,
    cell_key: str,
    dependencies: parallel.ParallelDependencies | None = None,
    event_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
    pre_seal_hook: Callable[[Mapping[str, Any], Path], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run, validate, seal and prune one exact production cell identity."""

    deps = parallel.ParallelDependencies() if dependencies is None else dependencies
    if pre_seal_hook is not None:
        if (
            deps.cell_diagnostics_fn is not None
            and deps.cell_diagnostics_fn is not pre_seal_hook
        ):
            raise campaign.GlobalCampaignError("two v2 pre-seal hooks were supplied")
        deps = replace(deps, cell_diagnostics_fn=pre_seal_hook)
    model_indices = [
        index
        for index, plan in enumerate(prepared.plans)
        if plan.variant_id == model_variant
    ]
    cell_indices = [
        index for index, cell in enumerate(prepared.cells) if cell.key == cell_key
    ]
    if len(model_indices) != 1 or len(cell_indices) != 1:
        raise campaign.GlobalCampaignError("requested v2 production cell is not unique")
    cell_index = cell_indices[0]
    cell = prepared.cells[cell_index]
    if cell.reference_dataset not in {None, cell.dataset}:
        raise campaign.GlobalCampaignError(
            "standalone v2 cell requires a previously sealed reference"
        )
    with _v2_api():
        return parallel._run_cell_task(
            prepared,
            model_index=model_indices[0],
            cell_index=cell_index,
            reference_summary=None,
            dependencies=deps,
            event_callback=event_callback,
        )


def run_global_parallel_v2_campaign(
    hydra_config: Any,
    *,
    root: Path | None = None,
    output_root: Path | None = None,
    dependencies: parallel.ParallelDependencies | None = None,
    require_cuda: bool | None = None,
    preflight_only: bool = False,
) -> Path:
    selected = with_h100_v2_profile(hydra_config)
    with _v2_api():
        return parallel.run_global_parallel_campaign(
            selected,
            root=root,
            output_root=output_root,
            dependencies=dependencies,
            require_cuda=require_cuda,
            preflight_only=preflight_only,
        )


def _hydra_main(arguments: Sequence[str] | None = None) -> None:
    argv = list(sys.argv[1:] if arguments is None else arguments)
    preflight_only = "--preflight-only" in argv
    output_root: Path | None = None
    if "--output-root" in argv:
        index = argv.index("--output-root")
        if index + 1 >= len(argv) or argv.count("--output-root") != 1:
            raise campaign.GlobalCampaignError("--output-root requires one path")
        output_root = Path(argv[index + 1])
        del argv[index : index + 2]
    overrides = tuple(value for value in argv if value != "--preflight-only")
    if any(value.startswith("output_root=") for value in overrides):
        raise campaign.GlobalCampaignError(
            "use --output-root so placement does not change campaign identity"
        )
    config = campaign.compose_global_campaign_v2_config(overrides)
    output = run_global_parallel_v2_campaign(
        config, output_root=output_root, preflight_only=preflight_only
    )
    print(output)


def main() -> None:
    _hydra_main()


if __name__ == "__main__":
    main()
