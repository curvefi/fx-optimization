from pathlib import Path
from typing import Any, Mapping, Sequence

from ..evaluation.selected import SelectedEvaluator
from ..specs.scenario import ScenarioSpec
from .adapter import SSHProcessAdapter
from .grouped_dispatch import GroupedDispatch, dispatch_grouped_evaluations
from .site import SiteProfile


def dispatch_grouped_grid(
    *,
    run_root: Path, run_id: str, selected: SelectedEvaluator,
    scenario: ScenarioSpec, points: Sequence[Any],
    pending_by_blade: Mapping[str, Sequence[Any]], repository: Path,
    site: SiteProfile, chunk_size: int, attempt_id: int,
    ssh: SSHProcessAdapter,
) -> GroupedDispatch:
    """Dispatch pending grid points through grouped remote execution."""
    by_ordinal = {point.ordinal: point for point in points}
    ordinals = {
        blade: tuple(value for shard in shards for value in shard.ordinals)
        for blade, shards in pending_by_blade.items()
    }
    pending = {value for values in ordinals.values() for value in values}
    evaluations = tuple(point.evaluation for point in points if point.ordinal in pending)
    assignments = {
        blade: tuple(str(by_ordinal[value].evaluation.evaluation_id) for value in values)
        for blade, values in ordinals.items()}
    return dispatch_grouped_evaluations(
        run_root=run_root, run_id=run_id, selected=selected,
        evaluations=evaluations, scenarios=(scenario,),
        evaluation_ids_by_blade=assignments, repository=repository,
        site=site, chunk_size=chunk_size,
        lane_count=site.runner.worker_concurrency,
        request_namespace=f"attempt_{attempt_id:04d}", ssh=ssh)
