"""Artifact-selected Cartesian grid execution, local or shared NFS."""

from __future__ import annotations

import json
import shlex
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..artifacts.io import sha256_path
from ..artifacts.manifest import load_manifest, write_manifest_atomic
from ..evaluation.plans import ScenarioKey
from ..evaluation.selected import SelectedEvaluator
from ..grids.model import CartesianGridPlan
from ..grids.runner import load_grid_plan
from ..specs.common import canonical_json_bytes
from ..specs.scenario import ScenarioSpec
from .adapter import LocalProcessAdapter, ProcessAdapter, SSHProcessAdapter
from .collection import collect_grid_results, is_grid_shard_complete
from .grouped_grid import dispatch_grid_requests, execute_grid_plan
from .grouped_remote import GridWorkRequest
from .paths import remote_run_paths, validate_run_id
from .shared_nfs import (
    SharedRunLease,
    fetch_authoritative_run,
    grid_identity_sha256,
    package_identity_sha256,
    shared_run_lease,
    stage_run_directory_atomic,
)
from .site import RunnerConfig, SiteProfile


class ExecutionBackendError(RuntimeError):
    """A grid execution or resume invariant failed."""


@dataclass(frozen=True)
class ExecutionSummary:
    run_id: str
    scope: str
    status: str
    total_pools: int
    duration_seconds: float
    output_path: Path | None = None
    manifest_path: Path | None = None
    attempts_count: int = 1
    skipped_shards: int = 0
    executed_shards: int = 0


@dataclass(frozen=True, slots=True)
class _Preflight:
    manifest: dict[str, Any]
    selected: SelectedEvaluator
    scenario: ScenarioSpec
    plan: CartesianGridPlan


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _preflight(path: Path, repository: Path) -> _Preflight:
    if not repository.is_dir():
        raise ExecutionBackendError(f"repository directory not found: {repository}")
    manifest = load_manifest(path, expected_kind="grid")
    run_dir = path.parent
    artifacts = {
        str(item.get("path")): item for item in manifest.get("artifacts", ())
        if isinstance(item, Mapping)
    }
    for relative, kind in (
        ("evaluator_artifact/artifact.json", "evaluator_artifact_receipt"),
        ("evaluator_artifact/evaluator", "evaluator_binary"),
    ):
        descriptor, artifact = artifacts.get(relative), run_dir / relative
        if (not isinstance(descriptor, Mapping) or descriptor.get("kind") != kind
                or not artifact.is_file() or descriptor.get("sha256") != sha256_path(artifact)
                or descriptor.get("bytes") != artifact.stat().st_size):
            raise ExecutionBackendError(f"grid artifact does not match {relative}")
    try:
        selected = SelectedEvaluator.load(run_dir / "evaluator_artifact")
        resolved = manifest["resolved_spec"]
        scenario_raw = resolved["scenario"]
        scenario = ScenarioSpec.from_dict(scenario_raw)
        if canonical_json_bytes(scenario.to_dict()) != canonical_json_bytes(scenario_raw):
            raise ValueError("resolved scenario is not canonical")
        plan_raw = manifest["grid"]["plan"]
        raw_key = plan_raw["scenario_key"]
        scenario_key = ScenarioKey(raw_key["identity_json"].encode(), raw_key["sha256"]).validated()
        plan = load_grid_plan(manifest, selected_evaluator=selected, scenario=scenario_key)
    except Exception as exc:
        raise ExecutionBackendError(f"grid plan, scenario, or evaluator is invalid: {exc}") from exc
    policy = resolved.get("policy")
    identity = selected.policy_identity
    if (manifest.get("core") != selected.manifest_core(binary_override="evaluator_artifact/evaluator")
            or resolved.get("evaluator_artifact_selection") != selected.provenance
            or not isinstance(policy, Mapping) or policy.get("id") != identity["id"]
            or policy.get("source_sha256") != identity["source_sha256"]
            or policy.get("policy_abi") != identity["abi"]):
        raise ExecutionBackendError("grid evaluator, policy, or selection provenance differs")
    return _Preflight(manifest, selected, scenario, plan)


def _shards(pool_count: int, chunk_size: int) -> tuple[dict[str, Any], ...]:
    return tuple({
        "shard_id": f"shard_{index:06d}", "shard_index": index,
        "ranges": [[start, min(start + chunk_size, pool_count)]],
    } for index, start in enumerate(range(0, pool_count, chunk_size)))


def _validate_shards(values: Any, pool_count: int) -> tuple[dict[str, Any], ...]:
    if not isinstance(values, list) or not values:
        raise ExecutionBackendError("grid has no persisted shard assignments")
    result, cursor = [], 0
    for index, raw in enumerate(values):
        if not isinstance(raw, Mapping) or set(raw) != {"shard_id", "shard_index", "ranges"}:
            raise ExecutionBackendError("persisted grid shard fields are invalid")
        expected_id, ranges = f"shard_{index:06d}", raw["ranges"]
        if (raw["shard_id"] != expected_id or raw["shard_index"] != index
                or not isinstance(ranges, list) or ranges != [[cursor, ranges[0][1]]]):
            raise ExecutionBackendError("persisted grid shard identity or ranges are invalid")
        end = ranges[0][1]
        if isinstance(end, bool) or not isinstance(end, int) or end <= cursor or end > pool_count or end - cursor > 2048:
            raise ExecutionBackendError("persisted grid shard range is invalid")
        result.append(dict(raw))
        cursor = end
    if cursor != pool_count:
        raise ExecutionBackendError("persisted grid shards do not exactly cover the plan")
    widths = [item["ranges"][0][1] - item["ranges"][0][0] for item in result]
    if len(set(widths[:-1])) > 1 or widths[-1] > widths[0]:
        raise ExecutionBackendError("persisted grid shards are not canonical chunks")
    return tuple(result)


def _append_attempt(path: Path, attempt: Mapping[str, Any]) -> None:
    manifest = load_manifest(path, expected_kind="grid")
    manifest["attempt_history"].append(dict(attempt))
    manifest["updated_at"] = _now()
    write_manifest_atomic(path, manifest, expected_kind="grid")


class ExecutionBackend:
    """Run a compact Cartesian plan through one process per session group."""

    def __init__(self, site_profile: SiteProfile | None = None, *,
                 process_adapter: ProcessAdapter | None = None) -> None:
        self.site = site_profile or SiteProfile(
            name="local", site_type="local", description="Built-in local profile",
            runner=RunnerConfig(max_workers=1, worker_concurrency=10),
        )
        self.site.validate()
        self.adapter = process_adapter or LocalProcessAdapter()

    def _mounted(self, path: Path) -> bool:
        if self.site.cluster.transport != "shared_nfs":
            return False
        try:
            path.resolve().relative_to(Path(str(self.site.cluster.remote_run_root)).resolve())
            return True
        except ValueError:
            return False

    def _via_coordinator(self, path: Path, *, resume: bool, blades: Sequence[str],
                         chunk_size: int | None, lease: SharedRunLease) -> ExecutionSummary:
        initial = load_manifest(path, expected_kind="grid")
        identity, run_id = grid_identity_sha256(initial), validate_run_id(initial["run_id"])
        if resume:
            fetch_authoritative_run(lease, path.parent.parent)
            if grid_identity_sha256(load_manifest(path, expected_kind="grid")) != identity:
                raise ExecutionBackendError("remote resume authority differs from local grid")
        else:
            stage_run_directory_atomic(lease, path.parent)
        remote = remote_run_paths(run_id, remote_base=self.site.cluster.remote_base)
        command = (
            f"cd {shlex.quote(str(self.site.cluster.repository_root))} && "
            f"env FXSIM_RUN_LEASE_TOKEN={shlex.quote(lease.token)} "
            f"{shlex.quote(self.site.cluster.worker_command)} grid run "
            f"{shlex.quote(str(remote['manifest']))} --site {shlex.quote(self.site.name)}"
        )
        command += "".join(f" --blades {shlex.quote(blade)}" for blade in blades)
        command += " --resume" if resume else ""
        command += f" --chunk-size {chunk_size}" if chunk_size is not None else ""
        started = time.monotonic()
        result = SSHProcessAdapter(ssh_config=self.site.ssh, process_runner=self.adapter).run_ssh(
            self.site.cluster.coordinator, command, timeout=self.site.harness.timeout_seconds,
        )
        fetch_authoritative_run(lease, path.parent.parent)
        if not result.ok:
            raise ExecutionBackendError(f"shared-NFS coordinator failed: {result.stderr}")
        finished = load_manifest(path, expected_kind="grid")
        attempts, output = finished["attempt_history"], path.parent / "evaluation_table.npz"
        if not attempts or attempts[-1].get("status") != "succeeded" or not output.is_file():
            raise ExecutionBackendError("coordinator returned without complete grid artifacts")
        topology = attempts[-1]["topology_metadata"]
        return ExecutionSummary(
            run_id, "cluster", "succeeded", int(finished["grid"]["pool_count"]),
            time.monotonic() - started, output, path, len(attempts),
            int(topology["skipped_shards"]), int(topology["executed_shards"]),
        )

    def run_grid(self, manifest_path: Path | str, *, resume: bool = False,
                 blades: Sequence[str] | None = None, chunk_size: int | None = None,
                 repository: Path | None = None) -> ExecutionSummary:
        path = Path(manifest_path).resolve()
        if repository is None:
            raise ExecutionBackendError("grid execution requires explicit repository context")
        preflight = _preflight(path, Path(repository).resolve())
        active = tuple(blades or self.site.cluster.blades) if self.site.site_type == "ssh" or blades else ()
        if active:
            if self.site.cluster.transport != "shared_nfs":
                raise ExecutionBackendError("cluster grids require shared_nfs")
            self.site.validate_blades(active)
            with shared_run_lease(self.site, preflight.manifest["run_id"], adapter=self.adapter) as lease:
                if not self._mounted(path):
                    return self._via_coordinator(
                        path, resume=resume, blades=active, chunk_size=chunk_size, lease=lease,
                    )
                return self._run(path, preflight, Path(repository).resolve(), resume, chunk_size, active)
        return self._run(path, preflight, Path(repository).resolve(), resume, chunk_size, ())

    def _run(self, path: Path, preflight: _Preflight, repository: Path, resume: bool,
             chunk_size: int | None, blades: Sequence[str]) -> ExecutionSummary:
        manifest, plan, run_dir = preflight.manifest, preflight.plan, path.parent
        size = int(chunk_size if chunk_size is not None else self.site.harness.chunk_size)
        if size < 1 or size > 2048:
            raise ExecutionBackendError("grid chunk_size must be between 1 and 2048")
        persisted = manifest["grid"]["shards"]
        if persisted:
            descriptors = _validate_shards(persisted, plan.pool_count)
            if chunk_size is not None and descriptors[0]["ranges"][0][1] != size:
                raise ExecutionBackendError("resume chunk_size differs from persisted shards")
        else:
            if resume:
                raise ExecutionBackendError("resume requires persisted grid shards")
            descriptors = _shards(plan.pool_count, size)
            manifest["grid"]["shards"] = list(descriptors)
            write_manifest_atomic(path, manifest, expected_kind="grid")
        started, started_at = time.monotonic(), _now()
        attempt_id = len(manifest["attempt_history"]) + 1
        pending, skipped = [], 0
        for descriptor in descriptors:
            receipt = run_dir / "results" / f"{descriptor['shard_id']}.receipt.json"
            if resume and is_grid_shard_complete(manifest, plan, descriptor, receipt):
                skipped += 1
            else:
                pending.append(descriptor)
        scope, workers = ("cluster", 0) if blades else ("local", 0)
        try:
            if blades and pending:
                package_sha = package_identity_sha256(repository)
                projection_sha = manifest["resolved_spec"]["metric_projection"]["projection_sha256"]
                manifest_sha = sha256_path(path)
                assigned = {blade: tuple(pending[index::len(blades)]) for index, blade in enumerate(blades)}
                requests = {
                    blade: GridWorkRequest(
                        manifest["run_id"], f"attempt_{attempt_id:04d}_{blade}", blade,
                        manifest_sha, plan.plan_sha256, preflight.selected.artifact_sha256,
                        projection_sha, package_sha, self.site.runner.worker_concurrency, shards,
                    ).validated()
                    for blade, shards in assigned.items() if shards
                }
                workers = dispatch_grid_requests(
                    run_root=run_dir, requests_by_blade=requests, manifest=manifest,
                    plan=plan, repository=repository, site=self.site,
                    ssh=SSHProcessAdapter(ssh_config=self.site.ssh, process_runner=self.adapter),
                )
            elif pending:
                local_request = GridWorkRequest(
                    manifest["run_id"], f"attempt_{attempt_id:04d}_local", "local",
                    sha256_path(path), plan.plan_sha256, preflight.selected.artifact_sha256,
                    manifest["resolved_spec"]["metric_projection"]["projection_sha256"],
                    package_identity_sha256(repository), self.site.runner.worker_concurrency,
                    tuple(pending),
                ).validated()
                workers = execute_grid_plan(
                    manifest=manifest, plan=plan, selected=preflight.selected,
                    scenario=preflight.scenario, shards=pending, repository=repository,
                    work_dir=run_dir, blade="local", work_request_sha256=local_request.sha256,
                    evaluator_workers=self.site.runner.worker_concurrency,
                    timeout_seconds=self.site.harness.timeout_seconds,
                ).evaluator_sessions
            output = collect_grid_results(path, run_dir / "evaluation_table.npz")
        except BaseException as exc:  # noqa: BLE001
            _append_attempt(path, {
                "attempt_id": attempt_id, "attempt_uid": uuid.uuid4().hex[:12],
                "scope": scope, "started_at": started_at, "finished_at": _now(),
                "status": "failed", "exit_code": 1,
                "blade": self.site.cluster.coordinator if blades else "local",
                "topology_metadata": {"workers": workers, "shards": len(descriptors),
                    "skipped_shards": skipped, "executed_shards": len(pending)},
                "error_message": str(exc),
            })
            raise
        _append_attempt(path, {
            "attempt_id": attempt_id, "attempt_uid": uuid.uuid4().hex[:12],
            "scope": scope, "started_at": started_at, "finished_at": _now(),
            "status": "succeeded", "exit_code": 0,
            "blade": self.site.cluster.coordinator if blades else "local",
            "topology_metadata": {"workers": workers, "shards": len(descriptors),
                "skipped_shards": skipped, "executed_shards": len(pending)},
            "error_message": None,
        })
        return ExecutionSummary(
            manifest["run_id"], scope, "succeeded", plan.pool_count,
            time.monotonic() - started, output, path, attempt_id, skipped, len(pending),
        )


__all__ = ["ExecutionBackend", "ExecutionBackendError", "ExecutionSummary"]
