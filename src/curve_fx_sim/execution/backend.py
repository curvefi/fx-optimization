"""Unified local and cluster execution backend for grids and optimization.

Coordinates staging, block-cyclic sharding, persistent evaluator execution,
append-only attempt recording, resume shard skipping, and strict collection
across local workstation cores or remote SSH blades.
"""

from __future__ import annotations
from datetime import UTC, datetime
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import shlex
import time
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

from curve_fx_harness_client import EvaluatorClient
from curve_fx_harness_client.models import BatchResultFrame, CandidateSpec, ObservationSpec

from .adapter import LocalProcessAdapter, ProcessAdapter, SSHProcessAdapter
from ..artifacts.manifest import load_manifest, write_manifest_atomic
from ..artifacts.io import sha256_path
from ..evaluation.grouping import (
    LocalSessionGroupBinding,
    SessionGroup,
    bind_local_session_group,
)
from ..evaluation.selected import SelectedEvaluator
from ..evaluation.session import LocalSessionMaterialization
from ..specs.common import canonical_json_bytes
from ..specs.scenario import ScenarioSpec
from .collection import (
    CollectionError,
    _load_shard_records,
    _rows_checksum,
    collect_grid_results,
    group_request_set_sha256,
    grid_request_set_sha256,
    is_grouped_shard_complete,
    normalize_session_attestation,
    write_shard_result,
)
from .grouped import execute_local_groups
from .grouped_grid import dispatch_grouped_grid
from .evaluator_pool import EvaluatorRegistry
from .sharding import ShardAssignment, make_assignments, write_ranges_file
from .site import RunnerConfig, SiteProfile
from .shared_nfs import (
    SharedRunLease,
    fetch_authoritative_run,
    grid_identity_sha256,
    shared_run_lease,
    stage_local_file_immutable,
    stage_run_directory_atomic,
)
from .staging import WorkBundle, prepare_work_bundle, remote_run_paths, validate_run_id


class ExecutionBackendError(RuntimeError):
    """Raised when an execution or resume operation fails."""


def _reject_artifact_bound_grid(manifest: Mapping[str, Any]) -> None:
    resolved = manifest.get("resolved_spec")
    if not isinstance(resolved, Mapping):
        return
    compilation = resolved.get("candidate_compilation")
    if not isinstance(compilation, Mapping):
        return
    mode = compilation.get("mode")
    if mode not in (None, ""):
        raise ExecutionBackendError(
            "artifact-bound named grid execution is not implemented by ExecutionBackend "
            f"(candidate_compilation mode {mode!r})"
        )


_NAMED_GRID_MODE = "schema_grouped_v1"


@dataclass(frozen=True, slots=True)
class _NamedGridPreflight:
    manifest: dict[str, Any]
    selected: SelectedEvaluator
    scenario: ScenarioSpec
    points: tuple[Any, ...]
    groups: tuple[SessionGroup, ...]


def _artifact_selected_mode(manifest: Mapping[str, Any]) -> str | None:
    resolved = manifest.get("resolved_spec")
    compilation = resolved.get("candidate_compilation") if isinstance(resolved, Mapping) else None
    if not isinstance(compilation, Mapping):
        return None
    mode = compilation.get("mode")
    return str(mode) if mode not in (None, "") else None


def _required_artifact_files(manifest: Mapping[str, Any], run_dir: Path) -> None:
    required = {
        "evaluator_artifact/artifact.json": "evaluator_artifact_receipt",
        "evaluator_artifact/evaluator": "evaluator_binary",
    }
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Sequence) or isinstance(artifacts, (str, bytes)):
        raise ExecutionBackendError("named grid manifest artifacts must be an array")
    by_path = {
        str(item.get("path")): item for item in artifacts if isinstance(item, Mapping)
    }
    for relative, kind in required.items():
        descriptor = by_path.get(relative)
        path = run_dir / relative
        if not isinstance(descriptor, Mapping) or descriptor.get("kind") != kind:
            raise ExecutionBackendError(f"named grid manifest does not list {relative}")
        if (
            not path.is_file()
            or descriptor.get("sha256") != sha256_path(path)
            or descriptor.get("bytes") != path.stat().st_size
        ):
            raise ExecutionBackendError(f"named grid artifact receipt does not match {relative}")


def _preflight_named_grid(
    manifest_file: Path,
    *,
    repository: Path,
) -> _NamedGridPreflight:
    if not repository.is_dir():
        raise ExecutionBackendError(
            f"artifact-selected repository directory not found: {repository}"
        )
    manifest = load_manifest(manifest_file, expected_kind="grid")
    run_dir = manifest_file.parent
    _required_artifact_files(manifest, run_dir)
    try:
        selected = SelectedEvaluator.load(run_dir / "evaluator_artifact")
    except Exception as exc:
        raise ExecutionBackendError(f"run-local SelectedEvaluator verification failed: {exc}") from exc
    expected_core = selected.manifest_core(binary_override="evaluator_artifact/evaluator")
    if manifest.get("core") != expected_core:
        raise ExecutionBackendError("named grid core differs from its run-local evaluator")
    resolved = manifest.get("resolved_spec")
    if not isinstance(resolved, Mapping):
        raise ExecutionBackendError("named grid resolved_spec must be an object")
    if resolved.get("evaluator_artifact_selection") != selected.provenance:
        raise ExecutionBackendError("named grid selection provenance differs from its artifact")
    compilation = resolved.get("candidate_compilation")
    if not isinstance(compilation, Mapping) or compilation.get("mode") != _NAMED_GRID_MODE:
        raise ExecutionBackendError("named grid candidate compilation mode is unsupported")
    policy = resolved.get("policy")
    selected_policy = selected.policy_identity
    if (
        not isinstance(policy, Mapping)
        or policy.get("id") != selected_policy["id"]
        or policy.get("source_sha256") != selected_policy["source_sha256"]
        or policy.get("policy_abi") != selected_policy["abi"]
        or compilation.get("policy_id") != selected_policy["id"]
        or compilation.get("parameter_schema_sha256") != selected.parameter_schema_sha256
        or compilation.get("parameter_schema_version")
        != selected.compiler.schema.schema_version
    ):
        raise ExecutionBackendError("named grid policy or parameter schema differs from its artifact")
    raw_scenario = resolved.get("scenario")
    if not isinstance(raw_scenario, Mapping):
        raise ExecutionBackendError("named grid resolved scenario must be an object")
    try:
        from ..grids.runner import load_grouped_grid

        scenario = ScenarioSpec.from_dict(raw_scenario)
        if canonical_json_bytes(scenario.to_dict()) != canonical_json_bytes(raw_scenario):
            raise ValueError("resolved scenario is not canonical")
        points, groups = load_grouped_grid(
            manifest,
            parameter_schema=selected.compiler.schema,
            artifact_sha256=selected.artifact_sha256,
        )
    except Exception as exc:
        raise ExecutionBackendError(f"named grid candidate or scenario evidence is invalid: {exc}") from exc
    if any(
        point.evaluation is None
        or not point.evaluation.candidate.candidate_sha256
        for point in points
    ):
        raise ExecutionBackendError(
            "named grid point is missing canonical candidate JSON or SHA-256 attestation"
        )
    return _NamedGridPreflight(
        manifest=manifest,
        selected=selected,
        scenario=scenario,
        points=points,
        groups=groups,
    )


@dataclass(frozen=True)
class ExecutionSummary:
    """High-level outcome summary of an executed grid or optimization run."""

    run_id: str
    scope: str
    status: str  # "succeeded" | "failed"
    total_pools: int
    duration_seconds: float
    output_path: Path | None = None
    manifest_path: Path | None = None
    attempts_count: int = 1
    skipped_shards: int = 0
    executed_shards: int = 0
def now_utc_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _append_attempt(manifest_file: Path, manifest: dict[str, Any], attempt: Mapping[str, Any]) -> None:
    try:
        current = load_manifest(manifest_file, expected_kind=str(manifest.get("run_kind", "grid")))
    except FileNotFoundError:
        current = manifest
    except Exception as exc:
        raise ExecutionBackendError(f"cannot load manifest before recording attempt: {exc}") from exc
    history = current.setdefault("attempt_history", [])
    if not isinstance(history, list):
        raise ExecutionBackendError("manifest attempt_history is not an array")
    history.append(dict(attempt))
    current["updated_at"] = now_utc_iso()
    write_manifest_atomic(manifest_file, current, expected_kind=str(current.get("run_kind", "grid")))


def _attempt_recorded(manifest_file: Path, attempt_id: int) -> bool:
    try:
        current = load_manifest(manifest_file)
    except Exception:
        return False
    history = current.get("attempt_history")
    return isinstance(history, list) and any(
        isinstance(item, Mapping) and item.get("attempt_id") == attempt_id
        for item in history
    )



def _is_shard_complete(
    shard_file: Path,
    assignment: ShardAssignment,
    *,
    run_id: str,
    request_set_sha256: str,
    session_attestation: Mapping[str, str],
) -> bool:
    """Accept resume state only when its complete identity still matches."""
    if not shard_file.is_file():
        return False
    try:
        payload, records = _load_shard_records(shard_file)
    except Exception:  # noqa: BLE001
        return False
    expected_indices = [
        index for start, end in assignment.ranges for index in range(start, end)
    ]
    try:
        observed_attestation = normalize_session_attestation(
            payload.get("session_attestation"),
            expected_session_id=str(session_attestation["session_id"]),
        )
        observed_indices = [int(record["pool_index"]) for record in records]
    except (CollectionError, KeyError, TypeError, ValueError):
        return False
    return (
        payload.get("run_id") == run_id
        and payload.get("shard_id") == assignment.shard_id
        and payload.get("shard_index") == assignment.shard_index
        and payload.get("ranges") == [list(item) for item in assignment.ranges]
        and payload.get("row_count") == len(expected_indices)
        and len(records) == len(expected_indices)
        and observed_indices == expected_indices
        and payload.get("rows_sha256") == _rows_checksum(records)
        and payload.get("request_set_sha256") == request_set_sha256
        and observed_attestation == dict(session_attestation)
    )


@dataclass(frozen=True, slots=True)
class _GroupedShard:
    shard_id: str
    shard_index: int
    ranges: tuple[tuple[int, int], ...]
    total_pools: int
    session_group_id: str
    observation_id: str

    @property
    def ordinals(self) -> tuple[int, ...]:
        return tuple(
            value for start, end in self.ranges for value in range(start, end)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "shard_index": self.shard_index,
            "blade": "local",
            "ranges": [list(value) for value in self.ranges],
            "chunk_size": len(self.ordinals),
            "total_pools": self.total_pools,
            "assigned_pools": len(self.ordinals),
            "session_group_id": self.session_group_id,
            "observation_id": self.observation_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> _GroupedShard:
        return cls(
            str(value["shard_id"]),
            int(value["shard_index"]),
            tuple((int(item[0]), int(item[1])) for item in value["ranges"]),
            int(value["total_pools"]),
            str(value["session_group_id"]),
            str(value["observation_id"]),
        )


def _ordinal_ranges(ordinals: Sequence[int]) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    for value in ordinals:
        if ranges and ranges[-1][1] == value:
            ranges[-1] = (ranges[-1][0], value + 1)
        else:
            ranges.append((value, value + 1))
    return tuple(ranges)


def _make_grouped_shards(
    points: Sequence[Any], chunk_size: int
) -> tuple[_GroupedShard, ...]:
    partitions: dict[tuple[str, str], list[int]] = {}
    for point in points:
        partitions.setdefault(
            (point.session_group_id, point.evaluation.observation_key.sha256), []
        ).append(point.ordinal)
    shards: list[_GroupedShard] = []
    total = len(points)
    for (group_id, observation_id), ordinals in partitions.items():
        for start in range(0, len(ordinals), chunk_size):
            selected = ordinals[start : start + chunk_size]
            index = len(shards)
            shards.append(
                _GroupedShard(
                    f"shard_{index:03d}_{group_id[:8]}_{observation_id[:8]}",
                    index,
                    _ordinal_ranges(selected),
                    total,
                    group_id,
                    observation_id,
                )
            )
    return tuple(shards)


def _extract_candidates_for_ranges(manifest: Mapping[str, Any], ranges: Sequence[tuple[int, int]]) -> list[dict[str, Any]]:
    """Build requests exclusively from canonical manifest.grid.pools records."""
    grid_obj = manifest.get("grid")
    if not isinstance(grid_obj, Mapping):
        raise ExecutionBackendError("manifest has no grid section")
    pools = grid_obj.get("pools")
    if not isinstance(pools, list) or not pools:
        raise ExecutionBackendError("manifest grid.pools is missing or empty")

    candidates: list[dict[str, Any]] = []
    for start, end in ranges:
        if start < 0 or start >= end or end > len(pools):
            raise ExecutionBackendError(f"candidate range [{start}, {end}) is outside grid.pools")
        for idx in range(start, end):
            pool_data = pools[idx]
            if not isinstance(pool_data, Mapping):
                raise ExecutionBackendError(f"manifest grid.pools[{idx}] is not an object")
            candidate_id = pool_data.get("id")
            if not isinstance(candidate_id, str) or not candidate_id:
                raise ExecutionBackendError(f"manifest grid.pools[{idx}] has no candidate id")
            policy_params = pool_data.get("policy_params", ())
            if isinstance(policy_params, (str, bytes)) or not isinstance(policy_params, Sequence):
                raise ExecutionBackendError(f"manifest grid.pools[{idx}] has invalid policy_params")
            pool_overrides = pool_data.get("pool_overrides", {})
            if not isinstance(pool_overrides, Mapping):
                raise ExecutionBackendError(f"manifest grid.pools[{idx}] has invalid pool_overrides")
            candidates.append({
                "ordinal": idx,
                "candidate_id": candidate_id,
                "policy_params": list(policy_params),
                "pool_overrides": dict(pool_overrides),
            })
    return candidates


def _ensure_session_opened(
    client: EvaluatorClient,
    bundle: WorkBundle,
    blade: str,
) -> dict[str, str]:
    """Ensure client has opened an attested session using the work bundle."""
    session_id = f"sess_{bundle.run_id}_{blade}"
    is_remote = blade != "local" and bool(blade)
    open_request = {
        "session_id": session_id,
        "template_sha256": bundle.template_sha256,
        "manifest_sha256": (
            bundle.session_manifest_remote_sha256
            if is_remote
            else bundle.session_manifest_local_sha256
        ),
        "session_config": bundle.session_config,
    }
    open_request_sha256 = hashlib.sha256(
        json.dumps(open_request, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    current_session_id = getattr(client, "current_session_id", None)
    cached_attestation = getattr(client, "_curve_fx_session_attestation", None)
    cached_open_request_sha256 = getattr(client, "_curve_fx_open_request_sha256", None)
    if current_session_id:
        if current_session_id != session_id:
            raise ExecutionBackendError(
                f"evaluator client for {blade!r} already holds session {current_session_id!r}, "
                f"expected {session_id!r}"
            )
        if cached_attestation is None:
            raise ExecutionBackendError(
                f"evaluator client for {blade!r} has an open session without a cached SessionReady proof"
            )
        if cached_open_request_sha256 != open_request_sha256:
            raise ExecutionBackendError(
                f"evaluator client for {blade!r} has session {session_id!r} opened for different inputs"
            )
        try:
            return normalize_session_attestation(
                cached_attestation,
                expected_session_id=session_id,
            )
        except CollectionError as exc:
            raise ExecutionBackendError(f"invalid cached session attestation on {blade!r}: {exc}") from exc
    template_path = str(bundle.template_remote if is_remote and bundle.template_remote else (bundle.template_local or ""))
    manifest_path_str = str(
        bundle.session_manifest_remote
        if is_remote
        else bundle.session_manifest_local.as_posix()
    )
    ready = client.open_session(
        session_id=session_id,
        template_path=template_path,
        manifest_path=manifest_path_str,
        template_sha256=bundle.template_sha256,
        manifest_sha256=(
            bundle.session_manifest_remote_sha256
            if is_remote
            else bundle.session_manifest_local_sha256
        ),
        **bundle.session_config,
    )
    try:
        attestation = normalize_session_attestation(ready, expected_session_id=session_id)
    except CollectionError as exc:
        raise ExecutionBackendError(f"evaluator on {blade!r} returned an invalid SessionReady proof: {exc}") from exc
    observed_session_id = client.current_session_id
    if observed_session_id != session_id:
        raise ExecutionBackendError(
            f"evaluator on {blade!r} opened {observed_session_id!r}, expected {session_id!r}"
        )
    try:
        setattr(client, "_curve_fx_session_attestation", attestation)
        setattr(client, "_curve_fx_open_request_sha256", open_request_sha256)
    except (AttributeError, TypeError) as exc:
        raise ExecutionBackendError(
            f"evaluator client for {blade!r} cannot retain its SessionReady proof"
        ) from exc
    return attestation


def _ensure_policy_identity(
    client: EvaluatorClient,
    manifest: Mapping[str, Any],
    blade: str,
) -> None:
    """Fail closed when a compiled run meets a different evaluator policy."""
    core = manifest.get("core", {})
    expected = {
        key: core.get(key)
        for key in (
            "policy_id",
            "policy_source_sha256",
            "policy_abi",
            "policy_parameter_count",
        )
        if core.get(key) is not None
    }
    if not expected:
        return
    hello = client.hello
    if hello is None:
        raise ExecutionBackendError(f"evaluator on {blade!r} did not provide a hello identity")
    identity = hello.evaluator_identity
    for key, value in expected.items():
        actual = getattr(identity, key)
        if str(actual).lower() != str(value).lower():
            raise ExecutionBackendError(
                f"evaluator on {blade!r} reports {key}={actual!r}, expected {value!r}"
            )


class ExecutionBackend:
    """Unified execution backend for local workstation and remote cluster blades."""

    def __init__(
        self,
        site_profile: SiteProfile | None = None,
        *,
        process_adapter: ProcessAdapter | None = None,
        harness_binary: Path | str | None = None,
        client_factory: Callable[[str, str], EvaluatorClient] | None = None,
    ) -> None:
        self.site = site_profile or SiteProfile(
            name="local",
            site_type="local",
            description="Built-in local execution profile",
            runner=RunnerConfig(max_workers=1),
        )
        self.site.validate()
        self.adapter = process_adapter or LocalProcessAdapter()
        self.harness_binary = Path(harness_binary) if harness_binary else None
        binary_path: Path | PurePosixPath | str = self.harness_binary or self.site.harness.binary_name
        if self.site.site_type == "ssh" and self.site.harness.remote_binary_path is not None:
            binary_path = self.site.harness.remote_binary_path
        self.evaluators = EvaluatorRegistry(
            client_factory=client_factory,
            binary_path=binary_path,
            ssh_config=self.site.ssh,
            default_timeout=self.site.harness.timeout_seconds,
        )

    def _shared_run_is_mounted(self, path: Path) -> bool:
        if self.site.cluster.transport != "shared_nfs":
            return False
        try:
            path.resolve().relative_to(Path(str(self.site.cluster.remote_run_root)).resolve())
        except ValueError:
            return False
        return True

    @staticmethod
    def _require_ok(result: Any, action: str) -> None:
        if not result.ok:
            detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
            raise ExecutionBackendError(f"{action}: {detail}")

    def _materialize_shared_bundle(self, bundle: WorkBundle) -> None:
        """Place immutable inputs once in the NFS run namespace."""
        copies: list[tuple[Path, Path]] = [
            (
                bundle.session_manifest_local.parent / "session_manifest.remote.json",
                Path(str(bundle.session_manifest_remote)),
            )
        ]
        if bundle.template_local is not None and bundle.template_remote is not None:
            copies.append((bundle.template_local, Path(str(bundle.template_remote))))
        copies.extend(
            (entry.local_path, Path(str(entry.remote_path))) for entry in bundle.market_files
        )
        for source, destination in copies:
            if source.resolve() != destination.resolve():
                stage_local_file_immutable(source, destination)

    def _stage_remote_bundle(
        self,
        bundle: WorkBundle,
        blades: Sequence[str],
        *,
        include_shards: bool,
    ) -> None:
        if self.site.cluster.transport == "shared_nfs" and self._shared_run_is_mounted(bundle.manifest_local):
            self._materialize_shared_bundle(bundle)
            return
        if self.site.cluster.transport == "shared_nfs":
            raise ExecutionBackendError(
                "off-mount shared-NFS staging must use the leased coordinator run path"
            )
        ssh = SSHProcessAdapter(ssh_config=self.site.ssh, process_runner=self.adapter)
        paths = remote_run_paths(bundle.run_id, remote_base=self.site.cluster.remote_base)
        targets = (
            [self.site.cluster.coordinator]
            if self.site.cluster.transport == "shared_nfs"
            else list(dict.fromkeys(blades))
        )
        for blade in targets:
            self._require_ok(
                ssh.run_ssh(
                    blade,
                    "mkdir -p " + " ".join(
                        shlex.quote(str(paths[key]))
                        for key in ("root", "inputs", "shards", "results", "logs", "data")
                    ),
                ),
                f"failed to prepare run namespace on {blade}",
            )
            uploads: list[tuple[Path, PurePosixPath]] = [
                (bundle.manifest_local, paths["manifest"]),
                (
                    bundle.session_manifest_local.parent / "session_manifest.remote.json",
                    bundle.session_manifest_remote,
                ),
            ]
            if bundle.template_local is not None and bundle.template_remote is not None:
                uploads.append((bundle.template_local, bundle.template_remote))
            uploads.extend((entry.local_path, entry.remote_path) for entry in bundle.market_files)
            if include_shards and bundle.shards_local_dir is not None:
                uploads.append((bundle.shards_local_dir, paths["shards"]))
            for source, destination in uploads:
                self._require_ok(
                    ssh.rsync_upload(source, blade, str(destination)),
                    f"failed to stage {source.name} through {blade}",
                )

    def _run_grid_via_coordinator(
        self,
        manifest_file: Path,
        *,
        resume: bool,
        blades: Sequence[str] | None,
        chunk_size: int | None,
        lease: SharedRunLease,
    ) -> ExecutionSummary:
        """Stage a run once, then let the NFS coordinator own execution."""
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        run_id = validate_run_id(str(manifest.get("run_id", manifest_file.parent.name)))
        paths = remote_run_paths(run_id, remote_base=self.site.cluster.remote_base)
        coordinator = self.site.cluster.coordinator
        ssh = SSHProcessAdapter(ssh_config=self.site.ssh, process_runner=self.adapter)
        immutable_identity = grid_identity_sha256(manifest)
        if resume:
            fetch_authoritative_run(lease, manifest_file.parent.parent)
            remote_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            if grid_identity_sha256(remote_manifest) != immutable_identity:
                raise ExecutionBackendError(
                    "remote resume authority does not match the local immutable grid request"
                )
        else:
            stage_run_directory_atomic(lease, manifest_file.parent)
        worker = shlex.quote(self.site.cluster.worker_command)
        command = (
            f"cd {shlex.quote(str(self.site.cluster.repository_root))} && "
            f"env FXSIM_RUN_LEASE_TOKEN={shlex.quote(lease.token)} "
            f"{worker} grid run {shlex.quote(str(paths['manifest']))} "
            f"--site {shlex.quote(self.site.name)}"
        )
        for blade in blades or ():
            command += f" --blades {shlex.quote(blade)}"
        if resume:
            command += " --resume"
        if chunk_size is not None:
            command += f" --chunk-size {int(chunk_size)}"
        executed = ssh.run_ssh(
            coordinator, command, timeout=self.site.harness.timeout_seconds
        )
        if not executed.ok:
            try:
                fetch_authoritative_run(lease, manifest_file.parent.parent)
            except Exception as recovery_error:
                raise ExecutionBackendError(
                    f"shared-NFS grid execution failed on {coordinator}: {executed.stderr}; "
                    f"failed to recover its run journal: {recovery_error}"
                ) from recovery_error
            self._require_ok(
                executed, f"shared-NFS grid execution failed on {coordinator}"
            )
        fetch_authoritative_run(lease, manifest_file.parent.parent)
        finished = json.loads(manifest_file.read_text(encoding="utf-8"))
        attempts = finished.get("attempt_history", [])
        if not isinstance(attempts, list) or not attempts or attempts[-1].get("status") != "succeeded":
            raise ExecutionBackendError("remote grid command returned without a succeeded attempt receipt")
        topology = attempts[-1].get("topology_metadata", {})
        output = manifest_file.parent / "grid_results.npz"
        if not output.is_file():
            raise ExecutionBackendError("remote grid command did not publish grid_results.npz")
        return ExecutionSummary(
            run_id=run_id,
            scope="cluster",
            status="succeeded",
            total_pools=int(finished["grid"]["pool_count"]),
            duration_seconds=executed.duration_seconds,
            output_path=output,
            manifest_path=manifest_file,
            attempts_count=len(attempts),
            skipped_shards=int(topology.get("skipped_shards", 0)),
            executed_shards=int(topology.get("executed_shards", 0)),
        )

    def dispatch_candidates(
        self,
        manifest_or_bundle: Path | str | WorkBundle,
        candidates: Sequence[dict[str, Any]],
        *,
        policy_identity: str | None = None,
        blades: Sequence[str] | None = None,
        observation: dict[str, Any] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Dispatch a candidate batch across configured blades using persistent evaluators.

        Reused by grid and interactive workflows.
        """
        if not candidates:
            return []

        if isinstance(manifest_or_bundle, WorkBundle):
            bundle = manifest_or_bundle
        else:
            manifest_path = Path(manifest_or_bundle).resolve()
            with manifest_path.open("r", encoding="utf-8") as handle:
                _reject_artifact_bound_grid(json.load(handle))
            bundle = prepare_work_bundle(
                manifest_path,
                root=manifest_path.parent.parent,
                remote_base=self.site.cluster.remote_base,
            )

        with bundle.manifest_local.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        _reject_artifact_bound_grid(manifest)

        resolved = manifest.get("resolved_spec", {})
        resolved_policy = resolved.get("policy", {}) if isinstance(resolved, Mapping) else {}
        core = manifest.get("core", {})
        effective_policy_id = policy_identity or str(
            (resolved_policy.get("id") if isinstance(resolved_policy, Mapping) else None)
            or (core.get("policy_id") if isinstance(core, Mapping) else None)
            or "default"
        )

        is_cluster = self.site.site_type == "ssh" or bool(blades)
        active_blades = list(blades) if blades else list(self.site.cluster.blades)
        if is_cluster:
            self.site.validate_blades(active_blades)
        if not active_blades:
            active_blades = ["local"] if not is_cluster else ["blade-b1"]

        if is_cluster:
            self._stage_remote_bundle(bundle, active_blades, include_shards=False)

        # Chunk candidates across blades
        chunk_size = min(len(candidates), self.site.harness.chunk_size)
        if chunk_size <= 0:
            chunk_size = len(candidates)

        evaluated_results: list[dict[str, Any]] = []
        blade_idx = 0
        total_candidates = len(candidates)

        for i in range(0, total_candidates, chunk_size):
            chunk = candidates[i : i + chunk_size]
            blade = active_blades[blade_idx % len(active_blades)]
            blade_idx += 1

            client = self.evaluators.get_or_create(blade, effective_policy_id, bundle.run_id)
            if client is None:
                raise ExecutionBackendError(f"failed to obtain evaluator client for ({blade!r}, {effective_policy_id!r})")

            _ensure_policy_identity(client, manifest, blade)
            _ensure_session_opened(client, bundle, blade)
            try:
                results = self._evaluate_candidates_with_client(client, chunk)
                evaluated_results.extend(results)
                if progress_callback is not None:
                    progress_callback(len(evaluated_results), total_candidates)
            except Exception as exc:
                raise ExecutionBackendError(f"candidate batch evaluation failed on {blade}: {exc}") from exc

        return evaluated_results

    def run_grid(
        self,
        manifest_path: Path | str,
        *,
        resume: bool = False,
        blades: Sequence[str] | None = None,
        chunk_size: int | None = None,
        repository: Path | None = None,
    ) -> ExecutionSummary:
        manifest_file = Path(manifest_path).resolve()
        if not manifest_file.is_file():
            raise FileNotFoundError(f"manifest file not found at {manifest_file}")
        manifest = load_manifest(manifest_file, expected_kind="grid")
        named_mode = _artifact_selected_mode(manifest)
        if named_mode is not None:
            if named_mode != _NAMED_GRID_MODE:
                _reject_artifact_bound_grid(manifest)
            if repository is None:
                raise ExecutionBackendError(
                    "artifact-selected grid execution requires explicit repository context"
                )
            preflight = _preflight_named_grid(
                manifest_file, repository=Path(repository).resolve()
            )
            is_cluster = self.site.site_type == "ssh" or bool(blades)
            if is_cluster:
                if self.site.cluster.transport != "shared_nfs":
                    raise ExecutionBackendError("artifact-selected cluster grids require shared_nfs")
                active = list(blades) if blades else list(self.site.cluster.blades)
                self.site.validate_blades(active)
                run_id = validate_run_id(str(manifest["run_id"]))
                with shared_run_lease(self.site, run_id, adapter=self.adapter) as lease:
                    if not self._shared_run_is_mounted(manifest_file):
                        return self._run_grid_via_coordinator(
                            manifest_file, resume=resume, blades=active,
                            chunk_size=chunk_size, lease=lease,
                        )
                    return self._run_named_local_grid(
                        manifest_file, preflight, repository=Path(repository).resolve(),
                        resume=resume, chunk_size=chunk_size, cluster_blades=active,
                    )
            return self._run_named_local_grid(
                manifest_file,
                preflight,
                repository=Path(repository).resolve(),
                resume=resume,
                chunk_size=chunk_size,
            )
        if blades:
            self.site.validate_blades(blades)
        if self.site.cluster.transport != "shared_nfs":
            return self._run_grid_unlocked(
                manifest_file, resume=resume, blades=blades, chunk_size=chunk_size
            )
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        run_id = validate_run_id(str(manifest.get("run_id", manifest_file.parent.name)))
        with shared_run_lease(self.site, run_id, adapter=self.adapter) as lease:
            if self._shared_run_is_mounted(manifest_file) and not lease.inherited:
                raise ExecutionBackendError(
                    "shared-NFS grid runs must enter through the coordinator staging path"
                )
            return self._run_grid_unlocked(
                manifest_file,
                resume=resume,
                blades=blades,
                chunk_size=chunk_size,
                lease=lease,
            )

    def _run_named_local_grid(
        self,
        manifest_file: Path,
        preflight: _NamedGridPreflight,
        *,
        repository: Path,
        resume: bool,
        chunk_size: int | None,
        cluster_blades: Sequence[str] = (),
    ) -> ExecutionSummary:
        """Persist around the path-local grouped executor."""
        manifest = preflight.manifest
        run_dir = manifest_file.parent
        run_id = validate_run_id(str(manifest["run_id"]))
        is_cluster = bool(cluster_blades)
        pool_count = len(preflight.points)
        size = self.site.harness.chunk_size if chunk_size is None else int(chunk_size)
        if size <= 0:
            raise ExecutionBackendError(f"chunk_size must be >= 1, got {size}")

        persisted = manifest["grid"].get("shards")
        if resume and isinstance(persisted, list) and persisted:
            try:
                shards = tuple(_GroupedShard.from_dict(item) for item in persisted)
                indices = [ordinal for shard in shards for ordinal in shard.ordinals]
                by_ordinal = {point.ordinal: point for point in preflight.points}
                if (
                    len({shard.shard_id for shard in shards}) != len(shards)
                    or sorted(indices) != list(range(pool_count))
                    or any(
                        shard.total_pools != pool_count
                        or any(
                            by_ordinal[ordinal].session_group_id != shard.session_group_id
                            or by_ordinal[ordinal].evaluation.observation_key.sha256
                            != shard.observation_id
                            for ordinal in shard.ordinals
                        )
                        for shard in shards
                    )
                ):
                    raise ValueError("persisted grouped shards are not canonical")
            except (KeyError, TypeError, ValueError) as exc:
                raise ExecutionBackendError(
                    "manifest contains invalid persisted grouped shards"
                ) from exc
        else:
            shards = _make_grouped_shards(preflight.points, size)

        results_dir = run_dir / "results"
        shards_dir = run_dir / "shards"
        results_dir.mkdir(parents=True, exist_ok=True)
        shards_dir.mkdir(parents=True, exist_ok=True)
        for shard in shards:
            write_ranges_file(shards_dir / f"{shard.shard_id}.ranges", shard.ranges)
        descriptors = [
            shard.to_dict() | {"blade": cluster_blades[index % len(cluster_blades)]}
            if is_cluster else shard.to_dict()
            for index, shard in enumerate(shards)
        ]
        if resume and persisted != descriptors:
            raise ExecutionBackendError("persisted grouped shards differ from canonical assignment")
        manifest["grid"]["shards"] = descriptors
        manifest["scope"] = "cluster" if is_cluster else "local"
        if is_cluster:
            manifest.update(remote_base=str(self.site.cluster.remote_base),
                            remote_transport="shared_nfs",
                            remote_coordinator=self.site.cluster.coordinator)
        write_manifest_atomic(manifest_file, manifest, expected_kind="grid")

        start_time = time.monotonic()
        started_iso = now_utc_iso()
        attempt_id = len(manifest.get("attempt_history", [])) + 1
        skipped = 0
        pending: list[_GroupedShard] = []
        try:
            bindings: dict[str, LocalSessionGroupBinding] = {}
            if not is_cluster:
                for group in preflight.groups:
                    materialization = LocalSessionMaterialization.from_scenario(
                        preflight.scenario, repository=repository,
                        manifest_root=run_dir / "session_transport",
                        session_id=f"sess_{run_id}_{group.key.sha256[:12]}",
                    ).validated()
                    bindings[group.key.sha256] = bind_local_session_group(group, materialization)

            for shard in shards:
                output = results_dir / f"{shard.shard_id}.json"
                if resume and is_grouped_shard_complete(
                    manifest, shard.to_dict(), output
                ):
                    skipped += 1
                else:
                    pending.append(shard)

            requested_ordinals = sorted(
                ordinal for shard in pending for ordinal in shard.ordinals
            )
            by_ordinal = {point.ordinal: point for point in preflight.points}
            requested_ids = [
                by_ordinal[ordinal].evaluation.evaluation_id
                for ordinal in requested_ordinals
            ]
            if is_cluster:
                pending_by_blade = {
                    blade: [shard for shard, descriptor in zip(shards, descriptors, strict=True)
                            if shard in pending and descriptor["blade"] == blade]
                    for blade in cluster_blades
                }
                execution = dispatch_grouped_grid(
                    run_root=run_dir, run_id=run_id, selected=preflight.selected,
                    scenario=preflight.scenario, points=preflight.points,
                    pending_by_blade=pending_by_blade, repository=repository, site=self.site,
                    chunk_size=size, attempt_id=attempt_id,
                    ssh=SSHProcessAdapter(ssh_config=self.site.ssh, process_runner=self.adapter),
                )
            else:
                execution = execute_local_groups(
                    preflight.selected, preflight.groups,
                    lambda group: bindings[group.key.sha256], requested_ids,
                    work_dir=run_dir, chunk_size=size,
                    max_workers=self.site.runner.max_workers,
                )
            for shard in pending:
                attestation = (
                    execution.attestations_by_session_group_id[shard.session_group_id]
                    if is_cluster else
                    execution.receipts_by_session_group_id[shard.session_group_id].session_attestation
                )
                rows = [
                    execution.results_by_evaluation_id[
                        by_ordinal[ordinal].evaluation.evaluation_id
                    ].model_dump()
                    for ordinal in shard.ordinals
                ]
                request_sha256 = group_request_set_sha256(manifest, shard.ordinals, attestation)
                write_shard_result(
                    results_dir / f"{shard.shard_id}.json",
                    run_id=run_id,
                    shard_id=shard.shard_id,
                    shard_index=shard.shard_index,
                    ranges=shard.ranges,
                    rows=rows,
                    request_set_sha256=request_sha256,
                    session_attestation=attestation,
                )
            output = collect_grid_results(
                manifest_file, run_dir / "grid_results.npz", adapter=self.adapter
            )
            duration = time.monotonic() - start_time
        except BaseException as exc:  # noqa: BLE001
            _append_attempt(
                manifest_file,
                manifest,
                {
                    "attempt_id": attempt_id,
                    "attempt_uid": uuid.uuid4().hex[:12],
                    "scope": "cluster" if is_cluster else "local",
                    "started_at": started_iso,
                    "finished_at": now_utc_iso(),
                    "status": "failed",
                    "exit_code": 1,
                    "blade": self.site.cluster.coordinator if is_cluster else "local",
                    "topology_metadata": {"shards": len(shards)},
                    "error_message": str(exc),
                },
            )
            raise

        _append_attempt(
            manifest_file,
            manifest,
            {
                "attempt_id": attempt_id,
                "attempt_uid": uuid.uuid4().hex[:12],
                "scope": "cluster" if is_cluster else "local",
                "started_at": started_iso,
                "finished_at": now_utc_iso(),
                "status": "succeeded",
                "exit_code": 0,
                "blade": self.site.cluster.coordinator if is_cluster else "local",
                "topology_metadata": {
                    "workers": execution.requests if is_cluster else execution.workers,
                    "shards": len(shards),
                    "skipped_shards": skipped,
                    "executed_shards": len(pending),
                },
                "error_message": None,
            },
        )
        return ExecutionSummary(
            run_id=run_id,
            scope="cluster" if is_cluster else "local",
            status="succeeded",
            total_pools=pool_count,
            duration_seconds=duration,
            output_path=output,
            manifest_path=manifest_file,
            attempts_count=attempt_id,
            skipped_shards=skipped,
            executed_shards=len(pending),
        )

    def _run_grid_unlocked(
        self,
        manifest_path: Path | str,
        *,
        resume: bool = False,
        blades: Sequence[str] | None = None,
        chunk_size: int | None = None,
        lease: SharedRunLease | None = None,
    ) -> ExecutionSummary:
        """Execute a grid run partitioned across local workers or cluster blades."""
        manifest_file = Path(manifest_path).resolve()
        if not manifest_file.is_file():
            raise FileNotFoundError(f"manifest file not found at {manifest_file}")
        if (
            self.site.site_type == "ssh"
            and self.site.cluster.transport == "shared_nfs"
            and not self._shared_run_is_mounted(manifest_file)
        ):
            assert lease is not None
            return self._run_grid_via_coordinator(
                manifest_file,
                resume=resume,
                blades=blades,
                chunk_size=chunk_size,
                lease=lease,
            )
        with manifest_file.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        run_id = validate_run_id(str(manifest.get("run_id", manifest_file.parent.name)))
        run_dir = manifest_file.parent
        results_dir = run_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        grid_obj = manifest.get("grid")
        if not isinstance(grid_obj, Mapping):
            raise ExecutionBackendError("grid manifest has no grid section")
        pools = grid_obj.get("pools")
        pool_count = int(grid_obj.get("pool_count", len(pools) if isinstance(pools, list) else 0))
        if pool_count <= 0 or not isinstance(pools, list) or len(pools) != pool_count:
            raise ExecutionBackendError(f"manifest {manifest_file} declares invalid grid.pools")
        resolved = manifest.get("resolved_spec", {})
        policy = resolved.get("policy", {}) if isinstance(resolved, Mapping) else {}
        policy_identity = str(policy.get("id") if isinstance(policy, Mapping) and policy.get("id") else "default")
        start_time = time.monotonic()
        attempt_id = len(manifest.get("attempt_history", [])) + 1
        started_iso = now_utc_iso()
        is_cluster = self.site.site_type == "ssh" or bool(blades)
        active_blades = list(blades) if blades else list(self.site.cluster.blades)
        if not active_blades and is_cluster:
            active_blades = ["blade-b1"]
        persisted_shards = grid_obj.get("shards")
        if resume and isinstance(persisted_shards, list) and persisted_shards:
            try:
                assignments = [ShardAssignment.from_dict(item) for item in persisted_shards]
                covered: set[int] = set()
                if len({a.shard_id for a in assignments}) != len(assignments):
                    raise ValueError("duplicate shard_id")
                for assignment in assignments:
                    if assignment.total_pools != pool_count:
                        raise ValueError("persisted shard total_pools mismatch")
                    for start, end in assignment.ranges:
                        if start < 0 or start >= end or end > pool_count:
                            raise ValueError("persisted shard range outside pool_count")
                        indices = set(range(start, end))
                        if covered.intersection(indices):
                            raise ValueError("persisted shard ranges overlap")
                        covered.update(indices)
                if covered != set(range(pool_count)):
                    raise ValueError("persisted shard ranges do not cover pool_count")
            except (AttributeError, TypeError, ValueError, KeyError) as exc:
                raise ExecutionBackendError("manifest contains invalid persisted shard assignments") from exc
            is_cluster = str(manifest.get("scope", "local")) == "cluster"
        else:
            effective_chunk_size = self.site.harness.chunk_size if chunk_size is None else int(chunk_size)
            if effective_chunk_size <= 0:
                raise ExecutionBackendError(f"chunk_size must be >= 1, got {effective_chunk_size}")
            assignments = make_assignments(pool_count, active_blades if is_cluster else ["local"], chunk_size=effective_chunk_size, run_id=run_id)
        shards_dir = run_dir / "shards"
        shards_dir.mkdir(parents=True, exist_ok=True)
        for assignment in assignments:
            write_ranges_file(shards_dir / f"{assignment.shard_id}.ranges", assignment.ranges)
        manifest.setdefault("grid", {})["shards"] = [a.to_dict() for a in assignments]
        manifest["scope"] = "cluster" if is_cluster else "local"
        if is_cluster:
            manifest["remote_base"] = str(self.site.cluster.remote_base)
            manifest["remote_transport"] = self.site.cluster.transport
            manifest["remote_coordinator"] = self.site.cluster.coordinator
        write_manifest_atomic(manifest_file, manifest, expected_kind=str(manifest.get("run_kind", "grid")))
        try:
            bundle = prepare_work_bundle(
                manifest_file,
                root=manifest_file.parent.parent,
                remote_base=self.site.cluster.remote_base,
            )
        except Exception as exc:
            _append_attempt(
                manifest_file,
                manifest,
                {
                    "attempt_id": attempt_id,
                    "attempt_uid": uuid.uuid4().hex[:12],
                    "scope": "cluster" if is_cluster else "local",
                    "started_at": started_iso,
                    "finished_at": now_utc_iso(),
                    "status": "failed",
                    "exit_code": 1,
                    "blade": "coordinator" if is_cluster else "local",
                    "topology_metadata": {"shards": len(assignments)},
                    "error_message": f"work bundle preparation failed: {exc}",
                },
            )
            raise ExecutionBackendError(f"work bundle preparation failed: {exc}") from exc
        try:
            if is_cluster:
                return self._run_cluster_shards(
                    bundle,
                    manifest,
                    assignments,
                    policy_identity,
                    start_time,
                    attempt_id,
                    started_iso,
                    resume=resume,
                )
            return self._run_local_shards(
                bundle,
                manifest,
                assignments,
                policy_identity,
                start_time,
                attempt_id,
                started_iso,
                resume=resume,
            )
        except Exception as exc:
            if not _attempt_recorded(manifest_file, attempt_id):
                _append_attempt(
                    manifest_file,
                    manifest,
                    {
                        "attempt_id": attempt_id,
                        "attempt_uid": uuid.uuid4().hex[:12],
                        "scope": "cluster" if is_cluster else "local",
                        "started_at": started_iso,
                        "finished_at": now_utc_iso(),
                        "status": "failed",
                        "exit_code": 1,
                        "blade": "coordinator" if is_cluster else "local",
                        "topology_metadata": {"shards": len(assignments)},
                        "error_message": f"execution setup failed: {exc}",
                    },
                )
            raise
    def _run_local_shards(
        self,
        bundle: WorkBundle,
        manifest: dict[str, Any],
        assignments: Sequence[ShardAssignment],
        policy_identity: str,
        start_time: float,
        attempt_id: int,
        started_iso: str,
        *,
        resume: bool = False,
    ) -> ExecutionSummary:
        manifest_file = bundle.manifest_local
        run_dir = manifest_file.parent
        results_dir = run_dir / "results"
        run_id = bundle.run_id
        total_pools = len(manifest.get("grid", {}).get("pools", ()))
        max_workers = int(self.site.runner.max_workers)
        if max_workers < 1:
            raise ExecutionBackendError("runner.max_workers must be >= 1")

        # Each worker slot owns a persistent evaluator process. The logical
        # blade remains "local", so session IDs and attestation remain stable.
        skipped_count = 0
        request_set_sha256: str | None = None
        slot_count = min(max_workers, max(1, len(assignments)))
        clients: list[EvaluatorClient] = []
        attestations: list[dict[str, str]] = []
        for slot in range(slot_count):
            client = self.evaluators.get_or_create(
                "local", policy_identity, run_id, worker_slot=slot
            )
            if client is None:
                raise ExecutionBackendError(
                    f"failed to obtain evaluator client for ('local', {policy_identity!r}, slot {slot})"
                )
            _ensure_policy_identity(client, manifest, "local")
            attestation = _ensure_session_opened(client, bundle, "local")
            clients.append(client)
            attestations.append(attestation)
            digest = grid_request_set_sha256(manifest, attestation)
            if request_set_sha256 is None:
                request_set_sha256 = digest
            elif digest != request_set_sha256:
                raise ExecutionBackendError("local evaluators opened non-identical attested sessions")

        assert request_set_sha256 is not None
        pending = []
        for assignment in assignments:
            shard_out = results_dir / f"{assignment.shard_id}.json"
            if resume and _is_shard_complete(
                shard_out,
                assignment,
                run_id=run_id,
                request_set_sha256=request_set_sha256,
                session_attestation=attestation,
            ):
                skipped_count += 1
            else:
                pending.append(assignment)

        def execute_lane(slot: int) -> list[tuple[str, str]]:
            client = clients[slot]
            attestation = attestations[slot]
            failures: list[tuple[str, str]] = []
            # One lane owns one persistent evaluator, so its requests remain
            # serialized even when another lane finishes a shard early.
            for assignment in pending[slot::slot_count]:
                try:
                    candidates = _extract_candidates_for_ranges(manifest, assignment.ranges)
                    records = self._evaluate_candidates_with_client(client, candidates)
                    write_shard_result(
                        results_dir / f"{assignment.shard_id}.json",
                        run_id=run_id,
                        shard_id=assignment.shard_id,
                        shard_index=assignment.shard_index,
                        ranges=assignment.ranges,
                        rows=records,
                        request_set_sha256=request_set_sha256,
                        session_attestation=attestation,
                    )
                except Exception as exc:  # noqa: BLE001
                    failures.append(
                        (assignment.shard_id, f"shard {assignment.shard_id} evaluation failed: {exc}")
                    )
            return failures

        errors: list[str] = []
        if pending:
            with ThreadPoolExecutor(max_workers=slot_count) as executor:
                futures = [executor.submit(execute_lane, slot) for slot in range(slot_count)]
                failures_by_shard: dict[str, str] = {}
                for future in futures:
                    try:
                        for shard_id, message in future.result():
                            failures_by_shard[shard_id] = message
                    except Exception as exc:  # noqa: BLE001
                        errors.append(f"local worker failed: {exc}")
                errors.extend(
                    failures_by_shard[assignment.shard_id]
                    for assignment in pending
                    if assignment.shard_id in failures_by_shard
                )

        executed_count = len(pending)
        error_msg = "; ".join(errors) if errors else None
        exit_code = 1 if errors else 0
        duration = time.monotonic() - start_time
        status = "succeeded" if exit_code == 0 else "failed"
        attempt = {
            "attempt_id": attempt_id,
            "attempt_uid": uuid.uuid4().hex[:12],
            "scope": "local",
            "started_at": started_iso,
            "finished_at": now_utc_iso(),
            "status": status,
            "exit_code": exit_code,
            "blade": "local",
            "topology_metadata": {
                "workers": slot_count,
                "shards": len(assignments),
                "skipped_shards": skipped_count,
                "executed_shards": executed_count,
            },
            "error_message": error_msg,
        }
        if exit_code != 0:
            _append_attempt(manifest_file, manifest, attempt)
            raise ExecutionBackendError(error_msg or "local execution failed")
        try:
            output_file = collect_grid_results(manifest_file, run_dir / "grid_results.npz", adapter=self.adapter)
        except Exception as exc:
            attempt.update({"status": "failed", "exit_code": 1, "error_message": f"collection failed: {exc}"})
            _append_attempt(manifest_file, manifest, attempt)
            raise ExecutionBackendError(f"local collection failed: {exc}") from exc
        _append_attempt(manifest_file, manifest, attempt)
        return ExecutionSummary(
            run_id=run_id,
            scope="local",
            status=status,
            total_pools=total_pools,
            duration_seconds=duration,
            output_path=output_file,
            manifest_path=manifest_file,
            attempts_count=attempt_id,
            skipped_shards=skipped_count,
            executed_shards=executed_count,
        )

    def _run_cluster_shards(
        self,
        bundle: WorkBundle,
        manifest: dict[str, Any],
        assignments: Sequence[ShardAssignment],
        policy_identity: str,
        start_time: float,
        attempt_id: int,
        started_iso: str,
        *,
        resume: bool = False,
    ) -> ExecutionSummary:
        manifest_file = bundle.manifest_local
        run_dir = manifest_file.parent
        results_dir = run_dir / "results"
        run_id = bundle.run_id
        total_pools = len(manifest.get("grid", {}).get("pools", ()))
        active_blades = list(dict.fromkeys(assignment.blade for assignment in assignments))
        self._stage_remote_bundle(bundle, active_blades, include_shards=True)

        clients: dict[str, EvaluatorClient] = {}
        session_attestations: dict[str, dict[str, str]] = {}
        request_set_digests: dict[str, str] = {}
        for blade in active_blades:
            client = self.evaluators.get_or_create(blade, policy_identity, run_id)
            if client is None:
                raise ExecutionBackendError(
                    f"failed to obtain evaluator client for ({blade!r}, {policy_identity!r})"
                )
            _ensure_policy_identity(client, manifest, blade)
            attestation = _ensure_session_opened(client, bundle, blade)
            clients[blade] = client
            session_attestations[blade] = attestation
            request_set_digests[blade] = grid_request_set_sha256(manifest, attestation)
        if len(set(request_set_digests.values())) != 1:
            raise ExecutionBackendError("cluster evaluators opened non-identical attested sessions")

        pending_by_blade: dict[str, list[ShardAssignment]] = {blade: [] for blade in active_blades}
        skipped_count = 0
        for assignment in assignments:
            blade = assignment.blade
            if resume and _is_shard_complete(
                results_dir / f"{assignment.shard_id}.json",
                assignment,
                run_id=run_id,
                request_set_sha256=request_set_digests[blade],
                session_attestation=session_attestations[blade],
            ):
                skipped_count += 1
            else:
                pending_by_blade[blade].append(assignment)

        def execute_blade(blade: str) -> list[str]:
            failures: list[str] = []
            client = clients[blade]
            attestation = session_attestations[blade]
            request_digest = request_set_digests[blade]
            # A persistent evaluator process accepts one request at a time.
            # Keeping this loop per blade serializes only same-blade work while
            # allowing independent blades to overlap.
            for assignment in pending_by_blade[blade]:
                try:
                    candidates = _extract_candidates_for_ranges(manifest, assignment.ranges)
                    records = self._evaluate_candidates_with_client(client, candidates)
                    write_shard_result(
                        results_dir / f"{assignment.shard_id}.json",
                        run_id=run_id,
                        shard_id=assignment.shard_id,
                        shard_index=assignment.shard_index,
                        ranges=assignment.ranges,
                        rows=records,
                        request_set_sha256=request_digest,
                        session_attestation=attestation,
                    )
                except Exception as exc:  # noqa: BLE001
                    failures.append(
                        f"cluster shard {assignment.shard_id} failed on {blade}: {exc}"
                    )
            return failures

        errors: list[str] = []
        nonempty_blades = [blade for blade in active_blades if pending_by_blade[blade]]
        max_workers = int(self.site.runner.max_workers)
        if max_workers < 1:
            raise ExecutionBackendError("runner.max_workers must be >= 1")
        if nonempty_blades:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(nonempty_blades))) as executor:
                futures = {
                    blade: executor.submit(execute_blade, blade)
                    for blade in nonempty_blades
                }
                failures_by_blade: dict[str, list[str]] = {}
                for blade in nonempty_blades:
                    try:
                        failures_by_blade[blade] = futures[blade].result()
                    except Exception as exc:  # noqa: BLE001
                        failures_by_blade[blade] = [f"cluster worker {blade} failed: {exc}"]
            # Retain complete messages by walking canonical assignments.
            message_by_shard = {
                message.split(" failed ", 1)[0].removeprefix("cluster shard "): message
                for messages in failures_by_blade.values()
                for message in messages
            }
            errors.extend(
                message_by_shard[assignment.shard_id]
                for assignment in assignments
                if assignment.shard_id in message_by_shard
            )
            errors.extend(
                message
                for blade in nonempty_blades
                for message in failures_by_blade[blade]
                if message.startswith("cluster worker ")
            )

        executed_count = len(assignments) - skipped_count
        error_msg = "; ".join(errors) if errors else None
        exit_code = 1 if errors else 0
        duration = time.monotonic() - start_time
        status = "succeeded" if exit_code == 0 else "failed"
        coordinator = self.site.cluster.coordinator or (active_blades[0] if active_blades else "blade-b6")
        attempt = {
            "attempt_id": attempt_id,
            "attempt_uid": uuid.uuid4().hex[:12],
            "scope": "cluster",
            "started_at": started_iso,
            "finished_at": now_utc_iso(),
            "status": status,
            "exit_code": exit_code,
            "blade": coordinator,
            "topology_metadata": {
                "blades": active_blades,
                "workers": min(max_workers, len(nonempty_blades)) if nonempty_blades else 0,
                "shards": len(assignments),
                "skipped_shards": skipped_count,
                "executed_shards": executed_count,
            },
            "error_message": error_msg,
        }
        if exit_code != 0:
            _append_attempt(manifest_file, manifest, attempt)
            raise ExecutionBackendError(error_msg or "cluster execution failed")
        try:
            output_file = collect_grid_results(
                manifest_file,
                run_dir / "grid_results.npz",
                ssh_config=self.site.ssh,
                adapter=self.adapter,
            )
        except Exception as exc:
            attempt.update({"status": "failed", "exit_code": 1, "error_message": f"collection failed: {exc}"})
            _append_attempt(manifest_file, manifest, attempt)
            raise ExecutionBackendError(f"cluster collection failed: {exc}") from exc
        _append_attempt(manifest_file, manifest, attempt)
        return ExecutionSummary(
            run_id=run_id,
            scope="cluster",
            status=status,
            total_pools=total_pools,
            duration_seconds=duration,
            output_path=output_file,
            manifest_path=manifest_file,
            attempts_count=attempt_id,
            skipped_shards=skipped_count,
            executed_shards=executed_count,
        )

    def _evaluate_candidates_with_client(
        self,
        client: EvaluatorClient,
        candidates: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Evaluate one exact canonical candidate set and fail closed on drift.

        Batches are intentionally not split: the evaluator has no candidate
        count cap (only the 4 MiB frame guard), so one shard = one batch,
        sized by the caller's chunk_size for blade saturation.
        """
        request = [CandidateSpec.model_validate(candidate) for candidate in candidates]
        expected = [(candidate.ordinal, candidate.candidate_id) for candidate in request]
        if len({ordinal for ordinal, _ in expected}) != len(expected):
            raise ExecutionBackendError("candidate request contains duplicate ordinals")
        if len({candidate_id for _, candidate_id in expected}) != len(expected):
            raise ExecutionBackendError("candidate request contains duplicate candidate IDs")

        response = client.evaluate_batch(candidates=request)
        if not isinstance(response, BatchResultFrame):
            raise ExecutionBackendError(
                "canonical evaluator returned a non-BatchResultFrame response"
            )
        if response.status != "complete":
            raise ExecutionBackendError(
                f"evaluator batch status is {response.status!r}, expected 'complete'"
            )
        if not client.current_session_id:
            raise ExecutionBackendError("evaluator client has no active session after evaluation")
        if response.session_id != client.current_session_id:
            raise ExecutionBackendError(
                f"evaluator batch session {response.session_id!r} != active "
                f"session {client.current_session_id!r}"
            )

        expected_order = sorted(expected)
        actual_order = [
            (result.ordinal, result.candidate_id) for result in response.results
        ]
        if len({ordinal for ordinal, _ in actual_order}) != len(actual_order):
            raise ExecutionBackendError("evaluator batch returned duplicate ordinals")
        if len({candidate_id for _, candidate_id in actual_order}) != len(actual_order):
            raise ExecutionBackendError("evaluator batch returned duplicate candidate IDs")
        if actual_order != expected_order:
            raise ExecutionBackendError(
                "evaluator batch candidate coverage/order mismatch: "
                f"expected {expected_order!r}, got {actual_order!r}"
            )
        return [result.model_dump() for result in response.results]


    def close(self) -> None:
        """Close evaluator registry and release resources."""
        self.evaluators.close_all()

    def __enter__(self) -> ExecutionBackend:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


__all__ = [
    "ExecutionBackend",
    "ExecutionBackendError",
    "ExecutionSummary",
]
