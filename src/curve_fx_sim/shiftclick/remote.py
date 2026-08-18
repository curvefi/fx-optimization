"""Scoped SSH transport for exact one-candidate shiftclick replay."""

from __future__ import annotations

import json
import secrets
import shlex
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from ..analysis.economics import compare_economics
from ..artifacts.attestation import find_attested_artifact, verify_manifest_artifacts
from ..artifacts.manifest import load_manifest, write_manifest_atomic
from ..artifacts.store import RunStore
from ..evaluation.selection import load_attested_evaluation_table, normalize_selection
from ..execution.adapter import SSHProcessAdapter
from ..execution.site import SiteProfile
from ..execution.shared_nfs import shared_run_lease
from ..specs.shiftclick import ShiftclickSpec
from .runner import selection_from_spec

def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"remote shiftclick {label} must be an object")
    return value

def _evaluator_identity(core: Mapping[str, Any]) -> dict[str, Any]:
    """Remove deployment receipt fields from a validated evaluator identity."""
    return {
        key: value
        for key, value in core.items()
        if key not in {"site", "remote_sha256"}
    }


def _verify_remote_receipt(
    manifest: Mapping[str, Any],
    *,
    spec: ShiftclickSpec,
    store: RunStore,
    run_dir: Path,
) -> None:
    """Bind a downloaded replay to its attested source row and request."""
    source_manifest = store.load_manifest(spec.source_run_id)
    if source_manifest.get("run_kind") != spec.source_kind:
        raise RuntimeError(
            "remote shiftclick source kind does not match the requested source run"
        )
    source_table = load_attested_evaluation_table(
        source_manifest,
        store=store,
        run_id=spec.source_run_id,
    )

    selection = selection_from_spec(spec)
    expected_plan = normalize_selection(
        selection,
        store=store,
        observation_level="full_trace",
        trace_interval=spec.trace_interval,
        trace_actions=spec.trace_actions,
        evaluation_table=source_table,
    )
    if (
        expected_plan.pair_spec.id != spec.pair_id
        or expected_plan.scenario_spec.id != spec.scenario_id
        or expected_plan.policy_id != spec.policy_id
    ):
        raise RuntimeError("remote shiftclick spec identity does not match its source run")

    shiftclick = _require_mapping(manifest.get("shiftclick"), "manifest branch")
    expected_branch = {
        "shiftclick_id": spec.id,
        "source_run_id": spec.source_run_id,
        "selection": selection.to_dict(),
        "resolution": "full",
    }
    for key, expected in expected_branch.items():
        if shiftclick.get(key) != expected:
            raise RuntimeError(
                f"remote shiftclick {key} does not match the requested replay"
            )

    resolved = _require_mapping(manifest.get("resolved_spec"), "resolved_spec")
    observed_spec = dict(_require_mapping(resolved.get("shiftclick"), "resolved spec"))
    expected_spec = spec.to_dict()
    observed_spec.pop("source_spec_path", None)
    expected_spec.pop("source_spec_path", None)
    if observed_spec != expected_spec:
        raise RuntimeError("remote shiftclick resolved spec differs from the request")

    observed_plan = dict(_require_mapping(resolved.get("replay_plan"), "replay plan"))
    expected_plan_payload = expected_plan.to_dict()
    observed_plan.pop("artifact_dir", None)
    expected_plan_payload.pop("artifact_dir", None)
    if observed_plan != expected_plan_payload:
        raise RuntimeError("remote shiftclick replay plan differs from the source selection")

    source_core = _require_mapping(source_manifest.get("core"), "source core")
    observed_core = _require_mapping(manifest.get("core"), "core")
    if _evaluator_identity(observed_core) != _evaluator_identity(source_core):
        raise RuntimeError("remote shiftclick evaluator core identity differs from the source run")

    replay_path = find_attested_artifact(
        manifest,
        run_dir=run_dir,
        kind="replay_result",
    )
    comparison_path = find_attested_artifact(
        manifest,
        run_dir=run_dir,
        kind="economic_comparison",
    )
    try:
        replay = _require_mapping(
            json.loads(replay_path.read_text(encoding="utf-8")),
            "replay result",
        )
        comparison = _require_mapping(
            json.loads(comparison_path.read_text(encoding="utf-8")),
            "economic comparison",
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("remote shiftclick receipt JSON is invalid") from exc

    source_row = expected_plan.source_row
    replay_fingerprint = replay.get("economic_fingerprint")
    if not isinstance(replay_fingerprint, str):
        raise RuntimeError("remote shiftclick replay fingerprint is invalid")
    if (
        replay.get("status") != "ok"
        or replay.get("candidate_id") != source_row.candidate_id
        or replay.get("ordinal") != source_row.ordinal
    ):
        raise RuntimeError("remote shiftclick candidate identity does not match the source row")
    projection = source_table.metric_projection
    if projection is None:
        raise RuntimeError("source evaluation table has no MetricProjection")
    try:
        recomputed = compare_economics(
            source_row.metrics,
            _require_mapping(replay.get("metrics"), "replay metrics"),
            expected_fingerprint=source_row.economic_fingerprint or "",
            observed_fingerprint=replay_fingerprint,
            fields=projection.fields,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError("remote shiftclick economics do not match the source row") from exc
    if dict(comparison) != recomputed.to_dict():
        raise RuntimeError(
            "remote shiftclick economic comparison is not bound to the source row"
        )







@dataclass(frozen=True)
class RemoteShiftclickResult:
    blade: str
    run_dir: Path
    manifest: dict[str, object]
    remote_workspace: str

    def to_dict(self) -> dict[str, object]:
        return {
            "blade": self.blade,
            "run_dir": self.run_dir.as_posix(),
            "manifest": self.manifest,
            "remote_workspace": self.remote_workspace,
        }


def run_remote_shiftclick(
    spec: ShiftclickSpec,
    *,
    spec_path: Path,
    store: RunStore,
    site: SiteProfile,
    blade: str,
) -> RemoteShiftclickResult:
    site.validate()
    site.validate_blades((blade,))
    run_id = f"shiftclick_{spec.id}"
    ssh = SSHProcessAdapter(ssh_config=site.ssh)
    if site.cluster.transport == "shared_nfs":
        with shared_run_lease(site, run_id, ssh_adapter=ssh) as lease:
            return _run_remote_shiftclick_unlocked(
                spec,
                spec_path=spec_path,
                store=store,
                site=site,
                blade=blade,
                attempt_token=lease.token,
                ssh=ssh,
            )
    return _run_remote_shiftclick_unlocked(
        spec,
        spec_path=spec_path,
        store=store,
        site=site,
        blade=blade,
        attempt_token=secrets.token_hex(16),
        ssh=ssh,
    )


def _run_remote_shiftclick_unlocked(
    spec: ShiftclickSpec,
    *,
    spec_path: Path,
    store: RunStore,
    site: SiteProfile,
    blade: str,
    attempt_token: str,
    ssh: SSHProcessAdapter,
) -> RemoteShiftclickResult:
    """Stage one source run into an isolated remote workspace and replay it."""
    if site.site_type != "ssh":
        raise ValueError(f"remote shiftclick requires an SSH site, got {site.site_type!r}")
    if not spec.source_run_id:
        raise ValueError("remote shiftclick requires source_run_id")
    source_dir = store.get_run_dir(spec.source_run_id)
    run_id = f"shiftclick_{spec.id}"
    local_run_dir = store.runs_dir / run_id
    if local_run_dir.exists():
        raise FileExistsError(f"immutable shiftclick output already exists: {local_run_dir}")

    shared_mounted = False
    if site.cluster.transport == "shared_nfs":
        try:
            source_dir.resolve().relative_to(
                Path(str(site.cluster.remote_run_root)).resolve()
            )
            spec_path.resolve().relative_to(
                Path(str(site.cluster.repository_root)).resolve()
            )
            shared_mounted = True
        except ValueError:
            pass
    workspace = (
        site.cluster.remote_base / ".workspaces" / run_id / attempt_token
    )
    workspace_runs = workspace / "runs"
    remote_spec = (
        spec_path.resolve() if shared_mounted else workspace / "shiftclick.toml"
    )
    remote_repo = site.cluster.repository_root
    transfer_host = (
        site.cluster.coordinator
        if site.cluster.transport == "shared_nfs"
        else blade
    )
    runs_setup = (
        f"ln -s {shlex.quote(str(site.cluster.remote_run_root))} "
        f"{shlex.quote(str(workspace_runs))}"
        if shared_mounted
        else f"mkdir -p {shlex.quote(str(workspace_runs))}"
    )
    setup_command = (
        f"mkdir -p {shlex.quote(str(workspace.parent))} && "
        f"mkdir {shlex.quote(str(workspace))} && "
        f"{runs_setup} && "
        f"touch {shlex.quote(str(workspace / 'pyproject.toml'))} && "
        f"ln -s {shlex.quote(str(remote_repo / 'data'))} {shlex.quote(str(workspace / 'data'))} && "
        f"ln -s {shlex.quote(str(remote_repo / 'configs'))} {shlex.quote(str(workspace / 'configs'))}"
    )
    setup = ssh.run_ssh(transfer_host, setup_command)
    if not setup.ok:
        raise RuntimeError(
            f"failed to create remote shiftclick workspace on {transfer_host}: {setup.stderr}; "
            f"attempt workspace: {workspace}"
        )
    if not shared_mounted:
        source_upload = ssh.rsync_upload(source_dir, transfer_host, f"{workspace_runs}/")
        if not source_upload.ok:
            raise RuntimeError(
                f"failed to stage source run on {transfer_host}: {source_upload.stderr}; "
                f"retained workspace: {workspace}"
            )
        spec_upload = ssh.rsync_upload(spec_path.resolve(), transfer_host, str(remote_spec))
        if not spec_upload.ok:
            raise RuntimeError(
                f"failed to stage shiftclick spec on {transfer_host}: {spec_upload.stderr}; "
                f"retained workspace: {workspace}"
            )

    worker = shlex.quote(site.cluster.worker_command)
    harness = shlex.quote(str(site.harness.remote_binary_path or site.harness.binary_name))
    command = (
        f"cd {shlex.quote(str(workspace))} && "
        f"{worker} replay shiftclick {shlex.quote(str(remote_spec))} --harness {harness}"
    )
    executed = ssh.run_ssh(blade, command, timeout=site.harness.timeout_seconds)
    if not executed.ok:
        raise RuntimeError(
            f"remote shiftclick failed on {blade}: {executed.stderr}; "
            f"retained workspace: {workspace}"
        )

    if shared_mounted:
        if not local_run_dir.is_dir():
            raise RuntimeError("shared-NFS shiftclick did not publish its run directory")
    else:
        remote_result = workspace_runs / run_id
        downloaded = ssh.rsync_download(transfer_host, str(remote_result), store.runs_dir)
        if not downloaded.ok:
            raise RuntimeError(
                f"failed to collect remote shiftclick from {transfer_host}: {downloaded.stderr}; "
                f"retained workspace: {workspace}"
            )
    manifest_path = local_run_dir / "manifest.json"
    manifest = load_manifest(manifest_path, expected_kind="shiftclick")
    verify_manifest_artifacts(manifest, run_dir=local_run_dir)
    find_attested_artifact(manifest, run_dir=local_run_dir, kind="trace")
    if spec.trace_actions:
        find_attested_artifact(manifest, run_dir=local_run_dir, kind="actions")
    _verify_remote_receipt(
        manifest,
        spec=spec,
        store=store,
        run_dir=local_run_dir,
    )
    manifest["shiftclick"]["execution"] = {
        "scope": "cluster",
        "site": site.name,
        "blade": blade,
        "remote_workspace": str(workspace),
    }
    manifest["updated_at"] = (
        datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
    write_manifest_atomic(manifest_path, manifest, expected_kind="shiftclick")
    manifest = load_manifest(manifest_path, expected_kind="shiftclick")
    cleanup = ssh.run_ssh(
        transfer_host,
        f"rm -rf -- {shlex.quote(str(workspace))}",
    )
    if not cleanup.ok:
        raise RuntimeError(
            f"verified shiftclick result but failed to clean workspace {workspace}: {cleanup.stderr}"
        )
    return RemoteShiftclickResult(
        blade=blade,
        run_dir=local_run_dir,
        manifest=manifest,
        remote_workspace=str(workspace),
    )


__all__ = ["RemoteShiftclickResult", "run_remote_shiftclick"]
