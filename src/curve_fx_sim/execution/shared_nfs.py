from __future__ import annotations
from contextlib import contextmanager
import hashlib, json, os
from pathlib import Path, PurePosixPath; from typing import Any, Iterator, Mapping, Sequence
import secrets, shlex, shutil
from .adapter import ProcessAdapter, SSHProcessAdapter
from .site import SiteProfile
from .staging import sha256_path, validate_run_id
class SharedNFSError(RuntimeError): pass
def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
def grid_identity_sha256(manifest: Mapping[str, Any]) -> str:
    grid = manifest.get("grid")
    if not isinstance(grid, Mapping):
        raise SharedNFSError("grid manifest has no immutable grid authority")
    return canonical_sha256({"run_id": manifest.get("run_id"),
        "resolved_spec": manifest.get("resolved_spec"), "core": manifest.get("core"),
        "grid": {key: grid.get(key) for key in
                 ("grid_id", "pool_count", "resolved_axes", "pools")},
    })
class SharedRunLease:
    def __init__(self, site: SiteProfile, run_id: str, *, adapter: ProcessAdapter | None = None,
                 ssh_adapter: SSHProcessAdapter | None = None) -> None:
        site.validate()
        if site.cluster.transport != "shared_nfs":
            raise SharedNFSError("shared run leases require shared_nfs transport")
        self.site, self.run_id = site, validate_run_id(run_id)
        self.token = os.environ.get("FXSIM_RUN_LEASE_TOKEN") or secrets.token_hex(16)
        self.inherited = "FXSIM_RUN_LEASE_TOKEN" in os.environ
        self.ssh = ssh_adapter or SSHProcessAdapter(
            ssh_config=site.ssh, process_runner=adapter)
        self.path = site.cluster.remote_base / ".leases" / self.run_id
    def acquire(self) -> None:
        owner = self.path / "owner"
        if self.inherited:
            command = (f"test -f {shlex.quote(str(owner))} && "
                       f"test \"$(cat {shlex.quote(str(owner))})\" = "
                       f"{shlex.quote(self.token)}")
        else:
            command = ("umask 077; "
                       f"mkdir -p {shlex.quote(str(self.path.parent))} && "
                       f"mkdir {shlex.quote(str(self.path))} && "
                       f"printf '%s\\n' {shlex.quote(self.token)} > "
                       f"{shlex.quote(str(owner))}")
        result = self.ssh.run_ssh(self.site.cluster.coordinator, command)
        if not result.ok:
            raise SharedNFSError(
                f"run {self.run_id!r} is already owned or its lease is invalid: {result.stderr}")
    def release(self) -> None:
        if self.inherited:
            return
        owner = self.path / "owner"
        command = (f"test \"$(cat {shlex.quote(str(owner))})\" = "
                   f"{shlex.quote(self.token)} && rm -f {shlex.quote(str(owner))} && "
                   f"rmdir {shlex.quote(str(self.path))}")
        result = self.ssh.run_ssh(self.site.cluster.coordinator, command)
        if not result.ok:
            raise SharedNFSError(f"failed to release run lease {self.run_id!r}: {result.stderr}")
@contextmanager
def shared_run_lease(site: SiteProfile, run_id: str, *, adapter: ProcessAdapter | None = None,
                     ssh_adapter: SSHProcessAdapter | None = None) -> Iterator[SharedRunLease]:
    lease = SharedRunLease(site, run_id, adapter=adapter, ssh_adapter=ssh_adapter)
    lease.acquire()
    try:
        yield lease
    finally:
        lease.release()
def stage_run_directory_atomic(lease: SharedRunLease, local_run_dir: Path) -> PurePosixPath:
    if local_run_dir.name != lease.run_id:
        raise SharedNFSError("local run directory name does not match its lease")
    destination = lease.site.cluster.remote_run_root / lease.run_id
    staging = lease.site.cluster.remote_base / ".staging" / lease.token
    staged_run = staging / lease.run_id
    artifact = local_run_dir / "evaluator_artifact"
    inputs = [local_run_dir / "manifest.json"]
    if artifact.exists():
        inputs += [artifact / "artifact.json", artifact / "evaluator"]
        if not all(path.is_file() for path in inputs):
            raise SharedNFSError("grouped run requires exactly two evaluator artifact files")
    setup = lease.ssh.run_ssh(
        lease.site.cluster.coordinator,
        f"test ! -e {shlex.quote(str(destination))} && "
        f"mkdir -p {shlex.quote(str(staging.parent))} && "
        f"mkdir {shlex.quote(str(staging))} && mkdir {shlex.quote(str(staged_run))}" +
        (f" && mkdir {shlex.quote(str(staged_run / 'evaluator_artifact'))}" if len(inputs) == 3 else ""))
    if not setup.ok:
        raise SharedNFSError(
            f"non-resume run destination already exists or staging failed: {setup.stderr}")
    staged = [staged_run / path.relative_to(local_run_dir).as_posix() for path in inputs]
    for source, target in zip(inputs, staged, strict=True):
        uploaded = lease.ssh.rsync_upload(source, lease.site.cluster.coordinator, str(target))
        if not uploaded.ok: raise SharedNFSError(f"failed to stage immutable run inputs: {uploaded.stderr}")
    checks = " && ".join(f"test \"$(sha256sum {shlex.quote(str(target))} | cut -d' ' -f1)\" = {sha256_path(source)}"
                         for source, target in zip(inputs, staged, strict=True))
    publish = lease.ssh.run_ssh(
        lease.site.cluster.coordinator,
        f"{checks} && test ! -e {shlex.quote(str(destination))} && "
        f"mv {shlex.quote(str(staged_run))} {shlex.quote(str(destination))} && "
        f"rmdir {shlex.quote(str(staging))}")
    if not publish.ok:
        raise SharedNFSError(f"immutable run publication failed: {publish.stderr}")
    return destination
def fetch_authoritative_run(lease: SharedRunLease, local_runs_dir: Path) -> Path:
    remote = lease.site.cluster.remote_run_root / lease.run_id; checked = lease.ssh.run_ssh(
        lease.site.cluster.coordinator,
        f"test -f {shlex.quote(str(remote / 'manifest.json'))}")
    if not checked.ok:
        raise SharedNFSError("resume requires an existing authoritative remote manifest")
    result = lease.ssh.rsync_download(
        lease.site.cluster.coordinator, str(remote), local_runs_dir)
    if not result.ok: raise SharedNFSError(f"failed to fetch authoritative remote run: {result.stderr}")
    return local_runs_dir / lease.run_id
def stage_local_file_immutable(source: Path, destination: Path) -> None:
    expected = sha256_path(source); destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or sha256_path(destination) != expected:
            raise SharedNFSError(f"immutable shared input differs: {destination}")
        return
    temporary = destination.with_name(f".{destination.name}.{secrets.token_hex(12)}.tmp")
    shutil.copyfile(source, temporary)
    try:
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if not destination.is_file() or sha256_path(destination) != expected:
                raise SharedNFSError(
                    f"immutable shared input raced with different content: {destination}")
    finally:
        temporary.unlink(missing_ok=True)
def package_identity_sha256(root: Path) -> str:
    files = [p.relative_to(root) for p in (root / "src" / "curve_fx_sim").rglob("*.py")
             if p.is_file()]
    if (root / "policies").is_dir():
        files += [p.relative_to(root) for p in (root / "policies").rglob("*.hpp")
                  if p.is_file()]
    files += [Path(name) for name in ("pyproject.toml", "uv.lock", "scripts/bootstrap_blade.sh")
              if (root / name).is_file()]
    digest = hashlib.sha256()
    for relative in sorted(files, key=lambda item: item.as_posix()):
        digest.update(relative.as_posix().encode() + b"\0"); digest.update(
            bytes.fromhex(sha256_path(root / relative)))
    return digest.hexdigest()
def execution_site_payload(site: SiteProfile, blades: Sequence[str], *,
                           artifact_selected: bool = False) -> dict[str, Any]:
    payload = {
        "name": site.name, "transport": site.cluster.transport,
        "coordinator": site.cluster.coordinator, "blades": list(blades),
        "remote_base": str(site.cluster.remote_base), "remote_run_root": str(site.cluster.remote_run_root),
        "repository_root": str(site.cluster.repository_root), "worker_command": site.cluster.worker_command,
        "ssh_user": site.ssh.user, "ssh_port": site.ssh.port, "ssh_options": list(site.ssh.options),
    }
    key, value = (("evaluator_source", "run_local_artifact") if artifact_selected else
                  ("harness_binary", str(site.harness.remote_binary_path or site.harness.binary_name)))
    payload[key] = value
    return payload
