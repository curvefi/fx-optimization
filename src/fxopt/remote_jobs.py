"""Launch, inspect, stop, and retrieve detached cluster jobs."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from .config import ConfigError
from .placement import REMOTE_BASE, RSYNC_SSH, SSH_OPTIONS
from .results import ArtifactPaths
from .config import RunConfig
from .run import prepare_remote, stage_remote_run

REMOTE_JOB_FILENAME = ".remote-job.json"

@dataclass(frozen=True, slots=True)
class RemoteJob:
    """The minimal local handle for one detached coordinator process."""

    run_id: str
    coordinator: str
    remote_output: str


@dataclass(frozen=True, slots=True)
class RemoteRunStatus:
    """Current coordinator state plus its latest operator-facing log line."""

    state: str
    coordinator: str
    remote_output: str | None
    detail: str = ""
    exit_code: int | None = None


def _remote_paths(output_dir: str | Path) -> tuple[Path, ArtifactPaths, Path]:
    destination = Path(output_dir).expanduser().resolve()
    paths = ArtifactPaths(destination / "run.json", destination / "results.npz")
    return destination, paths, destination / REMOTE_JOB_FILENAME


def _local_run_complete(config: RunConfig, paths: ArtifactPaths) -> bool:
    if paths.run_json.is_file() and paths.results_npz.is_file():
        try:
            run_id = json.loads(paths.run_json.read_text())["run_id"]
        except (KeyError, OSError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("invalid local run metadata") from exc
        if run_id != config.run_id:
            raise RuntimeError("local results do not match the configured run")
        return True
    if paths.run_json.exists() or paths.results_npz.exists():
        raise RuntimeError("local result publication is incomplete")
    return False


def _validate_remote_output(remote_output: str) -> None:
    if (
        not remote_output.startswith("/tmp/fxopt-grid.")
        or any(character.isspace() for character in remote_output)
    ):
        raise RuntimeError("coordinator returned an invalid output directory")


def _write_remote_job(path: Path, job: RemoteJob) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps({
        "run_id": job.run_id,
        "coordinator": job.coordinator,
        "remote_output": job.remote_output,
    }, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _read_remote_job(config: RunConfig, path: Path) -> RemoteJob:
    try:
        payload = json.loads(path.read_text())
        values = (
            payload["run_id"],
            payload["coordinator"],
            payload["remote_output"],
        )
        if any(not isinstance(value, str) or not value for value in values):
            raise TypeError
        job = RemoteJob(
            run_id=values[0],
            coordinator=values[1],
            remote_output=values[2],
        )
    except (KeyError, OSError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid remote job record: {path}") from exc
    if job.run_id != config.run_id or job.coordinator != config.hosts[0]:
        raise RuntimeError("remote job record does not match the configured run")
    _validate_remote_output(job.remote_output)
    return job


def _start_remote_job(
    config: RunConfig,
    output_dir: str | Path,
    *,
    transfer: bool,
    rebuild: bool,
) -> RemoteJob:
    destination, paths, job_path = _remote_paths(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if paths.run_json.exists() or paths.results_npz.exists():
        raise FileExistsError(f"completed result already exists in {destination}")
    if job_path.exists():
        raise FileExistsError(
            f"remote job already exists in {destination}; "
            "use --status, --follow, --retrieve, or --stop"
        )

    prepare_remote(config, transfer=transfer, rebuild=rebuild)
    remote_config = stage_remote_run(config)
    coordinator = config.hosts[0]
    created = subprocess.run(
        [
            "ssh", *SSH_OPTIONS, "--", coordinator,
            "mktemp", "-d", "/tmp/fxopt-grid.XXXXXXXX",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    remote_output = created.stdout.strip()
    _validate_remote_output(remote_output)
    job = RemoteJob(config.run_id, coordinator, remote_output)
    _write_remote_job(job_path, job)

    optimizer_root = next(
        parent
        for parent in config.path.parents
        if parent.name == "curve-fx-optimization"
    )
    workspace = optimizer_root.parent.resolve()
    remote_project = str(REMOTE_BASE / "curve-fx-optimization")
    worker = [
        f"{remote_project}/scripts/cluster-python",
        "-m", "fxopt.cli", "_cluster-worker", remote_config,
        "--output", remote_output,
        "--origin-workspace", str(workspace),
        "--origin-config", str(config.path),
    ]
    remote_script = " ".join((
        "set -euo pipefail;",
        "export PYTHONPATH=" + shlex.quote(
            f"{remote_project}/src:"
            f"{REMOTE_BASE}/curve-fx-arb-harness/python/src:"
            f"{remote_project}/.venv-cluster/lib/python3.12/site-packages"
        ) + ";",
        "export PYTHONNOUSERSITE=1;",
        "exec " + shlex.join(worker),
    ))
    remote_command = (
        f"nix-shell {shlex.quote(f'{remote_project}/shell.nix')} "
        f"--run {shlex.quote(remote_script)}"
    )
    log_path = f"{remote_output}/coordinator.log"
    exit_path = f"{remote_output}/exit_code"
    pid_path = f"{remote_output}/pid"
    wrapped = " ".join((
        f"{remote_command};",
        "status=$?;",
        f"printf '%s\\n' \"$status\" > {shlex.quote(exit_path)};",
        "exit \"$status\"",
    ))
    launch = " ".join((
        f": > {shlex.quote(log_path)};",
        f"nohup setsid /bin/sh -c {shlex.quote(wrapped)} </dev/null "
        f">>{shlex.quote(log_path)} 2>&1 &",
        f"printf '%s\\n' \"$!\" > {shlex.quote(pid_path)}",
    ))
    try:
        subprocess.run(
            ["ssh", *SSH_OPTIONS, "--", coordinator, launch],
            check=True,
        )
    except BaseException:
        print(
            f"cluster: launch state retained in {job_path}",
            file=sys.stderr,
        )
        raise
    print(
        f"cluster: started detached on {coordinator}:{remote_output}",
        file=sys.stderr,
    )
    return job


def remote_run_status(
    config_path: str | Path,
    output_dir: str | Path,
) -> RemoteRunStatus:
    """Read one detached coordinator's state without changing it."""
    config = RunConfig.from_toml(config_path)
    if not config.hosts:
        raise ConfigError("remote status requires placement hosts")
    _destination, paths, job_path = _remote_paths(output_dir)
    if _local_run_complete(config, paths):
        return RemoteRunStatus("retrieved", config.hosts[0], None)
    if not job_path.is_file():
        raise FileNotFoundError(f"remote job record not found: {job_path}")
    job = _read_remote_job(config, job_path)
    script = " ".join((
        f"work={shlex.quote(job.remote_output)};",
        "state=stopped; code=;",
        "if ! test -d \"$work\"; then state=missing;",
        "elif test -s \"$work/run.json\" && test -s \"$work/results.npz\"; "
        "then state=complete; code=0;",
        "elif test -s \"$work/stopped\"; then state=stopped;",
        "elif test -s \"$work/exit_code\"; then "
        "code=$(cat \"$work/exit_code\"); state=failed;",
        "elif test -s \"$work/pid\" && "
        "kill -0 \"$(cat \"$work/pid\")\" 2>/dev/null; then state=running;",
        "fi;",
        "printf '%s\\n%s\\n' \"$state\" \"$code\";",
        "tail -n 1 \"$work/coordinator.log\" 2>/dev/null || true",
    ))
    checked = subprocess.run(
        ["ssh", *SSH_OPTIONS, "--", job.coordinator, script],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = checked.stdout.splitlines()
    if len(lines) < 2 or lines[0] not in {
        "running", "complete", "failed", "stopped", "missing"
    }:
        raise RuntimeError("coordinator returned an invalid job status")
    try:
        exit_code = int(lines[1]) if lines[1] else None
    except ValueError as exc:
        raise RuntimeError("coordinator returned an invalid exit code") from exc
    return RemoteRunStatus(
        lines[0],
        job.coordinator,
        job.remote_output,
        detail=lines[2] if len(lines) > 2 else "",
        exit_code=exit_code,
    )


def stop_remote_run(
    config_path: str | Path,
    output_dir: str | Path,
) -> RemoteRunStatus:
    """Stop one detached coordinator while retaining its diagnostic state."""
    config = RunConfig.from_toml(config_path)
    if not config.hosts:
        raise ConfigError("remote stop requires placement hosts")
    destination, paths, job_path = _remote_paths(output_dir)
    if _local_run_complete(config, paths):
        raise RuntimeError("remote job was already retrieved")
    status = remote_run_status(config.path, destination)
    if status.state == "complete":
        raise RuntimeError("remote job is complete; use --retrieve")
    if status.state != "running":
        return status

    job = _read_remote_job(config, job_path)
    script = " ".join((
        "set -eu;",
        f"work={shlex.quote(job.remote_output)};",
        "pid=$(cat \"$work/pid\");",
        "case \"$pid\" in ''|*[!0-9]*) exit 2;; esac;",
        "if test -s \"$work/run.json\" && test -s \"$work/results.npz\"; "
        "then printf 'complete\\n'; exit 0; fi;",
        "signal_tree() { signal=$1; current=$2; "
        "for child in $(cat \"/proc/$current/task/$current/children\" "
        "2>/dev/null || true); do signal_tree \"$signal\" \"$child\"; done; "
        "kill -\"$signal\" \"$current\" 2>/dev/null || true; };",
        "pgid=$(ps -o pgid= -p \"$pid\" 2>/dev/null | tr -d ' ' || true);",
        "if test \"$pgid\" = \"$pid\"; then "
        "kill -TERM -- \"-$pid\" 2>/dev/null || true; "
        "else signal_tree TERM \"$pid\"; fi;",
        "attempt=0; while kill -0 \"$pid\" 2>/dev/null && "
        "test \"$attempt\" -lt 20; do sleep 0.25; attempt=$((attempt + 1)); done;",
        "if kill -0 \"$pid\" 2>/dev/null; then "
        "if test \"$pgid\" = \"$pid\"; then "
        "kill -KILL -- \"-$pid\" 2>/dev/null || true; "
        "else signal_tree KILL \"$pid\"; fi; fi;",
        "printf 'operator-stop\\n' > \"$work/stopped\";",
        "printf 'stopped\\n'",
    ))
    stopped = subprocess.run(
        ["ssh", *SSH_OPTIONS, "--", job.coordinator, script],
        capture_output=True,
        text=True,
        check=True,
    )
    state = stopped.stdout.strip()
    if state == "complete":
        raise RuntimeError(
            "remote job completed before it could be stopped; use --retrieve"
        )
    if state != "stopped":
        raise RuntimeError("coordinator returned an invalid stop result")
    return RemoteRunStatus(
        "stopped",
        job.coordinator,
        job.remote_output,
        detail=status.detail,
    )


def _retrieve_remote_job(
    destination: Path,
    paths: ArtifactPaths,
    job_path: Path,
    job: RemoteJob,
) -> ArtifactPaths:
    incoming = destination / ".fetch"
    shutil.rmtree(incoming, ignore_errors=True)
    incoming.mkdir(parents=True)
    print("cluster: fetching final results...", file=sys.stderr)
    fetch = [
        "rsync", "-a", "--partial",
        "--include=/run.json", "--include=/results.npz", "--exclude=*",
        "-e", RSYNC_SSH, "--",
        f"{job.coordinator}:{job.remote_output}/", f"{incoming}/",
    ]
    for attempt in range(1, 4):
        fetched = subprocess.run(fetch, check=False)
        if fetched.returncode == 0:
            break
        if attempt < 3:
            print(f"cluster: fetch retry {attempt + 1}/3...", file=sys.stderr)
    else:
        raise subprocess.CalledProcessError(fetched.returncode, fetch)
    incoming_run = incoming / "run.json"
    incoming_results = incoming / "results.npz"
    if not incoming_run.is_file() or not incoming_results.is_file():
        raise RuntimeError("coordinator did not return both result artifacts")
    os.replace(incoming_results, paths.results_npz)
    try:
        os.replace(incoming_run, paths.run_json)
    except BaseException:
        paths.results_npz.unlink(missing_ok=True)
        raise
    incoming.rmdir()
    job_path.unlink()
    cleanup = subprocess.run(
        [
            "ssh", *SSH_OPTIONS, "--", job.coordinator,
            "rm", "-rf", "--", job.remote_output,
        ],
        check=False,
    )
    if cleanup.returncode != 0:
        print(
            f"cluster: warning: could not remove "
            f"{job.coordinator}:{job.remote_output}",
            file=sys.stderr,
        )
    return paths


def retrieve_remote_run(
    config_path: str | Path,
    output_dir: str | Path,
) -> ArtifactPaths:
    """Fetch and atomically publish one completed detached remote run."""
    config = RunConfig.from_toml(config_path)
    if not config.hosts:
        raise ConfigError("remote retrieval requires placement hosts")
    destination, paths, job_path = _remote_paths(output_dir)
    if _local_run_complete(config, paths):
        return paths
    status = remote_run_status(config.path, destination)
    if status.state != "complete":
        detail = f": {status.detail}" if status.detail else ""
        raise RuntimeError(f"remote job is {status.state}{detail}")
    return _retrieve_remote_job(
        destination,
        paths,
        job_path,
        _read_remote_job(config, job_path),
    )


def _follow_remote_job(job: RemoteJob, *, from_start: bool) -> None:
    lines = "+1" if from_start else "1"
    script = " ".join((
        f"work={shlex.quote(job.remote_output)};",
        "pid=$(cat \"$work/pid\");",
        f"exec tail --pid=\"$pid\" -n {lines} -f \"$work/coordinator.log\"",
    ))
    subprocess.run(
        ["ssh", *SSH_OPTIONS, "--", job.coordinator, script],
        check=True,
    )


def _follow_and_retrieve(
    config: RunConfig,
    destination: Path,
    job: RemoteJob,
    *,
    from_start: bool,
) -> ArtifactPaths:
    try:
        _follow_remote_job(job, from_start=from_start)
    except BaseException:
        print(
            "cluster: connection ended; detached job retained "
            "(--status / --follow / --retrieve / --stop)",
            file=sys.stderr,
        )
        raise
    status = remote_run_status(config.path, destination)
    if status.state != "complete":
        detail = f": {status.detail}" if status.detail else ""
        raise RuntimeError(f"remote job is {status.state}{detail}")
    _destination, paths, job_path = _remote_paths(destination)
    return _retrieve_remote_job(destination, paths, job_path, job)


def follow_remote_run(
    config_path: str | Path,
    output_dir: str | Path,
) -> ArtifactPaths:
    """Follow a detached coordinator and retrieve its completed artifacts."""
    config = RunConfig.from_toml(config_path)
    if not config.hosts:
        raise ConfigError("remote follow requires placement hosts")
    destination, paths, job_path = _remote_paths(output_dir)
    if _local_run_complete(config, paths):
        return paths
    status = remote_run_status(config.path, destination)
    if status.state == "complete":
        return _retrieve_remote_job(
            destination,
            paths,
            job_path,
            _read_remote_job(config, job_path),
        )
    if status.state != "running":
        detail = f": {status.detail}" if status.detail else ""
        raise RuntimeError(f"remote job is {status.state}{detail}")
    job = _read_remote_job(config, job_path)
    return _follow_and_retrieve(config, destination, job, from_start=False)


def run_remote_config(
    config_path: str | Path | RunConfig,
    output_dir: str | Path,
    *,
    transfer: bool = False,
    rebuild: bool = False,
) -> ArtifactPaths:
    """Launch, follow, and retrieve one disconnect-safe remote grid."""
    config = (
        config_path
        if isinstance(config_path, RunConfig)
        else RunConfig.from_toml(config_path)
    )
    if not config.hosts:
        raise ConfigError("remote coordinator requires placement hosts")
    job = _start_remote_job(
        config,
        output_dir,
        transfer=transfer,
        rebuild=rebuild,
    )
    destination = Path(output_dir).expanduser().resolve()
    return _follow_and_retrieve(config, destination, job, from_start=True)
