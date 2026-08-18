"""Process and SSH execution adapters with injectable interfaces.

Provides uniform execution abstractions for local subprocesses, remote SSH commands,
and rsync file transfers, with a mock adapter for deterministic unit testing.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from .site import SSHConfig, validate_remote_host, validate_remote_path


@dataclass(frozen=True)
class ProcessResult:
    """Standardized outcome of an executed local or remote command."""

    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        """True if the process exited with returncode 0."""
        return self.returncode == 0


class ProcessAdapter(Protocol):
    """Protocol for command execution."""

    def run(self, argv: Sequence[str], **kwargs: Any) -> ProcessResult:
        """Execute a command represented by an argv sequence."""
        ...


class LocalProcessAdapter:
    """Executes commands on the local machine via subprocess."""

    def __init__(self, default_timeout: float = 3600.0) -> None:
        self.default_timeout = default_timeout

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        input_data: str | bytes | None = None,
        **kwargs: Any,
    ) -> ProcessResult:
        effective_timeout = timeout if timeout is not None else self.default_timeout
        start_time = time.monotonic()

        merged_env: dict[str, str] | None = None
        if env is not None:
            merged_env = dict(os.environ)
            merged_env.update(env)

        text_mode = not isinstance(input_data, bytes)
        stdin_val = input_data if input_data is not None else None

        try:
            proc = subprocess.run(
                list(argv),
                cwd=str(cwd) if cwd is not None else None,
                env=merged_env,
                timeout=effective_timeout,
                input=stdin_val,
                text=text_mode,
                capture_output=True,
                check=False,
            )
            duration = time.monotonic() - start_time
            stdout_str = proc.stdout if isinstance(proc.stdout, str) else proc.stdout.decode("utf-8", errors="replace")
            stderr_str = proc.stderr if isinstance(proc.stderr, str) else proc.stderr.decode("utf-8", errors="replace")
            return ProcessResult(
                returncode=proc.returncode,
                stdout=stdout_str,
                stderr=stderr_str,
                duration_seconds=duration,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.monotonic() - start_time
            stdout_str = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", errors="replace")
            stderr_str = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", errors="replace")
            return ProcessResult(
                returncode=-1,
                stdout=stdout_str,
                stderr=f"command timed out after {effective_timeout}s: {stderr_str}",
                duration_seconds=duration,
            )
        except Exception as exc:  # noqa: BLE001
            duration = time.monotonic() - start_time
            return ProcessResult(
                returncode=-1,
                stdout="",
                stderr=f"execution failed: {exc}",
                duration_seconds=duration,
            )


class SSHProcessAdapter:
    """Executes commands and transfers files on remote blades over SSH/rsync."""

    def __init__(
        self,
        ssh_config: SSHConfig | None = None,
        process_runner: ProcessAdapter | None = None,
    ) -> None:
        self.ssh_config = ssh_config or SSHConfig()
        self._runner = process_runner or LocalProcessAdapter()

    def build_ssh_argv(self, blade: str, command: str) -> list[str]:
        """Construct the full argv list for an SSH invocation."""
        self.ssh_config.validate()
        validate_remote_host(blade, "SSH target")
        argv: list[str] = ["ssh"]
        argv.extend(self.ssh_config.options)
        if self.ssh_config.key:
            argv.extend(["-i", str(self.ssh_config.key)])
        if self.ssh_config.port != 22:
            argv.extend(["-p", str(self.ssh_config.port)])
        user_host = f"{self.ssh_config.user}@{blade}" if self.ssh_config.user else blade
        argv.append(user_host)
        argv.append(command)
        return argv

    def run_ssh(
        self,
        blade: str,
        command: str,
        *,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> ProcessResult:
        """Run a remote shell command on a specific blade."""
        argv = self.build_ssh_argv(blade, command)
        return self._runner.run(argv, timeout=timeout, **kwargs)

    def run(self, argv: Sequence[str], **kwargs: Any) -> ProcessResult:
        """Default run delegates to underlying process runner."""
        return self._runner.run(argv, **kwargs)

    def rsync_upload(
        self,
        local_path: Path,
        blade: str,
        remote_path: str,
        *,
        timeout: float = 300.0,
        delete: bool = False,
    ) -> ProcessResult:
        """Upload a local file or directory to a remote blade using rsync."""
        self.ssh_config.validate()
        validate_remote_host(blade, "rsync target")
        remote_path = str(validate_remote_path(remote_path, "rsync remote path"))
        ssh_cmd_parts = ["ssh"] + list(self.ssh_config.options)
        if self.ssh_config.key:
            ssh_cmd_parts.extend(["-i", str(self.ssh_config.key)])
        if self.ssh_config.port != 22:
            ssh_cmd_parts.extend(["-p", str(self.ssh_config.port)])
        ssh_str = " ".join(shlex.quote(p) for p in ssh_cmd_parts)

        target = f"{self.ssh_config.user}@{blade}:{remote_path}" if self.ssh_config.user else f"{blade}:{remote_path}"
        argv: list[str] = ["rsync", "-az", "--protect-args", "-e", ssh_str]
        if delete:
            argv.append("--delete")
        argv.extend(["--", str(local_path), target])

        return self._runner.run(argv, timeout=timeout)

    def rsync_download(
        self,
        blade: str,
        remote_path: str,
        local_path: Path,
        *,
        timeout: float = 300.0,
    ) -> ProcessResult:
        """Download a remote file or directory to the local filesystem using rsync."""
        self.ssh_config.validate()
        validate_remote_host(blade, "rsync source")
        remote_path = str(validate_remote_path(remote_path, "rsync remote path"))
        ssh_cmd_parts = ["ssh"] + list(self.ssh_config.options)
        if self.ssh_config.key:
            ssh_cmd_parts.extend(["-i", str(self.ssh_config.key)])
        if self.ssh_config.port != 22:
            ssh_cmd_parts.extend(["-p", str(self.ssh_config.port)])
        ssh_str = " ".join(shlex.quote(p) for p in ssh_cmd_parts)

        source = f"{self.ssh_config.user}@{blade}:{remote_path}" if self.ssh_config.user else f"{blade}:{remote_path}"
        local_path.parent.mkdir(parents=True, exist_ok=True)
        argv: list[str] = [
            "rsync", "-az", "--protect-args", "-e", ssh_str,
            "--", source, str(local_path),
        ]

        return self._runner.run(argv, timeout=timeout)


class MockProcessAdapter:
    """Mock process adapter for deterministic unit testing without spawning processes."""

    def __init__(
        self,
        default_result: ProcessResult | None = None,
    ) -> None:
        self.default_result = default_result or ProcessResult(returncode=0, stdout="", stderr="")
        self.calls: list[dict[str, Any]] = []
        self._handlers: list[tuple[Callable[[Sequence[str]], bool], ProcessResult]] = []

    def register_handler(
        self,
        matcher: Callable[[Sequence[str]], bool],
        result: ProcessResult,
    ) -> None:
        """Register a custom result for commands matching a predicate."""
        self._handlers.append((matcher, result))

    def run(self, argv: Sequence[str], **kwargs: Any) -> ProcessResult:
        """Record the call and return a matched or default ProcessResult."""
        self.calls.append({"argv": list(argv), "kwargs": dict(kwargs)})
        for matcher, result in self._handlers:
            if matcher(argv):
                return result
        return self.default_result


__all__ = [
    "LocalProcessAdapter",
    "MockProcessAdapter",
    "ProcessAdapter",
    "ProcessResult",
    "SSHProcessAdapter",
]
