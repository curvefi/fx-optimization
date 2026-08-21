"""Site profiles for execution backends.

Profiles define local workstation or remote cluster execution environments,
including SSH topology, worker concurrency, and compiler configurations,
without embedding credentials or hardcoding blade lists in application code.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
import tomllib
from typing import Any, Mapping, Sequence

from ..specs.registry import SpecRegistry


class SiteProfileError(ValueError):
    """Raised when a site configuration profile is invalid or cannot be loaded."""


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise SiteProfileError(f"unsupported {label} fields: {', '.join(unknown)}")


_REMOTE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_REMOTE_PATH = re.compile(r"^/[A-Za-z0-9._/-]+$")


def validate_remote_host(value: str, label: str = "remote host") -> str:
    if not _REMOTE_TOKEN.fullmatch(value):
        raise SiteProfileError(f"{label} contains unsafe characters: {value!r}")
    return value


def validate_remote_path(value: str | PurePosixPath, label: str = "remote path") -> PurePosixPath:
    raw = str(value)
    path = PurePosixPath(raw)
    if raw == "/" or not _REMOTE_PATH.fullmatch(raw) or ".." in path.parts or "//" in raw:
        raise SiteProfileError(f"{label} must be a safe non-root absolute POSIX path")
    return path


@dataclass(frozen=True)
class SSHConfig:
    """SSH connection parameters for remote cluster execution."""

    user: str = "heswithme"
    key: Path | None = field(default_factory=lambda: Path("~/.ssh/id_rsa2").expanduser())
    port: int = 22
    connect_timeout: int = 10
    options: tuple[str, ...] = (
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10",
    )

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.user, str) or not (self.key is None or isinstance(self.key, Path)):
            raise SiteProfileError("ssh user and key have invalid types")
        if not isinstance(self.options, tuple) or not all(isinstance(x, str) for x in self.options):
            raise SiteProfileError("ssh.options must be a tuple of strings")
        if self.user:
            validate_remote_host(self.user, "ssh.user")
        if self.port < 1 or self.port > 65535 or self.connect_timeout < 1:
            raise SiteProfileError("ssh port and connect_timeout are invalid")
        if len(self.options) % 2 or any(
            self.options[index] != "-o" for index in range(0, len(self.options), 2)
        ):
            raise SiteProfileError("ssh.options must contain '-o', 'Key=Value' pairs")
        allowed_values = {
            "stricthostkeychecking": {"yes", "accept-new"},
            "batchmode": {"yes"},
            "controlmaster": {"no"},
            "controlpath": {"none"},
            "identitiesonly": {"yes"},
        }
        numeric_limits = {
            "connecttimeout": (1, 300),
            "connectionattempts": (1, 10),
            "serveraliveinterval": (1, 300),
            "serveralivecountmax": (1, 10),
        }
        for raw in self.options[1::2]:
            key, separator, value = raw.partition("=")
            normalized = key.lower()
            if not separator:
                raise SiteProfileError("ssh option values must use Key=Value syntax")
            if normalized in allowed_values and value.lower() in allowed_values[normalized]:
                continue
            if normalized in numeric_limits and value.isdigit():
                number = int(value)
                lower, upper = numeric_limits[normalized]
                if lower <= number <= upper:
                    continue
            raise SiteProfileError(f"ssh option is not allow-listed: {raw!r}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SSHConfig:
        _reject_unknown(
            data,
            {"user", "key", "port", "connect_timeout", "options"},
            "ssh",
        )
        key_raw = str(data.get("key", "~/.ssh/id_rsa2"))
        key_path = Path(key_raw).expanduser()
        connect_timeout = int(data.get("connect_timeout", 10))
        options_raw = data.get("options", ())
        if isinstance(options_raw, (str, bytes)) or not isinstance(options_raw, Sequence):
            raise SiteProfileError("ssh.options must be an array of strings")
        options = tuple(str(opt) for opt in options_raw)
        if not options:
            options = (
                "-o", "StrictHostKeyChecking=accept-new",
                "-o", "BatchMode=yes",
                "-o", f"ConnectTimeout={connect_timeout}",
            )
        return cls(
            user=str(data.get("user", "heswithme")),
            key=key_path,
            port=int(data.get("port", 22)),
            connect_timeout=connect_timeout,
            options=options,
        )


@dataclass(frozen=True)
class ClusterConfig:
    """Cluster topology and environment configuration."""

    coordinator: str = "blade-b6"
    transport: str = "rsync"
    remote_base: PurePosixPath = PurePosixPath("/home/heswithme/arb")
    remote_run_root: PurePosixPath = PurePosixPath("/home/heswithme/arb/runs")
    repository_root: PurePosixPath = PurePosixPath("/home/heswithme/arb")
    worker_command: str = "fxsim"
    logical_cores_per_blade: int = 128
    blades: tuple[str, ...] = ()
    nix_packages: tuple[str, ...] = ("gcc", "cmake", "boost", "gnumake")
    compiler_flags: tuple[str, ...] = (
        "-march=icelake-server",
        "-O3",
        "-fno-math-errno",
        "-funroll-loops",
        "-flto",
        "-pipe",
    )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ClusterConfig:
        _reject_unknown(
            data,
            {
                "coordinator", "transport", "remote_base", "remote_run_root",
                "repository_root", "worker_command", "logical_cores_per_blade",
                "blades", "nix_packages", "compiler_flags",
            },
            "cluster",
        )
        remote_base = PurePosixPath(str(data.get("remote_base", "/home/heswithme/arb")))
        run_root = data.get("remote_run_root")
        remote_run_root = (
            PurePosixPath(str(run_root)) if run_root else remote_base / "runs"
        )
        repository_root = PurePosixPath(str(data.get("repository_root", remote_base)))
        blades_raw = data.get("blades", ())
        packages_raw = data.get("nix_packages", ())
        flags_raw = data.get("compiler_flags", ())
        for label, value in (
            ("blades", blades_raw),
            ("nix_packages", packages_raw),
            ("compiler_flags", flags_raw),
        ):
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                raise SiteProfileError(f"cluster.{label} must be an array of strings")
        blades = tuple(str(b) for b in blades_raw)
        nix_pkgs = tuple(str(p) for p in packages_raw) or (
            "gcc", "cmake", "boost", "gnumake"
        )
        flags = tuple(str(f) for f in flags_raw) or (
            "-march=icelake-server",
            "-O3",
            "-fno-math-errno",
            "-funroll-loops",
            "-flto",
            "-pipe",
        )
        return cls(
            coordinator=str(data.get("coordinator", "blade-b6")),
            transport=str(data.get("transport", "rsync")),
            remote_base=remote_base,
            remote_run_root=remote_run_root,
            repository_root=repository_root,
            worker_command=str(data.get("worker_command", "fxsim")),
            logical_cores_per_blade=int(data.get("logical_cores_per_blade", 128)),
            blades=blades,
            nix_packages=nix_pkgs,
            compiler_flags=flags,
        )


@dataclass(frozen=True)
class HarnessConfig:
    """Evaluator request batching and timeout settings."""

    timeout_seconds: int = 3600
    chunk_size: int = 2048

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HarnessConfig:
        _reject_unknown(
            data,
            {"timeout_seconds", "chunk_size"},
            "harness",
        )
        return cls(
            timeout_seconds=int(data.get("timeout_seconds", 3600)),
            chunk_size=int(data.get("chunk_size", 2048)),
        )


@dataclass(frozen=True)
class RunnerConfig:
    """Runner process management settings."""

    type: str = "subprocess"
    max_workers: int = 12
    worker_concurrency: int = 1

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunnerConfig:
        _reject_unknown(data, {"type", "max_workers", "worker_concurrency"}, "runner")
        return cls(
            type=str(data.get("type", "subprocess")),
            max_workers=int(data.get("max_workers", 12)),
            worker_concurrency=int(data.get("worker_concurrency", 1)),
        )


@dataclass(frozen=True)
class SiteProfile:
    """Complete, self-contained configuration profile for one execution site."""

    name: str
    site_type: str  # "local" | "ssh"
    description: str = ""
    ssh: SSHConfig = field(default_factory=SSHConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    harness: HarnessConfig = field(default_factory=HarnessConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not all((isinstance(self.cluster, ClusterConfig),
                    isinstance(self.harness, HarnessConfig),
                    isinstance(self.runner, RunnerConfig))):
            raise SiteProfileError("site sections have invalid types")
        if any(
            not isinstance(value, PurePosixPath)
            for value in (
                self.cluster.remote_base,
                self.cluster.remote_run_root,
                self.cluster.repository_root,
            )
        ):
            raise SiteProfileError("cluster remote paths must be PurePosixPath values")
        if not isinstance(self.cluster.blades, tuple) or not all(
            isinstance(blade, str) for blade in self.cluster.blades):
            raise SiteProfileError("cluster.blades must be a tuple of strings")
        if not _REMOTE_TOKEN.fullmatch(self.name):
            raise SiteProfileError("site name must be a safe token")
        self.ssh.validate()
        if self.site_type not in {"local", "ssh"}:
            raise SiteProfileError("site_type must be 'local' or 'ssh'")
        if self.runner.max_workers < 1 or self.runner.worker_concurrency < 1:
            raise SiteProfileError("runner worker counts must be positive")
        if self.harness.timeout_seconds < 1 or self.harness.chunk_size < 1:
            raise SiteProfileError("harness timeout_seconds and chunk_size must be positive")
        if self.cluster.transport not in {"rsync", "shared_nfs"}:
            raise SiteProfileError("cluster.transport must be 'rsync' or 'shared_nfs'")
        if self.site_type == "local" and self.cluster.transport != "rsync":
            raise SiteProfileError("shared_nfs transport requires site_type='ssh'")
        if self.site_type == "ssh":
            if not self.cluster.blades:
                raise SiteProfileError("SSH sites must configure at least one blade")
            for blade in self.cluster.blades:
                validate_remote_host(blade, "cluster blade")
            if self.cluster.coordinator not in self.cluster.blades:
                raise SiteProfileError("cluster.coordinator must be one of cluster.blades")
            for label, value in (
                ("remote_base", self.cluster.remote_base),
                ("remote_run_root", self.cluster.remote_run_root),
                ("repository_root", self.cluster.repository_root),
            ):
                validate_remote_path(value, f"cluster.{label}")
            validate_remote_path(self.cluster.worker_command, "cluster.worker_command")
            if self.cluster.transport == "shared_nfs":
                expected = self.cluster.remote_base / "runs"
                if self.cluster.remote_run_root != expected:
                    raise SiteProfileError(
                        "shared_nfs requires remote_run_root == remote_base/runs"
                    )

    def validate_blades(self, blades: Sequence[str]) -> tuple[str, ...]:
        selected = tuple(blades)
        for blade in selected:
            validate_remote_host(blade, "selected blade")
        unknown = sorted(set(selected) - set(self.cluster.blades))
        if unknown:
            raise SiteProfileError(
                "selected blades are not configured for this site: " + ", ".join(unknown)
            )
        return selected

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SiteProfile:
        _reject_unknown(
            data,
            {"schema_version", "name", "site_type", "description", "ssh", "cluster", "harness", "runner"},
            "site profile",
        )
        sections: dict[str, Any] = {}
        for section in ("ssh", "cluster", "harness", "runner"):
            raw = data.get(section, {})
            if not isinstance(raw, Mapping):
                raise SiteProfileError(f"site profile {section} must be a table")
            cls_ = {"ssh": SSHConfig, "cluster": ClusterConfig,
                    "harness": HarnessConfig, "runner": RunnerConfig}[section]
            sections[section] = cls_.from_dict(raw) if raw else cls_()
        return cls(
            name=str(data.get("name", "unnamed")),
            site_type=str(data.get("site_type", "local")),
            description=str(data.get("description", "")),
            **sections,
        )


def find_site_profile_path(
    name_or_path: str | os.PathLike[str],
    root: Path,
) -> Path:
    """Locate a site profile by name or explicit file path."""
    registry = SpecRegistry.from_root(root)
    try:
        return registry.resolve("site", name_or_path)
    except FileNotFoundError as exc:
        raise SiteProfileError(str(exc)) from exc


def load_site_profile(
    name_or_path: str | os.PathLike[str] | None = None,
    *,
    root: Path,
) -> SiteProfile:
    """Load and parse a site profile TOML file.

    If ``name_or_path`` is None or 'local', loads 'local.toml' from configs/sites/
    or returns a default local profile if not found.
    """
    if name_or_path is None or name_or_path == "local":
        try:
            profile_path = find_site_profile_path("local", root)
        except SiteProfileError:
            return SiteProfile(
                name="local", site_type="local", description="Default local profile",
                runner=RunnerConfig(max_workers=1, worker_concurrency=10),
            )
    else:
        profile_path = find_site_profile_path(name_or_path, root)

    with profile_path.open("rb") as handle:
        data = tomllib.load(handle)

    schema_version = data.get("schema_version")
    if schema_version != "site_profile_v2":
        raise SiteProfileError(
            f"unsupported schema_version {schema_version!r} in {profile_path}; expected 'site_profile_v2'"
        )

    return SiteProfile.from_dict(data)


__all__ = [
    "ClusterConfig",
    "HarnessConfig",
    "RunnerConfig",
    "SSHConfig",
    "SiteProfile",
    "SiteProfileError",
    "find_site_profile_path",
    "load_site_profile",
    "validate_remote_host",
    "validate_remote_path",
]
