"""Site profiles for execution backends.

Profiles define local workstation or remote cluster execution environments,
including SSH topology, worker concurrency, and compiler configurations,
without embedding credentials or hardcoding blade lists in application code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
import tomllib
from typing import Any, Mapping, Sequence

from ..specs.common import repository_root


class SiteProfileError(ValueError):
    """Raised when a site configuration profile is invalid or cannot be loaded."""


@dataclass(frozen=True)
class SSHConfig:
    """SSH connection parameters for remote cluster execution."""

    user: str = "heswithme"
    key: Path = field(default_factory=lambda: Path("~/.ssh/id_rsa2").expanduser())
    port: int = 22
    connect_timeout: int = 10
    options: tuple[str, ...] = (
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10",
    )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SSHConfig:
        key_raw = str(data.get("key", "~/.ssh/id_rsa2"))
        key_path = Path(key_raw).expanduser()
        options = tuple(str(opt) for opt in data.get("options", ()))
        if not options:
            options = cls.options
        return cls(
            user=str(data.get("user", "heswithme")),
            key=key_path,
            port=int(data.get("port", 22)),
            connect_timeout=int(data.get("connect_timeout", 10)),
            options=options,
        )


@dataclass(frozen=True)
class ClusterConfig:
    """Cluster topology and environment configuration."""

    coordinator: str = "blade-b6"
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
        remote_base = PurePosixPath(str(data.get("remote_base", "/home/heswithme/arb")))
        run_root = data.get("remote_run_root")
        remote_run_root = (
            PurePosixPath(str(run_root)) if run_root else remote_base / "runs"
        )
        repository_root = PurePosixPath(str(data.get("repository_root", remote_base)))
        blades = tuple(str(b) for b in data.get("blades", ()))
        nix_pkgs = tuple(str(p) for p in data.get("nix_packages", ())) or (
            "gcc", "cmake", "boost", "gnumake"
        )
        flags = tuple(str(f) for f in data.get("compiler_flags", ())) or (
            "-march=icelake-server",
            "-O3",
            "-fno-math-errno",
            "-funroll-loops",
            "-flto",
            "-pipe",
        )
        return cls(
            coordinator=str(data.get("coordinator", "blade-b6")),
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
    """Harness executable settings."""

    binary_name: str = "arb_evaluator_ld"
    remote_binary_path: PurePosixPath | None = None
    default_real: str = "longdouble"
    timeout_seconds: int = 3600
    persistent_evaluator: bool = True
    chunk_size: int = 2048

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HarnessConfig:
        return cls(
            binary_name=str(data.get("binary_name", "arb_evaluator_ld")),
            remote_binary_path=(
                PurePosixPath(str(data["remote_binary_path"]))
                if data.get("remote_binary_path")
                else None
            ),
            default_real=str(data.get("default_real", "longdouble")),
            timeout_seconds=int(data.get("timeout_seconds", 3600)),
            persistent_evaluator=bool(data.get("persistent_evaluator", True)),
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

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SiteProfile:
        name = str(data.get("name", "unnamed"))
        site_type = str(data.get("site_type", "local"))
        description = str(data.get("description", ""))
        ssh = (
            SSHConfig.from_dict(data["ssh"])
            if "ssh" in data and isinstance(data["ssh"], Mapping)
            else SSHConfig()
        )
        cluster = (
            ClusterConfig.from_dict(data["cluster"])
            if "cluster" in data and isinstance(data["cluster"], Mapping)
            else ClusterConfig()
        )
        harness = (
            HarnessConfig.from_dict(data["harness"])
            if "harness" in data and isinstance(data["harness"], Mapping)
            else HarnessConfig()
        )
        runner = (
            RunnerConfig.from_dict(data["runner"])
            if "runner" in data and isinstance(data["runner"], Mapping)
            else RunnerConfig()
        )
        return cls(
            name=name,
            site_type=site_type,
            description=description,
            ssh=ssh,
            cluster=cluster,
            harness=harness,
            runner=runner,
        )


def find_site_profile_path(
    name_or_path: str | os.PathLike[str],
    root: Path | None = None,
) -> Path:
    """Locate a site profile by name or explicit file path."""
    candidate = Path(name_or_path)
    if candidate.is_file():
        return candidate.resolve()

    repo = repository_root(root)
    sites_dir = repo / "configs" / "sites"

    # Try name with .toml suffix
    name_str = str(name_or_path)
    if not name_str.endswith(".toml"):
        with_suffix = sites_dir / f"{name_str}.toml"
        if with_suffix.is_file():
            return with_suffix.resolve()

    direct = sites_dir / name_str
    if direct.is_file():
        return direct.resolve()

    raise SiteProfileError(
        f"site profile '{name_or_path}' not found in {sites_dir} or as a file path"
    )


def load_site_profile(
    name_or_path: str | os.PathLike[str] | None = None,
    root: Path | None = None,
) -> SiteProfile:
    """Load and parse a site profile TOML file.

    If ``name_or_path`` is None or 'local', loads 'local.toml' from configs/sites/
    or returns a default local profile if not found.
    """
    if name_or_path is None or name_or_path == "local":
        try:
            profile_path = find_site_profile_path("local", root)
        except SiteProfileError:
            return SiteProfile(name="local", site_type="local", description="Default local profile")
    else:
        profile_path = find_site_profile_path(name_or_path, root)

    with profile_path.open("rb") as handle:
        data = tomllib.load(handle)

    schema_version = data.get("schema_version")
    if schema_version != "site_profile_v1":
        raise SiteProfileError(
            f"unsupported schema_version {schema_version!r} in {profile_path}; expected 'site_profile_v1'"
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
]
