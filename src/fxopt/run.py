"""Execute a bounded local-or-SSH fxopt grid through one persistent harness session."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import tomllib
from typing import Any

from .candidates import CandidateSpec
from .config import CandidateConfig, ConfigError
from .contract import Candidate
from .engine import ClientFactory
from .placement import (
    EvaluatorFleet,
    PlacementLane,
    local_client_factory,
    ssh_client_factory,
)
from .results import ArtifactPaths, ResultWriter


_RUN_KEYS = frozenset({"id", "evaluator", "template", "manifest", "batch_size", "workers"})
_PLACEMENT_KEYS = frozenset({"hosts"})
_CANDIDATE_KEYS = frozenset({"defaults", "axes"})
_SCENARIO_KEYS = frozenset(
    {"id", "market", "market_sha256", "chainlink", "chainlink_sha256", "yb_mode"}
)


def _required_string(section: Mapping[str, Any], key: str, label: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label}.{key} must be a non-empty string")
    return value


def _resolve_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Resolved settings for the ordinary local-or-SSH ``fxopt run`` command."""

    path: Path
    run_id: str
    evaluator: Path
    template: Path
    manifest: Path | None
    batch_size: int
    workers: int
    hosts: tuple[str, ...]
    candidate: CandidateConfig
    session: Mapping[str, Any]
    scenario: Mapping[str, Any]

    @classmethod
    def from_toml(cls, path: str | Path) -> "RunConfig":
        config_path = Path(path).expanduser().resolve()
        try:
            with config_path.open("rb") as stream:
                raw = tomllib.load(stream)
        except OSError as exc:
            raise ConfigError(f"cannot read config {config_path}: {exc}") from exc

        run = raw.get("run")
        if not isinstance(run, Mapping):
            raise ConfigError("config requires a [run] table")
        unknown_run = set(run) - _RUN_KEYS
        if unknown_run:
            raise ConfigError(f"unknown [run] keys: {sorted(unknown_run)}")
        batch_size = run.get("batch_size")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ConfigError("run.batch_size must be a positive integer")
        workers = run.get("workers", 1)
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ConfigError("run.workers must be a positive integer")

        placement = raw.get("placement", {})
        if not isinstance(placement, Mapping):
            raise ConfigError("[placement] must be a mapping")
        unknown_placement = set(placement) - _PLACEMENT_KEYS
        if unknown_placement:
            raise ConfigError(f"unknown [placement] keys: {sorted(unknown_placement)}")
        raw_hosts = placement.get("hosts", [])
        if isinstance(raw_hosts, (str, bytes)) or not isinstance(raw_hosts, list):
            raise ConfigError("placement.hosts must be an array")
        hosts: list[str] = []
        for host in raw_hosts:
            if not isinstance(host, str) or not host.strip():
                raise ConfigError("placement.hosts entries must be non-empty strings")
            if host.startswith("-"):
                raise ConfigError("placement.hosts entries must not start with '-'")
            if any(character.isspace() or ord(character) < 32 for character in host):
                raise ConfigError("placement.hosts entries must not contain whitespace")
            if host in hosts:
                raise ConfigError(f"duplicate placement host: {host!r}")
            hosts.append(host)

        session = raw.get("session", {})
        if not isinstance(session, Mapping):
            raise ConfigError("[session] must be a mapping")
        forbidden_session = {"session_id", "template_path", "manifest_path"} & set(session)
        if forbidden_session:
            raise ConfigError(
                "[session] cannot set " + ", ".join(sorted(forbidden_session))
            )

        candidate = raw.get("candidate")
        if not isinstance(candidate, Mapping):
            raise ConfigError("config requires a [candidate] table")
        unknown_candidate = set(candidate) - _CANDIDATE_KEYS
        if unknown_candidate:
            raise ConfigError(f"unknown [candidate] keys: {sorted(unknown_candidate)}")

        manifest_value = run.get("manifest")
        manifest = (
            _resolve_path(manifest_value, config_path.parent)
            if isinstance(manifest_value, str) and manifest_value.strip()
            else None
        )
        if manifest_value is not None and manifest is None:
            raise ConfigError("run.manifest must be a non-empty string")
        scenario = raw.get("scenario")
        if manifest is None and not isinstance(scenario, Mapping):
            raise ConfigError("config without run.manifest requires a [scenario] table")
        scenario = dict(scenario) if isinstance(scenario, Mapping) else {}
        scenario_yb_mode = scenario.get("yb_mode")
        if scenario_yb_mode is not None and not isinstance(scenario_yb_mode, str):
            raise ConfigError("scenario.yb_mode must be a string")
        base = config_path.parent
        return cls(
            path=config_path,
            run_id=_required_string(run, "id", "run"),
            evaluator=_resolve_path(_required_string(run, "evaluator", "run"), base),
            template=_resolve_path(_required_string(run, "template", "run"), base),
            manifest=manifest,
            batch_size=batch_size,
            workers=workers,
            hosts=tuple(hosts),
            candidate=CandidateConfig.from_mapping(candidate),
            session=dict(session),
            scenario=scenario,
        )


def _session_manifest(config: RunConfig, directory: Path) -> Path:
    """Materialize the evaluator's narrow session manifest only for this run."""
    scenario = config.scenario
    if not scenario:
        raise ConfigError("config without run.manifest requires a [scenario] table")
    unknown = set(scenario) - _SCENARIO_KEYS
    if unknown:
        raise ConfigError(f"unknown [scenario] keys: {sorted(unknown)}")
    scenario_id = _required_string(scenario, "id", "scenario")
    market = _required_string(scenario, "market", "scenario")
    market_sha256 = _required_string(scenario, "market_sha256", "scenario")
    files = [{"path": str(_resolve_path(market, config.path.parent).resolve()), "kind": "market", "sha256": market_sha256}]
    chainlink = scenario.get("chainlink")
    chainlink_sha256 = scenario.get("chainlink_sha256")
    if chainlink is not None or chainlink_sha256 is not None:
        if not isinstance(chainlink, str) or not chainlink.strip() or not isinstance(chainlink_sha256, str) or not chainlink_sha256.strip():
            raise ConfigError("scenario.chainlink and scenario.chainlink_sha256 must be supplied together")
        files.append({"path": str(_resolve_path(chainlink, config.path.parent).resolve()), "kind": "chainlink", "sha256": chainlink_sha256})
    session = config.session
    payload = {
        "schema_version": "fxsim_manifest_v1",
        "run_kind": "session",
        "run_id": f"session-{scenario_id}",
        "resolved_spec": {
            "scenario": {
                "id": scenario_id,
                "start_time": session.get("start_time", 0),
                "end_time": session.get("end_time", 0),
                "n_candles": session.get("n_candles", 0),
                "candle_filter": session.get("candle_filter", 99.0),
                "market_files": files,
            }
        },
    }
    path = directory / "session.json"
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return path


def _candidate(spec: CandidateSpec) -> Candidate:
    payload = spec.payload
    expected = {"policy_params", "pool"}
    unknown = set(payload) - expected
    missing = expected - set(payload)
    if unknown or missing:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if unknown:
            details.append(f"unknown {sorted(unknown)}")
        raise ConfigError(
            "candidate payload must contain only policy_params and pool ("
            + "; ".join(details)
            + ")"
        )
    policy_params = payload["policy_params"]
    pool = payload["pool"]
    if not isinstance(policy_params, (list, tuple)):
        raise ConfigError("candidate.policy_params must be an array")
    if not isinstance(pool, Mapping):
        raise ConfigError("candidate.pool must be a mapping")
    return Candidate(
        candidate_id=spec.candidate_id,
        policy_params=tuple(policy_params),
        pool_overrides=pool,
    )


def _run_with_manifest(
    config: RunConfig,
    output_dir: str | Path,
    manifest: Path,
    *,
    client_factory: ClientFactory | None,
) -> ArtifactPaths:
    if client_factory is not None:
        lanes = (PlacementLane("injected", client_factory),)
    elif config.hosts:
        lanes = tuple(
            PlacementLane(
                host,
                ssh_client_factory(
                    host,
                    config.evaluator,
                    workers=config.workers,
                    verify_local_inputs=False,
                ),
            )
            for host in config.hosts
        )
    else:
        lanes = (
            PlacementLane(
                "local",
                local_client_factory(
                    config.evaluator,
                    work_dir=config.path.parent,
                    workers=config.workers,
                ),
            ),
        )
    candidate_grid = config.candidate.grid()
    lane_count = len(lanes)
    effective_batch = min(config.batch_size, (len(candidate_grid) + lane_count - 1) // lane_count)
    metadata = {
        "config": str(config.path),
        "evaluator": str(config.evaluator),
        "template": str(config.template),
        "manifest": str(config.manifest) if config.manifest is not None else "temporary",
        "placement": "ssh" if config.hosts and client_factory is None else "local",
        "hosts": list(config.hosts),
        "batch_size": config.batch_size,
        "effective_batch_size": effective_batch,
        "workers": config.workers,
    }
    writer = ResultWriter(output_dir, run_id=config.run_id, metadata=metadata)
    open_session = {
        "template_path": str(config.template),
        "manifest_path": str(manifest),
        **config.session,
    }
    if (yb_mode := config.scenario.get("yb_mode")) is not None:
        open_session.setdefault("yb_mode", yb_mode)
    with writer:
        with EvaluatorFleet(
            lanes,
            session_id=config.run_id,
            batch_size=effective_batch,
            open_session=open_session,
        ) as fleet:
            for specs in candidate_grid.iter_batches(lane_count * effective_batch):
                batch = tuple(_candidate(spec) for spec in specs)
                writer.append(batch, fleet.evaluate(batch))
        return writer.finalize()


def run_config(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    client_factory: ClientFactory | None = None,
) -> ArtifactPaths:
    """Run every grid point in bounded batches and publish the two artifacts."""
    config = RunConfig.from_toml(config_path)
    if config.manifest is not None:
        return _run_with_manifest(config, output_dir, config.manifest, client_factory=client_factory)
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".fxopt-session-", dir=output_path) as temporary:
        manifest = _session_manifest(config, Path(temporary))
        return _run_with_manifest(config, output_path, manifest, client_factory=client_factory)


__all__ = ["RunConfig", "run_config"]
