"""Verified local inputs for compiled evaluator sessions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..artifacts.io import atomic_write_json, sha256_path
from ..specs.common import assert_contained_path, canonical_json_bytes
from ..specs.scenario import ScenarioClosure, ScenarioSpec
from .plans import CandidatePlan, CandidatePlanError, CandidateSchema, ScenarioKey


class SessionMaterializationError(ValueError):
    """A local session or compiled request failed attestation."""


_TRANSPORT_FIELDS = frozenset(
    {"session_id", "template_path", "template_sha256", "manifest_path", "manifest_sha256"}
)


def _strict_object(data: bytes, *, label: str) -> dict[str, Any]:
    if not isinstance(data, bytes):
        raise SessionMaterializationError(f"{label} must be bytes")
    try:
        value = json.loads(
            data,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise SessionMaterializationError(f"{label} is not strict JSON") from exc
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise SessionMaterializationError(f"{label} must encode an object")
    if canonical_json_bytes(value) != data:
        raise SessionMaterializationError(f"{label} is not canonical JSON")
    return value


def _validate_legacy_alias(request: dict[str, Any], *, label: str) -> None:
    mode = request.get("yb_mode")
    alias = request.get("yb_releverage")
    if not isinstance(mode, str) or not isinstance(alias, bool) or alias != (mode != "off"):
        raise SessionMaterializationError(
            f"{label} must satisfy yb_releverage == (yb_mode != 'off')"
        )


@dataclass(frozen=True, slots=True)
class LocalSessionTransportReceipt:
    """Path-specific transport evidence for one local protocol request."""

    session_id: str
    template_path: str
    template_sha256: str
    manifest_path: str
    manifest_sha256: str
    schema_version: str = "curve_fx_local_session_transport_v1"

    @property
    def canonical_json(self) -> bytes:
        return canonical_json_bytes(asdict(self))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json).hexdigest()


@dataclass(frozen=True, slots=True, init=False)
class LocalSessionMaterialization:
    """Deeply immutable closure plus local, byte-attested protocol inputs."""

    repository: Path
    manifest_root: Path
    closure: ScenarioClosure
    scenario_key: ScenarioKey
    transport_receipt: LocalSessionTransportReceipt
    baseline_request_json: bytes
    baseline_request_sha256: str

    @classmethod
    def from_scenario(
        cls, scenario: ScenarioSpec, *, repository: Path, manifest_root: Path, session_id: str
    ) -> LocalSessionMaterialization:
        if not isinstance(scenario, ScenarioSpec):
            raise TypeError("scenario must be a validated ScenarioSpec")
        if ScenarioSpec.from_dict(scenario.to_dict()) != scenario:
            raise SessionMaterializationError("scenario is not a canonical validated ScenarioSpec")
        if not isinstance(session_id, str) or not session_id.strip():
            raise SessionMaterializationError("session_id must be a non-empty string")

        root = Path(repository).resolve()
        output_root = Path(manifest_root).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"repository directory not found: {root}")
        output_root.mkdir(parents=True, exist_ok=True)
        if not output_root.is_dir():
            raise FileNotFoundError(f"manifest directory not found: {output_root}")
        if scenario.template_path is None:
            raise FileNotFoundError("scenario template_path is required by the harness protocol")

        template = assert_contained_path(scenario.template_path, root, allow_symlinks=False)
        template_digest = _verified_digest(template, scenario.template_sha256, label="template file")
        market_files: list[dict[str, object]] = []
        market_digests: list[str] = []
        for market in scenario.market_files:
            path = assert_contained_path(market.path, root, allow_symlinks=False)
            digest = _verified_digest(path, market.sha256, label=f"market file {market.path}")
            market_digests.append(digest)
            market_files.append({"path": path.as_posix(), "sha256": digest, "kind": market.kind})

        closure = ScenarioClosure.from_verified(
            scenario, template_sha256=template_digest, market_sha256s=market_digests
        )
        manifest_payload = {
            "schema_version": "fxsim_manifest_v1",
            "run_kind": "session",
            "run_id": f"session-{scenario.id}",
            "resolved_spec": {"scenario": scenario.harness_manifest_scenario(market_files)},
        }
        manifest_name = hashlib.sha256(canonical_json_bytes(manifest_payload)).hexdigest()
        manifest = atomic_write_json(output_root / f"{manifest_name}.json", manifest_payload)
        manifest_digest = sha256_path(manifest)
        receipt = LocalSessionTransportReceipt(
            session_id=session_id,
            template_path=template.as_posix(),
            template_sha256=template_digest,
            manifest_path=manifest.resolve().as_posix(),
            manifest_sha256=manifest_digest,
        )
        request = {
            "session_id": session_id,
            "template_path": receipt.template_path,
            "template_sha256": template_digest,
            "manifest_path": receipt.manifest_path,
            "manifest_sha256": manifest_digest,
            "pool_index": 0,
            **scenario.harness_session_config(),
        }
        _validate_legacy_alias(request, label="scenario session request")
        request_json = canonical_json_bytes(request)
        instance = object.__new__(cls)
        for name, value in (
            ("repository", root),
            ("manifest_root", output_root),
            ("closure", closure),
            ("scenario_key", ScenarioKey.from_closure(closure)),
            ("transport_receipt", receipt),
            ("baseline_request_json", request_json),
            ("baseline_request_sha256", hashlib.sha256(request_json).hexdigest()),
        ):
            object.__setattr__(instance, name, value)
        return instance

    @property
    def baseline_open_session_fields(self) -> dict[str, Any]:
        """Return a detached copy of the exact baseline protocol-v1 fields."""
        return _strict_object(self.baseline_request_json, label="baseline session request")

    def validated(self) -> LocalSessionMaterialization:
        request = self.baseline_open_session_fields
        if self.scenario_key.validated() != ScenarioKey.from_closure(self.closure):
            raise SessionMaterializationError("materialization ScenarioKey does not match closure")
        if hashlib.sha256(self.baseline_request_json).hexdigest() != self.baseline_request_sha256:
            raise SessionMaterializationError("baseline session request hash mismatch")
        receipt = self.transport_receipt
        for field in _TRANSPORT_FIELDS:
            if request.get(field) != getattr(receipt, field):
                raise SessionMaterializationError(f"baseline transport field mismatch: {field}")
        _validate_legacy_alias(request, label="baseline session request")
        template = assert_contained_path(receipt.template_path, self.repository, allow_symlinks=False)
        manifest = assert_contained_path(receipt.manifest_path, self.manifest_root, allow_symlinks=False)
        _verified_digest(template, receipt.template_sha256, label="template file")
        _verified_digest(manifest, receipt.manifest_sha256, label="session manifest")
        _verify_manifest_markets(manifest, self.repository, self.closure)
        return self


def _verified_digest(path: Path, declared: str | None, *, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    digest = sha256_path(path)
    if declared is not None and declared != digest:
        raise SessionMaterializationError(
            f"{label} hash mismatch: expected {declared}, calculated {digest}"
        )
    return digest


def _verify_manifest_markets(manifest: Path, repository: Path, closure: ScenarioClosure) -> None:
    try:
        payload = json.loads(manifest.read_bytes())
        scenario = payload["resolved_spec"]["scenario"]
        files = scenario["market_files"]
        if scenario["id"] != closure.scenario_id or len(files) != len(closure.market_inputs):
            raise ValueError("scenario identity or market count mismatch")
        for item, expected in zip(files, closure.market_inputs, strict=True):
            if item["kind"] != expected.kind or item["sha256"] != expected.sha256:
                raise ValueError("market identity mismatch")
            path = assert_contained_path(item["path"], repository, allow_symlinks=False)
            _verified_digest(path, expected.sha256, label=f"market file {item['path']}")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        if isinstance(exc, SessionMaterializationError):
            raise
        raise SessionMaterializationError(
            "session manifest does not match the materialized scenario closure"
        ) from exc


def _verified_plan_request(
    plan: CandidatePlan,
    materialization: LocalSessionMaterialization,
    *,
    parameter_schema: CandidateSchema | None = None,
) -> tuple[dict[str, Any], str]:
    """Verify a compiled plan without admitting any caller-owned raw mapping."""
    if not isinstance(plan, CandidatePlan):
        raise TypeError("plan must be a CandidatePlan")
    materialization.validated()
    try:
        scenario_key = plan.scenario_key.validated()
    except CandidatePlanError as exc:
        raise SessionMaterializationError("compiled plan has an invalid ScenarioKey") from exc
    if scenario_key != materialization.scenario_key:
        raise SessionMaterializationError("compiled plan ScenarioKey does not match materialization")

    request = _strict_object(plan.session_request_json, label="compiled session request")
    baseline = materialization.baseline_open_session_fields
    if request.keys() != baseline.keys():
        raise SessionMaterializationError("compiled session request fields do not match protocol v1")
    receipt = materialization.transport_receipt
    for field in _TRANSPORT_FIELDS:
        if request[field] != getattr(receipt, field):
            raise SessionMaterializationError(f"compiled transport field mismatch: {field}")
    try:
        if parameter_schema is None:
            _validate_legacy_alias(request, label="compiled session request")
        elif parameter_schema.finalize_open_session(request) != request:
            raise SessionMaterializationError("compiled session request has a stale legacy alias")
    except CandidatePlanError as exc:
        raise SessionMaterializationError("compiled session request has invalid YB aliases") from exc

    try:
        plan.session_key.validated(parameter_schema)
    except (AttributeError, CandidatePlanError) as exc:
        raise SessionMaterializationError("compiled plan has an invalid SessionKey") from exc
    for name, value in plan.session_key.open_session_values.items():
        if parameter_schema is None:
            if not name.startswith("run."):
                raise SessionMaterializationError("SessionKey contains an invalid session identity name")
            field = name.removeprefix("run.")
        else:
            descriptor = parameter_schema.descriptor(name)
            field = descriptor.lowering_path.removeprefix("open_session.")
        if field not in request or request[field] != value:
            raise SessionMaterializationError(
                f"compiled request does not match its SessionKey field: {name}"
            )
    return request, hashlib.sha256(plan.session_request_json).hexdigest()


__all__ = ["LocalSessionMaterialization", "LocalSessionTransportReceipt", "SessionMaterializationError"]
