"""Portable compiled-evaluation identities and execution-local session binding."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from ..specs.common import canonical_json_bytes
from .plans import (
    CandidateCompiler, CandidatePlan, CandidatePlanError, CandidateSchema,
    ObservationKey, ScenarioKey, SessionKey,
)
from .session import (
    LocalSessionMaterialization, LocalSessionTransportReceipt,
    SessionMaterializationError,
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class EvaluationGroupingError(ValueError):
    """Compiled evaluation evidence cannot form a portable session group."""


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise EvaluationGroupingError(f"{label} must be lowercase SHA-256")
    return value


def _object(data: bytes, *, label: str) -> dict[str, Any]:
    if not isinstance(data, bytes):
        raise EvaluationGroupingError(f"{label} must be bytes")
    try:
        value = json.loads(
            data,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise EvaluationGroupingError(f"{label} is not strict JSON") from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) != data:
        raise EvaluationGroupingError(f"{label} must be a canonical JSON object")
    return value


@dataclass(frozen=True, slots=True)
class PortableCandidate:
    """Candidate and portable session identities with all transport bytes stripped."""
    scenario_key: ScenarioKey
    session_key: SessionKey
    policy_params: tuple[object, ...]
    pool_overrides_json: bytes
    candidate_json: bytes
    candidate_sha256: str

    @classmethod
    def from_plan(cls, plan: CandidatePlan) -> PortableCandidate:
        if not isinstance(plan, CandidatePlan):
            raise TypeError("plan must be a CandidatePlan")
        return cls(
            plan.scenario_key, plan.session_key, plan.policy_params,
            plan.pool_overrides_json, plan.candidate_json,
            plan.candidate_sha256,
        )


@dataclass(frozen=True, slots=True)
class CompiledEvaluation:
    """One portable candidate plus evaluate_batch observation semantics."""
    candidate: PortableCandidate
    artifact_sha256: str
    observation_key: ObservationKey
    ordinal: int | None = None
    evaluation_id: str | None = None

    @classmethod
    def from_plan(
        cls, plan: CandidatePlan, *, compiler: CandidateCompiler,
        artifact_sha256: str,
        observation: Mapping[str, object] | None = None,
        ordinal: int | None = None,
        evaluation_id: str | None = None,
    ) -> CompiledEvaluation:
        if not isinstance(compiler, CandidateCompiler):
            raise TypeError("compiler must be a CandidateCompiler")
        key, _ = compiler.compile_observation(observation)
        return cls(PortableCandidate.from_plan(plan), artifact_sha256, key,
                   ordinal, evaluation_id)


def encode_compiled_evaluation(evaluation: CompiledEvaluation) -> dict[str, Any]:
    """Encode one compiled evaluation into its canonical JSON record."""

    candidate = evaluation.candidate
    return {
        "artifact_sha256": evaluation.artifact_sha256,
        "ordinal": evaluation.ordinal,
        "evaluation_id": evaluation.evaluation_id,
        "scenario_key": _key_record(candidate.scenario_key),
        "session_key": _key_record(candidate.session_key),
        "observation_key": _key_record(evaluation.observation_key),
        "policy_params": list(candidate.policy_params),
        "candidate_sha256": candidate.candidate_sha256,
        "pool_overrides_json": candidate.pool_overrides_json.decode(),
        "candidate_json": candidate.candidate_json.decode(),
    }


def decode_compiled_evaluation(record: Mapping[str, Any]) -> CompiledEvaluation:
    """Decode and validate one canonical compiled-evaluation record."""

    fields = {
        "artifact_sha256", "ordinal", "evaluation_id", "scenario_key", "session_key",
        "observation_key", "policy_params", "pool_overrides_json", "candidate_json",
        "candidate_sha256",
    }
    if not isinstance(record, Mapping) or set(record) != fields:
        raise EvaluationGroupingError("compiled evaluation record has invalid fields")
    ordinal, identifier, params = record["ordinal"], record["evaluation_id"], record["policy_params"]
    if (
        isinstance(ordinal, bool)
        or not isinstance(ordinal, int)
        or ordinal < 0
        or not isinstance(identifier, str)
        or not identifier
        or not isinstance(params, list)
    ):
        raise EvaluationGroupingError("compiled evaluation ordinal, ID, or policy_params is invalid")
    scenario = _decode_key(record["scenario_key"], ScenarioKey, "scenario_key")
    session = _decode_key(record["session_key"], SessionKey, "session_key")
    observation = _decode_key(record["observation_key"], ObservationKey, "observation_key")
    overrides = _json_text(record["pool_overrides_json"], label="pool overrides")
    candidate_json = _json_text(record["candidate_json"], label="candidate payload")
    return CompiledEvaluation(
        PortableCandidate(
            scenario,
            session,
            tuple(params),
            overrides,
            candidate_json,
            _digest(record["candidate_sha256"], label="candidate_sha256"),
        ),
        _digest(record["artifact_sha256"], label="artifact_sha256"),
        observation,
        ordinal, identifier,
    )


def _key_record(key: Any) -> dict[str, str]:
    return {
        "identity_json": key.identity_json.decode("utf-8"),
        "sha256": key.sha256,
    }


def _decode_key(record: Any, cls: Any, label: str) -> Any:
    if not isinstance(record, Mapping) or set(record) != {"identity_json", "sha256"}:
        raise EvaluationGroupingError(f"{label} record has invalid fields")
    identity = _json_text(record["identity_json"], f"{label} identity")
    digest = _digest(record["sha256"], label=f"{label} sha256")
    if hashlib.sha256(identity).hexdigest() != digest:
        raise EvaluationGroupingError(f"{label} hash mismatch")
    return cls(identity, digest)


def _json_text(value: Any, label: str) -> bytes:
    if not isinstance(value, str):
        raise EvaluationGroupingError(f"{label} must be canonical JSON text")
    data = value.encode()
    _object(data, label=label)
    return data


@dataclass(frozen=True, slots=True)
class ObservationGroup:
    """Stable observation partition within one session group."""
    key: ObservationKey
    evaluations: tuple[CompiledEvaluation, ...]


@dataclass(frozen=True, slots=True)
class SessionGroupKey:
    """Portable global identity for one artifact/schema/scenario/session tuple."""
    identity_json: bytes
    sha256: str

    @classmethod
    def create(cls, artifact_sha256: str, schema: CandidateSchema,
               scenario_key: ScenarioKey, session_key: SessionKey) -> SessionGroupKey:
        identity = canonical_json_bytes({
            "version": "curve_fx_session_group_v1",
            "artifact_sha256": artifact_sha256,
            "parameter_schema_sha256": schema.sha256,
            "scenario_key_sha256": scenario_key.sha256,
            "session_key_sha256": session_key.sha256,
        })
        return cls(identity, hashlib.sha256(identity).hexdigest())

    def validated(self) -> SessionGroupKey:
        payload = _object(self.identity_json, label="SessionGroupKey identity")
        fields = {
            "version", "artifact_sha256", "parameter_schema_sha256",
            "scenario_key_sha256", "session_key_sha256",
        }
        if set(payload) != fields or payload.get("version") != "curve_fx_session_group_v1":
            raise EvaluationGroupingError("SessionGroupKey identity has invalid shape or version")
        for field in fields - {"version"}:
            _digest(payload[field], label=f"SessionGroupKey {field}")
        _digest(self.sha256, label="SessionGroupKey sha256")
        if hashlib.sha256(self.identity_json).hexdigest() != self.sha256:
            raise EvaluationGroupingError("SessionGroupKey hash mismatch")
        return self

@dataclass(frozen=True, slots=True)
class SessionGroup:
    """Order-preserving portable partition sharing one evaluator session."""
    key: SessionGroupKey
    artifact_sha256: str
    parameter_schema: CandidateSchema
    scenario_key: ScenarioKey
    session_key: SessionKey
    observation_groups: tuple[ObservationGroup, ...]
    evaluations: tuple[CompiledEvaluation, ...]


@dataclass(frozen=True, slots=True)
class LocalSessionGroupBinding:
    """Ephemeral exact-request and local transport binding for a portable group."""
    group: SessionGroup
    materialization: LocalSessionMaterialization
    session_request_json: bytes
    session_request_sha256: str
    transport_receipt: LocalSessionTransportReceipt


def _validate_evaluation(
    evaluation: CompiledEvaluation, artifact_sha256: str, schema: CandidateSchema
) -> CompiledEvaluation:
    if not isinstance(evaluation, CompiledEvaluation) or not isinstance(
        evaluation.candidate, PortableCandidate
    ):
        raise EvaluationGroupingError("evaluations must contain CompiledEvaluation values")
    if evaluation.artifact_sha256 != artifact_sha256:
        raise EvaluationGroupingError("compiled evaluation artifact mismatch")
    if evaluation.ordinal is not None and (isinstance(evaluation.ordinal, bool)
            or not isinstance(evaluation.ordinal, int) or evaluation.ordinal < 0):
        raise EvaluationGroupingError("evaluation ordinal must be a non-negative integer")
    if evaluation.evaluation_id is not None and (not isinstance(evaluation.evaluation_id, str)
                                                  or not evaluation.evaluation_id):
        raise EvaluationGroupingError("evaluation_id must be a non-empty string")
    candidate = evaluation.candidate
    if not isinstance(candidate.policy_params, tuple):
        raise EvaluationGroupingError("portable policy_params must be an immutable tuple")
    try:
        candidate.scenario_key.validated()
        candidate.session_key.validated(schema)
        evaluation.observation_key.validated(schema)
    except (AttributeError, CandidatePlanError) as exc:
        raise EvaluationGroupingError("compiled evaluation contains an invalid portable key") from exc
    _object(candidate.candidate_json, label="candidate payload")
    overrides = _object(candidate.pool_overrides_json, label="pool overrides")
    exact_candidate = canonical_json_bytes(
        {"policy_params": list(candidate.policy_params), "pool_overrides": overrides}
    )
    if candidate.candidate_json != exact_candidate:
        raise EvaluationGroupingError("portable candidate has inconsistent exact evidence")
    _digest(candidate.candidate_sha256, label="candidate_sha256")
    if hashlib.sha256(candidate.candidate_json).hexdigest() != candidate.candidate_sha256:
        raise EvaluationGroupingError("candidate payload hash mismatch")
    try:
        schema.validate_candidate(candidate.policy_params, overrides)
    except CandidatePlanError as exc:
        raise EvaluationGroupingError("candidate does not match parameter schema") from exc
    evaluation.observation_key.request_fragment(schema)
    return evaluation


def group_evaluations(evaluations: Sequence[CompiledEvaluation], *, artifact_sha256: str,
                      parameter_schema: CandidateSchema) -> tuple[SessionGroup, ...]:
    """Validate first, then partition in stable input and observation order."""
    artifact_sha256 = _digest(artifact_sha256, label="artifact_sha256")
    if not isinstance(parameter_schema, CandidateSchema):
        raise TypeError("parameter_schema must be a CandidateSchema")
    _digest(parameter_schema.sha256, label="parameter_schema.sha256")
    values = tuple(evaluations)
    validated = tuple(_validate_evaluation(item, artifact_sha256, parameter_schema)
                      for item in values)
    ordinals = [item.ordinal for item in validated if item.ordinal is not None]
    identifiers = [item.evaluation_id for item in validated if item.evaluation_id is not None]
    if len(ordinals) != len(set(ordinals)):
        raise EvaluationGroupingError("duplicate evaluation ordinal")
    if len(identifiers) != len(set(identifiers)):
        raise EvaluationGroupingError("duplicate evaluation_id")

    grouped: dict[tuple[str, str], list[CompiledEvaluation]] = {}
    for item in validated:
        pair = (item.candidate.scenario_key.sha256, item.candidate.session_key.sha256)
        grouped.setdefault(pair, []).append(item)
    result = []
    for items in grouped.values():
        observations: dict[str, list[CompiledEvaluation]] = {}
        for item in items:
            observations.setdefault(item.observation_key.sha256, []).append(item)
        first = items[0]
        key = SessionGroupKey.create(artifact_sha256, parameter_schema,
                                     first.candidate.scenario_key,
                                     first.candidate.session_key).validated()
        result.append(SessionGroup(
            key, artifact_sha256, parameter_schema, first.candidate.scenario_key,
            first.candidate.session_key,
            tuple(ObservationGroup(rows[0].observation_key, tuple(rows))
                  for rows in observations.values()),
            tuple(items),
        ))
    return tuple(result)

def bind_local_session_group(group: SessionGroup, materialization: LocalSessionMaterialization,
                             ) -> LocalSessionGroupBinding:
    """Attest every plan to one exact local request without opening a client."""
    if not isinstance(group, SessionGroup):
        raise TypeError("group must be a SessionGroup")
    if not isinstance(materialization, LocalSessionMaterialization):
        raise TypeError("materialization must be a LocalSessionMaterialization")
    group.key.validated()
    rebuilt = group_evaluations(group.evaluations, artifact_sha256=group.artifact_sha256,
                                parameter_schema=group.parameter_schema)
    if rebuilt != (group,):
        raise EvaluationGroupingError("SessionGroup does not match its portable evidence")
    materialization.validated()
    if group.scenario_key != materialization.scenario_key:
        raise SessionMaterializationError("SessionGroup ScenarioKey does not match materialization")
    request = materialization.baseline_open_session_fields
    expected_fields = {
        item.lowering_path.removeprefix("open_session.")
        for item in group.parameter_schema.descriptors
        if item.lowering_path.startswith("open_session.")
    }
    if set(request) != expected_fields:
        raise SessionMaterializationError("materialized request fields do not match parameter schema")
    group.session_key.validated(group.parameter_schema)
    for name, value in group.session_key.open_session_values.items():
        descriptor = group.parameter_schema.descriptor(name)
        request[descriptor.lowering_path.removeprefix("open_session.")] = value
    try:
        request = group.parameter_schema.finalize_open_session(request)
    except CandidatePlanError as exc:
        raise SessionMaterializationError("cannot reconcile materialized YB aliases") from exc
    request_json = canonical_json_bytes(request)
    return LocalSessionGroupBinding(
        group, materialization, request_json, hashlib.sha256(request_json).hexdigest(),
        materialization.transport_receipt,
    )

__all__ = [
    "CompiledEvaluation", "EvaluationGroupingError", "LocalSessionGroupBinding",
    "ObservationGroup", "SessionGroup", "SessionGroupKey",
    "PortableCandidate",
    "bind_local_session_group", "decode_compiled_evaluation",
    "encode_compiled_evaluation", "group_evaluations",
]
