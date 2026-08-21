"""Description-driven lowering of named values to evaluator protocol v1 payloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import re
from typing import Any

from ..specs.common import SpecError, canonical_json_bytes
from ..specs.scenario import ScenarioClosure


class CandidatePlanError(ValueError):
    """The evaluator schema or a proposed value cannot be compiled safely."""


_MISSING = object()
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_TYPES = frozenset({"boolean", "enum", "integer", "object", "real", "real_pair", "string"})
_CLASSIFICATIONS = frozenset({"candidate", "session", "observation"})
_OPEN_SESSION_TRANSPORT_FIELDS = frozenset(
    {"session_id", "template_path", "template_sha256", "manifest_path", "manifest_sha256"}
)
_WIRES_BY_TYPE = {
    "boolean": frozenset({"json_boolean", "json_boolean_or_binary64"}),
    "enum": frozenset({"utf8"}),
    "integer": frozenset({"finite_binary64", "uint64", "uint64_or_milliseconds"}),
    "object": frozenset({"json_object", "json_object_binary64_boundary"}),
    "real": frozenset({"binary64", "binary64_fraction_or_1e10", "binary64_from_wad_1e18", "finite_binary64"}),
    "real_pair": frozenset({"binary64_from_wad_1e18"}),
    "string": frozenset({"lower_hex_64", "utf8"}),
}


def _decimal(value: object, *, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise CandidatePlanError(f"{label} must be a real number")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise CandidatePlanError(f"{label} must be a finite real number") from exc
    if not result.is_finite():
        raise CandidatePlanError(f"{label} must be finite")
    return result


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return tuple((str(key), _freeze(item)) for key, item in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _json_value(value: object, *, label: str) -> object:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, (float, Decimal)):
        number = float(_decimal(value, label=label))
        if not math.isfinite(number):
            raise CandidatePlanError(f"{label} is outside the binary64 domain")
        return number
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise CandidatePlanError(f"{label} object keys must be strings")
        return {key: _json_value(item, label=f"{label}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item, label=f"{label}[]") for item in value]
    raise CandidatePlanError(f"{label} is not a JSON value")


@dataclass(frozen=True, slots=True)
class ParameterDescriptor:
    name: str
    lowering_path: str
    value_type: str
    unit: str
    wire_representation: str
    classification: str
    order: int | None = None
    default: object = _MISSING
    minimum: object = _MISSING
    maximum: object = _MISSING
    quantum: object = _MISSING
    choices: tuple[str, ...] = ()

    @property
    def has_default(self) -> bool:
        return self.default is not _MISSING


@dataclass(frozen=True, slots=True)
class CandidateSchema:
    schema_version: str
    sha256: str
    policy_id: str
    descriptors: tuple[ParameterDescriptor, ...]

    @classmethod
    def from_description(cls, description: Mapping[str, Any]) -> CandidateSchema:
        if description.get("schema_version") != "curve_fx_evaluator_description_v1":
            raise CandidatePlanError("unsupported evaluator description schema")
        policy = description.get("policy")
        if not isinstance(policy, Mapping):
            raise CandidatePlanError("evaluator description omitted policy identity")
        if policy.get("descriptor_abi_version") != 1:
            raise CandidatePlanError("unsupported policy descriptor ABI")
        count = policy.get("parameter_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise CandidatePlanError("policy parameter_count must be a non-negative integer")
        policy_id = policy.get("id")
        if not isinstance(policy_id, str) or not policy_id:
            raise CandidatePlanError("policy id must be a non-empty string")

        schema = description.get("parameter_schema")
        if not isinstance(schema, Mapping) or schema.get("schema_version") != "curve_fx_parameter_schema_v1":
            raise CandidatePlanError("unsupported parameter schema")
        expected_sha = description.get("parameter_schema_sha256")
        if not isinstance(expected_sha, str) or not _SHA256_RE.fullmatch(expected_sha):
            raise CandidatePlanError("parameter_schema_sha256 must be lowercase SHA-256")
        canonical = description.get("parameter_schema_canonical_json")
        if not isinstance(canonical, str):
            raise CandidatePlanError("evaluator description omitted parameter_schema_canonical_json")
        if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != expected_sha:
            raise CandidatePlanError("parameter_schema_sha256 does not match canonical schema bytes")
        try:
            canonical_schema = json.loads(
                canonical,
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise CandidatePlanError("parameter_schema_canonical_json is not strict JSON") from exc
        if canonical_json_bytes(canonical_schema) != canonical_json_bytes(schema):
            raise CandidatePlanError("canonical parameter schema does not match parameter_schema")
        raw_parameters = schema.get("parameters")
        if not isinstance(raw_parameters, list):
            raise CandidatePlanError("parameter schema omitted its parameter list")

        descriptors = tuple(_parse_descriptor(item, index) for index, item in enumerate(raw_parameters))
        names = [item.name for item in descriptors]
        paths = [item.lowering_path for item in descriptors]
        if len(names) != len(set(names)):
            raise CandidatePlanError("parameter schema contains duplicate names")
        if len(paths) != len(set(paths)):
            raise CandidatePlanError("parameter schema contains duplicate lowering paths")

        policy_descriptors = [item for item in descriptors if item.order is not None]
        orders = [item.order for item in policy_descriptors]
        if len(orders) != len(set(orders)):
            raise CandidatePlanError("parameter schema contains duplicate policy orders")
        if sorted(orders) != list(range(count)) or len(policy_descriptors) != count:
            raise CandidatePlanError("policy descriptors do not match policy parameter_count")
        if any(item.name.startswith("policy.") != (item.order is not None) for item in descriptors):
            raise CandidatePlanError("policy names and ordered policy descriptors do not match")
        for item in policy_descriptors:
            assert item.order is not None
            expected_path = f"evaluate_batch.candidates[].policy_params[{item.order}]"
            if not item.name.startswith("policy.") or item.lowering_path != expected_path:
                raise CandidatePlanError(f"{item.name} has an invalid policy lowering path")
            if item.classification != "candidate" or item.wire_representation != "finite_binary64":
                raise CandidatePlanError(f"{item.name} has an invalid policy wire contract")
            _validate_domain(item, item.default, label=f"{item.name} default")
        return cls(str(schema["schema_version"]), expected_sha, policy_id, descriptors)

    def descriptor(self, name: str) -> ParameterDescriptor:
        for descriptor in self.descriptors:
            if descriptor.name == name:
                return descriptor
        raise CandidatePlanError(f"unknown proposal key: {name}")

    def validate_candidate(
        self, policy_params: Sequence[object], pool_overrides: Mapping[str, object]
    ) -> None:
        """Validate already-lowered candidate wire values against this schema."""
        if isinstance(policy_params, (str, bytes)) or not isinstance(policy_params, Sequence):
            raise CandidatePlanError("policy_params must be a sequence")
        policy = sorted(
            (item for item in self.descriptors if item.order is not None),
            key=lambda item: item.order if item.order is not None else -1,
        )
        if len(policy_params) != len(policy):
            raise CandidatePlanError("policy_params length does not match parameter schema")
        for descriptor, value in zip(policy, policy_params, strict=True):
            _validate_lowered(descriptor, value)
        if not isinstance(pool_overrides, Mapping):
            raise CandidatePlanError("pool_overrides must be an object")
        leaves = {
            tuple(item.lowering_path.removeprefix("pool_overrides.").split(".")): item
            for item in self.descriptors
            if item.classification == "candidate" and item.order is None
        }
        _validate_override_tree(pool_overrides, (), leaves)

    def finalize_open_session(self, request: Mapping[str, object]) -> dict[str, object]:
        """Reconcile protocol-v1 legacy aliases into harness-ready wire fields."""
        if not isinstance(request, Mapping) or any(not isinstance(key, str) for key in request):
            raise CandidatePlanError("open_session request must be an object with string keys")
        mode_path = "open_session.yb_mode"
        alias_path = "open_session.yb_releverage"
        relevant = tuple(
            item for item in self.descriptors
            if item.name in {"run.yb_mode", "run.yb_releverage"}
            or item.lowering_path in {mode_path, alias_path}
        )
        if not relevant:
            if "yb_mode" in request or "yb_releverage" in request:
                raise CandidatePlanError("YB session fields are not described by parameter schema")
            return dict(request)
        try:
            mode = self.descriptor("run.yb_mode")
            alias = self.descriptor("run.yb_releverage")
        except CandidatePlanError as exc:
            raise CandidatePlanError("parameter schema has incomplete YB alias descriptors") from exc
        if (
            len(relevant) != 2
            or mode.lowering_path != mode_path
            or mode.classification != "session"
            or mode.value_type != "enum"
            or mode.wire_representation != "utf8"
            or "off" not in mode.choices
            or alias.lowering_path != alias_path
            or alias.classification != "session"
            or alias.value_type != "boolean"
            or alias.wire_representation != "json_boolean"
            or alias.unit != "legacy_alias"
        ):
            raise CandidatePlanError("parameter schema has conflicting YB alias descriptors")
        if "yb_mode" not in request or "yb_releverage" not in request:
            raise CandidatePlanError("open_session omitted required YB alias fields")
        result = dict(request)
        _validate_lowered(mode, result["yb_mode"])
        result["yb_releverage"] = result["yb_mode"] != "off"
        _validate_lowered(alias, result["yb_releverage"])
        return result


def _parse_descriptor(value: object, index: int) -> ParameterDescriptor:
    if not isinstance(value, Mapping):
        raise CandidatePlanError(f"parameter descriptor {index} must be an object")
    required = ("name", "lowering_path", "type", "unit", "wire_representation", "classification")
    if any(not isinstance(value.get(key), str) or not value.get(key) for key in required):
        raise CandidatePlanError(f"parameter descriptor {index} is incomplete")
    value_type = str(value["type"])
    classification = str(value["classification"])
    wire = str(value["wire_representation"])
    if value_type not in _TYPES:
        raise CandidatePlanError(f"unknown parameter type: {value_type}")
    if classification not in _CLASSIFICATIONS:
        raise CandidatePlanError(f"unknown parameter classification: {classification}")
    if wire not in _WIRES_BY_TYPE[value_type]:
        raise CandidatePlanError(f"unknown {value_type} wire representation: {wire}")
    name = str(value["name"])
    path = str(value["lowering_path"])
    raw_choices = value.get("choices", _MISSING)
    if value_type == "enum":
        if (
            not isinstance(raw_choices, list)
            or not raw_choices
            or any(not isinstance(item, str) or not item for item in raw_choices)
            or len(raw_choices) != len(set(raw_choices))
        ):
            raise CandidatePlanError(f"{name} enum choices must be non-empty unique strings")
        choices = tuple(raw_choices)
    else:
        if raw_choices is not _MISSING:
            raise CandidatePlanError(f"{name} defines choices but is not an enum")
        choices = ()
    order_value = value.get("order", _MISSING)
    order: int | None = None
    if order_value is not _MISSING:
        if isinstance(order_value, bool) or not isinstance(order_value, int) or order_value < 0:
            raise CandidatePlanError(f"{name} has an invalid policy order")
        order = order_value
    if classification == "session" and not path.startswith("open_session."):
        raise CandidatePlanError(f"{name} has an invalid session lowering path")
    if classification == "candidate" and order is None and not path.startswith("pool_overrides."):
        raise CandidatePlanError(f"{name} has an invalid candidate lowering path")
    domain = {key: value.get(key, _MISSING) for key in ("default", "minimum", "maximum", "quantum")}
    if value_type == "integer":
        for key, item in domain.items():
            if item is _MISSING:
                continue
            number = _decimal(item, label=f"{name} {key}")
            if number != number.to_integral_value():
                raise CandidatePlanError(f"{name} {key} must be an integer")
            domain[key] = int(number)
    descriptor = ParameterDescriptor(
        name, path, value_type, str(value["unit"]), wire, classification, order, **domain,
        choices=choices,
    )
    if any(item is not _MISSING for item in (descriptor.minimum, descriptor.maximum, descriptor.quantum)):
        if _MISSING in (descriptor.minimum, descriptor.maximum, descriptor.quantum):
            raise CandidatePlanError(f"{name} has an incomplete numeric domain")
        if not descriptor.has_default:
            raise CandidatePlanError(f"{name} has bounds but no default")
        minimum = _decimal(descriptor.minimum, label=f"{name} minimum")
        maximum = _decimal(descriptor.maximum, label=f"{name} maximum")
        quantum = _decimal(descriptor.quantum, label=f"{name} quantum")
        if minimum > maximum or quantum < 0:
            raise CandidatePlanError(f"{name} has an inconsistent numeric domain")
    if descriptor.has_default:
        _lower(descriptor, descriptor.default)
    return descriptor


def _validate_domain(descriptor: ParameterDescriptor, value: object, *, label: str) -> None:
    if descriptor.value_type == "real":
        number = _decimal(value, label=label)
    elif descriptor.value_type == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise CandidatePlanError(f"{label} must be an integer")
        number = Decimal(value)
    else:
        return
    if descriptor.minimum is _MISSING:
        return
    minimum = _decimal(descriptor.minimum, label=f"{descriptor.name} minimum")
    maximum = _decimal(descriptor.maximum, label=f"{descriptor.name} maximum")
    quantum = _decimal(descriptor.quantum, label=f"{descriptor.name} quantum")
    if not minimum <= number <= maximum:
        raise CandidatePlanError(f"{descriptor.name} value {value!r} is outside [{minimum}, {maximum}]")
    if quantum and (number - minimum) % quantum:
        raise CandidatePlanError(f"{descriptor.name} value {value!r} is off quantum {quantum}")


def _scaled_binary64(number: Decimal, exponent: int, *, label: str, lattice: str) -> int | float:
    parts = number.as_tuple()
    scaled = Decimal((parts.sign, parts.digits, parts.exponent + exponent))
    if scaled != scaled.to_integral_value():
        raise CandidatePlanError(f"{label} is finer than the {lattice} lattice")
    # Identity follows the protocol's binary64 materialization, including collisions.
    binary64 = float(scaled)
    if not math.isfinite(binary64):
        raise CandidatePlanError(f"{label} is outside the binary64 domain")
    if binary64.is_integer() and abs(binary64) <= 2**53:
        return int(binary64)
    return binary64


def _lower(descriptor: ParameterDescriptor, value: object) -> object:
    label = descriptor.name
    _validate_domain(descriptor, value, label=label)
    value_type = descriptor.value_type
    if value_type == "boolean":
        if not isinstance(value, bool):
            raise CandidatePlanError(f"{label} must be a boolean")
        return value
    if value_type in {"string", "enum"}:
        if not isinstance(value, str):
            raise CandidatePlanError(f"{label} must be a string")
        if value_type == "enum" and value not in descriptor.choices:
            raise CandidatePlanError(f"{label} must be one of {', '.join(descriptor.choices)}")
        if descriptor.wire_representation == "lower_hex_64" and not _SHA256_RE.fullmatch(value):
            raise CandidatePlanError(f"{label} must be lowercase SHA-256")
        return value
    if value_type == "integer":
        assert isinstance(value, int) and not isinstance(value, bool)
        if descriptor.wire_representation.startswith("uint64") and not 0 <= value < 2**64:
            raise CandidatePlanError(f"{label} is outside the uint64 domain")
        return value
    if value_type == "object":
        if not isinstance(value, Mapping):
            raise CandidatePlanError(f"{label} must be an object")
        return _json_value(value, label=label)
    values: Sequence[object]
    if value_type == "real_pair":
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
            raise CandidatePlanError(f"{label} must contain exactly two real values")
        values = value
    else:
        values = (value,)
    lowered: list[object] = []
    for item in values:
        number = _decimal(item, label=label)
        if descriptor.wire_representation == "binary64_fraction_or_1e10":
            lowered.append(_scaled_binary64(number, 10, label=label, lattice="1e-10 fee"))
        elif descriptor.wire_representation == "binary64_from_wad_1e18":
            lowered.append(_scaled_binary64(number, 18, label=label, lattice="1e-18 WAD"))
        else:
            binary64 = float(number)
            if not math.isfinite(binary64):
                raise CandidatePlanError(f"{label} is outside the binary64 domain")
            lowered.append(binary64)
    return lowered if value_type == "real_pair" else lowered[0]


def _set_path(target: dict[str, Any], path: Sequence[str], value: object, *, label: str) -> None:
    cursor = target
    for component in path[:-1]:
        existing = cursor.setdefault(component, {})
        if not isinstance(existing, dict):
            raise CandidatePlanError(f"{label} collides with another lowering path")
        cursor = existing
    leaf = path[-1]
    if leaf in cursor:
        raise CandidatePlanError(f"{label} collides with another lowering path")
    cursor[leaf] = value


def _identity_descriptors(
    schema: CandidateSchema, *, prefix: str
) -> tuple[ParameterDescriptor, ...]:
    descriptors = []
    for descriptor in schema.descriptors:
        if not descriptor.lowering_path.startswith(prefix) or (
            prefix == "evaluate_batch." and descriptor.classification != "observation"
        ):
            continue
        field = descriptor.lowering_path.removeprefix(prefix)
        if prefix == "open_session." and (
            field in _OPEN_SESSION_TRANSPORT_FIELDS or descriptor.unit == "legacy_alias"
        ):
            continue
        if prefix == "evaluate_batch." and field == "observation.artifact_dir":
            continue
        descriptors.append(descriptor)
    return tuple(descriptors)


def _validate_lowered(descriptor: ParameterDescriptor, value: object) -> None:
    if descriptor.value_type == "boolean":
        valid = isinstance(value, bool)
    elif descriptor.value_type in {"string", "enum"}:
        valid = isinstance(value, str) and (
            descriptor.value_type != "enum" or value in descriptor.choices
        )
        if valid and descriptor.wire_representation == "lower_hex_64":
            valid = bool(_SHA256_RE.fullmatch(value))
    elif descriptor.value_type == "integer":
        valid = isinstance(value, int) and not isinstance(value, bool)
        if valid and descriptor.wire_representation.startswith("uint64"):
            valid = 0 <= value < 2**64
    elif descriptor.value_type == "object":
        valid = isinstance(value, Mapping)
    elif descriptor.value_type == "real_pair":
        valid = (
            isinstance(value, list)
            and len(value) == 2
            and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)
        )
    else:
        valid = isinstance(value, (int, float)) and not isinstance(value, bool)
    if not valid:
        raise CandidatePlanError(f"{descriptor.name} has an invalid lowered wire value")
    if descriptor.value_type in {"integer", "real"}:
        semantic: object = value
        if descriptor.value_type == "real" and descriptor.wire_representation in {
            "binary64_fraction_or_1e10", "binary64_from_wad_1e18"
        }:
            exponent = 10 if descriptor.wire_representation.endswith("1e10") else 18
            semantic = _decimal(value, label=descriptor.name).scaleb(-exponent)
        _validate_domain(descriptor, semantic, label=descriptor.name)
        canonical = _lower(descriptor, semantic)
        if canonical != value or type(canonical) is not type(value):
            raise CandidatePlanError(f"{descriptor.name} is not a canonical lowered wire value")
    elif descriptor.value_type == "real_pair":
        semantic_pair = tuple(
            _decimal(item, label=descriptor.name).scaleb(-18) for item in value
        )
        canonical_pair = _lower(descriptor, semantic_pair)
        if canonical_pair != value or any(
            type(expected) is not type(actual)
            for expected, actual in zip(canonical_pair, value, strict=True)
        ):
            raise CandidatePlanError(f"{descriptor.name} is not a canonical lowered wire value")
    try:
        canonical_json_bytes(value)
    except (TypeError, ValueError) as exc:
        raise CandidatePlanError(f"{descriptor.name} has an invalid lowered wire value") from exc


def _validate_override_tree(
    value: object,
    path: tuple[str, ...],
    leaves: Mapping[tuple[str, ...], ParameterDescriptor],
) -> None:
    descriptor = leaves.get(path)
    if descriptor is not None:
        _validate_lowered(descriptor, value)
        return
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise CandidatePlanError(f"unknown pool override leaf: {'.'.join(path) or '<root>'}")
    if path and not value:
        raise CandidatePlanError(f"unknown pool override structure: {'.'.join(path)}")
    prefixes = {leaf[:len(path) + 1] for leaf in leaves if leaf[:len(path)] == path}
    for key, item in value.items():
        child = (*path, key)
        if child not in prefixes:
            raise CandidatePlanError(f"unknown pool override leaf: {'.'.join(child)}")
        _validate_override_tree(item, child, leaves)


def _key_payload(identity_json: bytes, *, version: str, label: str) -> dict[str, Any]:
    if not isinstance(identity_json, bytes):
        raise CandidatePlanError(f"{label} identity_json must be bytes")
    try:
        payload = json.loads(
            identity_json,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise CandidatePlanError(f"{label} identity is not strict JSON") from exc
    if not isinstance(payload, dict) or canonical_json_bytes(payload) != identity_json:
        raise CandidatePlanError(f"{label} identity must be a canonical JSON object")
    expected = {"version", "parameter_schema_sha256", label}
    if set(payload) != expected or payload.get("version") != version:
        raise CandidatePlanError(f"{label} identity has an invalid shape or version")
    if not isinstance(payload[label], dict) or any(
        not isinstance(name, str) for name in payload[label]
    ):
        raise CandidatePlanError(f"{label} identity parameters must be an object")
    return payload


@dataclass(frozen=True, slots=True)
class SessionKey:
    identity_json: bytes
    sha256: str

    @classmethod
    def from_request(cls, schema: CandidateSchema, request: Mapping[str, object]) -> SessionKey:
        values = {
            descriptor.name: request[descriptor.lowering_path.removeprefix("open_session.")]
            for descriptor in _identity_descriptors(schema, prefix="open_session.")
        }
        identity = canonical_json_bytes({
            "version": "curve_fx_session_key_v1",
            "parameter_schema_sha256": schema.sha256,
            "open_session": values,
        })
        return cls(identity, hashlib.sha256(identity).hexdigest())

    def validated(self, schema: CandidateSchema | None = None) -> SessionKey:
        payload = _key_payload(
            self.identity_json, version="curve_fx_session_key_v1", label="open_session"
        )
        if not isinstance(self.sha256, str) or not _SHA256_RE.fullmatch(self.sha256):
            raise CandidatePlanError("SessionKey sha256 must be lowercase SHA-256")
        if hashlib.sha256(self.identity_json).hexdigest() != self.sha256:
            raise CandidatePlanError("SessionKey hash does not match its canonical identity")
        schema_sha = payload["parameter_schema_sha256"]
        if not isinstance(schema_sha, str) or not _SHA256_RE.fullmatch(schema_sha):
            raise CandidatePlanError("SessionKey parameter schema must be lowercase SHA-256")
        if schema is not None:
            if schema_sha != schema.sha256:
                raise CandidatePlanError("SessionKey parameter schema mismatch")
            descriptors = _identity_descriptors(schema, prefix="open_session.")
            if set(payload["open_session"]) != {item.name for item in descriptors}:
                raise CandidatePlanError("SessionKey open_session membership mismatch")
            for descriptor in descriptors:
                _validate_lowered(descriptor, payload["open_session"][descriptor.name])
        return self

    @property
    def open_session_values(self) -> dict[str, object]:
        return dict(_key_payload(
            self.identity_json, version="curve_fx_session_key_v1", label="open_session"
        )["open_session"])


@dataclass(frozen=True, slots=True)
class ObservationKey:
    identity_json: bytes
    sha256: str

    def validated(self, schema: CandidateSchema) -> ObservationKey:
        payload = _key_payload(
            self.identity_json, version="curve_fx_observation_key_v1", label="evaluate_batch"
        )
        if not isinstance(self.sha256, str) or not _SHA256_RE.fullmatch(self.sha256):
            raise CandidatePlanError("ObservationKey sha256 must be lowercase SHA-256")
        if hashlib.sha256(self.identity_json).hexdigest() != self.sha256:
            raise CandidatePlanError("ObservationKey hash does not match its canonical identity")
        if payload["parameter_schema_sha256"] != schema.sha256:
            raise CandidatePlanError("ObservationKey parameter schema mismatch")
        descriptors = _identity_descriptors(schema, prefix="evaluate_batch.")
        if set(payload["evaluate_batch"]) != {item.name for item in descriptors}:
            raise CandidatePlanError("ObservationKey evaluate_batch membership mismatch")
        for descriptor in descriptors:
            _validate_lowered(descriptor, payload["evaluate_batch"][descriptor.name])
        return self

    def request_fragment(self, schema: CandidateSchema) -> dict[str, object]:
        self.validated(schema)
        values = _key_payload(
            self.identity_json, version="curve_fx_observation_key_v1", label="evaluate_batch"
        )["evaluate_batch"]
        fragment: dict[str, Any] = {}
        for descriptor in _identity_descriptors(schema, prefix="evaluate_batch."):
            _set_path(
                fragment,
                descriptor.lowering_path.removeprefix("evaluate_batch.").split("."),
                values[descriptor.name],
                label=descriptor.name,
            )
        return fragment


@dataclass(frozen=True, slots=True)
class ScenarioKey:
    identity_json: bytes
    sha256: str

    @classmethod
    def from_closure(cls, closure: ScenarioClosure) -> ScenarioKey:
        if not isinstance(closure, ScenarioClosure):
            raise CandidatePlanError("scenario must be a verified ScenarioClosure")
        identity_json = canonical_json_bytes(closure.to_identity())
        return cls(identity_json=identity_json, sha256=closure.sha256)

    def validated(self) -> ScenarioKey:
        if not isinstance(self.identity_json, bytes):
            raise CandidatePlanError("ScenarioKey identity_json must be bytes")
        if not isinstance(self.sha256, str) or not _SHA256_RE.fullmatch(self.sha256):
            raise CandidatePlanError("ScenarioKey sha256 must be lowercase SHA-256")
        try:
            identity = json.loads(self.identity_json)
            if not isinstance(identity, Mapping):
                raise CandidatePlanError("ScenarioKey identity must be an object")
            closure = ScenarioClosure.from_dict(identity)
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError, SpecError) as exc:
            raise CandidatePlanError("ScenarioKey is not a ScenarioClosure identity") from exc
        expected = ScenarioKey.from_closure(closure)
        if self != expected:
            raise CandidatePlanError("ScenarioKey bytes or hash do not match ScenarioClosure")
        return self


@dataclass(frozen=True, slots=True)
class CandidatePlan:
    scenario_key: ScenarioKey
    session_key: SessionKey
    session_request_json: bytes
    policy_params: tuple[object, ...]
    pool_overrides_json: bytes
    candidate_json: bytes
    candidate_sha256: str
    named_values: tuple[tuple[str, object], ...]

    @property
    def session_request(self) -> dict[str, Any]:
        return json.loads(self.session_request_json)

    @property
    def pool_overrides(self) -> dict[str, Any]:
        return json.loads(self.pool_overrides_json)

    @property
    def candidate_payload(self) -> dict[str, Any]:
        return json.loads(self.candidate_json)


@dataclass(frozen=True, slots=True)
class CandidateCompiler:
    schema: CandidateSchema

    @classmethod
    def from_description(cls, description: Mapping[str, Any]) -> CandidateCompiler:
        return cls(CandidateSchema.from_description(description))

    def compile(
        self,
        proposal: Mapping[str, object],
        *,
        open_session: Mapping[str, object],
        scenario: ScenarioClosure | ScenarioKey | None = None,
        scenario_identity: Mapping[str, object] | None = None,
    ) -> CandidatePlan:
        """Compile against a verified scenario; raw identity is legacy-only."""
        if scenario_identity is not None:
            if scenario is not None:
                raise CandidatePlanError("pass scenario or scenario_identity, not both")
            return self.compile_legacy(
                proposal,
                open_session=open_session,
                scenario_identity=scenario_identity,
            )
        if isinstance(scenario, ScenarioClosure):
            scenario_key = ScenarioKey.from_closure(scenario)
        elif isinstance(scenario, ScenarioKey):
            scenario_key = scenario.validated()
        else:
            raise CandidatePlanError("scenario must be a ScenarioClosure or validated ScenarioKey")
        return self._compile(proposal, open_session=open_session, scenario_key=scenario_key)

    def compile_legacy(
        self,
        proposal: Mapping[str, object],
        *,
        open_session: Mapping[str, object],
        scenario_identity: Mapping[str, object],
    ) -> CandidatePlan:
        """Compatibility path for pre-closure callers with caller-owned identity."""
        if not isinstance(scenario_identity, Mapping):
            raise CandidatePlanError("scenario_identity must be a mapping")
        scenario_json = canonical_json_bytes(
            _json_value(scenario_identity, label="scenario_identity")
        )
        return self._compile(
            proposal,
            open_session=open_session,
            scenario_key=ScenarioKey(
                identity_json=scenario_json,
                sha256=hashlib.sha256(scenario_json).hexdigest(),
            ),
        )

    def compile_observation(
        self, values: Mapping[str, object] | None = None
    ) -> tuple[ObservationKey, bytes]:
        """Lower evaluate_batch semantics, excluding execution-local artifact paths."""
        supplied = {} if values is None else dict(values)
        descriptors = _identity_descriptors(self.schema, prefix="evaluate_batch.")
        allowed = {item.name for item in descriptors}
        if any(not isinstance(name, str) for name in supplied):
            raise CandidatePlanError("observation keys must be strings")
        unknown = sorted(set(supplied) - allowed)
        if unknown:
            raise CandidatePlanError(f"unknown observation key: {', '.join(unknown)}")
        lowered: dict[str, object] = {}
        fragment: dict[str, Any] = {}
        for descriptor in descriptors:
            value = supplied.get(descriptor.name, descriptor.default)
            if value is _MISSING:
                raise CandidatePlanError(f"observation is missing required field: {descriptor.name}")
            wire_value = _lower(descriptor, value)
            lowered[descriptor.name] = wire_value
            _set_path(
                fragment,
                descriptor.lowering_path.removeprefix("evaluate_batch.").split("."),
                wire_value,
                label=descriptor.name,
            )
        identity_json = canonical_json_bytes({
            "version": "curve_fx_observation_key_v1",
            "parameter_schema_sha256": self.schema.sha256,
            "evaluate_batch": lowered,
        })
        return (
            ObservationKey(identity_json, hashlib.sha256(identity_json).hexdigest()),
            canonical_json_bytes(fragment),
        )

    def _compile(
        self,
        proposal: Mapping[str, object],
        *,
        open_session: Mapping[str, object],
        scenario_key: ScenarioKey,
    ) -> CandidatePlan:
        if not isinstance(proposal, Mapping) or not isinstance(open_session, Mapping):
            raise CandidatePlanError("proposal and open_session must be mappings")

        proposed: dict[str, object] = {}
        for name, value in proposal.items():
            if not isinstance(name, str):
                raise CandidatePlanError("proposal keys must be strings")
            descriptor = self.schema.descriptor(name)
            if descriptor.name == "run.session_id" or descriptor.unit in {"path", "sha256"}:
                raise CandidatePlanError(
                    f"transport materialization field cannot be proposed: {name}"
                )
            if descriptor.unit == "legacy_alias":
                raise CandidatePlanError(f"legacy alias cannot be optimized: {name}")
            if (
                descriptor.classification == "observation"
                and descriptor.lowering_path.startswith("evaluate_batch.")
            ):
                raise CandidatePlanError(f"observation-only parameter cannot be optimized: {name}")
            proposed[name] = value

        open_session_descriptors = [
            item for item in self.schema.descriptors if item.lowering_path.startswith("open_session.")
        ]
        by_field = {
            item.lowering_path.removeprefix("open_session."): item
            for item in open_session_descriptors
        }
        if any(not isinstance(field, str) for field in open_session):
            raise CandidatePlanError("open_session keys must be strings")
        unknown_base = sorted(set(open_session) - set(by_field))
        if unknown_base:
            raise CandidatePlanError(f"open_session contains unknown fields: {', '.join(unknown_base)}")

        request: dict[str, object] = {}
        for descriptor in open_session_descriptors:
            field = descriptor.lowering_path.removeprefix("open_session.")
            value = proposed.get(descriptor.name, open_session.get(field, descriptor.default))
            if value is _MISSING:
                raise CandidatePlanError(f"open_session is missing required field: {field}")
            lowered = _lower(descriptor, value)
            request[field] = lowered
        request = self.schema.finalize_open_session(request)

        policy_descriptors = sorted(
            (item for item in self.schema.descriptors if item.order is not None),
            key=lambda item: item.order if item.order is not None else -1,
        )
        policy_params = tuple(
            _lower(item, proposed.get(item.name, item.default)) for item in policy_descriptors
        )
        pool_overrides: dict[str, Any] = {}
        for descriptor in self.schema.descriptors:
            if descriptor.classification != "candidate" or descriptor.order is not None:
                continue
            value = proposed.get(descriptor.name, descriptor.default)
            if value is _MISSING:
                continue
            path = descriptor.lowering_path.split(".")[1:]
            _set_path(pool_overrides, path, _lower(descriptor, value), label=descriptor.name)

        request_json = canonical_json_bytes(request)
        session_key = SessionKey.from_request(self.schema, request)
        payload = {"policy_params": list(policy_params), "pool_overrides": pool_overrides}
        candidate_json = canonical_json_bytes(payload)
        named_values = dict(proposed)
        for descriptor in policy_descriptors:
            named_values.setdefault(descriptor.name, descriptor.default)
        return CandidatePlan(
            scenario_key=scenario_key,
            session_key=session_key,
            session_request_json=request_json,
            policy_params=policy_params,
            pool_overrides_json=canonical_json_bytes(pool_overrides),
            candidate_json=candidate_json,
            candidate_sha256=hashlib.sha256(candidate_json).hexdigest(),
            named_values=tuple((name, _freeze(named_values[name])) for name in sorted(named_values)),
        )


__all__ = [
    "CandidateCompiler",
    "CandidatePlan",
    "CandidatePlanError",
    "CandidateSchema",
    "ObservationKey",
    "ParameterDescriptor",
    "ScenarioKey",
    "SessionKey",
]
