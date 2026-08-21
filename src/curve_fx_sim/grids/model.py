"""Canonical lazy Cartesian grid planning."""

from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, MutableMapping

from ..specs.common import canonical_json_bytes
from ..specs.grid import AxisSpec, AxisTarget, GridSpec

if TYPE_CHECKING:
    from ..evaluation.grouping import CompiledEvaluation
    from ..evaluation.plans import CandidateCompiler, ObservationKey, ScenarioKey, SessionKey
    from ..specs.scenario import ScenarioClosure


GRID_PLAN_SCHEMA_VERSION = "fxsim_cartesian_grid_v1"
_AxisOption = tuple[tuple[tuple[str, Any], ...], tuple[tuple[str, Any], ...]]


class GridValidationError(ValueError):
    """A grid does not define one exact finite Cartesian product."""


def _exact_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else format(value, "f")
    if isinstance(value, Mapping):
        return {str(key): _exact_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_exact_value(item) for item in value]
    return value


def coordinate_signature(coordinate: Mapping[str, Any]) -> str:
    """Canonical exact-coordinate equality key."""

    def normalize(value: Any) -> Any:
        if isinstance(value, bool) or value is None:
            return ["value", value]
        if isinstance(value, (int, float, Decimal, str)):
            try:
                number = Decimal(str(value))
                if number.is_finite():
                    return ["decimal", "0" if number == 0 else str(number.normalize())]
            except (InvalidOperation, ValueError):
                pass
        if isinstance(value, Mapping):
            return ["mapping", [[str(key), normalize(item)] for key, item in sorted(value.items())]]
        if isinstance(value, (tuple, list)):
            return ["sequence", [normalize(item) for item in value]]
        return ["value", str(value)]

    return json.dumps(
        [[str(key), normalize(value)] for key, value in sorted(coordinate.items())],
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple((str(key), _freeze(item)) for key, item in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, tuple):
        if all(
            isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
            for item in value
        ):
            return {key: _thaw(item) for key, item in value}
        return [_thaw(item) for item in value]
    return value


def _immutable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _immutable(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_immutable(item) for item in value)
    return value


def _key_record(key: Any) -> dict[str, str]:
    return {"sha256": key.sha256, "identity_json": key.identity_json.decode("utf-8")}


def _load_key(value: Any, key_type: type[Any], label: str) -> Any:
    if not isinstance(value, Mapping) or set(value) != {"sha256", "identity_json"}:
        raise GridValidationError(f"grid plan {label} identity is incomplete")
    digest, identity = value["sha256"], value["identity_json"]
    if not isinstance(digest, str) or not isinstance(identity, str):
        raise GridValidationError(f"grid plan {label} identity is invalid")
    if hashlib.sha256(identity.encode()).hexdigest() != digest:
        raise GridValidationError(f"grid plan {label} identity hash mismatch")
    return key_type(identity.encode(), digest)


def _semantic_wire_value(descriptor: Any, value: Any) -> Any:
    if descriptor.wire_representation == "binary64_fraction_or_1e10":
        return Decimal(str(value)).scaleb(-10)
    if descriptor.wire_representation == "binary64_from_wad_1e18":
        if descriptor.value_type == "real_pair":
            return tuple(Decimal(str(item)).scaleb(-18) for item in value)
        return Decimal(str(value)).scaleb(-18)
    return value


def _baseline_open_session(compiler: CandidateCompiler, key: SessionKey) -> dict[str, Any]:
    """Provide compiler-only placeholders; execution must bind fresh local materialization."""
    identity_values = key.open_session_values
    request: dict[str, Any] = {}
    for descriptor in compiler.schema.descriptors:
        prefix = "open_session."
        if not descriptor.lowering_path.startswith(prefix):
            continue
        field = descriptor.lowering_path.removeprefix(prefix)
        if descriptor.name in identity_values:
            request[field] = _semantic_wire_value(descriptor, identity_values[descriptor.name])
        elif field == "session_id":
            request[field] = "cartesian_grid_plan"
        elif descriptor.unit == "path":
            request[field] = "cartesian_grid_plan"
        elif descriptor.unit == "sha256":
            request[field] = "0" * 64
        elif descriptor.unit == "legacy_alias":
            request[field] = False
        elif descriptor.has_default:
            request[field] = descriptor.default
        else:
            raise GridValidationError(f"cannot reconstruct baseline field {field!r}")
    return request


def _split_proposal(
    compiler: CandidateCompiler, proposal: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    observation: dict[str, Any] = {}
    candidate: dict[str, Any] = {}
    for name, value in proposal.items():
        descriptor = compiler.schema.descriptor(name)
        target = observation if (
            descriptor.classification == "observation"
            and descriptor.lowering_path.startswith("evaluate_batch.")
        ) else candidate
        target[name] = value
    return candidate, observation


@dataclass(frozen=True, slots=True)
class GridPoint:
    """One portable point reconstructed from a CartesianGridPlan ordinal."""

    ordinal: int
    candidate_id: str
    coordinate_indices: tuple[int, ...]
    coordinates: Mapping[str, Any]
    evaluation: CompiledEvaluation
    coordinate_signature: str
    proposal: tuple[tuple[str, Any], ...]
    session_group_id: str

    @property
    def policy_params(self) -> tuple[object, ...]:
        return self.evaluation.candidate.policy_params

    @property
    def pool_overrides(self) -> Mapping[str, Any]:
        return json.loads(self.evaluation.candidate.pool_overrides_json)

    @property
    def proposal_dict(self) -> dict[str, Any]:
        return {name: _thaw(value) for name, value in self.proposal}


@dataclass(frozen=True, slots=True)
class CartesianGridPlan:
    """Axes-only authority that compiles a portable ordinal on demand.

    Points deliberately contain no runnable session request. Execution must
    bind their session groups against a fresh LocalSessionMaterialization.
    """

    grid_id: str
    axis_options: tuple[tuple[_AxisOption, ...], ...]
    static_proposal: tuple[tuple[str, Any], ...]
    artifact_sha256: str
    parameter_schema_sha256: str
    scenario_key: ScenarioKey
    baseline_session_key: SessionKey
    baseline_observation_key: ObservationKey
    coordinate_shape: tuple[int, ...]
    pool_count: int
    plan_sha256: str
    compiler: CandidateCompiler

    def __len__(self) -> int:
        return self.pool_count

    def __getitem__(self, index: int | slice) -> GridPoint:
        if isinstance(index, slice):
            raise TypeError("CartesianGridPlan does not support materializing slices")
        if index < 0:
            index += self.pool_count
        return self.point_at(index)

    def __iter__(self) -> Iterator[GridPoint]:
        return (self.point_at(ordinal) for ordinal in range(self.pool_count))

    def candidate_id_at(self, ordinal: int) -> str:
        self.coordinate_indices_at(ordinal)
        return f"grid_{self.grid_id}_p{ordinal:06d}"

    def coordinate_indices_at(self, ordinal: int) -> tuple[int, ...]:
        if isinstance(ordinal, bool) or not isinstance(ordinal, int):
            raise TypeError("grid ordinal must be an integer")
        if ordinal < 0 or ordinal >= self.pool_count:
            raise IndexError(f"grid ordinal {ordinal} outside [0, {self.pool_count})")
        remainder = ordinal
        indices = [0] * len(self.coordinate_shape)
        for axis_index in range(len(indices) - 1, -1, -1):
            indices[axis_index] = remainder % self.coordinate_shape[axis_index]
            remainder //= self.coordinate_shape[axis_index]
        return tuple(indices)

    def coordinates_at(self, ordinal: int) -> dict[str, Any]:
        coordinates: dict[str, Any] = {}
        for axis_index, option_index in enumerate(self.coordinate_indices_at(ordinal)):
            display, _ = self.axis_options[axis_index][option_index]
            coordinates.update(_thaw(display))
        return coordinates

    def ordinal_at(self, coordinates: Mapping[str, Any]) -> int:
        """Return the exact canonical ordinal; no nearest or tolerance lookup."""
        if not isinstance(coordinates, Mapping):
            raise TypeError("grid coordinates must be a mapping")
        remaining = dict(coordinates)
        indices: list[int] = []
        for options in self.axis_options:
            names = tuple(name for name, _ in options[0][0])
            if any(name not in remaining for name in names):
                raise GridValidationError("coordinate does not contain every grid axis")
            partial = {name: remaining.pop(name) for name in names}
            signature = coordinate_signature(partial)
            matches = [
                index
                for index, (display, _) in enumerate(options)
                if coordinate_signature(_thaw(display)) == signature
            ]
            if len(matches) != 1:
                raise GridValidationError("coordinate is not an exact grid point")
            indices.append(matches[0])
        if remaining:
            raise GridValidationError("coordinate contains fields outside the grid axes")
        ordinal = 0
        for index, size in zip(indices, self.coordinate_shape, strict=True):
            ordinal = ordinal * size + index
        return ordinal

    def iter_points(self, ranges: Sequence[tuple[int, int]]) -> Iterator[GridPoint]:
        """Compile sorted, non-overlapping half-open ordinal ranges lazily."""
        previous_end = 0
        for start, end in ranges:
            if (
                isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, int)
                or not isinstance(end, int)
                or start < previous_end
                or start < 0
                or end <= start
                or end > self.pool_count
            ):
                raise GridValidationError("grid ranges must be sorted valid half-open ranges")
            yield from (self.point_at(ordinal) for ordinal in range(start, end))
            previous_end = end

    def point_at(self, ordinal: int) -> GridPoint:
        """Compile one mixed-radix ordinal; the final axis changes fastest."""
        from ..evaluation.grouping import CompiledEvaluation, SessionGroupKey
        from ..evaluation.plans import CandidatePlanError

        if isinstance(ordinal, bool) or not isinstance(ordinal, int):
            raise TypeError("grid ordinal must be an integer")
        if ordinal < 0 or ordinal >= self.pool_count:
            raise IndexError(f"grid ordinal {ordinal} outside [0, {self.pool_count})")
        indices = self.coordinate_indices_at(ordinal)

        coordinates: dict[str, Any] = {}
        proposal = {name: _thaw(value) for name, value in self.static_proposal}
        for axis_index, option_index in enumerate(indices):
            display, update = self.axis_options[axis_index][option_index]
            coordinates.update(_thaw(display))
            proposal.update(_thaw(update))
        candidate_proposal, observation_proposal = _split_proposal(self.compiler, proposal)
        candidate_id = self.candidate_id_at(ordinal)
        try:
            candidate = self.compiler.compile(
                candidate_proposal,
                open_session=_baseline_open_session(self.compiler, self.baseline_session_key),
                scenario=self.scenario_key,
            )
            evaluation = CompiledEvaluation.from_plan(
                candidate,
                compiler=self.compiler,
                artifact_sha256=self.artifact_sha256,
                observation=observation_proposal,
                ordinal=ordinal,
                evaluation_id=candidate_id,
            )
        except CandidatePlanError as exc:
            raise GridValidationError(f"grid point {candidate_id} is invalid: {exc}") from exc
        group_id = SessionGroupKey.create(
            self.artifact_sha256,
            self.compiler.schema,
            evaluation.candidate.scenario_key,
            evaluation.candidate.session_key,
        ).validated().sha256
        return GridPoint(
            ordinal=ordinal,
            candidate_id=candidate_id,
            coordinate_indices=indices,
            coordinates=_immutable(coordinates),
            evaluation=evaluation,
            coordinate_signature=coordinate_signature(coordinates),
            proposal=tuple((name, _freeze(value)) for name, value in sorted(proposal.items())),
            session_group_id=group_id,
        )

    def authority_dict(self) -> dict[str, Any]:
        axes = []
        for options in self.axis_options:
            axes.append([
                {"display": _exact_value(_thaw(display)), "proposal": _exact_value(_thaw(proposal))}
                for display, proposal in options
            ])
        return {
            "schema_version": GRID_PLAN_SCHEMA_VERSION,
            "grid_id": self.grid_id,
            "axes": axes,
            "static_proposal": _exact_value(_thaw(self.static_proposal)),
            "artifact_sha256": self.artifact_sha256,
            "parameter_schema_sha256": self.parameter_schema_sha256,
            "scenario_key": _key_record(self.scenario_key),
            "baseline_session_key": _key_record(self.baseline_session_key),
            "baseline_observation_key": _key_record(self.baseline_observation_key),
            "coordinate_shape": list(self.coordinate_shape),
            "pool_count": self.pool_count,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.authority_dict() | {"plan_sha256": self.plan_sha256}

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        compiler: CandidateCompiler,
        artifact_sha256: str,
        scenario: ScenarioClosure | ScenarioKey,
    ) -> CartesianGridPlan:
        from ..evaluation.plans import ObservationKey, ScenarioKey, SessionKey

        expected = {
            "schema_version", "grid_id", "axes", "static_proposal", "artifact_sha256",
            "parameter_schema_sha256", "scenario_key", "baseline_session_key",
            "baseline_observation_key", "coordinate_shape", "pool_count", "plan_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise GridValidationError("grid plan fields do not match fxsim_cartesian_grid_v1")
        authority = {key: value[key] for key in value if key != "plan_sha256"}
        digest = hashlib.sha256(canonical_json_bytes(authority)).hexdigest()
        if value["plan_sha256"] != digest:
            raise GridValidationError("grid plan SHA-256 mismatch")
        if value["schema_version"] != GRID_PLAN_SCHEMA_VERSION:
            raise GridValidationError(f"unsupported grid plan schema {value['schema_version']!r}")
        if value["artifact_sha256"] != artifact_sha256:
            raise GridValidationError("grid plan evaluator artifact mismatch")
        if value["parameter_schema_sha256"] != compiler.schema.sha256:
            raise GridValidationError("grid plan parameter schema mismatch")
        scenario_key = _load_key(value["scenario_key"], ScenarioKey, "scenario")
        supplied_scenario = (
            ScenarioKey.from_closure(scenario) if not isinstance(scenario, ScenarioKey) else scenario
        ).validated()
        if scenario_key.validated() != supplied_scenario:
            raise GridValidationError("grid plan scenario mismatch")
        session_key = _load_key(value["baseline_session_key"], SessionKey, "session")
        session_key.validated(compiler.schema)
        observation_key = _load_key(value["baseline_observation_key"], ObservationKey, "observation")
        observation_key.validated(compiler.schema)

        raw_axes = value["axes"]
        if not isinstance(raw_axes, list) or not raw_axes:
            raise GridValidationError("grid plan axes must be a non-empty array")
        axes: list[tuple[_AxisOption, ...]] = []
        coordinate_names: set[str] = set()
        controlled_names: set[str] = set()
        for raw_options in raw_axes:
            if not isinstance(raw_options, list) or not raw_options:
                raise GridValidationError("grid plan axis options must be non-empty arrays")
            options: list[_AxisOption] = []
            axis_coordinates: set[str] | None = None
            axis_proposals: set[str] | None = None
            display_signatures: set[str] = set()
            for raw in raw_options:
                if not isinstance(raw, Mapping) or set(raw) != {"display", "proposal"}:
                    raise GridValidationError("grid plan axis option is invalid")
                if not isinstance(raw["display"], Mapping) or not isinstance(raw["proposal"], Mapping):
                    raise GridValidationError("grid plan display and proposal must be objects")
                proposal = _typed_proposal(compiler, raw["proposal"])
                option_coordinates = set(raw["display"])
                option_proposals = set(proposal)
                if not option_coordinates or not option_proposals:
                    raise GridValidationError("grid plan axis options cannot be empty")
                if axis_coordinates is None:
                    axis_coordinates, axis_proposals = option_coordinates, option_proposals
                    if coordinate_names.intersection(axis_coordinates):
                        raise GridValidationError("grid plan axes reuse coordinate names")
                    if controlled_names.intersection(axis_proposals):
                        raise GridValidationError("grid plan axes reuse proposal descriptors")
                    coordinate_names.update(axis_coordinates)
                    controlled_names.update(axis_proposals)
                elif option_coordinates != axis_coordinates or option_proposals != axis_proposals:
                    raise GridValidationError("grid plan axis option ownership is inconsistent")
                signature = coordinate_signature(raw["display"])
                if signature in display_signatures:
                    raise GridValidationError("grid plan axis has duplicate display coordinates")
                display_signatures.add(signature)
                options.append((_freeze(raw["display"]), _freeze(proposal)))
            axes.append(tuple(options))
        shape = tuple(int(item) for item in value["coordinate_shape"])
        if shape != tuple(len(options) for options in axes):
            raise GridValidationError("grid plan coordinate_shape does not match its axes")
        count = 1
        for size in shape:
            count *= size
        if value["pool_count"] != count:
            raise GridValidationError("grid plan pool_count does not match coordinate_shape")
        static = _typed_proposal(compiler, value["static_proposal"])
        if controlled_names.intersection(static):
            raise GridValidationError("grid plan static proposal overlaps an axis")
        plan = cls(
            grid_id=str(value["grid_id"]),
            axis_options=tuple(axes),
            static_proposal=tuple((name, _freeze(item)) for name, item in sorted(static.items())),
            artifact_sha256=artifact_sha256,
            parameter_schema_sha256=compiler.schema.sha256,
            scenario_key=scenario_key,
            baseline_session_key=session_key,
            baseline_observation_key=observation_key,
            coordinate_shape=shape,
            pool_count=count,
            plan_sha256=digest,
            compiler=compiler,
        )
        candidate, observation = _split_proposal(compiler, static)
        baseline = compiler.compile(
            candidate,
            open_session=_baseline_open_session(compiler, session_key),
            scenario=scenario_key,
        )
        if baseline.session_key != session_key:
            raise GridValidationError("grid plan static proposal does not match baseline session")
        if compiler.compile_observation(observation)[0] != observation_key:
            raise GridValidationError("grid plan static proposal does not match baseline observation")
        if plan.to_dict() != dict(value):
            raise GridValidationError("grid plan is not canonically encoded")
        return plan


def _typed_proposal(compiler: CandidateCompiler, value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(name, str) for name in value):
        raise GridValidationError("grid plan proposal must be an object")
    result: dict[str, Any] = {}
    for name, item in value.items():
        descriptor = compiler.schema.descriptor(name)
        if descriptor.value_type == "integer":
            item = int(item)
        elif descriptor.value_type == "real" and isinstance(item, str):
            item = Decimal(str(item))
        elif descriptor.value_type == "real_pair":
            item = tuple(
                Decimal(str(part)) if isinstance(part, str) else part for part in item
            )
        result[name] = item
    return result


def _named_axis_options(
    axis: AxisSpec,
    compiler: CandidateCompiler,
    snapped: dict[str, int],
    collapsed: dict[str, int],
) -> tuple[_AxisOption, ...]:
    from ..evaluation.plans import CandidatePlanError

    values = axis.rows if axis.is_coupled else tuple((value,) for value in axis.values)
    names = axis.names if axis.is_coupled else (axis.name,)
    targets = axis.targets or tuple(AxisTarget(path=tuple(name.split("."))) for name in names)
    if axis.is_coupled and len(targets) != len(names):
        raise GridValidationError(f"axis {axis.name!r} must declare one target per coordinate")
    if not targets:
        raise GridValidationError(f"axis {axis.name!r} has no canonical targets")
    try:
        descriptors = tuple(compiler.schema.descriptor(target.proposal_name) for target in targets)
    except CandidatePlanError as exc:
        raise GridValidationError(f"axis {axis.name!r} has an unknown canonical target") from exc
    if len({item.name for item in descriptors}) != len(descriptors):
        raise GridValidationError(f"axis {axis.name!r} declares a canonical target twice")

    options: list[_AxisOption] = []
    requested_keys: set[str] = set()
    executed_keys: set[str] = set()
    for row in values:
        requested = coordinate_signature(dict(zip(names, row, strict=True)))
        if requested in requested_keys:
            raise GridValidationError(f"axis {axis.name!r} has a duplicate display coordinate")
        requested_keys.add(requested)
        display: dict[str, Any] = {}
        proposal: dict[str, Any] = {}
        bindings = (
            tuple(zip(names, targets, descriptors, row, strict=True))
            if axis.is_coupled
            else tuple((axis.name, target, descriptor, row[0]) for target, descriptor in zip(targets, descriptors, strict=True))
        )
        inverse_display: list[Any] = []
        for coordinate_name, target, descriptor, value in bindings:
            executed = target.transform_proposal_value(value)
            displayed = value
            if descriptor.value_type in {"integer", "real"}:
                try:
                    quantum = Decimal(str(descriptor.quantum))
                except InvalidOperation:
                    quantum = {
                        "binary64_fraction_or_1e10": Decimal("1e-10"),
                        "binary64_from_wad_1e18": Decimal("1e-18"),
                    }.get(descriptor.wire_representation, Decimal(0))
                try:
                    minimum = Decimal(str(descriptor.minimum))
                except InvalidOperation:
                    minimum = Decimal(0)
                try:
                    semantic = (
                        value * target.scale / target.display_scale
                        if target.kind in {"decimal", "integer", "bps"}
                        else executed
                    )
                    number = Decimal(str(semantic))
                except (ArithmeticError, TypeError, ValueError) as exc:
                    raise GridValidationError(f"axis {axis.name!r} cannot invert {descriptor.name!r}") from exc
                if quantum > 0:
                    with localcontext() as context:
                        context.prec = 80
                        tick = ((number - minimum) / quantum).to_integral_value(rounding=ROUND_HALF_EVEN)
                        executed = minimum + tick * quantum
                if Decimal(str(executed)) != number:
                    snapped[descriptor.name] = snapped.get(descriptor.name, 0) + 1
                displayed = Decimal(str(executed)) * target.display_scale / target.scale
            inverse_display.append(displayed)
            if axis.is_coupled:
                display[coordinate_name] = _exact_value(displayed)
            proposal[descriptor.name] = int(executed) if descriptor.value_type == "integer" else executed
        if not axis.is_coupled:
            canonical_display = inverse_display[0]
            if any(item != canonical_display for item in inverse_display[1:]):
                raise GridValidationError(f"axis {axis.name!r} targets do not share one snapped coordinate")
            display[axis.name] = _exact_value(canonical_display)
        signature = coordinate_signature(display)
        if signature in executed_keys:
            collapsed[axis.name] = collapsed.get(axis.name, 0) + 1
            continue
        executed_keys.add(signature)
        options.append((_freeze(display), _freeze(proposal)))
    return tuple(options)


def compile_grid_plan(
    grid_spec: GridSpec,
    *,
    compiler: CandidateCompiler,
    artifact_sha256: str,
    open_session: Mapping[str, object],
    scenario: ScenarioClosure | ScenarioKey,
    snap_receipt: MutableMapping[str, Any] | None = None,
) -> CartesianGridPlan:
    """Resolve only axes and immutable identities; never enumerate grid points."""
    from ..evaluation.plans import CandidatePlanError, ScenarioKey

    if not isinstance(grid_spec.static_overrides, Mapping) or any(
        not isinstance(name, str) for name in grid_spec.static_overrides
    ):
        raise GridValidationError("static_overrides must use canonical string names")
    base = dict(grid_spec.static_overrides)
    try:
        for name in base:
            compiler.schema.descriptor(name)
    except CandidatePlanError as exc:
        raise GridValidationError(f"static override {name!r} is not canonical") from exc

    coordinate_names: set[str] = set()
    controlled_names: set[str] = set()
    axes: list[tuple[_AxisOption, ...]] = []
    snapped: dict[str, int] = {}
    collapsed: dict[str, int] = {}
    collapsed_axes: list[str] = []
    for axis in grid_spec.axes:
        names = axis.names if axis.is_coupled else (axis.name,)
        duplicates = coordinate_names.intersection(names)
        if duplicates:
            raise GridValidationError("duplicate coordinate names: " + ", ".join(sorted(duplicates)))
        coordinate_names.update(names)
        options = _named_axis_options(axis, compiler, snapped, collapsed)
        if not options:
            raise GridValidationError(f"axis {axis.name!r} has no finite values")
        if axis.size > 1 and len(options) < 2:
            collapsed_axes.append(axis.name)
        option_names = {name for name, _ in options[0][1]}
        duplicates = controlled_names.intersection(option_names)
        if duplicates:
            raise GridValidationError("proposal names controlled by multiple axes: " + ", ".join(sorted(duplicates)))
        controlled_names.update(option_names)
        axes.append(options)
    if controlled_names.intersection(base):
        raise GridValidationError("static proposal names cannot also be controlled by axes")
    if snapped:
        details = ", ".join(f"{name}={count}" for name, count in sorted(snapped.items()))
        warnings.warn(
            f"grid {grid_spec.id!r} snapped axis values (adjusted={sum(snapped.values())}, "
            f"collapsed={sum(collapsed.values())}; {details})",
            UserWarning,
            stacklevel=2,
        )
    if snap_receipt is not None:
        snap_receipt.clear()
        snap_receipt.update(
            adjusted_count=sum(snapped.values()),
            collapsed_count=sum(collapsed.values()),
            adjusted_by_descriptor=dict(sorted(snapped.items())),
        )
    if collapsed_axes:
        raise GridValidationError("axes collapsed after schema snapping: " + ", ".join(collapsed_axes))

    candidate, observation = _split_proposal(compiler, base)
    baseline = compiler.compile(candidate, open_session=open_session, scenario=scenario)
    observation_key = compiler.compile_observation(observation)[0]
    scenario_key = (
        ScenarioKey.from_closure(scenario) if not isinstance(scenario, ScenarioKey) else scenario
    ).validated()
    shape = tuple(len(options) for options in axes)
    count = 1
    for size in shape:
        count *= size
    provisional = CartesianGridPlan(
        grid_id=grid_spec.id,
        axis_options=tuple(axes),
        static_proposal=tuple((name, _freeze(value)) for name, value in sorted(base.items())),
        artifact_sha256=artifact_sha256,
        parameter_schema_sha256=compiler.schema.sha256,
        scenario_key=scenario_key,
        baseline_session_key=baseline.session_key,
        baseline_observation_key=observation_key,
        coordinate_shape=shape,
        pool_count=count,
        plan_sha256="",
        compiler=compiler,
    )
    digest = hashlib.sha256(canonical_json_bytes(provisional.authority_dict())).hexdigest()
    return CartesianGridPlan(
        grid_id=provisional.grid_id,
        axis_options=provisional.axis_options,
        static_proposal=provisional.static_proposal,
        artifact_sha256=provisional.artifact_sha256,
        parameter_schema_sha256=provisional.parameter_schema_sha256,
        scenario_key=provisional.scenario_key,
        baseline_session_key=provisional.baseline_session_key,
        baseline_observation_key=provisional.baseline_observation_key,
        coordinate_shape=provisional.coordinate_shape,
        pool_count=provisional.pool_count,
        plan_sha256=digest,
        compiler=compiler,
    )


__all__ = [
    "GRID_PLAN_SCHEMA_VERSION",
    "CartesianGridPlan",
    "GridPoint",
    "GridValidationError",
    "compile_grid_plan",
    "coordinate_signature",
]
