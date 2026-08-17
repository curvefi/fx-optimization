"""Frozen policy specification contract and loader."""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping

from .common import (
    SpecError,
    assert_contained_path,
    repository_relative,
    repository_root,
)


@dataclass(frozen=True)
class PolicyParameter:
    """Definition of a single tunable parameter for a dynamic fee policy."""

    name: str
    type: str = "float"  # float or int; the evaluator receives one dense numeric vector
    default: Any = None
    min_val: float | None = None
    max_val: float | None = None
    step: float | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise SpecError("policy parameter name must be non-empty")
        if self.type not in {"float", "int"}:
            raise SpecError(
                f"parameter {self.name} has unsupported type {self.type!r}; "
                "compiled policy parameters must be float or int"
            )
        for label, raw in (
            ("min", self.min_val),
            ("max", self.max_val),
            ("step", self.step),
        ):
            if raw is not None:
                self._decimal(raw, label=label)
        if self.min_val is None or self.max_val is None or self.step is None:
            raise SpecError(
                f"parameter {self.name} requires explicit min, max, and step"
            )
        if self._decimal(self.min_val, label="min") > self._decimal(self.max_val, label="max"):
            raise SpecError(f"parameter {self.name} min exceeds max")
        if self._decimal(self.step, label="step") <= 0:
            raise SpecError(f"parameter {self.name} step must be positive")
        if self.default is None:
            raise SpecError(f"parameter {self.name} requires an explicit default")
        self.validate_value(self.default)

    def _decimal(self, value: Any, *, label: str = "value") -> Decimal:
        if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
            raise SpecError(
                f"parameter {self.name} {label} must be numeric, got {value!r}"
            )
        try:
            decimal_value = Decimal(str(value))
        except (InvalidOperation, ValueError) as exc:
            raise SpecError(
                f"parameter {self.name} {label} must be numeric, got {value!r}"
            ) from exc
        if not decimal_value.is_finite():
            raise SpecError(
                f"parameter {self.name} {label} must be finite, got {value!r}"
            )
        return decimal_value

    def validate_value(self, value: Any) -> Any:
        """Validate one value without truncation or off-lattice coercion."""
        if value is None:
            return self.default
        decimal_value = self._decimal(value)
        if self.type == "int":
            if decimal_value != decimal_value.to_integral_value():
                raise SpecError(f"parameter {self.name} must be an integer, got {value!r}")

        lower = self._decimal(self.min_val, label="min")
        upper = self._decimal(self.max_val, label="max")
        step = self._decimal(self.step, label="step")
        if decimal_value < lower:
            raise SpecError(f"parameter {self.name} value {value!r} < min {self.min_val}")
        if decimal_value > upper:
            raise SpecError(f"parameter {self.name} value {value!r} > max {self.max_val}")
        if (decimal_value - lower) % step != 0:
            raise SpecError(
                f"parameter {self.name} value {value!r} is off lattice "
                f"{self.min_val} + n*{self.step}"
            )
        return int(decimal_value) if self.type == "int" else float(decimal_value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "default": self.default,
            "min_val": self.min_val,
            "max_val": self.max_val,
            "step": self.step,
            "description": self.description,
        }


@dataclass(frozen=True)
class PolicySpec:
    """Immutable policy specification."""

    id: str
    header_file: Path | str
    source_sha256: str
    policy_abi: str = "twocrypto_policy_v1"
    policy_kind: str = "compiled"
    parameters: tuple[PolicyParameter, ...] = ()
    tags: tuple[str, ...] = ()
    source_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.id or not self.id.strip():
            raise SpecError("policy id must be non-empty")
        if self.policy_kind != "compiled":
            raise SpecError(
                f"policy {self.id!r} must use the sole supported kind 'compiled'"
            )
        if self.policy_abi != "twocrypto_policy_v1":
            raise SpecError(
                f"policy {self.id!r} must use ABI 'twocrypto_policy_v1'"
            )
        if not isinstance(self.source_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.source_sha256
        ):
            raise SpecError(
                f"policy {self.id!r} requires a lowercase 64-character source_sha256"
            )
        if not str(self.header_file).strip():
            raise SpecError(f"policy {self.id!r} requires a header_file")
        if not self.parameters:
            raise SpecError(f"policy {self.id!r} declares no parameters")
        names = [parameter.name for parameter in self.parameters]
        if len(names) != len(set(names)):
            raise SpecError(f"policy {self.id!r} has duplicate parameter names")

    def validate_params(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Validate and complete parameters against specification."""
        param_map = {p.name: p for p in self.parameters}
        unknown = sorted(set(params) - set(param_map))
        if unknown:
            raise SpecError(
                f"policy {self.id!r} received undeclared parameters: " + ", ".join(unknown)
            )
        return {
            parameter.name: parameter.validate_value(
                params.get(parameter.name, parameter.default)
            )
            for parameter in self.parameters
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "id": self.id,
            "header_file": self.header_file.as_posix() if isinstance(self.header_file, Path) else str(self.header_file),
            "source_sha256": self.source_sha256,
            "policy_abi": self.policy_abi,
            "policy_kind": self.policy_kind,
            "parameters": [p.to_dict() for p in self.parameters],
            "tags": list(self.tags),
            "source_path": self.source_path.as_posix() if self.source_path else None,
        }


def load_policy_spec(
    path_or_id: str | os.PathLike[str],
    *,
    repository: Path | None = None,
) -> PolicySpec:
    """Load and validate a policy TOML specification."""
    root = repository.resolve() if repository is not None else repository_root()
    candidate = Path(path_or_id)

    if not candidate.is_file():
        search_paths = [
            root / "configs" / "policies" / f"{path_or_id}.toml",
            root / "configs" / f"{path_or_id}.toml",
            root / "policies" / f"{path_or_id}.toml",
        ]
        found = None
        for p in search_paths:
            if p.is_file():
                found = p
                break
        if found is None:
            raise FileNotFoundError(f"Policy specification not found: {path_or_id}")
        candidate = found

    assert_contained_path(candidate, root, allow_symlinks=True)

    with candidate.open("rb") as stream:
        raw_data = tomllib.load(stream)

    policy_data = raw_data.get("policy", raw_data)
    config_dir = candidate.resolve().parent

    policy_id = policy_data.get("id") or candidate.stem
    header_raw = policy_data.get("header_file") or policy_data.get("header", f"{policy_id}.hpp")
    header_path = Path(header_raw)
    if not header_path.is_absolute() and (config_dir / header_path).is_file():
        header_file = repository_relative(config_dir / header_path, root)
    else:
        header_file = Path(header_raw)

    source_sha256 = policy_data.get("source_sha256")
    policy_abi = policy_data.get("policy_abi", "twocrypto_policy_v1")
    policy_kind = policy_data.get("policy_kind", "compiled")

    params_raw = policy_data.get("parameters", policy_data.get("params", []))
    params: list[PolicyParameter] = []
    if isinstance(params_raw, list):
        for item in params_raw:
            if isinstance(item, Mapping):
                params.append(
                    PolicyParameter(
                        name=item["name"],
                        type=item.get("type", "float"),
                        default=item.get("default"),
                        min_val=float(item["min"]) if "min" in item else None,
                        max_val=float(item["max"]) if "max" in item else None,
                        step=float(item["step"]) if "step" in item else None,
                        description=item.get("description", ""),
                    )
                )
    elif isinstance(params_raw, Mapping):
        for name, item in params_raw.items():
            if isinstance(item, Mapping):
                params.append(
                    PolicyParameter(
                        name=name,
                        type=item.get("type", "float"),
                        default=item.get("default"),
                        min_val=float(item["min"]) if "min" in item else None,
                        max_val=float(item["max"]) if "max" in item else None,
                        step=float(item["step"]) if "step" in item else None,
                        description=item.get("description", ""),
                    )
                )

    tags = tuple(policy_data.get("tags", []))
    source_path = repository_relative(candidate, root)

    return PolicySpec(
        id=policy_id,
        header_file=header_file,
        source_sha256=source_sha256,
        policy_abi=policy_abi,
        policy_kind=policy_kind,
        parameters=tuple(params),
        tags=tags,
        source_path=source_path,
    )


__all__ = ["PolicyParameter", "PolicySpec", "load_policy_spec"]
