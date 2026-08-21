"""Frozen optimization specification contract and loader."""

from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from .common import (
    SpecError,
    canonical_primitive,
    repository_relative,
    serializable,
)
from .parameters import build_parameter_registry, validate_parameter_space_names
from .policy import load_policy_spec
from .registry import SpecRegistry
from .scenario import load_scenario_spec


@dataclass(frozen=True)
class OptimizationSpec:
    """Immutable optimization run specification."""

    id: str
    pair_id: str
    policy_id: str
    algorithm: str = "tmrbcd"
    scenarios: tuple[str, ...] = ()
    parameter_space: dict[str, Any] = field(default_factory=dict)
    optimizer_config: dict[str, Any] = field(default_factory=dict)
    scoring_config: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    source_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise SpecError("optimization id must be non-empty")
        if not self.pair_id:
            raise SpecError("optimization pair_id must be non-empty")
        if not self.policy_id:
            raise SpecError("optimization policy_id must be non-empty")
        supported_algorithms = {"tmrbcd", "nevergrad_two_points_de"}
        if self.algorithm.lower() not in supported_algorithms:
            raise SpecError(
                "optimization algorithm must be one of: "
                + ", ".join(sorted(supported_algorithms))
            )
        unknown_optimizer = sorted(set(self.optimizer_config) - {"budget", "batch_size", "seed"})
        if unknown_optimizer:
            raise SpecError(
                "unsupported optimizer_config fields: " + ", ".join(unknown_optimizer)
            )
        unknown_scoring = sorted(set(self.scoring_config) - {"score_key", "require_yb"})
        if unknown_scoring:
            raise SpecError(
                "unsupported scoring_config fields: " + ", ".join(unknown_scoring)
            )
        score_key = self.scoring_config.get(
            "score_key", "score_fx_lp_e15_slippage_v1"
        )
        if score_key != "score_fx_lp_e15_slippage_v1":
            raise SpecError(
                "optimization score_key must be 'score_fx_lp_e15_slippage_v1'"
            )
        require_yb = self.scoring_config.get("require_yb", False)
        if not isinstance(require_yb, bool):
            raise SpecError("optimization require_yb must be a boolean")
        if not self.scenarios or any(not scenario for scenario in self.scenarios):
            raise SpecError("optimization scenarios must be a non-empty string array")

    def validate_parameter_space(self, registry: Mapping[str, Any]) -> None:
        """Resolve every parameter_space name through the parameter registry.

        Both policy and pool registry dimensions are legal; any unknown name
        is a SpecError.  Callers build the registry for the run's policy and
        pair template before invoking this (see
        :func:`curve_fx_sim.specs.parameters.build_parameter_registry`).
        """
        validate_parameter_space_names(self.parameter_space, registry)

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "id": self.id,
            "pair_id": self.pair_id,
            "policy_id": self.policy_id,
            "algorithm": self.algorithm,
            "scenarios": list(self.scenarios),
            "parameter_space": serializable(self.parameter_space),
            "optimizer_config": serializable(self.optimizer_config),
            "scoring_config": serializable(self.scoring_config),
            "tags": list(self.tags),
            "source_path": self.source_path.as_posix() if self.source_path else None,
        }


def load_optimization_spec(
    path_or_id: str | os.PathLike[str],
    *,
    repository: Path,
    parameter_space_authority: Literal["legacy_registry", "selected_schema"] = "legacy_registry",
) -> OptimizationSpec:
    """Load and validate an optimization specification from TOML."""
    if parameter_space_authority not in {"legacy_registry", "selected_schema"}:
        raise ValueError(f"unsupported parameter_space_authority: {parameter_space_authority!r}")
    registry = SpecRegistry.from_root(repository)
    root = registry.context.project_root
    candidate = registry.resolve("optimization", path_or_id)

    with candidate.open("rb") as stream:
        raw_data = tomllib.load(stream)

    opt_data = raw_data.get("optimization", raw_data)
    if not isinstance(opt_data, Mapping):
        raise SpecError("optimization specification must be an object")
    known = {
        "id",
        "pair_id",
        "policy_id",
        "algorithm",
        "scenarios",
        "parameter_space",
        "optimizer_config",
        "scoring_config",
        "tags",
    }
    unknown = sorted(set(opt_data) - known)
    if unknown:
        raise SpecError("unsupported optimization fields: " + ", ".join(unknown))

    opt_id = opt_data.get("id") or candidate.stem
    pair_id = opt_data.get("pair_id", "")
    policy_id = opt_data.get("policy_id", "")
    algorithm = opt_data.get("algorithm", "tmrbcd")

    scenarios_raw = opt_data.get("scenarios", [])
    if isinstance(scenarios_raw, (str, bytes)) or not isinstance(scenarios_raw, Sequence):
        raise SpecError("optimization scenarios must be a non-empty string array")
    scenarios = tuple(str(s) for s in scenarios_raw)

    parameter_space = dict(opt_data.get("parameter_space", {}))
    optimizer_config = dict(opt_data.get("optimizer_config", {}))
    scoring_config = dict(opt_data.get("scoring_config", {}))
    tags = tuple(opt_data.get("tags", []))
    source_path = repository_relative(candidate, root)

    if parameter_space_authority == "legacy_registry":
        # The default loader keeps the checked-in PolicySpec registry as its
        # authority. Artifact-selected callers defer only this validation and
        # must resolve canonical names against the selected evaluator schema.
        template_json = None
        if scenarios:
            try:
                primary_scenario = load_scenario_spec(scenarios[0], repository=root)
            except SpecError as exc:
                raise SpecError(
                    f"optimization scenario {scenarios[0]!r} is invalid: {exc}"
                ) from exc
            if primary_scenario.template_path is not None:
                template_file = root / primary_scenario.template_path
                if template_file.is_file():
                    with template_file.open("r", encoding="utf-8") as stream:
                        template_json = json.load(stream)
        policy_spec = load_policy_spec(policy_id, repository=root)
        registry = build_parameter_registry(policy_spec, template_json, parameter_space)
        validate_parameter_space_names(parameter_space, registry)

    return OptimizationSpec(
        id=opt_id,
        pair_id=pair_id,
        policy_id=policy_id,
        algorithm=algorithm,
        scenarios=scenarios,
        parameter_space=parameter_space,
        optimizer_config=optimizer_config,
        scoring_config=scoring_config,
        tags=tags,
        source_path=source_path,
    )


__all__ = ["OptimizationSpec", "load_optimization_spec"]
