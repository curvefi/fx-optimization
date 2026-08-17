"""Specifications for pairs, scenarios, policies, grids, optimizations, shiftclick, and runs."""

from .common import (
    PathContainmentError,
    SpecError,
    assert_contained_path,
    canonical_decimal,
    canonical_dict,
    canonical_json_bytes,
    canonical_primitive,
    format_exact_decimal,
    repository_relative,
    repository_root,
    serializable,
)
from .grid import (
    AxisSpec,
    AxisTarget,
    GridSpec,
    load_grid_spec,
)
from .optimization import (
    OptimizationSpec,
    load_optimization_spec,
)
from .pair import (
    PairSpec,
    load_pair_spec,
)
from .parameters import (
    ParameterDim,
    build_parameter_registry,
)
from .policy import (
    PolicyParameter,
    PolicySpec,
    load_policy_spec,
)
from .scenario import (
    MarketFileRef,
    ScenarioSpec,
    load_scenario_spec,
)
from .shiftclick import (
    ShiftclickSpec,
    load_shiftclick_spec,
)

__all__ = [
    "SpecError",
    "PathContainmentError",
    "canonical_decimal",
    "format_exact_decimal",
    "canonical_primitive",
    "canonical_dict",
    "canonical_json_bytes",
    "repository_root",
    "repository_relative",
    "assert_contained_path",
    "serializable",
    "PairSpec",
    "load_pair_spec",
    "ParameterDim",
    "build_parameter_registry",
    "MarketFileRef",
    "ScenarioSpec",
    "load_scenario_spec",
    "PolicyParameter",
    "PolicySpec",
    "load_policy_spec",
    "AxisTarget",
    "AxisSpec",
    "GridSpec",
    "load_grid_spec",
    "OptimizationSpec",
    "load_optimization_spec",
    "ShiftclickSpec",
    "load_shiftclick_spec",
]
