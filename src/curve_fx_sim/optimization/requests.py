"""Registry-backed proposal-to-request split for optimization runs.

The dense vector produced by the optimizer lattice holds policy parameters in
PolicySpec ABI order followed by active pool-economics dims in registry order.
This module splits one such vector into the two protocol request halves
consumed by the evaluator: dense ``policy_params`` (the exact policy ABI
vector) and nested ``pool_overrides`` (the harness pool-override schema,
mirroring the seam in :mod:`curve_fx_sim.grids.model`).
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Sequence

from .profiles import Profile


def _set_nested(target: dict[str, Any], path: Sequence[str], value: Any) -> None:
    """Write a nested pool-override value, mirroring grids.model's seam."""
    if not path:
        raise ValueError("pool override target path must be non-empty")
    current = target
    for part in path[:-1]:
        existing = current.get(part)
        if existing is None:
            existing = {}
            current[part] = existing
        if not isinstance(existing, dict):
            raise ValueError(
                f"pool override path {'.'.join(path)!r} crosses scalar namespace {part!r}"
            )
        current = existing
    current[path[-1]] = value


def split_request(
    profile: Profile,
    params: Sequence[float],
) -> tuple[list[float], dict[str, Any]]:
    """Split one dense registry vector into ``policy_params`` and ``pool_overrides``.

    ``policy_params`` is the exact policy ABI vector (policy ABI order, fixed
    parameters included) and ``pool_overrides`` nests each active pool dim
    value converted to the harness's raw scaled-integer units.  For a
    policy-only profile the result is ``(dense_vector, {})``.
    """
    expected = profile.dense_dim()
    if len(params) != expected:
        if profile.pool_dims:
            raise ValueError(
                f"profile {profile.name!r} requires {expected} parameters "
                f"(policy {profile.n_params()} + pool {len(profile.pool_dims)}), got {len(params)}"
            )
        raise ValueError(
            f"policy {profile.name!r} requires {profile.n_params()} parameters, got {len(params)}"
        )
    policy_params = [float(value) for value in params[: profile.n_params()]]
    pool_overrides: dict[str, Any] = {}
    for dim in profile.pool_dims:
        scaled = Decimal(str(params[dim.index])) * dim.override_scale
        if scaled == scaled.to_integral_value():
            _set_nested(pool_overrides, dim.target_path, int(scaled))
        else:
            _set_nested(pool_overrides, dim.target_path, float(scaled))
    return policy_params, pool_overrides


__all__ = ["split_request"]
