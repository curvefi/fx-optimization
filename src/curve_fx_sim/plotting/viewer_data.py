"""Thin run-data adapter: fxsim evaluation-table NPZ -> legacy NpzRun view.

The viewer core (plotting/viewer.py) is a near-verbatim copy of the legacy
``plot_heatmap_nd_opt.py`` explorer. Its data layer expects an ``NpzRun``
object (``load_array(name)``, ``n_pools``, ``metadata`` with ``metadata.grid``
axes in raw pool units) plus per-cell pool configs/metrics. This module builds
those views from a ``fxsim`` run directory (``manifest.json`` +
``evaluation_table.npz``), converting grid display coordinates to raw pool
units via the attested axis targets so the legacy axis formatters (A ÷1e4,
fees in bps, ma_time in hours) apply unchanged.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..artifacts.tables import EvaluationTable


def _raw_axis_value(display: float, scale: float, display_scale: float, kind: str) -> float:
    """AxisTarget.transform_value equivalent; float form for the viewer."""
    scaled = (display * scale) / display_scale
    if kind in ("integer", "bps"):
        return float(int(scaled))
    if scaled == math.floor(scaled):
        return float(int(scaled))
    return float(scaled)


def _axis_scale_from_target(target: Mapping[str, Any]) -> tuple[float, float, str]:
    try:
        scale = float(target.get("scale", 1))
    except (TypeError, ValueError):
        scale = 1.0
    try:
        display_scale = float(target.get("display_scale", 1))
    except (TypeError, ValueError):
        display_scale = 1.0
    kind = str(target.get("kind", "decimal"))
    return scale, display_scale, kind


@dataclass(frozen=True)
class ViewerAxis:
    """One grid axis in the legacy metadata.grid shape."""

    name: str
    raw_values: tuple[float, ...]
    display_values: tuple[Any, ...]
    key: str  # x1, x2, ...
    targets: tuple[Mapping[str, Any], ...] = ()
    names: tuple[str, ...] = ()

    def to_metadata_entry(self) -> dict[str, Any]:
        entry: dict[str, Any] = {"name": self.name, "values": list(self.raw_values)}
        if self.names:
            entry = {"names": list(self.names), "values": [list(r) for r in self.raw_values]}
        return entry


class TableRun:
    """NpzRun-compatible facade over one fxsim evaluation-table run."""

    def __init__(self, run_dir: Path) -> None:
        self.root = Path(run_dir).resolve()
        manifest_path = self.root / "manifest.json"
        table_path = self.root / "evaluation_table.npz"
        if not manifest_path.is_file() or not table_path.is_file():
            raise FileNotFoundError(
                f"{self.root} is not a collected fxsim run "
                "(manifest.json + evaluation_table.npz required)"
            )
        with manifest_path.open("r", encoding="utf-8") as stream:
            self.manifest: dict[str, Any] = json.load(stream)
        self.table = EvaluationTable.from_npz(table_path)
        self.n_pools = len(self.table.rows)
        self.metadata: dict[str, Any] = self._build_metadata()
        self._metric_names: list[str] = sorted(
            {name for row in self.table.rows for name in row.metrics}
        )
        self._metric_index: dict[str, int] = {
            name: index for index, name in enumerate(self._metric_names)
        }
        self._metric_columns: dict[str, np.ndarray] = {}
        for name in self._metric_names:
            column = np.full(self.n_pools, np.nan, dtype=float)
            for row_index, row in enumerate(self.table.rows):
                value = row.metrics.get(name)
                if value is not None and not isinstance(value, bool):
                    try:
                        column[row_index] = float(value)
                    except (TypeError, ValueError):
                        pass
            self._metric_columns[name] = column
        self._pool_configs: dict[tuple[int, ...], dict[str, Any]] = {}
        self._metrics_lookup: dict[tuple[int, ...], dict[str, Any]] = {}
        self._row_coordinates: dict[tuple[int, ...], Mapping[str, Any]] = {}
        self._build_cell_views()

    # ---- metadata ---------------------------------------------------------

    def _build_metadata(self) -> dict[str, Any]:
        meta: dict[str, Any] = {}
        grid_section = self.manifest.get("grid", {})
        if not isinstance(grid_section, Mapping):
            grid_section = {}
        axes = grid_section.get("resolved_axes")
        if isinstance(axes, list):
            grid: dict[str, Any] = {}
            for index, axis in enumerate(axes):
                if not isinstance(axis, Mapping):
                    continue
                key = f"x{index + 1}"
                raw, display, targets, names = self._axis_view(axis)
                if raw is None:
                    continue
                if names:
                    grid[key] = {"names": list(names), "values": list(raw)}
                else:
                    leaf = str(axis.get("name", key))
                    if targets:
                        first_path = targets[0].get("path")
                        if isinstance(first_path, list) and first_path:
                            leaf = str(first_path[-1])
                    grid[key] = {
                        "name": leaf,
                        "display_name": str(axis.get("name", key)),
                        "values": list(raw),
                    }
                if display is not None:
                    grid[key]["display_values"] = list(display)
                if targets:
                    grid[key]["targets"] = list(targets)
            meta["grid"] = grid
        meta["n_pools"] = self.n_pools
        meta["total_pools"] = self.n_pools
        meta["run_id"] = str(self.manifest.get("run_id", ""))
        core = self.manifest.get("core")
        if isinstance(core, Mapping):
            meta["policy_id"] = str(core.get("policy_id", "twocrypto_native"))
            meta["binary_sha256"] = str(core.get("sha256", core.get("binary_sha256", "")))
            numeric_mode = core.get("numeric_mode", "longdouble")
            meta["real"] = "longdouble" if numeric_mode == "longdouble" else str(numeric_mode)
        resolved = self.manifest.get("resolved_spec", {})
        if isinstance(resolved, Mapping):
            pair_spec = resolved.get("pair")
            if isinstance(pair_spec, Mapping):
                meta["pair_id"] = str(pair_spec.get("id", ""))
            scenario_spec = resolved.get("scenario")
            if isinstance(scenario_spec, Mapping):
                meta["scenario_id"] = str(scenario_spec.get("id", ""))
                try:
                    meta["n_candles"] = int(scenario_spec.get("n_candles", 0))
                except (TypeError, ValueError):
                    meta["n_candles"] = 0

        # Base pool/costs from the scenario template; harness args for inspect.
        resolved = self.manifest.get("resolved_spec", {})
        scenario = resolved.get("scenario") if isinstance(resolved, Mapping) else None
        if isinstance(scenario, Mapping):
            template_path = scenario.get("template_path")
            if template_path:
                candidate = self.root.parent.parent / str(template_path)
                if candidate.is_file():
                    try:
                        with candidate.open("r", encoding="utf-8") as stream:
                            template = json.load(stream)
                        pools = template.get("pools", [])
                        if isinstance(pools, list) and pools:
                            entry = pools[0]
                            if isinstance(entry, Mapping):
                                pool = entry.get("pool", entry)
                                if isinstance(pool, Mapping):
                                    meta["base_pool"] = dict(pool)
                                costs = entry.get("costs")
                                if isinstance(costs, Mapping):
                                    meta["base_costs"] = dict(costs)
                    except (OSError, ValueError):
                        pass
            market_files = scenario.get("market_files")
            if isinstance(market_files, list) and market_files:
                first = market_files[0]
                if isinstance(first, Mapping):
                    meta["candles_file"] = str(first.get("path", ""))
                    if first.get("kind") == "chainlink":
                        meta["chainlink_feed"] = str(first.get("path", ""))
            harness_args: dict[str, Any] = {}
            for src_key, dst_key in (
                ("start_time", "start_time"),
                ("dustswap_freq_s", "dustswapfreq"),
                ("user_swap_freq_s", "userswapfreq"),
                ("user_swap_size_frac", "userswapsize"),
                ("user_swap_thresh", "userswapthresh"),
                ("disable_slippage_probes", "disable_slippage_probes"),
            ):
                if scenario.get(src_key) is not None:
                    harness_args[dst_key] = scenario[src_key]
            if harness_args:
                meta["harness_args"] = harness_args
        core = self.manifest.get("core")
        if isinstance(core, Mapping):
            numeric_mode = core.get("numeric_mode", "longdouble")
            meta["real"] = "longdouble" if numeric_mode == "longdouble" else str(numeric_mode)
        return meta

    def _axis_view(
        self, axis: Mapping[str, Any]
    ) -> tuple[tuple[float, ...] | None, tuple[Any, ...] | None, tuple[Mapping[str, Any], ...], tuple[str, ...]]:
        raw_targets = axis.get("targets") or axis.get("target")
        targets: tuple[Mapping[str, Any], ...]
        if isinstance(raw_targets, list):
            targets = tuple(t for t in raw_targets if isinstance(t, Mapping))
        elif isinstance(raw_targets, Mapping):
            targets = (raw_targets,)
        else:
            targets = ()

        raw_values_raw = axis.get("values") or axis.get("rows")
        if not isinstance(raw_values_raw, list) or not raw_values_raw:
            return None, None, targets, ()
        # Coupled axis (rows of tuples).
        if isinstance(raw_values_raw[0], list):
            names = tuple(str(n) for n in (axis.get("names") or ()))
            if not names:
                # Derive from multi-target paths (e.g. mid_fee/out_fee).
                for target in targets:
                    path = target.get("path")
                    if isinstance(path, list) and path:
                        names = names + (str(path[-1]),)
            raw_rows: list[tuple[float, ...]] = []
            display_rows: list[tuple[Any, ...]] = []
            for row in raw_values_raw:
                if not isinstance(row, list):
                    return None, None, targets, ()
                if not targets:
                    targets = tuple({"path": tuple(str(n).split("."))} for n in names)
                vals: list[float] = []
                disps: list[Any] = []
                for value, target in zip(row, targets):
                    scale, display_scale, kind = _axis_scale_from_target(target)
                    try:
                        fvalue = float(value)
                    except (TypeError, ValueError):
                        return None, None, targets, ()
                    vals.append(_raw_axis_value(fvalue, scale, display_scale, kind))
                    disps.append(value)
                raw_rows.append(tuple(vals))
                display_rows.append(tuple(disps))
            return tuple(raw_rows), tuple(display_rows), targets, names
        # Scalar axis.
        display_values: list[Any] = []
        raw_values: list[float] = []
        for value in raw_values_raw:
            try:
                fvalue = float(value)
            except (TypeError, ValueError):
                return None, None, targets, ()
            display_values.append(value)
            if targets:
                scale, display_scale, kind = _axis_scale_from_target(targets[0])
                raw_values.append(_raw_axis_value(fvalue, scale, display_scale, kind))
            else:
                raw_values.append(fvalue)
        return tuple(raw_values), tuple(display_values), targets, ()

    # ---- arrays -----------------------------------------------------------

    def metric_names(self) -> set[str]:
        return set(self._metric_names)

    def load_array(self, name: str) -> np.ndarray:
        if name == "success":
            return np.asarray([1 if row.status == "ok" else 0 for row in self.table.rows], dtype=np.int8)
        try:
            return self._metric_columns[name]
        except KeyError as exc:
            raise KeyError(name) from exc

    # ---- per-cell views ---------------------------------------------------

    def _build_cell_views(self) -> None:
        """Build per-cell views from the manifest's authoritative ordinal map.

        ``grid.pools[].coordinate_indices`` (ordinal -> axis indices) plus
        ``grid.pools[].coordinates`` (exact display values) are the single
        source of truth; the table's ordinal column joins to them. This also
        covers coupled axes (row indices) and exact-decimal grids without any
        float matching.
        """
        grid = self.metadata.get("grid", {})
        axis_keys = sorted(
            (k for k in grid if isinstance(k, str) and k.startswith("x") and k[1:].isdigit()),
            key=lambda k: int(k[1:]),
        )
        if not axis_keys:
            return
        n_axes = len(axis_keys)

        pools = self.manifest.get("grid", {}).get("pools")
        by_ordinal: dict[int, tuple[tuple[int, ...], Mapping[str, Any] | None]] = {}
        if isinstance(pools, list):
            for entry in pools:
                if not isinstance(entry, Mapping):
                    continue
                try:
                    ordinal = int(entry.get("ordinal", -1))
                except (TypeError, ValueError):
                    continue
                indices = entry.get("coordinate_indices")
                if ordinal < 0 or not isinstance(indices, list) or len(indices) != n_axes:
                    continue
                try:
                    idx_tuple = tuple(int(index) for index in indices)
                except (TypeError, ValueError):
                    continue
                coords = entry.get("coordinates")
                by_ordinal[ordinal] = (idx_tuple, coords if isinstance(coords, Mapping) else None)

        for row in self.table.rows:
            info = by_ordinal.get(row.ordinal)
            if info is None:
                continue
            idx_tuple, coords = info
            if coords is not None:
                self._row_coordinates[idx_tuple] = dict(coords)
            metrics = dict(row.metrics)
            self._metrics_lookup[idx_tuple] = metrics
            pool = dict(self.metadata.get("base_pool", {}))
            costs = dict(self.metadata.get("base_costs", {}))
            overrides = row.pool_overrides
            if isinstance(overrides, Mapping):
                for key, value in overrides.items():
                    if key == "costs" and isinstance(value, Mapping):
                        costs.update(value)
                    else:
                        pool[key] = value
            if row.params:
                pool["policy_params"] = dict(row.params)
            self._pool_configs[idx_tuple] = {"pool": pool, "costs": costs}


    # ---- legacy-shaped accessors -------------------------------------------

    @property
    def pool_configs(self) -> dict[tuple[int, ...], dict[str, Any]]:
        return self._pool_configs

    @property
    def metrics_lookup(self) -> dict[tuple[int, ...], dict[str, Any]]:
        return self._metrics_lookup

    @property
    def row_coordinates(self) -> dict[tuple[int, ...], Mapping[str, Any]]:
        """Exact per-cell display coordinates (keys = axis display names)."""
        return self._row_coordinates


def load_viewer_data(run_dir: Path) -> dict[str, Any]:
    """Load one collected fxsim run into the legacy explorer data shape."""
    run = TableRun(run_dir)
    return {
        "metadata": run.metadata,
        "_npz_run": run,
        "runs": [],
    }


__all__ = ["TableRun", "load_viewer_data"]
