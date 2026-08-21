"""Deterministic trajectory trace loading and headless rendering."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from ..artifacts.io import atomic_write_json
from ..shiftclick.archive import ACTION_COLUMNS, TRACE_COLUMNS
from .theme import DEFAULT_THEME, PlotTheme, apply_theme


class TrajectoryError(ValueError):
    """Raised when a trace cannot be interpreted as a finite trajectory."""


@dataclass(frozen=True)
class Trajectory:
    """Column-oriented finite trace used by replay diagnostics and plotting."""

    records: tuple[Mapping[str, Any], ...]
    source: Path | None = None

    def __post_init__(self) -> None:
        if not self.records:
            raise TrajectoryError("trajectory trace is empty")

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(sorted({str(key) for row in self.records for key in row}))

    def series(self, field: str) -> tuple[float, ...]:
        if field not in self.fields:
            raise TrajectoryError(f"trace field {field!r} is unavailable")
        values: list[float] = []
        for index, row in enumerate(self.records):
            raw = row.get(field)
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise TrajectoryError(f"trace field {field!r} row {index} is not numeric") from exc
            if not math.isfinite(value):
                raise TrajectoryError(f"trace field {field!r} row {index} is not finite")
            values.append(value)
        return tuple(values)

    def x_values(self, field: str = "t") -> tuple[float, ...]:
        if field in self.fields:
            return self.series(field)
        return tuple(float(index) for index in range(len(self.records)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "fxsim_trajectory_v1",
            "record_count": len(self.records),
            "fields": list(self.fields),
            "source": self.source.as_posix() if self.source else None,
        }


def load_replay_records(
    path: Path | str, *, companion_path: Path | str, kind: str = "trace",
    scenario_index: int = 0,
) -> tuple[Mapping[str, Any], ...]:
    """Load one attested matrix from the candidate-wide replay archive."""
    supplied = Path(path).resolve()
    companion = Path(companion_path).resolve()
    try:
        payload = json.loads(companion.read_text(encoding="utf-8"))
        if (set(payload) != {"schema_version", "source_run_id", "candidate_id", "ordinal", "columns", "scenarios", "npz"}
                or payload.get("schema_version") != "curve_fx_replay_trace_v1"):
            raise TrajectoryError("unsupported replay trace schema")
        expected_columns = {"trace": list(TRACE_COLUMNS), **{name: list(value) for name, value in ACTION_COLUMNS.items()}}
        if payload["columns"] != expected_columns or kind not in expected_columns:
            raise TrajectoryError("replay columns differ from the fixed schema")
        npz_ref = payload["npz"]
        if set(npz_ref) != {"path", "sha256", "bytes"} or npz_ref["path"] != "replay_trace.npz":
            raise TrajectoryError("replay NPZ descriptor is invalid")
        archive_path = (companion.parent / npz_ref["path"]).resolve()
        if archive_path != supplied:
            raise TrajectoryError("replay NPZ and companion are not paired")
        from ..artifacts.io import sha256_path
        if archive_path.stat().st_size != npz_ref["bytes"] or sha256_path(archive_path) != npz_ref["sha256"]:
            raise TrajectoryError("replay NPZ attestation mismatch")
        scenarios = payload["scenarios"]
        expected_keys = {f"{name}_{index:03d}" for index in range(len(scenarios)) for name in expected_columns}
        with np.load(archive_path, allow_pickle=False) as archive:
            if set(archive.files) != expected_keys:
                raise TrajectoryError("replay NPZ has unexpected or missing matrices")
            for index, scenario in enumerate(scenarios):
                if (set(scenario) != {"index", "id", "evaluation_id", "economic_fingerprint", "row_counts", "source_sidecars"}
                        or scenario["index"] != index or set(scenario["row_counts"]) != set(expected_columns)
                        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                               for value in scenario["row_counts"].values())
                        or not isinstance(scenario["source_sidecars"], list)
                        or any(set(item) != {"kind", "sha256", "bytes"}
                               for item in scenario["source_sidecars"])):
                    raise TrajectoryError("replay scenario metadata is inconsistent")
                for matrix_kind, columns in expected_columns.items():
                    matrix = archive[f"{matrix_kind}_{index:03d}"]
                    if (matrix.dtype != np.dtype("<f8") or matrix.ndim != 2
                            or matrix.shape != (scenario["row_counts"][matrix_kind], len(columns))):
                        raise TrajectoryError(f"replay {matrix_kind} matrix has the wrong shape or dtype")
            matrix = archive[f"{kind}_{scenario_index:03d}"].copy()
        columns = expected_columns[kind]
        records = []
        for values in matrix:
            row = {name: float(value) for name, value in zip(columns, values, strict=True)}
            if kind != "trace":
                row["type"] = kind
            records.append(row)
        return tuple(records)
    except TrajectoryError:
        raise
    except (IndexError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise TrajectoryError(f"cannot read replay archive {companion}: {exc}") from exc


def load_trajectory(
    path: Path | str, *, companion_path: Path | str, scenario_index: int = 0
) -> Trajectory:
    """Load one scenario trace from a candidate-wide replay archive."""
    source = Path(path).resolve()
    return Trajectory(load_replay_records(
        source, companion_path=companion_path, scenario_index=scenario_index
    ), source)


def _default_fields(trace: Trajectory) -> tuple[str, ...]:
    preferred = ("price_scale", "price_oracle", "vp", "xcp", "profit", "donation_apy")
    available = set(trace.fields)
    selected = tuple(name for name in preferred if name in available)
    if selected:
        return selected
    return tuple(
        name for name in trace.fields if name != "t" and all(
            isinstance(row.get(name), (int, float)) and math.isfinite(float(row[name]))
            for row in trace.records
        )
    )[:4]


def _atomic_save(figure: object, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"immutable trajectory artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=path.suffix, dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        figure.savefig(temporary, format=path.suffix.lstrip("."), bbox_inches="tight")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def render_trajectory(
    trace: Trajectory,
    output: Path | str,
    *,
    fields: Sequence[str] | None = None,
    x_field: str = "t",
    theme: PlotTheme = DEFAULT_THEME,
) -> tuple[Path, Path]:
    """Render deterministic state trajectories and an immutable state sidecar."""
    if not isinstance(trace, Trajectory):
        raise TypeError("render_trajectory requires an explicitly loaded Trajectory")
    resolved = trace
    chosen = tuple(fields) if fields is not None else _default_fields(resolved)
    if not chosen:
        raise TrajectoryError("trace has no numeric fields to plot")
    x = resolved.x_values(x_field)
    series = {field: resolved.series(field) for field in chosen}
    image_path = Path(output)
    state_path = image_path.with_suffix(".state.json")
    if image_path.exists() or state_path.exists():
        raise FileExistsError(f"trajectory artifact already exists: {image_path}")
    figure, axis = plt.subplots(figsize=(theme.figure_width, theme.panel_height * max(1.0, len(chosen) / 2.0)))
    try:
        for field in chosen:
            axis.plot(x, series[field], label=field, linewidth=theme.line_width)
        axis.set_xlabel(x_field if x_field in resolved.fields else "record", fontfamily=theme.font_family)
        axis.set_ylabel("value", fontfamily=theme.font_family)
        axis.legend(loc="best", frameon=False)
        apply_theme(figure, np.asarray([axis], dtype=object), theme)
        _atomic_save(figure, image_path)
    finally:
        plt.close(figure)
    atomic_write_json(
        state_path,
        {
            "schema_version": "fxsim_trajectory_render_v1",
            "trace": resolved.to_dict(),
            "x_field": x_field,
            "fields": list(chosen),
            "record_count": len(resolved.records),
        },
    )
    return image_path, state_path


__all__ = ["Trajectory", "TrajectoryError", "load_replay_records", "load_trajectory", "render_trajectory"]
