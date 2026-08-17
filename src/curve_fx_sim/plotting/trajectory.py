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


def _records_from_payload(payload: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(payload, Mapping):
        for key in ("records", "trace", "rows", "observations"):
            if key in payload:
                payload = payload[key]
                break
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise TrajectoryError("trace must contain an array of observation objects")
    records: list[Mapping[str, Any]] = []
    for index, raw in enumerate(payload):
        if not isinstance(raw, Mapping):
            raise TrajectoryError(f"trace record {index} is not an object")
        records.append(dict(raw))
    return tuple(records)


def load_trajectory(path: Path | str) -> Trajectory:
    """Load harness JSON trace or NPZ column trace without directory scanning."""
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"trace artifact not found: {source}")
    if source.suffix == ".npz":
        try:
            archive = np.load(source, allow_pickle=False)
            names = tuple(archive.files)
            if not names:
                raise TrajectoryError("NPZ trace has no arrays")
            count = len(archive[names[0]])
            records = []
            for index in range(count):
                row: dict[str, Any] = {}
                for name in names:
                    value = archive[name][index]
                    if isinstance(value, np.generic):
                        value = value.item()
                    if isinstance(value, (str, int, float, bool)):
                        row[name] = value
                records.append(row)
            archive.close()
            return Trajectory(tuple(records), source)
        except TrajectoryError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise TrajectoryError(f"cannot read NPZ trace {source}: {exc}") from exc
    try:
        with source.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise TrajectoryError(f"cannot read JSON trace {source}: {exc}") from exc
    return Trajectory(_records_from_payload(payload), source)


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
    trace: Trajectory | Path | str,
    output: Path | str,
    *,
    fields: Sequence[str] | None = None,
    x_field: str = "t",
    theme: PlotTheme = DEFAULT_THEME,
) -> tuple[Path, Path]:
    """Render deterministic state trajectories and an immutable state sidecar."""
    resolved = trace if isinstance(trace, Trajectory) else load_trajectory(trace)
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


__all__ = ["Trajectory", "TrajectoryError", "load_trajectory", "render_trajectory"]
