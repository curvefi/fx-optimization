"""Compact, pickle-free NPZ evaluation tables, rows, and metric projections."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..specs.common import canonical_json_bytes, canonical_primitive

_TABLE_SCHEMA_VERSION = "fxsim_evaluation_table_npz_v1"


def _json_text(value: Any) -> str:
    return canonical_json_bytes(canonical_primitive(value)).decode("utf-8")


def _json_array(values: Sequence[Any]) -> np.ndarray:
    return np.asarray([_json_text(value) for value in values], dtype=np.str_)


def _optional_strings(values: Sequence[str | None]) -> tuple[np.ndarray, np.ndarray]:
    present = np.asarray([value is not None for value in values], dtype=np.bool_)
    strings = np.asarray(["" if value is None else value for value in values], dtype=np.str_)
    return strings, present


def _scalar_text(archive: Mapping[str, Any], name: str) -> str:
    value = np.asarray(archive[name])
    if value.shape != ():
        raise ValueError(f"evaluation-table NPZ field {name!r} must be scalar")
    return str(value.item())


@dataclass(frozen=True)
class MetricProjection:
    """Attested subset projection of raw simulation metrics."""

    fields: tuple[str, ...]
    projection_id: str = "default"
    projection_sha256: str = ""

    def __post_init__(self) -> None:
        if not self.projection_sha256:
            digest = hashlib.sha256(canonical_json_bytes(list(self.fields))).hexdigest()
            object.__setattr__(self, "projection_sha256", digest)

    @classmethod
    def from_fields(cls, fields: Sequence[str], projection_id: str = "custom") -> MetricProjection:
        sorted_fields = tuple(sorted(str(f) for f in fields))
        return cls(fields=sorted_fields, projection_id=projection_id)

    def project_metrics(self, raw_metrics: Mapping[str, Any]) -> dict[str, Any]:
        """Project a raw metrics dictionary down to this projection's fields."""
        return {f: raw_metrics.get(f) for f in self.fields if f in raw_metrics}

    def to_dict(self) -> dict[str, Any]:
        return {
            "projection_id": self.projection_id,
            "fields": list(self.fields),
            "projection_sha256": self.projection_sha256,
        }


@dataclass
class EvaluationRow:
    """A single candidate evaluation result row."""

    candidate_id: str
    ordinal: int = 0
    coordinates: dict[str, Any] | None = None
    params: dict[str, Any] = field(default_factory=dict)
    pool_overrides: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    economic_fingerprint: str | None = None
    trace_path: str | None = None
    actions_path: str | None = None
    tags: tuple[str, ...] = ()

    def canonical_sort_key(self) -> tuple[int, str, int, str]:
        """Deterministic sorting key: status, candidate, ordinal, coordinates."""
        status_pri = 0 if self.status == "ok" else 1
        coord_key = json.dumps(self.coordinates, sort_keys=True, default=str) if self.coordinates else ""
        return (status_pri, self.candidate_id, self.ordinal, coord_key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "ordinal": self.ordinal,
            "coordinates": canonical_primitive(self.coordinates),
            "params": canonical_primitive(self.params),
            "pool_overrides": canonical_primitive(self.pool_overrides),
            "metrics": canonical_primitive(self.metrics),
            "status": self.status,
            "economic_fingerprint": self.economic_fingerprint,
            "trace_path": self.trace_path,
            "actions_path": self.actions_path,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvaluationRow:
        return cls(
            candidate_id=str(data.get("candidate_id", "")),
            ordinal=int(data.get("ordinal", 0)),
            coordinates=dict(data["coordinates"]) if data.get("coordinates") is not None else None,
            params=dict(data.get("params", {})),
            pool_overrides=dict(data.get("pool_overrides", {})),
            metrics=dict(data.get("metrics", {})),
            status=str(data.get("status", "ok")),
            economic_fingerprint=data.get("economic_fingerprint"),
            trace_path=data.get("trace_path"),
            actions_path=data.get("actions_path"),
            tags=tuple(data.get("tags", ())),
        )


class EvaluationTable:
    """One compact, typed NPZ representation shared by grid and optimization."""

    def __init__(
        self,
        rows: Sequence[EvaluationRow] | None = None,
        *,
        metadata: Mapping[str, Any] | None = None,
        metric_projection: MetricProjection | None = None,
    ) -> None:
        self.rows: list[EvaluationRow] = list(rows) if rows is not None else []
        self.metadata: dict[str, Any] = dict(metadata) if metadata is not None else {}
        self.metric_projection = metric_projection

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, index: int) -> EvaluationRow:
        return self.rows[index]

    def add_row(self, row: EvaluationRow) -> None:
        self.rows.append(row)

    def sort_canonical(self) -> EvaluationTable:
        """Sort rows in-place into deterministic canonical order and return self."""
        self.rows.sort(key=lambda row: row.canonical_sort_key())
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _TABLE_SCHEMA_VERSION,
            "metadata": canonical_primitive(self.metadata),
            "metric_projection": self.metric_projection.to_dict() if self.metric_projection else None,
            "row_count": len(self.rows),
            "rows": [row.to_dict() for row in self.rows],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvaluationTable:
        meta = dict(data.get("metadata", {}))
        proj_data = data.get("metric_projection")
        proj = None
        if proj_data and isinstance(proj_data, Mapping):
            proj = MetricProjection(
                fields=tuple(proj_data.get("fields", ())),
                projection_id=proj_data.get("projection_id", "default"),
                projection_sha256=proj_data.get("projection_sha256", ""),
            )
        return cls(
            rows=[EvaluationRow.from_dict(row) for row in data.get("rows", [])],
            metadata=meta,
            metric_projection=proj,
        )

    def to_npz(self, path: Path | str) -> Path:
        """Atomically write a compressed, pickle-free columnar evaluation table."""
        destination = Path(path)
        if destination.suffix != ".npz":
            raise ValueError("evaluation table path must end in .npz")
        destination.parent.mkdir(parents=True, exist_ok=True)
        metric_names = (
            self.metric_projection.fields
            if self.metric_projection is not None
            else tuple(sorted({name for row in self.rows for name in row.metrics}))
        )
        metric_values = np.full((len(self.rows), len(metric_names)), np.nan, dtype=np.float64)
        metric_present = np.zeros(metric_values.shape, dtype=np.bool_)
        metric_extras: list[dict[str, Any]] = []
        for row_index, row in enumerate(self.rows):
            extras: dict[str, Any] = {}
            for metric_index, name in enumerate(metric_names):
                if name not in row.metrics or row.metrics[name] is None:
                    continue
                value = row.metrics[name]
                if not isinstance(value, bool) and isinstance(
                    value, (int, float, np.integer, np.floating)
                ):
                    metric_values[row_index, metric_index] = float(value)
                    metric_present[row_index, metric_index] = True
                else:
                    extras[name] = value
            metric_extras.append(extras)
        economic, economic_present = _optional_strings(
            [row.economic_fingerprint for row in self.rows]
        )
        traces, trace_present = _optional_strings([row.trace_path for row in self.rows])
        actions, actions_present = _optional_strings([row.actions_path for row in self.rows])
        payload = {
            "schema_version": np.asarray(_TABLE_SCHEMA_VERSION),
            "metadata_json": np.asarray(_json_text(self.metadata)),
            "metric_projection_json": np.asarray(
                _json_text(self.metric_projection.to_dict() if self.metric_projection else None)
            ),
            "candidate_id": np.asarray([row.candidate_id for row in self.rows], dtype=np.str_),
            "ordinal": np.asarray([row.ordinal for row in self.rows], dtype=np.int64),
            "coordinates_json": _json_array([row.coordinates for row in self.rows]),
            "params_json": _json_array([row.params for row in self.rows]),
            "pool_overrides_json": _json_array([row.pool_overrides for row in self.rows]),
            "metric_names": np.asarray(metric_names, dtype=np.str_),
            "metric_values": metric_values,
            "metric_present": metric_present,
            "metric_extras_json": _json_array(metric_extras),
            "status": np.asarray([row.status for row in self.rows], dtype=np.str_),
            "economic_fingerprint": economic,
            "economic_fingerprint_present": economic_present,
            "trace_path": traces,
            "trace_path_present": trace_present,
            "actions_path": actions,
            "actions_path_present": actions_present,
            "tags_json": _json_array([list(row.tags) for row in self.rows]),
        }
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                np.savez_compressed(stream, **payload)
            os.replace(temporary_name, destination)
        finally:
            Path(temporary_name).unlink(missing_ok=True)
        return destination

    @classmethod
    def from_npz(cls, path: Path | str) -> EvaluationTable:
        """Load and validate a pickle-free columnar evaluation table."""
        source = Path(path)
        try:
            with np.load(source, allow_pickle=False) as archive:
                if _scalar_text(archive, "schema_version") != _TABLE_SCHEMA_VERSION:
                    raise ValueError("unsupported evaluation-table NPZ schema")
                metadata = json.loads(_scalar_text(archive, "metadata_json"))
                projection_data = json.loads(_scalar_text(archive, "metric_projection_json"))
                candidate_ids = archive["candidate_id"].astype(str)
                ordinals = archive["ordinal"].astype(np.int64)
                statuses = archive["status"].astype(str)
                metric_names = archive["metric_names"].astype(str)
                metric_values = archive["metric_values"].astype(np.float64)
                metric_present = archive["metric_present"].astype(bool)
                row_count = len(candidate_ids)
                one_dimensional = (
                    "coordinates_json",
                    "params_json",
                    "pool_overrides_json",
                    "metric_extras_json",
                    "economic_fingerprint",
                    "economic_fingerprint_present",
                    "trace_path",
                    "trace_path_present",
                    "actions_path",
                    "actions_path_present",
                    "tags_json",
                )
                if any(np.asarray(archive[name]).shape != (row_count,) for name in one_dimensional):
                    raise ValueError("evaluation-table NPZ row columns have inconsistent shapes")
                if ordinals.shape != (row_count,) or statuses.shape != (row_count,):
                    raise ValueError("evaluation-table NPZ typed columns have inconsistent shapes")
                if metric_values.shape != (row_count, len(metric_names)):
                    raise ValueError("evaluation-table NPZ metric matrix has invalid shape")
                if metric_present.shape != metric_values.shape:
                    raise ValueError("evaluation-table NPZ metric presence matrix has invalid shape")
                coordinates = archive["coordinates_json"].astype(str)
                params = archive["params_json"].astype(str)
                overrides = archive["pool_overrides_json"].astype(str)
                metric_extras = archive["metric_extras_json"].astype(str)
                economic = archive["economic_fingerprint"].astype(str)
                economic_present = archive["economic_fingerprint_present"].astype(bool)
                traces = archive["trace_path"].astype(str)
                trace_present = archive["trace_path_present"].astype(bool)
                actions = archive["actions_path"].astype(str)
                actions_present = archive["actions_path_present"].astype(bool)
                tags = archive["tags_json"].astype(str)
                rows = []
                for index in range(row_count):
                    metrics = {
                        str(name): float(metric_values[index, metric_index])
                        for metric_index, name in enumerate(metric_names)
                        if metric_present[index, metric_index]
                    }
                    metrics.update(dict(json.loads(metric_extras[index])))
                    raw_coordinates = json.loads(coordinates[index])
                    rows.append(
                        EvaluationRow(
                            candidate_id=str(candidate_ids[index]),
                            ordinal=int(ordinals[index]),
                            coordinates=(
                                dict(raw_coordinates) if raw_coordinates is not None else None
                            ),
                            params=dict(json.loads(params[index])),
                            pool_overrides=dict(json.loads(overrides[index])),
                            metrics=metrics,
                            status=str(statuses[index]),
                            economic_fingerprint=(
                                str(economic[index]) if economic_present[index] else None
                            ),
                            trace_path=str(traces[index]) if trace_present[index] else None,
                            actions_path=str(actions[index]) if actions_present[index] else None,
                            tags=tuple(json.loads(tags[index])),
                        )
                    )
        except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid evaluation-table NPZ {source}: {exc}") from exc
        projection = (
            MetricProjection(
                fields=tuple(projection_data.get("fields", ())),
                projection_id=str(projection_data.get("projection_id", "default")),
                projection_sha256=str(projection_data.get("projection_sha256", "")),
            )
            if isinstance(projection_data, Mapping)
            else None
        )
        return cls(rows=rows, metadata=dict(metadata), metric_projection=projection)


__all__ = ["MetricProjection", "EvaluationRow", "EvaluationTable"]
