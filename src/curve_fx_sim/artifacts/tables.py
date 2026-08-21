"""Pickle-free evaluation tables for optimization rows and Cartesian grids."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from ..specs.common import canonical_json_bytes, canonical_primitive

if TYPE_CHECKING:
    from ..grids.model import CartesianGridPlan

OPTIMIZATION_TABLE_SCHEMA_VERSION = "fxsim_evaluation_table_npz_v2"
GRID_TABLE_SCHEMA_VERSION = "fxsim_evaluation_table_npz_v3"
_STATUS_NAMES = ("ok", "failed", "cancelled")
_LOWER_SHA = re.compile(r"^[0-9a-f]{64}$")


def _sha_bytes(value: bytes) -> bool:
    try:
        return len(value) == 64 and _LOWER_SHA.fullmatch(value.decode("ascii")) is not None
    except UnicodeDecodeError:
        return False


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
    item = value.item()
    return item.decode("ascii") if isinstance(item, bytes) else str(item)


def _write_npz(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return path


@dataclass(frozen=True)
class MetricProjection:
    """Attested subset projection of raw simulation metrics."""

    fields: tuple[str, ...]
    projection_id: str = "default"
    projection_sha256: str = ""

    def __post_init__(self) -> None:
        if not self.fields or any(not isinstance(name, str) or not name for name in self.fields):
            raise ValueError("metric projection fields must be non-empty strings")
        if len(self.fields) != len(set(self.fields)):
            raise ValueError("metric projection fields must be unique")
        digest = hashlib.sha256(canonical_json_bytes(list(self.fields))).hexdigest()
        if self.projection_sha256 and self.projection_sha256 != digest:
            raise ValueError("metric projection digest does not match fields")
        object.__setattr__(self, "projection_sha256", digest)

    @classmethod
    def from_fields(cls, fields: Sequence[str], projection_id: str = "custom") -> MetricProjection:
        return cls(tuple(sorted(str(field) for field in fields)), projection_id)

    def project_metrics(self, raw_metrics: Mapping[str, Any]) -> dict[str, Any]:
        return {field: raw_metrics[field] for field in self.fields if field in raw_metrics}

    def to_dict(self) -> dict[str, Any]:
        return {
            "projection_id": self.projection_id,
            "fields": list(self.fields),
            "projection_sha256": self.projection_sha256,
        }


@dataclass
class EvaluationRow:
    """One materialized result; Cartesian tables create it only on selection."""

    candidate_id: str
    ordinal: int = 0
    coordinates: dict[str, Any] | None = None
    params: dict[str, Any] = field(default_factory=dict)
    pool_overrides: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    economic_fingerprint: str | None = None
    error: str | None = None
    trace_path: str | None = None
    actions_path: str | None = None
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id, "ordinal": self.ordinal,
            "coordinates": canonical_primitive(self.coordinates),
            "params": canonical_primitive(self.params),
            "pool_overrides": canonical_primitive(self.pool_overrides),
            "metrics": canonical_primitive(self.metrics), "status": self.status,
            "economic_fingerprint": self.economic_fingerprint, "error": self.error,
            "trace_path": self.trace_path, "actions_path": self.actions_path,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvaluationRow:
        return cls(
            candidate_id=str(data.get("candidate_id", "")), ordinal=int(data.get("ordinal", 0)),
            coordinates=dict(data["coordinates"]) if data.get("coordinates") is not None else None,
            params=dict(data.get("params", {})), pool_overrides=dict(data.get("pool_overrides", {})),
            metrics=dict(data.get("metrics", {})), status=str(data.get("status", "ok")),
            economic_fingerprint=data.get("economic_fingerprint"), error=data.get("error"),
            trace_path=data.get("trace_path"), actions_path=data.get("actions_path"),
            tags=tuple(data.get("tags", ())),
        )


class EvaluationTable:
    """The sole table abstraction, lazy and ordinal-backed for Cartesian grids."""

    def __init__(self, rows: Sequence[EvaluationRow] | None = None, *,
                 metadata: Mapping[str, Any] | None = None,
                 metric_projection: MetricProjection | None = None,
                 _grid_path: Path | None = None, _grid_status: np.ndarray | None = None,
                 _row_count: int | None = None, _plan_sha256: str | None = None) -> None:
        self._rows = list(rows) if rows is not None else []
        self.metadata = dict(metadata or {})
        self.metric_projection = metric_projection
        self._grid_path, self._grid_status = _grid_path, _grid_status
        self._row_count = len(self._rows) if _row_count is None else _row_count
        self._plan_sha256 = _plan_sha256

    @property
    def rows(self) -> list[EvaluationRow]:
        if self._grid_path is not None:
            raise TypeError("Cartesian grid rows are lazy; use row_at(ordinal, plan)")
        return self._rows

    @property
    def row_count(self) -> int:
        return self._row_count

    @property
    def is_columnar_grid(self) -> bool:
        return self._grid_path is not None

    @property
    def plan_sha256(self) -> str | None:
        return self._plan_sha256

    def __len__(self) -> int:
        return self.row_count

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, index: int) -> EvaluationRow:
        return self.rows[index]

    def add_row(self, row: EvaluationRow) -> None:
        self.rows.append(row)
        self._row_count = len(self._rows)

    def metric_array(self, name: str) -> np.ndarray:
        projection = self.metric_projection
        if projection is None or name not in projection.fields:
            raise KeyError(f"unknown metric {name!r}")
        if self._grid_path is None:
            values = np.full(self.row_count, np.nan, dtype=np.float64)
            for index, row in enumerate(self._rows):
                if name in row.metrics and row.metrics[name] is not None:
                    values[index] = float(row.metrics[name])
            return values
        index = projection.fields.index(name)
        with np.load(self._grid_path, allow_pickle=False) as archive:
            values = np.asarray(archive[f"metric_{index:03d}_values"], dtype=np.float64)
            present = np.asarray(archive[f"metric_{index:03d}_present"], dtype=bool)
        if values.shape != (self.row_count,) or present.shape != values.shape:
            raise ValueError(f"evaluation-table metric {name!r} has invalid shape")
        values[~present] = np.nan
        return values

    def status_at(self, ordinal: int) -> str:
        self._check_ordinal(ordinal)
        if self._grid_path is None:
            return self._rows[ordinal].status
        assert self._grid_status is not None
        code = int(self._grid_status[ordinal])
        if code >= len(_STATUS_NAMES):
            raise ValueError(f"invalid Cartesian grid status code {code}")
        return _STATUS_NAMES[code]

    def economic_fingerprint_at(self, ordinal: int) -> str | None:
        self._check_ordinal(ordinal)
        if self._grid_path is None:
            return self._rows[ordinal].economic_fingerprint
        with np.load(self._grid_path, allow_pickle=False) as archive:
            present = bool(archive["economic_fingerprint_present"][ordinal])
            raw = archive["economic_fingerprint"][ordinal]
        return bytes(raw).decode("ascii") if present else None

    def _error_at(self, ordinal: int) -> str | None:
        if self._grid_path is None:
            return self._rows[ordinal].error
        with np.load(self._grid_path, allow_pickle=False) as archive:
            offsets = archive["error_offsets"]
            start, end = int(offsets[ordinal]), int(offsets[ordinal + 1])
            if start == end:
                return None
            return bytes(archive["error_utf8"][start:end]).decode("utf-8")

    def row_at(self, ordinal: int, plan: CartesianGridPlan) -> EvaluationRow:
        """Reconstruct one exact Cartesian row without materializing the grid."""
        self._check_ordinal(ordinal)
        if self._grid_path is None:
            return self._rows[ordinal]
        if plan.plan_sha256 != self._plan_sha256 or plan.pool_count != self.row_count:
            raise ValueError("Cartesian grid plan does not match evaluation table")
        point = plan.point_at(ordinal)
        metrics: dict[str, float] = {}
        assert self.metric_projection is not None
        for name in self.metric_projection.fields:
            value = self.metric_array(name)[ordinal]
            if np.isfinite(value):
                metrics[name] = float(value)
        return EvaluationRow(
            point.candidate_id, ordinal, dict(point.coordinates),
            {"vector": list(point.policy_params)}, dict(point.pool_overrides), metrics,
            self.status_at(ordinal), self.economic_fingerprint_at(ordinal),
            self._error_at(ordinal), tags=("grid",),
        )

    def _check_ordinal(self, ordinal: int) -> None:
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or not 0 <= ordinal < self.row_count:
            raise IndexError(f"evaluation ordinal {ordinal!r} outside [0, {self.row_count})")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": GRID_TABLE_SCHEMA_VERSION if self.is_columnar_grid else OPTIMIZATION_TABLE_SCHEMA_VERSION,
            "metadata": canonical_primitive(self.metadata),
            "metric_projection": self.metric_projection.to_dict() if self.metric_projection else None,
            "row_count": self.row_count,
            "rows": [row.to_dict() for row in self.rows] if not self.is_columnar_grid else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvaluationTable:
        return cls([EvaluationRow.from_dict(row) for row in data.get("rows", [])],
                   metadata=dict(data.get("metadata", {})),
                   metric_projection=_projection(data.get("metric_projection")))

    def to_npz(self, path: Path | str) -> Path:
        """Write optimization rows; Cartesian artifacts are produced by collection."""
        if self.is_columnar_grid:
            raise ValueError("a collected Cartesian grid table is immutable")
        destination = Path(path)
        if destination.suffix != ".npz":
            raise ValueError("evaluation table path must end in .npz")
        projection = self.metric_projection
        if projection is None:
            raise ValueError("evaluation table requires a non-empty metric projection")
        names = projection.fields
        values = np.full((self.row_count, len(names)), np.nan, dtype="<f8")
        present = np.zeros(values.shape, dtype=np.bool_)
        for row_index, row in enumerate(self._rows):
            unknown = set(row.metrics) - set(names)
            if unknown:
                raise ValueError(f"evaluation metrics are outside the projection: {sorted(unknown)!r}")
            for metric_index, name in enumerate(names):
                if name not in row.metrics or row.metrics[name] is None:
                    continue
                raw = row.metrics[name]
                if isinstance(raw, bool) or not np.isfinite(float(raw)):
                    raise ValueError(f"metric {name!r} row {row_index} must be finite numeric")
                values[row_index, metric_index], present[row_index, metric_index] = float(raw), True
        economic, economic_present = _optional_strings([row.economic_fingerprint for row in self._rows])
        traces, trace_present = _optional_strings([row.trace_path for row in self._rows])
        actions, actions_present = _optional_strings([row.actions_path for row in self._rows])
        return _write_npz(destination, {
            "schema_version": np.asarray(OPTIMIZATION_TABLE_SCHEMA_VERSION),
            "metadata_json": np.asarray(_json_text(self.metadata)),
            "metric_projection_json": np.asarray(_json_text(projection.to_dict())),
            "candidate_id": np.asarray([row.candidate_id for row in self._rows], dtype=np.str_),
            "ordinal": np.asarray([row.ordinal for row in self._rows], dtype="<i8"),
            "coordinates_json": _json_array([row.coordinates for row in self._rows]),
            "params_json": _json_array([row.params for row in self._rows]),
            "pool_overrides_json": _json_array([row.pool_overrides for row in self._rows]),
            "metric_names": np.asarray(names, dtype=np.str_), "metric_values": values,
            "metric_present": present, "status": np.asarray([row.status for row in self._rows], dtype=np.str_),
            "economic_fingerprint": economic, "economic_fingerprint_present": economic_present,
            "trace_path": traces, "trace_path_present": trace_present,
            "actions_path": actions, "actions_path_present": actions_present,
            "tags_json": _json_array([list(row.tags) for row in self._rows]),
        })

    @classmethod
    def from_npz(cls, path: Path | str) -> EvaluationTable:
        source = Path(path)
        try:
            with np.load(source, allow_pickle=False) as archive:
                schema = _scalar_text(archive, "schema_version")
                if schema == GRID_TABLE_SCHEMA_VERSION:
                    return cls._from_grid_npz(source, archive)
                if schema != OPTIMIZATION_TABLE_SCHEMA_VERSION:
                    raise ValueError(f"unsupported evaluation-table NPZ schema {schema!r}")
                return cls._from_optimization_npz(archive)
        except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid evaluation-table NPZ {source}: {exc}") from exc

    @classmethod
    def _from_grid_npz(cls, source: Path, archive: Mapping[str, Any]) -> EvaluationTable:
        count = int(np.asarray(archive["row_count"]).item())
        projection = _projection(json.loads(_scalar_text(archive, "metric_projection_json")))
        if _scalar_text(archive, "table_kind") != "cartesian_grid":
            raise ValueError("evaluation-table kind is not cartesian_grid")
        if projection is None or tuple(archive["metric_names"].astype(str)) != projection.fields:
            raise ValueError("Cartesian grid metric columns differ from projection")
        projection_sha = np.asarray(archive["metric_projection_sha256"])
        if projection_sha.shape != () or bytes(projection_sha.item()).decode() != projection.projection_sha256:
            raise ValueError("Cartesian grid projection identity is invalid")
        status = np.asarray(archive["status"])
        if status.dtype != np.dtype("u1") or status.shape != (count,) or np.any(status > 2):
            raise ValueError("Cartesian grid status column is invalid")
        fingerprint = np.asarray(archive["economic_fingerprint"])
        fingerprint_present = np.asarray(archive["economic_fingerprint_present"])
        if (fingerprint.dtype != np.dtype("S64") or fingerprint.shape != (count,)
                or fingerprint_present.dtype != np.dtype(bool)
                or fingerprint_present.shape != (count,)):
            raise ValueError("Cartesian grid fingerprint column is invalid")
        for raw, present in zip(fingerprint, fingerprint_present, strict=True):
            value = bytes(raw)
            if (present and not _sha_bytes(value)) or (not present and value):
                raise ValueError("Cartesian grid fingerprint presence is invalid")
        offsets, errors = np.asarray(archive["error_offsets"]), np.asarray(archive["error_utf8"])
        if (offsets.dtype != np.dtype("<i8") or offsets.shape != (count + 1,)
                or offsets[0] != 0 or np.any(np.diff(offsets) < 0)
                or errors.dtype != np.dtype("u1") or offsets[-1] != len(errors)):
            raise ValueError("Cartesian grid error columns are invalid")
        try:
            errors.tobytes().decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Cartesian grid errors are not UTF-8") from exc
        if np.any((status == 0) & (np.diff(offsets) != 0)):
            raise ValueError("successful Cartesian grid rows contain errors")
        expected_fields = {
            "schema_version", "table_kind", "run_id", "plan_sha256", "artifact_sha256",
            "metric_projection_sha256", "row_count", "metric_names",
            "metric_projection_json", "metadata_json", "status", "economic_fingerprint",
            "economic_fingerprint_present", "error_offsets", "error_utf8",
            *(f"metric_{index:03d}_{suffix}" for index in range(len(projection.fields)) for suffix in ("values", "present")),
        }
        if set(archive.files) != expected_fields:
            raise ValueError("Cartesian grid evaluation-table fields are invalid")
        for index in range(len(projection.fields)):
            values = np.asarray(archive[f"metric_{index:03d}_values"])
            present = np.asarray(archive[f"metric_{index:03d}_present"])
            if values.dtype != np.dtype("<f8") or values.shape != (count,) or present.dtype != np.dtype(bool) or present.shape != (count,):
                raise ValueError("Cartesian grid metric column is invalid")
            if np.any(present & ~np.isfinite(values)) or np.any(~present & ~np.isnan(values)) or np.any((status == 0) & ~present):
                raise ValueError("Cartesian grid metric presence semantics are invalid")
        return cls(metadata=json.loads(_scalar_text(archive, "metadata_json")),
                   metric_projection=projection, _grid_path=source,
                   _grid_status=status.copy(), _row_count=count,
                   _plan_sha256=_scalar_text(archive, "plan_sha256"))

    @classmethod
    def _from_optimization_npz(cls, archive: Mapping[str, Any]) -> EvaluationTable:
        projection = _projection(json.loads(_scalar_text(archive, "metric_projection_json")))
        if projection is None:
            raise ValueError("evaluation table requires a non-empty metric projection")
        candidate_ids, ordinals = archive["candidate_id"].astype(str), archive["ordinal"].astype(np.int64)
        names, values, present = archive["metric_names"].astype(str), archive["metric_values"], archive["metric_present"]
        count = len(candidate_ids)
        if tuple(names) != projection.fields or values.shape != (count, len(names)) or present.shape != values.shape:
            raise ValueError("evaluation-table metric matrix is invalid")
        if not np.array_equal(ordinals, np.arange(count, dtype=np.int64)):
            raise ValueError("evaluation-table NPZ ordinals must be contiguous from zero")
        if values.dtype != np.dtype("<f8") or present.dtype != np.dtype(bool):
            raise ValueError("evaluation-table metric dtypes are invalid")
        if np.any(present & ~np.isfinite(values)) or np.any(~present & ~np.isnan(values)):
            raise ValueError("evaluation-table metric values violate presence semantics")
        columns = {name: archive[name].astype(str) for name in ("coordinates_json", "params_json", "pool_overrides_json", "tags_json")}
        economic, economic_present = archive["economic_fingerprint"].astype(str), archive["economic_fingerprint_present"].astype(bool)
        traces, trace_present = archive["trace_path"].astype(str), archive["trace_path_present"].astype(bool)
        actions, actions_present = archive["actions_path"].astype(str), archive["actions_path_present"].astype(bool)
        statuses = archive["status"].astype(str)
        if any(column.shape != (count,) for column in (*columns.values(), economic, economic_present, traces, trace_present, actions, actions_present, statuses)):
            raise ValueError("evaluation-table row columns have inconsistent shapes")
        rows = []
        for index in range(count):
            coordinate = json.loads(columns["coordinates_json"][index])
            rows.append(EvaluationRow(str(candidate_ids[index]), int(ordinals[index]),
                dict(coordinate) if coordinate is not None else None,
                dict(json.loads(columns["params_json"][index])), dict(json.loads(columns["pool_overrides_json"][index])),
                {str(name): float(values[index, metric]) for metric, name in enumerate(names) if present[index, metric]},
                str(statuses[index]), str(economic[index]) if economic_present[index] else None,
                trace_path=str(traces[index]) if trace_present[index] else None,
                actions_path=str(actions[index]) if actions_present[index] else None,
                tags=tuple(json.loads(columns["tags_json"][index]))))
        return cls(rows, metadata=json.loads(_scalar_text(archive, "metadata_json")), metric_projection=projection)


def _projection(value: Any) -> MetricProjection | None:
    if not isinstance(value, Mapping):
        return None
    return MetricProjection(tuple(value.get("fields", ())), str(value.get("projection_id", "default")), str(value.get("projection_sha256", "")))


__all__ = ["GRID_TABLE_SCHEMA_VERSION", "OPTIMIZATION_TABLE_SCHEMA_VERSION",
           "MetricProjection", "EvaluationRow", "EvaluationTable"]
