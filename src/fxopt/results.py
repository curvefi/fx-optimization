"""Canonical two-file result artifacts for optimizer runs."""

from __future__ import annotations

import json
import os
import orjson
import shutil
import tempfile
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from .candidates import candidate_id
from .contract import Candidate, CandidateResult
from .grid import CartesianGrid

if TYPE_CHECKING:
    from .engine import ProjectedBatch


SCHEMA_VERSION = "fxopt.results.v2"
GRID_SCHEMA_VERSION = "fxopt.grid-results.v1"
RUN_FILENAME = "run.json"
RESULTS_FILENAME = "results.npz"

_STATUS_TO_CODE = {"ok": 0, "failed": 1, "cancelled": 2}
_CODE_TO_STATUS = tuple(_STATUS_TO_CODE)


@dataclass(frozen=True, slots=True)
class ResultBundle:
    run_id: str
    candidates: tuple[Candidate, ...]
    results: tuple[CandidateResult, ...]
    metadata: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        if len(self.candidates) != len(self.results):
            raise ValueError("candidates and results must have equal length")
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        result_ids = [result.candidate_id for result in self.results]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate IDs must be unique")
        if result_ids != candidate_ids:
            raise ValueError("results must be ordered like candidates")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")


@dataclass(frozen=True, slots=True)
class ArtifactPaths:
    run_json: Path
    results_npz: Path


@dataclass(frozen=True, slots=True)
class ResultColumns:
    """Selected NumPy columns plus lazy access to one stored candidate."""

    root: Path
    run_id: str
    metadata: Mapping[str, Any]
    available_metrics: tuple[str, ...]
    candidate_ids: np.ndarray | None
    ordinals: np.ndarray
    statuses: np.ndarray | None
    metrics: Mapping[str, np.ndarray]
    errors: np.ndarray | None
    error_present: np.ndarray | None
    status_codes: np.ndarray | None = None
    failures: Mapping[int, str] = MappingProxyType({})
    grid: CartesianGrid | None = None

    @property
    def row_count(self) -> int:
        return int(self.ordinals.shape[0])

    def row_for_ordinal(self, ordinal: int) -> int:
        if not isinstance(ordinal, int) or ordinal < 0:
            raise ValueError("ordinal must be a non-negative integer")
        rows = np.flatnonzero(self.ordinals == ordinal)
        if len(rows) != 1:
            raise ValueError(f"result artifact has no unique row for ordinal {ordinal}")
        return int(rows[0])

    @property
    def ok_mask(self) -> np.ndarray:
        if self.status_codes is not None:
            return self.status_codes == _STATUS_TO_CODE["ok"]
        assert self.statuses is not None
        return self.statuses == "ok"

    def status_at(self, row: int) -> str:
        if self.status_codes is not None:
            code = int(self.status_codes[row])
            if code < 0 or code >= len(_CODE_TO_STATUS):
                raise ValueError(f"invalid stored status code: {code}")
            return _CODE_TO_STATUS[code]
        assert self.statuses is not None
        return str(self.statuses[row])

    def error_at(self, row: int) -> str | None:
        if self.error_present is not None and self.errors is not None:
            return str(self.errors[row]) if bool(self.error_present[row]) else None
        return self.failures.get(int(self.ordinals[row]))

    def candidate_ids_array(self) -> np.ndarray:
        if self.candidate_ids is not None:
            return self.candidate_ids
        width = len(candidate_id(max(0, self.row_count - 1)))
        return np.fromiter(
            (candidate_id(int(ordinal)) for ordinal in self.ordinals),
            dtype=f"U{width}",
            count=self.row_count,
        )

    def candidate_at(self, ordinal: int) -> Candidate:
        """Construct only the candidate selected by its stored result ordinal."""
        return self._candidate_at_row(self.row_for_ordinal(ordinal))

    def _candidate_at_row(self, row: int) -> Candidate:
        if self.grid is not None:
            spec = self.grid.candidate_at(int(self.ordinals[row]))
            payload = spec.payload
            policy_params = payload.get("policy_params")
            pool = payload.get("pool")
            if not isinstance(policy_params, (list, tuple)) or not isinstance(pool, Mapping):
                raise ValueError("grid metadata cannot reconstruct candidate payloads")
            return Candidate(
                candidate_id=spec.candidate_id,
                policy_params=tuple(policy_params),
                pool_overrides=pool,
            )
        assert self.candidate_ids is not None
        try:
            with np.load(self.root / RESULTS_FILENAME, allow_pickle=False) as archive:
                offsets = archive["candidate_policy_offsets"]
                values = archive["candidate_policy_values"]
                overrides = archive["candidate_pool_overrides"]
        except (OSError, KeyError, ValueError) as exc:
            raise ValueError(f"invalid {RESULTS_FILENAME}") from exc
        start, stop = int(offsets[row]), int(offsets[row + 1])
        try:
            pool_overrides = json.loads(str(overrides[row]))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("invalid stored candidate pool overrides") from exc
        raw_id = self.candidate_ids[row]
        return Candidate(
            candidate_id=(raw_id.decode() if isinstance(raw_id, bytes) else str(raw_id)),
            policy_params=values[start:stop].tolist(),
            pool_overrides=pool_overrides,
        )


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TypeError("artifact metadata must be finite JSON") from exc


def _row_json_bytes(value: Any) -> bytes:
    """Encode already-validated candidate/result rows on the hot path."""
    return orjson.dumps(value, option=orjson.OPT_SORT_KEYS)


def _atomic_bytes(path: Path, data: bytes) -> None:
    handle = tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    handle = tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".npz", delete=False)
    temporary = Path(handle.name)
    handle.close()
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _artifact_paths(root: Path) -> ArtifactPaths:
    return ArtifactPaths(root / RUN_FILENAME, root / RESULTS_FILENAME)


def _metric_key(column: int) -> str:
    return f"metric_{column:04d}"


def _grid_member(kind: str, shard: int) -> str:
    return f"{kind}_{shard:08d}.npy"


def _grid_metric_member(column: int, shard: int) -> str:
    return f"metric_{column:04d}_{shard:08d}.npy"


def _read_npy_member(archive: zipfile.ZipFile, name: str) -> np.ndarray:
    with archive.open(name, "r") as stream:
        return np.lib.format.read_array(stream, allow_pickle=False)


def _write_npy_member(
    archive: zipfile.ZipFile, name: str, values: np.ndarray
) -> None:
    with archive.open(name, "w", force_zip64=True) as stream:
        np.lib.format.write_array(stream, np.asarray(values), allow_pickle=False)


def _grid_from_metadata(metadata: Mapping[str, Any], count: int) -> CartesianGrid:
    defaults = metadata.get("candidate_defaults")
    axes = metadata.get("axes")
    if not isinstance(defaults, Mapping) or not isinstance(axes, Mapping):
        raise ValueError("compact grid results require candidate defaults and axes")
    grid = CartesianGrid(
        dict(defaults),
        {
            str(name): tuple(values)
            for name, values in axes.items()
            if isinstance(name, str) and isinstance(values, list)
        },
    )
    if len(grid) != count or len(grid.axes) != len(axes):
        raise ValueError("compact grid metadata does not match candidate count")
    return grid


def write_results(bundle: ResultBundle, directory: str | Path) -> ArtifactPaths:
    """Write exactly ``run.json`` and ``results.npz`` for one completed run."""
    if not isinstance(bundle, ResultBundle):
        raise TypeError("bundle must be a ResultBundle")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(root)
    metric_names = sorted({name for result in bundle.results for name in result.metrics})
    metric_values = {
        name: np.full(len(bundle.results), np.nan, dtype=np.float64)
        for name in metric_names
    }
    for row, result in enumerate(bundle.results):
        for name, value in result.metrics.items():
            metric_values[name][row] = value

    policy_lengths = np.asarray([len(candidate.policy_params) for candidate in bundle.candidates])
    policy_offsets = np.empty(len(bundle.candidates) + 1, dtype=np.int64)
    policy_offsets[0] = 0
    np.cumsum(policy_lengths, out=policy_offsets[1:])
    policy_values = np.asarray(
        [value for candidate in bundle.candidates for value in candidate.policy_params],
        dtype=np.float64,
    )

    run_payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": bundle.run_id,
        "metric_names": metric_names,
        "candidate_count": len(bundle.candidates),
        "metadata": dict(bundle.metadata),
    }
    arrays = {
        "candidate_ids": np.asarray([r.candidate_id for r in bundle.results], dtype=str),
        "candidate_policy_offsets": policy_offsets,
        "candidate_policy_values": policy_values,
        "candidate_pool_overrides": np.asarray([
            _canonical_json_bytes(dict(candidate.pool_overrides)).decode()
            for candidate in bundle.candidates
        ], dtype=str),
        "ordinals": np.asarray([r.ordinal for r in bundle.results], dtype=np.int64),
        "statuses": np.asarray([r.status for r in bundle.results], dtype=str),
        "errors": np.asarray([r.error or "" for r in bundle.results], dtype=str),
        "error_present": np.asarray([r.error is not None for r in bundle.results], dtype=np.bool_),
        **{
            _metric_key(column): metric_values[name]
            for column, name in enumerate(metric_names)
        },
    }
    _atomic_npz(
        paths.results_npz,
        arrays,
    )
    _atomic_bytes(paths.run_json, _canonical_json_bytes(run_payload))
    return paths


def read_result_columns(
    directory: str | Path,
    *,
    metrics: Sequence[str] | None = None,
) -> ResultColumns:
    """Load base result arrays and only the requested metric columns."""
    root = Path(directory)
    run_path = root / RUN_FILENAME
    try:
        payload = json.loads(run_path.read_bytes())
        schema = payload.get("schema_version")
        if schema not in {SCHEMA_VERSION, GRID_SCHEMA_VERSION}:
            raise ValueError("unsupported result artifact schema")
        metric_names = tuple(payload["metric_names"])
        if len(set(metric_names)) != len(metric_names) or not all(
            isinstance(name, str) and name for name in metric_names
        ):
            raise ValueError("invalid metric names")
        count = int(payload["candidate_count"])
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("invalid metadata")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {RUN_FILENAME}") from exc

    selected = metric_names if metrics is None else tuple(dict.fromkeys(metrics))
    missing = [name for name in selected if name not in metric_names]
    if missing:
        raise ValueError(f"unknown result metrics: {', '.join(missing)}")
    columns = {name: index for index, name in enumerate(metric_names)}

    if schema == GRID_SCHEMA_VERSION:
        shard_count = payload.get("shard_count")
        if isinstance(shard_count, bool) or not isinstance(shard_count, int) or shard_count < 1:
            raise ValueError("compact grid result has invalid shard count")
        selected_metrics = {
            name: np.empty(count, dtype=np.float64) for name in selected
        }
        status_codes = np.empty(count, dtype=np.uint8)
        written = np.zeros(count, dtype=np.bool_)
        failures: dict[int, str] = {}
        try:
            with zipfile.ZipFile(root / RESULTS_FILENAME, "r") as archive:
                members = set(archive.namelist())
                for shard in range(shard_count):
                    index_name = _grid_member("index", shard)
                    metric_members = tuple(
                        _grid_metric_member(column, shard)
                        for column in range(len(metric_names))
                    )
                    if index_name not in members or any(
                        name not in members for name in metric_members
                    ):
                        raise ValueError(f"compact grid result is missing shard {shard}")
                    index = _read_npy_member(archive, index_name)
                    if (
                        index.dtype.names != ("ordinal", "status")
                        or index.ndim != 1
                    ):
                        raise ValueError(f"compact grid shard {shard} has invalid arrays")
                    ordinals = np.asarray(index["ordinal"], dtype=np.int64)
                    if (
                        np.any(ordinals < 0)
                        or np.any(ordinals >= count)
                        or len(np.unique(ordinals)) != len(ordinals)
                        or np.any(written[ordinals])
                    ):
                        raise ValueError(f"compact grid shard {shard} has invalid ordinals")
                    written[ordinals] = True
                    status_codes[ordinals] = index["status"]
                    for name in selected:
                        values = _read_npy_member(
                            archive, metric_members[columns[name]]
                        )
                        if values.shape != (len(index),):
                            raise ValueError(
                                f"compact grid shard {shard} has invalid arrays"
                            )
                        selected_metrics[name][ordinals] = values
                    errors_name = _grid_member("errors", shard)
                    if errors_name in members:
                        errors = _read_npy_member(archive, errors_name)
                        if errors.dtype.names != ("ordinal", "error") or errors.ndim != 1:
                            raise ValueError(
                                f"compact grid shard {shard} has invalid errors"
                            )
                        for item in errors:
                            ordinal = int(item["ordinal"])
                            if ordinal not in failures:
                                failures[ordinal] = str(item["error"])
        except (OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
            raise ValueError(f"invalid {RESULTS_FILENAME}") from exc
        if not np.all(written) or np.any(status_codes >= len(_CODE_TO_STATUS)):
            raise ValueError("compact grid shards do not cover the grid")
        return ResultColumns(
            root=root,
            run_id=payload["run_id"],
            metadata=dict(metadata),
            available_metrics=metric_names,
            candidate_ids=None,
            ordinals=np.arange(count, dtype=np.int64),
            statuses=None,
            metrics=MappingProxyType(selected_metrics),
            errors=None,
            error_present=None,
            status_codes=status_codes,
            failures=MappingProxyType(failures),
            grid=_grid_from_metadata(metadata, count),
        )

    try:
        with np.load(root / RESULTS_FILENAME, allow_pickle=False) as archive:
            candidate_ids = archive["candidate_ids"]
            ordinals = archive["ordinals"]
            statuses = archive["statuses"]
            errors = archive["errors"]
            error_present = archive["error_present"]
            selected_metrics = {
                name: archive[_metric_key(columns[name])]
                for name in selected
            }
    except (OSError, KeyError, ValueError) as exc:
        raise ValueError(f"invalid {RESULTS_FILENAME}") from exc
    arrays = (candidate_ids, ordinals, statuses, errors, error_present, *selected_metrics.values())
    if any(array.shape != (count,) for array in arrays):
        raise ValueError("result arrays have inconsistent lengths")
    if len(set(int(value) for value in ordinals.tolist())) != count:
        raise ValueError("result ordinals must be unique")
    return ResultColumns(
        root=root,
        run_id=payload["run_id"],
        metadata=dict(metadata),
        available_metrics=metric_names,
        candidate_ids=candidate_ids,
        ordinals=ordinals,
        statuses=statuses,
        metrics=MappingProxyType(selected_metrics),
        errors=errors,
        error_present=error_present,
    )


def read_results(directory: str | Path) -> ResultBundle:
    """Read and validate the canonical two-file result artifact."""
    columns = read_result_columns(directory)
    candidates = tuple(columns._candidate_at_row(row) for row in range(columns.row_count))
    results: list[CandidateResult] = []
    for row in range(columns.row_count):
        metrics = {
            name: float(values[row])
            for name, values in columns.metrics.items()
            if np.isfinite(values[row])
        }
        results.append(
            CandidateResult(
                candidate_id=candidates[row].candidate_id,
                status=columns.status_at(row),
                metrics=metrics,
                error=columns.error_at(row),
                ordinal=int(columns.ordinals[row]),
            )
        )
    return ResultBundle(
        run_id=columns.run_id,
        candidates=candidates,
        results=tuple(results),
        metadata=columns.metadata,
    )


class GridResultWriter:
    """Stream typed Cartesian shards into one atomically published NPZ."""

    def __init__(
        self,
        directory: str | Path,
        *,
        run_id: str,
        total: int,
        metadata: Mapping[str, Any],
        metric_names: Sequence[str] | None = None,
        shard_rows: int = 65_536,
    ) -> None:
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        if isinstance(total, bool) or not isinstance(total, int) or total < 1:
            raise ValueError("grid result total must be a positive integer")
        names = None if metric_names is None else tuple(sorted(metric_names))
        if names is not None and (
            not names
            or len(set(names)) != len(names)
            or any(not isinstance(name, str) or not name for name in names)
        ):
            raise ValueError("metric names must be unique non-empty strings")
        if (
            isinstance(shard_rows, bool)
            or not isinstance(shard_rows, int)
            or shard_rows < 1
        ):
            raise ValueError("shard_rows must be a positive integer")

        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._paths = _artifact_paths(self.directory)
        if self._paths.run_json.exists() or self._paths.results_npz.exists():
            raise FileExistsError(f"completed result already exists in {self.directory}")
        self._temporary = self.directory / f".{RESULTS_FILENAME}.tmp"
        self._temporary.unlink(missing_ok=True)
        self._archive = zipfile.ZipFile(
            self._temporary,
            mode="w",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        )
        self._run_id = run_id
        self._total = total
        self._metadata = dict(metadata)
        self._metric_names = names
        self._shard_rows = shard_rows
        self._written = np.zeros(total, dtype=np.bool_)
        self._ordinal_dtype = np.dtype("<u4" if total <= np.iinfo(np.uint32).max else "<u8")
        self._index_dtype = np.dtype(
            [("ordinal", self._ordinal_dtype), ("status", "u1")]
        )
        self._buffer_index = np.empty(shard_rows, dtype=self._index_dtype)
        self._buffer_values: np.ndarray | None = None
        self._buffer_errors: list[tuple[int, str]] = []
        self._buffer_count = 0
        self._status_counts = {name: 0 for name in _STATUS_TO_CODE}
        self._count = 0
        self._shard_count = 0
        self._closed = False

    @property
    def retained_rows(self) -> int:
        return self._buffer_count

    @property
    def row_count(self) -> int:
        return self._count

    def append(
        self,
        candidates: Sequence[Candidate],
        results: Sequence[CandidateResult],
    ) -> None:
        if self._closed:
            raise RuntimeError("grid result writer is closed")
        pairs = tuple(zip(candidates, results, strict=True))
        if not pairs:
            return
        for candidate, result in pairs:
            if not isinstance(candidate, Candidate) or not isinstance(result, CandidateResult):
                raise TypeError("append expects Candidate and CandidateResult values")
            if (
                candidate.candidate_id != result.candidate_id
                or candidate.candidate_id != candidate_id(result.ordinal)
            ):
                raise ValueError("grid candidate identity does not match its ordinal")

        ordinals = self._validated_ordinals(
            (result.ordinal for _candidate, result in pairs),
            len(pairs),
        )

        if self._metric_names is None:
            self._metric_names = tuple(sorted({
                name
                for _candidate, result in pairs
                for name in result.metrics
            }))
        metric_names = self._metric_names
        if self._buffer_values is None:
            self._buffer_values = np.empty(
                (len(metric_names), self._shard_rows), dtype=np.float64
            )
        metric_columns = {name: index for index, name in enumerate(metric_names)}
        values = np.full(
            (len(metric_names), len(pairs)), np.nan, dtype=np.float64
        )
        status = np.empty(len(pairs), dtype=np.uint8)
        errors: list[tuple[int, str]] = []
        for row, (_candidate, result) in enumerate(pairs):
            if (
                any(name not in metric_columns for name in result.metrics)
                or (
                    result.status == "ok"
                    and len(result.metrics) != len(metric_names)
                )
            ):
                raise ValueError(
                    "grid result metrics do not match the configured fields"
                )
            for name, value in result.metrics.items():
                values[metric_columns[name], row] = value
            status[row] = _STATUS_TO_CODE[result.status]
            self._status_counts[result.status] += 1
            if result.error is not None:
                errors.append((result.ordinal, result.error))

        self._append_encoded(ordinals, status, values, errors)

    def append_projected(
        self,
        ordinals: Sequence[int],
        batch: "ProjectedBatch",
    ) -> None:
        """Append one fixed-field evaluator batch without materializing maps."""
        if self._closed:
            raise RuntimeError("grid result writer is closed")
        metric_names = self._metric_names
        if metric_names is None or batch.metric_fields != metric_names:
            raise ValueError("projected metrics do not match the configured fields")
        rows = batch.rows
        pairs = tuple(zip(ordinals, rows, strict=True))
        if not pairs:
            return
        for ordinal, row in pairs:
            if (
                row.get("candidate_id") != candidate_id(ordinal)
            ):
                raise ValueError("grid candidate identity does not match its ordinal")

        ordinal_values = self._validated_ordinals(
            (ordinal for ordinal, _row in pairs),
            len(pairs),
        )
        values = np.asarray(
            [row["metrics"] for _ordinal, row in pairs],
            dtype=np.float64,
        )
        if values.shape != (len(pairs), len(metric_names)):
            raise ValueError("projected metric arrays have the wrong shape")
        values = values.T
        try:
            status = np.fromiter(
                (
                    _STATUS_TO_CODE[row.get("status", "ok")]
                    for _ordinal, row in pairs
                ),
                dtype=np.uint8,
                count=len(pairs),
            )
        except KeyError as exc:
            raise ValueError(f"unsupported result status: {exc.args[0]}") from exc
        values[:, status != _STATUS_TO_CODE["ok"]] = np.nan
        errors = [
            (ordinal, str(row["error"]))
            for ordinal, row in pairs
            if row.get("error") is not None
        ]
        for code, count in zip(*np.unique(status, return_counts=True), strict=True):
            self._status_counts[_CODE_TO_STATUS[int(code)]] += int(count)
        self._append_encoded(ordinal_values, status, values, errors)

    def _validated_ordinals(
        self,
        ordinals: Iterable[int],
        count: int,
    ) -> np.ndarray:
        values = np.fromiter(ordinals, dtype=np.int64, count=count)
        if (
            np.any(values < 0)
            or np.any(values >= self._total)
            or len(np.unique(values)) != len(values)
            or np.any(self._written[values])
        ):
            raise ValueError("grid result ordinals must be unique and in range")
        return values

    def _append_encoded(
        self,
        ordinals: np.ndarray,
        status: np.ndarray,
        values: np.ndarray,
        errors: Sequence[tuple[int, str]],
    ) -> None:
        if self._buffer_values is None:
            self._buffer_values = np.empty(
                (values.shape[0], self._shard_rows), dtype=np.float64
            )

        error_by_ordinal = dict(errors)
        offset = 0
        while offset < len(ordinals):
            space = self._shard_rows - self._buffer_count
            take = min(space, len(ordinals) - offset)
            target = slice(self._buffer_count, self._buffer_count + take)
            source = slice(offset, offset + take)
            self._buffer_index["ordinal"][target] = ordinals[source]
            self._buffer_index["status"][target] = status[source]
            assert self._buffer_values is not None
            self._buffer_values[:, target] = values[:, source]
            self._buffer_errors.extend(
                (int(ordinal), error_by_ordinal[int(ordinal)])
                for ordinal in ordinals[source]
                if int(ordinal) in error_by_ordinal
            )
            self._buffer_count += take
            offset += take
            if self._buffer_count == self._shard_rows:
                self._flush()
        self._written[ordinals] = True
        self._count += len(ordinals)

    def _flush(self) -> None:
        if not self._buffer_count:
            return
        shard = self._shard_count
        _write_npy_member(
            self._archive,
            _grid_member("index", shard),
            self._buffer_index[: self._buffer_count],
        )
        assert self._buffer_values is not None
        for column in range(len(self._metric_names or ())):
            _write_npy_member(
                self._archive,
                _grid_metric_member(column, shard),
                self._buffer_values[column, : self._buffer_count],
            )
        if self._buffer_errors:
            width = max(len(error) for _ordinal, error in self._buffer_errors)
            error_values = np.asarray(
                self._buffer_errors,
                dtype=np.dtype(
                    [("ordinal", self._ordinal_dtype), ("error", f"U{width}")]
                ),
            )
            _write_npy_member(
                self._archive, _grid_member("errors", shard), error_values
            )
        self._buffer_count = 0
        self._buffer_errors.clear()
        self._shard_count += 1

    def finalize(self) -> ArtifactPaths:
        if self._closed:
            raise RuntimeError("grid result writer is closed")
        if self._count != self._total or not np.all(self._written):
            raise ValueError("grid results do not cover every candidate")
        self._flush()
        self._archive.close()
        self._closed = True
        with self._temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        run_payload = {
            "schema_version": GRID_SCHEMA_VERSION,
            "run_id": self._run_id,
            "candidate_count": self._total,
            "metric_names": list(self._metric_names or ()),
            "shard_count": self._shard_count,
            "status_counts": self._status_counts,
            "metadata": self._metadata,
        }
        os.replace(self._temporary, self._paths.results_npz)
        try:
            _atomic_bytes(
                self._paths.run_json,
                _canonical_json_bytes(run_payload),
            )
        except BaseException:
            self._paths.results_npz.unlink(missing_ok=True)
            raise
        return self._paths

    def cleanup(self) -> None:
        if not self._closed:
            self._archive.close()
            self._closed = True
        self._temporary.unlink(missing_ok=True)

    def __enter__(self) -> "GridResultWriter":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if exc_type is None:
            if not self._closed:
                self.finalize()
        else:
            self.cleanup()


class ResultWriter:
    """Append bounded batches and finalize without retaining rows in memory."""

    def __init__(
        self,
        directory: str | Path,
        *,
        run_id: str,
        metadata: Mapping[str, Any] | None = None,
    ):
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._run_id = run_id
        self._metadata = dict(metadata or {})
        self._metric_names: set[str] = set()
        self._count = 0
        self._stats = {
            "max_id": 1,
            "max_status": 1,
            "max_error": 1,
            "max_overrides": 2,
            "params": 0,
            "min_params": None,
            "max_params": 0,
        }
        self._closed = False
        self._temporary = Path(tempfile.mkdtemp(prefix=".fxopt-", dir=self.directory))
        self._spool_path = self._temporary / "rows.jsonl"
        self._spool = self._spool_path.open("wb")

    @property
    def retained_rows(self) -> int:
        """Always zero: rows live in the temporary spool, not in Python objects."""
        return 0

    @property
    def row_count(self) -> int:
        return self._count

    def _account(self, row: Mapping[str, Any]) -> None:
        candidate_data = row["candidate"]
        result_data = row["result"]
        self._metric_names.update(result_data.get("metrics", {}))
        self._count += 1
        self._stats["max_id"] = max(
            self._stats["max_id"], len(candidate_data["candidate_id"])
        )
        self._stats["max_status"] = max(
            self._stats["max_status"], len(result_data["status"])
        )
        self._stats["max_error"] = max(
            self._stats["max_error"], len(result_data.get("error") or "")
        )
        overrides = _row_json_bytes(candidate_data.get("pool_overrides", {}))
        self._stats["max_overrides"] = max(
            self._stats["max_overrides"], len(overrides.decode())
        )
        self._stats["params"] += len(candidate_data.get("policy_params", ()))
        param_count = len(candidate_data.get("policy_params", ()))
        current_min = self._stats["min_params"]
        self._stats["min_params"] = (
            param_count if current_min is None else min(current_min, param_count)
        )
        self._stats["max_params"] = max(self._stats["max_params"], param_count)

    def update_metadata(self, **values: Any) -> None:
        """Add small run-level facts before final publication."""
        if self._closed:
            raise RuntimeError("result writer is closed")
        self._metadata.update(values)

    def append(self, candidates: Sequence[Candidate], results: Sequence[CandidateResult]) -> None:
        if self._closed:
            raise RuntimeError("result writer is closed")
        for candidate, result in zip(candidates, results, strict=True):
            if not isinstance(candidate, Candidate) or not isinstance(result, CandidateResult):
                raise TypeError("append expects Candidate and CandidateResult values")
            if candidate.candidate_id != result.candidate_id:
                raise ValueError("candidate and result IDs must match")
            row = {"candidate": candidate.to_dict(ordinal=result.ordinal), "result": result.to_dict()}
            self._spool.write(_row_json_bytes(row) + b"\n")
            self._account(row)
        self._spool.flush()

    def _rows(self):
        if not self._spool.closed:
            self._spool.flush()
        with self._spool_path.open("rb") as stream:
            for line in stream:
                yield orjson.loads(line)

    def finalize(self) -> ArtifactPaths:
        if self._closed:
            raise RuntimeError("result writer is closed")
        self._spool.close()
        self._closed = True
        try:
            metric_names = tuple(sorted(self._metric_names))
            stats = self._stats
            if stats["min_params"] != stats["max_params"]:
                raise ValueError("streamed candidates must have a fixed policy width")
            param_width = int(stats["max_params"])
            root = self.directory
            staging = Path(tempfile.mkdtemp(prefix=".fxopt-final-", dir=root))
            try:
                arrays = {
                    "candidate_ids": np.lib.format.open_memmap(staging / "ids.npy", mode="w+", dtype=f"U{stats['max_id']}", shape=(self._count,)),
                    "candidate_policy_offsets": np.lib.format.open_memmap(staging / "offsets.npy", mode="w+", dtype=np.int64, shape=(self._count + 1,)),
                    "candidate_policy_values": np.lib.format.open_memmap(staging / "params.npy", mode="w+", dtype=np.float64, shape=(stats["params"],)),
                    "candidate_pool_overrides": np.lib.format.open_memmap(staging / "overrides.npy", mode="w+", dtype=f"U{stats['max_overrides']}", shape=(self._count,)),
                    "ordinals": np.lib.format.open_memmap(staging / "ordinals.npy", mode="w+", dtype=np.int64, shape=(self._count,)),
                    "statuses": np.lib.format.open_memmap(staging / "statuses.npy", mode="w+", dtype=f"U{stats['max_status']}", shape=(self._count,)),
                    "errors": np.lib.format.open_memmap(staging / "errors.npy", mode="w+", dtype=f"U{stats['max_error']}", shape=(self._count,)),
                    "error_present": np.lib.format.open_memmap(staging / "error_present.npy", mode="w+", dtype=np.bool_, shape=(self._count,)),
                    **{
                        _metric_key(column): np.lib.format.open_memmap(
                            staging / f"metric-{column}.npy",
                            mode="w+",
                            dtype=np.float64,
                            shape=(self._count,),
                        )
                        for column, _name in enumerate(metric_names)
                    },
                }
                written = np.lib.format.open_memmap(
                    staging / "written.npy",
                    mode="w+",
                    dtype=np.bool_,
                    shape=(self._count,),
                )
                written.fill(False)
                offsets = arrays["candidate_policy_offsets"]
                offsets[:] = np.arange(self._count + 1, dtype=np.int64) * param_width
                columns = {name: index for index, name in enumerate(metric_names)}
                for column in columns.values():
                    arrays[_metric_key(column)].fill(np.nan)
                for row in self._rows():
                    candidate, result = row["candidate"], row["result"]
                    index = int(result["ordinal"])
                    if index < 0 or index >= self._count or written[index]:
                        raise ValueError("streamed result ordinals must form one grid permutation")
                    written[index] = True
                    arrays["candidate_ids"][index] = candidate["candidate_id"]
                    params = candidate.get("policy_params", ())
                    start = index * param_width
                    arrays["candidate_policy_values"][start : start + param_width] = params
                    arrays["candidate_pool_overrides"][index] = _canonical_json_bytes(candidate.get("pool_overrides", {})).decode()
                    arrays["ordinals"][index], arrays["statuses"][index] = index, result["status"]
                    for name, value in result.get("metrics", {}).items():
                        arrays[_metric_key(columns[name])][index] = value
                    arrays["errors"][index], arrays["error_present"][index] = result.get("error") or "", result.get("error") is not None
                if not np.all(written):
                    raise ValueError("streamed result ordinals do not cover the grid")
                for array in arrays.values():
                    array.flush()
                written.flush()
                run_payload = {"schema_version": SCHEMA_VERSION, "run_id": self._run_id, "candidate_count": self._count, "metric_names": list(metric_names), "metadata": self._metadata}
                _atomic_npz(root / RESULTS_FILENAME, arrays)
                _atomic_bytes(root / RUN_FILENAME, _canonical_json_bytes(run_payload))
                return ArtifactPaths(root / RUN_FILENAME, root / RESULTS_FILENAME)
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        if not self._spool.closed:
            self._spool.close()
        shutil.rmtree(self._temporary, ignore_errors=True)

    def __enter__(self) -> "ResultWriter":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if exc_type is None:
            if not self._closed:
                self.finalize()
        else:
            self.cleanup()
