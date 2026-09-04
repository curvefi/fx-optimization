"""Canonical two-file artifacts for exhaustive Cartesian grids."""

from __future__ import annotations

import json
import os
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
from .contract import Candidate
from .grid import CartesianGrid

if TYPE_CHECKING:
    from .engine import ProjectedBatch


SCHEMA_VERSION = "fxopt.grid-results.v1"
RUN_FILENAME = "run.json"
RESULTS_FILENAME = "results.npz"

_STATUS_TO_CODE = {"ok": 0, "failed": 1, "cancelled": 2, "uncalculated": 3}
_CODE_TO_STATUS = tuple(_STATUS_TO_CODE)


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
    ordinals: np.ndarray
    metrics: Mapping[str, np.ndarray]
    status_codes: np.ndarray
    failures: Mapping[int, str] = MappingProxyType({})
    grid: CartesianGrid | None = None

    @property
    def row_count(self) -> int:
        return int(self.ordinals.shape[0])

    def row_for_ordinal(self, ordinal: int) -> int:
        if not isinstance(ordinal, int) or ordinal < 0:
            raise ValueError("ordinal must be a non-negative integer")
        if ordinal >= self.row_count:
            raise ValueError(f"result artifact has no row for ordinal {ordinal}")
        return ordinal

    @property
    def ok_mask(self) -> np.ndarray:
        return self.status_codes == _STATUS_TO_CODE["ok"]

    def status_at(self, row: int) -> str:
        code = int(self.status_codes[row])
        if code < 0 or code >= len(_CODE_TO_STATUS):
            raise ValueError(f"invalid stored status code: {code}")
        return _CODE_TO_STATUS[code]

    def error_at(self, row: int) -> str | None:
        return self.failures.get(int(self.ordinals[row]))

    def candidate_at(self, ordinal: int) -> Candidate:
        """Construct only the candidate selected by its stored result ordinal."""
        row = self.row_for_ordinal(ordinal)
        if self.status_at(row) == "uncalculated":
            raise ValueError(f"ordinal {ordinal} was not calculated")
        return self._candidate_at_row(row)

    def _candidate_at_row(self, row: int) -> Candidate:
        assert self.grid is not None
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


def _artifact_paths(root: Path) -> ArtifactPaths:
    return ArtifactPaths(root / RUN_FILENAME, root / RESULTS_FILENAME)


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


def read_result_columns(
    directory: str | Path,
    *,
    metrics: Sequence[str] | None = None,
) -> ResultColumns:
    """Load the grid index and only the requested metric columns."""
    root = Path(directory)
    try:
        payload = json.loads((root / RUN_FILENAME).read_bytes())
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported result artifact schema")
        metric_names = tuple(payload["metric_names"])
        if len(set(metric_names)) != len(metric_names) or not all(
            isinstance(name, str) and name for name in metric_names
        ):
            raise ValueError("invalid metric names")
        count = payload["candidate_count"]
        shard_count = payload["shard_count"]
        run_id = payload["run_id"]
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
            or isinstance(shard_count, bool)
            or not isinstance(shard_count, int)
            or shard_count < 0
            or not isinstance(run_id, str)
            or not run_id
        ):
            raise ValueError("invalid grid result identity")
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

    selected_metrics = {
        name: np.full(count, np.nan, dtype=np.float64) for name in selected
    }
    status_codes = np.full(count, _STATUS_TO_CODE["uncalculated"], dtype=np.uint8)
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
                    raise ValueError(f"grid result is missing shard {shard}")
                index = _read_npy_member(archive, index_name)
                if index.dtype.names != ("ordinal", "status") or index.ndim != 1:
                    raise ValueError(f"grid shard {shard} has invalid arrays")
                ordinals = np.asarray(index["ordinal"], dtype=np.int64)
                if (
                    np.any(ordinals < 0)
                    or np.any(ordinals >= count)
                    or len(np.unique(ordinals)) != len(ordinals)
                    or np.any(written[ordinals])
                ):
                    raise ValueError(f"grid shard {shard} has invalid ordinals")
                written[ordinals] = True
                status_codes[ordinals] = index["status"]
                for name in selected:
                    values = _read_npy_member(
                        archive, metric_members[columns[name]]
                    )
                    if values.shape != (len(index),):
                        raise ValueError(f"grid shard {shard} has invalid arrays")
                    selected_metrics[name][ordinals] = values
                errors_name = _grid_member("errors", shard)
                if errors_name in members:
                    errors = _read_npy_member(archive, errors_name)
                    if errors.dtype.names != ("ordinal", "error") or errors.ndim != 1:
                        raise ValueError(f"grid shard {shard} has invalid errors")
                    for item in errors:
                        ordinal = int(item["ordinal"])
                        if ordinal < 0 or ordinal >= count or not written[ordinal]:
                            raise ValueError(f"grid shard {shard} has invalid errors")
                        failures.setdefault(ordinal, str(item["error"]))
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        raise ValueError(f"invalid {RESULTS_FILENAME}") from exc
    if np.any(status_codes >= len(_CODE_TO_STATUS)):
        raise ValueError("grid result contains an invalid status")
    return ResultColumns(
        root=root,
        run_id=run_id,
        metadata=dict(metadata),
        available_metrics=metric_names,
        ordinals=np.arange(count, dtype=np.int64),
        metrics=MappingProxyType(selected_metrics),
        status_codes=status_codes,
        failures=MappingProxyType(failures),
        grid=_grid_from_metadata(metadata, count),
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
        metric_names: Sequence[str],
        shard_rows: int = 65_536,
        expected_count: int | None = None,
    ) -> None:
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        if isinstance(total, bool) or not isinstance(total, int) or total < 1:
            raise ValueError("grid result total must be a positive integer")
        names = tuple(sorted(metric_names))
        if (
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
        if expected_count is not None and (
            isinstance(expected_count, bool)
            or not isinstance(expected_count, int)
            or expected_count < 1
            or expected_count > total
        ):
            raise ValueError("expected count must be in [1, total]")

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
        self._expected_count = total if expected_count is None else expected_count
        self._partition = expected_count is not None
        self._written = np.zeros(total, dtype=np.bool_)
        self._ordinal_dtype = np.dtype("<u4" if total <= np.iinfo(np.uint32).max else "<u8")
        self._index_dtype = np.dtype(
            [("ordinal", self._ordinal_dtype), ("status", "u1")]
        )
        self._buffer_index = np.empty(shard_rows, dtype=self._index_dtype)
        self._buffer_values = np.empty(
            (len(names), shard_rows), dtype=np.float64
        )
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

    def append_projected(
        self,
        ordinals: Sequence[int],
        batch: "ProjectedBatch",
    ) -> None:
        """Append one fixed-field evaluator batch without materializing maps."""
        if self._closed:
            raise RuntimeError("grid result writer is closed")
        if batch.metric_fields != self._metric_names:
            raise ValueError("projected metrics do not match the configured fields")
        metric_names = self._metric_names
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
        error_by_ordinal = dict(errors)
        offset = 0
        while offset < len(ordinals):
            space = self._shard_rows - self._buffer_count
            take = min(space, len(ordinals) - offset)
            target = slice(self._buffer_count, self._buffer_count + take)
            source = slice(offset, offset + take)
            self._buffer_index["ordinal"][target] = ordinals[source]
            self._buffer_index["status"][target] = status[source]
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
        for column in range(len(self._metric_names)):
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
        if self._count != self._expected_count or (
            not self._partition and not np.all(self._written)
        ):
            raise ValueError("grid results do not cover the expected candidates")
        self._flush()
        self._archive.close()
        self._closed = True
        with self._temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        run_payload = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self._run_id,
            "candidate_count": self._total,
            "metric_names": list(self._metric_names),
            "shard_count": self._shard_count,
            "status_counts": self._status_counts,
            "metadata": self._metadata,
        }
        if self._partition:
            run_payload["partition_count"] = self._expected_count
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

    def finalize_partition(self) -> ArtifactPaths:
        """Publish a dynamically sized partial grid for coordinator merging."""
        if self._closed:
            raise RuntimeError("grid result writer is closed")
        if self._count < 1:
            raise ValueError("grid partition must contain at least one candidate")
        self._expected_count = self._count
        self._partition = True
        return self.finalize()

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


def merge_grid_partitions(
    directory: str | Path,
    partitions: Sequence[str | Path],
    *,
    run_id: str,
    total: int,
    metadata: Mapping[str, Any],
    metric_names: Sequence[str],
) -> ArtifactPaths:
    """Validate disjoint worker archives and publish one canonical grid result."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(root)
    if paths.run_json.exists() or paths.results_npz.exists():
        raise FileExistsError(f"completed result already exists in {root}")
    names = tuple(sorted(metric_names))
    if not partitions or not names or total < 1:
        raise ValueError("partition merge requires inputs, metrics, and a positive total")

    temporary = root / f".{RESULTS_FILENAME}.tmp"
    temporary.unlink(missing_ok=True)
    output = zipfile.ZipFile(
        temporary,
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    )
    written = np.zeros(total, dtype=np.bool_)
    status_counts = {name: 0 for name in _STATUS_TO_CODE}
    output_shard = 0
    try:
        for partition_path in partitions:
            partition = Path(partition_path)
            try:
                payload = json.loads((partition / RUN_FILENAME).read_bytes())
                if (
                    payload.get("schema_version") != SCHEMA_VERSION
                    or payload.get("run_id") != run_id
                    or payload.get("candidate_count") != total
                    or tuple(payload.get("metric_names", ())) != names
                ):
                    raise ValueError("partition identity does not match the run")
                expected = payload["partition_count"]
                shard_count = payload["shard_count"]
                if (
                    isinstance(expected, bool)
                    or not isinstance(expected, int)
                    or expected < 1
                    or isinstance(shard_count, bool)
                    or not isinstance(shard_count, int)
                    or shard_count < 0
                ):
                    raise ValueError("partition counts are invalid")
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid grid partition: {partition}") from exc

            partition_count = 0
            with zipfile.ZipFile(partition / RESULTS_FILENAME, "r") as source:
                members = set(source.namelist())
                for source_shard in range(shard_count):
                    index_name = _grid_member("index", source_shard)
                    metric_members = tuple(
                        _grid_metric_member(column, source_shard)
                        for column in range(len(names))
                    )
                    if index_name not in members or any(
                        member not in members for member in metric_members
                    ):
                        raise ValueError(f"grid partition is missing shard {source_shard}")
                    index = _read_npy_member(source, index_name)
                    if index.dtype.names != ("ordinal", "status") or index.ndim != 1:
                        raise ValueError(f"grid partition shard {source_shard} is invalid")
                    ordinals = np.asarray(index["ordinal"], dtype=np.int64)
                    statuses = np.asarray(index["status"], dtype=np.uint8)
                    if (
                        np.any(ordinals < 0)
                        or np.any(ordinals >= total)
                        or len(np.unique(ordinals)) != len(ordinals)
                        or np.any(written[ordinals])
                        or np.any(statuses >= len(_CODE_TO_STATUS))
                    ):
                        raise ValueError(f"grid partition shard {source_shard} overlaps")
                    written[ordinals] = True
                    partition_count += len(index)
                    for code, count in zip(
                        *np.unique(statuses, return_counts=True), strict=True
                    ):
                        status_counts[_CODE_TO_STATUS[int(code)]] += int(count)

                    _write_npy_member(
                        output, _grid_member("index", output_shard), index
                    )
                    for column, member in enumerate(metric_members):
                        with source.open(member) as incoming, output.open(
                            _grid_metric_member(column, output_shard),
                            mode="w",
                            force_zip64=True,
                        ) as outgoing:
                            shutil.copyfileobj(incoming, outgoing)
                    errors_name = _grid_member("errors", source_shard)
                    if errors_name in members:
                        with source.open(errors_name) as incoming, output.open(
                            _grid_member("errors", output_shard),
                            mode="w",
                            force_zip64=True,
                        ) as outgoing:
                            shutil.copyfileobj(incoming, outgoing)
                    output_shard += 1
            if partition_count != expected:
                raise ValueError("grid partition row count does not match its receipt")

        status_counts["uncalculated"] = int(total - np.count_nonzero(written))
        output.close()
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, paths.results_npz)
        try:
            _atomic_bytes(paths.run_json, _canonical_json_bytes({
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "candidate_count": total,
                "metric_names": list(names),
                "shard_count": output_shard,
                "status_counts": status_counts,
                "metadata": dict(metadata),
            }))
        except BaseException:
            paths.results_npz.unlink(missing_ok=True)
            raise
        return paths
    except BaseException:
        output.close()
        temporary.unlink(missing_ok=True)
        raise
