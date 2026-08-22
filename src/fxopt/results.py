"""Canonical two-file result artifacts for optimizer runs."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .contract import Candidate, CandidateResult


SCHEMA_VERSION = "fxopt.results.v2"
RUN_FILENAME = "run.json"
RESULTS_FILENAME = "results.npz"


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
    candidate_ids: np.ndarray
    ordinals: np.ndarray
    statuses: np.ndarray
    metrics: Mapping[str, np.ndarray]
    errors: np.ndarray
    error_present: np.ndarray

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

    def candidate_at(self, ordinal: int) -> Candidate:
        """Construct only the candidate selected by its stored result ordinal."""
        return self._candidate_at_row(self.row_for_ordinal(ordinal))

    def _candidate_at_row(self, row: int) -> Candidate:
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
        return Candidate(
            candidate_id=str(self.candidate_ids[row]),
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


def _require_unused_artifacts(root: Path) -> ArtifactPaths:
    paths = _artifact_paths(root)
    existing = [path.name for path in (paths.run_json, paths.results_npz) if path.exists()]
    if existing:
        raise FileExistsError(f"result artifacts already exist: {', '.join(existing)}")
    return paths


def _metric_key(column: int) -> str:
    return f"metric_{column:04d}"


def write_results(bundle: ResultBundle, directory: str | Path) -> ArtifactPaths:
    """Write exactly ``run.json`` and ``results.npz`` for one completed run."""
    if not isinstance(bundle, ResultBundle):
        raise TypeError("bundle must be a ResultBundle")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    paths = _require_unused_artifacts(root)
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
        if payload.get("schema_version") != SCHEMA_VERSION:
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
        error = str(columns.errors[row]) if bool(columns.error_present[row]) else None
        results.append(
            CandidateResult(
                candidate_id=str(columns.candidate_ids[row]),
                status=str(columns.statuses[row]),
                metrics=metrics,
                error=error,
                ordinal=int(columns.ordinals[row]),
            )
        )
    return ResultBundle(
        run_id=columns.run_id,
        candidates=candidates,
        results=tuple(results),
        metadata=columns.metadata,
    )


class ResultWriter:
    """Append bounded batches and finalize without retaining rows in memory."""

    def __init__(self, directory: str | Path, *, run_id: str, metadata: Mapping[str, Any] | None = None):
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        _require_unused_artifacts(self.directory)
        self._temporary = Path(tempfile.mkdtemp(prefix=".fxopt-", dir=self.directory))
        self._spool_path = self._temporary / "rows.jsonl"
        self._spool = self._spool_path.open("wb")
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
        }
        self._closed = False

    @property
    def retained_rows(self) -> int:
        """Always zero: rows live in the temporary spool, not in Python objects."""
        return 0

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
            self._spool.write(_canonical_json_bytes(row) + b"\n")
            self._metric_names.update(result.metrics)
            self._count += 1
            candidate_data, result_data = row["candidate"], row["result"]
            self._stats["max_id"] = max(self._stats["max_id"], len(candidate_data["candidate_id"]))
            self._stats["max_status"] = max(self._stats["max_status"], len(result_data["status"]))
            self._stats["max_error"] = max(self._stats["max_error"], len(result_data.get("error") or ""))
            overrides = _canonical_json_bytes(candidate_data.get("pool_overrides", {})).decode()
            self._stats["max_overrides"] = max(self._stats["max_overrides"], len(overrides))
            self._stats["params"] += len(candidate_data.get("policy_params", ()))
        self._spool.flush()

    def _rows(self):
        if not self._spool.closed:
            self._spool.flush()
        with self._spool_path.open("rb") as stream:
            for line in stream:
                yield json.loads(line)

    def finalize(self) -> ArtifactPaths:
        if self._closed:
            raise RuntimeError("result writer is closed")
        self._spool.close()
        self._closed = True
        try:
            metric_names = tuple(sorted(self._metric_names))
            stats = self._stats
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
                offsets, param_at = arrays["candidate_policy_offsets"], 0
                columns = {name: index for index, name in enumerate(metric_names)}
                for column in columns.values():
                    arrays[_metric_key(column)].fill(np.nan)
                for index, row in enumerate(self._rows()):
                    candidate, result = row["candidate"], row["result"]
                    arrays["candidate_ids"][index] = candidate["candidate_id"]
                    params = candidate.get("policy_params", ())
                    arrays["candidate_policy_values"][param_at : param_at + len(params)] = params
                    param_at += len(params)
                    offsets[index + 1] = param_at
                    arrays["candidate_pool_overrides"][index] = _canonical_json_bytes(candidate.get("pool_overrides", {})).decode()
                    arrays["ordinals"][index], arrays["statuses"][index] = result["ordinal"], result["status"]
                    for name, value in result.get("metrics", {}).items():
                        arrays[_metric_key(columns[name])][index] = value
                    arrays["errors"][index], arrays["error_present"][index] = result.get("error") or "", result.get("error") is not None
                arrays["candidate_policy_offsets"][0] = 0
                for array in arrays.values():
                    array.flush()
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
        if exc_type is None and not self._closed:
            self.finalize()
        else:
            self.cleanup()
