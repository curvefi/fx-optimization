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


SCHEMA_VERSION = "fxopt.results.v1"
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


def write_results(bundle: ResultBundle, directory: str | Path) -> ArtifactPaths:
    """Write exactly ``run.json`` and ``results.npz`` for one completed run."""
    if not isinstance(bundle, ResultBundle):
        raise TypeError("bundle must be a ResultBundle")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    metric_names = sorted({name for result in bundle.results for name in result.metrics})
    metric_values = np.zeros((len(bundle.results), len(metric_names)), dtype=np.float64)
    metric_present = np.zeros(metric_values.shape, dtype=np.bool_)
    for row, result in enumerate(bundle.results):
        for column, name in enumerate(metric_names):
            if name in result.metrics:
                metric_values[row, column] = result.metrics[name]
                metric_present[row, column] = True

    run_payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": bundle.run_id,
        "metric_names": metric_names,
        "candidates": [
            {
                "candidate_id": candidate.candidate_id,
                "policy_params": list(candidate.policy_params),
                "pool_overrides": dict(candidate.pool_overrides),
            }
            for candidate in bundle.candidates
        ],
        "metadata": dict(bundle.metadata),
    }
    run_path = root / RUN_FILENAME
    npz_path = root / RESULTS_FILENAME
    _atomic_bytes(run_path, _canonical_json_bytes(run_payload))
    _atomic_npz(
        npz_path,
        {
            "candidate_ids": np.asarray([r.candidate_id for r in bundle.results], dtype=str),
            "ordinals": np.asarray([r.ordinal for r in bundle.results], dtype=np.int64),
            "statuses": np.asarray([r.status for r in bundle.results], dtype=str),
            "metrics": metric_values,
            "metric_present": metric_present,
            "errors": np.asarray([r.error or "" for r in bundle.results], dtype=str),
            "error_present": np.asarray([r.error is not None for r in bundle.results], dtype=np.bool_),
            "economic_fingerprints": np.asarray(
                [r.economic_fingerprint or "" for r in bundle.results], dtype=str
            ),
            "fingerprint_present": np.asarray(
                [r.economic_fingerprint is not None for r in bundle.results], dtype=np.bool_
            ),
        },
    )
    return ArtifactPaths(run_json=run_path, results_npz=npz_path)


def read_results(directory: str | Path) -> ResultBundle:
    """Read and validate the canonical two-file result artifact."""
    root = Path(directory)
    run_path, npz_path = root / RUN_FILENAME, root / RESULTS_FILENAME
    try:
        payload = json.loads(run_path.read_bytes())
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported result artifact schema")
        metric_names = tuple(payload["metric_names"])
        metadata = payload.get("metadata", {})
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {RUN_FILENAME}") from exc

    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid {RESULTS_FILENAME}") from exc

    ids = [str(value) for value in arrays["candidate_ids"].tolist()]
    if "candidates" in payload:
        candidates = tuple(
            Candidate(
                candidate_id=item["candidate_id"],
                policy_params=tuple(item.get("policy_params", ())),
                pool_overrides=item.get("pool_overrides", {}),
            )
            for item in payload["candidates"]
        )
        if ids != [candidate.candidate_id for candidate in candidates]:
            raise ValueError("result candidate IDs do not match run.json")
    else:
        count = int(payload["candidate_count"])
        offsets = arrays["candidate_policy_offsets"]
        if offsets.shape != (count + 1,):
            raise ValueError("candidate policy offsets have the wrong shape")
        candidates = tuple(
            Candidate(
                candidate_id=ids[row],
                policy_params=arrays["candidate_policy_values"][offsets[row] : offsets[row + 1]].tolist(),
                pool_overrides=json.loads(str(arrays["candidate_pool_overrides"][row])),
            )
            for row in range(count)
        )
    count = len(candidates)
    if arrays["metrics"].shape != (count, len(metric_names)):
        raise ValueError("result metric matrix has the wrong shape")
    if arrays["metric_present"].shape != arrays["metrics"].shape:
        raise ValueError("result metric presence matrix has the wrong shape")
    if any(
        array.shape[0] != count
        for name, array in arrays.items()
        if name not in {"metrics", "metric_present", "candidate_policy_values", "candidate_policy_offsets"}
    ):
        raise ValueError("result arrays have inconsistent lengths")
    results: list[CandidateResult] = []
    for row in range(count):
        metrics = {
            str(name): float(arrays["metrics"][row, column])
            for column, name in enumerate(metric_names)
            if bool(arrays["metric_present"][row, column])
        }
        error = str(arrays["errors"][row]) if bool(arrays["error_present"][row]) else None
        fingerprint = (
            str(arrays["economic_fingerprints"][row])
            if bool(arrays["fingerprint_present"][row])
            else None
        )
        results.append(
            CandidateResult(
                candidate_id=ids[row],
                status=str(arrays["statuses"][row]),
                metrics=metrics,
                error=error,
                economic_fingerprint=fingerprint,
                ordinal=int(arrays["ordinals"][row]),
            )
        )
    return ResultBundle(
        run_id=payload["run_id"], candidates=candidates, results=tuple(results), metadata=metadata
    )


class ResultWriter:
    """Append bounded batches and finalize without retaining rows in memory."""

    def __init__(self, directory: str | Path, *, run_id: str, metadata: Mapping[str, Any] | None = None):
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
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
            "max_fp": 1,
            "max_overrides": 2,
            "params": 0,
        }
        self._closed = False

    @property
    def retained_rows(self) -> int:
        """Always zero: rows live in the temporary spool, not in Python objects."""
        return 0

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
            self._stats["max_fp"] = max(self._stats["max_fp"], len(result_data.get("economic_fingerprint") or ""))
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
                shape = (self._count, len(metric_names))
                arrays = {
                    "candidate_ids": np.lib.format.open_memmap(staging / "ids.npy", mode="w+", dtype=f"U{stats['max_id']}", shape=(self._count,)),
                    "candidate_policy_offsets": np.lib.format.open_memmap(staging / "offsets.npy", mode="w+", dtype=np.int64, shape=(self._count + 1,)),
                    "candidate_policy_values": np.lib.format.open_memmap(staging / "params.npy", mode="w+", dtype=np.float64, shape=(stats["params"],)),
                    "candidate_pool_overrides": np.lib.format.open_memmap(staging / "overrides.npy", mode="w+", dtype=f"U{stats['max_overrides']}", shape=(self._count,)),
                    "ordinals": np.lib.format.open_memmap(staging / "ordinals.npy", mode="w+", dtype=np.int64, shape=(self._count,)),
                    "statuses": np.lib.format.open_memmap(staging / "statuses.npy", mode="w+", dtype=f"U{stats['max_status']}", shape=(self._count,)),
                    "metrics": np.lib.format.open_memmap(staging / "metrics.npy", mode="w+", dtype=np.float64, shape=shape),
                    "metric_present": np.lib.format.open_memmap(staging / "present.npy", mode="w+", dtype=np.bool_, shape=shape),
                    "errors": np.lib.format.open_memmap(staging / "errors.npy", mode="w+", dtype=f"U{stats['max_error']}", shape=(self._count,)),
                    "error_present": np.lib.format.open_memmap(staging / "error_present.npy", mode="w+", dtype=np.bool_, shape=(self._count,)),
                    "economic_fingerprints": np.lib.format.open_memmap(staging / "fingerprints.npy", mode="w+", dtype=f"U{stats['max_fp']}", shape=(self._count,)),
                    "fingerprint_present": np.lib.format.open_memmap(staging / "fp_present.npy", mode="w+", dtype=np.bool_, shape=(self._count,)),
                }
                offsets, param_at = arrays["candidate_policy_offsets"], 0
                columns = {name: index for index, name in enumerate(metric_names)}
                for index, row in enumerate(self._rows()):
                    candidate, result = row["candidate"], row["result"]
                    arrays["candidate_ids"][index] = candidate["candidate_id"]
                    params = candidate.get("policy_params", ())
                    arrays["candidate_policy_values"][param_at : param_at + len(params)] = params
                    param_at += len(params)
                    offsets[index + 1] = param_at
                    arrays["candidate_pool_overrides"][index] = _canonical_json_bytes(candidate.get("pool_overrides", {})).decode()
                    arrays["ordinals"][index], arrays["statuses"][index] = result["ordinal"], result["status"]
                    arrays["metrics"][index] = 0.0
                    arrays["metric_present"][index] = False
                    for name, value in result.get("metrics", {}).items():
                        arrays["metrics"][index, columns[name]], arrays["metric_present"][index, columns[name]] = value, True
                    arrays["errors"][index], arrays["error_present"][index] = result.get("error") or "", result.get("error") is not None
                    arrays["economic_fingerprints"][index], arrays["fingerprint_present"][index] = result.get("economic_fingerprint") or "", result.get("economic_fingerprint") is not None
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
