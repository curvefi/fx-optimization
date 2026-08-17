"""Strict shard fetch, attestation, validation, and deterministic collection."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence
import tempfile

from .adapter import ProcessAdapter, SSHProcessAdapter
from .site import SSHConfig
import numpy as np
from .staging import scoped_remote_path, sha256_path, validate_run_id


class CollectionError(RuntimeError):
    """Raised when shard collection or validation detects invalid shard data."""


GRID_RESULTS_SCHEMA_VERSION = "fxsim_grid_results_npz_v1"
SHARD_RESULT_SCHEMA_VERSION = "fxsim_grid_shard_v2"
GRID_REQUEST_SCHEMA_VERSION = "fxsim_grid_request_v1"
SESSION_ATTESTATION_FIELDS = (
    "scenario_set_sha256",
    "session_fingerprint",
    "session_config_sha256",
    "metric_schema_sha256",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SHARD_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "shard_id",
        "shard_index",
        "ranges",
        "row_count",
        "rows_sha256",
        "request_set_sha256",
        "session_attestation",
        "rows",
    }
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if item.get("pool_index") is None:
            if item.get("ordinal") is None:
                raise CollectionError("shard result row lacks pool_index and ordinal")
            item["pool_index"] = int(item["ordinal"])
        normalized.append(item)
    return normalized


def _rows_checksum(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(_canonical_json(list(rows))).hexdigest()


def normalize_session_attestation(
    value: Any,
    *,
    expected_session_id: str | None = None,
) -> dict[str, str]:
    """Return the exact evaluator session proof persisted with every shard."""
    if hasattr(value, "model_dump") and callable(value.model_dump):
        value = value.model_dump()
    if not isinstance(value, Mapping):
        raise CollectionError("evaluator session attestation must be an object")
    session_id = value.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise CollectionError("evaluator session attestation has no session_id")
    if expected_session_id is not None and session_id != expected_session_id:
        raise CollectionError(
            f"evaluator session attestation has session_id {session_id!r}, expected {expected_session_id!r}"
        )
    result = {"session_id": session_id}
    for field in SESSION_ATTESTATION_FIELDS:
        digest = value.get(field)
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise CollectionError(
                f"evaluator session attestation {field} must be a lowercase SHA-256 digest"
            )
        result[field] = digest
    return result


def _session_digest_view(value: Any) -> dict[str, str]:
    attestation = normalize_session_attestation(value)
    return {field: attestation[field] for field in SESSION_ATTESTATION_FIELDS}


def grid_request_set_sha256(
    manifest: Mapping[str, Any],
    session_attestation: Any,
) -> str:
    """Bind a shard to its exact candidates, economics, core, and opened session."""
    grid = manifest.get("grid")
    resolved = manifest.get("resolved_spec")
    core = manifest.get("core")
    if not isinstance(grid, Mapping) or not isinstance(grid.get("pools"), list):
        raise CollectionError("cannot attest a manifest without grid.pools")
    if not isinstance(resolved, Mapping):
        raise CollectionError("cannot attest a manifest without resolved_spec")
    if not isinstance(core, Mapping):
        raise CollectionError("cannot attest a manifest without core identity")
    request = {
        "schema_version": GRID_REQUEST_SCHEMA_VERSION,
        "run_id": manifest.get("run_id"),
        "grid_id": grid.get("grid_id"),
        "pools": grid["pools"],
        "scenario": resolved.get("scenario"),
        "policy": resolved.get("policy"),
        "metric_projection": resolved.get("metric_projection"),
        "core": core,
        "session_attestation": _session_digest_view(session_attestation),
    }
    return hashlib.sha256(_canonical_json(request)).hexdigest()


def make_shard_result(
    *,
    run_id: str,
    shard_id: str,
    shard_index: int,
    ranges: Sequence[tuple[int, int]],
    rows: Sequence[Mapping[str, Any]],
    request_set_sha256: str,
    session_attestation: Any,
) -> dict[str, Any]:
    """Build the attested, canonical shard result envelope."""
    if not isinstance(request_set_sha256, str) or not _SHA256_RE.fullmatch(request_set_sha256):
        raise CollectionError("request_set_sha256 must be a lowercase SHA-256 digest")
    normalized_attestation = normalize_session_attestation(session_attestation)
    normalized_rows = _canonical_rows(rows)
    return {
        "schema_version": SHARD_RESULT_SCHEMA_VERSION,
        "run_id": run_id,
        "shard_id": shard_id,
        "shard_index": int(shard_index),
        "ranges": [list(r) for r in ranges],
        "row_count": len(normalized_rows),
        "rows_sha256": _rows_checksum(normalized_rows),
        "request_set_sha256": request_set_sha256,
        "session_attestation": normalized_attestation,
        "rows": normalized_rows,
    }


def write_shard_result(
    path: Path,
    *,
    run_id: str,
    shard_id: str,
    shard_index: int,
    ranges: Sequence[tuple[int, int]],
    rows: Sequence[Mapping[str, Any]],
    request_set_sha256: str,
    session_attestation: Any,
) -> str:
    """Atomically write one canonical shard envelope and return its file digest."""
    payload = make_shard_result(
        run_id=run_id,
        shard_id=shard_id,
        shard_index=shard_index,
        ranges=ranges,
        rows=rows,
        request_set_sha256=request_set_sha256,
        session_attestation=session_attestation,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(_canonical_json(payload) + b"\n")
    os.replace(temporary, path)
    return sha256_path(path)


def _ranges(descriptor: Mapping[str, Any]) -> list[tuple[int, int]]:
    val = descriptor.get("ranges")
    if not isinstance(val, (list, tuple)):
        raise CollectionError(f"shard descriptor has no valid ranges: {descriptor.get('shard_id', '<unknown>')}")
    try:
        ranges = [(int(r[0]), int(r[1])) for r in val]
    except (TypeError, ValueError, IndexError) as exc:
        raise CollectionError(
            f"shard descriptor has no valid ranges: {descriptor.get('shard_id', '<unknown>')}"
        ) from exc
    if any(start < 0 or start >= end for start, end in ranges):
        raise CollectionError(f"shard descriptor has invalid ranges: {descriptor.get('shard_id', '<unknown>')}")
    return ranges


def _load_shard_records(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not path.is_file():
        raise CollectionError(f"missing shard result file: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise CollectionError(f"invalid shard result JSON in {path}: {exc}") from exc

    if not isinstance(payload, Mapping):
        raise CollectionError(
            f"invalid shard result schema in {path}: expected an object with schema_version and rows"
        )
    rows = payload.get("rows")
    if (
        payload.get("schema_version") != SHARD_RESULT_SCHEMA_VERSION
        or set(payload) != _SHARD_RESULT_FIELDS
        or not isinstance(rows, list)
    ):
        raise CollectionError(
            f"invalid shard result schema in {path}: expected {SHARD_RESULT_SCHEMA_VERSION!r}"
        )
    if not all(isinstance(row, Mapping) for row in rows):
        raise CollectionError(f"invalid shard result rows in {path}: every row must be an object")
    return dict(payload), [dict(row) for row in rows]


def _record_index(record: Mapping[str, Any]) -> int:
    if "pool_index" not in record or record["pool_index"] is None:
        raise CollectionError(f"record lacks canonical pool_index field: {record}")
    try:
        return int(record["pool_index"])
    except (TypeError, ValueError) as exc:
        raise CollectionError(f"record has invalid pool_index: {record}") from exc


def _descriptor_checksum_error(
    descriptor: Mapping[str, Any],
    path: Path,
    payload: Mapping[str, Any],
) -> str | None:
    rows = payload.get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        return f"shard result {path} has invalid rows"
    actual_rows_checksum = _rows_checksum([dict(row) for row in rows])
    if payload.get("rows_sha256") != actual_rows_checksum:
        return f"shard result {path} has invalid rows_sha256 checksum"

    expected_rows = descriptor.get("rows_sha256") or descriptor.get("content_sha256")
    if expected_rows and str(expected_rows) != actual_rows_checksum:
        return f"shard result {path} does not match attested rows checksum"

    expected_file = descriptor.get("result_sha256") or descriptor.get("file_sha256")
    if expected_file and str(expected_file) != sha256_path(path):
        return f"shard result {path} does not match attested file checksum"

    expected_generic = descriptor.get("checksum")
    if expected_generic and str(expected_generic) not in {actual_rows_checksum, sha256_path(path)}:
        return f"shard result {path} does not match attested checksum"
    return None


def _metric_schema(record: Mapping[str, Any]) -> tuple[str, ...]:
    metrics = record.get("metrics")
    if metrics is None:
        metrics = record
    if not isinstance(metrics, Mapping):
        raise CollectionError(f"record has invalid metrics schema: {record}")
    return tuple(sorted(str(key) for key in metrics))
def _manifest_grid_shards(manifest: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], int]:
    grid = manifest.get("grid")
    if not isinstance(grid, Mapping):
        raise CollectionError("grid manifest has no grid section")
    assignments = grid.get("shards")
    pools = grid.get("pools")
    if not isinstance(assignments, list) or not assignments:
        raise CollectionError("manifest grid.shards must be a non-empty array")
    if not isinstance(pools, list) or not pools:
        raise CollectionError("manifest grid.pools must be a non-empty array")
    return assignments, len(pools)



def validate_shards(
    manifest: Mapping[str, Any],
    results_dir: Path,
) -> tuple[list[dict[str, Any]], str]:
    """Validate all canonical shard envelopes against manifest assignments.

    Presence is checked for every assignment before any present file is parsed. This
    makes a partial collection report all absent shards instead of masking them with
    an unrelated malformed row in an earlier shard.
    """
    assignments, expected_total = _manifest_grid_shards(manifest)
    expected_indices: set[int] = set()
    descriptor_ranges: dict[str, set[int]] = {}
    for desc in assignments:
        if not isinstance(desc, Mapping):
            raise CollectionError(f"invalid shard descriptor: {desc!r}")
        shard_id = str(desc.get("shard_id", ""))
        if not shard_id or Path(shard_id).name != shard_id or shard_id in descriptor_ranges:
            raise CollectionError(f"invalid or duplicate shard_id {shard_id!r}")
        shard_expected: set[int] = set()
        for start, end in _ranges(desc):
            shard_expected.update(range(start, end))
        descriptor_ranges[shard_id] = shard_expected
        if expected_indices.intersection(shard_expected):
            raise CollectionError(f"shard assignments overlap at pool indices for {shard_id}")
        expected_indices.update(shard_expected)

    if expected_indices != set(range(expected_total)):
        raise CollectionError("shard assignments do not exactly partition grid.pools")

    # Presence pass: report every missing canonical file before reading any row.
    shard_paths: dict[str, Path] = {}
    missing: list[str] = []
    for desc in assignments:
        shard_id = str(desc["shard_id"])
        shard_file = results_dir / f"{shard_id}.json"
        shard_paths[shard_id] = shard_file
        if not shard_file.is_file():
            missing.append(shard_id)
    if missing:
        raise CollectionError(f"missing shard result files: {', '.join(missing)}")

    collected_records: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    schema_fields: tuple[str, ...] | None = None
    run_id = manifest.get("run_id")
    common_session_digests: dict[str, str] | None = None
    common_request_set_sha256: str | None = None

    for desc in assignments:
        shard_id = str(desc["shard_id"])
        shard_file = shard_paths[shard_id]
        payload, records = _load_shard_records(shard_file)
        if payload.get("shard_id") != shard_id:
            raise CollectionError(f"shard result {shard_file} has wrong shard_id")
        if run_id is not None and payload.get("run_id") != run_id:
            raise CollectionError(f"shard result {shard_file} has wrong run_id")
        if payload.get("shard_index") != int(desc.get("shard_index", -1)):
            raise CollectionError(f"shard result {shard_file} has wrong shard_index")
        if payload.get("ranges") != [list(r) for r in _ranges(desc)]:
            raise CollectionError(f"shard result {shard_file} has wrong assigned ranges")
        if payload.get("row_count") != len(records):
            raise CollectionError(f"shard result {shard_file} has wrong row_count")
        checksum_error = _descriptor_checksum_error(desc, shard_file, payload)
        if checksum_error:
            raise CollectionError(checksum_error)

        blade = str(desc.get("blade", ""))
        expected_session_id = f"sess_{run_id}_{blade}"
        raw_attestation = payload.get("session_attestation")
        if not isinstance(raw_attestation, Mapping) or set(raw_attestation) != {
            "session_id",
            *SESSION_ATTESTATION_FIELDS,
        }:
            raise CollectionError(f"shard result {shard_file} has invalid session_attestation fields")
        attestation = normalize_session_attestation(
            raw_attestation,
            expected_session_id=expected_session_id,
        )
        session_digests = _session_digest_view(attestation)
        if common_session_digests is None:
            common_session_digests = session_digests
        elif session_digests != common_session_digests:
            raise CollectionError(f"shard result {shard_file} was evaluated against a different opened session")

        expected_request_sha256 = grid_request_set_sha256(manifest, attestation)
        if payload.get("request_set_sha256") != expected_request_sha256:
            raise CollectionError(f"shard result {shard_file} does not match the manifest request set")
        if common_request_set_sha256 is None:
            common_request_set_sha256 = expected_request_sha256
        elif expected_request_sha256 != common_request_set_sha256:
            raise CollectionError(f"shard result {shard_file} has a different request set")

        shard_expected = descriptor_ranges[shard_id]
        shard_actual: set[int] = set()
        for record in records:
            idx = _record_index(record)
            if idx in shard_actual or idx in seen_indices:
                raise CollectionError(f"duplicate pool index {idx} in shard {shard_id}")
            shard_actual.add(idx)
            if idx not in shard_expected:
                raise CollectionError(f"unexpected pool index {idx} in shard {shard_id}")
            fields = _metric_schema(record)
            if schema_fields is None:
                schema_fields = fields
            elif fields != schema_fields:
                raise CollectionError(f"metric schema mismatch in shard {shard_id} at pool index {idx}")
            collected_records.append(record)

        if [int(record["pool_index"]) for record in records] != sorted(
            int(record["pool_index"]) for record in records
        ):
            raise CollectionError(f"shard {shard_id} rows are not in canonical pool_index order")
        if shard_actual != shard_expected:
            missing_in_shard = sorted(shard_expected - shard_actual)
            extra_in_shard = sorted(shard_actual - shard_expected)
            raise CollectionError(
                f"shard {shard_id} has non-exact pool index coverage: "
                f"missing={missing_in_shard[:5]}, extra={extra_in_shard[:5]}"
            )
        seen_indices.update(shard_actual)

    if seen_indices != expected_indices:
        missing_total = sorted(expected_indices - seen_indices)
        extra_total = sorted(seen_indices - expected_indices)
        raise CollectionError(
            f"collection has non-exact pool index coverage: "
            f"missing={missing_total[:5]}, extra={extra_total[:5]}"
        )

    sorted_records = sorted(collected_records, key=_record_index)
    schema_sig = json.dumps({"fields": list(schema_fields or ())}, sort_keys=True)
    return sorted_records, schema_sig


def fetch_remote_shards(
    manifest: Mapping[str, Any],
    local_results_dir: Path,
    ssh_config: SSHConfig | None = None,
    adapter: ProcessAdapter | None = None,
) -> None:
    """Download all canonical remote shard artifacts to local results directory."""
    run_id = validate_run_id(str(manifest.get("run_id", "")))
    remote_base = PurePosixPath(str(manifest.get("remote_base", "/home/heswithme/arb")))
    assignments, _ = _manifest_grid_shards(manifest)

    ssh_adapter = SSHProcessAdapter(ssh_config=ssh_config, process_runner=adapter)
    local_results_dir.mkdir(parents=True, exist_ok=True)

    for desc in assignments:
        blade = str(desc.get("blade", ""))
        shard_id = str(desc.get("shard_id", ""))
        remote_file = scoped_remote_path(run_id, f"results/{shard_id}.json", remote_base=remote_base)
        local_target = local_results_dir / f"{shard_id}.json"
        res = ssh_adapter.rsync_download(blade, str(remote_file), local_target)
        if not res.ok:
            raise CollectionError(
                f"failed to fetch shard {shard_id} from {blade}:{remote_file}: {res.stderr}"
            )


def write_grid_results_npz(
    path: Path,
    *,
    run_id: str,
    schema_signature: str,
    request_set_sha256: str,
    session_attestation: Mapping[str, str],
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    """Atomically write collected result rows as compressed, pickle-free NPZ."""
    destination = Path(path)
    if destination.suffix != ".npz":
        raise CollectionError("collected grid result path must end in .npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": np.asarray(GRID_RESULTS_SCHEMA_VERSION),
        "run_id": np.asarray(run_id),
        "schema_signature": np.asarray(schema_signature),
        "request_set_sha256": np.asarray(request_set_sha256),
        "session_attestation_json": np.asarray(
            _canonical_json(dict(session_attestation)).decode("utf-8")
        ),
        "total_rows": np.asarray(len(rows), dtype=np.int64),
        "rows_json": np.asarray(
            [_canonical_json(dict(row)).decode("utf-8") for row in rows],
            dtype=np.str_,
        ),
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


def load_grid_results_npz(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load and validate one compressed collected-result artifact."""
    source = Path(path)
    try:
        with np.load(source, allow_pickle=False) as archive:
            schema = str(np.asarray(archive["schema_version"]).item())
            if schema != GRID_RESULTS_SCHEMA_VERSION:
                raise CollectionError(f"unsupported grid result NPZ schema {schema!r}")
            rows_raw = archive["rows_json"].astype(str)
            total_rows = int(np.asarray(archive["total_rows"]).item())
            if rows_raw.shape != (total_rows,):
                raise CollectionError("grid result NPZ row count does not match rows_json")
            rows = [json.loads(value) for value in rows_raw]
            if not all(isinstance(row, Mapping) for row in rows):
                raise CollectionError("grid result NPZ rows must decode to objects")
            metadata = {
                "schema_version": schema,
                "run_id": str(np.asarray(archive["run_id"]).item()),
                "schema_signature": str(np.asarray(archive["schema_signature"]).item()),
                "request_set_sha256": str(np.asarray(archive["request_set_sha256"]).item()),
                "session_attestation": json.loads(
                    str(np.asarray(archive["session_attestation_json"]).item())
                ),
                "total_rows": total_rows,
            }
    except CollectionError:
        raise
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise CollectionError(f"invalid grid result NPZ {source}: {exc}") from exc
    return metadata, [dict(row) for row in rows]


def collect_grid_results(
    manifest_path: Path,
    output_file: Path | None = None,
    *,
    ssh_config: SSHConfig | None = None,
    adapter: ProcessAdapter | None = None,
) -> Path:
    """Validate and merge all canonical shard results into one NPZ output."""
    manifest_file = Path(manifest_path).resolve()
    with manifest_file.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    run_dir = manifest_file.parent
    results_dir = run_dir / "results"
    assignments, _ = _manifest_grid_shards(manifest)
    scope = manifest.get("scope", "local")
    if scope == "cluster":
        missing = [
            str(desc.get("shard_id", ""))
            for desc in assignments
            if not (results_dir / f"{desc.get('shard_id', '')}.json").is_file()
        ]
        if missing:
            fetch_remote_shards(manifest, results_dir, ssh_config=ssh_config, adapter=adapter)

    sorted_records, schema_sig = validate_shards(manifest, results_dir)
    first_shard_id = str(assignments[0]["shard_id"])
    first_payload, _ = _load_shard_records(results_dir / f"{first_shard_id}.json")
    target_output = output_file or (run_dir / "grid_results.npz")
    write_grid_results_npz(
        target_output,
        run_id=str(manifest.get("run_id", "")),
        schema_signature=schema_sig,
        request_set_sha256=str(first_payload["request_set_sha256"]),
        session_attestation=_session_digest_view(first_payload["session_attestation"]),
        rows=sorted_records,
    )
    if manifest.get("run_kind") == "grid" and isinstance(manifest.get("resolved_spec"), Mapping):
        try:
            from ..grids.runner import collect_grid_run

            collect_grid_run(manifest_file, results_path=target_output)
        except Exception as exc:  # noqa: BLE001
            raise CollectionError(f"canonical evaluation table collection failed: {exc}") from exc
    return target_output


__all__ = [
    "CollectionError",
    "GRID_REQUEST_SCHEMA_VERSION",
    "GRID_RESULTS_SCHEMA_VERSION",
    "SHARD_RESULT_SCHEMA_VERSION",
    "collect_grid_results",
    "fetch_remote_shards",
    "grid_request_set_sha256",
    "make_shard_result",
    "normalize_session_attestation",
    "load_grid_results_npz",
    "write_grid_results_npz",
    "validate_shards",
    "write_shard_result",
]
