"""Compact ordinal-only grid shard publication and streaming collection."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

import numpy as np

from ..artifacts.io import atomic_write_json, sha256_path
from ..artifacts.manifest import load_manifest, write_manifest_atomic
from ..artifacts.tables import GRID_TABLE_SCHEMA_VERSION, MetricProjection
from ..specs.common import canonical_json_bytes

if TYPE_CHECKING:
    from ..grids.model import CartesianGridPlan

SHARD_NPZ_SCHEMA_VERSION = "fxsim_grid_shard_npz_v1"
SHARD_RECEIPT_SCHEMA_VERSION = "fxsim_grid_shard_receipt_v1"
SESSION_ATTESTATION_FIELDS = (
    "scenario_set_sha256", "session_fingerprint", "session_config_sha256",
    "metric_schema_sha256",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STATUS = {"ok": 0, "failed": 1, "cancelled": 2}


class CollectionError(RuntimeError):
    """A shard or collected table violates its immutable grid contract."""


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise CollectionError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _sha_bytes(value: bytes) -> bool:
    try:
        return len(value) == 64 and _SHA256.fullmatch(value.decode("ascii")) is not None
    except UnicodeDecodeError:
        return False


def normalize_session_attestation(value: Any, *, expected_session_id: str | None = None) -> dict[str, str]:
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if not isinstance(value, Mapping):
        raise CollectionError("session attestation must be an object")
    session_id = value.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise CollectionError("session attestation has no session_id")
    if expected_session_id is not None and session_id != expected_session_id:
        raise CollectionError("session attestation has the wrong session_id")
    result = {"session_id": session_id}
    for field in SESSION_ATTESTATION_FIELDS:
        result[field] = _require_sha(value.get(field), f"session attestation {field}")
    return result


def _normalized_attestations(value: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    if not isinstance(value, Mapping) or not value:
        raise CollectionError("grid shard requires session attestations")
    return {
        _require_sha(group_id, "session group id"): normalize_session_attestation(value[group_id])
        for group_id in sorted(value)
    }


def _request_set_sha256(
    manifest: Mapping[str, Any], shard: Mapping[str, Any],
    ranges: Sequence[tuple[int, int]], projection: MetricProjection,
    session_attestations: Mapping[str, Any],
) -> str:
    """Stable resume identity; attempt-specific work-request identity is excluded."""
    plan = manifest["grid"]["plan"]
    return _digest({
        "schema_version": SHARD_RECEIPT_SCHEMA_VERSION,
        "run_id": manifest["run_id"], "plan_sha256": plan["plan_sha256"],
        "artifact_sha256": plan["artifact_sha256"],
        "metric_projection_sha256": projection.projection_sha256,
        "shard_id": shard["shard_id"], "shard_index": int(shard["shard_index"]),
        "ranges": [list(item) for item in ranges],
        "session_attestations": _normalized_attestations(session_attestations),
    })


def _projection(manifest: Mapping[str, Any]) -> MetricProjection:
    resolved = manifest.get("resolved_spec")
    raw = resolved.get("metric_projection") if isinstance(resolved, Mapping) else None
    if not isinstance(raw, Mapping):
        raise CollectionError("grid manifest has no metric projection")
    try:
        return MetricProjection(tuple(raw["fields"]), str(raw.get("projection_id", "grid")),
                                str(raw.get("projection_sha256", "")))
    except (KeyError, TypeError, ValueError) as exc:
        raise CollectionError(f"invalid grid metric projection: {exc}") from exc


def _grid_identity(manifest: Mapping[str, Any]) -> tuple[str, str, int]:
    grid = manifest.get("grid")
    plan = grid.get("plan") if isinstance(grid, Mapping) else None
    if not isinstance(plan, Mapping):
        raise CollectionError("grid manifest has no Cartesian plan")
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise CollectionError("grid manifest has no run_id")
    return run_id, _require_sha(plan.get("plan_sha256"), "grid plan SHA-256"), int(grid["pool_count"])


def _ranges(descriptor: Mapping[str, Any], pool_count: int) -> tuple[tuple[int, int], ...]:
    raw = descriptor.get("ranges")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise CollectionError("grid shard has no ranges")
    try:
        ranges = tuple((int(item[0]), int(item[1])) for item in raw)
    except (IndexError, TypeError, ValueError) as exc:
        raise CollectionError("grid shard ranges are invalid") from exc
    previous = -1
    for start, end in ranges:
        if start < 0 or start < previous or end <= start or end > pool_count:
            raise CollectionError("grid shard ranges are not sorted valid half-open ranges")
        previous = end
    if sum(end - start for start, end in ranges) > 2048:
        raise CollectionError("grid shard exceeds the 2,048-point persistence bound")
    return ranges


def _ordinals(ranges: Sequence[tuple[int, int]]) -> tuple[int, ...]:
    return tuple(ordinal for start, end in ranges for ordinal in range(start, end))


def _result_dict(result: Any) -> dict[str, Any]:
    if hasattr(result, "model_dump"):
        result = result.model_dump()
    if not isinstance(result, Mapping):
        raise CollectionError("grid result must be an object")
    return dict(result)


def _error_columns(errors: Sequence[str | None]) -> tuple[np.ndarray, np.ndarray]:
    encoded = [(error or "").encode("utf-8") for error in errors]
    offsets = np.zeros(len(encoded) + 1, dtype="<i8")
    for index, value in enumerate(encoded):
        offsets[index + 1] = offsets[index] + len(value)
    return offsets, np.frombuffer(b"".join(encoded), dtype="u1").copy()


def _atomic_npz(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return path


def write_grid_shard_result(
    npz_path: Path | str, receipt_path: Path | str, *, manifest: Mapping[str, Any],
    plan: CartesianGridPlan, shard: Mapping[str, Any], blade: str,
    work_request_sha256: str, session_attestations: Mapping[str, Any],
    results: Iterable[Any],
) -> Mapping[str, Any]:
    """Persist one ordinal-sorted shard NPZ, then its canonical receipt."""
    destination, receipt_destination = Path(npz_path), Path(receipt_path)
    if destination.suffix != ".npz":
        raise CollectionError("grid shard path must end in .npz")
    shard_id = str(shard.get("shard_id", ""))
    if (not shard_id or destination.name != f"{shard_id}.npz"
            or receipt_destination.name != f"{shard_id}.receipt.json"
            or destination.parent.resolve() != receipt_destination.parent.resolve()
            or destination.parent.name != "results"):
        raise CollectionError("grid shard paths do not match its shard id")
    run_id, plan_sha, pool_count = _grid_identity(manifest)
    if plan.plan_sha256 != plan_sha or plan.pool_count != pool_count:
        raise CollectionError("grid shard plan differs from manifest")
    ranges = _ranges(shard, pool_count)
    expected = _ordinals(ranges)
    projection = _projection(manifest)
    normalized = _normalized_attestations(session_attestations)
    expected_groups = {point.session_group_id for point in plan.iter_points(ranges)}
    if set(normalized) != expected_groups:
        raise CollectionError("grid shard session attestations do not exactly cover its plan groups")
    by_ordinal: dict[int, dict[str, Any]] = {}
    for raw in results:
        row = _result_dict(raw)
        try:
            ordinal = int(row["ordinal"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CollectionError("grid result has no valid ordinal") from exc
        if ordinal in by_ordinal:
            raise CollectionError(f"duplicate grid result ordinal {ordinal}")
        if row.get("candidate_id") != plan.candidate_id_at(ordinal):
            raise CollectionError(f"grid result ordinal {ordinal} has the wrong candidate id")
        by_ordinal[ordinal] = row
    if set(by_ordinal) != set(expected):
        raise CollectionError("grid shard result ordinals do not exactly cover its ranges")

    statuses = np.empty(len(expected), dtype="u1")
    fingerprints = np.zeros(len(expected), dtype="S64")
    fingerprint_present = np.zeros(len(expected), dtype=np.bool_)
    metric_values = np.full((len(expected), len(projection.fields)), np.nan, dtype="<f8")
    metric_present = np.zeros(metric_values.shape, dtype=np.bool_)
    errors: list[str | None] = []
    counts = {name: 0 for name in _STATUS}
    for index, ordinal in enumerate(expected):
        row = by_ordinal[ordinal]
        status = row.get("status", "ok")
        if status not in _STATUS:
            raise CollectionError(f"grid result ordinal {ordinal} has invalid status")
        statuses[index], counts[status] = _STATUS[status], counts[status] + 1
        error = row.get("error")
        if error is not None and not isinstance(error, str):
            raise CollectionError(f"grid result ordinal {ordinal} has invalid error")
        errors.append(error)
        fingerprint = row.get("economic_fingerprint")
        if fingerprint:
            fingerprints[index], fingerprint_present[index] = _require_sha(fingerprint, "economic fingerprint").encode(), True
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            raise CollectionError(f"grid result ordinal {ordinal} has invalid metrics")
        for metric_index, name in enumerate(projection.fields):
            value = metrics.get(name)
            if value is None:
                continue
            if isinstance(value, bool) or not np.isfinite(float(value)):
                raise CollectionError(f"grid result ordinal {ordinal} metric {name!r} is not finite")
            metric_values[index, metric_index], metric_present[index, metric_index] = float(value), True
        if status == "ok" and (error is not None or not fingerprint_present[index] or not np.all(metric_present[index])):
            raise CollectionError(f"successful grid result ordinal {ordinal} is incomplete")

    offsets, error_bytes = _error_columns(errors)
    shard_index = int(shard["shard_index"])
    payload: dict[str, Any] = {
        "schema_version": np.asarray(SHARD_NPZ_SCHEMA_VERSION), "run_id": np.asarray(run_id),
        "shard_id": np.asarray(str(shard["shard_id"])),
        "plan_sha256": np.asarray(plan_sha.encode(), dtype="S64"),
        "artifact_sha256": np.asarray(plan.artifact_sha256.encode(), dtype="S64"),
        "metric_projection_sha256": np.asarray(projection.projection_sha256.encode(), dtype="S64"),
        "shard_index": np.asarray(shard_index, dtype="<i8"),
        "row_count": np.asarray(len(expected), dtype="<i8"), "metric_names": np.asarray(projection.fields, dtype=np.str_),
        "ordinal": np.asarray(expected, dtype="<i8"), "status": statuses,
        "economic_fingerprint": fingerprints, "economic_fingerprint_present": fingerprint_present,
        "error_offsets": offsets, "error_utf8": error_bytes,
        "metric_values": metric_values, "metric_present": metric_present,
    }
    _atomic_npz(destination, payload)
    try:
        relative_npz = destination.resolve().relative_to(destination.parent.parent.resolve()).as_posix()
    except ValueError as exc:
        raise CollectionError("grid shard must stay beneath its run directory") from exc
    receipt = {
        "schema_version": SHARD_RECEIPT_SCHEMA_VERSION, "run_id": run_id,
        "shard_id": str(shard["shard_id"]), "shard_index": shard_index,
        "plan_sha256": plan_sha, "artifact_sha256": plan.artifact_sha256,
        "metric_projection_sha256": projection.projection_sha256,
        "ranges": [list(item) for item in ranges], "blade": str(blade),
        "work_request_sha256": _require_sha(work_request_sha256, "work request SHA-256"),
        "request_set_sha256": _request_set_sha256(manifest, shard, ranges, projection, normalized),
        "session_attestations": normalized, "row_count": len(expected),
        "result": {"path": relative_npz, "sha256": sha256_path(destination),
                   "bytes": destination.stat().st_size},
        "status_counts": counts,
    }
    atomic_write_json(receipt_destination, receipt)
    return receipt


def _read_receipt(path: Path | str) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CollectionError(f"invalid grid shard receipt {source}: {exc}") from exc
    expected = {
        "schema_version", "run_id", "plan_sha256", "artifact_sha256",
        "metric_projection_sha256", "shard_id", "shard_index", "ranges",
        "row_count", "blade", "work_request_sha256", "request_set_sha256",
        "session_attestations", "result", "status_counts",
    }
    if (not isinstance(value, Mapping)
            or value.get("schema_version") != SHARD_RECEIPT_SCHEMA_VERSION
            or set(value) != expected):
        raise CollectionError(f"unsupported grid shard receipt {source}")
    return dict(value)


def _validate_receipt(manifest: Mapping[str, Any], plan: CartesianGridPlan | None,
                      descriptor: Mapping[str, Any], receipt_path: Path,
                      *, verify_npz: bool) -> tuple[dict[str, Any], Path]:
    receipt = _read_receipt(receipt_path)
    run_id, plan_sha, pool_count = _grid_identity(manifest)
    if plan is not None and (plan.plan_sha256 != plan_sha or plan.pool_count != pool_count):
        raise CollectionError("grid shard plan differs from manifest")
    projection, ranges = _projection(manifest), _ranges(descriptor, pool_count)
    expected = {
        "run_id": run_id, "shard_id": str(descriptor["shard_id"]),
        "shard_index": int(descriptor["shard_index"]),
        "plan_sha256": plan_sha, "artifact_sha256": manifest["grid"]["plan"]["artifact_sha256"],
        "metric_projection_sha256": projection.projection_sha256,
        "ranges": [list(item) for item in ranges],
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise CollectionError(f"grid shard receipt {receipt_path} differs from its manifest descriptor")
    if descriptor.get("blade") is not None and receipt.get("blade") != descriptor["blade"]:
        raise CollectionError(f"grid shard receipt {receipt_path} has the wrong blade")
    expected_work = descriptor.get("work_request_sha256")
    if expected_work is not None and receipt.get("work_request_sha256") != expected_work:
        raise CollectionError(f"grid shard receipt {receipt_path} has the wrong work request")
    _require_sha(receipt.get("work_request_sha256"), "work request SHA-256")
    if not isinstance(receipt.get("blade"), str) or not receipt["blade"]:
        raise CollectionError(f"grid shard receipt {receipt_path} has no blade")
    normalized = _normalized_attestations(receipt.get("session_attestations", {}))
    if plan is not None:
        expected_groups = {
            point.session_group_id for point in plan.iter_points(ranges)
        }
        if set(normalized) != expected_groups:
            raise CollectionError(f"grid shard receipt {receipt_path} has the wrong session groups")
    if receipt.get("request_set_sha256") != _request_set_sha256(manifest, descriptor, ranges, projection, normalized):
        raise CollectionError(f"grid shard receipt {receipt_path} has an invalid request set")
    npz_record = receipt.get("result")
    expected_path = f"results/{descriptor['shard_id']}.npz"
    if not isinstance(npz_record, Mapping) or npz_record.get("path") != expected_path:
        raise CollectionError(f"grid shard receipt {receipt_path} has an invalid NPZ path")
    npz_path = receipt_path.parent.parent / str(npz_record["path"])
    if not npz_path.is_file() or npz_path.stat().st_size != npz_record.get("bytes") or (verify_npz and sha256_path(npz_path) != npz_record.get("sha256")):
        raise CollectionError(f"grid shard receipt {receipt_path} NPZ attestation failed")
    return receipt, npz_path


def load_grid_shard_receipt(
    receipt_path: Path | str, *, manifest: Mapping[str, Any], plan: CartesianGridPlan,
    shard: Mapping[str, Any], verify_npz: bool = True,
) -> Mapping[str, Any]:
    receipt, npz_path = _validate_receipt(
        manifest, plan, shard, Path(receipt_path), verify_npz=verify_npz
    )
    if verify_npz:
        _load_shard(npz_path, manifest, shard, receipt)
    return receipt


def is_grid_shard_complete(manifest: Mapping[str, Any], plan: CartesianGridPlan,
                           shard: Mapping[str, Any], receipt_path: Path | str) -> bool:
    """Return true only after receipt, NPZ identity, digest, and columns validate."""
    try:
        load_grid_shard_receipt(receipt_path, manifest=manifest, plan=plan,
                               shard=shard, verify_npz=True)
        return True
    except (CollectionError, KeyError, OSError, TypeError, ValueError):
        return False


def _scalar(archive: Mapping[str, Any], name: str) -> Any:
    value = np.asarray(archive[name])
    if value.shape != ():
        raise CollectionError(f"grid shard field {name!r} must be scalar")
    item = value.item()
    return item.decode("ascii") if isinstance(item, bytes) else item


def _load_shard(npz_path: Path, manifest: Mapping[str, Any], descriptor: Mapping[str, Any], receipt: Mapping[str, Any]) -> dict[str, np.ndarray]:
    projection = _projection(manifest)
    ranges = _ranges(descriptor, int(manifest["grid"]["pool_count"]))
    expected_ordinals = np.asarray(_ordinals(ranges), dtype="<i8")
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            scalars = {
                "schema_version": SHARD_NPZ_SCHEMA_VERSION, "run_id": manifest["run_id"],
                "shard_id": descriptor["shard_id"], "plan_sha256": manifest["grid"]["plan"]["plan_sha256"],
                "artifact_sha256": manifest["grid"]["plan"]["artifact_sha256"],
                "metric_projection_sha256": projection.projection_sha256,
                "shard_index": int(descriptor["shard_index"]),
                "row_count": len(expected_ordinals),
            }
            if any(_scalar(archive, key) != value for key, value in scalars.items()):
                raise CollectionError(f"grid shard {npz_path} scalar identity mismatch")
            expected_fields = {
                "schema_version", "run_id", "shard_id", "plan_sha256", "artifact_sha256",
                "metric_projection_sha256", "shard_index", "row_count", "metric_names",
                "ordinal", "status", "economic_fingerprint", "economic_fingerprint_present",
                "error_offsets", "error_utf8", "metric_values", "metric_present",
            }
            if set(archive.files) != expected_fields:
                raise CollectionError(f"grid shard {npz_path} fields are invalid")
            data = {name: np.asarray(archive[name]).copy() for name in (
                "metric_names", "ordinal", "status", "economic_fingerprint",
                "economic_fingerprint_present", "error_offsets", "error_utf8",
                "metric_values", "metric_present")}
    except (KeyError, OSError, ValueError) as exc:
        raise CollectionError(f"invalid grid shard NPZ {npz_path}: {exc}") from exc
    count, metrics = len(expected_ordinals), len(projection.fields)
    if tuple(data["metric_names"].astype(str)) != projection.fields or not np.array_equal(data["ordinal"], expected_ordinals):
        raise CollectionError(f"grid shard {npz_path} ordinal or metric identity mismatch")
    expected_shapes = {"status": (count,), "economic_fingerprint": (count,),
        "economic_fingerprint_present": (count,), "error_offsets": (count + 1,),
        "metric_values": (count, metrics), "metric_present": (count, metrics)}
    if any(data[name].shape != shape for name, shape in expected_shapes.items()):
        raise CollectionError(f"grid shard {npz_path} column shapes are invalid")
    if data["ordinal"].dtype != np.dtype("<i8") or data["status"].dtype != np.dtype("u1") or np.any(data["status"] > 2):
        raise CollectionError(f"grid shard {npz_path} typed columns are invalid")
    if data["economic_fingerprint"].dtype != np.dtype("S64") or data["metric_values"].dtype != np.dtype("<f8"):
        raise CollectionError(f"grid shard {npz_path} value dtypes are invalid")
    if data["economic_fingerprint_present"].dtype != np.dtype(bool) or data["metric_present"].dtype != np.dtype(bool):
        raise CollectionError(f"grid shard {npz_path} presence dtypes are invalid")
    offsets = data["error_offsets"]
    if (offsets.dtype != np.dtype("<i8") or offsets[0] != 0
            or data["error_utf8"].dtype != np.dtype("u1")
            or offsets[-1] != len(data["error_utf8"]) or np.any(np.diff(offsets) < 0)):
        raise CollectionError(f"grid shard {npz_path} error offsets are invalid")
    try:
        data["error_utf8"].tobytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CollectionError(f"grid shard {npz_path} errors are not UTF-8") from exc
    counts = {name: int(np.count_nonzero(data["status"] == code)) for name, code in _STATUS.items()}
    if receipt.get("status_counts") != counts or receipt.get("row_count") != count:
        raise CollectionError(f"grid shard receipt {npz_path} counts are invalid")
    ok = data["status"] == 0
    for raw, present in zip(data["economic_fingerprint"], data["economic_fingerprint_present"], strict=True):
        value = bytes(raw)
        if (present and not _sha_bytes(value)) or (not present and value):
            raise CollectionError(f"grid shard {npz_path} fingerprint presence is invalid")
    if (np.any(ok & ~data["economic_fingerprint_present"])
            or np.any(ok & (np.diff(offsets) != 0))
            or np.any(ok[:, None] & ~data["metric_present"])
            or np.any(data["metric_present"] & ~np.isfinite(data["metric_values"]))
            or np.any(~data["metric_present"] & ~np.isnan(data["metric_values"]))):
        raise CollectionError(f"successful rows in {npz_path} are incomplete")
    return data


def collect_grid_results(manifest_path: Path | str, output_path: Path | str | None = None) -> Path:
    """Stream verified shard columns into the sole evaluation_table.npz authority."""
    manifest_file = Path(manifest_path).resolve()
    manifest = load_manifest(manifest_file, expected_kind="grid")
    run_dir, results_dir = manifest_file.parent, manifest_file.parent / "results"
    try:
        from ..evaluation.plans import ScenarioKey
        from ..evaluation.selected import SelectedEvaluator
        from ..grids.runner import load_grid_plan

        selected = SelectedEvaluator.load(run_dir / "evaluator_artifact")
        raw = manifest["grid"]["plan"]["scenario_key"]
        scenario = ScenarioKey(str(raw["identity_json"]).encode(), str(raw["sha256"])).validated()
        plan_object = load_grid_plan(manifest, selected_evaluator=selected, scenario=scenario)
    except Exception as exc:
        raise CollectionError(f"cannot verify Cartesian grid plan: {exc}") from exc
    descriptors = manifest["grid"].get("shards")
    if not isinstance(descriptors, list) or not descriptors:
        raise CollectionError("grid manifest has no shard descriptors")
    descriptors = sorted(descriptors, key=lambda item: _ranges(item, manifest["grid"]["pool_count"])[0][0])
    pool_count, cursor = int(manifest["grid"]["pool_count"]), 0
    for descriptor in descriptors:
        values = _ordinals(_ranges(descriptor, pool_count))
        if values != tuple(range(cursor, cursor + len(values))):
            raise CollectionError("grid shard descriptors do not exactly partition ordinal order")
        cursor += len(values)
    if cursor != pool_count:
        raise CollectionError("grid shard descriptors do not cover the Cartesian plan")
    projection = _projection(manifest)
    target = Path(output_path).resolve() if output_path else run_dir / "evaluation_table.npz"
    if target.name != "evaluation_table.npz" or target.parent != run_dir:
        raise CollectionError("grid collection output must be the run evaluation_table.npz")

    with tempfile.TemporaryDirectory(prefix=".grid-collection.", dir=run_dir) as temporary:
        scratch = Path(temporary)
        status = np.memmap(scratch / "status", mode="w+", dtype="u1", shape=(pool_count,))
        fingerprint = np.memmap(scratch / "fingerprint", mode="w+", dtype="S64", shape=(pool_count,))
        fingerprint_present = np.memmap(scratch / "fingerprint-present", mode="w+", dtype=np.bool_, shape=(pool_count,))
        offsets = np.memmap(scratch / "error-offsets", mode="w+", dtype="<i8", shape=(pool_count + 1,))
        metric_values = [np.memmap(scratch / f"metric-{i}-values", mode="w+", dtype="<f8", shape=(pool_count,)) for i in range(len(projection.fields))]
        metric_present = [np.memmap(scratch / f"metric-{i}-present", mode="w+", dtype=np.bool_, shape=(pool_count,)) for i in range(len(projection.fields))]
        error_path, error_cursor = scratch / "errors", 0
        with error_path.open("wb") as error_stream:
            offsets[0] = 0
            for descriptor in descriptors:
                receipt_path = results_dir / f"{descriptor['shard_id']}.receipt.json"
                receipt, npz_path = _validate_receipt(
                    manifest, plan_object, descriptor, receipt_path, verify_npz=True
                )
                shard = _load_shard(npz_path, manifest, descriptor, receipt)
                for local, ordinal in enumerate(shard["ordinal"]):
                    value = int(ordinal)
                    status[value] = shard["status"][local]
                    fingerprint[value] = shard["economic_fingerprint"][local]
                    fingerprint_present[value] = shard["economic_fingerprint_present"][local]
                    start, end = int(shard["error_offsets"][local]), int(shard["error_offsets"][local + 1])
                    encoded = shard["error_utf8"][start:end].tobytes()
                    error_stream.write(encoded); error_cursor += len(encoded); offsets[value + 1] = error_cursor
                    for metric_index in range(len(projection.fields)):
                        metric_values[metric_index][value] = shard["metric_values"][local, metric_index]
                        metric_present[metric_index][value] = shard["metric_present"][local, metric_index]
            error_stream.flush(); os.fsync(error_stream.fileno())
        error_bytes = np.memmap(error_path, mode="r", dtype="u1", shape=(error_cursor,)) if error_cursor else np.empty(0, dtype="u1")
        plan = manifest["grid"]["plan"]
        payload: dict[str, Any] = {
            "schema_version": np.asarray(GRID_TABLE_SCHEMA_VERSION),
            "table_kind": np.asarray("cartesian_grid"), "run_id": np.asarray(manifest["run_id"]),
            "plan_sha256": np.asarray(plan["plan_sha256"].encode(), dtype="S64"),
            "artifact_sha256": np.asarray(plan["artifact_sha256"].encode(), dtype="S64"),
            "metric_projection_sha256": np.asarray(projection.projection_sha256.encode(), dtype="S64"),
            "row_count": np.asarray(pool_count, dtype="<i8"), "metric_names": np.asarray(projection.fields, dtype=np.str_),
            "metric_projection_json": np.asarray(canonical_json_bytes(projection.to_dict()).decode()),
            "metadata_json": np.asarray(canonical_json_bytes({"source_kind": "grid", "run_id": manifest["run_id"],
                "grid_id": manifest["grid"]["grid_id"], "shape": plan["coordinate_shape"]}).decode()),
            "status": status, "economic_fingerprint": fingerprint,
            "economic_fingerprint_present": fingerprint_present,
            "error_offsets": offsets, "error_utf8": error_bytes,
        }
        for index in range(len(projection.fields)):
            payload[f"metric_{index:03d}_values"] = metric_values[index]
            payload[f"metric_{index:03d}_present"] = metric_present[index]
        _atomic_npz(target, payload)

    table_ref = {"path": target.name, "sha256": sha256_path(target), "bytes": target.stat().st_size,
                 "row_count": pool_count, "metric_projection": projection.to_dict()}
    manifest["grid"]["table_ref"] = table_ref
    artifacts = [item for item in manifest.get("artifacts", ()) if item.get("kind") != "evaluation_table"]
    artifacts.append({"kind": "evaluation_table", "path": target.name,
                      "sha256": table_ref["sha256"], "bytes": table_ref["bytes"]})
    manifest["artifacts"] = artifacts
    write_manifest_atomic(manifest_file, manifest, expected_kind="grid")
    legacy = run_dir / "grid_results.npz"
    if legacy.exists():
        raise CollectionError("legacy grid_results.npz exists; refusing dual result authority")
    return target


__all__ = ["CollectionError", "SHARD_NPZ_SCHEMA_VERSION", "SHARD_RECEIPT_SCHEMA_VERSION",
           "collect_grid_results", "is_grid_shard_complete", "load_grid_shard_receipt",
           "normalize_session_attestation", "write_grid_shard_result"]
