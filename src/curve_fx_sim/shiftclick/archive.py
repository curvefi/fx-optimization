"""Pack verified evaluator sidecars into one candidate-wide replay archive."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..artifacts.io import atomic_write_json, sha256_path
from ..specs.common import assert_contained_path

TRACE_COLUMNS = (
    "t", "token0", "token1", "D", "xp_0", "xp_1", "price_oracle",
    "price_scale", "profit", "vp", "vp_boosted", "xcp", "lp_xcp_profit",
    "total_supply", "donation_apy", "donation_shares", "donation_unlocked",
    "last_prices", "last_timestamp", "open", "high", "low", "close", "p_cex",
    "p_chainlink", "fee", "slippage_1pct_0to1", "slippage_1pct_1to0",
    "n_trades", "n_rebalances", "yb_initialized", "yb_growth", "yb_fee",
    "yb_releverage_trades", "yb_stable_balance", "yb_debt", "yb_collateral_lp",
    "yb_lp_oracle", "yb_lp_fair",
)
ACTION_COLUMNS = {
    "donation": ("ts", "ts_due", "amount0", "amount1", "price_scale", "donation_ratio1", "apy_per_year", "freq_s"),
    "tick": ("ts", "p_cex", "ps_before", "ps_after", "oracle_before", "oracle_after", "xcp_profit_before", "xcp_profit_after", "vp_before", "vp_after"),
    "exchange": ("ts", "i", "j", "dx", "dy_after_fee", "fee_tokens", "profit_coin0", "p_cex", "p_pool_before", "p_pool_after", "oracle_before", "oracle_after", "ps_before", "ps_after", "lp_before", "lp_after", "xcp_profit_before", "xcp_profit_after", "vp_before", "vp_after"),
}
_INTEGER_FIELDS = {"t", "last_timestamp", "n_trades", "n_rebalances", "yb_initialized", "yb_releverage_trades", "ts", "ts_due", "freq_s", "i", "j"}


class ReplayArchiveError(ValueError):
    """Evaluator sidecars cannot be safely published as a replay archive."""


def _resolve(raw: str, root: Path, *, label: str) -> Path:
    try:
        path = assert_contained_path(root / raw if not Path(raw).is_absolute() else raw, root)
    except ValueError as exc:
        raise ReplayArchiveError(f"{label} escapes private staging") from exc
    if not path.is_file() or path.is_symlink():
        raise ReplayArchiveError(f"{label} is not a regular file")
    return path


def _verified(raw: str, digest: str, size: int | None, root: Path, label: str) -> Path:
    path = _resolve(raw, root, label=label)
    if size is not None and path.stat().st_size != size:
        raise ReplayArchiveError(f"{label} byte size mismatch")
    if sha256_path(path) != digest:
        raise ReplayArchiveError(f"{label} SHA-256 mismatch")
    return path


def _matrix(rows: Sequence[Any], columns: tuple[str, ...], *, kind: str) -> np.ndarray:
    matrix = np.empty((len(rows), len(columns)), dtype="<f8")
    expected = set(columns) | ({"type"} if kind != "trace" else set())
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != expected:
            raise ReplayArchiveError(f"{kind} row {row_index} has an unexpected schema")
        if kind != "trace" and row["type"] != kind:
            raise ReplayArchiveError(f"{kind} row {row_index} has the wrong type")
        for column_index, name in enumerate(columns):
            value = row[name]
            if value is None and kind == "trace":
                matrix[row_index, column_index] = np.nan
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ReplayArchiveError(f"{kind}.{name} row {row_index} is not numeric")
            number = float(value)
            if not np.isfinite(number):
                raise ReplayArchiveError(f"{kind}.{name} row {row_index} is not finite")
            if name in _INTEGER_FIELDS and (not isinstance(value, int) or abs(value) > 2**53):
                raise ReplayArchiveError(f"{kind}.{name} row {row_index} is not exact binary64 integer")
            matrix[row_index, column_index] = number
    return matrix


def pack_replay_archive(
    output_dir: Path, staging_dir: Path, *, source_run_id: str,
    candidate_id: str, ordinal: int, scenarios: Sequence[Mapping[str, Any]],
    require_actions: bool,
) -> tuple[Path, Path, dict[str, Any]]:
    """Verify, pack, publish, then remove one replay's private sidecars."""
    output, staging = Path(output_dir).resolve(), Path(staging_dir).resolve()
    if (output == Path(output.anchor) or staging != output / ".replay_staging"
            or Path(staging_dir).is_symlink() or not staging.is_dir()):
        raise ReplayArchiveError("private staging must be the output's .replay_staging directory")
    arrays: dict[str, np.ndarray] = {}
    records = []
    for index, scenario in enumerate(scenarios):
        artifacts = scenario["artifacts"]
        manifest = _verified(artifacts.manifest_path, artifacts.manifest_sha256, None, staging, "trace manifest")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if (payload.get("manifest_version") != "curve_fx_trace_manifest_v1"
                or payload.get("candidate_id") != scenario["evaluation_id"]
                or payload.get("economic_fingerprint") != scenario["economic_fingerprint"]
                or set(payload.get("scenarios", {})) != {scenario["id"]}):
            raise ReplayArchiveError("trace manifest identity mismatch")
        sidecars, trace_rows, action_rows = [{"kind": "trace_manifest", "sha256": artifacts.manifest_sha256, "bytes": manifest.stat().st_size}], None, []
        entry = payload.get("scenarios", {}).get(scenario["id"])
        if not isinstance(entry, Mapping):
            raise ReplayArchiveError("trace manifest lacks the requested scenario")
        for kind, count_name in (("trace", "record_count"), ("actions", "action_count")):
            desc = entry.get(kind)
            if kind == "actions" and desc is None:
                if require_actions:
                    raise ReplayArchiveError("required actions sidecar is missing")
                continue
            if not isinstance(desc, Mapping):
                raise ReplayArchiveError(f"trace manifest has no {kind} descriptor")
            if (getattr(artifacts, f"{kind}_path") != desc["path"]
                    or getattr(artifacts, f"{kind}_sha256") != desc["sha256"]):
                raise ReplayArchiveError(f"reported {kind} differs from its trace manifest")
            path = _verified(desc["path"], desc["sha256"], desc["size_bytes"], staging, kind)
            rows = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(rows, list) or len(rows) != desc[count_name]:
                raise ReplayArchiveError(f"{kind} row count mismatch")
            sidecars.append({"kind": kind, "sha256": desc["sha256"], "bytes": desc["size_bytes"]})
            if kind == "trace": trace_rows = rows
            else: action_rows = rows
        if trace_rows is None:
            raise ReplayArchiveError("trace sidecar is missing")
        if any(not isinstance(row, Mapping) or row.get("type") not in ACTION_COLUMNS
               for row in action_rows):
            raise ReplayArchiveError("actions sidecar contains a malformed or unknown action")
        arrays[f"trace_{index:03d}"] = _matrix(trace_rows, TRACE_COLUMNS, kind="trace")
        for kind, columns in ACTION_COLUMNS.items():
            arrays[f"{kind}_{index:03d}"] = _matrix([row for row in action_rows if isinstance(row, Mapping) and row.get("type") == kind], columns, kind=kind)
        records.append({"index": index, "id": scenario["id"], "evaluation_id": scenario["evaluation_id"], "economic_fingerprint": scenario["economic_fingerprint"], "row_counts": {"trace": len(trace_rows), **{kind: arrays[f"{kind}_{index:03d}"].shape[0] for kind in ACTION_COLUMNS}}, "source_sidecars": sidecars})
    npz_path = output / "replay_trace.npz"
    with tempfile.NamedTemporaryFile("wb", dir=output, prefix=".replay_trace.", suffix=".tmp", delete=False) as stream:
        temp = Path(stream.name)
        try:
            np.savez_compressed(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
        except BaseException:
            temp.unlink(missing_ok=True); raise
    os.replace(temp, npz_path)
    companion = {"schema_version": "curve_fx_replay_trace_v1", "source_run_id": source_run_id, "candidate_id": candidate_id, "ordinal": ordinal, "columns": {"trace": list(TRACE_COLUMNS), **{key: list(value) for key, value in ACTION_COLUMNS.items()}}, "scenarios": records, "npz": {"path": "replay_trace.npz", "sha256": sha256_path(npz_path), "bytes": npz_path.stat().st_size}}
    json_path = atomic_write_json(output / "replay_trace.json", companion)
    shutil.rmtree(staging)
    return npz_path, json_path, companion


__all__ = ["ACTION_COLUMNS", "TRACE_COLUMNS", "ReplayArchiveError", "pack_replay_archive"]
