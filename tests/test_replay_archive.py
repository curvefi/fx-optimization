from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from curve_fx_sim.plotting.trajectory import load_trajectory
from curve_fx_sim.shiftclick.archive import (
    ACTION_COLUMNS, TRACE_COLUMNS, ReplayArchiveError, pack_replay_archive,
)


def _write_bundle(root: Path, scenario: str, evaluation: str, fingerprint: str, *, tamper: bool = False):
    sidecars = root / "sidecars"
    sidecars.mkdir(parents=True, exist_ok=True)
    trace_row = {
        name: (None if name == "yb_growth" else (1 if name in {"t", "last_timestamp", "n_trades", "n_rebalances", "yb_initialized", "yb_releverage_trades"} else 1.25))
        for name in TRACE_COLUMNS
    }
    actions = [
        {"type": kind, **{name: (1 if name in {"ts", "ts_due", "freq_s", "i", "j"} else 2.5)
                          for name in columns}}
        for kind, columns in ACTION_COLUMNS.items()
    ]
    trace = sidecars / f"{scenario}.trace.json"
    action = sidecars / f"{scenario}.actions.json"
    trace.write_text(json.dumps([trace_row]), encoding="utf-8")
    action.write_text(json.dumps(actions), encoding="utf-8")
    trace_sha = hashlib.sha256(trace.read_bytes()).hexdigest()
    action_sha = hashlib.sha256(action.read_bytes()).hexdigest()
    manifest_payload = {
        "manifest_version": "curve_fx_trace_manifest_v1",
        "candidate_id": evaluation,
        "economic_fingerprint": fingerprint,
        "scenarios": {scenario: {
            "trace": {"path": f"sidecars/{trace.name}", "sha256": trace_sha,
                      "size_bytes": trace.stat().st_size, "record_count": 1},
            "actions": {"path": f"sidecars/{action.name}", "sha256": action_sha,
                        "size_bytes": action.stat().st_size, "action_count": len(actions)},
        }},
    }
    manifest = sidecars / f"{scenario}.manifest.json"
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if tamper:
        trace.write_text("[]", encoding="utf-8")
    artifacts = SimpleNamespace(
        trace_path=f"sidecars/{trace.name}", trace_sha256=trace_sha,
        actions_path=f"sidecars/{action.name}", actions_sha256=action_sha,
        manifest_path=f"sidecars/{manifest.name}", manifest_sha256=manifest_sha,
    )
    return {"id": scenario, "evaluation_id": evaluation,
            "economic_fingerprint": fingerprint, "artifacts": artifacts}


def test_candidate_wide_archive_roundtrip_and_tamper_rejection(tmp_path: Path) -> None:
    output = tmp_path / "published"
    staging = output / ".replay_staging"
    output.mkdir()
    scenarios = [_write_bundle(staging, f"scenario-{i}", f"eval-{i}", f"fp-{i}") for i in range(2)]
    npz_path, json_path, companion = pack_replay_archive(
        output, staging, source_run_id="source", candidate_id="candidate", ordinal=7,
        scenarios=scenarios, require_actions=True,
    )
    assert not staging.exists()
    assert companion["scenarios"][1]["evaluation_id"] == "eval-1"
    assert json.loads(json_path.read_text())["npz"]["path"] == npz_path.name
    assert load_trajectory(
        npz_path, companion_path=json_path, scenario_index=1).series("t") == (1.0,)
    with np.load(npz_path, allow_pickle=False) as archive:
        assert archive["trace_000"].dtype == np.dtype("<f8")
        assert archive["trace_000"].shape == (1, len(TRACE_COLUMNS))
        assert np.isnan(archive["trace_000"][0, TRACE_COLUMNS.index("yb_growth")])
        for kind, columns in ACTION_COLUMNS.items():
            assert archive[f"{kind}_001"].shape == (1, len(columns))

    bad_output = tmp_path / "bad"
    bad_staging = bad_output / ".replay_staging"
    bad_output.mkdir()
    bad = _write_bundle(bad_staging, "bad-scenario", "bad-eval", "bad-fp", tamper=True)
    with pytest.raises(ReplayArchiveError, match="byte size mismatch|SHA-256 mismatch"):
        pack_replay_archive(bad_output, bad_staging, source_run_id="source",
                            candidate_id="candidate", ordinal=0, scenarios=[bad], require_actions=True)
    assert not (bad_output / "replay_trace.npz").exists()
