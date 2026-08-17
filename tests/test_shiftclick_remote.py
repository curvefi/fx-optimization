"""One-blade shiftclick transport and receipt tests."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

import pytest

from curve_fx_sim.artifacts.io import sha256_path
from curve_fx_sim.artifacts.manifest import (
    new_grid_manifest,
    new_shiftclick_manifest,
    write_manifest_atomic,
)
from curve_fx_sim.artifacts.store import RunStore
from curve_fx_sim.artifacts.tables import EvaluationRow, EvaluationTable, MetricProjection
from curve_fx_sim.evaluation.selection import normalize_selection
from curve_fx_sim.execution.adapter import ProcessResult
from curve_fx_sim.execution.site import ClusterConfig, HarnessConfig, SiteProfile
from curve_fx_sim.shiftclick.remote import run_remote_shiftclick
from curve_fx_sim.shiftclick.runner import selection_from_spec
from curve_fx_sim.specs.pair import PairSpec
from curve_fx_sim.specs.scenario import ScenarioSpec
from curve_fx_sim.specs.shiftclick import ShiftclickSpec


def _core() -> dict[str, object]:
    return {
        "schema_version": "curve_fx_sim_identity_v2",
        "binary": "/srv/fx/bin/evaluator",
        "sha256": "a" * 64,
        "harness_version": "1.0.0",
        "pool_version": "0.1.0",
        "policy_id": "compiled_policy",
        "policy_source_sha256": "b" * 64,
        "policy_abi": "twocrypto_policy_v1",
        "policy_parameter_count": 1,
        "numeric_mode": "double",
        "real_type": "double",
        "compiler": "clang++",
        "build_target": "arb_evaluator_ld",
        "metric_schema": "twocrypto-summary-v1",
        "metric_fields": ["apy"],
    }


class FakeSSHAdapter:
    instances: list["FakeSSHAdapter"] = []
    resolved_spec: dict[str, object] = {}
    selection: dict[str, object] = {}
    source_run_id = "source_grid"
    fingerprint = "f" * 64

    def __init__(self, **_kwargs: object) -> None:
        self.calls: list[tuple[str, object, object]] = []
        self.instances.append(self)

    def run_ssh(self, blade: str, command: str, **_kwargs: object) -> ProcessResult:
        self.calls.append(("ssh", blade, command))
        return ProcessResult(0, "", "")

    def rsync_upload(self, local_path: Path, blade: str, remote_path: str) -> ProcessResult:
        self.calls.append(("upload", local_path, (blade, remote_path)))
        return ProcessResult(0, "", "")

    def rsync_download(self, blade: str, remote_path: str, local_path: Path) -> ProcessResult:
        self.calls.append(("download", blade, (remote_path, local_path)))
        run_dir = local_path / "shiftclick_exact"
        run_dir.mkdir()
        trace_path = run_dir / "trace.json"
        actions_path = run_dir / "actions.json"
        replay_path = run_dir / "replay_result.json"
        comparison_path = run_dir / "economic_comparison.json"
        trace_path.write_text("[]\n", encoding="utf-8")
        actions_path.write_text("[]\n", encoding="utf-8")
        replay_path.write_text(
            json.dumps(
                {
                    "ordinal": 0,
                    "candidate_id": "grid_p1",
                    "status": "ok",
                    "economic_fingerprint": self.fingerprint,
                    "metrics": {"apy": 0.01},
                }
            ),
            encoding="utf-8",
        )
        comparison_path.write_text(
            json.dumps(
                {
                    "fingerprint": self.fingerprint,
                    "metric_count": 1,
                    "relative_tolerance": 1e-12,
                    "absolute_tolerance": 0.0,
                    "max_absolute_error": 0.0,
                    "max_relative_error": 0.0,
                    "metrics": [
                        {
                            "field": "apy",
                            "expected": 0.01,
                            "observed": 0.01,
                            "absolute_error": 0.0,
                            "relative_error": 0.0,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        artifacts = [
            {
                "path": path.name,
                "kind": kind,
                "sha256": sha256_path(path),
                "bytes": path.stat().st_size,
            }
            for path, kind in (
                (trace_path, "trace"),
                (actions_path, "actions"),
                (replay_path, "replay_result"),
                (comparison_path, "economic_comparison"),
            )
        ]
        manifest = new_shiftclick_manifest(
            run_id="shiftclick_exact",
            shiftclick_id="exact",
            source_run_id=self.source_run_id,
            selection=self.selection,
            resolution="full",
            resolved_spec=self.resolved_spec,
            execution={"scope": "local"},
            core=_core(),
            artifacts=artifacts,
        )
        write_manifest_atomic(
            run_dir / "manifest.json",
            manifest,
            expected_kind="shiftclick",
        )
        return ProcessResult(0, "", "")


def _source_run(store: RunStore) -> None:
    run_dir = store.runs_dir / "source_grid"
    run_dir.mkdir()
    pair = PairSpec(
        id="chfusd",
        name="CHF/USD",
        base_token="CHF",
        quote_token="USD",
    )
    scenario = ScenarioSpec(
        id="chfusd-smoke",
        pair_id="chfusd",
        name="smoke",
        template_path=Path("template.json"),
        template_sha256="c" * 64,
    )
    projection = MetricProjection.from_fields(("apy",), projection_id="grid")
    table = EvaluationTable(
        [
            EvaluationRow(
                candidate_id="grid_p1",
                ordinal=0,
                coordinates={"weight": "0.5"},
                params={"vector": [0.5]},
                metrics={"apy": 0.01},
                economic_fingerprint=FakeSSHAdapter.fingerprint,
            )
        ],
        metric_projection=projection,
    )
    table_path = table.to_npz(run_dir / "evaluation_table.npz")
    table_ref = {
        "path": table_path.name,
        "sha256": sha256_path(table_path),
        "bytes": table_path.stat().st_size,
        "row_count": 1,
    }
    source_core = _core()
    source_core.update({"site": "blades", "remote_sha256": source_core["sha256"]})
    manifest = new_grid_manifest(
        run_id="source_grid",
        grid_id="source-grid",
        pool_count=1,
        resolved_spec={
            "pair": pair.to_dict(),
            "scenario": scenario.to_dict(),
            "policy": {"id": "compiled_policy"},
            "metric_projection": projection.to_dict(),
        },
        resolved_axes=[],
        pools=[
            {
                "id": "grid_p1",
                "ordinal": 0,
                "coordinate_indices": [0],
                "coordinates": {"weight": "0.5"},
                "policy_params": [0.5],
                "pool_overrides": {},
            }
        ],
        core=source_core,
        table_ref=table_ref,
        artifacts=[
            {
                "kind": "evaluation_table",
                "path": table_ref["path"],
                "sha256": table_ref["sha256"],
                "bytes": table_ref["bytes"],
            }
        ],
    )
    write_manifest_atomic(run_dir / "manifest.json", manifest, expected_kind="grid")


def _spec_and_remote_manifest(store: RunStore) -> ShiftclickSpec:
    spec = ShiftclickSpec(
        id="exact",
        source_kind="grid",
        source_run_id="source_grid",
        selection_kind="candidate_id",
        selection_value="grid_p1",
        pair_id="chfusd",
        scenario_id="chfusd-smoke",
        policy_id="compiled_policy",
    )
    selection = selection_from_spec(spec)
    plan = normalize_selection(selection, store=store)
    source = store.load_manifest("source_grid")
    FakeSSHAdapter.selection = selection.to_dict()
    FakeSSHAdapter.resolved_spec = {
        "shiftclick": spec.to_dict(),
        "replay_plan": plan.to_dict(),
        "metric_projection": source["resolved_spec"]["metric_projection"],
    }
    FakeSSHAdapter.source_run_id = "source_grid"
    return spec


def _site() -> SiteProfile:
    return SiteProfile(
        name="blades",
        site_type="ssh",
        cluster=ClusterConfig(
            remote_base=PurePosixPath("/srv/fx"),
            repository_root=PurePosixPath("/srv/fx/curve-fx-optimization"),
            worker_command="/srv/fx/bin/fxsim-worker",
        ),
        harness=HarnessConfig(
            remote_binary_path=PurePosixPath("/srv/fx/bin/evaluator")
        ),
    )


def test_remote_shiftclick_stages_source_and_persists_cluster_receipt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    store = RunStore(tmp_path)
    _source_run(store)
    spec = _spec_and_remote_manifest(store)
    spec_path = tmp_path / "shiftclick.toml"
    spec_path.write_text("[shiftclick]\nid='exact'\n", encoding="utf-8")
    FakeSSHAdapter.instances.clear()
    monkeypatch.setattr("curve_fx_sim.shiftclick.remote.SSHProcessAdapter", FakeSSHAdapter)

    result = run_remote_shiftclick(
        spec,
        spec_path=spec_path,
        store=store,
        site=_site(),
        blade="blade-a1",
    )

    assert result.blade == "blade-a1"
    assert result.run_dir == store.runs_dir / "shiftclick_exact"
    assert result.manifest["shiftclick"]["source_run_id"] == "source_grid"
    assert result.manifest["shiftclick"]["execution"] == {
        "scope": "cluster",
        "site": "blades",
        "blade": "blade-a1",
        "remote_workspace": "/srv/fx/runs/shiftclick_exact/replay_workspace",
    }
    calls = FakeSSHAdapter.instances[0].calls
    commands = [str(call[2]) for call in calls if call[0] == "ssh"]
    assert any("fxsim-worker replay shiftclick" in command for command in commands)
    assert any("--harness /srv/fx/bin/evaluator" in command for command in commands)
    uploads = [call for call in calls if call[0] == "upload"]
    assert any(call[1] == store.runs_dir / "source_grid" for call in uploads)
    assert any(call[1] == spec_path.resolve() for call in uploads)


def test_remote_shiftclick_rejects_wrong_source_receipt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    store = RunStore(tmp_path)
    _source_run(store)
    spec = _spec_and_remote_manifest(store)
    spec_path = tmp_path / "shiftclick.toml"
    spec_path.write_text("[shiftclick]\nid='exact'\n", encoding="utf-8")
    FakeSSHAdapter.source_run_id = "wrong_source"
    FakeSSHAdapter.instances.clear()
    monkeypatch.setattr("curve_fx_sim.shiftclick.remote.SSHProcessAdapter", FakeSSHAdapter)

    with pytest.raises(RuntimeError, match="source_run_id"):
        run_remote_shiftclick(
            spec,
            spec_path=spec_path,
            store=store,
            site=_site(),
            blade="blade-a1",
        )
