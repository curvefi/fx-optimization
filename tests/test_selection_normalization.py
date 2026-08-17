"""Tests for grid/optimizer selection normalization into canonical ReplayPlans."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from curve_fx_sim.artifacts.attestation import load_attested_evaluation_table
from curve_fx_sim.artifacts.io import sha256_path
from curve_fx_sim.artifacts.manifest import (
    ManifestError,
    new_grid_manifest,
    new_optimization_manifest,
)
from curve_fx_sim.artifacts.store import RunStore
from curve_fx_sim.artifacts.tables import (
    EvaluationRow,
    EvaluationTable,
)
from curve_fx_sim.evaluation.selection import (
    ReplayPlan,
    SelectionRef,
    normalize_selection,
)
from curve_fx_sim.specs.common import SpecError
from curve_fx_sim.specs.pair import PairSpec
from curve_fx_sim.specs.scenario import ScenarioSpec


def _core() -> dict[str, object]:
    return {
        "schema_version": "curve_fx_sim_identity_v2",
        "binary": "arb_evaluator_ld",
        "sha256": "a" * 64,
        "harness_version": "1.0.0",
        "pool_version": "0.1.0",
        "policy_id": "policy_v1",
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


def _attest_table(
    run_store: RunStore,
    run_id: str,
    table: EvaluationTable,
    *,
    run_kind: str,
) -> Path:
    table_path = run_store.save_evaluation_table(run_id, table)
    manifest = run_store.load_manifest(run_id, expected_kind=run_kind)
    manifest[run_kind]["table_ref"] = {
        "path": table_path.relative_to(table_path.parent).as_posix(),
        "sha256": sha256_path(table_path),
        "bytes": table_path.stat().st_size,
        "row_count": len(table),
    }
    run_store.save_manifest(run_id, manifest, expected_kind=run_kind)
    return table_path


def test_normalize_selection_grid_point(
    run_store: RunStore, sample_pair: PairSpec, sample_scenario: ScenarioSpec
) -> None:
    run_id = "grid_run_test_01"
    run_store.save_manifest(
        run_id,
        new_grid_manifest(
            run_id=run_id,
            grid_id="test_grid",
            pool_count=2,
            resolved_spec={
                "pair": sample_pair.to_dict(),
                "scenario": sample_scenario.to_dict(),
                "policy": {"id": "policy_v1"},
            },
            resolved_axes=[{"name": "mid_fee", "values": ["0.0001", "0.0005"]}],
            pools=[
                {
                    "id": "grid_p00",
                    "ordinal": 0,
                    "coordinates": {"weight": "0.1"},
                    "policy_params": [0.1],
                    "pool_overrides": {},
                },
                {
                    "id": "grid_p01",
                    "ordinal": 1,
                    "coordinates": {"weight": "0.5"},
                    "policy_params": [0.5],
                    "pool_overrides": {},
                },
            ],
            core=_core(),
        ),
        expected_kind="grid",
    )
    table = EvaluationTable([
        EvaluationRow(
            candidate_id="grid_p00",
            ordinal=0,
            coordinates={"weight": "0.1"},
            params={"vector": [0.1]},
            pool_overrides={},
            status="ok",
            economic_fingerprint="fp_grid_0",
        ),
        EvaluationRow(
            candidate_id="grid_p01",
            ordinal=1,
            coordinates={"weight": "0.5"},
            params={"vector": [0.5]},
            pool_overrides={},
            status="ok",
            economic_fingerprint="fp_grid_1",
        ),
    ])
    _attest_table(run_store, run_id, table, run_kind="grid")

    # 1. Select by ordinal index
    sel_idx = SelectionRef(run_id=run_id, kind="grid_point", index=1)
    plan_idx = normalize_selection(sel_idx, store=run_store)
    assert isinstance(plan_idx, ReplayPlan)
    assert plan_idx.policy_params == {"vector": [0.5]}
    assert plan_idx.pool_overrides == {}
    assert plan_idx.economic_fingerprint == "fp_grid_1"

    # 2. Select the same exact decimal coordinate across TOML/JSON scalar types.
    sel_coord = SelectionRef(run_id=run_id, kind="grid_point", coordinate={"weight": 0.1})
    plan_coord = normalize_selection(sel_coord, store=run_store)
    assert plan_coord.policy_params == {"vector": [0.1]}
    assert plan_coord.economic_fingerprint == "fp_grid_0"

    # 3. Missing index raises KeyError
    with pytest.raises(KeyError, match="grid point ordinal 99"):
        normalize_selection(
            SelectionRef(run_id=run_id, kind="grid_point", index=99),
            store=run_store,
        )


def test_normalize_selection_optimizer_winner(
    run_store: RunStore, sample_pair: PairSpec, sample_scenario: ScenarioSpec
) -> None:
    run_id = "opt_run_test_01"
    run_store.save_manifest(
        run_id,
        new_optimization_manifest(
            run_id=run_id,
            optimization_id="opt_arch",
            algorithm="tmrbcd",
            scenarios=["scenario_jan"],
            resolved_spec={
                "pair": sample_pair.to_dict(),
                "scenario": sample_scenario.to_dict(),
                "policy": {"id": "policy_v2"},
            },
            candidates_evaluated=10,
            best_candidate={
                "candidate_id": "cand_winner",
                "params": {"weight": 1.25},
                "pool_overrides": {},
                "economic_fingerprint": "fp_winner",
            },
            core=_core(),
        ),
        expected_kind="optimization",
    )
    table_path = _attest_table(
        run_store,
        run_id,
        EvaluationTable(
            [
                EvaluationRow(
                    candidate_id="cand_winner",
                    params={"weight": 1.25},
                    pool_overrides={},
                    economic_fingerprint="fp_winner",
                )
            ]
        ),
        run_kind="optimization",
    )

    sel = SelectionRef(run_id=run_id, kind="optimizer_winner")
    plan = normalize_selection(sel, store=run_store)
    assert plan.policy_params == {"weight": 1.25}
    assert plan.pool_overrides == {}
    assert plan.economic_fingerprint == "fp_winner"

    raw = table_path.read_bytes()
    table_path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 0xFF]))
    # Lean loads skip digest re-verification (no full-file read on selection);
    # the explicit verify path still catches size-preserving tampering.
    plan = normalize_selection(sel, store=run_store)
    assert plan.economic_fingerprint == "fp_winner"
    manifest = run_store.load_manifest(run_id)
    with pytest.raises(SpecError, match="SHA-256"):
        load_attested_evaluation_table(
            manifest, run_dir=run_store.get_run_dir(run_id), verify_digest=True
        )


def _save_single_point_grid(
    run_store: RunStore,
    sample_pair: PairSpec,
    sample_scenario: ScenarioSpec,
    *,
    run_id: str,
    row: EvaluationRow,
) -> Path:
    run_store.save_manifest(
        run_id,
        new_grid_manifest(
            run_id=run_id,
            grid_id="strict_grid",
            pool_count=1,
            resolved_spec={
                "pair": sample_pair.to_dict(),
                "scenario": sample_scenario.to_dict(),
                "policy": {"id": "policy_v1"},
            },
            resolved_axes=[{"name": "weight", "values": ["0.1"]}],
            pools=[
                {
                    "id": "grid_p00",
                    "ordinal": 0,
                    "coordinates": {"weight": "0.1"},
                    "policy_params": [0.1],
                    "pool_overrides": {},
                }
            ],
            core=_core(),
        ),
        expected_kind="grid",
    )
    return _attest_table(
        run_store,
        run_id,
        EvaluationTable([row]),
        run_kind="grid",
    )


def test_grid_replay_rejects_tampered_attested_table(
    run_store: RunStore,
    sample_pair: PairSpec,
    sample_scenario: ScenarioSpec,
) -> None:
    run_id = "grid_tampered_table"
    table_path = _save_single_point_grid(
        run_store,
        sample_pair,
        sample_scenario,
        run_id=run_id,
        row=EvaluationRow(
            candidate_id="grid_p00",
            ordinal=0,
            coordinates={"weight": "0.1"},
            params={"vector": [0.1]},
            economic_fingerprint="fp_grid_0",
        ),
    )
    raw = table_path.read_bytes()
    table_path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 0xFF]))

    # Lean loads skip digest re-verification; selection still resolves.
    plan = normalize_selection(
        SelectionRef(run_id=run_id, kind="grid_point", index=0),
        store=run_store,
    )
    assert plan.economic_fingerprint == "fp_grid_0"
    # The explicit verify path catches the size-preserving tamper.
    manifest = run_store.load_manifest(run_id)
    with pytest.raises(SpecError, match="SHA-256"):
        load_attested_evaluation_table(
            manifest, run_dir=run_store.get_run_dir(run_id), verify_digest=True
        )


@pytest.mark.parametrize(
    ("run_id", "row", "message"),
    [
        (
            "grid_wrong_id",
            EvaluationRow(
                candidate_id="foreign",
                ordinal=0,
                coordinates={"weight": "0.1"},
                params={"vector": [0.1]},
            ),
            "exactly one canonical pool",
        ),
        (
            "grid_wrong_ordinal",
            EvaluationRow(
                candidate_id="grid_p00",
                ordinal=1,
                coordinates={"weight": "0.1"},
                params={"vector": [0.1]},
            ),
            "ordinal does not match",
        ),
        (
            "grid_wrong_coordinate",
            EvaluationRow(
                candidate_id="grid_p00",
                ordinal=0,
                coordinates={"weight": "0.2"},
                params={"vector": [0.1]},
            ),
            "coordinates does not match",
        ),
        (
            "grid_wrong_vector",
            EvaluationRow(
                candidate_id="grid_p00",
                ordinal=0,
                coordinates={"weight": "0.1"},
                params={"vector": [0.2]},
            ),
            "policy vector does not match",
        ),
        (
            "grid_wrong_overrides",
            EvaluationRow(
                candidate_id="grid_p00",
                ordinal=0,
                coordinates={"weight": "0.1"},
                params={"vector": [0.1]},
                pool_overrides={"A": 100},
            ),
            "pool overrides does not match",
        ),
    ],
)
def test_grid_replay_rejects_row_that_differs_from_manifest_pool(
    run_store: RunStore,
    sample_pair: PairSpec,
    sample_scenario: ScenarioSpec,
    run_id: str,
    row: EvaluationRow,
    message: str,
) -> None:
    _save_single_point_grid(
        run_store,
        sample_pair,
        sample_scenario,
        run_id=run_id,
        row=row,
    )

    with pytest.raises(SpecError, match=message):
        normalize_selection(
            SelectionRef(run_id=run_id, kind="grid_point", candidate_id=row.candidate_id),
            store=run_store,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("bytes", 1, "byte size"),
        ("row_count", 2, "row count"),
        ("path", "../evaluation_table.json", "escapes the run directory"),
    ],
)
def test_grid_replay_rejects_invalid_table_reference(
    run_store: RunStore,
    sample_pair: PairSpec,
    sample_scenario: ScenarioSpec,
    field: str,
    value: object,
    message: str,
) -> None:
    run_id = f"grid_bad_ref_{field}"
    _save_single_point_grid(
        run_store,
        sample_pair,
        sample_scenario,
        run_id=run_id,
        row=EvaluationRow(
            candidate_id="grid_p00",
            ordinal=0,
            coordinates={"weight": "0.1"},
            params={"vector": [0.1]},
        ),
    )
    manifest = run_store.load_manifest(run_id, expected_kind="grid")
    manifest["grid"]["table_ref"][field] = value
    if field == "path":
        with pytest.raises(ManifestError, match="stay within the run directory"):
            run_store.save_manifest(run_id, manifest, expected_kind="grid")
        return
    run_store.save_manifest(run_id, manifest, expected_kind="grid")

    with pytest.raises(SpecError, match=message):
        normalize_selection(
            SelectionRef(run_id=run_id, kind="grid_point", index=0),
            store=run_store,
        )


def test_normalize_selection_fails_without_scenario(run_store: RunStore, sample_pair: PairSpec) -> None:
    run_id = "opt_no_scenario"
    run_store.save_manifest(
        run_id,
        new_optimization_manifest(
            run_id=run_id,
            optimization_id="opt_bare",
            algorithm="tmrbcd",
            scenarios=["s1"],
            resolved_spec={"pair": sample_pair.to_dict()},
            core=_core(),
        ),
        expected_kind="optimization",
    )

    sel = SelectionRef(run_id=run_id, kind="grid_point", index=0)
    with pytest.raises(SpecError, match="cannot resolve scenario"):
        normalize_selection(sel, store=run_store)
