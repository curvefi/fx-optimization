"""Focused checks for optimizer-owned grid result metadata."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

import curve_fx_sim.grids.model as grid_model
from curve_fx_sim.artifacts.store import RunStore
from curve_fx_sim.artifacts.tables import MetricProjection
from curve_fx_sim.execution.collection import write_grid_results_npz
from curve_fx_sim.grids.collection import GridCoverageError
from curve_fx_sim.grids.model import GridValidationError, expand_grid
from curve_fx_sim.grids.runner import _load_result_records, compile_grid_run
from curve_fx_sim.specs.grid import AxisSpec, AxisTarget, GridSpec
from curve_fx_sim.specs.pair import PairSpec
from curve_fx_sim.specs.scenario import ScenarioSpec

def _write_result(path: Path, *, pool_index: int) -> None:
    write_grid_results_npz(
        path,
        run_id="test",
        schema_signature='{"fields":["apy"]}',
        request_set_sha256="a" * 64,
        session_attestation={
            "scenario_set_sha256": "b" * 64,
            "session_fingerprint": "c" * 64,
            "session_config_sha256": "d" * 64,
            "metric_schema_sha256": "e" * 64,
        },
        rows=[
            {
                "pool_index": pool_index,
                "ordinal": 0,
                "candidate_id": "candidate_0",
                "status": "ok",
                "economic_fingerprint": "f" * 64,
                "metrics": {"apy": 0.01},
            }
        ],
    )


def test_grid_result_loader_strips_matching_optimizer_pool_index(tmp_path: Path) -> None:
    result_path = tmp_path / "grid_results.npz"
    _write_result(result_path, pool_index=0)

    records = _load_result_records(result_path)

    assert len(records) == 1
    assert records[0].ordinal == 0
    assert records[0].candidate_id == "candidate_0"


def test_grid_result_loader_rejects_mismatched_optimizer_pool_index(tmp_path: Path) -> None:
    result_path = tmp_path / "grid_results.npz"
    _write_result(result_path, pool_index=1)

    with pytest.raises(GridCoverageError, match="does not match evaluator ordinal"):
        _load_result_records(result_path)


def test_grid_points_detach_nested_static_overrides() -> None:
    grid = GridSpec(
        id="nested",
        pair_id="pair",
        axes=(
            AxisSpec(
                name="weight",
                values=(Decimal("1"), Decimal("2")),
                targets=(AxisTarget(path=("nested", "changed", "value"), kind="integer"),),
            ),
        ),
        static_overrides={"nested": {"stable": {"items": [1]}, "changed": {"baseline": [9]}}},
    )

    points = expand_grid(grid)

    points[0].pool_overrides["nested"]["stable"]["items"].append(2)
    points[0].pool_overrides["nested"]["changed"]["baseline"].append(10)
    assert points[1].pool_overrides["nested"]["stable"]["items"] == [1]
    assert points[1].pool_overrides["nested"]["changed"]["baseline"] == [9]


def test_grid_coordinate_signature_is_computed_once_per_point(monkeypatch: pytest.MonkeyPatch) -> None:
    original = grid_model.coordinate_signature
    calls: list[dict[str, object]] = []

    def counted(coordinate: dict[str, object]) -> str:
        calls.append(coordinate)
        return original(coordinate)

    monkeypatch.setattr(grid_model, "coordinate_signature", counted)
    grid = GridSpec(
        id="signature",
        pair_id="pair",
        axes=(
            AxisSpec(name="x", values=(Decimal("1"), Decimal("2"))),
            AxisSpec(name="y", values=(Decimal("3"), Decimal("4"))),
        ),
    )

    points = expand_grid(grid)

    assert len(calls) == grid.pool_count
    assert [point.coordinate_signature for point in points] == [
        original(point.coordinates) for point in points
    ]



def test_fee_equalize_mirrors_mid_fee_per_cell() -> None:
    grid = GridSpec(
        id="fee_eq",
        pair_id="pair",
        axes=(
            AxisSpec(
                name="mid_fee_bps",
                values=(Decimal("1"), Decimal("5"), Decimal("10")),
                targets=(AxisTarget(path=("mid_fee",), scale=Decimal("1000000"), kind="decimal"),),
            ),
        ),
        fee_equalize=True,
    )

    points = expand_grid(grid)

    assert [point.pool_overrides for point in points] == [
        {"mid_fee": 1000000, "out_fee": 1000000},
        {"mid_fee": 5000000, "out_fee": 5000000},
        {"mid_fee": 10000000, "out_fee": 10000000},
    ]
    assert grid.to_dict()["fee_equalize"] is True


def test_fee_equalize_rejects_independent_out_fee() -> None:
    grid = GridSpec(
        id="fee_eq_conflict",
        pair_id="pair",
        axes=(
            AxisSpec(
                name="mid_fee_bps",
                values=(Decimal("1"),),
                targets=(AxisTarget(path=("mid_fee",), scale=Decimal("1000000"), kind="decimal"),),
            ),
            AxisSpec(
                name="out_fee_bps",
                values=(Decimal("2"),),
                targets=(AxisTarget(path=("out_fee",), scale=Decimal("1000000"), kind="decimal"),),
            ),
        ),
        fee_equalize=True,
    )

    with pytest.raises(GridValidationError, match="may not override out_fee independently"):
        expand_grid(grid)


def test_native_grid_compiles_pool_overrides_without_policy(tmp_path: Path) -> None:
    grid = GridSpec(
        id="native_pool",
        pair_id="yb-weth",
        axes=(
            AxisSpec(
                name="A",
                values=(Decimal("1"), Decimal("2")),
                targets=(AxisTarget(path=("A",), kind="integer"),),
            ),
        ),
        static_overrides={"mid_fee": "0.001"},
    )
    identity = {
        "schema_version": "curve_fx_sim_identity_v2",
        "binary": "arb_evaluator_ld",
        "sha256": "a" * 64,
        "harness_version": "1.0.0",
        "pool_version": "1.0.0",
        "policy_id": "twocrypto_native",
        "policy_source_sha256": "none",
        "policy_abi": "twocrypto_policy_v1",
        "policy_parameter_count": 0,
        "numeric_mode": "longdouble",
        "real_type": "long double",
        "compiler": "clang",
        "build_target": "arb_evaluator_ld",
        "metric_schema": "twocrypto-summary-v1",
        "metric_fields": ["apy"],
    }

    compilation = compile_grid_run(
        grid,
        run_id="native_pool",
        pair_spec=PairSpec("yb-weth", "YB/WETH", "YB", "WETH"),
        scenario_spec=ScenarioSpec("eth", "yb-weth", "ETH"),
        policy_spec=None,
        store=RunStore(root=tmp_path),
        metric_projection=MetricProjection.from_fields(["apy"]),
        evaluator_identity=identity,
    )

    assert [point.policy_params for point in compilation.points] == [(), ()]
    assert [point.pool_overrides for point in compilation.points] == [
        {"mid_fee": "0.001", "A": 1},
        {"mid_fee": "0.001", "A": 2},
    ]
    assert compilation.manifest["resolved_spec"]["policy"] == {
        "id": "twocrypto_native",
        "policy_kind": "native",
        "parameters": [],
    }