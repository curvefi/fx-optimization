"""Tests for the NPZ evaluation table and metric projections."""

from __future__ import annotations

from pathlib import Path

from curve_fx_sim.artifacts.tables import EvaluationRow, EvaluationTable, MetricProjection


def test_unordered_row_canonical_sort() -> None:
    rows = [
        EvaluationRow(candidate_id="cand_003", ordinal=2, params={"A": 30}, metrics={"apy": 0.3}),
        EvaluationRow(candidate_id="cand_000", ordinal=99, status="failed"),
        EvaluationRow(candidate_id="cand_001", ordinal=0, params={"A": 10}, metrics={"apy": 0.1}),
    ]
    table = EvaluationTable(rows).sort_canonical()
    assert [row.candidate_id for row in table] == ["cand_001", "cand_003", "cand_000"]


def test_metric_projection() -> None:
    projection = MetricProjection.from_fields(["apy", "vp"], projection_id="summary_small")
    assert projection.project_metrics({"apy": 0.15, "vp": 1.02, "unwanted": 999}) == {"apy": 0.15, "vp": 1.02}
    assert len(projection.projection_sha256) == 64


def test_table_npz_roundtrip(tmp_path: Path) -> None:
    table = EvaluationTable([
        EvaluationRow(candidate_id="c1", ordinal=0, params={"A": 100}, metrics={"apy": 0.12}),
        EvaluationRow(candidate_id="c2", ordinal=1, params={"A": 200}, metrics={"apy": 0.15}),
    ], metadata={"tag": "demo"})
    path = tmp_path / "table.npz"
    table.to_npz(path)
    loaded = EvaluationTable.from_npz(path)
    assert loaded.to_dict() == table.to_dict()
