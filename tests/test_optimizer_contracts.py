from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

os.environ["MPLBACKEND"] = "Agg"

from curve_fx_sim.plotting.heatmap import (
    HeatmapAxis,
    HeatmapDataset,
    HeatmapValidationError,
    MaskSpec,
)
from curve_fx_sim.plotting.masked_metrics import masked_metric_slippage_sources
from curve_fx_sim.plotting.shiftclick_view import ROLLING_APY_WINDOW_S, _rolling_window_growth_apy
from fxopt import Candidate, EvaluatorSession
from fxopt.cli import main
from fxopt.config import ConfigError
from fxopt.engine import ProjectedBatch
from fxopt.results import GridResultWriter, merge_grid_partitions, read_result_columns
from fxopt.config import RunConfig
from fxopt.run import run_config, run_metadata, run_leased_worker


def _run_toml(
    path: Path,
    *,
    policy_params: str = "[]",
    compiled_policy: str = "",
    session: str = "",
    yb_mode: str = "off",
    price_feed: str = "",
    metrics: str = '["score"]',
    axes: str = "",
) -> Path:
    path.write_text(
        f'''[run]
id = "contract"
evaluator = "evaluator"
template = "template.json"
batch_size = 2
workers = 1
metric_fields = {metrics}
{session}
[scenario]
id = "scenario"
market = "market.json"
{price_feed}
yb_mode = "{yb_mode}"
{compiled_policy}
[candidate.defaults]
policy_params = {policy_params}
pool = {{}}
{axes}
'''
    )
    return path


def test_config_admission_table_covers_native_compiled_and_profiles(tmp_path: Path) -> None:
    cases = (
        ("native", {}, None),
        (
            "native-policy-rejected",
            {"policy_params": "[0.5]"},
            "policy_params must be empty",
        ),
        (
            "compiled",
            {
                "policy_params": "[0.5, 0.0003]",
                "compiled_policy": '[compiled_policy]\nheader = "policy.hpp"\nid = "compiled"',
            },
            None,
        ),
        (
            "price-feed",
            {"price_feed": 'price_feed = "nav.csv"'},
            None,
        ),
        (
            "exact-skip-profile",
            {"session": '[session]\nevent_cursor = "exact_skip"\nmetric_profile = "full_summary"'},
            "exact_skip requires metric_profile='grid_core'",
        ),
        (
            "singleton-mismatch",
            {"axes": '[candidate.axes]\n"pool.A" = {start = 1, stop = 2, count = 1}'},
            "pool.A count=1 requires start and stop to match",
        ),
        (
            "log-singleton-nonpositive",
            {"axes": '[candidate.axes]\n"pool.A" = {start = 0, stop = 0, count = 1, scale = "log"}'},
            "pool.A logarithmic endpoints must be positive",
        ),
        (
            "singleton-equal",
            {"axes": '[candidate.axes]\n"pool.A" = {start = 2, stop = 2, count = 1}'},
            None,
        ),
        (
            "grid-core-admission",
            {"session": '[session]\nmetric_profile = "grid_core"', "yb_mode": "active_2l"},
            "grid_core requires yb_mode='off' and slippage disabled",
        ),
        (
            "full-summary-yb-slippage",
            {
                "session": (
                    '[session]\nevent_cursor = "scalar"\n'
                    'metric_profile = "full_summary"\n'
                    "enable_slippage_probes = true"
                ),
                "yb_mode": "active_2l",
                "metrics": '["tw_real_slippage_1pct"]',
            },
            None,
        ),
    )
    for name, values, error in cases:
        config_path = _run_toml(tmp_path / f"{name}.toml", **values)
        if error is not None:
            with pytest.raises(ConfigError, match=error):
                RunConfig.from_toml(config_path)
            continue
        config = RunConfig.from_toml(config_path)
        if name == "compiled":
            assert config.compiled_policy_id == "compiled"
            assert run_metadata(config, effective_batch=2)["expected_evaluator_policy"] == {
                "policy_id": "compiled",
                "policy_abi": "twocrypto_policy_v1",
                "policy_parameter_count": 2,
            }
        else:
            assert config.compiled_policy_header is None
        if name == "full-summary-yb-slippage":
            assert config.scenario["yb_mode"] == "active_2l"
            assert config.session["enable_slippage_probes"] is True
        if name == "price-feed":
            metadata = run_metadata(config, effective_batch=2)
            assert metadata["open_session"]["price_feed_path"].endswith("nav.csv")


class _GridClient:
    def __init__(self) -> None:
        self.registered: list[dict[str, object]] = []
        self.requests: list[dict[str, object]] = []
        self.evaluations = 0

    def start(self) -> dict[str, bool]:
        return {"hello": True}

    def open_session(self, session_id: str, **request: object) -> None:
        self.session_id = session_id
        self.open_request = request

    def register_grid(self, grid_id: str, grid: dict[str, object], **request: object) -> dict[str, int]:
        self.registered.append(grid)
        count = int(np.prod(grid["shape"]))
        return {"candidate_count": count}

    def evaluate_batch(self, candidates: list[dict[str, object]], **request: object) -> dict[str, object]:
        self.evaluations += 1
        self.requests.append(request)
        ordinals = [
            ordinal
            for start, count in request["ranges"]
            for ordinal in range(int(start), int(start) + int(count))
        ]
        return {
            "metric_fields": request["metric_fields"],
            "results": [
                {
                    "ordinal": ordinal,
                    "candidate_id": f"p{ordinal:08d}",
                    "status": "ok",
                    "metrics": [float(ordinal)],
                }
                for ordinal in ordinals
            ],
        }

    def close_session(self, session_id: str | None = None) -> None:
        pass

    def shutdown(self) -> None:
        pass


def test_registered_grid_run_publishes_two_files_and_reconstructs_canonical_ordinals(
    tmp_path: Path,
) -> None:
    config = _run_toml(
        tmp_path / "run.toml",
        policy_params="[0.5]",
        compiled_policy='[compiled_policy]\nheader = "policy.hpp"\nid = "compiled"',
        axes='[candidate.axes]\n"pool.A" = [10, 20]\n"pool.donation_apy" = [0.0, 0.1]',
    )
    client = _GridClient()
    output = tmp_path / "run"
    paths = run_config(config, output, client_factory=lambda: client)

    assert paths.run_json.name == "run.json"
    assert paths.results_npz.name == "results.npz"
    assert {path.name for path in output.iterdir()} == {"run.json", "results.npz"}
    assert client.registered[0]["shape"] == [2, 2]
    columns = read_result_columns(output)
    assert columns.ordinals.tolist() == [0, 1, 2, 3]
    assert columns.candidate_at(3).candidate_id == "p00000003"
    assert columns.candidate_at(3).pool_overrides["A"] == 20
    assert columns.metrics["score"].tolist() == [0.0, 1.0, 2.0, 3.0]
    manifest = json.loads(paths.run_json.read_text())
    assert manifest["metadata"]["expected_evaluator_policy"] == {
        "policy_id": "compiled",
        "policy_abi": "twocrypto_policy_v1",
        "policy_parameter_count": 1,
    }

def _write_partition(path: Path, ordinals: tuple[int, ...]) -> Path:
    writer = GridResultWriter(
        path,
        run_id="partitioned",
        total=4,
        expected_count=len(ordinals),
        metadata={"worker": path.name},
        metric_names=("score",),
    )
    writer.append_projected(
        ordinals,
        ProjectedBatch(
            ("score",),
            tuple(
                {
                    "candidate_id": f"p{ordinal:08d}",
                    "status": "ok",
                    "metrics": [float(ordinal)],
                }
                for ordinal in ordinals
            ),
        ),
    )
    writer.finalize_partition()
    return path


def test_out_of_order_partitions_merge_to_integral_artifact_and_reject_overlap(
    tmp_path: Path,
) -> None:
    first = _write_partition(tmp_path / "worker-a", (3, 0))
    second = _write_partition(tmp_path / "worker-b", (2, 1))
    metadata = {
        "candidate_defaults": {"policy_params": [], "pool": {}},
        "axes": {"pool.A": [1, 2, 3, 4]},
        "shape": [4],
    }
    output = tmp_path / "merged"
    merge_grid_partitions(
        output,
        (first, second),
        run_id="partitioned",
        total=4,
        metadata=metadata,
        metric_names=("score",),
    )
    columns = read_result_columns(output)
    assert columns.ordinals.tolist() == [0, 1, 2, 3]
    assert columns.metrics["score"].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert {path.name for path in output.iterdir()} == {"run.json", "results.npz"}

    with pytest.raises(ValueError, match="overlaps"):
        merge_grid_partitions(
            tmp_path / "overlap",
            (first, first),
            run_id="partitioned",
            total=4,
            metadata=metadata,
            metric_names=("score",),
        )
    partial = tmp_path / "partial"
    merge_grid_partitions(
        partial, (first,), run_id="partitioned", total=4,
        metadata=metadata, metric_names=("score",),
    )
    columns = read_result_columns(partial)
    assert columns.ok_mask.tolist() == [True, False, False, True]
    assert np.isnan(columns.metrics["score"][[1, 2]]).all()
    assert columns.status_at(1) == "uncalculated"
    assert columns.candidate_at(3).pool_overrides["A"] == 4
    with pytest.raises(ValueError, match="not calculated"):
        columns.candidate_at(1)


def test_worker_keeps_completed_rows_after_evaluator_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Client(_GridClient):
        def evaluate_batch(self, candidates, **request):
            if request["ranges"][0][0] >= 2:
                raise RuntimeError("evaluator disconnected")
            return super().evaluate_batch(candidates, **request)

    monkeypatch.setattr("fxopt.run.local_client_factory", lambda *args, **kwargs: Client)
    config = _run_toml(
        tmp_path / "run.toml", axes='[candidate.axes]\n"pool.A" = [1, 2, 3, 4]',
    )
    partition = tmp_path / "worker"
    receipt = run_leased_worker(
        config, partition, worker_index=0,
        commands=({"type": "lease", "lease_id": 0}, {"type": "lease", "lease_id": 1}),
    )
    assert receipt["count"] == 2 and receipt["error"]
    output = tmp_path / "partial-worker"
    merge_grid_partitions(
        output, (partition,), run_id="contract", total=4,
        metadata=run_metadata(RunConfig.from_toml(config), effective_batch=2),
        metric_names=("score",),
    )
    columns = read_result_columns(output)
    assert columns.ok_mask.tolist() == [True, True, False, False]
    np.testing.assert_equal(columns.metrics["score"], [0.0, 1.0, np.nan, np.nan])


class _PointClient:
    def __init__(self, *, fields: list[str] | None = None) -> None:
        self.fields = fields or ["score", "apy"]
        self.requests: list[dict[str, object]] = []

    def start(self) -> None:
        pass

    def open_session(self, session_id: str, **request: object) -> None:
        pass

    def register_grid(self, grid_id: str, grid: object, **request: object) -> None:
        pass

    def evaluate_batch(self, candidates: list[dict[str, object]], **request: object) -> dict[str, object]:
        self.requests.append(request)
        return {
            "metric_fields": self.fields,
            "results": [
                {"candidate_id": item["candidate_id"], "metrics": [2.0, 0.1]}
                for item in reversed(candidates)
            ],
        }

    def close_session(self, session_id: str | None = None) -> None:
        pass

    def shutdown(self) -> None:
        pass


def test_point_batch_preserves_candidate_identity_and_metric_schema() -> None:
    client = _PointClient()
    with EvaluatorSession(
        lambda: client,
        session_id="points",
        metric_fields=("score", "apy"),
    ) as session:
        results = session.evaluate((Candidate("b", [2]), Candidate("a", [1])))

    assert [result.candidate_id for result in results] == ["b", "a"]
    assert [result.ordinal for result in results] == [0, 1]
    assert [dict(result.metrics) for result in results] == [
        {"score": 2.0, "apy": 0.1},
        {"score": 2.0, "apy": 0.1},
    ]
    assert client.requests[0]["metric_fields"] == ["score", "apy"]
    assert client.requests[0]["metrics_format"] == "array"


def _write_heatmap_result(path: Path) -> Path:
    writer = GridResultWriter(
        path,
        run_id="heatmap",
        total=4,
        metadata={
            "candidate_defaults": {"policy_params": [], "pool": {}},
            "axes": {"pool.A": [1, 2], "pool.donation_apy": [0.0, 0.1]},
            "shape": [2, 2],
        },
        metric_names=("score",),
    )
    writer.append_projected(
        range(4),
        ProjectedBatch(
            ("score",),
            tuple(
                {
                    "candidate_id": f"p{ordinal:08d}",
                    "status": "ok",
                    "metrics": [float(ordinal)],
                }
                for ordinal in range(4)
            ),
        ),
    )
    writer.finalize()
    return path


def test_public_heatmap_command_reads_run_json_and_results_npz(tmp_path: Path) -> None:
    result_dir = _write_heatmap_result(tmp_path / "run")
    output = tmp_path / "heatmap.png"
    result = CliRunner().invoke(
        main,
        [
            "heatmap",
            str(result_dir),
            "--metric",
            "score",
            "--x",
            "pool.A",
            "--y",
            "pool.donation_apy",
            "--columns",
            "2",
            "--output",
            str(output),
            "--no-show",
        ],
    )
    assert result.exit_code == 0, result.output
    state = json.loads(output.with_suffix(".state.json").read_text())
    assert output.is_file() and output.stat().st_size > 0
    assert state["data"]["shape"] == [2, 2]
    assert state["metric"] == "score"
    assert state["x_axis"] == "pool.A"
    assert state["y_axis"] == "pool.donation_apy"


def _mask_dataset(*, omit: str | None = None) -> HeatmapDataset:
    metrics: dict[str, np.ndarray] = {
        "apy_net": np.ones((2, 2)),
        "max_7d_rel_price_diff": np.array([[0.001, 0.003], [-1.0, 0.002]]),
        "detach_energy_ungated": np.array([[0.0, 1.0], [2.0, 3.0]]),
        "final_rel_price_diff": np.array([[0.001, 0.003], [-1.0, 0.002]]),
        "tw_real_slippage_1pct": np.array([[0.001, -1.0], [0.003, 0.002]]),
    }
    if omit is not None:
        metrics.pop(omit)
    return HeatmapDataset(
        axes=(HeatmapAxis(("x",), (1, 2)), HeatmapAxis(("y",), (10, 20))),
        metrics=metrics,
        valid=np.ones((2, 2), dtype=bool),
    )


def test_generic_mask_thresholds_fail_closed_and_require_source_metrics() -> None:
    cases = (
        (MaskSpec(), None, 4),
        (MaskSpec(max_price_diff_bps=20), "max_7d_rel_price_diff", 2),
        (MaskSpec(max_detach_energy=1), "detach_energy_ungated", 2),
        (MaskSpec(max_final_price_diff_bps=20), "final_rel_price_diff", 2),
        (MaskSpec(slippage_thr_bps=20), "tw_real_slippage_1pct", 2),
    )
    for mask, source, expected in cases:
        dataset = _mask_dataset()
        whole = dataset.metric_array("apy_net_masked", mask)
        assert np.isfinite(whole).sum() == expected
        np.testing.assert_equal(
            dataset.slice_metric("apy_net_masked", x_axis="x", y_axis="y", fixed_indices={}, mask=mask),
            whole.T,
        )
        if source is not None:
            with pytest.raises(HeatmapValidationError, match="mask metric.*unavailable"):
                _mask_dataset(omit=source).metric_array("apy_net_masked", mask)


def test_legacy_apy_mask_aliases_use_matching_slippage_sources() -> None:
    dataset = _mask_dataset()
    dataset.metrics["max_7d_rel_price_diff"] = np.full((2, 2), 0.001)
    dataset.metrics["tw_real_slippage_5pct"] = np.array(
        [[0.003, 0.001], [0.003, 0.001]]
    )
    mask = MaskSpec(slippage_thr_bps=20)

    assert np.isfinite(dataset.metric_array("apy_masked", mask)).all()
    assert np.isfinite(dataset.metric_array("apy_1_masked", mask)).tolist() == [
        [True, False],
        [False, True],
    ]
    assert np.isfinite(dataset.metric_array("apy_5_masked", mask)).tolist() == [
        [False, True],
        [False, True],
    ]
    assert np.isfinite(
        dataset.metric_array("tw_real_slippage_5pct_masked", mask)
    ).all()
    assert masked_metric_slippage_sources(
        ("apy_masked", "apy_1_masked", "apy_5_masked"), dataset.metrics
    ) == ("tw_real_slippage_1pct", "tw_real_slippage_5pct")


def test_rolling_growth_leaves_nonfinite_windows_unavailable() -> None:
    timestamps = np.array([0.0, ROLLING_APY_WINDOW_S])
    for growth in (np.array([1.0, np.nan]), np.array([1.0, np.inf]), np.array([np.inf, 1.0])):
        rolling, floored = _rolling_window_growth_apy(timestamps, growth)
        assert np.isnan(rolling[1]) and not floored[1]
    rolling, floored = _rolling_window_growth_apy(timestamps, np.array([1.0, 0.0]))
    assert rolling[1] == 0.0 and floored[1]


class _TraceClient:
    def __init__(self, trace_paths: dict[str, Path]) -> None:
        self.trace_paths = trace_paths
        self.payloads: list[dict[str, object]] = []
        self.open_requests: list[dict[str, object]] = []
        self.mode = "off"

    def start(self) -> None:
        pass

    def open_session(self, session_id: str, **request: object) -> None:
        self.open_requests.append(request)
        self.mode = str(request["yb_mode"])

    def evaluate_batch(self, candidates: list[dict[str, object]], **request: object) -> dict[str, object]:
        self.payloads.extend(candidates)
        item = candidates[0]
        return {
            "results": [{
                "candidate_id": item["candidate_id"],
                "status": "ok",
                "metrics": {"score": 1.0, "apy_net": 0.123, "apy_net_gm": 0.045},
                "artifacts": {
                    "trace_path": str(self.trace_paths[self.mode]),
                    "effective_inputs": {"pool.donation_frequency": 3600.0},
                },
            }],
        }

    def close_session(self, session_id: str | None = None) -> None:
        pass

    def shutdown(self) -> None:
        pass


def test_stored_ordinal_replay_passes_exact_candidate_for_yb_off_and_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fxopt.explorer import open_fxopt_explorer

    run_dir = _write_heatmap_result(tmp_path / "run")
    manifest = json.loads((run_dir / "run.json").read_text())
    manifest["metadata"].update(
        {
            "expected_evaluator_policy": {
                "policy_id": "none",
                "policy_abi": "none",
                "policy_parameter_count": 0,
            },
            "replay": {
                "evaluator": str(tmp_path / "evaluator"),
                "work_dir": str(tmp_path),
                "open_session": {
                    "scenario_id": "scenario",
                    "yb_mode": "off",
                },
            },
        }
    )
    (run_dir / "run.json").write_text(json.dumps(manifest))
    trace_paths: dict[str, Path] = {}
    for mode in ("off", "active_2l"):
        trace_paths[mode] = tmp_path / f"{mode}.json"
        trace_paths[mode].write_text(json.dumps([
            {
                "t": 1_700_000_000 + index * 8_640_000,
                "price_scale": 1.0,
                "p_cex": 1.0,
                "token0": 1_000.0,
                "token1": 1_000.0,
                "fee": 0.001,
                "slippage_1pct_0to1": 0.001,
                "slippage_1pct_1to0": 0.001,
                "lp_xcp_profit": 1.0 + index * 0.01,
                "donation_apy": 0.02,
                "yb_initialized": float(mode != "off"),
                "yb_growth": 1.0 + index * 0.01 if mode != "off" else None,
            }
            for index in range(2)
        ]))
    clients: list[_TraceClient] = []

    def local_factory(*_args: object, **_kwargs: object):
        client = _TraceClient(trace_paths)
        clients.append(client)
        return lambda: client

    monkeypatch.setattr("fxopt.shiftclick.local_client_factory", local_factory)
    ordinal = 3
    explorer = open_fxopt_explorer(
        run_dir,
        metrics=("score",),
        x_axis="pool.A",
        y_axis="pool.donation_apy",
        max_price_diff_bps=None,
    )
    try:
        selection = explorer.dataset.point((1, 1))
        assert selection.ordinal == ordinal and selection.candidate_id == "p00000003"
        shift_figure = explorer.on_replay(selection, "shift")
        right_figure = explorer.on_replay(selection, "right")
        for figure in (shift_figure, right_figure):
            assert any("12.3%" in axis.get_title() for axis in figure.axes)
        assert not (run_dir / "inspections").exists()
    finally:
        explorer.close()
        from matplotlib import pyplot as plt
        plt.close("all")

    assert [client.open_requests[0]["yb_mode"] for client in clients] == [
        "active_2l", "off"
    ]
    assert clients[0].open_requests[0]["yb_cash_multiplier"] == 3.0
    assert "yb_cash_multiplier" not in clients[1].open_requests[0]
    assert all(client.payloads[0]["candidate_id"] == "p00000003" for client in clients)
