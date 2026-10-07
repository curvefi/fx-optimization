from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fxopt.config import ConfigError, RunConfig
from fxopt.engine import ProjectedBatch
from fxopt.results import GridResultWriter, merge_grid_partitions, read_result_columns
from fxopt.run import run_config, run_metadata, run_leased_worker


def _run_toml(
    path: Path,
    *,
    market: str | None = "market.json",
    policy_params: str = "[]",
    compiled_policy: str = "",
    session: str = "",
    yb_mode: str = "off",
    price_feed: str = "",
    block_tape: str = "",
    metrics: str = '["score"]',
    axes: str = "",
) -> Path:
    market_line = f'market = "{market}"' if market is not None else ""
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
{market_line}
{price_feed}
{block_tape}
yb_mode = "{yb_mode}"
{compiled_policy}
[candidate.defaults]
policy_params = {policy_params}
pool = {{}}
{axes}
'''
    )
    return path


def test_block_config_omits_market_and_candle_controls(tmp_path):
    from fxopt.run import open_session_request
    from curve_fx_harness_client.models import OpenSessionFrame
    session = '[session]\nevent_mode = "block"\nevent_cursor = "fast_skip"'
    for mode in ("off", "active_2l"):
        path = _run_toml(tmp_path/"blocks.toml", market=None, yb_mode=mode,
                         block_tape='block_tape = "blocks.npz"', session=session)
        config = RunConfig.from_toml(path)
        request = open_session_request(config)
        frame = OpenSessionFrame.model_validate(dict(request, request_id="r", session_id="s"))
        assert frame.event_mode == "block" and frame.block_tape_path.endswith("blocks.npz")
        assert request["early_stop_max_7d_rel_price_diff"] == 0.3  # default pool stop
        assert "market_path" not in request and "market" not in run_metadata(config)
    for values, error in (({"block_tape": 'block_tape = "blocks.jsonl"'}, "requires NPZ"),
                          ({"block_tape": 'block_tape = "blocks.npz"'}, "block events require"),
                          ({"block_tape": 'trade_flow = "flow.npz"'}, "unsupported retired .scenario. option"),
                          ({"market": None, "block_tape": 'block_tape = "blocks.npz"',
                            "session": '[session]\nevent_mode = "block"\ncandle_filter = 10'}, "block events take no"),
                          ({"session": '[session]\narb_settle_offset_s = 1'}, "arb_settle_offset_s requires"),
                          ({"session": '[session]\nevent_mode = "trade_flow"'}, "event_mode must be candles or block"),
                          ({"session": '[session]\narb_role = "taker"'}, "unsupported retired"),
                          ({"yb_mode": "reference_2l"}, "yb_mode must be off or active_2l")):
        with pytest.raises(ConfigError, match=error):
            RunConfig.from_toml(_run_toml(tmp_path/"bad-blocks.toml", **values))


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
        ("block", {"market": None, "block_tape": 'block_tape = "blocks.npz"', "session": '[session]\nevent_mode = "block"'}, None),
        ("candles-yb", {"yb_mode": "active_2l"}, None),
        ("retired-depth", {"session": '[session]\ncex_depth_max_age_s = 30'}, "unsupported retired"),
        ("exact-skip", {"session": '[session]\nevent_cursor = "exact_skip"'}, "event_cursor must be scalar or fast_skip"),
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
            "full-summary-yb-slippage",
            {
                "session": (
                    '[session]\nevent_cursor = "scalar"\n'
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
            assert run_metadata(config)["expected_evaluator_policy"] == {
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
            metadata = run_metadata(config)
            assert metadata["open_session"]["price_feed_path"].endswith("nav.csv")
        if name == "block":
            opened = run_metadata(config)["open_session"]
            assert opened["block_tape_path"].endswith("blocks.npz") and opened["event_mode"] == "block"


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
        metadata=run_metadata(RunConfig.from_toml(config)),
        metric_names=("score",),
    )
    columns = read_result_columns(output)
    assert columns.ok_mask.tolist() == [True, True, False, False]
    np.testing.assert_equal(columns.metrics["score"], [0.0, 1.0, np.nan, np.nan])



def test_replay_without_active_yb_drops_the_yb_actor_choice(tmp_path: Path) -> None:
    from fxopt.config import EVALUATOR_POLICY_METADATA_KEY
    from fxopt.contract import Candidate
    from fxopt.shiftclick import trace_stored_candidate

    seen: list[tuple[dict[str, object], list[dict[str, object]]]] = []

    class Client(_GridClient):
        def evaluate_batch(self, candidates, **request):
            seen.append((self.open_request, list(candidates)))
            return {"results": [{"candidate_id": c["candidate_id"], "status": "ok", "metrics": {}} for c in candidates]}

    metadata = {
        "replay": {"evaluator": "evaluator", "work_dir": str(tmp_path),
                   "open_session": {"yb_mode": "active_2l", "yb_arb": "none", "metric_profile": "full_summary"}},
        EVALUATOR_POLICY_METADATA_KEY: {"policy_id": "p", "policy_abi": "1", "policy_parameter_count": 0},
    }
    candidate = Candidate("c0", (), {"run": {"arb_report_offset": 1}})
    for yb_mode in ("off", "active_2l"):
        trace_stored_candidate("run", metadata, candidate=candidate, ordinal=0, output_dir=tmp_path / yb_mode,
                               yb_mode=yb_mode, client_factory=Client)
    (off_session, off_batch), (active_session, active_batch) = seen
    assert "yb_arb" not in off_session and active_session["yb_arb"] == "none"
    assert "metric_profile" not in off_session and "metric_profile" not in active_session
    assert off_batch[0]["pool_overrides"] == active_batch[0]["pool_overrides"] == {"run": {"arb_report_offset": 1}}
