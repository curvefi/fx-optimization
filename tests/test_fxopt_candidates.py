from __future__ import annotations

import pytest

from fxopt.candidates import CandidateSpec
from fxopt.config import CandidateConfig, ConfigError, compile_candidates


def _config(*, reverse_axes: bool = False) -> CandidateConfig:
    axes = {
        "pool.A": {"values": tuple(range(1, 17)), "multiply": 10_000},
        "pool.donation_apy": tuple(index / 100 for index in range(16)),
    }
    if reverse_axes:
        axes = dict(reversed(tuple(axes.items())))
    return CandidateConfig.from_mapping(
        {
            "defaults": {
                "policy_params": [0.5, 0.0003],  # rpf=0.5, fee=3 bps
                "pool": {"A": 100, "donation_apy": 0.08},
            },
            "axes": axes,
        }
    )


def test_one_point_and_lazy_cartesian_grid_share_candidate_contract() -> None:
    config = _config()
    point = compile_candidates(config, "point")
    assert isinstance(point, CandidateSpec)
    assert point.payload == {
        "policy_params": [0.5, 0.0003],
        "pool": {"A": 100, "donation_apy": 0.08},
    }
    override = config.point({"pool.A": 101})
    assert override.payload["pool"] == {"A": 101, "donation_apy": 0.08}
    assert config.defaults["pool"] == {"A": 100, "donation_apy": 0.08}

    grid = compile_candidates(_config(), "grid")
    assert len(grid) == 16 * 16
    assert grid.candidate_at(0).payload["pool"]["A"] == 10_000
    assert grid.candidate_at(len(grid) - 1).payload["pool"]["A"] == 160_000
    first_batch = next(grid.iter_batches(7))
    assert len(first_batch) == 7
    assert all(isinstance(candidate, CandidateSpec) for candidate in first_batch)


def test_ids_and_payloads_are_independent_of_axis_order_and_batch_size() -> None:
    forward = compile_candidates(_config(), "grid")
    reverse = compile_candidates(_config(reverse_axes=True), "grid")
    forward_rows = [(item.candidate_id, dict(item.payload)) for item in forward]
    reverse_rows = [(item.candidate_id, dict(item.payload)) for item in reverse]
    assert forward_rows == reverse_rows

    with pytest.raises(ConfigError, match="collision"):
        CandidateConfig.from_mapping({"axes": {"pool": [1], "pool.A": [2]}})

    batched_rows = [
        (item.candidate_id, dict(item.payload))
        for batch in forward.iter_batches(11)
        for item in batch
    ]
    assert batched_rows == forward_rows


def test_million_candidate_grid_only_materializes_requested_batch() -> None:
    grid = CandidateConfig.from_mapping(
        {
            "defaults": {"policy_params": [0.5, 0.0003], "pool": {}},
            "axes": {"pool.A": range(1000), "pool.donation_apy": range(1000)},
        }
    ).grid()
    assert len(grid) == 1_000_000
    batch = next(grid.iter_batches(5))
    assert len(batch) == 5
    assert grid.candidate_at(999_999).payload == {
        "policy_params": [0.5, 0.0003],
        "pool": {"A": 999, "donation_apy": 999},
    }

    small = CandidateConfig.from_mapping({
        "defaults": {"policy_params": [], "pool": {}},
        "axes": {"pool.A": range(4), "pool.donation_apy": range(4)},
    }).grid()
    scheduled = list(small.iter_rotating_blocks(block_size=2, rotations=4))
    assert [ordinal for ordinal, _spec in scheduled] == [
        0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15,
    ]
    assert sorted(ordinal for ordinal, _spec in scheduled) == list(range(16))


def test_compact_ranges_support_log_spacing_and_linked_targets() -> None:
    grid = CandidateConfig.from_mapping({
        "defaults": {"policy_params": [], "pool": {}},
        "axes": {
            "flat_fee": {
                "start": 0.005,
                "stop": 0.035,
                "count": 3,
                "targets": ["pool.mid_fee", "pool.out_fee"],
            },
            "pool.ma_time": {"start": 300, "stop": 1200, "count": 3, "scale": "log"},
        },
    }).grid()
    assert grid.axes["pool.ma_time"] == (300, 600.0, 1200)
    assert grid.candidate_at(0).payload["pool"]["mid_fee"] == 0.005
    assert grid.candidate_at(len(grid) - 1).payload["pool"]["out_fee"] == 0.035
