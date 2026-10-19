#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
  echo "Usage: bash scripts/heatmap-pool.sh CONFIG RUN_DIR [heatmap options...]" >&2
  exit 2
fi

# Relative config and run paths are relative to this repository.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
config=$1
run_dir=$2
shift 2

# Completed local artifacts can be opened without contacting the cluster.
if [[ ! -f "$run_dir/run.json" || ! -f "$run_dir/results.npz" ]]; then
  env -u VIRTUAL_ENV uv run fxopt run "$config" --output "$run_dir" --follow
fi

exec env -u VIRTUAL_ENV uv run fxopt heatmap "$run_dir" \
  --x policy_params.0 --y policy_params.4 --columns 4 \
  --metric vp_growth_apy --metric donation_burn_apy --metric xcp_growth_apy --metric xcp_apy_gm30 \
  --metric fee_xcp_apy --metric net_step_cost_apy --metric arb_take_apy --metric xcp_apy_gm60 \
  --metric avg_rel_price_diff --metric max_7d_rel_price_diff --metric trades_per_year --metric xcp_apy_gm90 \
  --max-price-diff-bps 1500 \
  --shiftclick-yb-mode off \
  "$@"
