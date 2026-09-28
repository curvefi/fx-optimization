#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
  echo "Usage: bash scripts/heatmap-yb.sh CONFIG RUN_DIR [heatmap options...]" >&2
  exit 2
fi

# Relative config and run paths are relative to this repository.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
config=$1
run_dir=$2
shift 2

# Completed local artifacts can be opened without contacting the cluster.
if [[ ! -f "$run_dir/run.json" || ! -f "$run_dir/results.npz" ]]; then
  env -u VIRTUAL_ENV uv run fxopt run "$config" --output "$run_dir" --status
  env -u VIRTUAL_ENV uv run fxopt run "$config" --output "$run_dir" --follow
fi

exec env -u VIRTUAL_ENV uv run fxopt heatmap "$run_dir" \
  --x pool.donation_apy --y pool.reserved_profit_fraction --columns 3 \
  --metric apy_net --metric yb_apy --metric yb_apy_gm \
  --metric apy_net_masked --metric yb_apy_masked --metric yb_apy_gm_masked \
  --metric max_7d_rel_price_diff --metric apy_net_robust_90d --metric avg_imbalance \
  --max-price-diff-bps 1500 \
  --shiftclick-yb-mode active_2l --shiftclick-yb-cash-multiplier 3 \
  "$@"
