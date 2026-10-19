#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
  echo "Usage: bash scripts/hm-yb.sh CONFIG RUN_DIR [heatmap options...]" >&2
  exit 2
fi

cd "$(dirname "${BASH_SOURCE[0]}")/.."
config=$1
run_dir=$2
shift 2

if [[ ! -f "$run_dir/run.json" || ! -f "$run_dir/results.npz" ]]; then
  env -u VIRTUAL_ENV uv run fxopt run "$config" --output "$run_dir" --follow
fi

if (( $# == 0 )) && [[ ! -t 0 ]]; then
  set -- --output "$run_dir/heatmap.png" --no-show
fi

exec env -u VIRTUAL_ENV uv run fxopt heatmap "$run_dir" \
  --x pool.donation_apy --y pool.reserved_profit_fraction --columns 4 \
  --metric yb_apy_masked --metric yb_apy_gm30_masked --metric yb_apy_gm60_masked --metric yb_apy_gm90_masked \
  --max-price-diff-bps 1000 --final-price-diff-bps 100 \
  --shiftclick-yb-mode active_2l --shiftclick-yb-cash-multiplier 3 \
  "$@"
