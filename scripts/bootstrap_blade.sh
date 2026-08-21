#!/usr/bin/env bash
set -euo pipefail

root=${1:-"$HOME/arb"}
pool_repo="$root/twocrypto-cpp"
harness_repo="$root/curve-fx-arb-harness"
optimization_repo="$root/curve-fx-optimization"
nixpkgs_revision=bb8b5735d6f7e06b9ddd27de115b0600c1ffbdb4
nixpkgs_url="https://github.com/NixOS/nixpkgs/archive/${nixpkgs_revision}.tar.gz"
uv_version=0.12.4
python_version=3.12.6
policy_id=native_policy_dual_ema_stale_cap_v1
policy_header="$optimization_repo/policies/$policy_id.hpp"
policy_sha256=e00777e7cdcc2ca9de947e4539d5df184d5b32417f2b0894c09676f511f6cf6d

for repo in "$pool_repo" "$harness_repo" "$optimization_repo"; do
    if [[ ! -d "$repo" ]]; then
        printf 'Missing deployed repository: %s\n' "$repo" >&2
        exit 1
    fi
done

mkdir -p "$root/bin" "$root/build" "$root/install" "$root/tools/uv" "$root/tools/uv-cache"

nix-shell -I "nixpkgs=$nixpkgs_url" -p cmake gcc boost gnumake --run "
    cmake -S '$pool_repo' -B '$root/build/twocrypto-cpp' -DCMAKE_BUILD_TYPE=Release &&
    cmake --build '$root/build/twocrypto-cpp' -j 16 &&
    cmake --install '$root/build/twocrypto-cpp' --prefix '$root/install/twocrypto-cpp' &&
    cmake -S '$harness_repo' -B '$root/build/curve-fx-arb-harness' \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH='$root/install/twocrypto-cpp' \
      -DPOLICY_HEADER_PATH='$policy_header' \
      -DPOLICY_ID='$policy_id' \
      -DPOLICY_EXPECTED_SHA256='$policy_sha256' \
      -DPOLICY_ABI='twocrypto_policy_v1' &&
    cmake --build '$root/build/curve-fx-arb-harness' --target arb_evaluator_ld -j 16
"
install -m 755 "$root/build/curve-fx-arb-harness/arb_evaluator_ld" "$root/bin/arb_evaluator_ld"

if [[ ! -x "$root/tools/uv/uv" ]]; then
    curl -LsSf "https://astral.sh/uv/${uv_version}/install.sh" |
        env UV_INSTALL_DIR="$root/tools/uv" sh
fi

NIXPKGS_ALLOW_UNFREE=1 nix-build \
    -I "nixpkgs=$nixpkgs_url" '<nixpkgs>' -A steam-run \
    -o "$root/tools/steam-run"

steam_run="$root/tools/steam-run/bin/steam-run"
UV_CACHE_DIR="$root/tools/uv-cache" "$steam_run" "$root/tools/uv/uv" python install "$python_version"
(
    cd "$optimization_repo"
    UV_CACHE_DIR="$root/tools/uv-cache" "$steam_run" "$root/tools/uv/uv" sync \
        --frozen --python "$python_version"
)

cat > "$root/bin/fxsim-worker" <<EOF
#!/bin/sh
exec "$steam_run" "$optimization_repo/.venv/bin/fxsim" "\$@"
EOF
chmod 755 "$root/bin/fxsim-worker"

identity_json=$("$root/bin/arb_evaluator_ld" --identity-json)
case "$identity_json" in
    *\"policy_id\":\"$policy_id\"*) ;;
    *)
        printf 'Compiled evaluator policy id does not match %s\n' "$policy_id" >&2
        exit 1
        ;;
esac
case "$identity_json" in
    *\"policy_source_sha256\":\"$policy_sha256\"*) ;;
    *)
        printf 'Compiled evaluator policy SHA-256 does not match %s\n' "$policy_sha256" >&2
        exit 1
        ;;
esac
"$root/bin/fxsim-worker" --version
