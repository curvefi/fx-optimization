// Native-fee, policy-owned dual EMA with a touch-age linear movement cap.
//
// Params:
//   p0 fast_half_life_s
//   p1 slow_half_life_s
//   p2 kappa
//   p3 min_cap_bps
//   p4 deadband_bps
//
// cap_bps = clamp(dt / 60, min_cap_bps, 60),
// where dt is seconds since the last policy touch. The returned target is
// clipped to +/-5*cap_bps because native tweak_price applies target_gap/5.
#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "pools/twocrypto_fx/policies/common.hpp"
#include "pools/twocrypto_fx/policy_descriptor.hpp"

namespace arb::pools::twocrypto_fx {

template <typename T>
struct ChallengeFeePolicy {
    inline static constexpr PolicyDescriptor<5> DESCRIPTOR{
        "native_policy_dual_ema_stale_cap_v1",
        {{{"fast_half_life_s", 0, "seconds",
           3600.0L, 60.0L, 86400.0L, 10.0L},
          {"slow_half_life_s", 1, "seconds",
           86400.0L, 60.0L, 604800.0L, 10.0L},
          {"kappa", 2, "dimensionless",
           1.0L, 0.0L, 5.0L, 0.05L},
          {"min_cap_bps", 3, "basis_points",
           10.0L, 0.0L, 250.0L, 0.5L},
          {"deadband_bps", 4, "basis_points",
           0.0L, 0.0L, 100.0L, 0.5L}}},
    };
    static constexpr const char* NAME = DESCRIPTOR.name.data();
    static constexpr std::size_t PARAM_COUNT = DESCRIPTOR.size();
    static constexpr long double LN2 = 0.69314718055994530942L;

    enum class Parameter : std::size_t {
        FastHalfLife = 0,
        SlowHalfLife,
        Kappa,
        MinCapBps,
        DeadbandBps,
    };

    struct State {
        bool initialized{false};
        uint64_t sample_ts{0};
        T pending_print{T(0)};
        T fast_ema{T(0)};
        T slow_ema{T(0)};
        T price_scale{T(0)};
    };

    struct EmaPair {
        T fast{T(0)};
        T slow{T(0)};
    };

    static T param(
        const PolicyConfig<T>& params,
        Parameter parameter,
        T fallback
    ) {
        const std::size_t index = static_cast<std::size_t>(parameter);
        return index < params.n_params ? params.params[index] : fallback;
    }

    static T clamp_local(T value, T lo, T hi) {
        if (value < lo) return lo;
        if (value > hi) return hi;
        return value;
    }

    static T fast_half_life_s(const PolicyConfig<T>& params) {
        return clamp_local(param(params, Parameter::FastHalfLife, T(3600)), T(60), T(86400));
    }

    static T slow_half_life_s(const PolicyConfig<T>& params) {
        return clamp_local(param(params, Parameter::SlowHalfLife, T(86400)), T(60), T(604800));
    }

    static T kappa(const PolicyConfig<T>& params) {
        return clamp_local(param(params, Parameter::Kappa, T(1)), T(0), T(5));
    }

    static T min_cap_bps(const PolicyConfig<T>& params) {
        return clamp_local(param(params, Parameter::MinCapBps, T(10)), T(0), T(250));
    }

    static T deadband_bps(const PolicyConfig<T>& params) {
        return clamp_local(param(params, Parameter::DeadbandBps, T(0)), T(0), T(100));
    }

    static T max_cap_bps() {
        return T(60);
    }

    static T scale_seconds_per_bp() {
        return T(60);
    }

    static T get_fee(
        const State&,
        const PolicyConfig<T>&,
        const PolicyPoolConfig<T>&,
        const PolicyResearchContext<T>&,
        const std::array<T, 2>&
    ) {
        return T(0);
    }

    static T fee_floor(
        const PolicyConfig<T>&,
        const PolicyPoolConfig<T>&,
        const T& native_floor
    ) {
        return native_floor;
    }

    static T ema_half_life(T before, T sample, uint64_t dt, T half_life_s) {
        if (dt == 0 || !(half_life_s > T(0))) return before;
        if constexpr (std::is_same_v<T, uint256>) {
            return sample;
        } else {
            using std::exp;
            const T keep = exp(-T(LN2) * T(dt) / half_life_s);
            return sample * (T(1) - keep) + before * keep;
        }
    }

    static T cap_sample(T sample, T price_scale) {
        if (!(price_scale > T(0))) return sample;
        return clamp_local(sample, price_scale / T(2), price_scale * T(2));
    }

    static EmaPair project_emas(
        const State& state,
        uint64_t now,
        const PolicyConfig<T>& params
    ) {
        EmaPair projected{state.fast_ema, state.slow_ema};
        if (!state.initialized || !(state.pending_print > T(0)) ||
            now <= state.sample_ts) {
            return projected;
        }

        const uint64_t dt = now - state.sample_ts;
        projected.fast = ema_half_life(
            state.fast_ema,
            state.pending_print,
            dt,
            fast_half_life_s(params)
        );
        projected.slow = ema_half_life(
            state.slow_ema,
            state.pending_print,
            dt,
            slow_half_life_s(params)
        );
        return projected;
    }

    static T assembled_target(
        const EmaPair& emas,
        T current,
        const PolicyConfig<T>& params
    ) {
        if (!(emas.fast > T(0)) || !(emas.slow > T(0))) return T(0);

        const T lead = kappa(params);
        if (emas.fast >= emas.slow) {
            return emas.slow + lead * (emas.fast - emas.slow);
        }
        const T down = lead * (emas.slow - emas.fast);
        return down < emas.slow ? emas.slow - down : current / T(2);
    }

    static T get_price_scale(
        State& state,
        PolicyResearchContext<T>& research,
        const PolicyConfig<T>& params,
        const PolicyPoolConfig<T>& config
    ) {
        (void)config;
        if (!state.initialized || !(state.price_scale > T(0))) return T(0);

        const uint64_t dt = research.block_timestamp > state.sample_ts
            ? research.block_timestamp - state.sample_ts
            : 0;
        const EmaPair emas = project_emas(
            state,
            research.block_timestamp,
            params
        );
        const T current = state.price_scale;
        const T target = assembled_target(emas, current, params);
        if (!(target > T(0))) return T(0);

        const T delta = target >= current
            ? target - current
            : current - target;
        const T gap_bps = delta * T(10000) / current;
        if (deadband_bps(params) > T(0) &&
            gap_bps <= deadband_bps(params)) {
            return current;
        }

        const T lo = min_cap_bps(params);
        const T hi = max_cap_bps();
        const T ordered_hi = hi >= lo ? hi : lo;
        const T cap_bps = clamp_local(
            T(dt) / scale_seconds_per_bp(),
            lo,
            ordered_hi
        );
        const T target_gap_bps = T(5) * cap_bps;
        const T offset = current * target_gap_bps / T(10000);
        if (target > current + offset) return current + offset;
        if (target < current - offset) return current - offset;
        return target;
    }

    static void update_state(
        State& state,
        PolicyResearchContext<T>& research,
        const PolicyConfig<T>& params,
        const PolicyPoolConfig<T>& config,
        const PolicyUpdate<T>& update
    ) {
        (void)config;
        state.price_scale = update.price_scale;

        const uint64_t now = research.block_timestamp != 0
            ? research.block_timestamp
            : update.oracle_timestamp;
        if (now == 0 || !(update.last_prices > T(0))) return;

        const T sample = cap_sample(update.last_prices, update.price_scale);
        if (!state.initialized) {
            state.initialized = true;
            state.sample_ts = now;
            state.pending_print = sample;
            state.fast_ema = sample;
            state.slow_ema = sample;
            return;
        }
        if (now < state.sample_ts) return;

        const EmaPair projected = project_emas(state, now, params);
        state.fast_ema = projected.fast;
        state.slow_ema = projected.slow;
        state.sample_ts = now;
        state.pending_print = sample;
    }
};

} // namespace arb::pools::twocrypto_fx
