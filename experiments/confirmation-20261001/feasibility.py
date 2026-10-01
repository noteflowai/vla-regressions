"""Sample-based wall forecast, including possible full-horizon episodes."""
import math


def primary_forecast(durations, episodes, primary_pairs=80, horizon=520):
    if (type(primary_pairs) is not int or primary_pairs != 80
            or type(horizon) is not int or horizon != 520):
        raise ValueError("Keep the frozen 80 pairs and 520-step horizon")
    if len(durations) != len(episodes) or not durations:
        raise ValueError("Complete control durations and episode pairs required")
    overheads, rates = [], []
    for duration, pair in zip(durations, episodes, strict=True):
        if (type(duration) not in (float, int) or not math.isfinite(duration)
                or duration <= 0 or len(pair) != 2):
            raise ValueError("Invalid control duration")
        elapsed = []
        for side in pair:
            steps, seconds = side.get("steps"), side.get("elapsed_seconds")
            if (type(steps) is not int or not 0 < steps <= horizon
                    or type(seconds) not in (float, int) or not math.isfinite(seconds)
                    or seconds <= 0):
                raise ValueError("Invalid native episode timing")
            elapsed.append(seconds)
            rates.append(seconds / steps)
        overhead = duration - sum(elapsed)
        if overhead < 0:
            raise ValueError("Control pair duration is shorter than its episodes")
        overheads.append(overhead)
    fixed = max(overheads)
    full_horizon_episode = max(rates) * horizon
    pair = fixed + 2 * full_horizon_episode
    return {
        "estimated_primary_wall_seconds": math.ceil(pair * primary_pairs * 1.5) + 1,
        "maximum_observed_pair_fixed_overhead_seconds": fixed,
        "maximum_observed_episode_seconds_per_step": max(rates),
        "full_horizon_episode_seconds": full_horizon_episode,
        "full_horizon_pair_seconds_before_margin": pair,
        "margin_factor": 1.5, "horizon": horizon, "primary_pairs": primary_pairs,
        "scope": "Sample-based planning, not a worst-case guarantee. Episode time includes "
                 "reset/persistence; scaling it to the full horizon is conservative for "
                 "those fixed costs. New precision behavior and host load can differ.",
    }
