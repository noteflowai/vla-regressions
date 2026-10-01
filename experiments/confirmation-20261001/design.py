"""Prospective fixed-sample design. This module never imports a policy or simulator."""
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import beta, binom

ALPHA = .05
PRIMARY_PAIRS = 80
CONTROL_PAIRS = 2
PRIMARY_SEED = 2026100110
CONTROL_SEED = 2026100111
ORDER_SEED = 2026100112
STATE = {"task": 3, "init_state": 3, "scene_seed": 10007306, "weight": 1.0}


def paired_pvalue(harmful, beneficial):
    if any(type(k) is not int or k < 0 for k in (harmful, beneficial)):
        raise ValueError("Discordance counts must be nonnegative integers")
    return float(binom.sf(harmful - 1, harmful + beneficial, .5))


def exact_power(n, harmful_probability, beneficial_probability, alpha=ALPHA):
    """Integrate the exact conditional test over the binomial discordance count."""
    h, g = harmful_probability, beneficial_probability
    if (type(n) is not int or n < 1 or not 0 < alpha < 1
            or any(not math.isfinite(p) or p < 0 or p > 1 for p in (h, g))
            or h + g > 1):
        raise ValueError("Invalid sample size, probabilities or test level")
    r = h + g
    if r == 0:
        return 0.0
    m = np.arange(n + 1)
    # isf returns the largest acceptance count; reject strictly above it.
    cutoff = binom.isf(alpha, m, .5).astype(int) + 1
    return float(np.sum(binom.pmf(m, n, r) * binom.sf(cutoff - 1, m, h / r)))


def pair_plan(mode):
    if mode not in ("control", "primary"):
        raise ValueError("Undeclared cohort")
    n, seed = ((CONTROL_PAIRS, CONTROL_SEED) if mode == "control"
               else (PRIMARY_PAIRS, PRIMARY_SEED))
    # Balance order in advance; do not use collected outcomes to set it.
    order = np.array([0] * (n // 2) + [1] * (n // 2))
    np.random.default_rng(ORDER_SEED + (mode == "control")).shuffle(order)
    return [{"repeat": repeat,
             "policy_seed": int(np.random.SeedSequence([seed, 0, repeat]).generate_state(1)[0]),
             "side_order": ["old", "new"] if first == 0 else ["new", "old"]}
            for repeat, first in enumerate(order)]


def final_analysis(values):
    """Primary inference is available only for the complete frozen 80-pair cohort."""
    if len(values) != PRIMARY_PAIRS or any(
            len(pair) != 2 or any(type(v) is not bool for v in pair) for pair in values):
        raise ValueError("The complete valid fixed-sample cohort is required")
    harmful = sum(old and not new for old, new in values)
    beneficial = sum(new and not old for old, new in values)
    # Two 97.5% marginal Clopper-Pearson intervals and a union bound give a
    # conservative 95% interval for p(harmful) - p(beneficial).
    def marginal_interval(k):
        return (0.0 if k == 0 else float(beta.ppf(.0125, k, PRIMARY_PAIRS - k + 1)),
                1.0 if k == PRIMARY_PAIRS else float(beta.ppf(
                    .9875, k + 1, PRIMARY_PAIRS - k)))
    h_low, h_high = marginal_interval(harmful)
    g_low, g_high = marginal_interval(beneficial)
    blocks = []
    for index in range(8):
        block = values[index * 10:(index + 1) * 10]
        blocks.append(sum(int(old) - int(new) for old, new in block))
    return {
        "pairs": PRIMARY_PAIRS, "fp32_successes": sum(old for old, _ in values),
        "bf16_successes": sum(new for _, new in values),
        "harmful_discordances": harmful, "beneficial_discordances": beneficial,
        "paired_loss": (harmful - beneficial) / PRIMARY_PAIRS,
        "paired_loss_95pct_interval": [h_low - g_high, h_high - g_low],
        "interval_method": "Conservative exact marginal Clopper-Pearson with Bonferroni union bound",
        "one_sided_exact_p": paired_pvalue(harmful, beneficial),
        "alpha": ALPHA, "reject_no_loss": paired_pvalue(harmful, beneficial) <= ALPHA,
        "block_success_differences": blocks,
        "coarse_block_sign_p": paired_pvalue(
            sum(d > 0 for d in blocks), sum(d < 0 for d in blocks)),
        "scope": "One preselected state in a newly audited serial evaluator. The primary "
                 "test assumes independent pairs; block sensitivity uses a different null "
                 "and does not establish robustness to arbitrary dependence. Non-rejection "
                 "is inconclusive, not evidence of preservation or suite-wide prevalence.",
    }


def power_report():
    rows = []
    for n in (20, 40, 80, 100, 160):
        for loss in (.1, .2, .3):
            rates = np.linspace(loss, 1, 701)
            powers = [exact_power(n, float((r + loss) / 2), float((r - loss) / 2))
                      for r in rates]
            index = int(np.argmin(powers))
            rows.append({"pairs": n, "absolute_loss": loss,
                         "grid_minimum_power": powers[index],
                         "minimizing_discordance_probability": float(rates[index]),
                         "power_at_all_discordant": powers[-1]})
    return {
        "method": "Exact binomial-mixture integration of one-sided paired conditional test",
        "alpha": ALPHA, "fixed_primary_pairs": PRIMARY_PAIRS,
        "grid_points_per_loss": 701, "rows": rows,
        "limitations": "Grid minima, not a proved continuous worst-case bound. Alternatives "
                       "are planning assumptions, not estimates from historical outcomes. "
                       "Power is conditional on independent paired trials.",
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(power_report(), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")
