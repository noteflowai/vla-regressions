"""Compare two variants state by state.

A state is (task, init_state, scene_seed). For each state both variants have n repeated rollouts.
Reports:
  - the aggregate success rates and their paired difference, with a bootstrap over states;
  - naive flips: what a single rollout per state would show (repeat 0 only);
  - per-state regressions: a one-sided exact paired discordance test,
    Benjamini-Yekutieli across states, and a paired discordance posterior.

Usage: analyze.py OLD.jsonl NEW.jsonl [--alpha 0.1] [--json out.json] [--ignore-scene]
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from math import comb
from scipy.stats import beta


def load(path, match_scene=True):
    states = defaultdict(list)
    with open(path) as handle:
        for line in handle:
            r = json.loads(line)
            if type(r.get("success")) is not bool or type(r.get("repeat")) is not int:
                raise ValueError("A row needs a boolean outcome and integer repeat")
            if any(k not in r for k in ("task", "init_state", "scene_seed", "seed")):
                raise ValueError("Incomplete episode identity")
            states[(r["task"], r["init_state"], r["scene_seed"] if match_scene else None)].append(r)
    for rows in states.values():
        rows.sort(key=lambda r: r["repeat"])
        if len({r["repeat"] for r in rows}) != len(rows):
            raise ValueError("Duplicate state/repeat identity")
    return states


def fisher_less(k_new, n_new, k_old, n_old):
    """P(new has k_new or fewer successes) given the margins: small when new is worse."""
    total_k, total_n = k_new + k_old, n_new + n_old
    denom = comb(total_n, total_k)
    lo = max(0, total_k - n_old)
    return sum(comb(n_new, x) * comb(n_old, total_k - x) for x in range(lo, k_new + 1)) / denom


def bh(pvalues, alpha):
    """Benjamini-Hochberg: indices rejected at FDR alpha."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    m, cutoff = len(pvalues), -1
    for rank, i in enumerate(order, 1):
        if pvalues[i] <= alpha * rank / m:
            cutoff = rank
    return set(order[:cutoff]) if cutoff > 0 else set()


def by(pvalues, alpha):
    """FDR control under arbitrary dependence of valid marginal p-values."""
    if not pvalues:
        return set()
    harmonic = sum(1/i for i in range(1, len(pvalues)+1))
    return bh(pvalues, alpha/harmonic)


def paired_pvalue(harm, gain):
    """Conditional exact one-sided McNemar/binomial test on discordant pairs."""
    n = harm + gain
    return sum(comb(n, k) for k in range(harm, n+1)) / 2**n if n else 1.


def pair_rows(old, new, allow_seed_change=False):
    if {r["repeat"] for r in old} != {r["repeat"] for r in new}:
        raise ValueError("Paired repeat identities differ")
    lookup = {r["repeat"]: r for r in new}
    pairs = [(r, lookup[r["repeat"]]) for r in old]
    if not allow_seed_change and any(a["seed"] != b["seed"] for a, b in pairs):
        raise ValueError("Policy seeds differ; declare the independent-seed control explicitly")
    return pairs


def prob_worse_paired(harm, gain):
    """Jeffreys posterior for the direction of a discordant pair."""
    return float(beta.sf(.5, harm+.5, gain+.5))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("old")
    p.add_argument("new")
    p.add_argument("--alpha", type=float, default=0.1, help="FDR level for per-state regressions")
    p.add_argument("--json")
    p.add_argument("--ignore-scene", action="store_true",
                   help="pair on (task, init_state) only, to compare runs with different scene seeds")
    p.add_argument("--allow-seed-change", action="store_true",
                   help="explicit independent-policy-seed noise control, paired by repeat")
    args = p.parse_args()

    old, new = load(args.old, not args.ignore_scene), load(args.new, not args.ignore_scene)
    if set(old) != set(new):
        raise ValueError("State inventories differ; silently intersecting states is not allowed")
    keys = sorted(old)
    if not keys:
        raise SystemExit("no states in common")

    rows = []
    for key in keys:
        a, b = old[key], new[key]
        pairs = pair_rows(a, b, args.allow_seed_change)
        harm = sum(x["success"] and not y["success"] for x, y in pairs)
        gain = sum(not x["success"] and y["success"] for x, y in pairs)
        ka, na = sum(r["success"] for r in a), len(a)
        kb, nb = sum(r["success"] for r in b), len(b)
        rows.append(
            {
                "task": key[0],
                "init_state": key[1],
                "old": [ka, na],
                "new": [kb, nb],
                "discordant_harm": harm,
                "discordant_gain": gain,
                "p_regress": paired_pvalue(harm, gain),
                "p_improve": paired_pvalue(gain, harm),
                "prob_worse": prob_worse_paired(harm, gain),
                "first_old": a[0]["success"],
                "first_new": b[0]["success"],
            }
        )

    rate = lambda side: sum(r[side][0] for r in rows) / sum(r[side][1] for r in rows)
    diffs = [r["new"][0] / r["new"][1] - r["old"][0] / r["old"][1] for r in rows]
    rng = random.Random(1)
    boot = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(4000))
    ci = (boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot))])

    naive_neg = sum(r["first_old"] and not r["first_new"] for r in rows)
    naive_pos = sum(r["first_new"] and not r["first_old"] for r in rows)
    regressed = by([r["p_regress"] for r in rows], args.alpha)
    improved = by([r["p_improve"] for r in rows], args.alpha)

    summary = {
        "old": args.old,
        "new": args.new,
        "states": len(rows),
        "repeats": sorted({r["old"][1] for r in rows} | {r["new"][1] for r in rows}),
        "success_old": rate("old"),
        "success_new": rate("new"),
        "delta": sum(diffs) / len(diffs),
        "delta_ci95": ci,
        "naive_negative_flips": naive_neg,
        "naive_positive_flips": naive_pos,
        "regressed_by": len(regressed),
        "improved_by": len(improved),
        "regressed_bh_exploratory": len(bh([r["p_regress"] for r in rows], args.alpha)),
        "improved_bh_exploratory": len(bh([r["p_improve"] for r in rows], args.alpha)),
        "prob_worse_over_0.9": sum(r["prob_worse"] > 0.9 for r in rows),
        "prob_better_over_0.9": sum(r["prob_worse"] < 0.1 for r in rows),
        "alpha": args.alpha,
        "test": "one-sided exact paired discordance (conditional binomial)",
        "multiplicity": "Benjamini-Yekutieli; BH secondary and assumption-dependent",
        "posterior": "Jeffreys Beta posterior for discordance direction",
        "allow_seed_change": args.allow_seed_change,
        "ignore_scene": args.ignore_scene,
    }

    print(f"{summary['states']} states, repeats {summary['repeats']}")
    print(f"success  old {summary['success_old']:.3f}  new {summary['success_new']:.3f}  "
          f"delta {summary['delta']:+.3f}  95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    print(f"single rollout per state: {naive_neg} states flip to failure, {naive_pos} to success")
    print(f"paired per-state tests, BY at {args.alpha}: {len(regressed)} regressed, {len(improved)} improved")
    print(f"posterior P(worse) > 0.9: {summary['prob_worse_over_0.9']} states; "
          f"P(better) > 0.9: {summary['prob_better_over_0.9']}")
    by_task = defaultdict(lambda: [0, 0, 0, 0])
    for r in rows:
        t = by_task[r["task"]]
        t[0] += r["old"][0]; t[1] += r["old"][1]; t[2] += r["new"][0]; t[3] += r["new"][1]
    print("\ntask  old    new")
    for task, (ko, no, kn, nn) in sorted(by_task.items()):
        print(f"{task:>4}  {ko / no:.2f}   {kn / nn:.2f}")
    worst = sorted(rows, key=lambda r: r["p_regress"])[:8]
    print("\nmost regressed states (task, init_state): old k/n -> new k/n, p")
    for r in worst:
        print(f"  ({r['task']}, {r['init_state']}): {r['old'][0]}/{r['old'][1]} -> {r['new'][0]}/{r['new'][1]}, "
              f"p={r['p_regress']:.3f}{' *' if rows.index(r) in regressed else ''}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"summary": summary, "states": rows}, f, indent=1)


if __name__ == "__main__":
    main()
