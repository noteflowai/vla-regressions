"""Compare two variants state by state.

A state is (task, init_state, scene_seed). For each state both variants have n repeated rollouts.
Reports:
  - the aggregate success rates and their paired difference, with a bootstrap over states;
  - naive flips: what a single rollout per state would show (repeat 0 only);
  - per-state regressions: one-sided Fisher exact test that the new variant succeeds less often,
    Benjamini-Hochberg across states, and the posterior probability of being worse under
    uniform Beta priors.

Usage: analyze.py OLD.jsonl NEW.jsonl [--alpha 0.1] [--json out.json] [--ignore-scene]
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from math import comb


def load(path, match_scene=True):
    states = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        states[(r["task"], r["init_state"], r.get("scene_seed") if match_scene else None)].append(r)
    for rows in states.values():
        rows.sort(key=lambda r: r["repeat"])
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


def prob_worse(k_new, n_new, k_old, n_old, draws=4000, rng=random.Random(0)):
    worse = 0
    for _ in range(draws):
        if rng.betavariate(k_new + 1, n_new - k_new + 1) < rng.betavariate(k_old + 1, n_old - k_old + 1):
            worse += 1
    return worse / draws


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("old")
    p.add_argument("new")
    p.add_argument("--alpha", type=float, default=0.1, help="FDR level for per-state regressions")
    p.add_argument("--json")
    p.add_argument("--ignore-scene", action="store_true",
                   help="pair on (task, init_state) only, to compare runs with different scene seeds")
    args = p.parse_args()

    old, new = load(args.old, not args.ignore_scene), load(args.new, not args.ignore_scene)
    keys = sorted(set(old) & set(new))
    if not keys:
        raise SystemExit("no states in common")

    rows = []
    for key in keys:
        a, b = old[key], new[key]
        ka, na = sum(r["success"] for r in a), len(a)
        kb, nb = sum(r["success"] for r in b), len(b)
        rows.append(
            {
                "task": key[0],
                "init_state": key[1],
                "old": [ka, na],
                "new": [kb, nb],
                "p_regress": fisher_less(kb, nb, ka, na),
                "p_improve": fisher_less(ka, na, kb, nb),
                "prob_worse": prob_worse(kb, nb, ka, na),
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
    regressed = bh([r["p_regress"] for r in rows], args.alpha)
    improved = bh([r["p_improve"] for r in rows], args.alpha)

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
        "regressed_bh": len(regressed),
        "improved_bh": len(improved),
        "prob_worse_over_0.9": sum(r["prob_worse"] > 0.9 for r in rows),
        "prob_better_over_0.9": sum(r["prob_worse"] < 0.1 for r in rows),
        "alpha": args.alpha,
    }

    print(f"{summary['states']} states, repeats {summary['repeats']}")
    print(f"success  old {summary['success_old']:.3f}  new {summary['success_new']:.3f}  "
          f"delta {summary['delta']:+.3f}  95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    print(f"single rollout per state: {naive_neg} states flip to failure, {naive_pos} to success")
    print(f"per-state tests, BH at {args.alpha}: {len(regressed)} regressed, {len(improved)} improved")
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
