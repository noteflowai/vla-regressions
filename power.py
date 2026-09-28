"""Power of the per-state test: probability that a state is flagged as regressed.

For a state with success rates p_old and p_new and n rollouts of each variant, the one-sided
Fisher exact test at level alpha (a single state, before any multiplicity correction) flags it
with the probability printed here. A single rollout per state flags it (old succeeds, new fails)
with probability p_old * (1 - p_new), and flags an unchanged state with p (1 - p).

Usage: power.py [--alpha 0.05]
"""

import argparse
from math import comb

from analyze import fisher_less


def power(p_old, p_new, n, alpha):
    pmf = lambda k, p: comb(n, k) * p**k * (1 - p) ** (n - k)
    return sum(
        pmf(ko, p_old) * pmf(kn, p_new)
        for ko in range(n + 1)
        for kn in range(n + 1)
        if fisher_less(kn, n, ko, n) <= alpha
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    ns = (1, 5, 10, 20)
    print(f"alpha {args.alpha}; single rollout = naive flip old-success/new-failure")
    print(f"{'p_old':>6} {'p_new':>6}  {'naive':>6}  " + "  ".join(f"n={n:<4}" for n in ns[1:]))
    for p_old in (0.95, 0.8):
        for p_new in sorted({p_old, 0.8, 0.5, 0.2, 0.0}, reverse=True):
            if p_new > p_old:
                continue
            row = [power(p_old, p_new, n, args.alpha) for n in ns[1:]]
            print(f"{p_old:6.2f} {p_new:6.2f}  {p_old * (1 - p_new):6.3f}  " + "  ".join(f"{x:6.3f}" for x in row))


if __name__ == "__main__":
    main()
