"""Reanalysis of VLAQuantBench's public LIBERO episode records, state by state.

VLAQuantBench (arXiv 2609.25376, MIT licence, https://github.com/jiuyixu25/VLAQuantBench) runs
each configuration once per (task, initial state), and a few configurations again with seeds 1
and 2. Those repeats show how many states change outcome when nothing changes at all: the noise
floor that any single-rollout, per-state comparison has to clear.

Reports:
  1. repeated cells: success per seed, states that change outcome between seeds, and how the
     states split into always-fail / always-succeed / mixed, against what equal per-state
     success rates would give (a test for real per-state difficulty);
  2. same-seed pairing: whether two configurations run with the same seed share random numbers
     (if they did, flips between them at the same seed would be fewer than across seeds);
  3. updates: baseline against every other configuration of the same model and suite, seed 0,
     with the flips a single rollout per state shows, next to the noise floor of that
     configuration where it has repeats.

Usage: reanalyze_vqb.py VQB_REPO [--json out.json]
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

SEEDS = ("", "seed1-", "seed2-")
BASELINE = "rtn-BASELINE-e2e"


def load(path):
    states = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("kind") != "episode":
            continue
        states[(r["task_id"], r["extra"].get("init_state_id", r["episode"]))] = {
            "success": bool(r["success"]),
            "steps": r["steps"],
            "seed": r["seed"],
        }
    return states


def runs_of(cell_dir, name):
    return [load(cell_dir / f"{p}{name}.jsonl") for p in SEEDS if (cell_dir / f"{p}{name}.jsonl").exists()]


def flips(a, b, keys):
    neg = sum(a[k]["success"] and not b[k]["success"] for k in keys)
    pos = sum(b[k]["success"] and not a[k]["success"] for k in keys)
    return neg, pos


def split(runs, keys):
    """States with 0 successes, all successes, and mixed; and the same under equal rates."""
    n = len(runs)
    ks = [sum(r[k]["success"] for r in runs) for k in keys]
    p = sum(ks) / (n * len(keys))
    exp_none = len(keys) * (1 - p) ** n
    exp_all = len(keys) * p**n
    observed = {"none": ks.count(0), "all": ks.count(n), "mixed": len(ks) - ks.count(0) - ks.count(n)}
    expected = {"none": exp_none, "all": exp_all, "mixed": len(keys) - exp_none - exp_all}
    return observed, expected


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("repo", type=Path)
    ap.add_argument("--json")
    args = ap.parse_args()

    root = args.repo / "results" / "libero"
    out = {"repeated": [], "pairing": [], "updates": []}

    print("1. Same configuration, different seeds: states that change outcome")
    print(f"{'suite/model':28s} {'configuration':36s} {'success by seed':22s} {'flips':12s} "
          f"{'none/all/mixed obs':19s} {'if rates equal':s}")
    repeated = {}
    for first in sorted(root.glob("*/*/seed1-*.jsonl")):
        cell_dir, name = first.parent, first.name[len("seed1-"):-len(".jsonl")]
        runs = runs_of(cell_dir, name)
        keys = sorted(set.intersection(*(set(r) for r in runs)))
        rates = [sum(r[k]["success"] for k in keys) / len(keys) for r in runs]
        pair = [sum(flips(a, b, keys)) for a, b in itertools.combinations(runs, 2)]
        obs, exp = split(runs, keys)
        cell = f"{cell_dir.parent.name}/{cell_dir.name}"
        repeated[(cell, name)] = sum(pair) / len(pair) / len(keys)
        print(f"{cell:28s} {name:36s} {' '.join(f'{x:.3f}' for x in rates):22s} "
              f"{','.join(map(str, pair)) + f'/{len(keys)}':12s} "
              f"{obs['none']:>3}/{obs['all']:>3}/{obs['mixed']:>3}{'':7s} "
              f"{exp['none']:5.1f}/{exp['all']:5.1f}/{exp['mixed']:5.1f}")
        out["repeated"].append({"cell": cell, "config": name, "states": len(keys), "success": rates,
                                "pair_flips": pair, "split_observed": obs, "split_if_equal": exp})

    print("\n2. Do runs with the same seed share random numbers? (flips between two configurations)")
    for cell_dir in sorted(root.glob("*/*")):
        names = sorted({p.name[len("seed1-"):-len(".jsonl")] for p in cell_dir.glob("seed1-*.jsonl")})
        names = [n for n in names if len(runs_of(cell_dir, n)) == 3]
        for x, y in itertools.combinations(names, 2):
            X, Y = runs_of(cell_dir, x), runs_of(cell_dir, y)
            keys = sorted(set.intersection(*(set(r) for r in X + Y)))
            same = [sum(flips(X[i], Y[i], keys)) for i in range(3)]
            cross = [sum(flips(X[i], Y[j], keys)) for i in range(3) for j in range(3) if i != j]
            cell = f"{cell_dir.parent.name}/{cell_dir.name}"
            print(f"{cell:28s} {x} vs {y}: same seed {sum(same) / 3:.1f}, different seeds {sum(cross) / 6:.1f}")
            out["pairing"].append({"cell": cell, "a": x, "b": y, "same_seed": same, "cross_seed": cross})

    print("\n3. Baseline against each configuration, seed 0, one rollout per state")
    print(f"{'suite/model':28s} {'configuration':44s} {'base':>6s} {'new':>6s} {'to fail':>8s} "
          f"{'to succ':>8s} {'flip %':>7s} {'noise %':>8s}")
    for cell_dir in sorted(root.glob("*/*")):
        base_path = cell_dir / f"{BASELINE}.jsonl"
        if not base_path.exists():
            continue
        base = load(base_path)
        cell = f"{cell_dir.parent.name}/{cell_dir.name}"
        for path in sorted(cell_dir.glob("*.jsonl")):
            name = path.stem
            if name == BASELINE or name.startswith("seed"):
                continue
            new = load(path)
            keys = sorted(set(base) & set(new))
            if not keys:
                continue
            neg, pos = flips(base, new, keys)
            sb = sum(base[k]["success"] for k in keys) / len(keys)
            sn = sum(new[k]["success"] for k in keys) / len(keys)
            noise = repeated.get((cell, name))
            print(f"{cell:28s} {name:44s} {sb:6.3f} {sn:6.3f} {neg:8d} {pos:8d} "
                  f"{100 * (neg + pos) / len(keys):7.1f} {'' if noise is None else f'{100 * noise:8.1f}'}")
            out["updates"].append({"cell": cell, "config": name, "states": len(keys), "success_base": sb,
                                   "success_new": sn, "to_fail": neg, "to_success": pos,
                                   "noise_floor_new": noise})

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
