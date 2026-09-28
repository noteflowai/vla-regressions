"""Figure: states that change outcome, against success rate.

Same-configuration seed repeats (the noise floor) and baseline-versus-update comparisons from
`reanalyze_vqb.py --json`, with the flip rate two independent rollouts of a state give if every
state has the same success rate p: 2p(1-p).

Usage: plot_noise_floor.py results/vqb/reanalysis.json out.pdf
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    data = json.load(open(sys.argv[1]))
    fig, ax = plt.subplots(figsize=(3.4, 2.6))

    p = np.linspace(0, 1, 200)
    ax.plot(p, 2 * p * (1 - p), color="0.6", lw=1, ls="--", label="$2p(1-p)$, equal states")

    up_x, up_y = [], []
    for u in data["updates"]:
        up_x.append((u["success_base"] + u["success_new"]) / 2)
        up_y.append((u["to_fail"] + u["to_success"]) / u["states"])
    ax.scatter(up_x, up_y, s=6, color="tab:blue", alpha=0.35, lw=0, label="baseline vs update")

    rep_x, rep_y = [], []
    for r in data["repeated"]:
        rep_x.append(np.mean(r["success"]))
        rep_y.append(np.mean(r["pair_flips"]) / r["states"])
    ax.scatter(rep_x, rep_y, s=14, color="tab:red", marker="x", lw=1, label="same config, new seed")

    ax.set_xlabel("mean success rate of the two runs")
    ax.set_ylabel("fraction of states flipped")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(sys.argv[2])


if __name__ == "__main__":
    main()
