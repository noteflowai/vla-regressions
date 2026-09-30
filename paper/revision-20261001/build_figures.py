"""Build vector manuscript figures directly from archived binary episode outcomes."""
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
from scipy.stats import binom

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = REPO / "results"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
spec = importlib.util.spec_from_file_location("analysis", REPO / "analyze.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
    "axes.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.unicode_minus": False, "pdf.fonttype": 42, "svg.fonttype": "none",
})
BLUE, ORANGE, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#687783"
inputs, facts = {}, {}


def read(path):
    inputs[str(path.relative_to(REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path.read_text()


def load(path):
    rows = [json.loads(line) for line in read(path).splitlines()]
    keys = [(r["task"], r["init_state"], r["scene_seed"], r["repeat"]) for r in rows]
    assert len(keys) == len(set(keys)) and all(type(r["success"]) is bool for r in rows)
    return dict(zip(keys, rows))


def save(fig, name):
    # Fixed physical size: no cropping that changes the final manuscript font scale.
    fig.savefig(OUT / (name + ".pdf"), metadata={
        "Title": name.replace("-", " "), "Author": "Qiang Guo",
        "Subject": "Archived outcomes; no additional rollouts",
        "CreationDate": None, "ModDate": None,
    })
    fig.savefig(OUT / (name + ".svg"))
    fig.savefig(OUT / (name + ".png"), dpi=180)
    plt.close(fig)


old = load(DATA / "pilot/libero_10-fp32.jsonl")
states = sorted({k[:3] for k in old})
assert len(old) == 500 and len(states) == 100
variants = ["w4", "w3", "steps2", "bf16"]
labels = ["W4", "W3", "2 steps", "BF16"]
matrix, intervals, discoveries = [], [], []
for name in variants:
    new = load(DATA / f"pilot/libero_10-{name}.jsonl")
    assert old.keys() == new.keys() and all(old[k]["seed"] == new[k]["seed"] for k in old)
    diffs, pvalues = [], []
    for state in states:
        keys = sorted(k for k in old if k[:3] == state)
        assert len(keys) == 5
        diffs.append(sum(int(new[k]["success"])-int(old[k]["success"]) for k in keys)/5)
        harm = sum(old[k]["success"] and not new[k]["success"] for k in keys)
        gain = sum(not old[k]["success"] and new[k]["success"] for k in keys)
        pvalues.append(analysis.paired_pvalue(harm, gain))
    rng = random.Random(1)
    boot = sorted(sum(rng.choice(diffs) for _ in diffs)/100 for _ in range(4000))
    matrix.append(diffs)
    intervals.append([boot[100]*100, boot[3900]*100])
    discoveries.append(len(analysis.by(pvalues, .1)))
matrix = np.asarray(matrix)*100
means = matrix.mean(axis=1)
assert discoveries == [0, 0, 0, 0]
facts["pilot"] = dict(states=100, repeats=5, variants=variants,
                     delta_pp=means.tolist(), descriptive_ci95_pp=intervals,
                     by_discoveries=discoveries)

fig = plt.figure(figsize=(5.5, 3.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2], width_ratios=[1, .025],
                     left=.15, right=.87, top=.91, bottom=.15, hspace=.85)
ax = fig.add_subplot(gs[0, 0])
image = ax.imshow(matrix, aspect="auto", cmap="RdBu", norm=TwoSlopeNorm(0, -100, 100))
ax.set_yticks(range(4), labels)
ax.set_xticks(np.arange(10)*10+4.5, [str(i) for i in range(10)])
ax.set_xlabel("Task (10 initial states per task)")
ax.set_title("(a) All 100 states: new - old success", loc="left", pad=7)
for x in np.arange(9.5, 99, 10):
    ax.axvline(x, lw=.65, color=GREY, alpha=.6)
selected = states.index(next(s for s in states if s[:2] == (3, 3)))
ax.add_patch(Rectangle((selected-.5, 2.5), 1, 1, fill=False, lw=1.2, edgecolor="black"))
cb = fig.colorbar(image, cax=fig.add_subplot(gs[0, 1]), ticks=[-100, 0, 100])
cb.set_label("Change (pp)")
ax = fig.add_subplot(gs[1, 0])
cis = np.asarray(intervals)
ax.errorbar(means, np.arange(4), xerr=[means-cis[:, 0], cis[:, 1]-means],
            fmt="o", color=BLUE, capsize=3, markersize=4)
ax.axvline(0, color=GREY, lw=.8)
ax.set_yticks(range(4), labels)
ax.invert_yaxis()
ax.set(xlim=(-30, 5), xlabel="Aggregate change (pp)")
ax.set_title("(b) Descriptive 95% state-bootstrap intervals", loc="left", pad=7)
ax.grid(axis="x", alpha=.15)
save(fig, "statewise-pilot")

a = load(DATA / "followup/libero_10-t3s3-fp32.jsonl")
b = load(DATA / "followup/libero_10-t3s3-bf16.jsonl")
assert a.keys() == b.keys() and len(a) == 20
keys = sorted(a)
assert all(a[k]["seed"] == b[k]["seed"] for k in keys)
assert not {r["seed"] for r in old.values()} & {r["seed"] for r in a.values()}
assert all(r["steps"] == 520 for rows in (a, b) for r in rows.values() if not r["success"])
pairs = np.array([[int(a[k]["success"]), int(b[k]["success"])] for k in keys])
table = np.zeros((2, 2), dtype=int)
for x, y in pairs:
    table[x, y] += 1
harm, gain = int(table[1, 0]), int(table[0, 1])
pvalue = float(binom.sf(harm-1, harm+gain, .5))
assert (harm, gain) == (15, 1) and pairs.sum(axis=0).tolist() == [18, 4]
facts["followup"] = dict(pairs=20, fp32_success=18, bf16_success=4, harm=harm,
                        gain=gain, exact_p=pvalue, table=table.tolist(),
                        pilot_policy_seeds_excluded=True)
fig = plt.figure(figsize=(5.5, 3.0))
gs = fig.add_gridspec(2, 2, height_ratios=[.65, 1.5], width_ratios=[1.45, 1],
                     left=.16, right=.95, bottom=.18, top=.90, hspace=.78, wspace=.9)
ax = fig.add_subplot(gs[0, :])
ax.imshow(pairs.T, aspect="auto", cmap=ListedColormap(["#EAEFF2", BLUE]), vmin=0, vmax=1)
ax.set_yticks([0, 1], ["FP32", "BF16"])
ax.set_xticks([0, 4, 9, 14, 19], [1, 5, 10, 15, 20])
ax.set_title("(a) Fresh paired repeats: blue = success", loc="left", pad=7)
ax = fig.add_subplot(gs[1, 0])
display = table[::-1, ::-1]
ax.imshow(display, cmap="Blues", vmin=0, vmax=20, aspect="auto")
ax.set_xticks([0, 1], ["BF16\nsuccess", "BF16\ntimeout"])
ax.set_yticks([0, 1], ["FP32\nsuccess", "FP32\ntimeout"])
for (i, j), value in np.ndenumerate(display):
    ax.text(j, i, str(value), ha="center", va="center", fontsize=13,
            color="white" if value > 10 else "#18252E")
ax.set_title("(b) Paired outcome table", loc="left", pad=7)
ax = fig.add_subplot(gs[1, 1])
ax.bar(["Harm", "Gain"], [harm, gain], color=[ORANGE, BLUE], width=.65)
ax.set(ylim=(0, 18), ylabel="Discordant pairs", yticks=[0, 5, 10, 15])
ax.set_title("(c) Exact test", loc="left", pad=7)
ax.grid(axis="y", alpha=.15)
for i, value in enumerate([harm, gain]):
    ax.text(i, value+.4, str(value), ha="center", fontsize=8)
save(fig, "fresh-confirmation")

fig, ax = plt.subplots(figsize=(5.5, 1.95))
fig.subplots_adjust(left=.01, right=.99, bottom=.01, top=.99)
ax.set(xlim=(0, 10), ylim=(0, 3))
ax.axis("off")
boxes = [
    (.10, "Paired pilot", "100 states x 5 repeats\nFixed scene seeds\nOld/new shared seeds", BLUE),
    (3.50, "Candidate screen", "Exact paired tests\nBY q = 0.1 per update\n0 corrected discoveries", ORANGE),
    (6.90, "Fresh confirmation", "One selected BF16 state\n20 new paired repeats\n18/20 old vs 4/20 new", GREEN),
]
for x, title, body, color in boxes:
    ax.add_patch(FancyBboxPatch((x, 1.05), 2.95, 1.65, boxstyle="round,pad=.05",
                               edgecolor=color, facecolor="#F5F8FA", lw=1))
    ax.text(x+.12, 2.46, title, fontweight="bold", va="top", fontsize=8, color=color)
    ax.text(x+.12, 2.00, body, va="top", fontsize=7.5, linespacing=1.6)
for start, end in [(3.1, 3.4), (6.5, 6.8)]:
    ax.annotate("", xy=(end, 1.85), xytext=(start, 1.85),
                arrowprops={"arrowstyle": "-|>", "color": GREY, "lw": 1})
ax.text(.15, .49, "Old-versus-old controls", fontsize=8, color=GREY)
ax.text(3.55, .49, "Fresh data test", fontsize=8, color=GREY)
ax.text(6.95, .49, "One-state conclusion only", fontsize=8, color=GREY)
save(fig, "paired-protocol")

vqb = json.loads(read(DATA / "vqb/reanalysis.json"))
fig, ax = plt.subplots(figsize=(5.5, 3.1))
fig.subplots_adjust(left=.12, right=.97, bottom=.18, top=.96)
x = np.linspace(0, 1, 200)
ax.plot(x, 2*x*(1-x), color=GREY, lw=1, ls="--", label="Homogeneous reference: 2p(1-p)")
updates = vqb["updates"]
ax.scatter([(u["success_base"]+u["success_new"])/2 for u in updates],
           [(u["to_fail"]+u["to_success"])/u["states"] for u in updates],
           s=9, color=BLUE, alpha=.28, lw=0, label="VLAQuantBench: baseline vs update")
repeated = vqb["repeated"]
ax.scatter([np.mean(r["success"]) for r in repeated],
           [np.mean(r["pair_flips"])/r["states"] for r in repeated],
           s=22, color=ORANGE, marker="x", lw=1, label="VLAQuantBench: same config, new seed")
own = []
for base, rerun in [("fp32", "fp32-seed2000"), ("w3", "w3-seed2000")]:
    xold = old if base == "fp32" else load(DATA / f"pilot/libero_10-{base}.jsonl")
    xnew = load(DATA / f"pilot/libero_10-{rerun}.jsonl")
    assert xold.keys() == xnew.keys()
    first = [k for k in xold if k[-1] == 0]
    flips = sum(xold[k]["success"] != xnew[k]["success"] for k in first)/100
    success = sum(int(r["success"]) for rows in (xold, xnew) for r in rows.values())/1000
    own.append((success, flips))
ax.scatter(*zip(*own), s=36, facecolor="none", edgecolor="#18252E", lw=1,
           label="This work: same policy, new seed")
ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean success rate", ylabel="Fraction of states flipped")
ax.legend(fontsize=7, frameon=False, loc="upper left")
save(fig, "noise-floor")
facts["public_data"] = dict(repeated_configurations=len(repeated), own_controls=own)

h100 = sum(1/i for i in range(1, 101))
n = np.arange(1, 21)
fig, ax = plt.subplots(figsize=(5.5, 2.5))
fig.subplots_adjust(left=.15, right=.96, bottom=.23, top=.94)
ax.semilogy(n, 2.**(-n), "o-", color=BLUE, lw=1.3, ms=3,
            label="Smallest exact p: all pairs harmful")
threshold = .1/(100*h100)
ax.axhline(threshold, color=ORANGE, ls="--", lw=1.3,
           label=f"BY rank-1 threshold: {threshold:.6f}")
ax.scatter([5], [1/32], color=ORANGE, s=35, zorder=3)
ax.annotate("Pilot n = 5", (5, 1/32), xytext=(7, .09), fontsize=8,
            arrowprops=dict(arrowstyle="-", color=GREY))
ax.set(xlim=(1, 20), ylim=(5e-7, 1), xticks=[1, 5, 10, 13, 15, 20],
       xlabel="Paired repeats (best possible discordance pattern)", ylabel="One-sided exact p")
ax.legend(loc="upper right", frameon=False, fontsize=7)
ax.grid(axis="y", which="major", alpha=.12)
save(fig, "test-resolution")
facts["test_resolution"] = dict(min_p_n5=1/32, by_rank1_threshold=threshold,
                               harmonic100=h100, smallest_all_harmful_n_rank1=13,
                               is_general_power_calculation=False)
newbf = load(DATA / "pilot/libero_10-bf16.jsonl")
facts["bf16_changed_episode_lengths"] = sum(old[k]["steps"] != newbf[k]["steps"] for k in old)
(HERE / "figure-provenance.json").write_text(json.dumps({
    "sources": inputs, "facts": facts, "new_rollouts_in_manuscript": 0,
    "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}, indent=2)+"\n")
print(json.dumps({"figures": 5, "facts": facts}))
