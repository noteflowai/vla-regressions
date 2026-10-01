"""Plot a qualified engineering control, without loading policies or raw frames."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("audit", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
data = json.loads(args.audit.read_text())
events = data["isolated_side_lifecycle_events"]
assert data["complete_control_qualified"] is True and len(events) == 4
assert all(e["execution"]["clean_process_exit"] is True
           and e["execution"]["worker_exit_code"] == 0
           and e["parent_torch_imported"] is False for e in events)
gib = 1024 ** 3
labels = [f"Pair {int(e['episode_name'].split('-')[2]) + 1}\n{e['side']} (FP32)"
          for e in events]
x = np.arange(4)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none",
    "axes.spines.top": False, "axes.spines.right": False,
})
fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.8))
ax = axes[0]
for offset, key, color, label in (
        (-.18, "admission_before_exec", "#496B99", "Before child exec"),
        (.18, "resource_probe_after_exit", "#009E73", "After clean child exit")):
    values = [e[key]["host_available_bytes"] / gib for e in events]
    bars = ax.bar(x + offset, values, .34, color=color, label=label)
    ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=3, fontsize=8)
ax.axhline(18, color="#B14433", linestyle="--", linewidth=1.2, label="18 GiB gate")
ax.set_title("Host availability at lifecycle probes", loc="left", weight="bold")
ax.set_ylabel("GiB")
ax.set_ylim(0, max(25, max(values) * 1.18))
ax.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="none",
          framealpha=.96, fontsize=8)

ax = axes[1]
episode = np.array([e["episode_elapsed_seconds"] for e in events])
total = np.array([e["execution"]["elapsed_seconds"] for e in events])
assert np.all(total >= episode)
ax.bar(x, episode, .65, color="#009E73", label="Episode kernel")
bars = ax.bar(x, total - episode, .65, bottom=episode, color="#496B99",
              label="Remaining child lifecycle")
ax.bar_label(bars, labels=[f"{v:.1f}" for v in total], padding=3, fontsize=9)
ax.set_title("Measured child wall time", loc="left", weight="bold")
ax.set_ylabel("Seconds")
ax.set_ylim(0, max(total) * 1.22)
ax.legend(loc="upper right", frameon=False, fontsize=8)
for ax in axes:
    ax.set_xticks(x, labels)
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
fig.suptitle("V6: four owned workers, two qualified FP32 control pairs",
             x=.07, ha="left", fontsize=14, weight="bold")
fig.text(.07, .025,
         "Sequential workers; controller imports no Torch. Host values include concurrent work.\n"
         "Lifecycle remainder includes imports, hashing, loading and cleanup; no causal speedup or efficacy claim.",
         fontsize=9, color="#444444")
fig.subplots_adjust(left=.07, right=.985, top=.80, bottom=.23, wspace=.28)
args.output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.output.with_suffix(".svg"), metadata={"Date": None})
fig.savefig(args.output.with_suffix(".png"), dpi=180)
plt.close(fig)
