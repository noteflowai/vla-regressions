"""Plot the measured cleanup events, not policy efficacy or a causal host effect."""
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
events = data["cpu_release_events"]
assert len(events) == 2
gib = 1024 ** 3
labels = ["After FP32 episode", "After denied next load"]
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none",
    "axes.spines.top": False, "axes.spines.right": False,
})
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5))
x = np.arange(len(events))
for ax, key, title in (
        (axes[0], "process_rss_bytes", "Owned process RSS"),
        (axes[1], "host_available_bytes", "Host available memory")):
    for offset, phase, color in ((-.18, "before", "#496B99"), (.18, "after", "#009E73")):
        values = [e["report"][phase][key] / gib for e in events]
        bars = ax.bar(x + offset, values, width=.34, color=color, label=phase.capitalize())
        if key == "host_available_bytes":
            for bar, value in zip(bars, values, strict=True):
                ax.text(bar.get_x() + bar.get_width() / 2, value - .7, f"{value:.2f}",
                        ha="center", va="top", fontsize=9, color="white", weight="bold")
        else:
            ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=3, fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylabel("GiB")
    ax.set_title(title, loc="left", weight="bold")
    ax.set_ylim(0, max(20 if key == "host_available_bytes" else 4.4,
                       ax.get_ylim()[1] * 1.12))
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
axes[1].axhline(18, color="#B14433", linestyle="--", linewidth=1.2)
axes[1].text(.99, 18.8, "18 GiB admission gate", transform=axes[1].get_yaxis_transform(),
             ha="right", color="#B14433", fontsize=9)
axes[0].legend(frameon=False, ncol=2)
fig.suptitle("V5 engineering control: measured allocator release", x=.07, ha="left",
             fontsize=14, weight="bold")
fig.text(.07, .025,
         "One FP32 episode; the next load was denied. Host changes include concurrent work.\n"
         "Cleanup measurements do not qualify the complete control or establish policy efficacy.",
         fontsize=9, color="#444444")
fig.subplots_adjust(left=.07, right=.985, top=.80, bottom=.22, wspace=.30)
args.output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.output.with_suffix(".svg"), metadata={"Date": None})
fig.savefig(args.output.with_suffix(".png"), dpi=180)
plt.close(fig)
