#!/usr/bin/env python3
"""Generate selectivity comparison figure.

Shows channelized vs whole-vector vs random selectivity across held-out
cases, demonstrating that channelized steering achieves comparable
selectivity with far fewer dimensions.

Output: figures/fig7_selectivity_comparison.{pdf,png}
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGS = Path("figures")
FIGS.mkdir(exist_ok=True)

SELECTIVITY_RUNS = [
    ("Full (all heads)", "runs/selectivity_e2_recency_full/selectivity_comparison.json"),
]

CONDITION_COLORS = {
    "channelized": "#2176AE",
    "whole_vector": "#D64045",
    "random_channel": "#CCCCCC",
    "random_direction": "#888888",
}
CONDITION_LABELS = {
    "channelized": "Channelized (L0 H5)",
    "whole_vector": "Whole-vector",
    "random_channel": "Random channel",
    "random_direction": "Random direction",
}


def _style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "figure.dpi": 150,
    })


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    print(f"  saved {name}.pdf / .png")


def main():
    _style()

    data = json.loads(Path(SELECTIVITY_RUNS[0][1]).read_text())
    results = data["results"]

    case_ids = [r["heldout_case_id"] for r in results]
    conditions = ["channelized", "whole_vector", "random_channel", "random_direction"]

    selectivities = {
        cond: [r[cond]["selectivity"] for r in results]
        for cond in conditions
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                    gridspec_kw={"width_ratios": [3, 1.2]})

    # Panel 1: grouped bar chart by case
    x = np.arange(len(case_ids))
    width = 0.2
    offsets = [-1.5, -0.5, 0.5, 1.5]

    for cond, offset in zip(conditions, offsets):
        vals = selectivities[cond]
        ax1.bar(x + offset * width, vals, width,
                color=CONDITION_COLORS[cond],
                label=CONDITION_LABELS[cond],
                edgecolor="white", linewidth=0.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels([c.replace("recency_", "rec_") for c in case_ids],
                        fontsize=9)
    ax1.set_ylabel("Selectivity")
    ax1.set_xlabel("Held-out case")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_title("Per-Case Selectivity Comparison", fontsize=12, fontweight="bold")

    # Panel 2: mean selectivity summary
    means = {cond: np.mean(selectivities[cond]) for cond in conditions}
    bars = ax2.barh(range(len(conditions)), [means[c] for c in conditions],
                    color=[CONDITION_COLORS[c] for c in conditions],
                    edgecolor="white", linewidth=0.5)
    ax2.set_yticks(range(len(conditions)))
    ax2.set_yticklabels([CONDITION_LABELS[c] for c in conditions], fontsize=9)
    ax2.set_xlabel("Mean selectivity")
    ax2.set_title("Mean Across Cases", fontsize=12, fontweight="bold")
    ax2.axvline(0, color="black", linewidth=0.5)

    for i, (cond, bar) in enumerate(zip(conditions, bars)):
        ax2.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                 f"{means[cond]:.2f}", va="center", fontsize=9)

    fig.suptitle("Steering Selectivity: Channelized vs Whole-Vector (Recency Bias, E2 Model)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig7_selectivity_comparison")
    plt.close(fig)


if __name__ == "__main__":
    main()
