#!/usr/bin/env python3
"""Generate selectivity comparison figure across task families.

Shows channelized vs whole-vector mean selectivity for recency (DST-independent)
and induction (DST-cascade), with random baselines.

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

FAMILIES = [
    {
        "name": "Recency bias\n(DST-cascade)",
        "path": "runs/selectivity_c71_recency_full/selectivity_comparison.json",
    },
    {
        "name": "Induction\n(DST-cascade)",
        "path": "runs/selectivity_c71_induction_full/selectivity_comparison.json",
    },
]

CONDITIONS = ["channelized", "whole_vector", "random_channel", "random_direction"]
CONDITION_COLORS = {
    "channelized": "#2176AE",
    "whole_vector": "#D64045",
    "random_channel": "#CCCCCC",
    "random_direction": "#888888",
}
CONDITION_LABELS = {
    "channelized": "Channelized",
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

    # Compute mean selectivity per family per condition
    family_means = []
    for fam in FAMILIES:
        data = json.loads(Path(fam["path"]).read_text())
        results = data["results"]
        means = {}
        for cond in CONDITIONS:
            means[cond] = np.mean([r[cond]["selectivity"] for r in results])
        family_means.append({"name": fam["name"], "means": means,
                             "n_folds": len(results)})

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(FAMILIES))
    width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]

    for cond, offset in zip(CONDITIONS, offsets):
        vals = [fm["means"][cond] for fm in family_means]
        bars = ax.bar(x + offset * width, vals, width,
                      color=CONDITION_COLORS[cond],
                      label=CONDITION_LABELS[cond],
                      edgecolor="white", linewidth=0.5)
        # Label bars
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([fm["name"] + f"\n(n={fm['n_folds']})" for fm in family_means],
                       fontsize=10)
    ax.set_ylabel("Mean selectivity (higher = more selective)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title("Steering Selectivity Across Task Families",
                 fontsize=13, fontweight="bold")

    fig.tight_layout()
    _save(fig, "fig7_selectivity_comparison")
    plt.close(fig)


if __name__ == "__main__":
    main()
