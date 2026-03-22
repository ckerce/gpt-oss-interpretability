#!/usr/bin/env python3
"""Generate channel probe figure: held-out accuracy by head and task family.

Shows per-head held-out sign accuracy for each task family, with the
promotion gate threshold and null ceiling marked. Highlights the
concentration of signal in head H4.

Output: figures/fig6_channel_probe.{pdf,png}
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS = Path("runs") / "channel_probe_c71_phase1"
FIGS = Path("figures")
FIGS.mkdir(exist_ok=True)

FAMILY_COLORS = {
    "recency_bias": "#D64045",
    "induction": "#2176AE",
    "coreference": "#57B894",
}
FAMILY_LABELS = {
    "recency_bias": "Recency bias",
    "induction": "Induction",
    "coreference": "Coreference",
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
    data = json.loads((RUNS / "c_71_channel_probe.json").read_text())

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    for ax, family in zip(axes, ["recency_bias", "induction", "coreference"]):
        ranking = data["ranking"][family]
        rows = ranking["rows"]

        # Deduplicate: take best accuracy per head (rows repeat across layers)
        head_best = {}
        for row in rows:
            h = row["head"]
            acc = row["heldout_sign_accuracy"]
            if h not in head_best or acc > head_best[h]["acc"]:
                head_best[h] = {"acc": acc, "passes": row["passes_gate"]}

        heads = sorted(head_best.keys())
        accs = [head_best[h]["acc"] for h in heads]
        passes = [head_best[h]["passes"] for h in heads]
        color = FAMILY_COLORS[family]

        # Bar chart
        bars = ax.bar(heads, accs, color=[color if p else "#CCCCCC" for p in passes],
                      edgecolor="white", linewidth=0.5)

        # Null ceiling
        null_ceiling = max(r["null_accuracy_ceiling"] for r in rows)
        ax.axhline(null_ceiling, color="#888888", linestyle="--", linewidth=1,
                   label=f"Null ceiling ({null_ceiling:.2f})")

        # Promotion gate
        ax.axhline(0.70, color="#333333", linestyle=":", linewidth=1,
                   label="Promotion gate (0.70)")

        ax.set_xlabel("Head index")
        ax.set_title(FAMILY_LABELS[family], fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(heads)
        ax.legend(fontsize=8, loc="upper right")

    axes[0].set_ylabel("Held-out sign accuracy")
    fig.suptitle("Channel Probe: Per-Head Accuracy by Task Family (DST-cascade)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig6_channel_probe")
    plt.close(fig)


if __name__ == "__main__":
    main()
