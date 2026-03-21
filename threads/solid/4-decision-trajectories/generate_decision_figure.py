#!/usr/bin/env python3
"""Generate a decision trajectory figure for key positions.

Shows the top-1 prediction at each layer for 3 key decision positions,
one from each task family, with decision transitions highlighted.

Output: figures/fig4_decision_trajectories.{pdf,png}
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RUNS = Path("runs")
FIGS = Path("figures")
FIGS.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
})

COLORS = {
    "recency": "#D64045",
    "syntax": "#7B2D8E",
    "induction": "#2176AE",
}


def _load_position(prompt_name, position_idx):
    """Load a specific position's trajectory from logit-lens data."""
    path = RUNS / f"logit_lens_{prompt_name}" / "logit_lens_data.json"
    data = json.loads(path.read_text())
    for tp in data["tracked_positions"]:
        if tp["position"] == position_idx:
            return tp, data["prompt"]
    return None, None


def _plot_position(ax, pos_data, prompt_label, color, title):
    """Plot target token probability trajectory with decision markers."""
    trajectory = pos_data["trajectory"]
    target_token = pos_data["target_token"]
    conv_layer = pos_data["convergence_layer"]

    layers = [e["layer"] for e in trajectory]

    # Target token probability at each layer
    target_probs = []
    for e in trajectory:
        lp = e.get("target_logprob", -50)
        prob = math.exp(lp) if lp > -50 else 0.0
        target_probs.append(prob)

    # Plot probability trajectory
    ax.plot(layers, target_probs, "-", color=color, linewidth=2, alpha=0.8, zorder=3)
    ax.fill_between(layers, 0, target_probs, color=color, alpha=0.1)

    # Mark convergence layer
    if conv_layer is not None and conv_layer < len(target_probs):
        ax.axvline(conv_layer, color=color, linestyle="--", alpha=0.5, linewidth=1)
        ax.annotate(f"conv L{conv_layer}",
                    xy=(conv_layer, target_probs[conv_layer]),
                    xytext=(conv_layer + 1, target_probs[conv_layer] + 0.05),
                    fontsize=8, color=color,
                    arrowprops=dict(arrowstyle="->", color=color, lw=0.8))

    # Mark decision transitions (top-1 changes) with vertical ticks
    prev_top1 = None
    for e in trajectory:
        top1 = e["top_token_ids"][0]
        if prev_top1 is not None and top1 != prev_top1:
            l = e["layer"]
            ax.plot(l, target_probs[l], "v", color="black", markersize=5,
                    zorder=5, alpha=0.6)
        prev_top1 = top1

    # Add top-1 token labels at key layers
    label_layers = set()
    prev_top1 = None
    for e in trajectory:
        top1 = e["top_token_ids"][0]
        if prev_top1 is not None and top1 != prev_top1:
            label_layers.add(e["layer"])
        prev_top1 = top1

    # Label a subset to avoid clutter
    labeled = 0
    for e in trajectory:
        l = e["layer"]
        top1_tok = e["top_tokens"][0].strip()
        if len(top1_tok) > 12:
            top1_tok = top1_tok[:10] + "…"
        if l in label_layers and labeled < 6:
            y = target_probs[l]
            offset = 0.03 if y < 0.5 else -0.06
            ax.text(l, y + offset, f"→{top1_tok}", fontsize=7,
                    ha="center", va="bottom", color="black", alpha=0.7,
                    rotation=45)
            labeled += 1

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Layer")
    ax.set_xlim(-0.5, 23.5)
    ax.set_ylim(-0.02, 1.05)


def main():
    print("Generating decision trajectory figure...")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Recency: position 12, target " small"
    pos_data, prompt = _load_position("recency", 12)
    _plot_position(axes[0], pos_data, "recency", COLORS["recency"],
                   'Recency: "...suitcase was too ___"\ntarget = "small"')
    axes[0].set_ylabel("P(target token)")

    # Syntax: position 11, target " can"
    pos_data, prompt = _load_position("syntax", 11)
    _plot_position(axes[1], pos_data, "syntax", COLORS["syntax"],
                   'Syntax: "...so they ___"\ntarget = "can"')

    # Induction: position 13, target " D"
    pos_data, prompt = _load_position("induction", 13)
    _plot_position(axes[2], pos_data, "induction", COLORS["induction"],
                   'Induction: "A7 B2 C9 D4 A7 B2 C9 ___"\ntarget = "D"')

    # Add legend for markers
    decision_marker = plt.Line2D([], [], marker="v", color="black", linestyle="None",
                                 markersize=5, alpha=0.6, label="Top-1 change (decision)")
    conv_line = plt.Line2D([], [], color="gray", linestyle="--", linewidth=1,
                           label="Convergence layer")
    fig.legend(handles=[decision_marker, conv_line], loc="lower center",
               ncol=2, fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Decision trajectories: P(target) vs layer with top-1 changes marked (gpt-oss-20b)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig4_decision_trajectories.{ext}",
                    bbox_inches="tight", dpi=200)
    print(f"  saved fig4_decision_trajectories.pdf / .png")
    plt.close(fig)


if __name__ == "__main__":
    main()
