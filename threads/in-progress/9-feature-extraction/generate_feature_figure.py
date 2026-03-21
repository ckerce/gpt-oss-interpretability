#!/usr/bin/env python3
"""Generate feature-space PCA scatter colored by task family.

Loads the 257D feature vectors from all three task runs, projects onto
the first two principal components (computed jointly), and colors by
task family to show whether computational modes cluster.

Output: figures/fig8_feature_pca.{pdf,png}
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

FIGS = Path("figures")
FIGS.mkdir(exist_ok=True)

TASKS = {
    "induction": {"dir": "runs/features_induction", "color": "#2176AE", "marker": "o"},
    "recency": {"dir": "runs/features_recency", "color": "#D64045", "marker": "s"},
    "syntax": {"dir": "runs/features_syntax", "color": "#7B2D8E", "marker": "^"},
}
TASK_LABELS = {
    "induction": "Induction",
    "recency": "Recency bias",
    "syntax": "Syntax agreement",
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

    # Load all features and tokens
    all_features = []
    all_labels = []
    all_tokens = []
    for task, info in TASKS.items():
        feats = torch.load(Path(info["dir"]) / "all_features.pt",
                           map_location="cpu", weights_only=True).numpy()
        tokens = json.loads((Path(info["dir"]) / "all_tokens.json").read_text())
        all_features.append(feats)
        all_labels.extend([task] * len(feats))
        all_tokens.extend(tokens)

    X = np.vstack(all_features)
    # Center and PCA
    X_centered = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    pcs = X_centered @ Vt[:2].T
    var_explained = (S[:2] ** 2) / (S ** 2).sum()

    fig, ax = plt.subplots(figsize=(7, 5.5))

    for task, info in TASKS.items():
        mask = [l == task for l in all_labels]
        ax.scatter(pcs[mask, 0], pcs[mask, 1],
                   c=info["color"], marker=info["marker"], s=60,
                   label=TASK_LABELS[task], alpha=0.8, edgecolors="white",
                   linewidth=0.5, zorder=3)

    # Annotate a few tokens
    for i, (tok, label) in enumerate(zip(all_tokens, all_labels)):
        tok_str = tok if isinstance(tok, str) else str(tok)
        if len(tok_str) > 6:
            tok_str = tok_str[:5] + "..."
        ax.annotate(tok_str, (pcs[i, 0], pcs[i, 1]),
                    fontsize=7, alpha=0.6,
                    xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%} variance)")
    ax.set_title("Feature-Space PCA: 39 Tokens Across 3 Task Families\n"
                 "(257D computational-mode features, gpt-oss-20b)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    _save(fig, "fig8_feature_pca")
    plt.close(fig)


if __name__ == "__main__":
    main()
