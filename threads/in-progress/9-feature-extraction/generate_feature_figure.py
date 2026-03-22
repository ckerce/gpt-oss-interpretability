#!/usr/bin/env python3
"""Generate feature-space PCA scatter colored by task family.

Loads feature vectors from all task runs, projects onto the first two
principal components (computed jointly), and colors by task family to
show whether computational modes cluster.

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
    "induction": {"dir": "runs/features_induction_scaled", "color": "#2176AE", "marker": "o"},
    "recency_bias": {"dir": "runs/features_recency_bias_scaled", "color": "#D64045", "marker": "s"},
    "syntax_agreement": {"dir": "runs/features_syntax_agreement_scaled", "color": "#7B2D8E", "marker": "^"},
    "coreference": {"dir": "runs/features_coreference_scaled", "color": "#57B894", "marker": "D"},
    "capitalization": {"dir": "runs/features_capitalization_scaled", "color": "#FBB13C", "marker": "v"},
}
TASK_LABELS = {
    "induction": "Induction",
    "recency_bias": "Recency bias",
    "syntax_agreement": "Syntax agreement",
    "coreference": "Coreference",
    "capitalization": "Capitalization",
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
    total_tokens = 0
    for task, info in TASKS.items():
        run_dir = Path(info["dir"])
        if not run_dir.exists():
            print(f"  skipping {task} (no run directory)")
            continue
        feats = torch.load(run_dir / "all_features.pt",
                           map_location="cpu", weights_only=True).numpy()
        all_features.append(feats)
        all_labels.extend([task] * len(feats))
        total_tokens += len(feats)

    X = np.vstack(all_features)
    ndim = X.shape[1]

    # Center and PCA
    X_centered = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    pcs = X_centered @ Vt[:2].T
    var_explained = (S[:2] ** 2) / (S ** 2).sum()

    fig, ax = plt.subplots(figsize=(8, 6))

    for task, info in TASKS.items():
        mask = [l == task for l in all_labels]
        if not any(mask):
            continue
        ax.scatter(pcs[mask, 0], pcs[mask, 1],
                   c=info["color"], marker=info["marker"], s=30,
                   label=TASK_LABELS[task], alpha=0.7, edgecolors="white",
                   linewidth=0.3, zorder=3)

    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%} variance)")
    ax.set_title(f"Feature-Space PCA: {total_tokens} Tokens Across {len(TASKS)} Task Families\n"
                 f"({ndim:,}D computational-mode features, gpt-oss-20b)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    _save(fig, "fig8_feature_pca")
    plt.close(fig)


if __name__ == "__main__":
    main()
