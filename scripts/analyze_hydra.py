#!/usr/bin/env python3
"""Analyze per-head ablation results for the Hydra effect.

Reads head_ablation_L20 results and computes:
1. Ablation-effect variance across 64 heads (the Hydra metric)
2. Per-task ablation profiles
3. Comparison to NeurIPS paper's PLS vs control values

Output: figures/fig5_hydra_variance.{pdf,png} and runs/head_ablation_L20/hydra_analysis.json
"""

import json
import csv
import math
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs" / "head_ablation_L20"
FIGS = REPO / "figures"
FIGS.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
})


def load_results():
    """Load case-level results from the head ablation sweep."""
    csv_path = RUNS / "case_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No results at {csv_path}. Run the head ablation sweep first.")

    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["margin"] = float(row["margin"])
            row["correct"] = int(row["correct"])
            rows.append(row)
    return rows


def compute_hydra_metrics(rows):
    """Compute per-head ablation effects and variance."""
    # Group by head
    by_head = defaultdict(list)
    by_head_task = defaultdict(lambda: defaultdict(list))

    for row in rows:
        run = row["run_name"]
        if not run.startswith("head_L20_H"):
            continue
        head_idx = int(run.split("_H")[1].split("@")[0])
        by_head[head_idx].append(row["margin"])
        by_head_task[head_idx][row["task_name"]].append(row["margin"])

    # Compute mean margin per head
    head_means = {}
    for h in sorted(by_head.keys()):
        head_means[h] = np.mean(by_head[h])

    # Overall statistics
    all_means = np.array(list(head_means.values()))
    overall_mean = np.mean(all_means)
    overall_std = np.std(all_means)
    overall_var = np.var(all_means)

    # Per-task variance
    task_names = set()
    for h in by_head_task:
        task_names.update(by_head_task[h].keys())

    task_profiles = {}
    for task in sorted(task_names):
        task_head_means = []
        for h in sorted(by_head.keys()):
            if task in by_head_task[h]:
                task_head_means.append(np.mean(by_head_task[h][task]))
            else:
                task_head_means.append(np.nan)
        task_head_means = np.array(task_head_means)
        valid = task_head_means[~np.isnan(task_head_means)]
        task_profiles[task] = {
            "head_means": task_head_means.tolist(),
            "mean": float(np.mean(valid)),
            "std": float(np.std(valid)),
            "min": float(np.min(valid)),
            "max": float(np.max(valid)),
        }

    return {
        "num_heads": len(head_means),
        "head_means": {int(k): float(v) for k, v in head_means.items()},
        "overall_mean": float(overall_mean),
        "overall_std": float(overall_std),
        "overall_var": float(overall_var),
        "task_profiles": task_profiles,
        "neurips_comparison": {
            "gpt_oss_20b_std": float(overall_std),
            "neurips_control_std": 0.08,
            "neurips_pls_std": 0.47,
            "interpretation": (
                "Low σ (near 0.08) = Hydra active (distributed redundancy). "
                "High σ (near 0.47) = modular computation exposed."
            ),
        },
    }


def plot_hydra(metrics):
    """Generate the Hydra variance figure."""
    head_means = metrics["head_means"]
    heads = sorted(head_means.keys())
    means = [head_means[h] for h in heads]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [3, 1]})

    # --- Panel A: Per-head ablation effects ---
    ax = axes[0]
    colors = ["#2176AE" if m > 0 else "#D64045" for m in means]
    ax.bar(range(len(heads)), means, color=colors, edgecolor="white",
           linewidth=0.3, width=0.8, zorder=3)
    ax.axhline(metrics["overall_mean"], color="black", linestyle="--",
               linewidth=1, alpha=0.6, label=f"Mean = {metrics['overall_mean']:.2f}")
    ax.axhline(0, color="k", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Head index (0-63)")
    ax.set_ylabel("Mean margin under ablation")
    ax.set_title("(a) Per-head ablation effect at L20\n(soft main-analysis set, n=9 cases)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(8))

    # --- Panel B: Variance comparison ---
    ax2 = axes[1]
    labels = ["gpt-oss-20b\n(standard)", "NeurIPS\ncontrol", "NeurIPS\nPLS"]
    values = [
        metrics["neurips_comparison"]["gpt_oss_20b_std"],
        metrics["neurips_comparison"]["neurips_control_std"],
        metrics["neurips_comparison"]["neurips_pls_std"],
    ]
    bar_colors = ["#2176AE", "#888888", "#57B894"]
    bars = ax2.bar(labels, values, color=bar_colors, edgecolor="white",
                   linewidth=0.8, width=0.6, zorder=3)
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                 f"σ={val:.3f}", ha="center", va="bottom", fontsize=9,
                 fontweight="bold")
    ax2.set_ylabel("Ablation-effect σ")
    ax2.set_title("(b) Hydra comparison\n(σ across heads)",
                   fontsize=11, fontweight="bold")
    ax2.set_ylim(0, max(values) * 1.3)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig5_hydra_variance.{ext}", bbox_inches="tight", dpi=200)
    print(f"  Saved fig5_hydra_variance.pdf / .png")
    plt.close(fig)


def main():
    print("Analyzing Hydra effect from head ablation sweep...")
    rows = load_results()
    metrics = compute_hydra_metrics(rows)

    # Save analysis
    out_path = RUNS / "hydra_analysis.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Analysis saved to {out_path}")
    print(f"  Heads: {metrics['num_heads']}")
    print(f"  Overall σ: {metrics['overall_std']:.4f}")
    print(f"  NeurIPS control σ: {metrics['neurips_comparison']['neurips_control_std']}")
    print(f"  NeurIPS PLS σ: {metrics['neurips_comparison']['neurips_pls_std']}")

    # Generate figure
    plot_hydra(metrics)
    print("\nDone.")


if __name__ == "__main__":
    main()
