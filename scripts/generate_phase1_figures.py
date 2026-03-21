#!/usr/bin/env python3
"""Generate Phase 1 figures for the gpt-oss-interp portfolio.

Produces:
  figures/fig1_convergence_trajectories.{pdf,png}
  figures/fig2_late_layer_ablation.{pdf,png}
  figures/fig3_analysis_set_stratification.{pdf,png}
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

RUNS = Path("runs")
FIGS = Path("figures")
FIGS.mkdir(exist_ok=True)

# ---------- shared style ----------

COLORS = {
    "induction": "#2176AE",
    "coreference": "#57B894",
    "capitalization": "#FBB13C",
    "recency_bias": "#D64045",
    "syntax_agreement": "#7B2D8E",
}

FAMILY_LABELS = {
    "induction": "Induction",
    "coreference": "Coreference",
    "capitalization": "Capitalization",
    "recency_bias": "Recency bias",
    "syntax_agreement": "Syntax agreement",
}

STRATUM_COLORS = {
    "correct_late_stable": "#2176AE",
    "correct_late_unstable": "#FBB13C",
    "incorrect_early_expected": "#D64045",
    "incorrect_never_expected": "#888888",
}

STRATUM_LABELS = {
    "correct_late_stable": "Correct, late-stable",
    "correct_late_unstable": "Correct, late-unstable",
    "incorrect_early_expected": "Incorrect (early expected)",
    "incorrect_never_expected": "Incorrect (never expected)",
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


# =====================================================================
# Figure 1: Convergence trajectories
# =====================================================================

def fig1_convergence_trajectories():
    """Choice-margin vs layer for representative cases from each family."""
    print("Figure 1: convergence trajectories")

    conv = json.loads((RUNS / "convergence_calibration" / "convergence_calibration.json").read_text())

    # Pick one representative case per family (prefer correct_late_stable)
    representatives = {
        "induction": "induction_002",      # "red blue green..." — clean, intuitive
        "coreference": "coref_003",         # "John called Mary..." — clean, simple
        "capitalization": "caps_003",       # "new York" — stable
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [3, 2]})

    # --- Panel A: representative trajectories ---
    ax = axes[0]

    for case in conv["cases"]:
        cid = case["case_id"]
        family = case["task_name"]
        if cid not in representatives.values():
            continue

        layers = [d["layer"] for d in case["layer_details"]]
        # Signed margin: positive = expected choice winning
        margins = []
        for d in case["layer_details"]:
            expected_lp = d["expected_logprob"]
            other_lps = [v for k, v in d["choice_logprobs"].items()
                         if k != case["expected_label"]]
            best_other = max(other_lps)
            margins.append(expected_lp - best_other)

        color = COLORS[family]
        label = f"{FAMILY_LABELS[family]}"
        ax.plot(layers, margins, "-o", color=color, markersize=4,
                linewidth=1.8, label=label, zorder=3)

        # Mark convergence layer
        conv_layer = case.get("expected_choice_convergence")
        if conv_layer is not None and conv_layer < len(margins):
            ax.axvline(conv_layer, color=color, linestyle="--", alpha=0.4, linewidth=1)

    ax.axhline(0, color="k", linewidth=0.6, alpha=0.4, zorder=1)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Margin (expected − runner-up log-prob)")
    ax.set_title("(a) Choice-relative convergence trajectories", fontsize=11, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.set_xlim(-0.5, 23.5)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))

    # --- Panel B: convergence layer distribution (all 9 soft-main cases) ---
    ax2 = axes[1]
    strat = json.loads((RUNS / "analysis_set_stratification" / "analysis_set_stratification.json").read_text())
    soft_main_ids = {c["case_id"] for c in strat["cases"] if c["soft_stratum"] == "correct_late_stable"}

    family_conv_layers = {}
    for case in conv["cases"]:
        if case["case_id"] in soft_main_ids:
            fam = case["task_name"]
            cl = case.get("expected_choice_convergence", case.get("final_choice_convergence"))
            if cl is not None:
                family_conv_layers.setdefault(fam, []).append(cl)

    families_present = sorted(family_conv_layers.keys())
    positions = list(range(len(families_present)))
    for i, fam in enumerate(families_present):
        vals = family_conv_layers[fam]
        color = COLORS[fam]
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
        ax2.scatter([i + j for j in jitter], vals, color=color, s=50,
                    edgecolors="white", linewidths=0.5, zorder=3)
        ax2.barh(y=np.mean(vals), width=0.6, left=i - 0.3, height=0,
                 color=color, alpha=0)  # invisible, just for alignment
        # mean marker
        ax2.plot([i - 0.2, i + 0.2], [np.mean(vals), np.mean(vals)],
                 color=color, linewidth=2.5, zorder=4)

    ax2.set_xticks(positions)
    ax2.set_xticklabels([FAMILY_LABELS[f] for f in families_present],
                        rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("Convergence layer")
    ax2.set_title("(b) Convergence layer distribution\n(soft main-analysis set)", fontsize=11, fontweight="bold")
    ax2.set_ylim(-1, 24)
    ax2.yaxis.set_major_locator(mticker.MultipleLocator(4))

    fig.tight_layout()
    _save(fig, "fig1_convergence_trajectories")
    plt.close(fig)


# =====================================================================
# Figure 2: Late-layer ablation
# =====================================================================

def fig2_late_layer_ablation():
    """Dual panel: margin + accuracy under layer ablation (L18-L23) on soft main set."""
    print("Figure 2: late-layer ablation")

    sweep = json.loads((RUNS / "soft_main_late_layer_sweep" / "summary.json").read_text())
    baseline_margin = 6.260  # from filtered_benchmark_analysis

    layers = list(range(18, 24))
    layer_labels = [f"L{l}" for l in layers]
    margins = []
    accuracies = []

    for l in layers:
        key = f"late_layer_L{l}@0"
        entry = sweep["summary"][key]
        margins.append(entry["mean_margin"])
        accuracies.append(entry["accuracy"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    # --- Panel A: Mean margin ---
    bar_colors = []
    for acc in accuracies:
        if acc < 0.5:
            bar_colors.append("#D64045")
        elif acc < 0.8:
            bar_colors.append("#FBB13C")
        else:
            bar_colors.append("#2176AE")

    bars = ax1.bar(layer_labels, margins, color=bar_colors,
                   edgecolor="white", linewidth=0.8, zorder=3)
    ax1.axhline(baseline_margin, color="k", linewidth=1.2, linestyle="--",
                alpha=0.6, zorder=2, label=f"Baseline ({baseline_margin:.2f})")
    ax1.set_ylabel("Mean margin (expected − runner-up)")
    ax1.set_xlabel("Ablated layer")
    ax1.set_title("(a) Margin under layer ablation", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9, loc="upper left")
    ax1.set_ylim(0, 7.5)

    # --- Panel B: Accuracy ---
    bars2 = ax2.bar(layer_labels, [a * 100 for a in accuracies], color=bar_colors,
                    edgecolor="white", linewidth=0.8, zorder=3)
    ax2.axhline(100, color="k", linewidth=1.2, linestyle="--",
                alpha=0.6, zorder=2, label="Baseline (100%)")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_xlabel("Ablated layer")
    ax2.set_title("(b) Accuracy under layer ablation", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9, loc="upper left")
    ax2.set_ylim(0, 115)

    # Add percentage labels
    for bar, acc in zip(bars2, accuracies):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 f"{acc:.0%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.suptitle("Late-layer ablation on soft main-analysis set (n = 9 cases, gpt-oss-20b)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig2_late_layer_ablation")
    plt.close(fig)


# =====================================================================
# Figure 3: Analysis-set stratification
# =====================================================================

def fig3_stratification():
    """Stacked bar chart: 4-way stratification by task family."""
    print("Figure 3: analysis-set stratification")

    strat = json.loads((RUNS / "analysis_set_stratification" / "analysis_set_stratification.json").read_text())

    families = ["induction", "coreference", "capitalization", "syntax_agreement", "recency_bias"]
    strata_order = ["correct_late_stable", "correct_late_unstable",
                    "incorrect_early_expected", "incorrect_never_expected"]

    by_task = strat["by_task"]["soft"]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    x = np.arange(len(families))
    width = 0.6
    bottoms = np.zeros(len(families))

    for stratum in strata_order:
        counts = []
        for fam in families:
            counts.append(by_task.get(fam, {}).get(stratum, 0))
        counts = np.array(counts, dtype=float)
        if counts.sum() == 0:
            continue
        bars = ax.bar(x, counts, width, bottom=bottoms,
                      label=STRATUM_LABELS[stratum],
                      color=STRATUM_COLORS[stratum],
                      edgecolor="white", linewidth=0.8, zorder=3)
        # Add count labels on bars
        for i, (c, b) in enumerate(zip(counts, bottoms)):
            if c > 0:
                ax.text(x[i], b + c / 2, str(int(c)),
                        ha="center", va="center", fontsize=10,
                        fontweight="bold", color="white")
        bottoms += counts

    ax.set_xticks(x)
    ax.set_xticklabels([FAMILY_LABELS[f] for f in families], fontsize=10)
    ax.set_ylabel("Number of cases")
    ax.set_title("Analysis-set stratification by task family\n"
                 "(soft late-commitment rule, tail = 4 layers)",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_ylim(0, 5.5)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1))

    # Annotate main analysis set
    ax.annotate("9 cases in main\nanalysis set",
                xy=(1, 4.5), fontsize=9, color="#2176AE",
                fontweight="bold", ha="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#2176AE",
                          alpha=0.1, edgecolor="#2176AE"))

    fig.tight_layout()
    _save(fig, "fig3_analysis_set_stratification")
    plt.close(fig)


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    _style()
    fig1_convergence_trajectories()
    fig2_late_layer_ablation()
    fig3_stratification()
    print("\nAll Phase 1 figures generated.")
