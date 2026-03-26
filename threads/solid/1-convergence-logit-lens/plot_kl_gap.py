#!/usr/bin/env python3
"""Plot raw vs tuned KL gap per layer for gpt-oss-20b.

Reads the JSON produced by train_tuned_lens.py --gap-output and produces:
  1. kl_gap_curve.png  — raw vs tuned KL per layer (line plot)
  2. kl_reduction.png  — percentage KL reduction per layer (bar chart)

Usage::

    python threads/solid/1-convergence-logit-lens/plot_kl_gap.py \\
        --gap runs/tuned_lens/kl_gap.json \\
        --output runs/tuned_lens/

The figure is also written to threads/solid/1-convergence-logit-lens/figures/
for the thread README.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot KL gap curve")
    parser.add_argument("--gap", default="runs/tuned_lens/kl_gap.json",
                        help="JSON file from train_tuned_lens.py --gap-output")
    parser.add_argument("--output", default="runs/tuned_lens/",
                        help="Output directory for figures")
    parser.add_argument("--convergence-layer", type=int, default=21,
                        help="Mark raw logit-lens validity threshold (default 21)")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    gap_path = Path(args.gap)
    if not gap_path.exists():
        print(f"ERROR: {gap_path} not found. Run train_tuned_lens.py --gap-output first.")
        return 1

    data = json.loads(gap_path.read_text())
    raw = data["raw"]
    tuned = data.get("tuned", [])
    layers = list(range(len(raw)))
    n_layers = len(layers)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Also write to thread figures dir
    fig_dir = Path(__file__).parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Figure 1: KL gap curve ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(layers, raw, "o-", color="#e05c5c", linewidth=2, markersize=5, label="Raw logit lens")
    if tuned:
        ax.plot(layers, tuned, "s-", color="#4a90d9", linewidth=2, markersize=5, label="Tuned lens")

    # Mark the raw validity threshold
    L = args.convergence_layer
    ax.axvline(L, color="#888888", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.text(L + 0.3, max(raw) * 0.92,
            f"L{L}: raw\nlogit lens\nvalid",
            fontsize=8, color="#555555", va="top")

    # Shade the "opaque zone"
    ax.axvspan(0, L, alpha=0.06, color="#e05c5c", label=f"Opaque zone (L0–L{L-1})")

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Mean KL divergence (nats)", fontsize=12)
    ax.set_title("Logit-lens readout quality per layer — gpt-oss-20b", fontsize=13)
    ax.set_xticks(layers)
    ax.set_xticklabels([f"L{l}" for l in layers], rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    kl_curve_path = out_dir / "kl_gap_curve.png"
    fig.savefig(kl_curve_path, dpi=150, bbox_inches="tight")
    fig.savefig(fig_dir / "kl_gap_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {kl_curve_path}")

    # ── Figure 2: KL reduction bar chart ───────────────────────────────────
    if tuned and len(tuned) == len(raw):
        reductions = [
            (r - t) / (r + 1e-12) * 100
            for r, t in zip(raw, tuned)
        ]

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        colors = ["#4a90d9" if red >= 0 else "#e05c5c" for red in reductions]
        ax2.bar(layers, reductions, color=colors, alpha=0.85)
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.axvline(L, color="#888888", linestyle="--", linewidth=1.2, alpha=0.8)
        ax2.text(L + 0.3, min(reductions) * 0.85,
                 f"L{L}", fontsize=8, color="#555555")

        ax2.set_xlabel("Layer", fontsize=12)
        ax2.set_ylabel("KL reduction (%)", fontsize=12)
        ax2.set_title("Tuned-lens KL reduction per layer — gpt-oss-20b", fontsize=13)
        ax2.set_xticks(layers)
        ax2.set_xticklabels([f"L{l}" for l in layers], rotation=45, ha="right", fontsize=8)
        ax2.grid(axis="y", alpha=0.3)

        fig2.tight_layout()
        red_path = out_dir / "kl_reduction.png"
        fig2.savefig(red_path, dpi=150, bbox_inches="tight")
        fig2.savefig(fig_dir / "kl_reduction.png", dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved: {red_path}")

        # Summary stats
        mean_red = sum(reductions[:L]) / L if L else 0
        print(f"\nMean KL reduction L0–L{L-1} (opaque zone): {mean_red:.1f}%")
        print(f"Layers with >80% reduction: "
              f"{[l for l, r in enumerate(reductions) if r > 80]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
