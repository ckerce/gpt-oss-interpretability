#!/usr/bin/env python3
"""Generate memo figures from the direct-vocabulary steering artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ALIASES = {
    "dns-dns_S_G_gpt2": "SS-71",
    "dns-dns_S_G_cascade_gpt2": "C-71",
    "dns-dns_S_G_cascade_pls_xe_gpt2": "CP-71",
    "dns-dns_S_G_cascade_pls_xe_codelion_80M": "CP-80",
    "dns-dns_S_G_cascade_pls_xe_gpt2_base": "CP-124",
}

CASE_ORDER = ["caps_005", "induction_009", "coref_010"]
CASE_LABELS = {
    "caps_005": "Capitalization",
    "induction_009": "Induction",
    "coref_010": "Coreference",
}
MATCHED_PAIR = ["dns-dns_S_G_gpt2", "dns-dns_S_G_cascade_gpt2"]
PAIR_COLORS = {
    "dns-dns_S_G_gpt2": "#345995",
    "dns-dns_S_G_cascade_gpt2": "#c44536",
}
ALL_MODEL_COLORS = {
    "dns-dns_S_G_gpt2": "#345995",
    "dns-dns_S_G_cascade_gpt2": "#c44536",
    "dns-dns_S_G_cascade_pls_xe_gpt2": "#2d936c",
    "dns-dns_S_G_cascade_pls_xe_codelion_80M": "#a06cd5",
    "dns-dns_S_G_cascade_pls_xe_gpt2_base": "#f4a259",
}


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _load_payload(path: Path) -> dict:
    return json.loads(path.read_text())


def _rows_for(payload: dict, model: str, source: str, target: str) -> list[dict]:
    return [
        row
        for row in payload["rows"]
        if row["model"] == model
        and row["source_case_id"] == source
        and row["target_case_id"] == target
    ]


def _best_on_target_shift(rows: list[dict]) -> float:
    return max(abs(row["local_logit_shift"]) for row in rows)


def _mean_off_target_shift(rows: list[dict]) -> float:
    return float(np.mean([abs(row["local_logit_shift"]) for row in rows]))


def plot_baseline_local_gaps(payload: dict, output_dir: Path) -> None:
    models = [model["label"] for model in payload["models"]]
    x = np.arange(len(CASE_ORDER))
    width = 0.15

    fig, ax = plt.subplots(figsize=(10, 4.8))
    for idx, model in enumerate(models):
        vals = [
            payload["models"][models.index(model)]["baselines"][case]["local"]["local_logit_gap"]
            for case in CASE_ORDER
        ]
        ax.bar(
            x + (idx - (len(models) - 1) / 2) * width,
            vals,
            width=width,
            color=ALL_MODEL_COLORS[model],
            label=ALIASES[model],
        )

    ax.set_xticks(x)
    ax.set_xticklabels([CASE_LABELS[c] for c in CASE_ORDER])
    ax.set_ylabel("Baseline local logit gap")
    ax.set_title("Baseline Answer Margins Across Evaluated Models")
    ax.legend(frameon=False, ncol=3)
    ax.axhline(0.0, color="black", linewidth=0.8)
    _save_figure(fig, output_dir, "fig_baseline_local_gaps")


def plot_matched_pair_strength_scans(payload: dict, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), sharey=False)
    for ax, case in zip(axes, CASE_ORDER):
        for model in MATCHED_PAIR:
            rows = _rows_for(payload, model, case, case)
            best_layer = max(
                sorted({row["layer"] for row in rows}),
                key=lambda layer: max(abs(r["local_logit_shift"]) for r in rows if r["layer"] == layer),
            )
            subset = sorted(
                [row for row in rows if row["layer"] == best_layer],
                key=lambda row: row["scale"],
            )
            ax.plot(
                [row["scale"] for row in subset],
                [row["local_logit_shift"] for row in subset],
                marker="o",
                linewidth=2.0,
                color=PAIR_COLORS[model],
                label=f"{ALIASES[model]} (L{best_layer})",
            )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.axvline(0.0, color="black", linewidth=0.6, alpha=0.5)
        ax.set_title(CASE_LABELS[case])
        ax.set_xlabel("Scale α")
        ax.set_ylabel("Local logit-gap shift")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Perturbation-Strength Scans for the Matched 71M Pair", y=1.05)
    fig.tight_layout()
    _save_figure(fig, output_dir, "fig_matched_pair_strength_scans")


def plot_matched_pair_layer_profiles(payload: dict, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), sharey=False)
    for ax, case in zip(axes, CASE_ORDER):
        for model in MATCHED_PAIR:
            rows = _rows_for(payload, model, case, case)
            layers = sorted({row["layer"] for row in rows})
            vals = [
                max(abs(row["local_logit_shift"]) for row in rows if row["layer"] == layer)
                for layer in layers
            ]
            ax.plot(
                layers,
                vals,
                marker="o",
                linewidth=2.0,
                color=PAIR_COLORS[model],
                label=ALIASES[model],
            )
        ax.set_title(CASE_LABELS[case])
        ax.set_xlabel("Layer")
        ax.set_ylabel("Best |local shift|")
        ax.set_xticks(layers)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Layer Profiles for the Matched 71M Pair", y=1.05)
    fig.tight_layout()
    _save_figure(fig, output_dir, "fig_matched_pair_layer_profiles")


def plot_matched_pair_heatmaps(payload: dict, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.2), sharex=True, sharey=True)
    scales = sorted({row["scale"] for row in payload["rows"]})
    layers = sorted({row["layer"] for row in payload["rows"]})
    vmax = max(
        abs(row["local_logit_shift"])
        for row in payload["rows"]
        if row["model"] in MATCHED_PAIR and row["source_case_id"] == row["target_case_id"]
    )

    for row_idx, model in enumerate(MATCHED_PAIR):
        for col_idx, case in enumerate(CASE_ORDER):
            rows = _rows_for(payload, model, case, case)
            matrix = np.zeros((len(layers), len(scales)))
            for i, layer in enumerate(layers):
                for j, scale in enumerate(scales):
                    row = next(r for r in rows if r["layer"] == layer and r["scale"] == scale)
                    matrix[i, j] = row["local_logit_shift"]
            ax = axes[row_idx, col_idx]
            im = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax, origin="lower")
            ax.set_title(f"{ALIASES[model]} | {CASE_LABELS[case]}")
            ax.set_xticks(range(len(scales)))
            ax.set_xticklabels(scales)
            ax.set_yticks(range(len(layers)))
            ax.set_yticklabels(layers)
            if row_idx == 1:
                ax.set_xlabel("Scale α")
            if col_idx == 0:
                ax.set_ylabel("Layer")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label("Local logit-gap shift")
    fig.suptitle("Layer × Scale Heatmaps for the Matched 71M Pair", y=0.98)
    fig.tight_layout()
    _save_figure(fig, output_dir, "fig_matched_pair_heatmaps")


def plot_selectivity_scatter(payload: dict, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for model in [model["label"] for model in payload["models"]]:
        for case in CASE_ORDER:
            on_rows = _rows_for(payload, model, case, case)
            off_rows = [
                row
                for row in payload["rows"]
                if row["model"] == model
                and row["source_case_id"] == case
                and row["target_case_id"] != case
            ]
            on_val = _best_on_target_shift(on_rows)
            off_val = _mean_off_target_shift(off_rows)
            ax.scatter(
                on_val,
                off_val,
                s=80,
                color=ALL_MODEL_COLORS[model],
                alpha=0.9,
            )
            ax.text(
                on_val + 0.02,
                off_val + 0.005,
                f"{ALIASES[model]}:{case.split('_')[0]}",
                fontsize=8,
            )
    max_val = max(ax.get_xlim()[1], ax.get_ylim()[1], 1.0)
    ax.plot([0, max_val], [0, max_val], linestyle="--", color="black", linewidth=1.0, alpha=0.6)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Best on-target |local shift|")
    ax.set_ylabel("Mean off-target |local shift|")
    ax.set_title("Selectivity: On-Target Strength vs Off-Target Drift")
    _save_figure(fig, output_dir, "fig_selectivity_scatter")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate memo figures from direct-vocab results")
    parser.add_argument(
        "--input",
        default="runs/direct_vocab_large_models_cpu/direct_vocab_steering.json",
        help="Path to steering JSON artifact",
    )
    parser.add_argument(
        "--output-dir",
        default="doc/memo/figs",
        help="Directory for figure outputs",
    )
    args = parser.parse_args()

    payload = _load_payload(Path(args.input))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_baseline_local_gaps(payload, output_dir)
    plot_matched_pair_strength_scans(payload, output_dir)
    plot_matched_pair_layer_profiles(payload, output_dir)
    plot_matched_pair_heatmaps(payload, output_dir)
    plot_selectivity_scatter(payload, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
