"""Figure generation for Thread 15: MoE Expert Readouts.

Reads JSON outputs from run_expert_analysis.py and produces publication-quality
figures under runs/expert_readouts/figures/.

Usage::

    python generate_expert_figures.py --run-dir runs/expert_readouts

Figures produced:
    fig_routing_heatmap.png   -- Layer x expert utilization (heatmap)
    fig_routing_entropy.png   -- Routing entropy per layer (bar chart)
    fig_logit_delta.png       -- Top-token logit delta per layer (heatmap)
    fig_expert_profiles.png   -- Expert vocabulary profile matrix (non-quantized)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Lazy matplotlib import — keep module importable without display server
# ---------------------------------------------------------------------------

def _require_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import numpy as np
        return matplotlib, plt, mcolors, np
    except ImportError as e:
        print(f"[generate_expert_figures] matplotlib / numpy not available: {e}", file=sys.stderr)
        print("Install with:  pip install matplotlib numpy", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Figure 1: Routing heatmap (Layer × Expert utilization)
# ---------------------------------------------------------------------------

def fig_routing_heatmap(routing_patterns: dict, out_path: Path) -> None:
    """Heatmap of expert utilization across layers.

    routing_patterns schema:
        {task_family: {layer_str: {expert_str: token_count}}}

    We collapse across task families to get an overall utilization matrix.
    """
    matplotlib, plt, mcolors, np = _require_mpl()

    # Aggregate token counts: {layer_idx: {expert_idx: count}}
    aggregate: dict[int, dict[int, int]] = {}
    for _family, layer_data in routing_patterns.items():
        for layer_str, expert_data in layer_data.items():
            li = int(layer_str)
            if li not in aggregate:
                aggregate[li] = {}
            for expert_str, count in expert_data.items():
                ei = int(expert_str)
                aggregate[li][ei] = aggregate[li].get(ei, 0) + count

    if not aggregate:
        print("[fig_routing_heatmap] No routing data — skipping.", file=sys.stderr)
        return

    n_layers = max(aggregate) + 1
    n_experts = max(
        max(experts.keys()) for experts in aggregate.values()
    ) + 1

    # Build matrix [n_layers, n_experts]
    mat = np.zeros((n_layers, n_experts), dtype=float)
    for li, experts in aggregate.items():
        row_total = sum(experts.values()) or 1.0
        for ei, count in experts.items():
            mat[li, ei] = count / row_total  # fraction of tokens at this layer

    fig, ax = plt.subplots(figsize=(max(8, n_experts * 0.4), max(5, n_layers * 0.35)))
    im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0)

    # Uniform baseline for reference
    uniform = 1.0 / n_experts
    ax.set_title(
        f"Expert utilization (fraction of tokens routed)\n"
        f"Uniform baseline = {uniform:.3f}",
        fontsize=10,
    )
    ax.set_xlabel("Expert index")
    ax.set_ylabel("Layer index")
    ax.set_xticks(range(n_experts))
    ax.set_yticks(range(n_layers))
    ax.set_xticklabels(range(n_experts), fontsize=7)
    ax.set_yticklabels(range(n_layers), fontsize=7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Fraction of tokens", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[fig_routing_heatmap] saved → {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: Routing entropy per layer
# ---------------------------------------------------------------------------

def fig_routing_entropy(routing_entropy: dict, out_path: Path) -> None:
    """Bar chart of routing entropy per layer.

    routing_entropy schema:
        {layer_str: entropy_float}

    Draws a horizontal dashed line at log(n_experts) (uniform upper bound).
    """
    matplotlib, plt, mcolors, np = _require_mpl()

    if not routing_entropy:
        print("[fig_routing_entropy] No entropy data — skipping.", file=sys.stderr)
        return

    layers = sorted(int(k) for k in routing_entropy)
    entropies = [routing_entropy[str(li)] for li in layers]

    # Infer n_experts from max possible entropy (if recorded) or default 32
    uniform_entropy = max(entropies) if entropies else math.log(32)
    # Use log(32) ≈ 3.47 for gpt-oss-20b default
    n_experts_guess = round(math.exp(uniform_entropy)) if uniform_entropy > 0 else 32
    uniform_line = math.log(n_experts_guess)

    fig, ax = plt.subplots(figsize=(max(6, len(layers) * 0.4), 4))
    bars = ax.bar(layers, entropies, color="#4878D0", alpha=0.85, width=0.7)

    ax.axhline(uniform_line, color="red", linestyle="--", linewidth=1.2,
               label=f"Uniform max ≈ {uniform_line:.2f} nats (log {n_experts_guess})")

    ax.set_title("Routing entropy per MoE layer\n(lower = more concentrated / specialised)", fontsize=10)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Entropy (nats)")
    ax.set_xticks(layers)
    ax.set_xticklabels(layers, fontsize=7)
    ax.legend(fontsize=8)
    ax.set_ylim(0, uniform_line * 1.15)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[fig_routing_entropy] saved → {out_path}")


# ---------------------------------------------------------------------------
# Figure 3: Top-token logit delta per layer
# ---------------------------------------------------------------------------

def fig_logit_delta(layer_logit_delta: dict, out_path: Path, top_k: int = 8) -> None:
    """Heatmap showing top-promoted tokens per layer.

    layer_logit_delta schema:
        {layer_str: {pos_str: {"promoted": [(token, delta), ...],
                                "suppressed": [(token, delta), ...]}}}

    We pick the token with the largest delta magnitude at each layer × position
    cell and display a colour-coded matrix.  Text labels are the token strings.
    """
    matplotlib, plt, mcolors, np = _require_mpl()

    if not layer_logit_delta:
        print("[fig_logit_delta] No logit-delta data — skipping.", file=sys.stderr)
        return

    layers = sorted(int(k) for k in layer_logit_delta)
    positions = sorted(
        int(p)
        for data in layer_logit_delta.values()
        for p in data
    )
    positions = sorted(set(positions))

    n_layers = len(layers)
    n_pos = len(positions)

    # Matrix of max delta values and corresponding token labels
    delta_mat = np.zeros((n_layers, n_pos), dtype=float)
    label_mat = [["" for _ in positions] for _ in layers]

    for row, li in enumerate(layers):
        pos_data = layer_logit_delta[str(li)]
        for col, pi in enumerate(positions):
            entry = pos_data.get(str(pi), {})
            promoted = entry.get("promoted", [])
            if promoted:
                top_tok, top_delta = promoted[0]
                delta_mat[row, col] = top_delta
                label_mat[row][col] = str(top_tok)[:6]  # truncate long tokens

    # Diverging colour map centred at 0
    vmax = max(abs(delta_mat).max(), 0.01)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(max(8, n_pos * 0.55), max(5, n_layers * 0.4)))
    im = ax.imshow(delta_mat, aspect="auto", cmap="RdBu_r", norm=norm)

    # Overlay token text
    for row in range(n_layers):
        for col in range(n_pos):
            label = label_mat[row][col]
            if label:
                ax.text(col, row, label, ha="center", va="center",
                        fontsize=6, color="black")

    ax.set_title(
        "Top-promoted token logit-delta per layer × position\n"
        "(blue = promoted, red = suppressed)",
        fontsize=10,
    )
    ax.set_xlabel("Token position")
    ax.set_ylabel("Layer index")
    ax.set_xticks(range(n_pos))
    ax.set_yticks(range(n_layers))
    ax.set_xticklabels(positions, fontsize=7)
    ax.set_yticklabels(layers, fontsize=7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Δ log p", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[fig_logit_delta] saved → {out_path}")


# ---------------------------------------------------------------------------
# Figure 4: Expert vocabulary profile matrix (non-quantized only)
# ---------------------------------------------------------------------------

def fig_expert_profiles(expert_vocab_profiles: dict, out_path: Path, top_k: int = 5) -> None:
    """Grid of per-expert top-token vocabulary profiles.

    expert_vocab_profiles schema:
        {layer_str: {expert_str: [(token, logp), ...]}}

    Lays out one panel per (layer, expert) showing the top-k tokens and their
    log-probabilities as a horizontal bar chart.
    """
    matplotlib, plt, mcolors, np = _require_mpl()

    if not expert_vocab_profiles:
        print("[fig_expert_profiles] No profile data — skipping.", file=sys.stderr)
        return

    layers = sorted(int(k) for k in expert_vocab_profiles)
    n_layers = len(layers)

    # Collect expert indices across all layers
    all_experts: set[int] = set()
    for layer_data in expert_vocab_profiles.values():
        all_experts.update(int(e) for e in layer_data)
    experts = sorted(all_experts)
    n_experts = len(experts)

    if n_experts == 0:
        print("[fig_expert_profiles] Empty profile data — skipping.", file=sys.stderr)
        return

    cell_h = 1.2 * top_k
    cell_w = 2.0
    fig, axes = plt.subplots(
        n_layers, n_experts,
        figsize=(cell_w * n_experts, cell_h * n_layers),
        squeeze=False,
    )

    for row, li in enumerate(layers):
        layer_str = str(li)
        layer_data = expert_vocab_profiles.get(layer_str, {})
        for col, ei in enumerate(experts):
            ax = axes[row][col]
            profile = layer_data.get(str(ei), [])
            if profile:
                tokens = [str(t)[:10] for t, _ in profile[:top_k]]
                logps = [lp for _, lp in profile[:top_k]]
                # Reverse so highest logp at top
                tokens = tokens[::-1]
                logps = logps[::-1]
                ax.barh(range(len(tokens)), logps, color="#4878D0", alpha=0.8)
                ax.set_yticks(range(len(tokens)))
                ax.set_yticklabels(tokens, fontsize=6)
                ax.set_xticks([])
            else:
                ax.text(0.5, 0.5, "—", ha="center", va="center",
                        fontsize=10, transform=ax.transAxes, color="grey")
                ax.set_xticks([])
                ax.set_yticks([])

            if row == 0:
                ax.set_title(f"E{ei}", fontsize=8)
            if col == 0:
                ax.set_ylabel(f"L{li}", fontsize=8, rotation=0, labelpad=20)

    fig.suptitle(
        f"Expert vocabulary profiles — top-{top_k} tokens per expert\n"
        "(non-quantized checkpoint only)",
        fontsize=10,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig_expert_profiles] saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict | None:
    if not path.exists():
        print(f"[generate_expert_figures] {path.name} not found — skipping figure.", file=sys.stderr)
        return None
    with open(path) as f:
        return json.load(f)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate Thread 15 figures from run_expert_analysis.py outputs."
    )
    parser.add_argument(
        "--run-dir",
        default="runs/expert_readouts",
        help="Directory containing routing_patterns.json etc.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top tokens to display in profile figure.",
    )
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    routing_patterns = _load_json(run_dir / "routing_patterns.json")
    routing_entropy = _load_json(run_dir / "routing_entropy.json")
    layer_logit_delta = _load_json(run_dir / "layer_logit_delta.json")
    expert_vocab_profiles = _load_json(run_dir / "expert_vocab_profiles.json")

    if routing_patterns is not None:
        fig_routing_heatmap(routing_patterns, fig_dir / "fig_routing_heatmap.png")

    if routing_entropy is not None:
        fig_routing_entropy(routing_entropy, fig_dir / "fig_routing_entropy.png")

    if layer_logit_delta is not None:
        fig_logit_delta(layer_logit_delta, fig_dir / "fig_logit_delta.png", top_k=args.top_k)

    if expert_vocab_profiles is not None:
        fig_expert_profiles(expert_vocab_profiles, fig_dir / "fig_expert_profiles.png", top_k=args.top_k)

    print(f"\n[generate_expert_figures] Done. Figures written to {fig_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
