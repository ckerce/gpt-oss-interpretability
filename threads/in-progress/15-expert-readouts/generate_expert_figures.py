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

    # Use log(32) for gpt-oss-20b; infer from matrix width if routing_entropy
    # came from a different model.  Observed max entropy is always ≤ log(n_experts)
    # so we round up to the nearest power-of-two to avoid off-by-one from
    # load-balancing penalties.
    observed_max = max(entropies) if entropies else math.log(32)
    n_experts_guess = 1
    while math.log(n_experts_guess) < observed_max - 1e-6:
        n_experts_guess *= 2
    # Clamp: if the data isn't a power-of-two model, fall back to nearest int
    if abs(math.log(n_experts_guess) - observed_max) > 0.15:
        n_experts_guess = round(math.exp(observed_max))
    uniform_line = math.log(n_experts_guess)

    fig, ax = plt.subplots(figsize=(max(6, len(layers) * 0.4), 4))
    bars = ax.bar(layers, entropies, color="#4878D0", alpha=0.85, width=0.7)

    ax.axhline(uniform_line, color="red", linestyle="--", linewidth=1.2,
               label=f"Uniform max = log({n_experts_guess}) ≈ {uniform_line:.2f} nats")

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

def fig_expert_profiles(
    expert_vocab_profiles: dict,
    out_path: Path,
    top_k: int = 5,
    max_display_layers: int = 7,
    max_display_experts: int = 12,
) -> None:
    """Grid of per-expert top-token vocabulary profiles.

    expert_vocab_profiles schema:
        {layer_str: {expert_str: [(token, logp), ...]}}

    Subsamples layers and experts to keep the grid readable.  Layers are chosen
    to span the depth range (first, last, and the L19–21 causal bottleneck).
    Experts are sampled evenly across the expert index range.
    """
    matplotlib, plt, mcolors, np = _require_mpl()

    if not expert_vocab_profiles:
        print("[fig_expert_profiles] No profile data — skipping.", file=sys.stderr)
        return

    all_layers = sorted(int(k) for k in expert_vocab_profiles)
    all_experts_set: set[int] = set()
    for layer_data in expert_vocab_profiles.values():
        all_experts_set.update(int(e) for e in layer_data)
    all_experts = sorted(all_experts_set)

    if not all_experts:
        print("[fig_expert_profiles] Empty profile data — skipping.", file=sys.stderr)
        return

    # Subsample layers: always include first, last, and neighbourhood of L19-21
    n_layers_total = len(all_layers)
    if n_layers_total <= max_display_layers:
        display_layers = all_layers
    else:
        # Anchor points: first, L8 (mid-early), L17, L19, L20, L21, last
        anchors = {all_layers[0], all_layers[-1]}
        for target in [8, 17, 19, 20, 21]:
            closest = min(all_layers, key=lambda l: abs(l - target))
            anchors.add(closest)
        # Fill remaining slots with evenly spaced layers
        step = max(1, n_layers_total // max_display_layers)
        for i in range(0, n_layers_total, step):
            anchors.add(all_layers[i])
        display_layers = sorted(anchors)[:max_display_layers]

    # Subsample experts: evenly spaced
    if len(all_experts) <= max_display_experts:
        display_experts = all_experts
    else:
        step = len(all_experts) / max_display_experts
        display_experts = [all_experts[int(i * step)] for i in range(max_display_experts)]

    n_rows = len(display_layers)
    n_cols = len(display_experts)

    cell_h = 2.0 * top_k
    cell_w = 3.2
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(cell_w * n_cols, cell_h * n_rows),
        squeeze=False,
    )

    for row, li in enumerate(display_layers):
        layer_str = str(li)
        layer_data = expert_vocab_profiles.get(layer_str, {})
        for col, ei in enumerate(display_experts):
            ax = axes[row][col]
            profile = layer_data.get(str(ei), [])
            if profile:
                tokens = [str(t)[:10] for t, _ in profile[:top_k]]
                logps = [lp for _, lp in profile[:top_k]]
                tokens = tokens[::-1]
                logps = logps[::-1]
                ax.barh(range(len(tokens)), logps, color="#4878D0", alpha=0.8)
                ax.set_yticks(range(len(tokens)))
                ax.set_yticklabels(tokens, fontsize=9)
                ax.set_xticks([])
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
            else:
                ax.text(0.5, 0.5, "—", ha="center", va="center",
                        fontsize=12, transform=ax.transAxes, color="grey")
                ax.set_xticks([])
                ax.set_yticks([])

            if row == 0:
                ax.set_title(f"Expert {ei}", fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"Layer {li}", fontsize=10, fontweight="bold",
                              rotation=0, labelpad=40, va="center")

    n_experts_total = len(all_experts)
    fig.suptitle(
        f"Expert vocabulary profiles — top-{top_k} tokens\n"
        f"Showing {n_rows} of {n_layers_total} layers × {n_cols} of {n_experts_total} experts"
        f" (non-quantized checkpoint only)",
        fontsize=10,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
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


# ---------------------------------------------------------------------------
# Figure 5: Routing specialization gain (KL from uniform)
# ---------------------------------------------------------------------------

def fig_specialization_gain(kl_from_uniform: dict, out_path: Path) -> None:
    """Bar chart of D_KL(routing_l ‖ uniform) per layer.

    Interpretation: nats *saved* by knowing the routing policy vs. guessing
    uniform.  This is log(n_experts) - H(routing_l), but framed as information
    value rather than entropy deficit.
    """
    matplotlib, plt, mcolors, np = _require_mpl()

    if not kl_from_uniform:
        print("[fig_specialization_gain] No KL data — skipping.", file=sys.stderr)
        return

    layers = sorted(int(k) for k in kl_from_uniform)
    gains = [kl_from_uniform[str(li)] for li in layers]

    fig, ax = plt.subplots(figsize=(max(6, len(layers) * 0.4), 4))
    colors = ["#C44E52" if g == max(gains) else "#4878D0" for g in gains]
    ax.bar(layers, gains, color=colors, alpha=0.85, width=0.7)

    ax.set_title(
        "Routing specialization gain D_KL(routing ‖ uniform) per layer\n"
        "(nats saved by knowing the routing policy vs. guessing uniform)",
        fontsize=10,
    )
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Specialization gain (nats)")
    ax.set_xticks(layers)
    ax.set_xticklabels(layers, fontsize=7)
    ax.axhline(0, color="black", linewidth=0.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[fig_specialization_gain] saved → {out_path}")


# ---------------------------------------------------------------------------
# Figure 6: Routing velocity D_KL(routing_l ‖ routing_{l-1})
# ---------------------------------------------------------------------------

def fig_routing_velocity(routing_velocity: dict, out_path: Path) -> None:
    """Bar chart of routing velocity — where does routing policy change fastest?

    A spike marks a routing phase transition.  The hypothesis is that velocity
    peaks *before* the causal bottleneck (near L16-17), not within it.
    """
    matplotlib, plt, mcolors, np = _require_mpl()

    if not routing_velocity:
        print("[fig_routing_velocity] No velocity data — skipping.", file=sys.stderr)
        return

    layers = sorted(int(k) for k in routing_velocity)
    vels = [routing_velocity[str(li)] for li in layers]
    peak_layer = layers[vels.index(max(vels))] if vels else None

    fig, ax = plt.subplots(figsize=(max(6, len(layers) * 0.4), 4))
    colors = ["#C44E52" if li == peak_layer else "#4878D0" for li in layers]
    ax.bar(layers, vels, color=colors, alpha=0.85, width=0.7)

    if peak_layer is not None:
        ax.annotate(
            f"Peak at L{peak_layer}\n(routing phase transition)",
            xy=(peak_layer, max(vels)),
            xytext=(peak_layer + 1.5, max(vels) * 0.9),
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=8,
        )

    ax.set_title(
        "Routing velocity D_KL(routing_l ‖ routing_{l−1}) per layer\n"
        "(spike = routing policy changes fastest here; hypothesis: peaks before L19–21)",
        fontsize=10,
    )
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Velocity (nats)")
    ax.set_xticks(layers)
    ax.set_xticklabels(layers, fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[fig_routing_velocity] saved → {out_path}")


# ---------------------------------------------------------------------------
# Figure 7: Task routing mutual information curve
# ---------------------------------------------------------------------------

def fig_mi_task(mi_task: dict, out_path: Path) -> None:
    """Line chart of I(expert_l; task_family) per layer.

    Shows how many nats of task identity are encoded in the routing decision
    at each layer.  The peak layer is the routing-based task-resolution depth.
    """
    matplotlib, plt, mcolors, np = _require_mpl()

    if not mi_task:
        print("[fig_mi_task] No MI data — skipping.", file=sys.stderr)
        return

    layers = sorted(int(k) for k in mi_task)
    mi_vals = [mi_task[str(li)] for li in layers]
    peak_layer = layers[mi_vals.index(max(mi_vals))] if mi_vals else None

    fig, ax = plt.subplots(figsize=(max(6, len(layers) * 0.4), 4))
    ax.plot(layers, mi_vals, color="#4878D0", linewidth=2, marker="o", markersize=4)
    ax.fill_between(layers, mi_vals, alpha=0.2, color="#4878D0")

    if peak_layer is not None:
        ax.axvline(peak_layer, color="#C44E52", linestyle="--", linewidth=1.2,
                   label=f"Peak at L{peak_layer} ({mi_task[str(peak_layer)]:.5f} nats)")
        ax.legend(fontsize=8)

    ax.set_title(
        "Task routing mutual information I(expert; task_family) per layer\n"
        "(nats of task identity encoded in the routing decision; peak = task-routing alignment depth)",
        fontsize=10,
    )
    ax.set_xlabel("Layer index")
    ax.set_ylabel("I(expert; task) (nats)")
    ax.set_xticks(layers)
    ax.set_xticklabels(layers, fontsize=7)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[fig_mi_task] saved → {out_path}")


# ---------------------------------------------------------------------------
# Figure 8: Cross-task JSD distinguishability matrix at key layers
# ---------------------------------------------------------------------------

def fig_jsd_matrix(jsd_cross_task: dict, out_path: Path, probe_layers: list[int] | None = None) -> None:
    """Heatmap grid of cross-task JSD at selected layers.

    jsd_cross_task schema:
        {layer_str: {task_a: {task_b: jsd_float}}}

    Shows the 5×5 pairwise task-routing distinguishability matrix at a few
    representative depth layers.  JSD = 0 → tasks route identically.
    JSD = ln(2) ≈ 0.693 → tasks route to completely disjoint expert sets.
    """
    matplotlib, plt, mcolors, np = _require_mpl()

    if not jsd_cross_task:
        print("[fig_jsd_matrix] No JSD data — skipping.", file=sys.stderr)
        return

    all_layers = sorted(int(k) for k in jsd_cross_task)
    if not all_layers:
        return

    # Default probe layers: early, mid, late-onset, bottleneck, final
    if probe_layers is None:
        n = len(all_layers)
        probe_layers = sorted({
            all_layers[0],
            all_layers[n // 4],
            all_layers[n // 2],
            all_layers[3 * n // 4],
            all_layers[-1],
        })

    # Filter to available layers
    probe_layers = [li for li in probe_layers if str(li) in jsd_cross_task]
    if not probe_layers:
        return

    # Get task names from first available layer
    first_mat = jsd_cross_task[str(probe_layers[0])]
    task_names = sorted(first_mat.keys())
    n_tasks = len(task_names)
    ln2 = math.log(2)

    n_cols = len(probe_layers)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 3.5), squeeze=False)

    for col, li in enumerate(probe_layers):
        ax = axes[0][col]
        mat_dict = jsd_cross_task.get(str(li), {})
        mat = np.array([
            [mat_dict.get(ta, {}).get(tb, 0.0) for tb in task_names]
            for ta in task_names
        ])
        im = ax.imshow(mat, vmin=0, vmax=ln2, cmap="YlOrRd", aspect="equal")
        ax.set_title(f"L{li}", fontsize=10, fontweight="bold")
        ax.set_xticks(range(n_tasks))
        ax.set_yticks(range(n_tasks))
        short = [t[:4] for t in task_names]
        ax.set_xticklabels(short, fontsize=7, rotation=30)
        ax.set_yticklabels(short if col == 0 else [], fontsize=7)

        # Annotate cells
        for r in range(n_tasks):
            for c in range(n_tasks):
                v = mat[r, c]
                ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if v > ln2 * 0.6 else "black")

    fig.suptitle(
        "Cross-task routing distinguishability JSD(routing_A, routing_B) at key layers\n"
        f"(0 = identical routing; {ln2:.3f} nats = disjoint expert sets)",
        fontsize=10, y=1.02,
    )
    fig.colorbar(im, ax=axes[0][-1], fraction=0.046, label="JSD (nats)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig_jsd_matrix] saved → {out_path}")


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
    parser.add_argument(
        "--profile-layers",
        type=int,
        default=7,
        help="Max layers to show in expert profile grid (subsampled, anchored at L19-21).",
    )
    parser.add_argument(
        "--profile-experts",
        type=int,
        default=12,
        help="Max experts to show in expert profile grid (evenly subsampled).",
    )
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    routing_patterns = _load_json(run_dir / "routing_patterns.json")
    routing_entropy = _load_json(run_dir / "routing_entropy.json")
    layer_logit_delta = _load_json(run_dir / "layer_logit_delta.json")
    expert_vocab_profiles = _load_json(run_dir / "expert_vocab_profiles.json")
    kl_from_uniform = _load_json(run_dir / "routing_kl_from_uniform.json")
    routing_velocity = _load_json(run_dir / "routing_velocity.json")
    mi_task = _load_json(run_dir / "routing_mi_task.json")
    jsd_cross_task = _load_json(run_dir / "routing_jsd_matrix.json")

    if routing_patterns is not None:
        # Real run output wraps patterns under "by_task"; demo data is flat.
        rp_by_task = routing_patterns.get("by_task", routing_patterns)
        fig_routing_heatmap(rp_by_task, fig_dir / "fig_routing_heatmap.png")

    # Normalise layer_logit_delta if it came from a real run.
    # Real run schema: {layer_str: {"promoted": [{"token": t, "delta": d}, ...], "suppressed": [...]}}
    # Figure expects:  {layer_str: {pos_str: {"promoted": [(tok, delta), ...], ...}}}
    if layer_logit_delta is not None:
        first_layer_val = next(iter(layer_logit_delta.values()), {})
        if "promoted" in first_layer_val and not any(
            k.isdigit() for k in first_layer_val
        ):
            # Convert: wrap each layer's data in a single position "0"
            normalized: dict = {}
            for layer_str, pdata in layer_logit_delta.items():
                promoted = [
                    [e["token"], e["delta"]] if isinstance(e, dict) else e
                    for e in pdata.get("promoted", [])
                ]
                suppressed = [
                    [e["token"], e["delta"]] if isinstance(e, dict) else e
                    for e in pdata.get("suppressed", [])
                ]
                normalized[layer_str] = {"0": {"promoted": promoted, "suppressed": suppressed}}
            layer_logit_delta = normalized

    if routing_entropy is not None:
        fig_routing_entropy(routing_entropy, fig_dir / "fig_routing_entropy.png")

    if layer_logit_delta is not None:
        fig_logit_delta(layer_logit_delta, fig_dir / "fig_logit_delta.png", top_k=args.top_k)

    if expert_vocab_profiles is not None:
        fig_expert_profiles(
            expert_vocab_profiles,
            fig_dir / "fig_expert_profiles.png",
            top_k=args.top_k,
            max_display_layers=args.profile_layers,
            max_display_experts=args.profile_experts,
        )

    if kl_from_uniform is not None:
        fig_specialization_gain(kl_from_uniform, fig_dir / "fig_specialization_gain.png")

    if routing_velocity is not None:
        fig_routing_velocity(routing_velocity, fig_dir / "fig_routing_velocity.png")

    if mi_task is not None:
        fig_mi_task(mi_task, fig_dir / "fig_mi_task.png")

    if jsd_cross_task is not None:
        fig_jsd_matrix(jsd_cross_task, fig_dir / "fig_jsd_matrix.png")

    print(f"\n[generate_expert_figures] Done. Figures written to {fig_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
