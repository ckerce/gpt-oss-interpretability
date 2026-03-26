#!/usr/bin/env python3
"""Compare logit-lens vs tuned-lens convergence depth across task families.

For each prompt in the built-in corpus, measures at which layer the target
token first enters the top-1 prediction under:
  (a) raw logit lens
  (b) tuned lens (requires trained translators)

Produces:
  convergence_comparison.png — scatter/box plot of convergence depths by task
  convergence_rank_trajectory.png — rank trajectory for a selected induction prompt

Usage::

    python threads/solid/1-convergence-logit-lens/plot_convergence_comparison.py \\
        --model openai/gpt-oss-20b \\
        --tuned-lens runs/tuned_lens/translators.pt \\
        --output runs/tuned_lens/

Requires translators trained with train_tuned_lens.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


# Representative prompts per task family for convergence comparison
COMPARISON_PROMPTS: dict[str, list[tuple[str, str]]] = {
    "induction": [
        ("The sequence continues: alpha beta gamma alpha beta", "gamma"),
        ("In the pattern red blue green red blue", "green"),
        ("The series goes: 1 2 3 1 2", "3"),
        ("The letters repeat: A B C D A B C D A B C", "D"),
    ],
    "coreference": [
        ("The trophy didn't fit in the suitcase because it was too small. The", "suitcase"),
        ("Paul called Tom because he wanted to ask for advice. He", "Tom"),
    ],
    "capitalization": [
        ("The name of the first president of the United States is george", "Washington"),
        ("My favorite city is paris, which is known for", "the"),
    ],
    "factual": [
        ("The chemical symbol for water is", "H"),
        ("The largest planet in our solar system is", "Jupiter"),
    ],
}


def get_convergence_layer(
    backend,
    prompt: str,
    target_token: str,
    translators=None,
) -> int | None:
    """Return the first layer where target_token is the top-1 prediction.

    Returns None if the target never reaches top-1 within the model depth.
    """
    from gpt_oss_interp.readouts.logit_lens import run_logit_lens

    result = run_logit_lens(
        backend,
        prompt,
        top_k=1,
        positions=None,
        translators=translators,
    )

    # Find the last token position (the one whose next token we're predicting)
    all_positions = sorted({p.position for p in result.predictions})
    if not all_positions:
        return None
    last_pos = all_positions[-1]

    preds_at_pos = result.position_slice(last_pos)
    for pred in preds_at_pos:
        if pred.top_tokens and pred.top_tokens[0].strip() == target_token.strip():
            return pred.layer_idx
    return None


def get_rank_trajectory(
    backend,
    prompt: str,
    target_token: str,
    translators=None,
) -> list[int]:
    """Return target token rank at each layer for the last token position."""
    from gpt_oss_interp.readouts.logit_lens import run_logit_lens

    result = run_logit_lens(
        backend,
        prompt,
        top_k=10,
        positions=None,
        translators=translators,
    )

    all_positions = sorted({p.position for p in result.predictions})
    if not all_positions:
        return []
    last_pos = all_positions[-1]

    preds_at_pos = sorted(result.position_slice(last_pos), key=lambda p: p.layer_idx)
    return [pred.target_rank for pred in preds_at_pos]


def main() -> int:
    parser = argparse.ArgumentParser(description="Convergence comparison: logit vs tuned lens")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--tuned-lens", required=True,
                        help="Path to translators.pt from train_tuned_lens.py")
    parser.add_argument("--output", default="runs/tuned_lens/")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from gpt_oss_interp.backends.transformers_gpt_oss import GPTOSSTransformersBackend
    from gpt_oss_interp.readouts.tuned_lens import TunedLensTranslators

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(__file__).parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading backend: {args.model}")
    backend = GPTOSSTransformersBackend(model_name=args.model, local_files_only=True)

    print(f"Loading translators from {args.tuned_lens}")
    translators = TunedLensTranslators.load(args.tuned_lens)

    # ── Collect convergence layers ──────────────────────────────────────────
    results: dict[str, dict[str, list[int | None]]] = {
        family: {"raw": [], "tuned": []}
        for family in COMPARISON_PROMPTS
    }

    for family, prompts in COMPARISON_PROMPTS.items():
        print(f"\nFamily: {family}")
        for prompt, target in prompts:
            raw_conv = get_convergence_layer(backend, prompt, target, translators=None)
            tuned_conv = get_convergence_layer(backend, prompt, target, translators=translators)
            print(f"  [{target!r:12s}] raw={raw_conv}  tuned={tuned_conv}")
            results[family]["raw"].append(raw_conv)
            results[family]["tuned"].append(tuned_conv)

    # ── Figure 1: Convergence depth comparison ──────────────────────────────
    families = list(COMPARISON_PROMPTS.keys())
    n_fam = len(families)
    x = np.arange(n_fam)
    width = 0.35

    raw_means = []
    tuned_means = []
    for fam in families:
        raw_vals = [v for v in results[fam]["raw"] if v is not None]
        tuned_vals = [v for v in results[fam]["tuned"] if v is not None]
        raw_means.append(np.mean(raw_vals) if raw_vals else 24)
        tuned_means.append(np.mean(tuned_vals) if tuned_vals else 24)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_raw = ax.bar(x - width / 2, raw_means, width, label="Raw logit lens",
                      color="#e05c5c", alpha=0.85)
    bars_tuned = ax.bar(x + width / 2, tuned_means, width, label="Tuned lens",
                        color="#4a90d9", alpha=0.85)

    ax.axhline(21, color="#888888", linestyle="--", linewidth=1.2, alpha=0.8,
               label="L21: raw lens validity threshold")

    ax.set_xlabel("Task family", fontsize=12)
    ax.set_ylabel("Convergence layer (lower = earlier)", fontsize=12)
    ax.set_title("When does the model 'know' the answer?\nLogit lens vs tuned lens — gpt-oss-20b",
                 fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(families, fontsize=11)
    ax.set_ylim(0, 25)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    conv_path = out_dir / "convergence_comparison.png"
    fig.savefig(conv_path, dpi=150, bbox_inches="tight")
    fig.savefig(fig_dir / "convergence_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {conv_path}")

    # ── Figure 2: Rank trajectory for induction prompt ─────────────────────
    ind_prompt, ind_target = COMPARISON_PROMPTS["induction"][0]
    print(f"\nRank trajectory for: {ind_prompt!r} → {ind_target!r}")

    raw_ranks = get_rank_trajectory(backend, ind_prompt, ind_target, translators=None)
    tuned_ranks = get_rank_trajectory(backend, ind_prompt, ind_target, translators=translators)
    n_layers = max(len(raw_ranks), len(tuned_ranks))

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    if raw_ranks:
        ax2.semilogy(range(len(raw_ranks)), raw_ranks, "o-",
                     color="#e05c5c", linewidth=2, markersize=5, label="Raw logit lens")
    if tuned_ranks:
        ax2.semilogy(range(len(tuned_ranks)), tuned_ranks, "s-",
                     color="#4a90d9", linewidth=2, markersize=5, label="Tuned lens")

    ax2.axvline(21, color="#888888", linestyle="--", linewidth=1.2, alpha=0.8)
    ax2.text(21.3, max(raw_ranks or [1]) * 0.7, "L21\nraw valid",
             fontsize=8, color="#555555", va="top")
    ax2.axhline(1, color="#aaaaaa", linestyle=":", linewidth=1)
    ax2.text(0.5, 1.3, "rank 1 (top prediction)", fontsize=8, color="#888888")

    ax2.set_xlabel("Layer", fontsize=12)
    ax2.set_ylabel("Target token rank (log scale)", fontsize=12)
    ax2.set_title(
        f"Rank trajectory: {ind_target!r} in induction prompt\n"
        f"Prompt: \"{ind_prompt[:50]}…\"",
        fontsize=11
    )
    ax2.set_xticks(range(n_layers))
    ax2.set_xticklabels([f"L{l}" for l in range(n_layers)],
                        rotation=45, ha="right", fontsize=8)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    fig2.tight_layout()

    traj_path = out_dir / "convergence_rank_trajectory.png"
    fig2.savefig(traj_path, dpi=150, bbox_inches="tight")
    fig2.savefig(fig_dir / "convergence_rank_trajectory.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {traj_path}")

    # ── Save summary JSON ──────────────────────────────────────────────────
    summary = {
        family: {
            "prompts": [p for p, _ in COMPARISON_PROMPTS[family]],
            "targets": [t for _, t in COMPARISON_PROMPTS[family]],
            "raw_convergence": results[family]["raw"],
            "tuned_convergence": results[family]["tuned"],
            "mean_raw": float(np.mean([v for v in results[family]["raw"] if v is not None]) or 24),
            "mean_tuned": float(np.mean([v for v in results[family]["tuned"] if v is not None]) or 24),
        }
        for family in families
    }
    (out_dir / "convergence_comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved: {out_dir / 'convergence_comparison.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
