#!/usr/bin/env python3
"""Linearity probe: sequential rank-64+64 and per-layer MLP translators.

Two experiments to distinguish whether the tuned-lens KL floor (~4.9 nats)
is due to insufficient rank in the affine correction, or genuine nonlinearity
in the early-layer → output-space geometry gap.

Experiment 1 — Sequential affine (rank-64 + rank-64)
-----------------------------------------------------
Freeze the trained T1 (rank-64). Train T2 (rank-64) on the residual KL
after T1. The composition T2(T1(h)) is an affine map with effective rank
up to 128. If the floor drops significantly, it is rank-limited.

Experiment 2 — Per-layer MLP (bottleneck=256, GELU)
----------------------------------------------------
Train a nonlinear 2-layer MLP translator for L00 and L12 only.
  - L00: highest raw KL (17.1 nats), 66% reducible by affine — does
    nonlinearity close the remaining 34%?
  - L12: MI-peak layer (Thread 15), only 21% reducible by affine — the
    most diagnostically interesting case.
If the MLP significantly outperforms the affine at L12, there is genuine
nonlinear structure in the representation at that depth.

Usage::

    python threads/solid/1-convergence-logit-lens/train_residual_and_mlp.py \\
        --model openai/gpt-oss-20b \\
        --base-translators runs/tuned_lens/translators.pt \\
        --corpus runs/tuned_lens/corpus_fineweb2k.txt \\
        --output-dir runs/tuned_lens/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Linearity probe: residual + MLP")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument(
        "--base-translators", required=True,
        help="Path to trained T1 translators (runs/tuned_lens/translators.pt)",
    )
    parser.add_argument(
        "--corpus", default="runs/tuned_lens/corpus_fineweb2k.txt",
        help="Training corpus (one prompt per line)",
    )
    parser.add_argument("--rank", type=int, default=64,
                        help="Rank for residual T2 (default 64, matching T1)")
    parser.add_argument("--mlp-bottleneck", type=int, default=256,
                        help="MLP hidden width (default 256, memory-safe on 24GB)")
    parser.add_argument("--mlp-layers", default="0,12",
                        help="Comma-separated layer indices for MLP probe (default 0,12)")
    parser.add_argument("--n-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--max-prompts", type=int, default=None,
                        help="Truncate corpus to this many prompts (useful for smoke tests)")
    parser.add_argument("--output-dir", default="runs/tuned_lens/")
    args = parser.parse_args()

    from gpt_oss_interp.backends.transformers_gpt_oss import GPTOSSTransformersBackend
    from gpt_oss_interp.readouts.tuned_lens import (
        TunedLensTranslators,
        MLPTranslator,
        make_chained_translator,
        measure_translation_gap,
        train_tuned_lens,
        train_mlp_translator,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mlp_layer_indices = [int(x.strip()) for x in args.mlp_layers.split(",")]

    print(f"Loading backend: {args.model}")
    backend = GPTOSSTransformersBackend(model_name=args.model, local_files_only=True)

    prompts = [
        line.strip()
        for line in Path(args.corpus).read_text().splitlines()
        if line.strip()
    ]
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    print(f"Loaded {len(prompts)} prompts from {args.corpus}", flush=True)

    print(f"\nLoading base translators (T1) from {args.base_translators}")
    t1 = TunedLensTranslators.load(args.base_translators)

    # ── Experiment 1: residual T2 ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Experiment 1: Residual T2 (rank-64 on frozen T1 residual)")
    print(f"{'='*60}")
    t2_path = out_dir / "translators_t2_residual.pt"

    if t2_path.exists() and args.n_epochs == 0:
        # Load saved T2 rather than overwriting it with an untrained one
        print(f"Loading saved T2 from {t2_path} (--n-epochs 0 = skip training)", flush=True)
        t2 = TunedLensTranslators.load(t2_path)
    else:
        print(f"Rank={args.rank}, epochs={args.n_epochs}, lr={args.lr}")
        print("T1 is frozen; T2 learns to correct KL remaining after T1.\n")
        t2 = train_tuned_lens(
            backend,
            prompts,
            rank=args.rank,
            n_epochs=args.n_epochs,
            lr=args.lr,
            base_translators=t1,
            verbose=True,
        )
        t2.save(t2_path)
        print(f"\nT2 saved to {t2_path}", flush=True)

    # Measure gap: raw / T1 alone / T1+T2 chained
    print("\nMeasuring KL gap: raw vs T1 vs T1+T2 ...")
    gap_raw_t1 = measure_translation_gap(backend, prompts[:20], translators=t1)
    chained = make_chained_translator(t1, t2)
    gap_chained = measure_translation_gap(backend, prompts[:20], translators=chained)

    print(f"\n{'Layer':>6}  {'Raw KL':>10}  {'T1 KL':>10}  {'T1+T2 KL':>10}  {'T2 extra':>10}")
    print("-" * 58)
    residual_results = []
    for l, (raw, t1_kl, chain_kl) in enumerate(zip(
        gap_chained["raw"],
        gap_raw_t1.get("tuned", []),
        gap_chained.get("tuned", []),
    )):
        extra = (t1_kl - chain_kl) / (t1_kl + 1e-12) * 100
        marker = " ◀" if raw > 1.0 else ""
        print(f"  L{l:02d}  {raw:10.4f}  {t1_kl:10.4f}  {chain_kl:10.4f}  {extra:9.1f}%{marker}")
        residual_results.append({
            "layer": l, "raw_kl": raw, "t1_kl": t1_kl,
            "chained_kl": chain_kl, "t2_extra_reduction_pct": extra,
        })

    (out_dir / "residual_t2_gap.json").write_text(
        json.dumps(residual_results, indent=2) + "\n"
    )

    # ── Experiment 2: MLP per target layer ────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Experiment 2: MLP translators for layers {mlp_layer_indices}")
    print(f"{'='*60}")
    print(f"Bottleneck={args.mlp_bottleneck}, epochs={args.n_epochs}, lr={args.lr}")
    print("Nonlinear probe: does GELU(W1 h) + W2 outperform affine T1?\n")

    mlp_results = []
    for layer_idx in mlp_layer_indices:
        affine_kl = gap_raw_t1["tuned"][layer_idx] if gap_raw_t1.get("tuned") else None
        raw_kl = gap_chained["raw"][layer_idx]
        print(f"\n--- L{layer_idx:02d}  raw={raw_kl:.4f}  affine_T1={affine_kl:.4f} ---")

        mlp = train_mlp_translator(
            backend,
            prompts,
            layer_idx,
            bottleneck=args.mlp_bottleneck,
            n_epochs=args.n_epochs,
            lr=args.lr,
            verbose=True,
        )
        mlp_path = out_dir / f"mlp_translator_L{layer_idx:02d}.pt"
        mlp.save(mlp_path)
        print(f"MLP translator saved to {mlp_path}")

        # Measure MLP KL at this layer on the same 20 prompts
        import torch
        import torch.nn.functional as F
        from gpt_oss_interp.capture.activation_cache import ActivationCache

        block_name = backend.structure.block_names[layer_idx]
        final_block = backend.structure.block_names[-1]
        norm = backend.structure.final_norm
        lm_head = backend.structure.lm_head
        norm_device = next(norm.parameters()).device
        model_device = next(backend.model.parameters()).device

        mlp_kl_sum = 0.0
        mlp_count = 0
        for prompt in prompts[:20]:
            ids = backend.tokenizer.encode(prompt, add_special_tokens=True)
            if len(ids) < 2:
                continue
            input_ids = torch.tensor([ids], device=model_device)

            cache = ActivationCache(detach=True, to_cpu=False)
            handles = cache.register(backend.model, [block_name, final_block])
            try:
                with torch.no_grad():
                    backend.model(input_ids)
            finally:
                for h in handles:
                    h.remove()

            rec_l = cache.last(block_name)
            rec_f = cache.last(final_block)
            if rec_l is None or rec_f is None:
                continue

            h_l = rec_l.tensor[0]
            h_f = rec_f.tensor[0]

            with torch.no_grad():
                p_final = F.softmax(
                    lm_head(norm(h_f.to(norm_device))).float(), dim=-1
                ).detach()
                h_mlp = mlp(h_l.to(next(mlp.parameters()).device))
                p_mlp = F.softmax(
                    lm_head(norm(h_mlp.to(norm_device))).float(), dim=-1
                )
                kl = float(F.kl_div(
                    p_mlp.log(), p_final, reduction="batchmean", log_target=False
                ).item())
            mlp_kl_sum += kl * h_l.shape[0]
            mlp_count += h_l.shape[0]

        mlp_kl = mlp_kl_sum / mlp_count if mlp_count else float("nan")
        affine_reduction = (raw_kl - affine_kl) / (raw_kl + 1e-12) * 100 if affine_kl else 0
        mlp_reduction = (raw_kl - mlp_kl) / (raw_kl + 1e-12) * 100
        extra_over_affine = (affine_kl - mlp_kl) / (affine_kl + 1e-12) * 100 if affine_kl else 0

        print(f"\n  L{layer_idx:02d} results:")
        print(f"    Raw KL:          {raw_kl:.4f} nats")
        print(f"    Affine T1 KL:    {affine_kl:.4f} nats  ({affine_reduction:.1f}% reduction)")
        print(f"    MLP KL:          {mlp_kl:.4f} nats  ({mlp_reduction:.1f}% reduction)")
        print(f"    MLP vs affine:   {extra_over_affine:+.1f}% extra reduction")
        if extra_over_affine > 5:
            print(f"    → Nonlinearity HELPS at L{layer_idx}: geometry gap is partially nonlinear")
        elif extra_over_affine > -2:
            print(f"    → MLP ≈ affine at L{layer_idx}: gap is rank-limited, not nonlinear")
        else:
            print(f"    → MLP WORSE than affine at L{layer_idx}: overfitting or training issue")

        mlp_results.append({
            "layer": layer_idx,
            "raw_kl": raw_kl,
            "affine_t1_kl": affine_kl,
            "mlp_kl": mlp_kl,
            "affine_reduction_pct": affine_reduction,
            "mlp_reduction_pct": mlp_reduction,
            "mlp_extra_over_affine_pct": extra_over_affine,
        })

    (out_dir / "mlp_probe_results.json").write_text(
        json.dumps(mlp_results, indent=2) + "\n"
    )

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    avg_t2_extra = sum(r["t2_extra_reduction_pct"] for r in residual_results[:12]) / 12
    print(f"Sequential T2: mean extra KL reduction L0-L11: {avg_t2_extra:.1f}%")
    for r in mlp_results:
        print(
            f"MLP L{r['layer']:02d}: affine={r['affine_t1_kl']:.3f}  "
            f"mlp={r['mlp_kl']:.3f}  extra={r['mlp_extra_over_affine_pct']:+.1f}%"
        )

    if avg_t2_extra > 10:
        print("\nInterpretation: KL floor is partly RANK-LIMITED (affine, insufficient capacity).")
    else:
        print("\nInterpretation: KL floor is NOT rank-limited by rank-64 affine.")

    mlp_helps = any(r["mlp_extra_over_affine_pct"] > 5 for r in mlp_results)
    if mlp_helps:
        print("MLP outperforms affine → NONLINEAR geometry gap at probed layers.")
        print("Recommendation: train MLP translators for all 24 layers.")
    else:
        print("MLP ≈ affine → geometry gap is AFFINE but rank-limited (or irreducible).")
        print("Recommendation: rank-128+ affine may help; MLP unlikely to add value at scale.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
