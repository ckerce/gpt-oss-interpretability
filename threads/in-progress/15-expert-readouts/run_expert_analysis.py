#!/usr/bin/env python3
"""Thread 15: MoE expert readout analysis.

Three measurements in priority order:

  1. Routing patterns  — sidecar, always works (MXFP4-safe)
  2. Layer logit-delta — ActivationCache + logit-lens projection, always works
  3. Expert vocab profiles — ExpertCapture, non-quantized checkpoints only

All three write to --output.  Measurements 1 and 2 always run.
Measurement 3 is silently skipped if no hookable expert modules are found
(MXFP4 fused model) — a note is printed instead.

Usage:
    # All three measurements, single prompt
    python threads/in-progress/15-expert-readouts/run_expert_analysis.py \\
        --model openai/gpt-oss-20b \\
        --prompt "The trophy did not fit in the suitcase because it was too small." \\
        --output runs/expert_readouts/

    # Multiple task families (shows cross-family routing differences)
    python threads/in-progress/15-expert-readouts/run_expert_analysis.py \\
        --model openai/gpt-oss-20b \\
        --task-suite \\
        --output runs/expert_readouts/

    # Expert vocab profiles (non-quantized checkpoint only)
    python threads/in-progress/15-expert-readouts/run_expert_analysis.py \\
        --model openai/gpt-oss-20b-base \\
        --task-suite \\
        --expert-profiles \\
        --output runs/expert_readouts/
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

# Representative prompts for each task family used in the main analysis set
TASK_SUITE = {
    "capitalization": [
        "The name of the first president of the United States is george",
        "My favorite city is paris, which is known for",
    ],
    "coreference": [
        "The trophy didn't fit in the suitcase because it was too small. The",
        "The developer argued with the designer because she didn't like the",
    ],
    "induction": [
        "The sequence continues: alpha beta gamma alpha beta",
        "In the pattern red blue green red blue",
    ],
    "syntax_agreement": [
        "The keys to the cabinet",
        "The player with the best statistics on both teams",
    ],
    "recency": [
        "I bought milk, eggs, and bread. Then I bought coffee. And finally I bought",
        "She visited Rome, then Paris, then London. Her last stop was",
    ],
}


###############################################################################
# Measurement 1: Routing patterns via sidecar
###############################################################################

def measure_routing_patterns(
    backend,
    prompts_by_task: dict[str, list[str]],
    arch,
) -> dict:
    """Capture routing decisions for all prompts and aggregate statistics."""
    from gossh.sidecar import MoeSidecar, RouterWeightExtractor

    print("\n=== Measurement 1: Routing patterns (sidecar) ===")

    weights = RouterWeightExtractor().extract(backend.model, backend.structure)
    if not weights:
        print("  WARNING: No router weights found. Skipping routing measurement.")
        return {}

    print(f"  Router weights: {len(weights)} layers extracted")

    # token_counts[layer][expert] = total tokens routed there
    token_counts: dict[str, dict[int, dict[int, int]]] = {}
    # Also accumulate cross-task
    combined_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    with MoeSidecar(weights, top_k=arch.top_k) as sidecar:
        backend.attach_sidecar(sidecar)

        for task_name, prompts in prompts_by_task.items():
            print(f"  [{task_name}]", end="", flush=True)
            task_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

            for prompt in prompts:
                decisions = backend.capture_routing(prompt)
                for d in decisions:
                    for token_experts in d.selected_experts.tolist():
                        for ei in token_experts:
                            task_counts[d.layer_idx][ei] += 1
                            combined_counts[d.layer_idx][ei] += 1
                print(".", end="", flush=True)

            token_counts[task_name] = {
                li: dict(expert_counts)
                for li, expert_counts in task_counts.items()
            }
            print()

        backend.detach_sidecar()

    # Compute routing entropy per layer
    n_experts = arch.num_experts
    entropy_by_layer: dict[int, float] = {}
    for layer_idx, counts in combined_counts.items():
        total = sum(counts.values()) or 1
        probs = [counts.get(e, 0) / total for e in range(n_experts)]
        ent = -sum(p * math.log(p + 1e-12) for p in probs)
        entropy_by_layer[layer_idx] = ent

    max_entropy = math.log(n_experts)
    print(f"\n  Routing entropy (max possible = {max_entropy:.3f} nats):")
    for li in sorted(entropy_by_layer):
        bar = "█" * int(entropy_by_layer[li] / max_entropy * 20)
        print(f"    L{li:2d}  {entropy_by_layer[li]:.3f}  {bar}")

    return {
        "by_task": {
            task: {str(li): counts for li, counts in by_layer.items()}
            for task, by_layer in token_counts.items()
        },
        "combined": {
            str(li): dict(counts) for li, counts in combined_counts.items()
        },
        "entropy_by_layer": {str(k): v for k, v in entropy_by_layer.items()},
        "max_entropy": max_entropy,
        "n_experts": n_experts,
    }


###############################################################################
# Measurement 2: Layer logit-delta
###############################################################################

def measure_layer_logit_delta(
    backend,
    prompts: list[str],
    top_k: int = 8,
) -> dict:
    """For each layer, compute Δ log p_l = log p_l - log p_{l-1}.

    Returns top promoted/suppressed tokens per layer, averaged over prompts.
    """
    from gossh.capture import ActivationCache

    print("\n=== Measurement 2: Layer logit-delta ===")

    final_norm = backend.structure.final_norm
    lm_head = backend.structure.lm_head
    norm_device = next(final_norm.parameters()).device

    # Accumulate logprob deltas: {layer: {token_id: [delta, ...]}}
    delta_accum: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))

    for p_idx, prompt in enumerate(prompts):
        print(f"  [{p_idx+1}/{len(prompts)}] {prompt[:60]!r}")
        input_ids = backend.tokenizer.encode(prompt, return_tensors="pt").to(backend.device)
        seq_len = input_ids.shape[1]

        cache = ActivationCache(detach=True, to_cpu=True)
        handles = cache.register(backend.model, backend.structure.block_names)
        try:
            with torch.no_grad():
                backend.model(input_ids)
        finally:
            for h in handles:
                h.remove()

        # Project each layer's hidden state → log probs
        layer_logp: list[torch.Tensor] = []
        for block_name in backend.structure.block_names:
            record = cache.last(block_name)
            if record is None:
                layer_logp.append(None)
                continue
            hidden = record.tensor  # [1, seq_len, D]
            with torch.no_grad():
                normed = final_norm(hidden.to(norm_device))
                logits = lm_head(normed).cpu().float()
                lp = torch.log_softmax(logits[0], dim=-1)  # [seq_len, V]
            layer_logp.append(lp)

        # Compute deltas at last token position
        pos = seq_len - 1
        for l_idx in range(1, len(layer_logp)):
            if layer_logp[l_idx] is None or layer_logp[l_idx - 1] is None:
                continue
            delta = (layer_logp[l_idx][pos] - layer_logp[l_idx - 1][pos])  # [V]
            topk_vals, topk_ids = delta.topk(top_k * 2)
            for v, i in zip(topk_vals.tolist(), topk_ids.tolist()):
                delta_accum[l_idx][i].append(v)

    # Average and find top promoted tokens per layer
    results: dict[int, list[dict]] = {}
    for layer_idx in sorted(delta_accum.keys()):
        avg_delta = {
            tid: sum(vals) / len(vals)
            for tid, vals in delta_accum[layer_idx].items()
        }
        top_promoted = sorted(avg_delta.items(), key=lambda x: -x[1])[:top_k]
        top_suppressed = sorted(avg_delta.items(), key=lambda x: x[1])[:top_k]

        results[layer_idx] = {
            "promoted": [
                {"token": backend.tokenizer.decode([tid]), "token_id": tid, "delta": d}
                for tid, d in top_promoted
            ],
            "suppressed": [
                {"token": backend.tokenizer.decode([tid]), "token_id": tid, "delta": d}
                for tid, d in top_suppressed
            ],
        }

    # Print summary table
    print(f"\n  Top promoted tokens at last position (averaged over {len(prompts)} prompts):")
    print(f"  {'Layer':6s}  {'Top promoted':40s}  {'Top suppressed'}")
    print(f"  {'------':6s}  {'-'*40}  {'-'*40}")
    for layer_idx, info in sorted(results.items()):
        promoted_str = " ".join(
            f"{e['token']!r}({e['delta']:+.2f})" for e in info["promoted"][:3]
        )
        suppressed_str = " ".join(
            f"{e['token']!r}({e['delta']:+.2f})" for e in info["suppressed"][:3]
        )
        print(f"  L{layer_idx:2d}     {promoted_str:40s}  {suppressed_str}")

    return {str(k): v for k, v in results.items()}


###############################################################################
# Measurement 3: Expert vocabulary profiles (non-quantized only)
###############################################################################

def measure_expert_vocab_profiles(
    backend,
    prompts: list[str],
    top_k: int = 10,
) -> dict | None:
    """Project each expert's output through lm_head to get its vocabulary profile.

    Returns None if no hookable expert modules are found (MXFP4).
    """
    from gossh.capture import ExpertCapture

    print("\n=== Measurement 3: Expert vocabulary profiles ===")

    expert_names = ExpertCapture.discover_expert_names(backend.model, backend.structure)
    if not expert_names:
        print("  No hookable expert modules found.")
        print("  This model is likely running under MXFP4 quantization.")
        print("  Expert vocabulary profiles require a non-quantized checkpoint.")
        print("  → Routing patterns (measurement 1) are available via the sidecar.")
        return None

    n_layers = len(expert_names)
    n_experts_per_layer = max(len(v) for v in expert_names.values())
    print(f"  Found {n_experts_per_layer} hookable experts × {n_layers} layers")

    final_norm = backend.structure.final_norm
    lm_head = backend.structure.lm_head
    norm_device = next(final_norm.parameters()).device

    # Accumulate mean logp per expert: {layer: {expert: [vocab_tensor, ...]}}
    # We accumulate log-probs to average in log space (geometric mean of distributions)
    accum: dict[int, dict[int, list[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))
    token_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    with ExpertCapture(detach=True, to_cpu=True) as cap:
        cap.register(backend.model, expert_names)

        for p_idx, prompt in enumerate(prompts):
            print(f"  [{p_idx+1}/{len(prompts)}] {prompt[:60]!r}")
            cap.clear()

            input_ids = backend.tokenizer.encode(prompt, return_tensors="pt").to(backend.device)
            with torch.no_grad():
                backend.model(input_ids)

            # Project captured expert outputs
            for layer_idx, experts in cap.captured.items():
                for expert_idx, outputs in experts.items():
                    if not outputs:
                        continue
                    stacked = torch.cat(outputs, dim=0)  # [n_tokens, hidden_dim]
                    with torch.no_grad():
                        normed = final_norm(stacked.to(norm_device))
                        logits = lm_head(normed).cpu().float()
                        lp = torch.log_softmax(logits, dim=-1)  # [n, V]
                    accum[layer_idx][expert_idx].append(lp.mean(dim=0))
                    token_counts[layer_idx][expert_idx] += stacked.shape[0]

    # Average log-probs → expert vocabulary profile
    profiles: dict[int, dict[int, dict]] = {}
    for layer_idx in sorted(accum.keys()):
        profiles[layer_idx] = {}
        for expert_idx in sorted(accum[layer_idx].keys()):
            lp_list = accum[layer_idx][expert_idx]
            if not lp_list:
                continue
            mean_lp = torch.stack(lp_list).mean(dim=0)  # [V]
            topk_vals, topk_ids = mean_lp.topk(top_k)
            profiles[layer_idx][expert_idx] = {
                "top_tokens": [
                    {"token": backend.tokenizer.decode([tid]), "token_id": tid, "logp": float(v)}
                    for tid, v in zip(topk_ids.tolist(), topk_vals.tolist())
                ],
                "tokens_routed": token_counts[layer_idx][expert_idx],
            }

    # Print summary (early, mid, late layers)
    n_layers = backend.structure.num_layers
    show_layers = sorted({0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1}
                         & set(profiles.keys()))

    for layer_idx in show_layers:
        print(f"\n  Layer {layer_idx} expert profiles (top-5 tokens):")
        print(f"  {'Expert':8s}  {'Tokens routed':14s}  Top-5 vocabulary")
        print(f"  {'------':8s}  {'-'*14}  {'-'*50}")
        for expert_idx in sorted(profiles[layer_idx].keys())[:8]:
            info = profiles[layer_idx][expert_idx]
            tokens_str = " ".join(
                f"{e['token']!r}" for e in info["top_tokens"][:5]
            )
            print(f"  E{expert_idx:2d}       {info['tokens_routed']:8d}        {tokens_str}")

    return {
        str(li): {
            str(ei): info
            for ei, info in experts.items()
        }
        for li, experts in profiles.items()
    }


###############################################################################
# Report generation
###############################################################################

def write_report(
    output_dir: Path,
    routing_data: dict,
    delta_data: dict,
    profile_data: dict | None,
    prompts_by_task: dict[str, list[str]],
) -> None:
    lines = [
        "# Thread 15: MoE Expert Readout Analysis",
        "",
        f"Prompts analyzed: {sum(len(v) for v in prompts_by_task.values())} "
        f"across {len(prompts_by_task)} task families",
        "",
    ]

    if routing_data:
        lines += [
            "## Measurement 1: Routing Entropy by Layer",
            "",
            "| Layer | Entropy (nats) | % of max |",
            "| ---: | ---: | ---: |",
        ]
        max_ent = routing_data.get("max_entropy", 1.0)
        for li_str, ent in sorted(routing_data.get("entropy_by_layer", {}).items(), key=lambda x: int(x[0])):
            pct = 100 * ent / max_ent
            lines.append(f"| L{li_str} | {ent:.3f} | {pct:.1f}% |")
        lines.append("")

    if delta_data:
        lines += [
            "## Measurement 2: Layer Logit-Delta (Top Promoted Tokens at Last Position)",
            "",
            "| Layer | Top promoted | Top suppressed |",
            "| ---: | --- | --- |",
        ]
        for li_str in sorted(delta_data.keys(), key=int):
            info = delta_data[li_str]
            promoted = " ".join(
                f"`{e['token']}`({e['delta']:+.2f})" for e in info["promoted"][:3]
            )
            suppressed = " ".join(
                f"`{e['token']}`({e['delta']:+.2f})" for e in info["suppressed"][:3]
            )
            lines.append(f"| L{li_str} | {promoted} | {suppressed} |")
        lines.append("")

    if profile_data is None:
        lines += [
            "## Measurement 3: Expert Vocabulary Profiles",
            "",
            "_Skipped: no hookable expert modules found (MXFP4 quantization)._",
            "_Use a non-quantized checkpoint to enable per-expert readouts._",
            "",
        ]
    else:
        lines += [
            "## Measurement 3: Expert Vocabulary Profiles",
            "",
            "_Top-5 vocabulary predictions for each expert (averaged over all activations)._",
            "",
        ]
        for li_str in sorted(profile_data.keys(), key=int):
            lines.append(f"### Layer {li_str}\n")
            lines.append("| Expert | Tokens routed | Top-5 tokens |")
            lines.append("| ---: | ---: | --- |")
            for ei_str, info in sorted(profile_data[li_str].items(), key=lambda x: int(x[0])):
                tokens = " ".join(f"`{t['token']}`" for t in info["top_tokens"][:5])
                lines.append(f"| E{ei_str} | {info['tokens_routed']} | {tokens} |")
            lines.append("")

    report = "\n".join(lines) + "\n"
    (output_dir / "expert_readout_report.md").write_text(report)
    print(f"\nReport written to {output_dir / 'expert_readout_report.md'}")


###############################################################################
# Main
###############################################################################

def main() -> int:
    parser = argparse.ArgumentParser(description="MoE expert readout analysis")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--prompt", action="append", default=[],
                        help="One or more prompts (repeatable)")
    parser.add_argument("--task-suite", action="store_true",
                        help="Use the built-in 5-family task suite")
    parser.add_argument("--expert-profiles", action="store_true",
                        help="Attempt measurement 3 (non-quantized models only)")
    parser.add_argument("--top-k", type=int, default=8,
                        help="Top-k tokens in logit-delta and expert profiles")
    parser.add_argument("--output", default="runs/expert_readouts/")
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()

    from gossh.backends.gpt_oss import GPTOSSTransformersBackend

    print(f"Loading backend: {args.model}")
    backend = GPTOSSTransformersBackend(
        args.model, local_files_only=args.local_files_only
    )
    arch = backend._arch

    # Build prompt set
    if args.task_suite:
        prompts_by_task = TASK_SUITE
    elif args.prompt:
        prompts_by_task = {"custom": args.prompt}
    else:
        # Default: a single cross-family demo
        prompts_by_task = {
            "induction": ["The sequence: alpha beta gamma alpha beta"],
            "coreference": ["The trophy didn't fit in the suitcase because it was too small."],
        }

    all_prompts = [p for prompts in prompts_by_task.values() for p in prompts]
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run measurements
    routing_data = measure_routing_patterns(backend, prompts_by_task, arch)
    delta_data = measure_layer_logit_delta(backend, all_prompts, top_k=args.top_k)
    profile_data = (
        measure_expert_vocab_profiles(backend, all_prompts, top_k=args.top_k)
        if args.expert_profiles else None
    )

    if not args.expert_profiles:
        print("\n(Skipping measurement 3. Pass --expert-profiles to enable.)")

    # Save outputs
    if routing_data:
        (out_dir / "routing_patterns.json").write_text(json.dumps(routing_data, indent=2))
    if delta_data:
        (out_dir / "layer_logit_delta.json").write_text(json.dumps(delta_data, indent=2))
    if profile_data is not None:
        (out_dir / "expert_vocab_profiles.json").write_text(json.dumps(profile_data, indent=2))

    write_report(out_dir, routing_data, delta_data, profile_data, prompts_by_task)

    print(f"\nAll outputs written to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
