#!/usr/bin/env python3
"""Run a starter Bregman-conditioning analysis on gpt-oss prompts.

This script computes the hidden-space softmax Hessian at each transformer
layer for one or more prompts, then summarizes:

- trace(H)
- effective rank(H)
- condition number(H)
- retained top-k probability mass
- optional cosine between a vocab steering direction W[a] - W[b] and H v
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch


def _load_prompts(args: argparse.Namespace) -> list[str]:
    prompts: list[str] = []
    if args.prompt:
        prompts.extend(args.prompt)
    if args.prompt_file:
        for line in Path(args.prompt_file).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                prompts.append(line)
    if not prompts:
        raise ValueError("Provide at least one --prompt or --prompt_file")
    if args.max_prompts is not None:
        prompts = prompts[: args.max_prompts]
    return prompts


def _resolve_single_token_id(tokenizer, text: str) -> int:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if len(encoded) != 1:
        raise ValueError(
            f"Token string {text!r} maps to {len(encoded)} tokens, but this runner "
            "currently requires single-token steering directions."
        )
    return int(encoded[0])


def _parse_positions(raw: str | None, seq_len: int) -> list[int]:
    if raw is None or raw == "last":
        return [seq_len - 1]
    positions = [int(part.strip()) for part in raw.split(",") if part.strip()]
    valid = [pos for pos in positions if 0 <= pos < seq_len]
    if not valid:
        raise ValueError(f"No requested positions were valid for seq_len={seq_len}")
    return valid


def main() -> int:
    parser = argparse.ArgumentParser(description="Bregman-conditioning analysis on gpt-oss")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--prompt_file", default=None)
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--positions", default="last", help="Comma-separated token positions or 'last'")
    parser.add_argument("--top_k_vocab", type=int, default=2048)
    parser.add_argument("--token_a", default=None, help="Optional token text for W[a]-W[b] direction")
    parser.add_argument("--token_b", default=None, help="Optional token text for W[a]-W[b] direction")
    parser.add_argument("--output", default=None, help="Optional output directory")
    args = parser.parse_args()

    from gpt_oss_interp.backends.transformers_gpt_oss import GPTOSSTransformersBackend
    from gpt_oss_interp.capture.activation_cache import ActivationCache
    from gpt_oss_interp.steering.bregman import (
        analyze_bregman_state,
        format_bregman_summary,
        summarize_bregman_metrics,
        unembedding_direction,
    )

    prompts = _load_prompts(args)
    print(f"Initializing backend: {args.model}")
    backend = GPTOSSTransformersBackend(model_name=args.model)

    structure = backend.structure
    model = backend.model
    tokenizer = backend.tokenizer
    lm_head = structure.lm_head
    final_norm = structure.final_norm
    if lm_head is None or final_norm is None:
        raise RuntimeError("Model structure is missing lm_head or final_norm.")

    # Convert to fp32 once here; softmax_hessian_from_hidden will skip
    # the per-call conversion when the dtype is already float32.
    lm_head_weight = lm_head.weight.detach().float().cpu()
    lm_head_bias = getattr(lm_head, "bias", None)
    if lm_head_bias is not None:
        lm_head_bias = lm_head_bias.detach().float().cpu()

    direction = None
    direction_info = None
    if args.token_a or args.token_b:
        if not args.token_a or not args.token_b:
            raise ValueError("Provide both --token_a and --token_b for a direction cosine.")
        token_a_id = _resolve_single_token_id(tokenizer, args.token_a)
        token_b_id = _resolve_single_token_id(tokenizer, args.token_b)
        direction = unembedding_direction(lm_head_weight, token_a_id, token_b_id)
        direction_info = {
            "token_a": args.token_a,
            "token_b": args.token_b,
            "token_a_id": token_a_id,
            "token_b_id": token_b_id,
        }

    metrics_by_layer = defaultdict(list)
    sample_rows: list[dict[str, object]] = []
    norm_device = next(final_norm.parameters()).device

    for prompt_idx, prompt in enumerate(prompts):
        print(f"[{prompt_idx + 1}/{len(prompts)}] {prompt!r}")
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(backend.device)
        seq_len = input_ids.shape[1]
        positions = _parse_positions(args.positions, seq_len)

        cache = ActivationCache(detach=True, to_cpu=True)
        handles = cache.register(model, structure.block_names)
        try:
            with torch.no_grad():
                model(input_ids)
        finally:
            for handle in handles:
                handle.remove()

        for layer_idx, block_name in enumerate(structure.block_names):
            record = cache.last(block_name)
            if record is None:
                continue
            hidden = record.tensor
            with torch.no_grad():
                normed = final_norm(hidden.to(norm_device))[0].cpu()

            for pos in positions:
                metrics = analyze_bregman_state(
                    normed[pos],
                    lm_head_weight,
                    lm_head_bias=lm_head_bias,
                    top_k_vocab=args.top_k_vocab,
                    direction=direction,
                )
                metrics_by_layer[layer_idx].append(metrics)
                sample_rows.append(
                    {
                        "prompt_index": prompt_idx,
                        "prompt": prompt,
                        "layer_idx": layer_idx,
                        "position": pos,
                        "trace": metrics.trace,
                        "effective_rank": metrics.effective_rank,
                        "condition_number": metrics.condition_number,
                        "numerical_rank": metrics.numerical_rank,
                        "mass_covered": metrics.mass_covered,
                        "cosine_primal_dual": metrics.cosine_primal_dual,
                    }
                )

    summaries = summarize_bregman_metrics(metrics_by_layer, top_k_vocab=args.top_k_vocab)
    table = format_bregman_summary(summaries)
    print()
    print(table)

    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)

        report_lines = [
            "# Bregman Conditioning",
            "",
            f"Model: `{args.model}`",
            f"Prompts analyzed: {len(prompts)}",
            f"Top-k vocabulary approximation: {args.top_k_vocab}",
            "",
        ]
        if direction_info is not None:
            report_lines.extend(
                [
                    "Direction:",
                    f"- token_a: `{direction_info['token_a']}` (id={direction_info['token_a_id']})",
                    f"- token_b: `{direction_info['token_b']}` (id={direction_info['token_b_id']})",
                    "",
                ]
            )
        report_lines.extend(
            [
                table,
                "",
                "Interpretation:",
                "- Higher effective rank means more of the local geometry is available to linear steering.",
                "- Lower condition number means less distortion between primal and dual directions.",
                "- Lower mass coverage means the top-k approximation may be too aggressive for that layer.",
            ]
        )
        (out_dir / "bregman_conditioning.md").write_text("\n".join(report_lines) + "\n")

        payload = {
            "model": args.model,
            "prompts": prompts,
            "top_k_vocab": args.top_k_vocab,
            "direction": direction_info,
            "layer_summaries": [
                {
                    "layer_idx": row.layer_idx,
                    "n_samples": row.n_samples,
                    "top_k_vocab": row.top_k_vocab,
                    "trace_mean": row.trace_mean,
                    "effective_rank_mean": row.effective_rank_mean,
                    "condition_number_median": row.condition_number_median,
                    "mass_covered_mean": row.mass_covered_mean,
                    "cosine_mean": row.cosine_mean,
                    "cosine_std": row.cosine_std,
                }
                for row in summaries
            ],
            "samples": sample_rows,
        }
        (out_dir / "bregman_conditioning.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nReports written to {out_dir}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
