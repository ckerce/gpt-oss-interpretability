#!/usr/bin/env python3
"""Capture and visualize MoE routing decisions for gpt-oss-20b.

Shows which experts are selected per layer and their weights,
providing insight into the model's routing behavior.

Usage:
    python scripts/capture_routing.py \
        --model openai/gpt-oss-20b \
        --prompt "The trophy would not fit in the suitcase because" \
        --output runs/routing_demo/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture MoE routing decisions")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    from gpt_oss_interp.backends.transformers_gpt_oss import GPTOSSTransformersBackend

    print(f"Initializing backend: {args.model}")
    backend = GPTOSSTransformersBackend(model_name=args.model)

    print(f"\nCapturing routing for: {args.prompt!r}")
    decisions = backend.capture_routing(args.prompt)

    if not decisions:
        print("No routing decisions captured. Model may not have discoverable gate modules.")
        print("Run scripts/inspect_model.py to examine the model structure.")
        return 1

    # Print summary
    print(f"\nCaptured {len(decisions)} routing decisions across layers\n")
    print("| Layer | Tokens | Top Expert | Weight | 2nd Expert | Weight |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: |")

    report_data = []
    for d in decisions:
        # Average expert weights across tokens
        avg_weights = d.expert_weights.float().mean(dim=0)  # [top_k] after averaging batch
        if avg_weights.ndim > 1:
            avg_weights = avg_weights.mean(dim=0)
        avg_indices = d.selected_experts[0] if d.selected_experts.ndim > 1 else d.selected_experts
        if avg_indices.ndim > 1:
            avg_indices = avg_indices[0]  # Take first token as representative

        top1_expert = avg_indices[0].item()
        top1_weight = avg_weights[0].item()
        top2_expert = avg_indices[1].item() if avg_indices.shape[0] > 1 else -1
        top2_weight = avg_weights[1].item() if avg_weights.shape[0] > 1 else 0.0

        print(f"| {d.layer_idx} | {d.token_count} | {top1_expert} | {top1_weight:.3f} | {top2_expert} | {top2_weight:.3f} |")

        report_data.append({
            "layer": d.layer_idx,
            "token_count": d.token_count,
            "selected_experts": d.selected_experts.tolist(),
            "expert_weights": d.expert_weights.tolist(),
        })

    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "routing_decisions.json").write_text(
            json.dumps({"prompt": args.prompt, "model": args.model, "decisions": report_data}, indent=2) + "\n"
        )
        print(f"\nData written to {out_dir}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
