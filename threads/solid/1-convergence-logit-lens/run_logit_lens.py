#!/usr/bin/env python3
"""Run a logit-lens analysis on gpt-oss-20b.

Produces per-layer next-token predictions showing when the model
"knows" the answer across layers.

Usage:
    python threads/solid/1-convergence-logit-lens/run_logit_lens.py \
        --model openai/gpt-oss-20b \
        --prompt "The capital of France is" \
        --output runs/logit_lens_demo/

    python threads/solid/1-convergence-logit-lens/run_logit_lens.py \
        --model openai/gpt-oss-20b \
        --prompt "The trophy would not fit in the suitcase because the suitcase was too" \
        --top_k 10 \
        --output runs/logit_lens_demo/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_positions(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Logit-lens analysis on gpt-oss")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--last_n", type=int, default=5, help="Show last N token positions")
    parser.add_argument(
        "--positions",
        default=None,
        help="Optional comma-separated token positions to analyze; default is all positions",
    )
    parser.add_argument("--output", default=None, help="Output directory for reports")
    args = parser.parse_args()

    from gpt_oss_interp.backends.transformers_gpt_oss import GPTOSSTransformersBackend
    from gpt_oss_interp.readouts.logit_lens import format_logit_lens_table

    print(f"Initializing backend: {args.model}")
    backend = GPTOSSTransformersBackend(model_name=args.model)

    print(f"\nRunning logit lens on: {args.prompt!r}")
    positions = _parse_positions(args.positions)
    result = backend.run_logit_lens(args.prompt, top_k=args.top_k, positions=positions)

    table = format_logit_lens_table(result, last_n_positions=args.last_n)
    print(f"\n{table}")

    # Convergence analysis
    all_positions = sorted({p.position for p in result.predictions})
    print("\n## Convergence Analysis")
    for pos in all_positions[-args.last_n:]:
        conv = result.convergence_layer(pos)
        preds = result.position_slice(pos)
        final_pred = preds[-1].top_tokens[0] if preds else "?"
        token_at_pos = f"position {pos}"
        print(f"  {token_at_pos}: converges at layer {conv}, final prediction = {final_pred!r}")

    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save markdown report
        report_lines = [
            f"# Logit Lens: {args.prompt!r}",
            f"\nModel: `{args.model}`\n",
            table,
            "\n## Convergence\n",
        ]
        for pos in all_positions[-args.last_n:]:
            conv = result.convergence_layer(pos)
            preds = result.position_slice(pos)
            final_pred = preds[-1].top_tokens[0] if preds else "?"
            report_lines.append(f"- Position {pos}: converges at layer {conv} → {final_pred!r}")

        (out_dir / "logit_lens_report.md").write_text("\n".join(report_lines) + "\n")

        tracked_positions: list[dict[str, object]] = []
        for pos in all_positions:
            preds = result.position_slice(pos)
            if not preds:
                continue
            target_id = preds[0].target_token_id
            target_token = preds[0].target_token
            tracked_positions.append(
                {
                    "position": pos,
                    "target_token_id": target_id,
                    "target_token": target_token,
                    "convergence_layer": result.convergence_layer(pos),
                    "trajectory": [
                        {
                            "layer": pred.layer_idx,
                            "target_logprob": pred.target_logprob,
                            "target_rank": pred.target_rank,
                            "top_token_ids": pred.top_token_ids,
                            "top_tokens": pred.top_tokens,
                            "top_logprobs": pred.top_logprobs,
                        }
                        for pred in preds
                    ],
                }
            )

        # Save raw data as JSON
        data = {
            "prompt": args.prompt,
            "model": args.model,
            "num_layers": result.num_layers,
            "tracked_positions": tracked_positions,
            "predictions": [
                {
                    "layer": p.layer_idx,
                    "position": p.position,
                    "top_token_ids": p.top_token_ids,
                    "top_tokens": p.top_tokens,
                    "top_logprobs": p.top_logprobs,
                    "target_token_id": p.target_token_id,
                    "target_token": p.target_token,
                    "target_rank": p.target_rank,
                    "target_logprob": p.target_logprob,
                }
                for p in result.predictions
            ],
        }
        (out_dir / "logit_lens_data.json").write_text(
            json.dumps(data, indent=2) + "\n"
        )
        print(f"\nReports written to {out_dir}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
