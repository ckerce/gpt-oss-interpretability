#!/usr/bin/env python3
"""Compute a gauge-safe CASCADE target for one prompt position.

This is a reference implementation of the centered least-squares target:

    x_e* = argmin_x ||A x - b||_2^2

where:

    A = C W
    b = C(log p_teacher - W x_t)

and C is the centering operator on the vocabulary dimension.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


###############################################################################
#
# Helpers
#
###############################################################################


def _relative_residual(matrix: torch.Tensor, solution: torch.Tensor, rhs: torch.Tensor) -> float:
    residual = matrix @ solution - rhs
    denom = rhs.norm().item()
    if denom == 0.0:
        return residual.norm().item()
    return residual.norm().item() / denom


def _distribution_kl(
    teacher_log_probs: torch.Tensor,
    reconstructed_logits: torch.Tensor,
) -> float:
    teacher_probs = teacher_log_probs.exp()
    recon_log_probs = torch.log_softmax(reconstructed_logits, dim=-1)
    return torch.sum(teacher_probs * (teacher_log_probs - recon_log_probs)).item()


###############################################################################
#
# Main
#
###############################################################################


def main() -> int:
    parser = argparse.ArgumentParser(description="Reference CASCADE target computation")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--position",
        type=int,
        default=-1,
        help="Token position to analyze; negative values index from the end",
    )
    parser.add_argument("--output", default=None, help="Optional output directory")
    args = parser.parse_args()

    from gpt_oss_interp.backends.transformers_gpt_oss import GPTOSSTransformersBackend

    print(f"Initializing backend: {args.model}")
    backend = GPTOSSTransformersBackend(model_name=args.model)

    tokenizer = backend.tokenizer
    model = backend.model
    structure = backend.structure

    if structure.embed is None or structure.lm_head is None:
        raise RuntimeError("Model structure is missing the embedding layer or lm_head")

    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(backend.device)
    seq_len = input_ids.shape[1]
    position = args.position if args.position >= 0 else seq_len + args.position
    if position < 0 or position >= seq_len:
        raise ValueError(f"Position {args.position} resolves to invalid index {position} for seq_len={seq_len}")

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, position].float().cpu()
        log_probs = torch.log_softmax(logits, dim=-1)

        embed_out = structure.embed(input_ids)
        if isinstance(embed_out, tuple):
            embed_out = embed_out[0]
        x_t = embed_out[0, position].float().cpu()

    W = structure.lm_head.weight.detach().float().cpu()  # [V, d]
    W_centered = W - W.mean(dim=0, keepdim=True)

    centered_teacher = log_probs - log_probs.mean()
    centered_xt = W @ x_t
    centered_xt = centered_xt - centered_xt.mean()
    rhs = centered_teacher - centered_xt

    solution = torch.linalg.lstsq(W_centered, rhs).solution
    reconstructed_logits = W @ (x_t + solution)

    target_id = int(torch.argmax(log_probs).item())
    metrics = {
        "position": position,
        "prompt": args.prompt,
        "model": args.model,
        "target_token_id": target_id,
        "target_token": tokenizer.decode([target_id]),
        "relative_residual": _relative_residual(W_centered, solution, rhs),
        "teacher_top_logprob": float(log_probs[target_id].item()),
        "reconstructed_top_logprob": float(torch.log_softmax(reconstructed_logits, dim=-1)[target_id].item()),
        "kl_teacher_to_reconstructed": _distribution_kl(log_probs, reconstructed_logits),
        "solution_norm": float(solution.norm().item()),
        "rhs_norm": float(rhs.norm().item()),
        "rank_upper_bound": int(min(W_centered.shape)),
    }

    print("\n## CASCADE Reference Metrics")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "cascade_reference.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"\nWrote {out_dir / 'cascade_reference.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
