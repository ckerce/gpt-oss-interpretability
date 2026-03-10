###############################################################################
#
# Logit-lens style per-layer readouts
#
###############################################################################

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


@dataclass
class LayerPrediction:
    layer_idx: int
    position: int
    top_tokens: list[str]
    top_logprobs: list[float]
    target_token: str | None = None
    target_rank: int | None = None
    target_logprob: float | None = None


@dataclass
class LogitLensResult:
    """Per-layer logit-lens readouts for a single forward pass."""
    predictions: list[LayerPrediction]
    num_layers: int
    num_positions: int

    def layer_slice(self, layer_idx: int) -> list[LayerPrediction]:
        return [p for p in self.predictions if p.layer_idx == layer_idx]

    def position_slice(self, position: int) -> list[LayerPrediction]:
        return [p for p in self.predictions if p.position == position]

    def convergence_layer(self, position: int) -> int | None:
        """Return the earliest layer whose top prediction matches the final layer."""
        preds = self.position_slice(position)
        if not preds:
            return None
        final = preds[-1].top_tokens[0] if preds[-1].top_tokens else None
        for p in preds:
            if p.top_tokens and p.top_tokens[0] == final:
                return p.layer_idx
        return None


def run_logit_lens(
    model: nn.Module,
    input_ids: torch.Tensor,
    tokenizer: Any,
    final_norm: nn.Module,
    lm_head: nn.Module,
    block_modules: list[nn.Module],
    top_k: int = 5,
    target_ids: torch.Tensor | None = None,
    positions: list[int] | None = None,
) -> LogitLensResult:
    """Run a logit-lens pass over all transformer blocks.

    For each layer, projects the hidden state through the final norm and
    unembedding to recover per-layer next-token predictions.

    Args:
        model: The full model (used only for the embedding pass).
        input_ids: Token ids, shape [1, seq_len].
        tokenizer: Tokenizer for decoding token ids.
        final_norm: The final RMSNorm / LayerNorm before the lm_head.
        lm_head: The unembedding linear layer.
        block_modules: List of transformer block modules in order.
        top_k: Number of top predictions to record per position per layer.
        target_ids: Optional token ids to track rank and logprob for.
        positions: If given, only report these token positions (0-indexed).

    Returns:
        LogitLensResult with per-layer, per-position predictions.
    """
    from gpt_oss_interp.capture.activation_cache import ActivationCache

    # Capture hidden states at the output of every block
    cache = ActivationCache(detach=True, to_cpu=True)
    named = dict(model.named_modules())

    # Find block names
    block_names = []
    for name, module in named.items():
        if module in block_modules:
            block_names.append(name)

    if len(block_names) != len(block_modules):
        # Fallback: register by index
        block_names = []
        for idx, block in enumerate(block_modules):
            for name, module in named.items():
                if module is block:
                    block_names.append(name)
                    break

    handles = cache.register(model, block_names)
    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        for h in handles:
            h.remove()

    seq_len = input_ids.shape[1]
    if positions is None:
        positions = list(range(seq_len))
    num_layers = len(block_modules)

    # Move final_norm and lm_head weights to CPU for projection
    predictions: list[LayerPrediction] = []
    for layer_idx, block_name in enumerate(block_names):
        record = cache.last(block_name)
        if record is None:
            continue
        hidden = record.tensor  # [1, seq_len, hidden_dim], on CPU

        # Project through final norm and lm_head
        with torch.no_grad():
            normed = final_norm(hidden.to(next(final_norm.parameters()).device))
            logits = lm_head(normed).cpu().float()  # [1, seq_len, vocab]
            log_probs = torch.log_softmax(logits[0], dim=-1)

        for pos in positions:
            if pos >= seq_len:
                continue
            lp = log_probs[pos]
            topk_vals, topk_ids = torch.topk(lp, k=top_k)
            top_tokens = [tokenizer.decode([tid]) for tid in topk_ids.tolist()]
            top_logprobs = topk_vals.tolist()

            pred = LayerPrediction(
                layer_idx=layer_idx,
                position=pos,
                top_tokens=top_tokens,
                top_logprobs=top_logprobs,
            )

            if target_ids is not None and pos < target_ids.shape[-1]:
                target_id = target_ids[0, pos].item() if target_ids.ndim > 1 else target_ids[pos].item()
                pred.target_token = tokenizer.decode([target_id])
                pred.target_logprob = lp[target_id].item()
                pred.target_rank = int((lp > lp[target_id]).sum().item())

            predictions.append(pred)

    return LogitLensResult(
        predictions=predictions,
        num_layers=num_layers,
        num_positions=len(positions),
    )


def format_logit_lens_table(result: LogitLensResult, last_n_positions: int = 5) -> str:
    """Format logit-lens results as a readable markdown table."""
    lines = ["| Layer | Position | Top-1 Prediction | Logprob | Target | Target Rank |"]
    lines.append("| ---: | ---: | --- | ---: | --- | ---: |")

    # Only show the last N positions for readability
    all_positions = sorted({p.position for p in result.predictions})
    show_positions = set(all_positions[-last_n_positions:])

    for pred in result.predictions:
        if pred.position not in show_positions:
            continue
        top1 = repr(pred.top_tokens[0]) if pred.top_tokens else "?"
        lp = f"{pred.top_logprobs[0]:.3f}" if pred.top_logprobs else "?"
        target = repr(pred.target_token) if pred.target_token else ""
        rank = str(pred.target_rank) if pred.target_rank is not None else ""
        lines.append(f"| {pred.layer_idx} | {pred.position} | {top1} | {lp} | {target} | {rank} |")

    return "\n".join(lines)
