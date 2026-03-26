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
    top_token_ids: list[int]
    top_tokens: list[str]
    top_logprobs: list[float]
    target_token_id: int | None = None
    target_token: str | None = None
    target_rank: int | None = None
    target_logprob: float | None = None


@dataclass
class LogitLensResult:
    """Per-layer logit-lens readouts for a single forward pass."""
    predictions: list[LayerPrediction]
    num_layers: int
    num_positions: int
    tracked_target_ids: dict[int, int] = field(default_factory=dict)

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

    def tracked_target_id(self, position: int) -> int | None:
        return self.tracked_target_ids.get(position)

    def target_convergence_layer(self, position: int) -> int | None:
        """Return the earliest layer where the tracked target reaches rank 0."""
        preds = self.position_slice(position)
        if not preds:
            return None
        for p in preds:
            if p.target_rank == 0:
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
    translators: "Any | None" = None,
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

    # First pass: determine the fixed tracked token per position from the final layer.
    final_layer_targets: dict[int, int] = {}
    final_block_name = block_names[-1] if block_names else None
    if final_block_name is not None:
        record = cache.last(final_block_name)
        if record is not None:
            hidden = record.tensor
            with torch.no_grad():
                normed = final_norm(hidden.to(next(final_norm.parameters()).device))
                logits = lm_head(normed).cpu().float()
                final_log_probs = torch.log_softmax(logits[0], dim=-1)
            for pos in positions:
                if pos >= seq_len:
                    continue
                final_layer_targets[pos] = int(torch.argmax(final_log_probs[pos]).item())

    tracked_target_ids: dict[int, int] = dict(final_layer_targets)
    if target_ids is not None:
        for pos in positions:
            if pos >= seq_len:
                continue
            tracked_target_ids[pos] = int(
                target_ids[0, pos].item() if target_ids.ndim > 1 else target_ids[pos].item()
            )

    predictions: list[LayerPrediction] = []
    for layer_idx, block_name in enumerate(block_names):
        record = cache.last(block_name)
        if record is None:
            continue
        hidden = record.tensor  # [1, seq_len, hidden_dim], on CPU

        # Project through (optional translator +) final norm and lm_head
        with torch.no_grad():
            h = hidden  # [1, seq_len, hidden_dim]
            if translators is not None and layer_idx < translators.n_layers:
                # Apply tuned-lens translator: T_l(h) = h + U(V^T h) + b
                t_dev = translators.U[layer_idx].device
                h_flat = h[0].to(t_dev)  # [seq_len, hidden_dim]
                h_flat = translators.translate(h_flat, layer_idx)
                h = h_flat.unsqueeze(0).cpu()
            normed = final_norm(h.to(next(final_norm.parameters()).device))
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
                top_token_ids=topk_ids.tolist(),
                top_tokens=top_tokens,
                top_logprobs=top_logprobs,
            )

            tracked_target_id: int | None = tracked_target_ids.get(pos)

            if tracked_target_id is not None:
                pred.target_token_id = int(tracked_target_id)
                pred.target_token = tokenizer.decode([tracked_target_id])
                pred.target_logprob = lp[tracked_target_id].item()
                pred.target_rank = int((lp > lp[tracked_target_id]).sum().item())

            predictions.append(pred)

    return LogitLensResult(
        predictions=predictions,
        num_layers=num_layers,
        num_positions=len(positions),
        tracked_target_ids=tracked_target_ids,
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
