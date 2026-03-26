"""MoE router decision types and hook-based capture for gossh."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class RouterDecision:
    """Routing decisions for one layer and one forward pass."""
    layer_idx: int
    selected_experts: torch.Tensor      # [seq_len, top_k] int64
    expert_weights: torch.Tensor        # [seq_len, top_k] float32
    gate_logits: torch.Tensor | None    # [seq_len, num_experts] float32
    token_count: int

    @property
    def top_k(self) -> int:
        return self.selected_experts.shape[-1]


class RouterCapture:
    """Capture MoE routing decisions via forward hooks on gate modules.

    Works for non-quantized models where gate output is accessible via hooks.
    For MXFP4-quantized models, use ``MoeSidecar`` instead.

    Usage::

        capture = RouterCapture(top_k=4)
        handles = capture.register(model, gate_names)
        with torch.no_grad():
            model(input_ids)
        decisions = capture.decisions
        capture.clear()
        for h in handles:
            h.remove()
    """

    def __init__(self, top_k: int = 4):
        self.decisions: list[RouterDecision] = []
        self._top_k = top_k

    def clear(self) -> None:
        self.decisions.clear()

    def register(self, model: nn.Module, gate_names: list[str]) -> list[torch.utils.hooks.RemovableHook]:
        named = dict(model.named_modules())
        handles: list[torch.utils.hooks.RemovableHook] = []
        for idx, name in enumerate(gate_names):
            module = named.get(name)
            if module is None:
                raise KeyError(f"Gate module not found: {name}")
            handles.append(module.register_forward_hook(self._make_hook(idx)))
        return handles

    def _make_hook(self, layer_idx: int):
        top_k = self._top_k

        def hook(_module: nn.Module, _input: Any, output: Any) -> None:
            logits = output[0] if isinstance(output, tuple) else output
            if not isinstance(logits, torch.Tensor):
                return
            logits_cpu = logits.detach().cpu().float()
            weights, indices = torch.topk(
                torch.softmax(logits_cpu, dim=-1),
                k=min(top_k, logits_cpu.shape[-1]),
                dim=-1,
            )
            self.decisions.append(RouterDecision(
                layer_idx=layer_idx,
                selected_experts=indices,
                expert_weights=weights,
                gate_logits=logits_cpu,
                token_count=logits_cpu.shape[-2] if logits_cpu.ndim >= 2 else 1,
            ))
        return hook
