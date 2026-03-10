###############################################################################
#
# MoE router decision capture
#
###############################################################################

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


@dataclass
class RouterDecision:
    layer_idx: int
    selected_experts: torch.Tensor      # [batch, seq_len, top_k]
    expert_weights: torch.Tensor        # [batch, seq_len, top_k]
    gate_logits: torch.Tensor | None    # [batch, seq_len, num_experts]
    token_count: int

    @property
    def top_k(self) -> int:
        return self.selected_experts.shape[-1]


class RouterCapture:
    """Captures MoE routing decisions via forward hooks on gate modules.

    Usage::

        capture = RouterCapture()
        handles = capture.register(model, gate_names=["model.layers.0.mlp.gate", ...])
        with torch.no_grad():
            model(input_ids)
        decisions = capture.decisions   # list[RouterDecision]
        capture.clear()
    """

    def __init__(self, top_k: int = 4):
        self.decisions: list[RouterDecision] = []
        self._top_k = top_k

    def clear(self) -> None:
        self.decisions.clear()

    def register(
        self,
        model: nn.Module,
        gate_names: list[str],
    ) -> list[torch.utils.hooks.RemovableHook]:
        named = dict(model.named_modules())
        handles: list[torch.utils.hooks.RemovableHook] = []
        for idx, name in enumerate(gate_names):
            module = named.get(name)
            if module is None:
                raise KeyError(f"Gate module not found: {name}")
            handle = module.register_forward_hook(self._make_hook(idx))
            handles.append(handle)
        return handles

    def _make_hook(self, layer_idx: int):
        top_k = self._top_k

        def hook(_module: nn.Module, _input: Any, output: Any) -> None:
            # Gate modules typically output logits over experts
            logits = output
            if isinstance(logits, tuple):
                logits = logits[0]
            if not isinstance(logits, torch.Tensor):
                return

            logits_cpu = logits.detach().cpu()
            weights, indices = torch.topk(
                torch.softmax(logits_cpu, dim=-1),
                k=min(top_k, logits_cpu.shape[-1]),
                dim=-1,
            )
            token_count = logits_cpu.shape[-2] if logits_cpu.ndim >= 2 else 1
            self.decisions.append(RouterDecision(
                layer_idx=layer_idx,
                selected_experts=indices,
                expert_weights=weights,
                gate_logits=logits_cpu,
                token_count=token_count,
            ))
        return hook
