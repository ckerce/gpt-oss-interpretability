"""Router weight extraction and the RouterSidecarModel.

The MoE gate/router weights remain bf16 even in MXFP4-quantized checkpoints
because routing is a control-flow decision that cannot tolerate low precision.
RouterWeightExtractor walks model.named_parameters() to locate these weights;
RouterSidecarModel wraps them as nn.Linear layers for CPU inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from gossh.capture.router_capture import RouterDecision

if TYPE_CHECKING:
    from gossh.backends.structure import ModelStructure


class RouterWeightExtractor:
    """Extract bf16 gate/router weights from a (possibly quantized) model.

    The extractor matches gate module names discovered by ``ModelStructure``
    against ``model.named_parameters()`` to retrieve the weight tensors
    without triggering dequantization of the expert weights.
    """

    def extract(
        self,
        model: nn.Module,
        structure: "ModelStructure",
    ) -> dict[int, torch.Tensor]:
        """Return ``{layer_idx: weight_tensor}`` for each discovered gate.

        The returned tensors are float32 copies on CPU.
        """
        named_params = dict(model.named_parameters())
        weights: dict[int, torch.Tensor] = {}

        for layer_idx, gate_name in enumerate(structure.gate_names):
            if not gate_name:
                continue
            # Try <gate_name>.weight first, then the gate_name itself
            weight = None
            for candidate in (f"{gate_name}.weight", gate_name):
                if candidate in named_params:
                    weight = named_params[candidate].detach().cpu().to(torch.float32)
                    break
            if weight is not None:
                weights[layer_idx] = weight

        return weights


class RouterSidecarModel(nn.Module):
    """Minimal model: one nn.Linear per layer for the gate projection.

    Each linear projects hidden_dim → num_experts (no bias), matching the
    gate module in the original model.  Weights are loaded from the dict
    returned by ``RouterWeightExtractor.extract()``.
    """

    def __init__(self, router_weights: dict[int, torch.Tensor], top_k: int):
        super().__init__()
        self.top_k = top_k
        self.gates = nn.ModuleDict()

        for layer_idx, weight in router_weights.items():
            # weight: [num_experts, hidden_dim]
            num_experts, hidden_dim = weight.shape
            gate = nn.Linear(hidden_dim, num_experts, bias=False)
            gate.weight = nn.Parameter(weight, requires_grad=False)
            self.gates[str(layer_idx)] = gate

        self.eval()

    @property
    def layer_indices(self) -> list[int]:
        return sorted(int(k) for k in self.gates._modules.keys())

    def route(self, layer_idx: int, hidden: torch.Tensor) -> RouterDecision:
        """Route one layer. ``hidden`` is ``[seq_len, hidden_dim]`` float32."""
        key = str(layer_idx)
        if key not in self.gates:
            raise KeyError(f"No router for layer {layer_idx}")
        gate = self.gates[key]

        with torch.no_grad():
            logits = gate(hidden)                               # [seq, num_experts]
            probs = F.softmax(logits, dim=-1)
            k = min(self.top_k, logits.shape[-1])
            weights, indices = torch.topk(probs, k=k, dim=-1)  # [seq, top_k]

        return RouterDecision(
            layer_idx=layer_idx,
            selected_experts=indices.cpu(),
            expert_weights=weights.cpu(),
            gate_logits=logits.cpu(),
            token_count=hidden.shape[0],
        )

    def route_all(
        self,
        layer_hidden: dict[int, torch.Tensor],
    ) -> list[RouterDecision]:
        """Route all layers in ``layer_hidden``, returned in layer order."""
        return [
            self.route(layer_idx, hidden)
            for layer_idx, hidden in sorted(layer_hidden.items())
            if str(layer_idx) in self.gates._modules
        ]
