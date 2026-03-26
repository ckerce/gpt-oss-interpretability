"""Per-expert output capture for MoE models.

Hooks individual expert submodules within MoE layers and captures their
output tensors before gating and summation.

**Availability constraint**: requires Python-visible expert modules.
Under MXFP4 quantization, expert dispatch is fused into a single native
kernel — individual expert outputs are not observable via Python hooks.
In that case, use ``MoeSidecar`` for routing information and fall back to
block-level ``ActivationCache`` for the combined MoE output.

Typical usage (non-quantized checkpoint)::

    names = ExpertCapture.discover_expert_names(model, structure)
    if not names:
        print("No hookable expert modules found (MXFP4?). Using sidecar.")
    else:
        with ExpertCapture() as cap:
            cap.register(model, names)
            with torch.no_grad():
                model(input_ids)
            profiles = cap.captured   # {layer_idx: {expert_idx: [Tensor, ...]}}
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class ExpertCapture:
    """Captures per-expert output tensors within MoE layers.

    Args:
        detach: Detach tensors from the computation graph.
        to_cpu: Move tensors to CPU after capture.
    """

    def __init__(self, detach: bool = True, to_cpu: bool = True):
        self._detach = detach
        self._to_cpu = to_cpu
        self._captured: dict[int, dict[int, list[torch.Tensor]]] = {}
        self._handles: list[torch.utils.hooks.RemovableHook] = []

    # ── Discovery ──────────────────────────────────────────────────────────────

    @staticmethod
    def discover_expert_names(
        model: nn.Module,
        structure: Any,
    ) -> dict[int, dict[int, str]]:
        """Walk each MoE/MLP module and find hookable expert submodule names.

        Handles two common patterns:
        - ``mlp.experts`` is an ``nn.ModuleList`` (Mixtral-style)
        - ``mlp.expert_0``, ``mlp.expert_1``, … are direct numeric children

        Returns:
            ``{layer_idx: {expert_idx: fully_qualified_module_name}}``
            Empty dict if no expert modules are found (MXFP4 fused model).
        """
        named = dict(model.named_modules())
        expert_names: dict[int, dict[int, str]] = {}

        for layer_idx, mlp_name in enumerate(structure.mlp_names):
            if not mlp_name:
                continue
            mlp_mod = named.get(mlp_name)
            if mlp_mod is None:
                continue

            layer_experts: dict[int, str] = {}

            for child_name, child_mod in mlp_mod.named_children():
                # Pattern 1: experts is a ModuleList
                if child_name == "experts" and isinstance(child_mod, nn.ModuleList):
                    for i in range(len(child_mod)):
                        layer_experts[i] = f"{mlp_name}.experts.{i}"
                    break

                # Pattern 2: numeric child names (e.g. "0", "1", "expert_0")
                raw = child_name.removeprefix("expert_").removeprefix("expert")
                try:
                    expert_idx = int(raw)
                    layer_experts[expert_idx] = f"{mlp_name}.{child_name}"
                except ValueError:
                    pass

            if layer_experts:
                expert_names[layer_idx] = layer_experts

        return expert_names

    # ── Registration ───────────────────────────────────────────────────────────

    def register(
        self,
        model: nn.Module,
        expert_names: dict[int, dict[int, str]],
    ) -> None:
        """Register forward hooks on named expert modules.

        Args:
            model: The loaded model.
            expert_names: Output of ``discover_expert_names``.
        """
        named = dict(model.named_modules())
        for layer_idx, experts in expert_names.items():
            if layer_idx not in self._captured:
                self._captured[layer_idx] = {}
            for expert_idx, module_name in experts.items():
                mod = named.get(module_name)
                if mod is None:
                    continue
                if expert_idx not in self._captured[layer_idx]:
                    self._captured[layer_idx][expert_idx] = []
                handle = mod.register_forward_hook(
                    self._make_hook(layer_idx, expert_idx)
                )
                self._handles.append(handle)

    def _make_hook(self, layer_idx: int, expert_idx: int):
        def hook(_module: nn.Module, _input: Any, output: Any) -> None:
            tensor = output
            if isinstance(tensor, tuple):
                tensor = tensor[0]
            if not isinstance(tensor, torch.Tensor):
                return
            if self._detach:
                tensor = tensor.detach()
            if self._to_cpu:
                tensor = tensor.cpu()
            self._captured[layer_idx][expert_idx].append(tensor)
        return hook

    # ── Data access ───────────────────────────────────────────────────────────

    @property
    def captured(self) -> dict[int, dict[int, list[torch.Tensor]]]:
        """``{layer_idx: {expert_idx: [output_tensor, ...]}}``."""
        return self._captured

    def clear(self) -> None:
        """Clear captured tensors without removing hooks."""
        for layer in self._captured.values():
            for outputs in layer.values():
                outputs.clear()

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def activation_count(self, layer_idx: int, expert_idx: int) -> int:
        """Return the number of activations captured for a given expert."""
        try:
            return len(self._captured[layer_idx][expert_idx])
        except KeyError:
            return 0

    def total_tokens_routed(self) -> dict[int, dict[int, int]]:
        """Return total tokens routed to each expert, per layer.

        Each captured tensor has shape ``[tokens_routed, hidden_dim]``;
        this method sums ``tensor.shape[0]`` across all captures.
        """
        counts: dict[int, dict[int, int]] = {}
        for layer_idx, experts in self._captured.items():
            counts[layer_idx] = {}
            for expert_idx, outputs in experts.items():
                counts[layer_idx][expert_idx] = sum(
                    t.shape[0] for t in outputs if t.ndim >= 1
                )
        return counts

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "ExpertCapture":
        return self

    def __exit__(self, *args: Any) -> None:
        self.remove_hooks()
