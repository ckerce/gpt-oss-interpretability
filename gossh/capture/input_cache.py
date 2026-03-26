"""InputCapture: pre-hook based capture of MoE layer inputs.

For MXFP4-quantized models, the residual stream state entering each MoE
block is accessible via ``register_forward_pre_hook`` even though the
router output is not.  InputCapture uses pre-hooks to record these hidden
states, which are then forwarded to the MoE sidecar for routing inference.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class InputCapture:
    """Capture residual-stream inputs to MoE/MLP modules via pre-hooks.

    Usage::

        capture = InputCapture()
        capture.register(model, structure)
        with torch.no_grad():
            model(input_ids)
        hidden_by_layer = capture.captured   # {layer_idx: Tensor[seq, hidden]}
        capture.clear()
        capture.remove_hooks()
    """

    def __init__(self):
        self._cache: dict[int, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHook] = []

    def register(self, model: nn.Module, mlp_names: list[str]) -> None:
        """Register pre-hooks on each MoE/MLP module in mlp_names.

        Args:
            model: The loaded model.
            mlp_names: List of fully-qualified module names (one per layer),
                typically ``structure.mlp_names``.  Empty strings are skipped.
        """
        named = dict(model.named_modules())
        for layer_idx, mlp_name in enumerate(mlp_names):
            if not mlp_name:
                continue
            module = named.get(mlp_name)
            if module is None:
                continue
            handle = module.register_forward_pre_hook(self._make_hook(layer_idx))
            self._handles.append(handle)

    def _make_hook(self, layer_idx: int):
        def hook(module: nn.Module, args: tuple) -> None:
            if args:
                hidden = args[0]
                if isinstance(hidden, torch.Tensor):
                    # Store on CPU to avoid holding GPU memory between forward passes
                    self._cache[layer_idx] = hidden.detach().cpu()
        return hook

    @property
    def captured(self) -> dict[int, torch.Tensor]:
        """Return a snapshot of captured hidden states by layer index."""
        return dict(self._cache)

    def clear(self) -> None:
        """Clear captured tensors (keep hooks registered)."""
        self._cache.clear()

    def remove_hooks(self) -> None:
        """Remove all registered pre-hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> "InputCapture":
        return self

    def __exit__(self, *args) -> None:
        self.remove_hooks()
