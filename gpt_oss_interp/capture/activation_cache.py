###############################################################################
#
# Activation capture with PyTorch hooks
#
###############################################################################

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


@dataclass
class ActivationRecord:
    layer_name: str
    tensor: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def tensor_shape(self) -> tuple[int, ...]:
        return tuple(self.tensor.shape)


class ActivationCache:
    """Collects hidden-state tensors via forward hooks.

    Usage::

        cache = ActivationCache()
        handles = cache.register(model, layer_names=["model.layers.0", ...])
        with torch.no_grad():
            model(input_ids)
        records = cache.records          # list[ActivationRecord]
        cache.clear()
        for h in handles:
            h.remove()
    """

    def __init__(self, detach: bool = True, to_cpu: bool = True):
        self.records: list[ActivationRecord] = []
        self._detach = detach
        self._to_cpu = to_cpu

    def clear(self) -> None:
        self.records.clear()

    def register(
        self,
        model: nn.Module,
        layer_names: list[str],
    ) -> list[torch.utils.hooks.RemovableHook]:
        """Register forward hooks on named modules and return hook handles."""
        named = dict(model.named_modules())
        handles: list[torch.utils.hooks.RemovableHook] = []
        for name in layer_names:
            module = named.get(name)
            if module is None:
                raise KeyError(f"Module not found: {name}")
            handle = module.register_forward_hook(self._make_hook(name))
            handles.append(handle)
        return handles

    def _make_hook(self, layer_name: str):
        def hook(_module: nn.Module, _input: Any, output: Any) -> None:
            tensor = output
            # Many modules return tuples; take the first element
            if isinstance(tensor, tuple):
                tensor = tensor[0]
            if not isinstance(tensor, torch.Tensor):
                return
            if self._detach:
                tensor = tensor.detach()
            if self._to_cpu:
                tensor = tensor.cpu()
            self.records.append(ActivationRecord(layer_name=layer_name, tensor=tensor))
        return hook

    def get(self, layer_name: str) -> list[ActivationRecord]:
        return [r for r in self.records if r.layer_name == layer_name]

    def last(self, layer_name: str) -> ActivationRecord | None:
        matches = self.get(layer_name)
        return matches[-1] if matches else None
