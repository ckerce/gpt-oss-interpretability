"""Hook factories for GOSSH interventions.

Each factory returns a ``torch`` forward hook closure suitable for
``module.register_forward_hook()``.  All hooks are pure functions of
their parameters with no global state — safe to register on multiple
modules simultaneously.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def head_mask_hook(head_indices: tuple[int, ...], scale: float, num_heads: int, head_dim: int):
    """Return a hook that scales specific attention head outputs.

    When ``scale=0.0`` the listed heads are zeroed (ablation).  When
    ``scale=1.0`` the hook is a no-op.

    The hook handles two output shapes:
    - Pre-projection: ``[batch, seq, num_heads * head_dim]`` — head slices
      are reshaped and scaled exactly.
    - Post-projection: ``[batch, seq, hidden_dim]`` where hidden_dim is not
      a clean multiple of num_heads * head_dim — proportional slices are
      scaled as an approximation.
    """
    def hook(module: nn.Module, input: Any, output: Any) -> Any:
        if isinstance(output, tuple):
            hidden, rest = output[0], output[1:]
        else:
            hidden, rest = output, None

        batch, seq_len, hidden_dim = hidden.shape
        if hidden_dim == num_heads * head_dim:
            h = hidden.view(batch, seq_len, num_heads, head_dim)
            for hi in head_indices:
                if hi < num_heads:
                    h[:, :, hi, :] = h[:, :, hi, :] * scale
            hidden = h.view(batch, seq_len, hidden_dim)
        else:
            slice_size = hidden_dim // num_heads
            for hi in head_indices:
                start = hi * slice_size
                end = start + slice_size
                if end <= hidden_dim:
                    hidden[:, :, start:end] = hidden[:, :, start:end] * scale

        return (hidden,) + rest if rest is not None else hidden
    return hook


def expert_output_scale_hook(expert_indices: tuple[int, ...], scale: float, num_experts: int):
    """Return a hook that scales the contribution of specific MoE experts.

    MXFP4-quantized models fuse the router and expert execution into a
    single kernel, bypassing Python-level router hooks.  This hook targets
    the MLP output instead and scales the fraction of output attributable
    to the targeted experts.

    For full ablation (``scale=0.0``) with top-k-of-N routing, the
    adjustment factor is ``(1 - k/N)`` where k is the number of targeted
    experts.  The blend preserves the contribution of untargeted experts.
    """
    def hook(module: nn.Module, input: Any, output: Any) -> Any:
        if isinstance(output, tuple):
            hidden, rest = output[0], output[1:]
        else:
            hidden, rest = output, None

        fraction = len(expert_indices) / num_experts
        adjusted = hidden * (1.0 - fraction + fraction * scale)

        return (adjusted,) + rest if rest is not None else adjusted
    return hook


def layer_scale_hook(scale: float, preserve_residual: bool = True):
    """Return a hook that scales the decoder-layer delta.

    With ``preserve_residual=True`` (default), the intervention is:

        output = residual_in + scale * (output - residual_in)

    This makes ``scale=0`` behave as an identity skip over the block rather
    than zeroing the entire residual stream — the semantically useful
    ablation for decoder layers with skip connections.

    With ``preserve_residual=False``:

        output = scale * output
    """
    def hook(module: nn.Module, input: Any, output: Any) -> Any:
        residual_in = input[0] if input else None

        if isinstance(output, tuple):
            hidden, rest = output[0], output[1:]
        else:
            hidden, rest = output, None

        if preserve_residual and residual_in is not None and isinstance(hidden, torch.Tensor):
            adjusted = residual_in + scale * (hidden - residual_in)
        else:
            adjusted = hidden * scale

        return (adjusted,) + rest if rest is not None else adjusted
    return hook


def temperature_hook(scale: float):
    """Return a hook that scales attention module output (pre-softmax temperature proxy).

    Hooks the attention module output directly.  For attention modules that
    return ``(hidden_states, ...)`` tuples this scales the first element;
    for modules that return a plain tensor it scales the tensor.
    """
    def hook(module: nn.Module, input: Any, output: Any) -> Any:
        if isinstance(output, tuple):
            return (output[0] * scale,) + output[1:]
        return output * scale
    return hook
