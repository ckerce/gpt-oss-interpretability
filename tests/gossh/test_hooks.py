"""Tests for gossh.interventions.hooks — pure hook factory functions."""

import pytest
import torch
import torch.nn as nn

from gossh.interventions.hooks import (
    expert_output_scale_hook,
    head_mask_hook,
    layer_scale_hook,
    temperature_hook,
)


def _dummy_module():
    return nn.Linear(4, 4)


def _call_hook(hook_fn, output, input_tensor=None):
    """Invoke a hook closure directly (simulating PyTorch forward hook call)."""
    module = _dummy_module()
    inp = (input_tensor,) if input_tensor is not None else (torch.zeros(1, 1, 4),)
    return hook_fn(module, inp, output)


class TestHeadMaskHook:
    def test_scale_zero_zeros_target_head(self):
        # [batch=1, seq=3, num_heads*head_dim = 4*4 = 16]
        num_heads, head_dim = 4, 4
        hidden = torch.ones(1, 3, num_heads * head_dim)
        hook = head_mask_hook((0,), scale=0.0, num_heads=num_heads, head_dim=head_dim)
        result = _call_hook(hook, hidden)
        # Head 0 should be zeroed
        assert result[0, :, :head_dim].sum() == 0.0
        # Other heads untouched
        assert result[0, :, head_dim:].sum() > 0.0

    def test_scale_one_is_identity(self):
        num_heads, head_dim = 4, 4
        hidden = torch.randn(1, 5, num_heads * head_dim)
        hook = head_mask_hook((0, 1, 2, 3), scale=1.0, num_heads=num_heads, head_dim=head_dim)
        result = _call_hook(hook, hidden)
        assert torch.allclose(result, hidden)

    def test_tuple_output_preserved(self):
        num_heads, head_dim = 2, 4
        hidden = torch.ones(1, 3, num_heads * head_dim)
        extra = torch.tensor([99.0])
        hook = head_mask_hook((0,), scale=0.0, num_heads=num_heads, head_dim=head_dim)
        result = _call_hook(hook, (hidden, extra))
        assert isinstance(result, tuple)
        assert torch.equal(result[1], extra)

    def test_out_of_range_head_index_ignored(self):
        num_heads, head_dim = 2, 4
        hidden = torch.ones(1, 3, num_heads * head_dim)
        hook = head_mask_hook((99,), scale=0.0, num_heads=num_heads, head_dim=head_dim)
        result = _call_hook(hook, hidden)
        # No valid head index — output unchanged
        assert torch.allclose(result, hidden)


class TestExpertOutputScaleHook:
    def test_full_ablation_reduces_output(self):
        num_experts = 8
        hidden = torch.ones(1, 4, 16)
        hook = expert_output_scale_hook((0, 1, 2, 3), scale=0.0, num_experts=num_experts)
        result = _call_hook(hook, hidden)
        # Fraction targeted = 4/8 = 0.5; adjustment = 1.0 - 0.5 + 0.5*0 = 0.5
        assert torch.allclose(result, hidden * 0.5)

    def test_scale_one_is_identity(self):
        hidden = torch.randn(1, 4, 16)
        hook = expert_output_scale_hook((0,), scale=1.0, num_experts=8)
        result = _call_hook(hook, hidden)
        assert torch.allclose(result, hidden)

    def test_tuple_output_preserved(self):
        hidden = torch.ones(1, 4, 16)
        extra = torch.tensor([42.0])
        hook = expert_output_scale_hook((0,), scale=0.0, num_experts=8)
        result = _call_hook(hook, (hidden, extra))
        assert isinstance(result, tuple)
        assert torch.equal(result[1], extra)


class TestLayerScaleHook:
    def test_scale_zero_preserve_residual(self):
        # With preserve_residual=True, scale=0 → output = residual_in
        residual = torch.ones(1, 3, 8) * 2.0
        block_output = torch.ones(1, 3, 8) * 5.0
        hook = layer_scale_hook(scale=0.0, preserve_residual=True)
        module = _dummy_module()
        result = hook(module, (residual,), block_output)
        assert torch.allclose(result, residual)

    def test_scale_one_preserve_residual_is_identity(self):
        residual = torch.randn(1, 3, 8)
        block_output = torch.randn(1, 3, 8)
        hook = layer_scale_hook(scale=1.0, preserve_residual=True)
        module = _dummy_module()
        result = hook(module, (residual,), block_output)
        assert torch.allclose(result, block_output)

    def test_scale_zero_no_preserve_zeros_output(self):
        residual = torch.ones(1, 3, 8)
        block_output = torch.ones(1, 3, 8) * 5.0
        hook = layer_scale_hook(scale=0.0, preserve_residual=False)
        module = _dummy_module()
        result = hook(module, (residual,), block_output)
        assert torch.allclose(result, torch.zeros_like(block_output))

    def test_tuple_output_preserved(self):
        residual = torch.ones(1, 3, 8)
        block_output = torch.ones(1, 3, 8) * 3.0
        extra = torch.tensor([7.0])
        hook = layer_scale_hook(scale=0.5, preserve_residual=True)
        module = _dummy_module()
        result = hook(module, (residual,), (block_output, extra))
        assert isinstance(result, tuple)
        assert torch.equal(result[1], extra)


class TestTemperatureHook:
    def test_scale_applied_to_tensor_output(self):
        output = torch.ones(1, 4, 8) * 3.0
        hook = temperature_hook(scale=2.0)
        result = _call_hook(hook, output)
        assert torch.allclose(result, output * 2.0)

    def test_scale_zero_zeros_output(self):
        output = torch.randn(1, 4, 8)
        hook = temperature_hook(scale=0.0)
        result = _call_hook(hook, output)
        assert torch.allclose(result, torch.zeros_like(output))

    def test_tuple_output_scales_first_element(self):
        hidden = torch.ones(1, 4, 8) * 2.0
        extra = torch.tensor([99.0])
        hook = temperature_hook(scale=3.0)
        result = _call_hook(hook, (hidden, extra))
        assert isinstance(result, tuple)
        assert torch.allclose(result[0], hidden * 3.0)
        assert torch.equal(result[1], extra)
