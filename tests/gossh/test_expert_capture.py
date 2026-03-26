"""Tests for ExpertCapture — per-expert output capture for MoE models."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from gossh.capture.expert_capture import ExpertCapture


###############################################################################
# Minimal mock MoE model
###############################################################################

HIDDEN_DIM = 32
NUM_EXPERTS = 4
SEQ_LEN = 8


class FakeExpert(nn.Linear):
    """A single expert FFN."""
    pass


class FakeMoE(nn.Module):
    """MoE layer with experts as a ModuleList (Mixtral-style)."""
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(HIDDEN_DIM, NUM_EXPERTS, bias=False)
        self.experts = nn.ModuleList([
            FakeExpert(HIDDEN_DIM, HIDDEN_DIM, bias=False)
            for _ in range(NUM_EXPERTS)
        ])

    def forward(self, x):
        # Simplified: route top-2, sum outputs (no gating weights)
        logits = self.gate(x)
        _, top_idx = logits.topk(2, dim=-1)
        out = torch.zeros_like(x)
        for t in range(x.shape[0]):
            for k in range(2):
                eidx = top_idx[t, k].item()
                out[t] += self.experts[eidx](x[t:t+1])[0]
        return out


class FakeMoENumericChildren(nn.Module):
    """MoE layer with experts as numerically-named direct children."""
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(HIDDEN_DIM, NUM_EXPERTS, bias=False)
        for i in range(NUM_EXPERTS):
            setattr(self, str(i), FakeExpert(HIDDEN_DIM, HIDDEN_DIM, bias=False))

    def forward(self, x):
        out = getattr(self, "0")(x)
        return out


class FakeBlock(nn.Module):
    def __init__(self, moe):
        super().__init__()
        self.mlp = moe

    def forward(self, x):
        return self.mlp(x)


class FakeModel(nn.Module):
    def __init__(self, moe_cls=FakeMoE):
        super().__init__()
        self.layers = nn.ModuleList([FakeBlock(moe_cls()) for _ in range(2)])

    def forward(self, x):
        for block in self.layers:
            x = block(x)
        return x


class FakeStructure:
    def __init__(self, n_layers=2):
        self.mlp_names = [f"layers.{i}.mlp" for i in range(n_layers)]


###############################################################################
# Discovery tests
###############################################################################

class TestDiscoverExpertNames:
    def test_modulelist_pattern(self):
        model = FakeModel(FakeMoE)
        structure = FakeStructure()
        names = ExpertCapture.discover_expert_names(model, structure)

        assert set(names.keys()) == {0, 1}
        for layer_idx in names:
            assert set(names[layer_idx].keys()) == set(range(NUM_EXPERTS))
            for ei, name in names[layer_idx].items():
                assert f".experts.{ei}" in name

    def test_numeric_children_pattern(self):
        model = FakeModel(FakeMoENumericChildren)
        structure = FakeStructure()
        names = ExpertCapture.discover_expert_names(model, structure)

        assert set(names.keys()) == {0, 1}
        for layer_idx in names:
            assert set(names[layer_idx].keys()) == set(range(NUM_EXPERTS))

    def test_empty_mlp_names_returns_empty(self):
        model = FakeModel()

        class EmptyStructure:
            mlp_names = ["", ""]

        names = ExpertCapture.discover_expert_names(model, EmptyStructure())
        assert names == {}

    def test_missing_module_skipped(self):
        model = FakeModel()

        class BadStructure:
            mlp_names = ["does_not_exist", "layers.0.mlp"]

        names = ExpertCapture.discover_expert_names(model, BadStructure())
        # Only layer 1 (index 1, name "layers.0.mlp") should appear
        assert 0 not in names
        assert 1 in names


###############################################################################
# Registration and capture tests
###############################################################################

class TestExpertCaptureRegistration:
    def test_register_hooks_on_experts(self):
        model = FakeModel(FakeMoE)
        structure = FakeStructure()
        names = ExpertCapture.discover_expert_names(model, structure)

        cap = ExpertCapture()
        cap.register(model, names)

        x = torch.randn(SEQ_LEN, HIDDEN_DIM)
        model(x)

        # Should have captured something for each layer
        for layer_idx in names:
            assert layer_idx in cap.captured
            # At least some experts should have fired
            total = sum(len(v) for v in cap.captured[layer_idx].values())
            assert total > 0, f"No captures at layer {layer_idx}"

        cap.remove_hooks()

    def test_captured_tensor_shape(self):
        model = FakeModel(FakeMoE)
        structure = FakeStructure()
        names = ExpertCapture.discover_expert_names(model, structure)

        cap = ExpertCapture(to_cpu=True)
        cap.register(model, names)
        x = torch.randn(SEQ_LEN, HIDDEN_DIM)
        model(x)

        for layer_idx, experts in cap.captured.items():
            for expert_idx, outputs in experts.items():
                for t in outputs:
                    # Expert outputs are 2D: [tokens_routed, hidden_dim]
                    assert t.ndim == 2
                    assert t.shape[-1] == HIDDEN_DIM

        cap.remove_hooks()

    def test_context_manager_removes_hooks(self):
        model = FakeModel(FakeMoE)
        structure = FakeStructure()
        names = ExpertCapture.discover_expert_names(model, structure)

        with ExpertCapture() as cap:
            cap.register(model, names)
            x = torch.randn(SEQ_LEN, HIDDEN_DIM)
            model(x)

        # After exit, hooks are removed — a second forward should not add captures
        count_after = {
            li: {ei: len(v) for ei, v in experts.items()}
            for li, experts in cap.captured.items()
        }
        model(x)
        count_after2 = {
            li: {ei: len(v) for ei, v in experts.items()}
            for li, experts in cap.captured.items()
        }
        assert count_after == count_after2

    def test_clear_empties_tensors(self):
        model = FakeModel(FakeMoE)
        structure = FakeStructure()
        names = ExpertCapture.discover_expert_names(model, structure)

        cap = ExpertCapture()
        cap.register(model, names)
        x = torch.randn(SEQ_LEN, HIDDEN_DIM)
        model(x)

        cap.clear()
        for layer_idx, experts in cap.captured.items():
            for expert_idx, outputs in experts.items():
                assert outputs == []

        cap.remove_hooks()

    def test_detach_is_applied(self):
        model = FakeModel(FakeMoE)
        structure = FakeStructure()
        names = ExpertCapture.discover_expert_names(model, structure)

        cap = ExpertCapture(detach=True)
        cap.register(model, names)
        x = torch.randn(SEQ_LEN, HIDDEN_DIM)
        model(x)

        for experts in cap.captured.values():
            for outputs in experts.values():
                for t in outputs:
                    assert not t.requires_grad

        cap.remove_hooks()


###############################################################################
# Introspection tests
###############################################################################

class TestExpertCaptureIntrospection:
    def test_activation_count(self):
        model = FakeModel(FakeMoE)
        structure = FakeStructure()
        names = ExpertCapture.discover_expert_names(model, structure)

        with ExpertCapture() as cap:
            cap.register(model, names)
            model(torch.randn(SEQ_LEN, HIDDEN_DIM))
            # Count should be >= 0 for any registered expert
            for li in names:
                for ei in names[li]:
                    assert cap.activation_count(li, ei) >= 0

        # Unknown layer/expert returns 0 without error
        assert cap.activation_count(999, 999) == 0

    def test_total_tokens_routed(self):
        model = FakeModel(FakeMoE)
        structure = FakeStructure()
        names = ExpertCapture.discover_expert_names(model, structure)

        with ExpertCapture() as cap:
            cap.register(model, names)
            model(torch.randn(SEQ_LEN, HIDDEN_DIM))
            counts = cap.total_tokens_routed()

        # Each layer should have token counts for all experts
        for layer_idx in names:
            assert layer_idx in counts
            for expert_idx in names[layer_idx]:
                assert expert_idx in counts[layer_idx]
                assert counts[layer_idx][expert_idx] >= 0
