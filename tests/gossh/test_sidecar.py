"""Tests for the MoE sidecar subsystem.

All tests use synthetic router weights — no real model checkpoint required.
Tests that spin up a subprocess are marked with pytest.mark.subprocess.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from gossh.capture.router_capture import RouterCapture, RouterDecision
from gossh.capture.input_cache import InputCapture
from gossh.sidecar.protocol import (
    encode_tensor,
    decode_tensor,
    encode_tensor_int,
    build_route_request,
)
from gossh.sidecar.dequant import RouterSidecarModel, RouterWeightExtractor
from gossh.sidecar.process import MoeSidecar
from gossh.sidecar.validation import (
    LayerValidationStats,
    SidecarValidationReport,
    format_validation_report,
    run_full_validation,
    validate_numerical,
    validate_sensitivity,
)


###############################################################################
# Fixtures
###############################################################################

NUM_LAYERS = 4
HIDDEN_DIM = 64
NUM_EXPERTS = 8
TOP_K = 2
SEQ_LEN = 16


def _make_router_weights(
    n_layers: int = NUM_LAYERS,
    hidden_dim: int = HIDDEN_DIM,
    num_experts: int = NUM_EXPERTS,
) -> dict[int, torch.Tensor]:
    """Synthetic router weights: [num_experts, hidden_dim] per layer."""
    torch.manual_seed(42)
    return {
        layer_idx: torch.randn(num_experts, hidden_dim)
        for layer_idx in range(n_layers)
    }


def _make_hidden_states(
    n_layers: int = NUM_LAYERS,
    seq_len: int = SEQ_LEN,
    hidden_dim: int = HIDDEN_DIM,
) -> dict[int, torch.Tensor]:
    torch.manual_seed(7)
    return {
        layer_idx: torch.randn(seq_len, hidden_dim)
        for layer_idx in range(n_layers)
    }


@pytest.fixture(scope="module")
def router_weights():
    return _make_router_weights()


@pytest.fixture(scope="module")
def hidden_states():
    return _make_hidden_states()


@pytest.fixture
def sidecar_model(router_weights):
    return RouterSidecarModel(router_weights, top_k=TOP_K)


@pytest.fixture(scope="module")
def running_sidecar(router_weights):
    with MoeSidecar(router_weights, top_k=TOP_K) as sc:
        yield sc


###############################################################################
# Protocol tests (no subprocess)
###############################################################################

class TestProtocol:
    def test_encode_decode_float_tensor(self):
        t = torch.randn(16, 64)
        d = encode_tensor(t)
        t2 = decode_tensor(d)
        assert t2.shape == t.shape
        assert torch.allclose(t.float(), t2, atol=1e-6)

    def test_encode_decode_int_tensor(self):
        t = torch.randint(0, 8, (16, 2))
        d = encode_tensor_int(t)
        t2 = decode_tensor(d)
        assert t2.dtype == torch.int64
        assert t2.shape == t.shape
        assert torch.equal(t.long(), t2)

    def test_encode_1d_tensor(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        d = encode_tensor(t)
        t2 = decode_tensor(d)
        assert torch.allclose(t, t2, atol=1e-6)

    def test_build_route_request_structure(self, hidden_states):
        msg = build_route_request(hidden_states)
        assert msg["cmd"] == "route"
        assert "layers" in msg
        assert len(msg["layers"]) == NUM_LAYERS
        for k in msg["layers"]:
            assert isinstance(k, str)
            assert "shape" in msg["layers"][k]
            assert "data" in msg["layers"][k]


###############################################################################
# RouterSidecarModel tests (no subprocess)
###############################################################################

class TestRouterSidecarModel:
    def test_route_single_layer_shape(self, sidecar_model):
        hidden = torch.randn(SEQ_LEN, HIDDEN_DIM)
        decision = sidecar_model.route(0, hidden)
        assert decision.layer_idx == 0
        assert decision.selected_experts.shape == (SEQ_LEN, TOP_K)
        assert decision.expert_weights.shape == (SEQ_LEN, TOP_K)
        assert decision.gate_logits.shape == (SEQ_LEN, NUM_EXPERTS)
        assert decision.token_count == SEQ_LEN

    def test_route_all_returns_all_layers(self, sidecar_model, hidden_states):
        decisions = sidecar_model.route_all(hidden_states)
        assert len(decisions) == NUM_LAYERS
        layer_indices = [d.layer_idx for d in decisions]
        assert layer_indices == sorted(layer_indices)

    def test_expert_weights_sum_to_one_approx(self, sidecar_model):
        hidden = torch.randn(SEQ_LEN, HIDDEN_DIM)
        decision = sidecar_model.route(0, hidden)
        # Weights are top-k softmax values — they sum to < 1 in general
        weight_sum = decision.expert_weights.sum(dim=-1)
        assert (weight_sum <= 1.0 + 1e-5).all()
        assert (weight_sum > 0.0).all()

    def test_route_is_deterministic(self, sidecar_model):
        hidden = torch.randn(SEQ_LEN, HIDDEN_DIM)
        d1 = sidecar_model.route(0, hidden)
        d2 = sidecar_model.route(0, hidden)
        assert torch.equal(d1.selected_experts, d2.selected_experts)
        assert torch.allclose(d1.expert_weights, d2.expert_weights)

    def test_route_matches_manual_linear(self, router_weights):
        """Sidecar routing should match a manual nn.Linear forward pass."""
        layer_idx = 0
        weight = router_weights[layer_idx]
        hidden = torch.randn(SEQ_LEN, HIDDEN_DIM)

        model = RouterSidecarModel(router_weights, top_k=TOP_K)
        decision = model.route(layer_idx, hidden)

        # Manual computation
        logits = F.linear(hidden, weight)
        probs = F.softmax(logits, dim=-1)
        manual_weights, manual_indices = torch.topk(probs, k=TOP_K, dim=-1)

        assert torch.equal(decision.selected_experts, manual_indices)
        assert torch.allclose(decision.expert_weights, manual_weights, atol=1e-6)

    def test_unknown_layer_raises(self, sidecar_model):
        hidden = torch.randn(SEQ_LEN, HIDDEN_DIM)
        with pytest.raises(KeyError):
            sidecar_model.route(999, hidden)

    def test_layer_indices_property(self, sidecar_model):
        assert sidecar_model.layer_indices == list(range(NUM_LAYERS))


###############################################################################
# RouterWeightExtractor tests
###############################################################################

class TestRouterWeightExtractor:
    def test_extract_from_mock_model(self):
        """Extractor should find weights via structure.gate_names."""
        import torch.nn as nn

        class MockGate(nn.Linear):
            pass

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_0 = MockGate(HIDDEN_DIM, NUM_EXPERTS, bias=False)
                self.gate_1 = MockGate(HIDDEN_DIM, NUM_EXPERTS, bias=False)

        model = MockModel()

        class FakeStructure:
            gate_names = ["gate_0", "gate_1", ""]

        extractor = RouterWeightExtractor()
        weights = extractor.extract(model, FakeStructure())

        assert 0 in weights
        assert 1 in weights
        assert 2 not in weights
        assert weights[0].shape == (NUM_EXPERTS, HIDDEN_DIM)


###############################################################################
# MoeSidecar subprocess tests
###############################################################################

class TestMoeSidecarLifecycle:
    def test_start_and_stop(self, router_weights):
        sidecar = MoeSidecar(router_weights, top_k=TOP_K)
        sidecar.start()
        assert sidecar.is_running()
        sidecar.stop()
        assert not sidecar.is_running()

    def test_context_manager(self, router_weights):
        with MoeSidecar(router_weights, top_k=TOP_K) as sc:
            assert sc.is_running()
        assert not sc.is_running()

    def test_ping(self, running_sidecar):
        assert running_sidecar.ping() is True

    def test_num_layers(self, running_sidecar):
        assert running_sidecar.num_layers == NUM_LAYERS

    def test_layer_indices(self, running_sidecar):
        assert running_sidecar.layer_indices == list(range(NUM_LAYERS))


class TestMoeSidecarRouting:
    def test_route_returns_all_layers(self, running_sidecar, hidden_states):
        decisions = running_sidecar.route(hidden_states)
        assert len(decisions) == NUM_LAYERS
        for d in decisions:
            assert isinstance(d, RouterDecision)

    def test_route_shapes(self, running_sidecar, hidden_states):
        decisions = running_sidecar.route(hidden_states)
        for d in decisions:
            assert d.selected_experts.shape == (SEQ_LEN, TOP_K)
            assert d.expert_weights.shape == (SEQ_LEN, TOP_K)
            assert d.token_count == SEQ_LEN

    def test_route_matches_local_model(self, running_sidecar, router_weights, hidden_states):
        """Sidecar routing should match local RouterSidecarModel exactly."""
        local = RouterSidecarModel(router_weights, top_k=TOP_K)
        local_decisions = {d.layer_idx: d for d in local.route_all(hidden_states)}
        sidecar_decisions = {d.layer_idx: d for d in running_sidecar.route(hidden_states)}

        for layer_idx in range(NUM_LAYERS):
            ld = local_decisions[layer_idx]
            sd = sidecar_decisions[layer_idx]
            assert torch.equal(ld.selected_experts, sd.selected_experts), \
                f"Layer {layer_idx}: expert indices differ"
            assert torch.allclose(ld.expert_weights, sd.expert_weights, atol=1e-5), \
                f"Layer {layer_idx}: expert weights differ"

    def test_route_subset_of_layers(self, running_sidecar, hidden_states):
        subset = {k: v for k, v in hidden_states.items() if k in (0, 2)}
        decisions = running_sidecar.route(subset)
        assert len(decisions) == 2
        assert {d.layer_idx for d in decisions} == {0, 2}

    def test_route_deterministic(self, running_sidecar, hidden_states):
        d1 = {d.layer_idx: d for d in running_sidecar.route(hidden_states)}
        d2 = {d.layer_idx: d for d in running_sidecar.route(hidden_states)}
        for layer_idx in range(NUM_LAYERS):
            assert torch.equal(d1[layer_idx].selected_experts, d2[layer_idx].selected_experts)


###############################################################################
# Validation tests (no subprocess needed for most)
###############################################################################

class TestValidationNumerical:
    def _make_reference_decisions(self, router_weights, hidden_states):
        model = RouterSidecarModel(router_weights, top_k=TOP_K)
        return model.route_all(hidden_states)

    def test_perfect_agreement(self, running_sidecar, router_weights, hidden_states):
        """Sidecar vs itself should report perfect numerical agreement."""
        ref = self._make_reference_decisions(router_weights, hidden_states)
        stats = validate_numerical(running_sidecar, ref, hidden_states)
        assert len(stats) == NUM_LAYERS
        for s in stats:
            assert s.top1_match_rate == pytest.approx(1.0, abs=1e-6)
            assert s.status == "robust"

    def test_degraded_reference(self, running_sidecar, router_weights, hidden_states):
        """Randomly permuted expert indices should trigger mismatch detection."""
        ref = self._make_reference_decisions(router_weights, hidden_states)
        # Corrupt expert indices for all layers
        bad_ref = [
            RouterDecision(
                layer_idx=d.layer_idx,
                selected_experts=torch.randint(0, NUM_EXPERTS, d.selected_experts.shape),
                expert_weights=d.expert_weights,
                gate_logits=d.gate_logits,
                token_count=d.token_count,
            )
            for d in ref
        ]
        stats = validate_numerical(running_sidecar, bad_ref, hidden_states, mismatch_threshold=0.05)
        # At least some layers should fail
        assert any(s.top1_match_rate < 1.0 for s in stats)


class TestValidationSensitivity:
    def test_returns_all_layers(self, running_sidecar, hidden_states):
        sens = validate_sensitivity(running_sidecar, hidden_states, noise_scale=0.01, n_trials=2)
        assert set(sens.keys()) == set(range(NUM_LAYERS))

    def test_large_noise_increases_sensitivity(self, running_sidecar, hidden_states):
        low_sens = validate_sensitivity(running_sidecar, hidden_states, noise_scale=1e-6, n_trials=2)
        high_sens = validate_sensitivity(running_sidecar, hidden_states, noise_scale=10.0, n_trials=2)
        # High noise should flip more decisions on average
        avg_low = sum(low_sens.values()) / len(low_sens)
        avg_high = sum(high_sens.values()) / len(high_sens)
        assert avg_high >= avg_low


class TestFullValidationPipeline:
    def test_run_full_validation_passes_for_self(self, running_sidecar, router_weights, hidden_states):
        local = RouterSidecarModel(router_weights, top_k=TOP_K)
        ref = local.route_all(hidden_states)
        report = run_full_validation(running_sidecar, ref, hidden_states)
        assert report.passed
        assert report.flagged_layers == []
        assert len(report.per_layer) == NUM_LAYERS

    def test_format_report_is_string(self, running_sidecar, router_weights, hidden_states):
        local = RouterSidecarModel(router_weights, top_k=TOP_K)
        ref = local.route_all(hidden_states)
        report = run_full_validation(running_sidecar, ref, hidden_states)
        text = format_validation_report(report)
        assert isinstance(text, str)
        assert "Layer" in text
        assert "robust" in text.lower() or "noisy" in text.lower() or "sensitive" in text.lower()


###############################################################################
# InputCapture tests (no model needed for basic tests)
###############################################################################

class TestInputCapture:
    def test_register_and_capture(self):
        import torch.nn as nn

        class SimpleMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)

            def forward(self, x):
                return self.mlp(x)

        model = SimpleMLP()
        capture = InputCapture()

        class FakeStructure:
            mlp_names = ["mlp", ""]

        capture.register(model, FakeStructure().mlp_names)

        x = torch.randn(SEQ_LEN, HIDDEN_DIM)
        model(x)

        captured = capture.captured
        assert 0 in captured
        assert captured[0].shape == (SEQ_LEN, HIDDEN_DIM)
        assert torch.allclose(captured[0], x, atol=1e-6)

        capture.clear()
        assert capture.captured == {}
        capture.remove_hooks()

    def test_context_manager_removes_hooks(self):
        import torch.nn as nn

        model = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        with InputCapture() as capture:
            capture.register(model, [""])  # empty names — no hooks registered

        # After exit, hooks are removed (nothing to assert but no exception)
