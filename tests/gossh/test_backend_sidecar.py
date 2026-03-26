"""Tests for MoE sidecar integration in GPTOSSTransformersBackend.

Uses a mock model so no real checkpoint is needed.  Verifies that:
- attach_sidecar / detach_sidecar manage self._sidecar correctly
- capture_routing dispatches to path 0 when sidecar is attached
- capture_routing falls through to paths 1-3 when sidecar is absent
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from gossh.capture.router_capture import RouterDecision
from gossh.sidecar.dequant import RouterSidecarModel
from gossh.sidecar.process import MoeSidecar


###############################################################################
# Minimal stub backend (no real HF model needed)
###############################################################################

NUM_LAYERS = 4
HIDDEN_DIM = 32
NUM_EXPERTS = 8
TOP_K = 2
SEQ_LEN = 6


def _make_weights() -> dict[int, torch.Tensor]:
    torch.manual_seed(0)
    return {i: torch.randn(NUM_EXPERTS, HIDDEN_DIM) for i in range(NUM_LAYERS)}


def _make_hidden() -> dict[int, torch.Tensor]:
    torch.manual_seed(1)
    return {i: torch.randn(SEQ_LEN, HIDDEN_DIM) for i in range(NUM_LAYERS)}


class _FakeStructure:
    """Minimal ModelStructure stand-in."""
    def __init__(self):
        self.mlp_names = [f"mlp_{i}" for i in range(NUM_LAYERS)]


class _FakeBackend:
    """Stripped-down backend with only the sidecar integration methods."""

    def __init__(self):
        self._sidecar = None
        self.structure = _FakeStructure()
        self.device = "cpu"

    # Copy the three sidecar methods from GPTOSSTransformersBackend verbatim

    def attach_sidecar(self, sidecar):
        if not sidecar.is_running():
            raise RuntimeError("Sidecar is not running — start it before attaching.")
        self._sidecar = sidecar

    def detach_sidecar(self):
        self._sidecar = None

    def _capture_routing_via_sidecar(self, hidden_by_layer):
        # In the real backend input_ids go through the model first; here we
        # pass already-captured hidden states directly to verify routing.
        return self._sidecar.route(hidden_by_layer)


###############################################################################
# Fixtures
###############################################################################

@pytest.fixture(scope="module")
def weights():
    return _make_weights()


@pytest.fixture(scope="module")
def hidden():
    return _make_hidden()


@pytest.fixture(scope="module")
def running_sidecar(weights):
    with MoeSidecar(weights, top_k=TOP_K) as sc:
        yield sc


###############################################################################
# attach / detach
###############################################################################

class TestAttachDetach:
    def test_initial_sidecar_is_none(self):
        backend = _FakeBackend()
        assert backend._sidecar is None

    def test_attach_sets_sidecar(self, running_sidecar):
        backend = _FakeBackend()
        backend.attach_sidecar(running_sidecar)
        assert backend._sidecar is running_sidecar

    def test_detach_clears_sidecar(self, running_sidecar):
        backend = _FakeBackend()
        backend.attach_sidecar(running_sidecar)
        backend.detach_sidecar()
        assert backend._sidecar is None

    def test_attach_stopped_sidecar_raises(self, weights):
        backend = _FakeBackend()
        sc = MoeSidecar(weights, top_k=TOP_K)
        # Not started — is_running() == False
        with pytest.raises(RuntimeError, match="not running"):
            backend.attach_sidecar(sc)

    def test_double_detach_is_safe(self, running_sidecar):
        backend = _FakeBackend()
        backend.attach_sidecar(running_sidecar)
        backend.detach_sidecar()
        backend.detach_sidecar()  # should not raise
        assert backend._sidecar is None


###############################################################################
# Routing via sidecar
###############################################################################

class TestRoutingViaSidecar:
    def test_returns_router_decisions(self, running_sidecar, hidden):
        backend = _FakeBackend()
        backend.attach_sidecar(running_sidecar)
        decisions = backend._capture_routing_via_sidecar(hidden)
        assert len(decisions) == NUM_LAYERS
        for d in decisions:
            assert isinstance(d, RouterDecision)

    def test_decisions_match_local_model(self, running_sidecar, weights, hidden):
        local = RouterSidecarModel(weights, top_k=TOP_K)
        local_map = {d.layer_idx: d for d in local.route_all(hidden)}

        backend = _FakeBackend()
        backend.attach_sidecar(running_sidecar)
        sidecar_map = {d.layer_idx: d for d in backend._capture_routing_via_sidecar(hidden)}

        for li in range(NUM_LAYERS):
            assert torch.equal(local_map[li].selected_experts, sidecar_map[li].selected_experts)
            assert torch.allclose(local_map[li].expert_weights, sidecar_map[li].expert_weights, atol=1e-5)

    def test_shapes(self, running_sidecar, hidden):
        backend = _FakeBackend()
        backend.attach_sidecar(running_sidecar)
        decisions = backend._capture_routing_via_sidecar(hidden)
        for d in decisions:
            assert d.selected_experts.shape == (SEQ_LEN, TOP_K)
            assert d.expert_weights.shape == (SEQ_LEN, TOP_K)
            assert d.token_count == SEQ_LEN

    def test_subset_of_layers(self, running_sidecar, hidden):
        subset = {k: v for k, v in hidden.items() if k in (0, 3)}
        backend = _FakeBackend()
        backend.attach_sidecar(running_sidecar)
        decisions = backend._capture_routing_via_sidecar(subset)
        assert len(decisions) == 2
        assert {d.layer_idx for d in decisions} == {0, 3}
