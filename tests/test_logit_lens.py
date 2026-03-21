"""Tests for logit-lens readout logic using synthetic data."""

import pytest

from gpt_oss_interp.readouts.logit_lens import (
    LayerPrediction,
    LogitLensResult,
    format_logit_lens_table,
)


def _make_predictions(num_layers=6, num_positions=3, convergence_at=4):
    """Create synthetic predictions where position 0 converges at a given layer.

    Before convergence: top token changes each layer.
    At and after convergence: top token is fixed (token 42, "answer").
    """
    preds = []
    for layer in range(num_layers):
        for pos in range(num_positions):
            if pos == 0 and layer >= convergence_at:
                # Converged: target token is rank 0
                top_ids = [42, 10, 20, 30, 40]
                top_tokens = ["answer", "wrong1", "wrong2", "wrong3", "wrong4"]
                top_logprobs = [-0.1, -2.0, -3.0, -4.0, -5.0]
                target_rank = 0
                target_logprob = -0.1
            elif pos == 0:
                # Pre-convergence: target token is buried
                top_ids = [10 + layer, 20, 30, 40, 50]
                top_tokens = [f"noise_{layer}", "wrong1", "wrong2", "wrong3", "wrong4"]
                top_logprobs = [-1.0, -2.0, -3.0, -4.0, -5.0]
                target_rank = 100 - layer * 10
                target_logprob = -10.0 + layer
            else:
                # Other positions: stable
                top_ids = [100 + pos, 200, 300, 400, 500]
                top_tokens = [f"stable_{pos}", "a", "b", "c", "d"]
                top_logprobs = [-0.5, -2.0, -3.0, -4.0, -5.0]
                target_rank = 0
                target_logprob = -0.5

            preds.append(LayerPrediction(
                layer_idx=layer,
                position=pos,
                top_token_ids=top_ids,
                top_tokens=top_tokens,
                top_logprobs=top_logprobs,
                target_token_id=42 if pos == 0 else 100 + pos,
                target_token="answer" if pos == 0 else f"stable_{pos}",
                target_rank=target_rank,
                target_logprob=target_logprob,
            ))

    return LogitLensResult(
        predictions=preds,
        num_layers=num_layers,
        num_positions=num_positions,
        tracked_target_ids={0: 42, 1: 101, 2: 102},
    )


class TestLogitLensResult:
    def test_layer_slice(self):
        result = _make_predictions()
        layer_0 = result.layer_slice(0)
        assert len(layer_0) == 3
        assert all(p.layer_idx == 0 for p in layer_0)

    def test_position_slice(self):
        result = _make_predictions()
        pos_0 = result.position_slice(0)
        assert len(pos_0) == 6
        assert all(p.position == 0 for p in pos_0)

    def test_convergence_layer_detected(self):
        result = _make_predictions(convergence_at=4)
        conv = result.convergence_layer(0)
        # convergence_layer finds earliest layer where top-1 matches final top-1
        assert conv is not None
        assert conv <= 4

    def test_convergence_layer_stable_position(self):
        result = _make_predictions()
        # Position 1 is stable from layer 0
        conv = result.convergence_layer(1)
        assert conv == 0

    def test_target_convergence_layer(self):
        result = _make_predictions(convergence_at=4)
        target_conv = result.target_convergence_layer(0)
        assert target_conv == 4

    def test_tracked_target_id(self):
        result = _make_predictions()
        assert result.tracked_target_id(0) == 42
        assert result.tracked_target_id(1) == 101


class TestLogitLensTableFormatting:
    def test_format_produces_string(self):
        result = _make_predictions()
        table = format_logit_lens_table(result, last_n_positions=2)
        assert isinstance(table, str)
        assert len(table) > 0

    def test_format_contains_layer_info(self):
        result = _make_predictions()
        table = format_logit_lens_table(result)
        assert "Layer" in table or "layer" in table or "0" in table
