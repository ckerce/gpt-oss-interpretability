"""Tests for synthetic unembedding-validation helpers."""

import math

import torch

from gpt_oss_interp.readouts.unembedding_validation import (
    build_periodic_prompt,
    distribution_metrics,
    summarize_results,
    CaseValidationResult,
    LayerValidationRecord,
)


class DummyTokenizer:
    def __init__(self):
        self.mapping = {0: " bird", 1: " cat", 2: " fish"}

    def decode(self, token_ids):
        return "".join(self.mapping[token_id] for token_id in token_ids)


def test_build_periodic_prompt_returns_expected_next_word():
    prompt, expected = build_periodic_prompt(("bird", "cat"), repeats=3)
    assert prompt == "bird cat bird cat bird cat"
    assert expected == "bird"


def test_distribution_metrics_tracks_tail_mass_and_top_token():
    tokenizer = DummyTokenizer()
    log_probs = torch.log(torch.tensor([0.85, 0.10, 0.05], dtype=torch.float32))
    metrics = distribution_metrics(log_probs, (0,), tokenizer, top_k=3)

    assert metrics["expected_token"] == " bird"
    assert metrics["top_token"] == " bird"
    assert metrics["top_is_valid"] is True
    assert math.isclose(metrics["valid_mass"], 0.85, rel_tol=1e-6)
    assert math.isclose(metrics["tail_mass"], 0.15, rel_tol=1e-6)
    assert metrics["expected_rank"] == 0


def test_summarize_results_aggregates_layer_statistics():
    layers_a = [
        LayerValidationRecord(0, 0, " bird", -0.1, 0.9, 0, 0.9, 0.1, 0, " bird", -0.1, 0.9, True, [" bird"], [0], [-0.1]),
        LayerValidationRecord(1, 0, " bird", -0.02, 0.98, 0, 0.98, 0.02, 0, " bird", -0.02, 0.98, True, [" bird"], [0], [-0.02]),
    ]
    layers_b = [
        LayerValidationRecord(0, 0, " bird", -0.7, 0.5, 1, 0.5, 0.5, 1, " cat", -0.6, 0.55, False, [" cat"], [1], [-0.6]),
        LayerValidationRecord(1, 0, " bird", -0.03, 0.97, 0, 0.97, 0.03, 0, " bird", -0.03, 0.97, True, [" bird"], [0], [-0.03]),
    ]
    results = [
        CaseValidationResult("c1", 2, ("bird", "cat"), "bird cat bird cat", "bird", 0, True, layers_a),
        CaseValidationResult("c2", 2, ("bird", "cat"), "bird cat bird cat", "bird", 1, True, layers_b),
    ]

    summaries = summarize_results(results, [2])
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.period == 2
    assert summary.n_cases == 2
    assert math.isclose(summary.final_top_match_rate, 1.0, rel_tol=1e-6)
    assert math.isclose(summary.mean_convergence_layer, 0.5, rel_tol=1e-6)
    assert math.isclose(summary.layers[0].top_is_valid_rate, 0.5, rel_tol=1e-6)
