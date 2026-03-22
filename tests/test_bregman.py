"""Tests for Bregman-conditioning utilities."""

import math

import torch

from gpt_oss_interp.steering.bregman import (
    analyze_bregman_hessian,
    analyze_bregman_state,
    condition_number_from_eigenvalues,
    effective_rank_from_eigenvalues,
    format_bregman_summary,
    summarize_bregman_metrics,
    unembedding_direction,
)


def test_effective_rank_identity_spectrum():
    eigvals = torch.tensor([1.0, 1.0, 1.0, 1.0])
    rank = effective_rank_from_eigenvalues(eigvals)
    assert math.isclose(rank, 4.0, rel_tol=1e-5)


def test_condition_number_reflects_full_spectrum():
    """A near-singular spectrum must report a large condition number,
    not 1.0 from the old supported-subspace logic."""
    eigvals = torch.tensor([1.0, 1e-8, 1e-10])
    cond, numerical_rank = condition_number_from_eigenvalues(eigvals)
    assert cond > 1e9, f"Expected large cond, got {cond}"
    assert numerical_rank == 1


def test_condition_number_isotropic():
    eigvals = torch.tensor([1.0, 1.0, 1.0])
    cond, numerical_rank = condition_number_from_eigenvalues(eigvals)
    assert math.isclose(cond, 1.0, rel_tol=1e-5)
    assert numerical_rank == 3


def test_analyze_bregman_hessian_reports_isotropic_cosine():
    hessian = torch.eye(3)
    direction = torch.tensor([1.0, -2.0, 0.5])
    metrics = analyze_bregman_hessian(hessian, mass_covered=1.0, direction=direction)
    assert math.isclose(metrics.trace, 3.0, rel_tol=1e-5)
    assert math.isclose(metrics.effective_rank, 3.0, rel_tol=1e-5)
    assert math.isclose(metrics.condition_number, 1.0, rel_tol=1e-5)
    assert math.isclose(metrics.cosine_primal_dual, 1.0, rel_tol=1e-5)


def test_analyze_bregman_state_returns_reasonable_metrics():
    hidden = torch.tensor([0.2, -0.1])
    weight = torch.tensor(
        [
            [2.0, 0.0],
            [0.0, 2.0],
            [-1.0, -1.0],
        ]
    )
    metrics = analyze_bregman_state(hidden, weight, top_k_vocab=3)
    assert metrics.trace > 0.0
    assert metrics.effective_rank >= 1.0
    assert metrics.mass_covered > 0.99


def test_unembedding_direction_is_row_difference():
    weight = torch.tensor(
        [
            [3.0, 1.0],
            [0.5, -2.0],
        ]
    )
    direction = unembedding_direction(weight, 0, 1)
    assert torch.equal(direction, torch.tensor([2.5, 3.0]))


def test_summary_and_formatting():
    metrics_by_layer = {
        0: [
            analyze_bregman_hessian(torch.eye(2), mass_covered=1.0, direction=torch.tensor([1.0, 0.0])),
            analyze_bregman_hessian(torch.eye(2), mass_covered=0.95, direction=torch.tensor([0.0, 1.0])),
        ]
    }
    summaries = summarize_bregman_metrics(metrics_by_layer, top_k_vocab=128)
    assert len(summaries) == 1
    assert summaries[0].n_samples == 2
    table = format_bregman_summary(summaries)
    assert "Layer" in table
    assert "Mass Covered" in table
