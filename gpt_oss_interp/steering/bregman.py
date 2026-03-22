"""Bregman-geometry diagnostics for steering reliability.

This module implements a lightweight version of the conditioning analysis used
in the companion Bregman-conditioning paper, adapted to this repository's
existing hidden-state / unembedding interface.

The key object is the softmax Hessian in hidden-state coordinates:

    H(h) = Cov[w_y | h]

where ``w_y`` is the unembedding row for token ``y`` and the covariance is taken
under the model's predictive distribution at hidden state ``h``.

This formulation keeps the Hessian in the model's hidden dimension instead of
the vocabulary dimension, making the geometry practical to analyze even for
large-vocabulary models like gpt-oss-20b.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
import statistics

import torch


EPS = 1e-8


@dataclass
class BregmanMetrics:
    """Geometry summary for a single layer/position sample."""

    trace: float
    effective_rank: float
    condition_number: float
    numerical_rank: int
    mass_covered: float
    cosine_primal_dual: float | None = None


@dataclass
class LayerBregmanSummary:
    """Aggregate Bregman summary over many samples for one layer."""

    layer_idx: int
    n_samples: int
    top_k_vocab: int
    trace_mean: float
    effective_rank_mean: float
    condition_number_median: float
    mass_covered_mean: float
    cosine_mean: float | None = None
    cosine_std: float | None = None


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.reshape(-1).float()
    b_flat = b.reshape(-1).float()
    denom = torch.norm(a_flat) * torch.norm(b_flat)
    if float(denom.item()) <= EPS:
        return float("nan")
    return float(torch.dot(a_flat, b_flat).item() / denom.item())


def effective_rank_from_eigenvalues(eigenvalues: torch.Tensor) -> float:
    """Return the entropy-based effective rank from non-negative eigenvalues."""

    positive = eigenvalues[eigenvalues > EPS]
    if positive.numel() == 0:
        return 0.0
    weights = positive / positive.sum()
    entropy = float(-(weights * torch.log(weights.clamp_min(EPS))).sum().item())
    return exp(entropy)


def condition_number_from_eigenvalues(
    eigenvalues: torch.Tensor,
    *,
    relative_floor: float = 1e-6,
) -> tuple[float, int]:
    """Return condition number and numerical rank from eigenvalues.

    The condition number is max/min over *all* positive eigenvalues — the
    true conditioning of the Hessian.  The numerical rank counts how many
    eigenvalues exceed ``max * relative_floor``, indicating the dimension
    of the numerically supported eigenspace.

    A near-singular Hessian like [1, 1e-8, 1e-10] correctly reports a
    large condition number (1e10) and numerical rank 1.
    """

    # Use a tight positivity floor (machine-epsilon scale) to exclude
    # only truly zero eigenvalues.  The larger relative_floor is used
    # separately to compute numerical rank.
    positivity_floor = float(eigenvalues.max().item()) * 1e-15 if eigenvalues.max().item() > 0 else EPS
    positive = eigenvalues[eigenvalues > positivity_floor]
    if positive.numel() == 0:
        return float("inf"), 0

    cond = float((positive.max() / positive.min()).item())

    cutoff = float(positive.max().item()) * relative_floor
    numerical_rank = int((positive >= cutoff).sum().item())

    return cond, numerical_rank


def softmax_hessian_from_hidden(
    hidden_state: torch.Tensor,
    lm_head_weight: torch.Tensor,
    *,
    lm_head_bias: torch.Tensor | None = None,
    top_k_vocab: int = 2048,
) -> tuple[torch.Tensor, float]:
    """Approximate the hidden-space softmax Hessian at one hidden state."""

    hidden = hidden_state.float().reshape(-1)
    weight = lm_head_weight.float()

    logits = weight @ hidden
    if lm_head_bias is not None:
        logits = logits + lm_head_bias.float()

    vocab_size = logits.shape[0]
    keep = min(max(1, top_k_vocab), vocab_size)
    if keep == vocab_size:
        top_logits = logits
        top_idx = torch.arange(vocab_size, device=logits.device)
    else:
        top_logits, top_idx = torch.topk(logits, k=keep)

    log_z = torch.logsumexp(logits, dim=0)
    selected_probs = torch.exp(top_logits - log_z)
    mass_covered = float(selected_probs.sum().item())

    if mass_covered <= EPS:
        raise ValueError("Selected probability mass is numerically zero.")

    # Use the actual probabilities from the full partition function.
    # Do NOT renormalize over the top-k support — renormalization would
    # measure the geometry of a conditional distribution, not the model's
    # actual softmax distribution.  The approximation error is bounded by
    # (1 - mass_covered); callers should check mass_covered is high enough.
    selected_vectors = weight[top_idx]
    mean = selected_probs @ selected_vectors
    centered = selected_vectors - mean
    weighted = centered * torch.sqrt(selected_probs).unsqueeze(1)
    hessian = weighted.T @ weighted

    return hessian.cpu(), mass_covered


def analyze_bregman_hessian(
    hessian: torch.Tensor,
    *,
    mass_covered: float,
    direction: torch.Tensor | None = None,
    relative_floor: float = 1e-6,
) -> BregmanMetrics:
    """Compute scalar geometry diagnostics from a hidden-space Hessian."""

    sym = 0.5 * (hessian.float() + hessian.float().T)
    eigvals = torch.linalg.eigvalsh(sym).clamp_min(0)

    cond, numerical_rank = condition_number_from_eigenvalues(
        eigvals,
        relative_floor=relative_floor,
    )
    cosine = None
    if direction is not None:
        primal = direction.float().reshape(-1)
        dual = sym @ primal
        cosine = _cosine_similarity(primal, dual)

    return BregmanMetrics(
        trace=float(eigvals.sum().item()),
        effective_rank=effective_rank_from_eigenvalues(eigvals),
        condition_number=cond,
        numerical_rank=numerical_rank,
        mass_covered=mass_covered,
        cosine_primal_dual=cosine,
    )


def analyze_bregman_state(
    hidden_state: torch.Tensor,
    lm_head_weight: torch.Tensor,
    *,
    lm_head_bias: torch.Tensor | None = None,
    top_k_vocab: int = 2048,
    direction: torch.Tensor | None = None,
    relative_floor: float = 1e-6,
) -> BregmanMetrics:
    """Convenience wrapper: hidden state -> Hessian -> scalar metrics."""

    hessian, mass_covered = softmax_hessian_from_hidden(
        hidden_state,
        lm_head_weight,
        lm_head_bias=lm_head_bias,
        top_k_vocab=top_k_vocab,
    )
    return analyze_bregman_hessian(
        hessian,
        mass_covered=mass_covered,
        direction=direction,
        relative_floor=relative_floor,
    )


def unembedding_direction(
    lm_head_weight: torch.Tensor,
    token_a_id: int,
    token_b_id: int,
) -> torch.Tensor:
    """Return the exact vocabulary-space steering direction W[a] - W[b]."""

    return lm_head_weight[token_a_id].float() - lm_head_weight[token_b_id].float()


def summarize_bregman_metrics(
    metrics_by_layer: dict[int, list[BregmanMetrics]],
    *,
    top_k_vocab: int,
) -> list[LayerBregmanSummary]:
    """Aggregate per-sample metrics into per-layer summaries."""

    summaries: list[LayerBregmanSummary] = []
    for layer_idx in sorted(metrics_by_layer):
        metrics = metrics_by_layer[layer_idx]
        cosines = [
            m.cosine_primal_dual
            for m in metrics
            if m.cosine_primal_dual is not None and m.cosine_primal_dual == m.cosine_primal_dual
        ]
        summaries.append(
            LayerBregmanSummary(
                layer_idx=layer_idx,
                n_samples=len(metrics),
                top_k_vocab=top_k_vocab,
                trace_mean=statistics.fmean(m.trace for m in metrics),
                effective_rank_mean=statistics.fmean(m.effective_rank for m in metrics),
                condition_number_median=float(statistics.median(m.condition_number for m in metrics)),
                mass_covered_mean=statistics.fmean(m.mass_covered for m in metrics),
                cosine_mean=(statistics.fmean(cosines) if cosines else None),
                cosine_std=(statistics.pstdev(cosines) if len(cosines) > 1 else 0.0 if cosines else None),
            )
        )
    return summaries


def format_bregman_summary(summaries: list[LayerBregmanSummary]) -> str:
    """Format layer summaries as a markdown table."""

    lines = [
        "| Layer | Samples | Trace | Effective Rank | Cond. Median | Mass Covered | Cosine |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        cosine = f"{row.cosine_mean:.3f}" if row.cosine_mean is not None else "—"
        lines.append(
            "| "
            f"{row.layer_idx} | "
            f"{row.n_samples} | "
            f"{row.trace_mean:.4f} | "
            f"{row.effective_rank_mean:.2f} | "
            f"{row.condition_number_median:.2f} | "
            f"{row.mass_covered_mean:.4f} | "
            f"{cosine} |"
        )
    return "\n".join(lines)
