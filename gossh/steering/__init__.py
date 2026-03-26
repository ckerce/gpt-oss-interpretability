"""gossh.steering — Bregman-geometry diagnostics for steering reliability."""
from .bregman import (
    BregmanMetrics,
    LayerBregmanSummary,
    analyze_bregman_hessian,
    analyze_bregman_state,
    condition_number_from_eigenvalues,
    effective_rank_from_eigenvalues,
    format_bregman_summary,
    softmax_hessian_from_hidden,
    summarize_bregman_metrics,
    unembedding_direction,
)

__all__ = [
    "BregmanMetrics",
    "LayerBregmanSummary",
    "analyze_bregman_hessian",
    "analyze_bregman_state",
    "condition_number_from_eigenvalues",
    "effective_rank_from_eigenvalues",
    "format_bregman_summary",
    "softmax_hessian_from_hidden",
    "summarize_bregman_metrics",
    "unembedding_direction",
]
