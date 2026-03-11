###############################################################################
#
# Geometric analysis of feature-space structure
#
# Implements the metric-space analysis described in GEOMETRIC_FRAMEWORK.md:
# - Pairwise distance matrices (pullback metric on data)
# - Intrinsic dimensionality estimation
# - Stratification analysis (by processing depth, routing pattern)
# - Inspectability scoring
#
###############################################################################

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import numpy as np


@dataclass
class GeometricSummary:
    """Summary statistics of the feature-space geometry."""
    n_tokens: int
    feature_dim: int
    intrinsic_dim_pca: float         # Estimated by PCA eigenvalue decay
    depth_stratification: float      # How cleanly k* separates feature space
    mean_pairwise_distance: float
    std_pairwise_distance: float
    inspectability_scores: np.ndarray  # Per-token inspectability
    pca_explained_variance: np.ndarray  # Eigenvalue spectrum


def compute_pairwise_distances(features: torch.Tensor) -> torch.Tensor:
    """Compute pairwise Euclidean distance matrix.

    Args:
        features: [N, D] feature vectors.

    Returns:
        [N, N] distance matrix.
    """
    # Efficient computation: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    norms_sq = (features ** 2).sum(dim=1)
    dists_sq = norms_sq.unsqueeze(0) + norms_sq.unsqueeze(1) - 2 * features @ features.T
    return torch.sqrt(dists_sq.clamp(min=0))


def estimate_intrinsic_dimension(features: torch.Tensor, threshold: float = 0.9) -> float:
    """Estimate intrinsic dimensionality via PCA eigenvalue decay.

    Returns the number of components needed to explain `threshold` fraction
    of total variance. This measures how many "directions" the computation
    actually uses, regardless of the ambient dimensionality.
    """
    centered = features - features.mean(dim=0)
    # Use SVD for numerical stability
    _, S, _ = torch.svd(centered)
    eigenvalues = S ** 2 / (features.shape[0] - 1)
    cumulative = eigenvalues.cumsum(0) / eigenvalues.sum()
    intrinsic = (cumulative < threshold).sum().item() + 1
    return float(intrinsic)


def depth_stratification_score(
    features: torch.Tensor,
    processing_depth: torch.Tensor,
) -> float:
    """Measure how cleanly processing depth separates the feature space.

    Computes the ratio of between-depth variance to total variance.
    1.0 = depth perfectly partitions feature space (ideal stratification).
    0.0 = depth is uncorrelated with feature structure.
    """
    unique_depths = processing_depth.unique()
    if len(unique_depths) <= 1:
        return 0.0

    total_mean = features.mean(dim=0)
    total_var = ((features - total_mean) ** 2).sum()

    between_var = 0.0
    for d in unique_depths:
        mask = processing_depth == d
        if mask.sum() == 0:
            continue
        group = features[mask]
        group_mean = group.mean(dim=0)
        between_var += mask.sum().item() * ((group_mean - total_mean) ** 2).sum().item()

    return between_var / max(total_var.item(), 1e-12)


def compute_inspectability(
    features: torch.Tensor,
    k: int = 10,
) -> np.ndarray:
    """Compute per-token inspectability scores.

    Inspectability is high when a token's neighborhood in feature space
    is dense and low-dimensional (consistent computational mode).

    Score = local_density × (1 / local_intrinsic_dim)

    Args:
        features: [N, D] feature vectors.
        k: Number of neighbors to consider.

    Returns:
        [N] inspectability scores (higher = more inspectable).
    """
    N = features.shape[0]
    k = min(k, N - 1)
    if k < 2:
        return np.ones(N)

    dists = compute_pairwise_distances(features)
    # For each token, find k-nearest neighbors
    knn_dists, knn_idx = dists.topk(k + 1, dim=1, largest=False)
    # Skip self (distance 0)
    knn_dists = knn_dists[:, 1:]  # [N, k]

    # Local density: inverse of mean k-NN distance
    mean_dist = knn_dists.mean(dim=1)
    density = 1.0 / (mean_dist + 1e-8)

    # Local intrinsic dimension (MLE estimator):
    # d_hat = 1 / (mean(log(r_k / r_j)) for j < k)
    # Simplified: use ratio of last to first neighbor distance
    r_ratio = knn_dists[:, -1] / (knn_dists[:, 0] + 1e-8)
    local_dim = torch.log(r_ratio + 1e-8).clamp(min=0.1)

    scores = density / local_dim
    # Normalize to [0, 1]
    scores = scores - scores.min()
    scores = scores / (scores.max() + 1e-8)

    return scores.numpy()


def analyze_geometry(
    features: torch.Tensor,
    processing_depth: torch.Tensor,
) -> GeometricSummary:
    """Full geometric analysis of a feature point cloud.

    Args:
        features: [N, D] feature vectors (standardized recommended).
        processing_depth: [N] stability layer k* per token.

    Returns:
        GeometricSummary with all analysis results.
    """
    N, D = features.shape

    # Standardize features for analysis
    std = features.std(dim=0).clamp(min=1e-8)
    standardized = (features - features.mean(dim=0)) / std

    # PCA spectrum
    _, S, _ = torch.svd(standardized)
    eigenvalues = (S ** 2 / (N - 1)).numpy()
    explained = eigenvalues / eigenvalues.sum()

    # Intrinsic dimension
    cumulative = np.cumsum(explained)
    intrinsic_90 = float((cumulative < 0.9).sum() + 1)

    # Depth stratification
    strat = depth_stratification_score(standardized, processing_depth)

    # Pairwise distances
    dists = compute_pairwise_distances(standardized)
    # Upper triangle only (avoid double-counting and self-distances)
    upper = dists[torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)]
    mean_dist = upper.mean().item()
    std_dist = upper.std().item()

    # Inspectability
    inspect_scores = compute_inspectability(standardized, k=min(10, N - 1))

    return GeometricSummary(
        n_tokens=N,
        feature_dim=D,
        intrinsic_dim_pca=intrinsic_90,
        depth_stratification=strat,
        mean_pairwise_distance=mean_dist,
        std_pairwise_distance=std_dist,
        inspectability_scores=inspect_scores,
        pca_explained_variance=explained,
    )


def format_geometric_report(summary: GeometricSummary, token_strings: list[str] | None = None) -> str:
    """Format geometric analysis as a markdown report."""
    lines = ["# Geometric Analysis of Feature Space\n"]

    lines.append(f"| Property | Value |")
    lines.append(f"| --- | --- |")
    lines.append(f"| Tokens analyzed | {summary.n_tokens} |")
    lines.append(f"| Feature dimensions | {summary.feature_dim} |")
    lines.append(f"| Intrinsic dimension (90% var) | {summary.intrinsic_dim_pca:.0f} |")
    lines.append(f"| Depth stratification score | {summary.depth_stratification:.3f} |")
    lines.append(f"| Mean pairwise distance | {summary.mean_pairwise_distance:.3f} |")
    lines.append(f"| Distance std dev | {summary.std_pairwise_distance:.3f} |")
    lines.append("")

    # PCA spectrum (top 20)
    lines.append("## PCA Eigenvalue Spectrum (top 20)\n")
    lines.append("| Component | Variance Explained | Cumulative |")
    lines.append("| ---: | ---: | ---: |")
    cumulative = 0.0
    for i, v in enumerate(summary.pca_explained_variance[:20]):
        cumulative += v
        lines.append(f"| {i+1} | {v:.4f} | {cumulative:.4f} |")
    lines.append("")

    # Most / least inspectable tokens
    if token_strings and len(token_strings) == summary.n_tokens:
        scores = summary.inspectability_scores
        ranked = np.argsort(scores)[::-1]

        lines.append("## Most Inspectable Tokens\n")
        lines.append("| Rank | Token | Score |")
        lines.append("| ---: | --- | ---: |")
        for i in range(min(10, len(ranked))):
            idx = ranked[i]
            lines.append(f"| {i+1} | {repr(token_strings[idx])} | {scores[idx]:.3f} |")
        lines.append("")

        lines.append("## Least Inspectable Tokens (computational outliers)\n")
        lines.append("| Rank | Token | Score |")
        lines.append("| ---: | --- | ---: |")
        for i in range(min(10, len(ranked))):
            idx = ranked[-(i+1)]
            lines.append(f"| {i+1} | {repr(token_strings[idx])} | {scores[idx]:.3f} |")

    return "\n".join(lines)
