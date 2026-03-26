"""gossh.features — feature extraction and geometric analysis."""
from .extractor import FeatureConfig, FeatureResult, MoEFeatureExtractor, extract_features_from_backend
from .geometry import (
    GeometricSummary,
    analyze_geometry,
    compute_pairwise_distances,
    compute_inspectability,
    depth_stratification_score,
    estimate_intrinsic_dimension,
    format_geometric_report,
)

__all__ = [
    "FeatureConfig",
    "FeatureResult",
    "MoEFeatureExtractor",
    "extract_features_from_backend",
    "GeometricSummary",
    "analyze_geometry",
    "compute_pairwise_distances",
    "compute_inspectability",
    "depth_stratification_score",
    "estimate_intrinsic_dimension",
    "format_geometric_report",
]
