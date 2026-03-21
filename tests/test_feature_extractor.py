"""Tests for the MoE feature extractor using synthetic tensors."""

import pytest
import torch
import numpy as np

from gpt_oss_interp.features.extractor import FeatureConfig, MoEFeatureExtractor
from gpt_oss_interp.features.geometry import (
    compute_pairwise_distances,
    estimate_intrinsic_dimension,
    depth_stratification_score,
)


def _default_config():
    return FeatureConfig(
        n_layers=6,
        n_query_heads=8,
        n_kv_heads=2,
        n_experts=4,
        top_k_experts=2,
        head_dim=32,
    )


def _make_synthetic_inputs(config, seq_len=10):
    """Create synthetic layer_logits, layer_attentions, expert_routing."""
    vocab_size = 100
    layer_logits = torch.randn(config.n_layers, seq_len, vocab_size)
    layer_attentions = torch.randn(
        config.n_layers, config.n_query_heads, seq_len, seq_len
    ).softmax(dim=-1)
    expert_routing = torch.randn(
        config.n_layers, seq_len, config.n_experts
    ).softmax(dim=-1)
    token_strings = [f"token_{i}" for i in range(seq_len)]
    return layer_logits, layer_attentions, expert_routing, token_strings


class TestFeatureConfig:
    def test_feature_dim_all_enabled(self):
        config = _default_config()
        dim = config.feature_dim
        assert dim > 0
        # Should be sum of all component dimensions

    def test_feature_dim_trajectory_only(self):
        config = _default_config()
        config.include_head_activation = False
        config.include_head_entropy = False
        config.include_expert_routing = False
        config.include_routing_entropy = False
        config.include_attention_scale = False
        dim_traj = config.feature_dim
        assert dim_traj > 0
        assert dim_traj < _default_config().feature_dim

    def test_feature_dim_zero_when_all_disabled(self):
        config = _default_config()
        config.include_trajectory = False
        config.include_stability = False
        config.include_head_activation = False
        config.include_head_entropy = False
        config.include_expert_routing = False
        config.include_routing_entropy = False
        config.include_attention_scale = False
        assert config.feature_dim == 0


class TestMoEFeatureExtractor:
    def test_extract_produces_correct_shape(self):
        config = _default_config()
        extractor = MoEFeatureExtractor(config)
        logits, attn, routing, tokens = _make_synthetic_inputs(config, seq_len=10)
        result = extractor.extract(logits, attn, routing, tokens)
        assert result.feature_vectors.shape == (10, config.feature_dim)

    def test_extract_without_attention(self):
        config = _default_config()
        config.include_head_activation = False
        config.include_head_entropy = False
        config.include_attention_scale = False
        extractor = MoEFeatureExtractor(config)
        logits, _, routing, tokens = _make_synthetic_inputs(config)
        result = extractor.extract(logits, None, routing, tokens)
        assert result.num_tokens == 10

    def test_extract_without_routing(self):
        config = _default_config()
        config.include_expert_routing = False
        config.include_routing_entropy = False
        extractor = MoEFeatureExtractor(config)
        logits, attn, _, tokens = _make_synthetic_inputs(config)
        result = extractor.extract(logits, attn, None, tokens)
        assert result.num_tokens == 10

    def test_extract_trajectory_only(self):
        config = _default_config()
        config.include_head_activation = False
        config.include_head_entropy = False
        config.include_expert_routing = False
        config.include_routing_entropy = False
        config.include_attention_scale = False
        extractor = MoEFeatureExtractor(config)
        logits, _, _, tokens = _make_synthetic_inputs(config)
        result = extractor.extract(logits, None, None, tokens)
        assert result.feature_vectors.shape[0] == 10
        assert result.feature_vectors.shape[1] == config.feature_dim

    def test_processing_depth_is_valid(self):
        config = _default_config()
        extractor = MoEFeatureExtractor(config)
        logits, attn, routing, tokens = _make_synthetic_inputs(config)
        result = extractor.extract(logits, attn, routing, tokens)
        assert result.processing_depth is not None
        assert len(result.processing_depth) == 10
        for d in result.processing_depth:
            assert 0 <= d <= config.n_layers

    def test_same_input_same_output(self):
        config = _default_config()
        extractor = MoEFeatureExtractor(config)
        logits, attn, routing, tokens = _make_synthetic_inputs(config)
        r1 = extractor.extract(logits, attn, routing, tokens)
        r2 = extractor.extract(logits, attn, routing, tokens)
        assert torch.allclose(r1.feature_vectors, r2.feature_vectors)


class TestGeometry:
    def test_pairwise_distances_shape(self):
        features = torch.randn(20, 50)
        dists = compute_pairwise_distances(features)
        assert dists.shape == (20, 20)

    def test_pairwise_distances_diagonal_zero(self):
        features = torch.randn(10, 30)
        dists = compute_pairwise_distances(features)
        dists_np = dists.numpy() if isinstance(dists, torch.Tensor) else dists
        np.testing.assert_allclose(np.diag(dists_np), 0, atol=0.01)

    def test_pairwise_distances_symmetric(self):
        features = torch.randn(10, 30)
        dists = compute_pairwise_distances(features)
        dists_np = dists.numpy() if isinstance(dists, torch.Tensor) else dists
        np.testing.assert_allclose(dists_np, dists_np.T, atol=1e-5)

    def test_intrinsic_dimension_reasonable(self):
        # 5D signal embedded in 50D space
        rng = np.random.default_rng(42)
        basis = rng.standard_normal((5, 50))
        raw = rng.standard_normal((100, 5)) @ basis
        features = torch.tensor(raw, dtype=torch.float32)
        dim = estimate_intrinsic_dimension(features, threshold=0.9)
        assert 3 <= dim <= 10  # Should be close to 5

    def test_depth_stratification_clustered(self):
        # Two clusters with very different depths
        rng = np.random.default_rng(42)
        group_a = rng.standard_normal((50, 20)) + 5.0
        group_b = rng.standard_normal((50, 20)) - 5.0
        features = torch.tensor(np.vstack([group_a, group_b]), dtype=torch.float32)
        depths = torch.tensor([0] * 50 + [5] * 50, dtype=torch.float32)
        score = depth_stratification_score(features, depths)
        assert score > 0.5  # Should show strong stratification

    def test_depth_stratification_random(self):
        rng = np.random.default_rng(42)
        features = torch.tensor(rng.standard_normal((100, 20)), dtype=torch.float32)
        depths = torch.tensor(rng.integers(0, 6, 100), dtype=torch.float32)
        score = depth_stratification_score(features, depths)
        assert score < 0.3  # Random assignment → weak stratification
