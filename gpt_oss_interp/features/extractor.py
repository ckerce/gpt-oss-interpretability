###############################################################################
#
# Extended feature extraction for gpt-oss-20b
#
# Adapts the NeurIPS 2026 Tier-2 feature system to MoE architectures.
# Computes per-token feature vectors from forward-pass data, enabling
# computational mode discovery via clustering.
#
# The feature map φ: Tokens-in-Context → ℝ^D simultaneously equips both
# the input data and the model's components with natural metric structure.
# See GEOMETRIC_FRAMEWORK.md for the mathematical exposition.
#
###############################################################################

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class FeatureConfig:
    """Configuration for the extended feature extractor."""
    n_layers: int = 24
    n_query_heads: int = 64
    n_kv_heads: int = 8
    n_experts: int = 32
    top_k_experts: int = 4
    head_dim: int = 64

    # Feature selection
    include_trajectory: bool = True       # Component A
    include_stability: bool = True        # Component B
    include_head_activation: bool = True  # Component C
    include_head_entropy: bool = True     # Component D
    include_expert_routing: bool = True   # Component E (new for MoE)
    include_routing_entropy: bool = True  # Component F (new for MoE)
    include_attention_scale: bool = True  # Component G (new for sliding/full)

    # Entropy gating threshold (suppress noise from inactive heads)
    entropy_gate_threshold: float = 0.1

    # Sliding attention window size (for computing local/global contrast)
    sliding_window: int = 128

    @property
    def feature_dim(self) -> int:
        """Total feature dimensionality."""
        d = 0
        L, H = self.n_layers, self.n_query_heads
        if self.include_trajectory:
            d += 3 * L - 1  # probs + margins + drops
        if self.include_stability:
            d += 2  # k*, κ
        if self.include_head_activation:
            d += 2 * L * H  # stable + final
        if self.include_head_entropy:
            d += 2 * L * H  # stable + final
        if self.include_expert_routing:
            d += 2 * self.n_experts  # routing at stable + final layers
            d += L * self.top_k_experts  # sorted top-k weights at all layers
        if self.include_routing_entropy:
            d += L
        if self.include_attention_scale:
            d += L  # local fraction per layer
        return d


@dataclass
class FeatureResult:
    """Per-token feature vectors with metadata."""
    feature_vectors: torch.Tensor     # [seq_len, feature_dim]
    processing_depth: torch.Tensor    # [seq_len] — stability layer k*
    confidence: torch.Tensor          # [seq_len] — final-layer confidence
    token_strings: list[str]          # Decoded tokens
    config: FeatureConfig
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_tokens(self) -> int:
        return self.feature_vectors.shape[0]

    @property
    def feature_dim(self) -> int:
        return self.feature_vectors.shape[1]


class MoEFeatureExtractor:
    """Extended feature extractor for MoE transformer models.

    Computes the Tier-2 feature system from the NeurIPS 2026 paper, extended
    with MoE routing features (Components E, F, G). The resulting feature
    vectors define a metric on the input that captures computational structure
    orthogonal to vocabulary/semantics.

    Requires:
        - Per-layer logits: [n_layers, seq_len, vocab_size]
        - Per-layer attention weights: [n_layers, n_heads, seq_len, seq_len]
        - (Optional) Per-layer expert routing: [n_layers, seq_len, n_experts]

    For gpt-oss-20b, these are obtained from:
        - Logit-lens projection of hidden states through final_norm + lm_head
        - Attention weight hooks on self_attn modules
        - Router hooks (requires non-quantized checkpoint) or proxy features
    """

    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()

    def extract(
        self,
        layer_logits: torch.Tensor,
        layer_attentions: torch.Tensor | None = None,
        expert_routing: torch.Tensor | None = None,
        token_strings: list[str] | None = None,
    ) -> FeatureResult:
        """Extract feature vectors from forward-pass data.

        Args:
            layer_logits: [n_layers, seq_len, vocab_size] — per-layer logits
                from logit-lens projection.
            layer_attentions: [n_layers, n_heads, seq_len, seq_len] — attention
                weights. If None, head activation/entropy features are zeroed.
            expert_routing: [n_layers, seq_len, n_experts] — expert routing
                weights. If None, routing features are zeroed (MXFP4 fallback).
            token_strings: Decoded token strings for metadata.

        Returns:
            FeatureResult with per-token feature vectors.
        """
        cfg = self.config
        n_layers, seq_len, vocab_size = layer_logits.shape
        device = layer_logits.device

        # --- Component B: Stability (needed first, indexes other components) ---
        layer_preds = layer_logits.argmax(dim=-1)  # [L, T]
        final_preds = layer_preds[-1]  # [T]

        # k*: earliest layer where prediction matches final and stays stable
        matches = layer_preds == final_preds.unsqueeze(0)  # [L, T]
        stability_mask = matches.flip(0).cumprod(dim=0).flip(0)
        any_stable = stability_mask.any(dim=0)
        first_stable = stability_mask.float().argmax(dim=0)
        k_star = torch.where(
            any_stable, first_stable,
            torch.full_like(first_stable, n_layers),
        )
        # If only the final layer matches, mark as non-converged
        if n_layers > 1:
            final_only = k_star == (n_layers - 1)
            k_star = torch.where(
                final_only,
                torch.full_like(k_star, n_layers),
                k_star,
            )

        # κ: max consecutive correct layers
        streak = torch.zeros(seq_len, dtype=torch.long, device=device)
        kappa = torch.zeros_like(streak)
        for ell in range(n_layers):
            streak = torch.where(matches[ell], streak + 1, torch.zeros_like(streak))
            kappa = torch.maximum(kappa, streak)

        processing_depth = torch.where(
            k_star == n_layers,
            torch.tensor(float(n_layers), device=device),
            k_star.float(),
        )

        # --- Component A: Trajectory ---
        traj_probs_full = F.softmax(layer_logits, dim=-1)  # [L, T, V]
        traj_probs = traj_probs_full.gather(
            -1, final_preds.unsqueeze(0).unsqueeze(-1).expand(n_layers, -1, -1),
        ).squeeze(-1)  # [L, T]

        # Margin over second-best
        top2_vals, top2_idx = traj_probs_full.topk(2, dim=-1)
        second_best = torch.where(
            top2_idx[..., 0] == final_preds.unsqueeze(0),
            top2_vals[..., 1],
            top2_vals[..., 0],
        )
        traj_margin = traj_probs - second_best  # [L, T]

        # Confidence drops between layers
        traj_drops = torch.clamp(traj_probs[:-1] - traj_probs[1:], min=0.0)  # [L-1, T]

        # Final-layer confidence
        confidence = traj_probs[-1]  # [T]

        # --- Component C, D: Head Activation and Entropy ---
        n_heads = cfg.n_query_heads
        head_act_stable = torch.zeros(seq_len, n_layers * n_heads, device=device)
        head_act_final = torch.zeros_like(head_act_stable)
        head_ent_stable = torch.zeros_like(head_act_stable)
        head_ent_final = torch.zeros_like(head_act_stable)

        if layer_attentions is not None:
            actual_heads = layer_attentions.shape[1]
            for t in range(seq_len):
                query_pos = max(t - 1, 0)
                stable_layer = min(int(k_star[t].item()), n_layers - 1)

                if query_pos > 0:
                    # Stable layer
                    attn_slice = layer_attentions[stable_layer, :, query_pos, :query_pos]
                    peak_attn = attn_slice.max(dim=-1).values  # [H]
                    entropy = self._batch_entropy(attn_slice)
                    entropy = torch.where(
                        peak_attn >= cfg.entropy_gate_threshold,
                        entropy, torch.zeros_like(entropy),
                    )
                    base = stable_layer * actual_heads
                    head_act_stable[t, base:base + actual_heads] = peak_attn
                    head_ent_stable[t, base:base + actual_heads] = entropy

                    # Final layer
                    attn_slice = layer_attentions[-1, :, query_pos, :query_pos]
                    peak_attn = attn_slice.max(dim=-1).values
                    entropy = self._batch_entropy(attn_slice)
                    entropy = torch.where(
                        peak_attn >= cfg.entropy_gate_threshold,
                        entropy, torch.zeros_like(entropy),
                    )
                    base = (n_layers - 1) * actual_heads
                    head_act_final[t, base:base + actual_heads] = peak_attn
                    head_ent_final[t, base:base + actual_heads] = entropy

        # --- Component E: Expert Routing ---
        n_experts = cfg.n_experts
        route_stable = torch.zeros(seq_len, n_experts, device=device)
        route_final = torch.zeros_like(route_stable)
        sorted_topk_weights = torch.zeros(
            seq_len, n_layers * cfg.top_k_experts, device=device
        )

        if expert_routing is not None:
            for t in range(seq_len):
                stable_layer = min(int(k_star[t].item()), n_layers - 1)
                route_stable[t] = expert_routing[stable_layer, t]
                route_final[t] = expert_routing[-1, t]

            # Sorted top-k weights at all layers (permutation-invariant)
            for ell in range(n_layers):
                topk_vals, _ = expert_routing[ell].topk(cfg.top_k_experts, dim=-1)
                # Sort descending for consistency
                topk_sorted, _ = topk_vals.sort(dim=-1, descending=True)
                base = ell * cfg.top_k_experts
                sorted_topk_weights[:, base:base + cfg.top_k_experts] = topk_sorted

        # --- Component F: Routing Entropy ---
        routing_entropy = torch.zeros(seq_len, n_layers, device=device)
        if expert_routing is not None:
            for ell in range(n_layers):
                weights = expert_routing[ell]  # [T, E]
                # Only compute entropy over nonzero entries (top-k routing)
                mask = weights > 1e-12
                safe_weights = weights.clamp(min=1e-12)
                ent = -(safe_weights * safe_weights.log() * mask.float()).sum(dim=-1)
                routing_entropy[:, ell] = ent

        # --- Component G: Attention Scale (sliding vs full contrast) ---
        local_fraction = torch.zeros(seq_len, n_layers, device=device)
        if layer_attentions is not None:
            for ell in range(n_layers):
                for t in range(seq_len):
                    query_pos = max(t - 1, 0)
                    if query_pos == 0:
                        local_fraction[t, ell] = 1.0
                        continue
                    attn = layer_attentions[ell, :, query_pos, :query_pos]
                    total = attn.sum()
                    if total < 1e-8:
                        local_fraction[t, ell] = 1.0
                        continue
                    window_start = max(0, query_pos - cfg.sliding_window)
                    local = attn[:, window_start:query_pos].sum()
                    local_fraction[t, ell] = (local / total).item()

        # --- Assemble feature vector ---
        components = []

        if cfg.include_trajectory:
            components.append(traj_probs.T)     # [T, L]
            components.append(traj_margin.T)    # [T, L]
            components.append(traj_drops.T)     # [T, L-1]

        if cfg.include_stability:
            components.append(k_star.unsqueeze(-1).float())  # [T, 1]
            components.append(kappa.unsqueeze(-1).float())   # [T, 1]

        if cfg.include_head_activation:
            components.append(head_act_stable)  # [T, L*H]
            components.append(head_act_final)   # [T, L*H]

        if cfg.include_head_entropy:
            components.append(head_ent_stable)  # [T, L*H]
            components.append(head_ent_final)   # [T, L*H]

        if cfg.include_expert_routing:
            components.append(route_stable)           # [T, E]
            components.append(route_final)            # [T, E]
            components.append(sorted_topk_weights)    # [T, L*top_k]

        if cfg.include_routing_entropy:
            components.append(routing_entropy)  # [T, L]

        if cfg.include_attention_scale:
            components.append(local_fraction)   # [T, L]

        feature_vectors = torch.cat(components, dim=-1)

        return FeatureResult(
            feature_vectors=feature_vectors,
            processing_depth=processing_depth,
            confidence=confidence,
            token_strings=token_strings or [],
            config=cfg,
            metadata={
                "n_layers": n_layers,
                "seq_len": seq_len,
                "has_attention": layer_attentions is not None,
                "has_routing": expert_routing is not None,
            },
        )

    @staticmethod
    def _batch_entropy(attn_slice: torch.Tensor) -> torch.Tensor:
        """Compute entropy for each head's attention distribution.

        Args:
            attn_slice: [n_heads, context_len] — attention weights for query.

        Returns:
            [n_heads] — entropy per head.
        """
        total = attn_slice.sum(dim=-1, keepdim=True)
        safe_total = total.clamp(min=1e-12)
        probs = attn_slice / safe_total
        log_probs = (probs + 1e-12).log()
        return -(probs * log_probs).sum(dim=-1)


###############################################################################
#
# Adapter: gpt-oss-interp backend → feature extractor
#
###############################################################################

def extract_features_from_backend(
    backend: Any,
    prompt: str,
    config: FeatureConfig | None = None,
) -> FeatureResult:
    """Extract feature vectors from a GPTOSSTransformersBackend.

    This is the primary entry point for feature extraction on gpt-oss-20b.
    It performs a single forward pass with hooks to capture:
    - Per-layer hidden states (projected through lm_head for logit-lens)
    - Per-layer attention weights (via attention hooks)
    - Expert routing weights (if available; requires non-quantized model)

    Args:
        backend: A GPTOSSTransformersBackend instance.
        prompt: The input prompt.
        config: Feature configuration. Defaults to gpt-oss-20b settings.

    Returns:
        FeatureResult with per-token feature vectors.
    """
    from gpt_oss_interp.capture.activation_cache import ActivationCache

    cfg = config or FeatureConfig()
    structure = backend.structure
    model = backend.model
    tokenizer = backend.tokenizer

    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(backend.device)
    seq_len = input_ids.shape[1]
    token_strings = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]

    # --- Capture hidden states for logit-lens ---
    cache = ActivationCache(detach=True, to_cpu=True)
    block_names = structure.block_names
    handles = cache.register(model, block_names)

    # --- Capture attention weights ---
    attn_weights: dict[int, torch.Tensor] = {}

    def make_attn_hook(layer_idx: int):
        def hook(module, input, output):
            # GptOssAttention returns (hidden_states, attn_weights, ...) when
            # output_attentions=True. We capture the attention weights.
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                attn_weights[layer_idx] = output[1].detach().cpu()
        return hook

    attn_handles = []
    for li, attn_name in enumerate(structure.attn_names):
        if not attn_name:
            continue
        module = dict(model.named_modules()).get(attn_name)
        if module is not None:
            h = module.register_forward_hook(make_attn_hook(li))
            attn_handles.append(h)

    try:
        with torch.no_grad():
            # Request attention weights from the model
            out = model(input_ids, output_attentions=True)
    finally:
        for h in handles:
            h.remove()
        for h in attn_handles:
            h.remove()

    # --- Build per-layer logits via logit-lens projection ---
    final_norm = structure.final_norm
    lm_head = structure.lm_head
    norm_device = next(final_norm.parameters()).device

    layer_logits_list = []
    for layer_idx, block_name in enumerate(block_names):
        record = cache.last(block_name)
        if record is None:
            continue
        hidden = record.tensor  # [1, seq_len, hidden_dim]
        with torch.no_grad():
            normed = final_norm(hidden.to(norm_device))
            logits = lm_head(normed).cpu().float()
        layer_logits_list.append(logits[0])  # [seq_len, vocab_size]

    layer_logits = torch.stack(layer_logits_list, dim=0)  # [L, T, V]

    # --- Build attention tensor ---
    layer_attentions = None
    if attn_weights:
        # Try to get from model output first (more reliable)
        if hasattr(out, 'attentions') and out.attentions is not None:
            attn_list = [a.cpu().float().squeeze(0) for a in out.attentions]
            layer_attentions = torch.stack(attn_list, dim=0)  # [L, H, T, T]
        elif attn_weights:
            # Fall back to hooked weights
            n_layers = len(block_names)
            n_heads = cfg.n_query_heads
            attn_tensor = torch.zeros(n_layers, n_heads, seq_len, seq_len)
            for li, w in attn_weights.items():
                if w.ndim == 4:
                    w = w.squeeze(0)
                actual_h = min(w.shape[0], n_heads)
                actual_t = min(w.shape[1], seq_len)
                attn_tensor[li, :actual_h, :actual_t, :actual_t] = w[:actual_h, :actual_t, :actual_t]
            layer_attentions = attn_tensor

    # --- Expert routing (may not be available under MXFP4) ---
    expert_routing = None
    if hasattr(out, 'router_logits') and out.router_logits is not None:
        route_list = []
        for rl in out.router_logits:
            if rl is not None:
                weights = F.softmax(rl.cpu().float(), dim=-1)
                route_list.append(weights.squeeze(0) if weights.ndim == 3 else weights)
        if route_list:
            expert_routing = torch.stack(route_list, dim=0)  # [L, T, E]

    # --- Extract features ---
    extractor = MoEFeatureExtractor(cfg)
    return extractor.extract(
        layer_logits=layer_logits,
        layer_attentions=layer_attentions,
        expert_routing=expert_routing,
        token_strings=token_strings,
    )
