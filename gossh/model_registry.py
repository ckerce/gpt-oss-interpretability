"""Model registry: architecture constants for supported gpt-oss models."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelArchSpec:
    """Architecture constants for a supported model."""

    model_id: str
    num_layers: int = 0
    hidden_dim: int = 0
    num_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    num_experts: int = 0
    top_k: int = 0
    quantization: str = "none"   # "mxfp4" | "bf16" | "none"
    vocab_size: int = 0
    max_seq_len: int = 0

    @classmethod
    def from_hf_config(cls, model_id: str) -> "ModelArchSpec":
        """Read architecture constants from HuggingFace config.json at runtime."""
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_id)
        return cls(
            model_id=model_id,
            num_layers=getattr(cfg, "num_hidden_layers", 0),
            hidden_dim=getattr(cfg, "hidden_size", 0),
            num_heads=getattr(cfg, "num_attention_heads", 0),
            num_kv_heads=getattr(cfg, "num_key_value_heads", 0),
            head_dim=getattr(cfg, "head_dim", 0),
            num_experts=getattr(cfg, "num_local_experts", 0),
            top_k=getattr(cfg, "num_experts_per_tok", 0),
            quantization="mxfp4",   # assumed for gpt-oss family
            vocab_size=getattr(cfg, "vocab_size", 0),
            max_seq_len=getattr(cfg, "max_position_embeddings", 0),
        )


# ── Known models ──────────────────────────────────────────────────────────────

_REGISTRY: dict[str, ModelArchSpec] = {
    "openai/gpt-oss-20b": ModelArchSpec(
        model_id="openai/gpt-oss-20b",
        num_layers=24,
        hidden_dim=4096,
        num_heads=64,
        num_kv_heads=8,
        head_dim=64,
        num_experts=32,
        top_k=4,
        quantization="mxfp4",
        vocab_size=200019,
        max_seq_len=8192,
    ),
    # 120b constants will be populated from config.json at first use
    "openai/gpt-oss-120b": ModelArchSpec(
        model_id="openai/gpt-oss-120b",
        quantization="mxfp4",
    ),
}


def get_arch_spec(model_id: str) -> ModelArchSpec:
    """Return the ModelArchSpec for a model, falling back to HF config."""
    if model_id in _REGISTRY:
        spec = _REGISTRY[model_id]
        if spec.num_layers == 0:
            # Stub entry — populate from HuggingFace config.json
            spec = ModelArchSpec.from_hf_config(model_id)
            _REGISTRY[model_id] = spec
        return spec
    # Unknown model — try HuggingFace
    spec = ModelArchSpec.from_hf_config(model_id)
    _REGISTRY[model_id] = spec
    return spec


def list_supported_models() -> list[str]:
    """Return model IDs with known architecture constants."""
    return [k for k, v in _REGISTRY.items() if v.num_layers > 0]
