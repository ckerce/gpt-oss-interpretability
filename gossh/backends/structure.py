"""Model structure discovery for GOSSH hook registration."""
from __future__ import annotations

import torch.nn as nn


def _find_child(parent: nn.Module, candidates: list[str]) -> nn.Module | None:
    """Return the first named child that exists."""
    for name in candidates:
        parts = name.split(".")
        mod = parent
        for p in parts:
            mod = getattr(mod, p, None)
            if mod is None:
                break
        if mod is not None:
            return mod
    return None


def _find_child_name(parent_name: str, parent: nn.Module, candidates: list[str]) -> str | None:
    """Return the fully-qualified name of the first matching child."""
    for name in candidates:
        parts = name.split(".")
        mod = parent
        for p in parts:
            mod = getattr(mod, p, None)
            if mod is None:
                break
        if mod is not None:
            return f"{parent_name}.{name}" if parent_name else name
    return None


class ModelStructure:
    """Discovered model structure for hook registration.

    Walks a loaded HuggingFace model and discovers the canonical names for
    transformer blocks, attention modules, MoE/MLP modules, router/gate
    modules, final norm, LM head, and token embedding.  All path discovery
    is done once at construction; downstream code should index into the
    ``block_names``, ``attn_names``, ``mlp_names``, and ``gate_names`` lists.
    """

    def __init__(self, model: nn.Module):
        named = dict(model.named_modules())
        self.blocks: list[nn.Module] = []
        self.block_names: list[str] = []
        self.attn_names: list[str] = []
        self.mlp_names: list[str] = []
        self.gate_names: list[str] = []
        self.final_norm: nn.Module | None = None
        self.lm_head: nn.Module | None = None
        self.embed: nn.Module | None = None

        # Discover the transformer block list
        container_prefix = ""
        for prefix in ["model.model.layers", "model.layers", "transformer.h", "model.transformer.h"]:
            if f"{prefix}.0" in named:
                container_prefix = prefix
                break

        if not container_prefix:
            raise RuntimeError(
                "Could not discover transformer blocks. "
                "Run scripts/inspect_model.py to examine the model structure."
            )

        idx = 0
        while f"{container_prefix}.{idx}" in named:
            block_name = f"{container_prefix}.{idx}"
            block = named[block_name]
            self.blocks.append(block)
            self.block_names.append(block_name)

            attn_name = _find_child_name(
                block_name, block,
                ["self_attn", "attn", "attention"],
            )
            self.attn_names.append(attn_name or "")

            mlp_name = _find_child_name(
                block_name, block,
                ["block_sparse_moe", "sparse_moe", "moe", "mlp"],
            )
            self.mlp_names.append(mlp_name or "")

            if mlp_name:
                mlp_mod = named.get(mlp_name)
                if mlp_mod is not None:
                    gate_name = _find_child_name(
                        mlp_name, mlp_mod,
                        ["gate", "router", "gate_proj"],
                    )
                    self.gate_names.append(gate_name or "")
                else:
                    self.gate_names.append("")
            else:
                self.gate_names.append("")

            idx += 1

        for name in ["model.model.norm", "model.norm", "transformer.ln_f", "model.transformer.ln_f"]:
            if name in named:
                self.final_norm = named[name]
                break

        for name in ["lm_head", "model.lm_head"]:
            if name in named:
                self.lm_head = named[name]
                break

        for name in ["model.model.embed_tokens", "model.embed_tokens", "transformer.wte", "model.transformer.wte"]:
            if name in named:
                self.embed = named[name]
                break

    @property
    def num_layers(self) -> int:
        return len(self.blocks)

    def summary(self) -> str:
        lines = [f"Discovered {self.num_layers} transformer blocks"]
        if self.blocks:
            lines.append(f"  block pattern : {self.block_names[0]}")
            lines.append(f"  attn pattern  : {self.attn_names[0]}")
            lines.append(f"  mlp pattern   : {self.mlp_names[0]}")
            lines.append(f"  gate pattern  : {self.gate_names[0]}")
        lines.append(f"  final norm    : {'found' if self.final_norm else 'NOT FOUND'}")
        lines.append(f"  lm_head       : {'found' if self.lm_head else 'NOT FOUND'}")
        lines.append(f"  embed         : {'found' if self.embed else 'NOT FOUND'}")
        return "\n".join(lines)
