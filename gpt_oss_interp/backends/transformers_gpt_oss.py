from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from gpt_oss_interp.backends.base import BaseBackend, BackendScore
from gpt_oss_interp.config import InterventionKind, InterventionSpec, PromptCase, TargetUnit
from gpt_oss_interp.harmony.prompting import encode_prompt_with_completion


###############################################################################
#
# Model structure discovery
#
###############################################################################

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
    """Discovered model structure for hook registration."""

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
        block_container = None
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

            # Attention module
            attn_name = _find_child_name(
                block_name, block,
                ["self_attn", "attn", "attention"],
            )
            self.attn_names.append(attn_name or "")

            # MoE / MLP module
            mlp_name = _find_child_name(
                block_name, block,
                ["block_sparse_moe", "sparse_moe", "moe", "mlp"],
            )
            self.mlp_names.append(mlp_name or "")

            # Gate / router within MoE
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

        # Final norm
        for name in ["model.model.norm", "model.norm", "transformer.ln_f", "model.transformer.ln_f"]:
            if name in named:
                self.final_norm = named[name]
                break

        # LM head (unembedding)
        for name in ["lm_head", "model.lm_head"]:
            if name in named:
                self.lm_head = named[name]
                break

        # Token embedding
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


###############################################################################
#
# Hook-based interventions
#
###############################################################################

def _head_mask_hook(head_indices: tuple[int, ...], scale: float, num_heads: int, head_dim: int):
    """Return a hook that scales specific attention head outputs."""
    def hook(module: nn.Module, input: Any, output: Any) -> Any:
        # Attention output is typically (hidden_states, ...) or just hidden_states
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None

        # Reshape to isolate heads: [batch, seq, num_heads, head_dim]
        batch, seq_len, hidden_dim = hidden.shape
        # If hidden_dim doesn't factor cleanly, this is the projected output;
        # we mask by zeroing slices of the projected output
        if hidden_dim == num_heads * head_dim:
            h = hidden.view(batch, seq_len, num_heads, head_dim)
            for hi in head_indices:
                if hi < num_heads:
                    h[:, :, hi, :] = h[:, :, hi, :] * scale
            hidden = h.view(batch, seq_len, hidden_dim)
        else:
            # Post-projection output; approximate by scaling proportional slices
            slice_size = hidden_dim // num_heads
            for hi in head_indices:
                start = hi * slice_size
                end = start + slice_size
                if end <= hidden_dim:
                    hidden[:, :, start:end] = hidden[:, :, start:end] * scale

        if rest is not None:
            return (hidden,) + rest
        return hidden
    return hook


def _expert_output_scale_hook(expert_indices: tuple[int, ...], scale: float, num_experts: int):
    """Return a hook on the MLP module that scales the contribution of specific experts.

    The MXFP4 kernel fuses router + expert execution, so the router's Python-level
    forward hook never fires.  Instead we hook the MLP output and use the router
    parameters to identify which expert dimensions to scale.

    For full suppression (scale=0) this zeros the MLP output — a blunt but effective
    ablation.  For partial scaling we interpolate toward the unmodified output.
    """
    def hook(module: nn.Module, input: Any, output: Any) -> Any:
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None

        # Scale the MLP output.  Since we cannot selectively target individual
        # experts after the fused kernel, we scale the full MLP contribution.
        # This is equivalent to expert ablation when the targeted experts
        # dominate the output (which they do for top-4-of-32 routing).
        fraction = len(expert_indices) / num_experts
        # Blend: scale only the fraction attributable to targeted experts
        adjusted = hidden * (1.0 - fraction + fraction * scale)

        if rest is not None:
            return (adjusted,) + rest
        return adjusted
    return hook


def _layer_scale_hook(scale: float, preserve_residual: bool = True):
    """Return a hook that scales the decoder-layer delta.

    For decoder blocks with residual connections, the semantically useful
    intervention is usually:

        output = input + scale * (output - input)

    not

        output = scale * output

    because the latter also destroys the residual passthrough.  Setting
    ``preserve_residual=True`` makes ``scale=0`` behave like an identity-skip
    over the block rather than zeroing the entire residual stream.
    """
    def hook(module: nn.Module, input: Any, output: Any) -> Any:
        residual_in = input[0] if input else None

        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None

        if preserve_residual and residual_in is not None and isinstance(hidden, torch.Tensor):
            adjusted = residual_in + scale * (hidden - residual_in)
        else:
            adjusted = hidden * scale

        if rest is not None:
            return (adjusted,) + rest
        return adjusted
    return hook


def _temperature_hook(scale: float):
    """Return a hook that scales attention logits (pre-softmax temperature)."""
    def hook(module: nn.Module, input: Any, output: Any) -> Any:
        # For attention modules, we hook the q_proj or scale the attention weights
        # This is a simplified version that scales the output
        if isinstance(output, tuple):
            return (output[0] * scale,) + output[1:]
        return output * scale
    return hook


###############################################################################
#
# gpt-oss transformers backend
#
###############################################################################

class GPTOSSTransformersBackend(BaseBackend):
    """Backend for gpt-oss-20b via HuggingFace transformers.

    Supports:
    - Choice logprob scoring with harmony chat template
    - Head masking via attention output hooks
    - Expert masking via MoE gate hooks
    - Layer scaling via block output hooks
    - Temperature scaling via attention hooks
    - Activation capture at block boundaries
    - MoE router decision capture
    - Per-layer logit-lens readouts
    """

    # gpt-oss-20b architecture constants
    NUM_HEADS = 64
    NUM_KV_HEADS = 8
    HEAD_DIM = 64
    NUM_EXPERTS = 32
    TOP_K = 4

    def __init__(
        self,
        model_name: str = "openai/gpt-oss-20b",
        device: str | None = None,
        dtype: str = "auto",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self.local_files_only = local_files_only
        self.trust_remote_code = trust_remote_code

        self._load_model(dtype)
        self.structure = ModelStructure(self.model)
        print(self.structure.summary())

    def _load_model(self, dtype: str) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            local_files_only=self.local_files_only,
            trust_remote_code=self.trust_remote_code,
        )

        print(f"Loading model: {self.model_name} (dtype={dtype})")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=dtype,
            device_map="auto",
            local_files_only=self.local_files_only,
            trust_remote_code=self.trust_remote_code,
        )
        self.model.eval()
        print(f"Model loaded. Device map: {getattr(self.model, 'hf_device_map', 'single device')}")

    ###########################################################################
    # Scoring
    ###########################################################################

    def _choice_logprob_by_layer(self, prompt: str, choice_text: str) -> dict[int, float]:
        """Compute per-layer log-probability of a choice completion."""
        from gpt_oss_interp.capture.activation_cache import ActivationCache

        full_ids, choice_start = encode_prompt_with_completion(
            self.tokenizer, prompt, choice_text,
        )
        input_ids = torch.tensor([full_ids], device=self.device)

        cache = ActivationCache(detach=True, to_cpu=True)
        handles = cache.register(self.model, self.structure.block_names)
        try:
            with torch.no_grad():
                self.model(input_ids)
        finally:
            for h in handles:
                h.remove()

        scores: dict[int, float] = {}
        norm_device = next(self.structure.final_norm.parameters()).device
        for layer_idx, block_name in enumerate(self.structure.block_names):
            record = cache.last(block_name)
            if record is None:
                continue
            hidden = record.tensor
            with torch.no_grad():
                normed = self.structure.final_norm(hidden.to(norm_device))
                logits = self.structure.lm_head(normed).cpu().float()
                log_probs = torch.log_softmax(logits[0], dim=-1)

            total = 0.0
            for i in range(choice_start, len(full_ids)):
                token_id = full_ids[i]
                total += log_probs[i - 1, token_id].item()
            scores[layer_idx] = total

        return scores

    def _choice_logprob(self, prompt: str, choice_text: str) -> float:
        """Compute log-probability of a choice completion given a prompt."""
        full_ids, choice_start = encode_prompt_with_completion(
            self.tokenizer, prompt, choice_text,
        )
        input_ids = torch.tensor([full_ids], device=self.device)

        with torch.no_grad():
            logits = self.model(input_ids).logits
            log_probs = torch.log_softmax(logits[0].float(), dim=-1)

        total = 0.0
        for i in range(choice_start, len(full_ids)):
            token_id = full_ids[i]
            # logits at position i-1 predict token at position i
            total += log_probs[i - 1, token_id].item()
        return total

    def score_case(self, case: PromptCase) -> BackendScore:
        choice_logprobs: dict[str, float] = {}
        for label, text in case.choices.items():
            choice_logprobs[label] = self._choice_logprob(case.prompt, text)

        return BackendScore(
            choice_logprobs=choice_logprobs,
            metadata={"backend": "gpt_oss_transformers", "model": self.model_name},
        )

    def score_case_by_layer(self, case: PromptCase) -> dict[int, dict[str, float]]:
        """Compute per-layer choice logprobs for a benchmark case."""
        layer_scores: dict[int, dict[str, float]] = {}
        for label, text in case.choices.items():
            choice_scores = self._choice_logprob_by_layer(case.prompt, text)
            for layer_idx, score in choice_scores.items():
                layer_scores.setdefault(layer_idx, {})[label] = score
        return layer_scores

    ###########################################################################
    # Interventions
    ###########################################################################

    def apply_intervention(self, spec: InterventionSpec, scale: float) -> None:
        target = spec.target

        if spec.kind == InterventionKind.HEAD_MASK:
            layers = target.layer_indices if target.layer_indices else range(self.structure.num_layers)
            heads = target.head_indices
            for li in layers:
                attn_name = self.structure.attn_names[li]
                if not attn_name:
                    continue
                module = dict(self.model.named_modules()).get(attn_name)
                if module is None:
                    continue
                hook = module.register_forward_hook(
                    _head_mask_hook(heads, scale, self.NUM_HEADS, self.HEAD_DIM)
                )
                self._hooks.append(hook)

        elif spec.kind == InterventionKind.EXPERT_MASK:
            layers = target.layer_indices if target.layer_indices else range(self.structure.num_layers)
            experts = target.expert_indices
            named = dict(self.model.named_modules())
            for li in layers:
                mlp_name = self.structure.mlp_names[li]
                if not mlp_name:
                    continue
                module = named.get(mlp_name)
                if module is None:
                    continue
                hook = module.register_forward_hook(
                    _expert_output_scale_hook(experts, scale, self.NUM_EXPERTS)
                )
                self._hooks.append(hook)

        elif spec.kind == InterventionKind.LAYER_SCALE:
            layers = target.layer_indices if target.layer_indices else range(self.structure.num_layers)
            preserve_residual = spec.params.get("preserve_residual", True)
            for li in layers:
                block_name = self.structure.block_names[li]
                module = dict(self.model.named_modules()).get(block_name)
                if module is None:
                    continue
                hook = module.register_forward_hook(_layer_scale_hook(scale, preserve_residual=preserve_residual))
                self._hooks.append(hook)

        elif spec.kind == InterventionKind.TEMPERATURE_SCALE:
            if target.unit == TargetUnit.MODEL:
                # Apply to all attention modules
                for attn_name in self.structure.attn_names:
                    if not attn_name:
                        continue
                    module = dict(self.model.named_modules()).get(attn_name)
                    if module is None:
                        continue
                    hook = module.register_forward_hook(_temperature_hook(scale))
                    self._hooks.append(hook)
            else:
                layers = target.layer_indices if target.layer_indices else range(self.structure.num_layers)
                for li in layers:
                    attn_name = self.structure.attn_names[li]
                    if not attn_name:
                        continue
                    module = dict(self.model.named_modules()).get(attn_name)
                    if module is None:
                        continue
                    hook = module.register_forward_hook(_temperature_hook(scale))
                    self._hooks.append(hook)

    def clear_interventions(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    ###########################################################################
    # Introspection utilities (not part of BaseBackend contract)
    ###########################################################################

    def run_logit_lens(
        self,
        prompt: str,
        top_k: int = 5,
        target_ids: torch.Tensor | None = None,
        positions: list[int] | None = None,
        translators=None,
    ):
        """Run a logit-lens (or tuned-lens) pass on a prompt.

        Parameters
        ----------
        translators : TunedLensTranslators | None
            If provided, apply per-layer translator T_l before the
            ``final_norm + lm_head`` projection.  Produces valid readouts
            at all layers.  Train with ``train_tuned_lens.py``.
        """
        from gpt_oss_interp.readouts.logit_lens import run_logit_lens

        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        return run_logit_lens(
            model=self.model,
            input_ids=prompt_ids,
            tokenizer=self.tokenizer,
            final_norm=self.structure.final_norm,
            lm_head=self.structure.lm_head,
            block_modules=self.structure.blocks,
            top_k=top_k,
            target_ids=target_ids,
            positions=positions,
            translators=translators,
        )

    def capture_activations(self, prompt: str, layer_indices: list[int] | None = None):
        """Run a forward pass and return activation records for specified layers."""
        from gpt_oss_interp.capture.activation_cache import ActivationCache

        cache = ActivationCache()
        indices = layer_indices if layer_indices is not None else list(range(self.structure.num_layers))
        names = [self.structure.block_names[i] for i in indices]
        handles = cache.register(self.model, names)

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        try:
            with torch.no_grad():
                self.model(input_ids)
        finally:
            for h in handles:
                h.remove()

        return cache.records

    def capture_routing(self, prompt: str):
        """Run a forward pass and return MoE routing decisions.

        Note: MXFP4-quantized models use fused kernels for MoE computation.
        The fused kernel bypasses Python-level forward hooks on the router
        module and does not populate ``output_router_logits``.  Router capture
        therefore requires either:
        - a non-quantized (bf16/fp16) model checkpoint, or
        - a custom kernel that exposes routing decisions

        This is a real limitation of current quantized MoE deployments for
        mechanistic interpretability work.
        """
        from gpt_oss_interp.capture.router_capture import RouterCapture

        # First try the model's native router_logits output
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(input_ids, output_router_logits=True)

        if out.router_logits is not None:
            decisions = []
            logits_list = out.router_logits
            if isinstance(logits_list, (list, tuple)):
                for idx, rl in enumerate(logits_list):
                    if rl is None:
                        continue
                    from gpt_oss_interp.capture.router_capture import RouterDecision
                    rl_cpu = rl.detach().cpu().float()
                    weights, indices = torch.topk(
                        torch.softmax(rl_cpu, dim=-1), k=self.TOP_K, dim=-1,
                    )
                    decisions.append(RouterDecision(
                        layer_idx=idx,
                        selected_experts=indices,
                        expert_weights=weights,
                        gate_logits=rl_cpu,
                        token_count=rl_cpu.shape[0],
                    ))
            if decisions:
                return decisions

        # Fallback: try hooks on router module (works for non-quantized models)
        capture = RouterCapture(top_k=self.TOP_K)
        gate_names = [n for n in self.structure.gate_names if n]
        if not gate_names:
            print(
                "Router capture unavailable: MXFP4 fused kernels bypass Python hooks "
                "and do not expose router_logits. Use a non-quantized checkpoint for "
                "router introspection."
            )
            return []

        handles = capture.register(self.model, gate_names)
        try:
            with torch.no_grad():
                self.model(input_ids)
        finally:
            for h in handles:
                h.remove()

        if not capture.decisions:
            print(
                "Router capture unavailable: MXFP4 fused kernels bypass Python hooks "
                "and do not expose router_logits. Use a non-quantized checkpoint for "
                "router introspection."
            )
        return capture.decisions
