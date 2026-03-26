"""HuggingFace transformers backend for gpt-oss models."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn

from gossh.backends.base import BaseBackend, BackendScore
from gossh.backends.structure import ModelStructure
from gossh.capture.router_capture import RouterCapture, RouterDecision
from gossh.config import InterventionKind, InterventionSpec, PromptCase, TargetUnit
from gossh.interventions.hooks import (
    expert_output_scale_hook,
    head_mask_hook,
    layer_scale_hook,
    temperature_hook,
)
from gossh.model_registry import get_arch_spec

if TYPE_CHECKING:
    from gossh.sidecar.process import MoeSidecar


class GPTOSSTransformersBackend(BaseBackend):
    """Backend for gpt-oss models via HuggingFace transformers.

    Architecture constants are read from the model registry (``get_arch_spec``)
    rather than being hardcoded — the same backend class works for both
    gpt-oss-20b and gpt-oss-120b.

    Supports:
    - Choice logprob scoring with harmony chat template
    - Head masking via attention output hooks
    - Expert masking via MoE gate hooks (post-kernel approximation for MXFP4)
    - Layer scaling via block output hooks
    - Temperature scaling via attention hooks
    - Activation capture at block boundaries
    - MoE router decision capture (non-quantized or output_router_logits path)
    - Per-layer logit-lens readouts
    """

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

        self._arch = get_arch_spec(model_name)
        self._load_model(dtype)
        self.structure = ModelStructure(self.model)
        self._sidecar: Optional["MoeSidecar"] = None
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

    # ── Sidecar management ────────────────────────────────────────────────────

    def attach_sidecar(self, sidecar: "MoeSidecar") -> None:
        """Attach a running MoeSidecar for MXFP4-safe routing capture (path 0).

        The sidecar must already be started (i.e. used as a context manager or
        ``sidecar.start()`` called).  Call ``detach_sidecar()`` before stopping it.
        """
        if not sidecar.is_running():
            raise RuntimeError("Sidecar is not running — start it before attaching.")
        self._sidecar = sidecar

    def detach_sidecar(self) -> None:
        """Remove the sidecar reference; routing falls back to paths 1–3."""
        self._sidecar = None

    def _capture_routing_via_sidecar(self, input_ids: torch.Tensor) -> list[RouterDecision]:
        """Path 0: capture routing via the MoE sidecar subprocess.

        Captures hidden states entering each MoE/MLP block via pre-hooks
        (which fire before the fused MXFP4 kernel), then sends them to the
        sidecar for routing.
        """
        from gossh.capture.input_cache import InputCapture

        capture = InputCapture()
        capture.register(self.model, self.structure.mlp_names)
        try:
            with torch.no_grad():
                self.model(input_ids)
        finally:
            capture.remove_hooks()

        if not capture.captured:
            return []
        return self._sidecar.route(capture.captured)  # type: ignore[union-attr]

    ###########################################################################
    # Scoring
    ###########################################################################

    def _encode_prompt_with_completion(self, prompt: str, choice_text: str) -> tuple[list[int], int]:
        from gpt_oss_interp.harmony.prompting import encode_prompt_with_completion
        return encode_prompt_with_completion(self.tokenizer, prompt, choice_text)

    def _choice_logprob(self, prompt: str, choice_text: str) -> float:
        full_ids, choice_start = self._encode_prompt_with_completion(prompt, choice_text)
        input_ids = torch.tensor([full_ids], device=self.device)

        with torch.no_grad():
            logits = self.model(input_ids).logits
            log_probs = torch.log_softmax(logits[0].float(), dim=-1)

        total = 0.0
        for i in range(choice_start, len(full_ids)):
            total += log_probs[i - 1, full_ids[i]].item()
        return total

    def _choice_logprob_by_layer(self, prompt: str, choice_text: str) -> dict[int, float]:
        from gossh.capture.activation_cache import ActivationCache

        full_ids, choice_start = self._encode_prompt_with_completion(prompt, choice_text)
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
                total += log_probs[i - 1, full_ids[i]].item()
            scores[layer_idx] = total
        return scores

    def score_case(self, case: PromptCase) -> BackendScore:
        choice_logprobs: dict[str, float] = {}
        for label, text in case.choices.items():
            choice_logprobs[label] = self._choice_logprob(case.prompt, text)
        return BackendScore(
            choice_logprobs=choice_logprobs,
            metadata={"backend": "gpt_oss_transformers", "model": self.model_name},
        )

    def score_case_by_layer(self, case: PromptCase) -> dict[int, dict[str, float]]:
        layer_scores: dict[int, dict[str, float]] = {}
        for label, text in case.choices.items():
            for layer_idx, score in self._choice_logprob_by_layer(case.prompt, text).items():
                layer_scores.setdefault(layer_idx, {})[label] = score
        return layer_scores

    ###########################################################################
    # Interventions
    ###########################################################################

    def apply_intervention(self, spec: InterventionSpec, scale: float) -> None:
        arch = self._arch
        target = spec.target
        named = dict(self.model.named_modules())

        if spec.kind == InterventionKind.HEAD_MASK:
            layers = target.layer_indices if target.layer_indices else range(self.structure.num_layers)
            for li in layers:
                attn_name = self.structure.attn_names[li]
                if not attn_name:
                    continue
                module = named.get(attn_name)
                if module is None:
                    continue
                hook = module.register_forward_hook(
                    head_mask_hook(target.head_indices, scale, arch.num_heads, arch.head_dim)
                )
                self._hooks.append(hook)

        elif spec.kind == InterventionKind.EXPERT_MASK:
            layers = target.layer_indices if target.layer_indices else range(self.structure.num_layers)
            for li in layers:
                mlp_name = self.structure.mlp_names[li]
                if not mlp_name:
                    continue
                module = named.get(mlp_name)
                if module is None:
                    continue
                hook = module.register_forward_hook(
                    expert_output_scale_hook(target.expert_indices, scale, arch.num_experts)
                )
                self._hooks.append(hook)

        elif spec.kind == InterventionKind.LAYER_SCALE:
            layers = target.layer_indices if target.layer_indices else range(self.structure.num_layers)
            preserve_residual = spec.params.get("preserve_residual", True)
            for li in layers:
                block_name = self.structure.block_names[li]
                module = named.get(block_name)
                if module is None:
                    continue
                hook = module.register_forward_hook(
                    layer_scale_hook(scale, preserve_residual=preserve_residual)
                )
                self._hooks.append(hook)

        elif spec.kind == InterventionKind.TEMPERATURE_SCALE:
            if target.unit == TargetUnit.MODEL:
                attn_iter = self.structure.attn_names
            else:
                layers = target.layer_indices if target.layer_indices else range(self.structure.num_layers)
                attn_iter = [self.structure.attn_names[li] for li in layers]
            for attn_name in attn_iter:
                if not attn_name:
                    continue
                module = named.get(attn_name)
                if module is None:
                    continue
                hook = module.register_forward_hook(temperature_hook(scale))
                self._hooks.append(hook)

    def clear_interventions(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    ###########################################################################
    # Introspection utilities
    ###########################################################################

    def run_logit_lens(
        self,
        prompt: str,
        top_k: int = 5,
        target_ids: torch.Tensor | None = None,
        positions: list[int] | None = None,
    ):
        """Run a logit-lens pass on a prompt and return per-layer predictions."""
        from gossh.readouts.logit_lens import run_logit_lens

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
        )

    def capture_activations(self, prompt: str, layer_indices: list[int] | None = None):
        """Run a forward pass and return activation records for specified layers."""
        from gossh.capture.activation_cache import ActivationCache

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

        Routing capture is attempted in priority order:

        - **Path 0 (sidecar)**: If a ``MoeSidecar`` is attached, capture hidden
          states via pre-hooks and route through the sidecar's bf16 gate clones.
          This is the only path that works correctly under MXFP4 quantization.
        - **Path 1 (output_router_logits)**: Ask the model to return router
          logits directly.  Works for non-MXFP4 checkpoints.
        - **Path 2 (hook-based)**: Register forward hooks on gate modules.
          Works for non-quantized checkpoints where the gate is a Python module.
        - **Path 3 (unavailable)**: MXFP4 fused kernels — emit an informative
          message and return an empty list.
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # Path 0: sidecar (MXFP4-safe)
        if self._sidecar is not None:
            return self._capture_routing_via_sidecar(input_ids)

        with torch.no_grad():
            out = self.model(input_ids, output_router_logits=True)

        if out.router_logits is not None:
            decisions = []
            logits_list = out.router_logits
            if isinstance(logits_list, (list, tuple)):
                for idx, rl in enumerate(logits_list):
                    if rl is None:
                        continue
                    rl_cpu = rl.detach().cpu().float()
                    weights, indices = torch.topk(
                        torch.softmax(rl_cpu, dim=-1), k=self._arch.top_k, dim=-1,
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

        # Fallback: hook-based capture (non-quantized models)
        capture = RouterCapture(top_k=self._arch.top_k)
        gate_names = [n for n in self.structure.gate_names if n]
        if not gate_names:
            print(
                "Router capture unavailable: MXFP4 fused kernels bypass Python hooks "
                "and do not expose router_logits. Use a non-quantized checkpoint for "
                "router introspection, or use the MoE sidecar (gossh.sidecar)."
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
                "Router capture unavailable: MXFP4 fused kernels bypass Python hooks. "
                "Use a non-quantized checkpoint or the MoE sidecar (gossh.sidecar)."
            )
        return capture.decisions
