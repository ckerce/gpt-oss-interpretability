"""Tuned lens for gpt-oss-20b.

The logit lens (nostalgebraist 2020; Belrose et al. 2023) applies
``final_norm + lm_head`` to intermediate hidden states to produce a
per-layer vocabulary prediction.  It is *free* — no training required —
but empirically valid only in the last few layers of most models (L21+
for gpt-oss-20b; see runs/unembedding_validation/).

The **tuned lens** (Belrose et al. 2023) trains a per-layer affine
translator T_l such that::

    lm_head(final_norm(T_l(h_l)))

closely approximates the final-layer distribution at every depth.
With translators trained, the logit lens becomes valid at all 24 layers.

This reveals *when* the model's hidden state first encodes the answer —
as opposed to when that encoding happens to be in output-space geometry.
For gpt-oss-20b the two are separated by ~8 layers: the answer is encoded
around L12 (MI peak, Thread 15) but only becomes logit-lens-readable at L21.

Architecture note — DST motivation
-----------------------------------
Training per-layer translators is a post-hoc correction for an
architectural choice: hidden states are free to occupy any geometry,
and output-space alignment is only guaranteed at the final layer.

The Dual-Stream PLS Transformer (DST, Threads 13–14) inverts this
by construction: the PLS decomposition enforces that every layer's
hidden state lives in a geometry that is interpretable without a
trained corrector.  The tuned lens is the empirical measure of the
*gap* between a standard transformer and a readout-ready architecture;
training it quantifies what the DST design choice buys.

References
----------
Belrose et al. 2023 — "Eliciting Latent Predictions from Transformers
with the Tuned Lens"  https://arxiv.org/abs/2303.08112
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Translator module
# ---------------------------------------------------------------------------

class TunedLensTranslators(nn.Module):
    """Per-layer low-rank residual translators.

    Each translator maps hidden state h_l toward a "final-layer-like"
    geometry so that ``final_norm + lm_head`` produces a meaningful
    vocabulary distribution at any depth.

    Parameterisation (residual low-rank + bias)::

        T_l(h) = h  +  U_l @ (V_l.T @ h)  +  b_l

    where ``U_l, V_l ∈ R^{hidden_dim × rank}`` and ``b_l ∈ R^{hidden_dim}``.
    Initialised as the identity (U=0, V=0, b=0).

    Parameters
    ----------
    hidden_dim : int
        Transformer hidden dimension (2880 for gpt-oss-20b).
    n_layers : int
        Number of transformer blocks (24 for gpt-oss-20b).
    rank : int
        Low-rank bottleneck dimension.  rank=32 gives ~4.5M total params
        for gpt-oss-20b — trivial to train.
    """

    def __init__(self, hidden_dim: int, n_layers: int, rank: int = 32) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rank = rank

        # U and V are initialised to zero so T_l starts as identity
        self.U = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden_dim, rank)) for _ in range(n_layers)
        ])
        self.V = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden_dim, rank)) for _ in range(n_layers)
        ])
        self.b = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden_dim)) for _ in range(n_layers)
        ])

    def translate(self, hidden: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Apply T_{layer_idx} to hidden states.

        Parameters
        ----------
        hidden : torch.Tensor
            Shape ``[..., hidden_dim]``.
        layer_idx : int

        Returns
        -------
        torch.Tensor
            Same shape as *hidden*.
        """
        U = self.U[layer_idx]   # [hidden_dim, rank]
        V = self.V[layer_idx]   # [hidden_dim, rank]
        b = self.b[layer_idx]   # [hidden_dim]
        projected = hidden @ V  # [..., rank]
        correction = projected @ U.T  # [..., hidden_dim]
        return hidden + correction + b

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "rank": self.rank,
        }
        torch.save({"meta": meta, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu") -> "TunedLensTranslators":
        payload = torch.load(path, map_location=map_location, weights_only=False)
        meta = payload["meta"]
        obj = cls(
            hidden_dim=meta["hidden_dim"],
            n_layers=meta["n_layers"],
            rank=meta["rank"],
        )
        obj.load_state_dict(payload["state_dict"])
        return obj


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_tuned_lens(
    backend: Any,
    prompts: Sequence[str],
    *,
    rank: int = 32,
    n_epochs: int = 3,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    device: str | None = None,
    verbose: bool = True,
) -> TunedLensTranslators:
    """Train per-layer translators on a corpus of prompts.

    The model is frozen.  For each prompt, hidden states at all layers
    are collected in a single forward pass.  The translators are then
    updated to minimise KL divergence from the final layer's vocabulary
    distribution.

    Training objective (per token position, per layer l)::

        L_l = KL( softmax(lm_head(final_norm(T_l(h_l)))) || P_L )

    where P_L is the final layer's distribution (target, detached).

    Parameters
    ----------
    backend : GPTOSSTransformersBackend
    prompts : sequence of str
        Training corpus.  The Thread 15 task suite (135 prompts) works
        for a first pass; more diverse text improves generalisation.
    rank : int
        Translator rank.
    n_epochs : int
        Training epochs over the corpus.
    lr : float
        Adam learning rate.
    verbose : bool
        Print per-epoch loss.

    Returns
    -------
    TunedLensTranslators
        Trained translator set, moved to CPU.
    """
    from gpt_oss_interp.capture.activation_cache import ActivationCache

    model = backend.model
    structure = backend.structure
    tokenizer = backend.tokenizer
    model_device = next(model.parameters()).device
    work_device = torch.device(device) if device else model_device

    hidden_dim = model.config.hidden_size
    n_layers = len(structure.block_names)

    translators = TunedLensTranslators(hidden_dim=hidden_dim, n_layers=n_layers, rank=rank)
    translators = translators.to(work_device)

    optimizer = torch.optim.Adam(
        translators.parameters(), lr=lr, weight_decay=weight_decay
    )

    norm = structure.final_norm
    lm_head = structure.lm_head

    total_steps = 0
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        for prompt in prompts:
            # --- collect hidden states (model frozen, no grad) --------
            ids = tokenizer.encode(prompt, add_special_tokens=True)
            if len(ids) < 2:
                continue
            input_ids = torch.tensor([ids], device=model_device)

            cache = ActivationCache(detach=True, to_cpu=False)
            handles = cache.register(model, structure.block_names)
            try:
                with torch.no_grad():
                    model(input_ids)
            finally:
                for h in handles:
                    h.remove()

            # Gather hidden states [n_layers, seq_len, hidden_dim]
            hiddens: list[torch.Tensor] = []
            for name in structure.block_names:
                rec = cache.last(name)
                if rec is None:
                    continue
                hiddens.append(rec.tensor[0].to(work_device))  # [seq_len, hidden_dim]

            if len(hiddens) != n_layers:
                continue

            # Final layer target distribution (detached)
            with torch.no_grad():
                h_final = hiddens[-1]
                norm_device = next(norm.parameters()).device
                logits_final = lm_head(norm(h_final.to(norm_device))).to(work_device)
                # Only predict next token: shift by 1
                # logits_final[i] predicts token at position i+1
                target_logprobs = F.log_softmax(logits_final[:-1].float(), dim=-1).detach()

            # --- train translators ------------------------------------
            for l, h_l in enumerate(hiddens[:-1]):  # skip final layer (identity)
                optimizer.zero_grad()

                translated = translators.translate(h_l.to(work_device), l)
                norm_device = next(norm.parameters()).device
                logits_l = lm_head(norm(translated.to(norm_device))).to(work_device)
                pred_logprobs = F.log_softmax(logits_l[:-1].float(), dim=-1)

                # KL(pred || target) = Σ target * (target_logprob - pred_logprob)
                loss = F.kl_div(
                    pred_logprobs,
                    target_logprobs.exp(),
                    reduction="batchmean",
                    log_target=False,
                )
                loss.backward()
                nn.utils.clip_grad_norm_(translators.parameters(), 1.0)
                optimizer.step()

                epoch_loss += float(loss.item())
                epoch_steps += 1

        total_steps += epoch_steps
        if verbose and epoch_steps > 0:
            print(
                f"  [tuned_lens] epoch {epoch + 1}/{n_epochs}  "
                f"mean_kl={epoch_loss / epoch_steps:.4f}  "
                f"steps={epoch_steps}"
            )

    return translators.cpu()


# ---------------------------------------------------------------------------
# Inference — integrate with existing logit-lens infrastructure
# ---------------------------------------------------------------------------

def layer_log_probs_with_tuned_lens(
    backend: Any,
    prompt: str,
    translators: TunedLensTranslators,
    encoding_mode: str = "raw",
) -> list[torch.Tensor]:
    """Return per-layer log-prob vectors using tuned-lens translators.

    Drop-in replacement for
    ``unembedding_validation.layer_log_probs_for_next_token`` that applies
    the trained T_l before projecting through ``final_norm + lm_head``.

    Parameters
    ----------
    backend : GPTOSSTransformersBackend
    prompt : str
    translators : TunedLensTranslators
    encoding_mode : str
        ``"raw"`` bypasses the chat template (required for sequence
        continuation tasks).  See ``unembedding_validation`` for details.

    Returns
    -------
    list[torch.Tensor]
        One ``[vocab_size]`` log-prob tensor per layer.
    """
    from gpt_oss_interp.capture.activation_cache import ActivationCache

    if encoding_mode == "raw":
        prompt_ids = backend.tokenizer.encode(prompt, add_special_tokens=True)
    else:
        from gpt_oss_interp.harmony.prompting import encode_prompt
        prompt_ids = encode_prompt(backend.tokenizer, prompt)

    input_ids = torch.tensor([prompt_ids], device=backend.device)

    cache = ActivationCache(detach=True, to_cpu=True)
    handles = cache.register(backend.model, backend.structure.block_names)
    try:
        with torch.no_grad():
            backend.model(input_ids)
    finally:
        for h in handles:
            h.remove()

    norm_device = next(backend.structure.final_norm.parameters()).device
    layer_log_probs: list[torch.Tensor] = []

    for layer_idx, block_name in enumerate(backend.structure.block_names):
        rec = cache.last(block_name)
        if rec is None:
            continue
        hidden = rec.tensor  # [1, seq_len, hidden_dim]

        with torch.no_grad():
            # Apply translator if available for this layer
            if layer_idx < translators.n_layers:
                h = hidden[0, -1].to(translators.U[layer_idx].device)
                h = translators.translate(h, layer_idx)
                h = h.unsqueeze(0)  # [1, hidden_dim]
            else:
                h = hidden[0, -1:].to(norm_device)

            normed = backend.structure.final_norm(h.to(norm_device))
            logits = backend.structure.lm_head(normed).cpu().float()[0]
            log_probs = torch.log_softmax(logits, dim=-1)

        layer_log_probs.append(log_probs)

    return layer_log_probs


# ---------------------------------------------------------------------------
# Diagnostic: per-layer KL gap
# ---------------------------------------------------------------------------

def measure_translation_gap(
    backend: Any,
    prompts: Sequence[str],
    translators: TunedLensTranslators | None = None,
) -> dict[str, list[float]]:
    """Measure mean KL(P_l || P_L) before and after tuned-lens correction.

    Returns a dict with keys ``"raw"`` and (if translators provided)
    ``"tuned"``, each mapping to a list of mean KL values indexed by layer.

    This is the primary diagnostic for whether the tuned lens is working:
    raw KL should be large in early layers; tuned KL should be small
    everywhere.  The residual gap at each layer quantifies how much of
    early-layer computation is still opaque to the unembedding.
    """
    from gpt_oss_interp.capture.activation_cache import ActivationCache

    model = backend.model
    structure = backend.structure
    tokenizer = backend.tokenizer
    model_device = next(model.parameters()).device
    norm = structure.final_norm
    lm_head = structure.lm_head
    n_layers = len(structure.block_names)

    raw_kl_sums = [0.0] * n_layers
    tuned_kl_sums = [0.0] * (n_layers if translators else 0)
    counts = [0] * n_layers

    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_special_tokens=True)
        if len(ids) < 2:
            continue
        input_ids = torch.tensor([ids], device=model_device)

        cache = ActivationCache(detach=True, to_cpu=False)
        handles = cache.register(model, structure.block_names)
        try:
            with torch.no_grad():
                model(input_ids)
        finally:
            for h in handles:
                h.remove()

        hiddens = []
        for name in structure.block_names:
            rec = cache.last(name)
            if rec is None:
                continue
            hiddens.append(rec.tensor[0])  # [seq_len, hidden_dim]

        if len(hiddens) != n_layers:
            continue

        norm_device = next(norm.parameters()).device

        with torch.no_grad():
            logits_final = lm_head(norm(hiddens[-1].to(norm_device)))
            p_final = F.softmax(logits_final.float(), dim=-1).detach()

        for l, h_l in enumerate(hiddens):
            with torch.no_grad():
                logits_raw = lm_head(norm(h_l.to(norm_device)))
                p_raw = F.softmax(logits_raw.float(), dim=-1)
                kl_raw = float(F.kl_div(
                    p_raw.log(), p_final, reduction="mean", log_target=False
                ).item())
                raw_kl_sums[l] += kl_raw * h_l.shape[0]
                counts[l] += h_l.shape[0]

                if translators and l < translators.n_layers:
                    t_device = translators.U[l].device
                    h_t = translators.translate(h_l.to(t_device), l)
                    logits_t = lm_head(norm(h_t.to(norm_device)))
                    p_t = F.softmax(logits_t.float(), dim=-1)
                    kl_t = float(F.kl_div(
                        p_t.log(), p_final, reduction="mean", log_target=False
                    ).item())
                    tuned_kl_sums[l] += kl_t * h_l.shape[0]

    result: dict[str, list[float]] = {
        "raw": [
            raw_kl_sums[l] / counts[l] if counts[l] else float("nan")
            for l in range(n_layers)
        ]
    }
    if translators:
        result["tuned"] = [
            tuned_kl_sums[l] / counts[l] if counts[l] else float("nan")
            for l in range(n_layers)
        ]
    return result
