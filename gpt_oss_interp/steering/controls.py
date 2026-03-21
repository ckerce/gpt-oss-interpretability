"""Null and matched control helpers for probing and causal validation."""

from __future__ import annotations

import torch


def make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def shuffled_head_scores(scores: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Permute the head axis of a [cases, layers, heads] score tensor."""
    if scores.dim() != 3:
        raise ValueError(f"Expected [cases, layers, heads], got {tuple(scores.shape)}")
    n_heads = scores.shape[-1]
    perm = torch.randperm(n_heads, generator=generator)
    return scores.index_select(-1, perm)


def label_permuted_scores(scores: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Randomly flip case labels by multiplying each case by +/-1."""
    if scores.dim() != 3:
        raise ValueError(f"Expected [cases, layers, heads], got {tuple(scores.shape)}")
    flips = torch.randint(0, 2, (scores.shape[0],), generator=generator, dtype=torch.int64)
    signs = (flips * 2 - 1).to(dtype=scores.dtype).view(-1, 1, 1)
    return scores * signs


def random_direction_slices_like(
    vocab_by_head: torch.Tensor,
    reference_slices: torch.Tensor,
    generator: torch.Generator,
    exclude_token_ids: set[int] | None = None,
) -> torch.Tensor:
    """Sample same-norm random token-slice directions for each case.

    Args:
        vocab_by_head: [vocab, heads, head_dim]
        reference_slices: [cases, heads, head_dim]
        exclude_token_ids: token ids that should not be sampled

    Returns:
        Random directions with the same shape and per-head norm as reference_slices.
    """
    if reference_slices.dim() != 3:
        raise ValueError(f"Expected [cases, heads, head_dim], got {tuple(reference_slices.shape)}")

    n_vocab = vocab_by_head.shape[0]
    n_cases = reference_slices.shape[0]
    exclude = exclude_token_ids or set()
    candidates = [idx for idx in range(n_vocab) if idx not in exclude]
    if len(candidates) < 2:
        raise ValueError("Need at least two candidate tokens for random-direction control")

    candidate_tensor = torch.tensor(candidates, dtype=torch.long)
    rand_a = candidate_tensor[torch.randint(0, len(candidates), (n_cases,), generator=generator)]
    rand_b = candidate_tensor[torch.randint(0, len(candidates), (n_cases,), generator=generator)]

    # Avoid degenerate identical-token directions.
    same = rand_a == rand_b
    while bool(same.any()):
        rand_b[same] = candidate_tensor[torch.randint(0, len(candidates), (int(same.sum().item()),), generator=generator)]
        same = rand_a == rand_b

    random_slices = vocab_by_head[rand_a] - vocab_by_head[rand_b]
    ref_norm = torch.norm(reference_slices, dim=-1, keepdim=True).clamp_min(1e-8)
    rand_norm = torch.norm(random_slices, dim=-1, keepdim=True).clamp_min(1e-8)
    return random_slices * (ref_norm / rand_norm)
