# Thread 10: Bridge / Cross-Model

**Status**: In progress (early) — **Objective**: Validate cross-model generalization

## Problem
Every finding in this project is about a single model (gpt-oss-20b). The question of whether late-layer criticality, task-dependent convergence, vocabulary steering, or Hydra redundancy are properties of *this model* versus properties of *transformer MoE architectures in general* cannot be answered without testing on other models. A single-model study demonstrates methodology; a cross-model study reveals architecture-level principles.

## Why it matters
Interpretability results that hold across models are far more valuable than model-specific observations. If direct vocabulary steering works on Gemma, LLaMA, and gpt-oss-20b, that suggests it's exploiting a fundamental property of how transformers organize vocabulary space — a finding with broad implications. If it works only on gpt-oss-20b, it might be an artifact of that model's specific training. The bridge infrastructure also tests whether the toolkit is genuinely backend-agnostic or has hidden dependencies on gpt-oss-20b's architecture.

## Contribution
This thread is **original infrastructure work** — building a screening pipeline that can evaluate whether a new model is amenable to the same interpretability analyses. The candidate screening approach (checking convergence patterns, attention structure, and routing behavior before committing to full experiments) is a practical methodology contribution. This thread is early-stage: only one non-gpt-oss model (Gemma-3-1B) has been screened, and no cross-model comparison results exist yet.

## Scripts
- `screen_bridge_candidates.py` — screens candidate models for compatibility
- `bridge_cascade_intervention.py` — applies CASCADE-style interventions to bridge models
- `reference_cascade_target.py` — computes reference CASCADE targets

## Runs (in `runs/`)
- `bridge_candidate_screen/`
- `bridge_preflight/`
- `cascade_reference_induction/`, `cascade_reference_recency/`, `cascade_reference_syntax/`

## Current state
- One Gemma-3-1B screening run completed
- CASCADE reference targets computed for 3 task families on gpt-oss-20b
- Shows the toolkit *can* screen other models, but only one tested

## Gaps
- Only one non-gpt-oss model tested (Gemma-3-1B)
- No cross-model comparison results yet
- Depends on theoretical geometric framework (thread 12) for principled comparison

## Package dependencies
`benchmarks.tasks`, `benchmarks.pools`, `capture.activation_cache`, `backends.transformers_gpt_oss`

## Related threads
- [12-geometric-framework](../../theoretical/12-geometric-framework/) — principled metrics for cross-model comparison
- [11-cascade-distillation](../../theoretical/11-cascade-distillation/) — CASCADE targets used in bridge experiments

## References

Cross-model comparison in interpretability is an open problem. Most mechanistic interpretability work studies a single model; comparing computational organization across architectures requires metrics that are invariant to model-specific details:

- [Elhage et al. 2022 — "Toy Models of Superposition"](../../../doc/references/papers/t02-t05-t07-elhage-toy_models_of_superposition.pdf) — The superposition hypothesis implies that models with different architectures may organize the same information differently. Cross-model comparison requires disentangling representation format from computational content.

This thread is early-stage infrastructure. The screening pipeline tests whether convergence patterns, attention structure, and routing behavior are compatible with the interpretability analyses developed on gpt-oss-20b, before committing to full experiments on a new model.
