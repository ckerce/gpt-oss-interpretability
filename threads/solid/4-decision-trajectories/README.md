# Thread 4: Decision Trajectories

**Status**: Solid — **Objective**: Characterize decision structure

## Problem
Current steering methods like Contrastive Activation Addition (CAA; Rimsky et al. 2023) require curating 100+ positive/negative example pairs to extract a steering direction — a labor-intensive process that doesn't scale and may inject the researcher's assumptions about what contrast matters. An alternative would be to read steering directions directly from the model's own computation: if the model changes its prediction at layer L, the direction of that change in logit space *is* a steering direction, self-supervised by the model itself.

## Why it matters
Self-supervised steering directions are a potential bridge between observational interpretability (what does the model compute?) and interventional interpretability (can we change what it computes?). If the model's own decision transitions provide steering directions of comparable quality to hand-crafted ones, it eliminates the human bottleneck in interpretability pipelines. This is also the empirical foundation for CASCADE distillation (thread 11), which would automate direction extraction via matrix factorization.

## Contribution
The decision trajectory extraction — identifying layers where the model's top-1 prediction changes and computing the logit-space difference as a direction — is an **original contribution** of this project. While logit-lens readouts are prior work, the specific use of prediction-transition layers as self-supervised steering signals has not been published elsewhere. The connection to classical topic modeling (treating decision directions as analogous to LSA basis vectors) is a novel framing.

## Scripts
- `extract_decision_vectors.py` — extracts per-layer decision points and logit-space directions
- `generate_decision_figure.py` — generates decision trajectory visualizations

## Runs (in `runs/`)
- `decision_vectors/`

## Figures (in `figures/`)
- `fig4_decision_trajectories.{pdf,png}`

## Results

![Decision trajectories](../../../figures/fig4_decision_trajectories.png)

### Decision trajectory summary across 3 task families

| Task | Prompt (truncated) | Key positions | Convergence layers | Transitions per position |
|------|-------------------|:-------------:|:------------------:|:------------------------:|
| Recency | "...suitcase was too" | 5 | L17–L18 | 6–12 |
| Induction | "A B C D A B C" | 5 | L13–L21 | 5–10 |
| Syntax | "The keys to the cabinet" | 5 | L16–L19 | 5–16 |

### Example decision arc — recency, position 12 ("suitcase was too ___")

```
L0: noise → L8: '‑' → L13: 'pack' → L14: <eos> → L16: '(' → L17: 'too' → L18: 'small'
```

Each transition (`→`) is a decision point where the model's top-1 prediction changes. The logit-space difference at each transition is a self-supervised steering direction — no curated contrastive examples needed.

### Key findings
- Each layer where the model's top-1 prediction changes is a "decision point"
- The logit-space difference at decision layers is a self-supervised steering direction
- Late transitions (L16–L18) are where semantically meaningful decisions happen; early transitions are noise
- Fundamentally different from CAA (which requires 100+ curated example pairs) — CASCADE reads directions from the model's own computation

## Package dependencies
`capture.activation_cache`, `readouts.logit_lens`, `backends.transformers_gpt_oss`, `benchmarks.tasks`

## Related threads
- [6-direct-vocab-steering](../6-direct-vocab-steering/) — decision directions can be used as steering vectors
- [11-cascade-distillation](../../theoretical/11-cascade-distillation/) — formalizes automatic extraction of these directions

## References

Decision trajectories occupy a middle ground between logit-lens readouts (thread 1) and contrastive steering methods. The key comparison is with CAA, which requires externally curated contrast pairs:

- [Rimsky et al. 2023 — "Steering Llama 2 via Contrastive Activation Addition"](../../../doc/references/papers/t04-t06-rimsky-contrastive_activation_addition.pdf) — CAA computes steering directions as the mean activation difference between 100+ curated positive/negative example pairs. Decision trajectories extract directions from the model's own layer-by-layer prediction changes, eliminating the curation bottleneck. The tradeoff: CAA directions are averaged over many examples and may generalize better; decision directions are per-input and capture the model's specific computational path.
- [Belrose et al. 2023 — "Eliciting Latent Predictions from Transformers with the Tuned Lens"](../../../doc/references/papers/t01-belrose-logit_lens_formalized.pdf) — The logit lens provides the per-layer predictions that decision trajectories are built from. Decision trajectories extend this by focusing specifically on *transitions* — layers where the prediction changes — rather than the full prediction profile.
