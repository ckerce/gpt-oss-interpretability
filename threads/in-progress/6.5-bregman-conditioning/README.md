# Thread 6.5: Bregman Conditioning

**Status**: In progress — **Objective**: Predict when linear steering is geometrically reliable
**Model**: gpt-oss-20b (201K vocab, 24 layers) — analysis targets the production model directly; Bregman geometry is vocabulary-scale and this thread is where that cost is justified.

## Problem
Threads 4–6 establish that steering directions can be extracted and that exact vocabulary-space interventions can flip model answers. But they do not yet answer a stricter question: **when should Euclidean steering be trusted at all?** A vector can flip the target answer while still redistributing probability mass in uncontrolled ways. Park et al. 2026 shows why: the natural geometry of softmax representation spaces is information geometry (Bregman / dually flat), not Euclidean. Probe vectors live in the dual space; adding them directly to primal representations is a type error. The companion paper (Kerce 2026) operationalizes this as a per-layer conditioning diagnostic: when the softmax Hessian at a given layer is ill-conditioned, Euclidean steering directions and their dual images diverge, and linear interventions leak probability mass off-target.

## Why it matters
This thread is the natural bridge between Thread 6 ([direct vocabulary steering](../../solid/6-direct-vocab-steering/)) and Thread 8 ([selectivity](../../solid/8-selectivity/)). It upgrades the narrative from:

- we can steer

to:

- we can predict when steering will be reliable before running the intervention.

Layers with high Bregman conditioning (well-conditioned softmax Hessian, high effective rank, cosine close to 1) should show cleaner steering and less off-target KL; layers with poor conditioning should predict selectivity failures. That makes this a Phase 2.5 result. Phase 1 tells us where the computation happens; Phase 2 shows we can intervene; this thread asks whether the geometry at those layers supports trustworthy intervention.

## Contribution
This thread applies the Bregman-conditioning diagnostics from the companion paper (Kerce 2026, *Stream separation improves Bregman conditioning in transformers*) to gpt-oss-20b. That paper works on symbolic dual-stream models; this thread is the first application to a production-scale MoE transformer. The theoretical basis is Park et al. 2026: the Hessian of the log-normalizer A(λ) is H = ∇²A = Cov[γ|λ], and it mediates the relationship between primal (Euclidean) and dual directions. The key move is to compute this Hessian in **hidden-state coordinates**:

```text
H(h) = Cov[w_y | h]
```

where `w_y` is the unembedding row for token `y` and the covariance is taken under the model's predictive distribution at hidden state `h`. This keeps the Hessian in the model's hidden dimension (4096D) instead of the vocabulary dimension (201K), making the analysis feasible. A top-k approximation (default k=2048) concentrates computation on the tokens that carry most of the probability mass.

The first implementation is observational:

- per-layer trace, effective rank, and condition number of `H`
- retained-probability diagnostics for top-k vocabulary approximation
- cosine between an exact vocabulary steering direction `W[a] − W[b]` and its dual image `H v`

The missing causal step is validated in the companion paper but not yet implemented here:

- matched Euclidean-vs-dual steering comparisons at equal target effect
- validation that low cosine / low effective rank predicts larger off-target KL

## Scripts
- `run_bregman_conditioning.py` — per-layer hidden-space Hessian analysis on gpt-oss-20b prompts; outputs Markdown table and per-sample JSON

## Runs (in `runs/`)
No runs yet. The runner is ready for first use against gpt-oss-20b.

## Figures (in `figures/`)
No figures yet.

## Preliminary results

No conditioning runs have been completed against gpt-oss-20b yet. However, the existing thread 6 and thread 8 results can be read as indirect Bregman conditioning evidence, because the geometry predicts exactly the patterns we already observe.

### The geometric reading of position specificity (thread 6)

The key diagnostic is the cosine between a steering direction `v = W[a] − W[b]` and its dual image `Hv`. When the model is already competing between tokens a and b (high probability mass on both), their unembedding directions dominate H = Cov[w_y | h], so v lies approximately in the dominant eigenspace of H and cosine(v, Hv) ≈ 1. When the model's distribution is concentrated elsewhere, v is approximately in the null space of H, Hv ≈ 0, and Euclidean steering cannot efficiently move the dual coordinate — it perturbs the hidden state but redistributes mass unpredictably.

This is exactly the position-specificity result from thread 6:

> **Prompt**: "The trophy would not fit in the suitcase because the suitcase was too small. The word 'small' refers to the"
>
> **Direction**: `W[trophy] − W[suitcase]`
>
> | Layer/position | Prediction | Gap | Bregman reading |
> |----------------|:----------:|----:|-----------------|
> | Baseline | **suitcase** | +2.40 | — |
> | Steer at decision position (L3) | **trophy** | −0.66 | v in dominant eigenspace of H → cosine ≈ 1 → clean flip |
> | Steer at token 0 (L0) | **suitcase** | +2.73 | model not yet competing on {trophy, suitcase} → v ⊥ dominant eigenspace → Hv ≈ 0 → no effect |

The same direction at the same scale produces opposite outcomes depending on layer and position — not because the model "changes its mind" between positions, but because the local geometry of the softmax manifold is different. Position specificity is a Bregman conditioning effect.

### Task families differ in how sharply H concentrates (threads 6, 7, 8)

Three task families with very different selectivity outcomes in thread 8 form a natural ordering:

| Task | Thread 8 channelized/whole ratio | Geometric prediction |
|------|:---------------------------------:|---------------------|
| Recency bias | 0.80–0.99 | H concentrates on {W[trophy], W[suitcase]}; steering direction well-aligned with dominant eigenspace |
| Induction | 0.60 | Pattern completion distributes probability across more tokens simultaneously; H eigenspace broader; v aligns less cleanly; more off-target leakage |
| Coreference | not measured (0 promoted channels) | Long-range antecedent integration is the most distributed; H most diffuse; Euclidean steering least predictable |

Thread 7 adds a finer-grained reading: the probe-causal dissociation (H4 probes well but H5 drives causally) is a signature of H having a dominant eigenspace that does not align with the steering direction — exactly what low cosine(v, Hv) would predict.

The companion paper's symbolic-model results predict that effective rank rises and condition number falls at convergence layers (threads 1 and 4), and that cosine similarity to the steering direction is lower for induction than recency at matched convergence depth — consistent with the selectivity ratios above.

## Gaps
- No causal validation: Euclidean-vs-dual intervention benchmark not yet implemented
- No cross-thread integration: selectivity outputs (thread 8) not yet compared against Bregman conditioning scores
- No DST-baseline comparison — companion paper's symbolic-model conditioning profiles haven't been replicated on a real transformer
- No empirical threshold calibration analogous to the companion paper's cosine cutoff

## Limitations
- **Logit-lens approximation**: The runner applies the model's *final* layer norm to intermediate hidden states, following the standard logit-lens approach. This is an approximation — the final norm is calibrated for the output of the last block, not for intermediate residual-stream states. Results at early layers should be interpreted cautiously.
- **Top-k renormalization artifact**: The Hessian approximation uses actual softmax probabilities over the full vocabulary partition function, *not* renormalized probabilities over the top-k support. Renormalizing would measure the geometry of a conditional distribution, not the model's distribution. Callers should verify that `mass_covered` is high (≥ 0.95) before trusting per-layer metrics; results at layers with low mass coverage are not interpretable.
- **Condition number vs. near-singular spectra**: The condition number is computed over the full positive spectrum using a machine-epsilon positivity floor, not over a "numerically supported" subspace. A spectrum like `[1, 1e-8, 1e-10]` correctly reports condition number ≈ 1e10 and numerical rank 1 rather than masking the near-singular directions. This is the intended behavior, but condition numbers in the range 1e6–1e12 should be treated as qualitative ("highly ill-conditioned") rather than precise.

## Package dependencies
`steering.bregman`, `capture.activation_cache`, `backends.transformers_gpt_oss`

## Related threads
- [6-direct-vocab-steering](../../solid/6-direct-vocab-steering/) — the steering behavior this thread is intended to explain
- [8-selectivity](../../solid/8-selectivity/) — off-target effect measurements this thread should predict
- [4-decision-trajectories](../../solid/4-decision-trajectories/) — identifies convergence layers; Bregman conditioning should be high at those layers
- [12-geometric-framework](../../theoretical/12-geometric-framework/) — broader theoretical home for the geometry story

## References

Bregman geometry formalizes the curvature of the softmax manifold and provides a rigorous basis for predicting when linear (Euclidean) steering will and will not produce controlled interventions:

- [Park et al. 2026 — "The Information Geometry of Softmax: Probing and Steering"](../../../doc/references/papers/t065-park-information_geometry_softmax.pdf) — the theoretical foundation (arXiv:2602.15293). Key results: (1) the natural geometry of softmax representation spaces is Bregman (dually flat), induced by the KL divergence; (2) probe vectors are elements of the dual space, so Euclidean steering commits a type error — adding a dual-space vector to a primal-space representation; (3) dual steering, which applies the probe in dual space, provably minimizes off-target distributional change (Theorem 3). Demonstrated on Gemma-3-4B and MetaCLIP-2.
- [Kerce 2026 — *Stream separation improves Bregman conditioning in transformers*](../../../doc/references/papers/t065-kerce-stream_separation_bregman_conditioning.pdf) (companion paper, in preparation) — operationalizes the Park et al. framework as per-layer conditioning diagnostics: trace(H), effective rank, condition number, and cosine between a steering direction `v` and its dual image `Hv`. Establishes that stream-separated (dual-stream) architectures improve Hessian conditioning relative to standard transformers on symbolic models. Thread 6.5 replicates these diagnostics on gpt-oss-20b.
- [Belrose et al. 2023 — "Eliciting Latent Predictions from Transformers with the Tuned Lens"](../../../doc/references/papers/t01-belrose-logit_lens_formalized.pdf) — The logit lens provides the per-layer hidden states that this thread's runner passes through the Hessian computation. The runner applies the same final-norm + unembedding pattern as the logit lens, inheriting both its interpretability and its approximation error at early layers.
