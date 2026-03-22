# Thread 1: Convergence / Logit Lens

**Status**: Solid — **Objective**: Measure convergence depth

## Problem
A fundamental question in mechanistic interpretability is *where* inside a transformer a given behavior gets resolved. Without knowing which layers are responsible for which computations, intervention-based interpretability is a needle-in-a-haystack search across thousands of components. For MoE models this is especially acute: the expert routing adds a second axis of variation (which experts, not just which layers), and production-scale MoE architectures like gpt-oss-20b have received far less interpretability attention than dense models.

## Why it matters
If different tasks converge at different depths, that tells us the model has a structured computational pipeline — not a diffuse, spread-out representation. This is a prerequisite for targeted intervention: you can't steer what you can't locate. Convergence depth also constrains which layers are candidates for ablation (thread 2) and steering (thread 6), and defines the filtering criterion for honest analysis sets (thread 3).

## Contribution
This thread applies the **logit lens** technique (nostalgebraist 2020; formalized by Belrose et al. 2023) to a production-scale MoE transformer. The logit lens itself is prior work — the contribution here is **empirical validation at MoE scale** and the demonstration that task-dependent convergence depth is a robust organizing principle in gpt-oss-20b. The choice-relative convergence metric (tracking when the correct choice first dominates, not just when the final prediction stabilizes) is a minor methodological extension.

## Scripts
- `run_logit_lens.py` — per-layer token prediction readouts with choice-relative convergence
- `calibrate_convergence.py` — convergence metric calibration across tasks

## Configs
- `configs/dry_run_recency.py` — smoke test (synthetic backend)

## Runs (in `runs/`)
- `convergence_calibration/`
- `logit_lens_induction/`
- `logit_lens_recency/`
- `logit_lens_syntax/`

## Figures (in `figures/`)
- `fig1_convergence_trajectories.{pdf,png}`

## Results

### Illustrative example — when does the model "know" the answer?

The logit lens reads off the model's best guess at each layer by projecting the hidden state through the unembedding matrix. For three different tasks, the answer emerges at different depths:

> **Capitalization** — "The lord of the ___" (expects "Rings")
> - L1: correct answer appears. The model resolves this almost immediately.
>
> **Coreference** — "Alice told Bob that she would leave early. The word 'she' refers to ___" (expects "Alice")
> - L1–4: wrong answer. L5: correct answer appears and holds. Semantic role assignment requires mid-depth processing.
>
> **Induction** — "D 5 Z 7 B 2 D 5 Z 7 B 2 D 5 Z 7 B ___" (expects "2")
> - L1–18: wrong or uncertain. L19: correct answer appears. Pattern completion with arbitrary (non-memorizable) tokens requires deep processing through the full attention stack.
>
> **Noisy induction** — "D 5 Z 7 B 2 D 5 A 7 B 2 ... D 5 R ___" (expects "7"; position 3 varies each cycle)
> - L1–19: wrong or uncertain. L20: correct answer appears. The model extracts the stable structure `7 B 2 D 5 [letter]` despite the varying letter — this is structural pattern recognition, not memorization.

The depth at which the correct answer first appears and stabilizes is the **convergence layer** — a fingerprint of how much computation each task type requires.

![Convergence trajectories](../../../figures/fig1_convergence_trajectories.png)

### Choice-relative convergence by task

| Task | Cases | Final correct rate | Final convergence (mean) | Final convergence (std) | Range |
|------|------:|-------------------:|-------------------------:|------------------------:|-------|
| capitalization | 4 | 0.50 | 1.0 | 1.7 | 0–4 |
| coreference | 4 | 1.00 | 5.0 | 7.5 | 0–18 |
| induction | 4 | 1.00 | 4.3 | 7.4 | 0–17 |
| recency_bias | 4 | 0.25 | 7.0 | 6.1 | 0–16 |
| syntax_agreement | 4 | 0.50 | 4.8 | 8.2 | 0–19 |

### Key findings
- **Capitalization** converges early (L1–2) — minimal depth processing
- **Coreference** converges mid-depth (L5) — requires semantic resolution
- **Induction** converges late (L17+) — consistent with induction heads as a late-layer phenomenon
- **Recency bias** and **syntax agreement** largely fail to converge to the correct answer — only 25% and 50% final correct rates, respectively
- Task-dependent convergence depth is a fundamental organizing principle of gpt-oss-20b's computation

## Package dependencies
`readouts.logit_lens`, `backends.transformers_gpt_oss`, `benchmarks.tasks`

## Related threads
- [2-late-layer-ablation](../2-late-layer-ablation/) — causal validation of which layers matter
- [3-analysis-set-filtering](../3-analysis-set-filtering/) — convergence stability defines analysis sets

## References

The logit lens was introduced informally by [nostalgebraist (2020)](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) as a way to read off per-layer predictions by projecting hidden states through the unembedding matrix. Belrose et al. formalized this into a family of "tuned lens" variants with theoretical justification for why the approach works in residual-stream architectures:

- [Belrose et al. 2023 — "Eliciting Latent Predictions from Transformers with the Tuned Lens"](../../../doc/references/papers/t01-belrose-logit_lens_formalized.pdf) — Formalizes the logit lens and introduces the tuned lens; shows that intermediate predictions are meaningful and improve monotonically with depth in most models.

This thread applies the logit lens at production MoE scale. The key extension is the **choice-relative convergence metric**, which tracks when the model's prediction first favors the correct choice rather than when the final prediction stabilizes — a distinction that matters for cases where the model oscillates before settling.
