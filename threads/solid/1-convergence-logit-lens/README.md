# Thread 1: Convergence / Logit Lens

**Status**: Solid
**Narrative beat**: Measure

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

## Key findings
- **Capitalization** converges early (L1–2) — minimal depth processing
- **Coreference** converges mid-depth (L5) — requires semantic resolution
- **Induction** converges late (L17+) — consistent with induction heads as a late-layer phenomenon
- Task-dependent convergence depth is a fundamental organizing principle of gpt-oss-20b's computation

## Package dependencies
`readouts.logit_lens`, `backends.transformers_gpt_oss`, `benchmarks.tasks`

## Related threads
- [2-late-layer-ablation](../2-late-layer-ablation/) — causal validation of which layers matter
- [3-analysis-set-filtering](../3-analysis-set-filtering/) — convergence stability defines analysis sets
