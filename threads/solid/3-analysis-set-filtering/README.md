# Thread 3: Analysis Set Filtering

**Status**: Solid
**Narrative beat**: Measure

## Problem
Mechanistic interpretability papers routinely claim that a model "uses mechanism X" based on a handful of cherry-picked examples. But models are messy: some inputs trigger clean, interpretable circuits; others produce noisy, unstable behavior where no clean mechanistic story holds. If you don't separate these cases *before* running ablation or steering experiments, you contaminate your results with noise and risk overclaiming. The field largely lacks systematic methodology for deciding which test cases to include in a mechanistic analysis.

## Why it matters
Honest filtering is the methodological backbone of every other thread in this project. The ablation results (thread 2), Hydra measurements (thread 5), and steering experiments (thread 6) are all run on the filtered "soft main-analysis set" — the 9/20 cases that meet convergence stability criteria. Without this filtering, the ablation results would show weaker effects diluted by noisy cases, and steering success rates would be misleadingly low. This thread makes the project's claims more conservative but more defensible.

## Contribution
This is a **methodological contribution** that is largely original to this project. While individual papers sometimes note that "some examples work better than others," the systematic 4-way stratification by convergence stability (correct+late-stable, correct+unstable, incorrect+late-stable, incorrect+unstable) and the explicit reporting that only 45% of cases pass the filter is uncommon in the literature. The key insight — that honest filtering is a *feature* that tells you where the model's behavior supports mechanistic claims — reframes what is usually treated as a limitation.

## Scripts
- `stratify_analysis_set.py` — 4-way stratification by convergence stability
- `analyze_filtered_benchmark.py` — benchmark analysis on filtered sets
- Shared: `scripts/run_benchmark.py`

## Configs
Uses same configs as thread 2 (soft-main configs)

## Runs (in `runs/`)
- `analysis_set_stratification/`
- `filtered_benchmark_analysis/`
- `retained_case_decision_audit/`

## Figures (in `figures/`)
- `fig3_analysis_set_stratification.{pdf,png}`

## Key findings
- Only 9/20 cases (45%) are "correct, late-stable" — the soft main-analysis set
- Recency bias and syntax agreement cases largely fail convergence criteria
- This is a feature: it tells you where the model's behavior is robust enough for causal claims
- All other threads (ablation, steering, Hydra) use this filtered set

## Package dependencies
`benchmarks.tasks`, `benchmarks.pools`

## Related threads
- [1-convergence-logit-lens](../1-convergence-logit-lens/) — convergence stability is the filtering criterion
- [2-late-layer-ablation](../2-late-layer-ablation/) — ablation results use the filtered set
