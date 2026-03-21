# Thread 5: Hydra / Head Redundancy

**Status**: Solid
**Narrative beat**: Structure

## Problem
Thread 2 shows that layers 19–21 are critical — ablating them collapses performance. But *within* those layers, are all attention heads equally important, or are a few heads doing most of the work? The "Hydra effect" hypothesis (from the companion PLS preprint) predicts that standard training produces extreme distributed redundancy: all heads learn similar functions, so removing any one has negligible impact. If true, this has profound implications for interpretability: it means you cannot find "the coreference head" or "the induction head" in a standard model because no single head is specialized.

## Why it matters
The Hydra effect is one of the central arguments for *why* standard transformers are hard to interpret: if computation is maximally distributed within layers, there are no clean circuits to find. This is the motivation behind per-layer supervision (PLS), which breaks the Hydra effect by encouraging head specialization. Measuring the Hydra effect at production scale validates the theoretical argument and provides a quantitative baseline: gpt-oss-20b's σ = 0.042 is the reference point against which PLS-trained models (σ = 0.47) are compared.

## Contribution
This thread **validates the Hydra hypothesis from the companion PLS preprint** (arxiv 2603.18029) at production scale. The Hydra effect itself is described and named in that preprint; the contribution here is the **first measurement on a production-scale MoE model** (21B params, 64 GQA heads). The finding that gpt-oss-20b's head redundancy is even more extreme than the PLS paper's dense control (σ = 0.042 vs σ = 0.08) strengthens the original paper's argument that standard training systematically produces distributed redundancy.

## Scripts
- `analyze_hydra.py` — per-head ablation variance analysis
- Shared: `scripts/run_benchmark.py`

## Configs
- `configs/head_ablation_L20.py` — ablates each of 64 heads individually at L20

## Runs (in `runs/`)
- `head_ablation_L20/`

## Figures (in `figures/`)
- `fig5_hydra_variance.{pdf,png}`

## Key findings
- All 64 heads at L20 are nearly equally redundant (σ = 0.042)
- gpt-oss-20b's σ = 0.042 is **half** the PLS-paper control (σ = 0.08) and **11x smaller** than PLS-trained models (σ = 0.47)
- Validates the Hydra hypothesis at production scale: standard training produces extreme distributed redundancy
- Resolves the paradox: layers are critical (thread 2) but individual heads within them are not

## Package dependencies
`benchmarks.runner`, `backends.transformers_gpt_oss`, `config`

## Related threads
- [2-late-layer-ablation](../2-late-layer-ablation/) — layer-level criticality vs head-level redundancy
- Companion: [PLS preprint](https://arxiv.org/abs/2603.18029)
