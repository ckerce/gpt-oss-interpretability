# Thread 5: Hydra / Head Redundancy

**Status**: Solid — **Objective**: Measure head redundancy

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

## Results

![Hydra variance](../../../figures/fig5_hydra_variance.png)

### Per-head ablation at L20 (64 heads, 9-case soft main-analysis set)

All 64 heads maintain 100% accuracy when individually ablated. Margin variation is minimal:

| Statistic | Value |
|-----------|------:|
| Heads tested | 64 |
| Accuracy (all heads) | 100% |
| Mean margin | 6.692 |
| Margin std (σ) | 0.042 |
| Min margin (H20) | 6.608 |
| Max margin (H52) | 6.778 |

### Cross-model comparison (Hydra effect strength)

| Model | Head margin σ | Interpretation |
|-------|-------------:|----------------|
| gpt-oss-20b (this work) | 0.042 | Extreme redundancy — Hydra fully active |
| PLS-paper dense control | 0.08 | High redundancy — standard training baseline |
| PLS-trained models | 0.47 | Low redundancy — Hydra broken by per-layer supervision |

### Per-task Hydra profiles

| Task | Mean margin | Margin σ | Range |
|------|------------:|---------:|-------|
| Capitalization | 3.239 | 0.157 | 2.842–3.653 |
| Induction | 6.463 | 0.055 | 6.332–6.561 |
| Coreference | 9.299 | 0.062 | 9.159–9.448 |

### Key findings
- All 64 heads at L20 are nearly equally redundant (σ = 0.042)
- gpt-oss-20b's σ = 0.042 is **half** the PLS-paper control (σ = 0.08) and **11x smaller** than PLS-trained models (σ = 0.47)
- Per-task σ values (0.055–0.157) are all small, but capitalization shows the most head-to-head variation
- Validates the Hydra hypothesis at production scale: standard training produces extreme distributed redundancy
- Resolves the paradox: layers are critical (thread 2) but individual heads within them are not

## Package dependencies
`benchmarks.runner`, `backends.transformers_gpt_oss`, `config`

## Related threads
- [2-late-layer-ablation](../2-late-layer-ablation/) — layer-level criticality vs head-level redundancy
- Companion: [PLS preprint](https://arxiv.org/abs/2603.18029)
