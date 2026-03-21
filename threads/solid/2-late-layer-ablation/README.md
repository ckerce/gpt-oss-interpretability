# Thread 2: Late-Layer Ablation

**Status**: Solid
**Narrative beat**: Measure

## Problem
Logit-lens readouts (thread 1) show *when* convergence happens, but correlation is not causation. A layer might show convergence because it *computes* the answer, or because an earlier layer already did and the representation is merely preserved. Ablation — zeroing or scaling a layer's contribution — provides the causal test: if removing a layer destroys performance, that layer is doing essential work. For production MoE transformers, the interaction between attention paths and MoE paths within each layer adds complexity: is it the attention heads, the expert MLP, or their combination that matters?

## Why it matters
Identifying the critical computation zone is the first step toward surgical intervention. If only 3 of 24 layers are decision-critical, the interpretability problem shrinks from "understand the whole network" to "understand these 3 layers deeply." The attention-vs-MoE decomposition further narrows the search: knowing that L20's MoE path carries 70% of the task signal (for example) tells you where to focus probing and steering efforts. This also validates or refutes claims from the companion preprints about late-layer criticality.

## Contribution
Causal ablation studies are a standard methodology in mechanistic interpretability (Elhage et al. 2022, Conmy et al. 2023). This thread applies layer-level and component-level ablation to a **production-scale MoE** and introduces component decomposition (separating attention-path from MoE-path contributions within the same layer). The finding that layers 19–21 are critical while layer 23 refines-but-doesn't-decide is an **empirical verification** of late-layer criticality at a scale larger than most published ablation studies. The component decomposition methodology is a modest extension of standard practice.

## Scripts
- `rank_soft_main_interventions.py` — ranks intervention effects across the soft-main analysis set
- Shared: `scripts/run_benchmark.py` — config-driven benchmark runner

## Configs
- `configs/soft_main_component_decomposition.py` — attention-path vs MoE-path decomposition at L20–22
- `configs/soft_main_component_decomposition_delta.py` — delta variant
- `configs/soft_main_late_layer_sweep.py` — layer ablation sweep L18–23
- `configs/soft_main_late_layer_delta_sweep.py` — delta variant

## Runs (in `runs/`)
- `soft_main_component_decomposition/`
- `soft_main_component_decomposition_delta/`
- `soft_main_late_layer_sweep/`
- `soft_main_late_layer_delta_sweep/`
- `soft_main_intervention_ranking/`
- `gpt_oss_20b_sweep/`

## Figures (in `figures/`)
- `fig2_late_layer_ablation.{pdf,png}`

## Results

![Late-layer ablation](../../figures/fig2_late_layer_ablation.png)

### Layer ablation sweep (zeroing layer output, 9-case soft main-analysis set)

| Layer ablated | Accuracy | Mean margin | Margin vs baseline |
|:-------------:|---------:|------------:|-------------------:|
| L18 | 66.7% | 2.126 | -68% |
| L19 | 44.4% | 0.726 | -89% |
| L20 | 44.4% | 0.945 | -86% |
| L21 | 44.4% | 0.607 | -91% |
| L22 | 33.3% | 1.258 | -81% |
| L23 | 100% | 1.357 | -80% |
| Baseline (no ablation) | 100% | 6.692 | — |

### Key findings
- Layers 19–21 are where gpt-oss-20b resolves task-relevant behavior
- Ablating L19–21 drops accuracy from 100% to 44% and collapses margin by 85–90%
- Layer 23 preserves accuracy (100%) despite margin loss — it refines, not decides
- L22 ablation produces the worst accuracy (33%) but higher margin than L19–21, suggesting it handles different cases
- Computation is concentrated across layers but distributed within layers (see thread 5)

## Package dependencies
`benchmarks.runner`, `backends.transformers_gpt_oss`, `config`, `interventions.specs`

## Related threads
- [1-convergence-logit-lens](../1-convergence-logit-lens/) — non-invasive view of the same phenomenon
- [5-hydra-head-redundancy](../5-hydra-head-redundancy/) — within-layer redundancy complements across-layer criticality
