# TODO: Targeted Interventions on gpt-oss-20b

## Motivated by current results

The logit-lens runs identified specific convergence layers for each behavior. The intervention benchmark confirmed that layer 20 has the strongest causal role. The next experiments should exploit these findings.

## 1. Convergence-layer ablation sweep

The logit lens shows that different behaviors converge at different layers. Ablate at and around the convergence point to test whether these layers are causally necessary.

| Behavior | Convergence layer | Ablation targets |
| --- | ---: | --- |
| Recency bias ("small") | 18 | L16, L17, L18, L19, L20 |
| Syntax agreement ("are") | 20 | L18, L19, L20, L21, L22 |
| Induction copying | 21 | L19, L20, L21, L22, L23 |

**Config**: One `LAYER_SCALE` intervention per target layer, scales `(0.0, 0.5, 1.0)`.

**Question**: Does ablating the convergence layer destroy only the converging behavior, or does it cause collateral damage to other tasks?

## 2. Collateral-damage matrix

Run the full task suite under each single-layer ablation and measure:
- **Direct effect**: accuracy change on the targeted behavior
- **Collateral damage**: accuracy change on all other behaviors

This produces a `task x layer` matrix showing which layers are shared vs. specialized.

**Config**: 24 `LAYER_SCALE` interventions (one per layer), scale=0.0. All 5 tasks.

**Question**: Is the model modular (each behavior localizes to different layers) or distributed (all layers contribute to all behaviors)?

## 3. Head-level induction search

Induction heads are a well-characterized circuit. Since induction converges at layer 21, sweep attention heads at layers 19-23 to find which heads implement the induction pattern.

**Config**: For each of layers 19-23, create `HEAD_MASK` interventions targeting heads in groups of 8 (0-7, 8-15, 16-23, ..., 56-63), scale=0.0. Run only the `induction` task.

**Question**: Which specific attention heads are necessary for induction in gpt-oss-20b? Are they in sliding-attention or full-attention layers?

## 4. Attention-pattern contrast: sliding vs. full

gpt-oss-20b alternates between sliding-window (128 tokens) and full-attention layers. This creates a natural experiment:
- Ablate only sliding-attention layers (L0, L2, L4, ...)
- Ablate only full-attention layers (L1, L3, L5, ...)

**Question**: Do recency bias and syntax agreement (which require long-range dependencies) depend more on full-attention layers?

## 5. Per-layer logit-lens under intervention

Run the logit lens *while* an intervention is active. For example:
- Logit lens on the recency prompt with layer 18 ablated
- Logit lens on the induction prompt with layer 21 ablated

**Question**: When you ablate the convergence layer, does the prediction fail to converge at all, or does it shift to a later layer?

## 6. MoE expert scaling at convergence layers

The MLP output scaling showed small effects at layer 8. Repeat at the convergence layers (18, 20, 21) where expert computation may be more behavior-specific.

**Config**: `EXPERT_MASK` at layers 18, 20, 21. Scales `(0.0, 0.25, 0.5, 0.75, 1.0)`.

**Question**: Is MoE expert computation more causally important at convergence layers than at arbitrary layers?

## 7. Bidirectional head intervention at L12

The current results show that ablating mid-layer heads at L12 *increases* margin (3.178 vs. 3.114 baseline). This suggests L12 heads add noise for these tasks.

**Config**: `HEAD_MASK` at L12, sweep individual heads (0-63 one at a time) or groups of 4, scale=0.0.

**Question**: Which specific L12 heads degrade performance? Are they in the sliding or full attention pattern? What behaviors do they encode?

## 8. Non-quantized checkpoint for router analysis

The MXFP4 fused kernel prevents router introspection. Loading the model in bf16 (non-quantized) would enable:
- Direct router hook capture
- Per-layer expert selection visualization
- Gate-level expert suppression (not just MLP output scaling)

**Requirement**: ~40GB VRAM (dual GPU or offload). Alternatively, analyze one layer at a time by loading only that layer's weights in bf16.

## Priority order

1. Convergence-layer ablation sweep (directly validates logit-lens findings)
2. Collateral-damage matrix (most publishable result)
3. Head-level induction search (ties to known mech-interp literature)
4. Per-layer logit-lens under intervention (novel methodology)
5. Sliding vs. full attention contrast (exploits architecture-specific structure)
6. MoE expert scaling at convergence (strengthens expert-level story)
7. Bidirectional head intervention at L12 (explains anomalous result)
8. Non-quantized router analysis (requires more hardware)
