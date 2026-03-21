# Thread 6: Direct Vocabulary Steering

**Status**: Solid (best-supported thread)
**Narrative beat**: Steer

## Problem
Activation steering methods (Turner et al. 2023; Rimsky et al. 2023) add learned directions to hidden states to shift model behavior. But a persistent criticism is that these methods might work by *disrupting* computation (a diffuse perturbation) rather than by *communicating* in the model's own representational language. If steering only works when applied everywhere, or works equally well at all positions, it's hard to distinguish from noise injection. The strongest possible evidence for genuine mechanistic steering would be: (a) using directions derived purely from the model's own vocabulary embeddings, with no learned component, and (b) showing that the effect is position-specific — steering at the decision-relevant position flips the answer while identical steering elsewhere has no effect.

## Why it matters
Direct vocabulary steering is the "proof of understanding" step in the narrative arc. Threads 1–3 measure where computation happens; threads 4–5 characterize its structure. This thread demonstrates that the understanding is *actionable*: we can precisely intervene in the model's computation using directions we can fully interpret (they are literal vocabulary differences, not opaque learned vectors). Position specificity is the critical control — it proves this is circuit-level intervention, not a blunt perturbation artifact. This is also the empirical foundation for the steering memo, the most publication-ready artifact in the project.

## Contribution
The use of exact vocabulary-space directions (`W[token_A] - W[token_B]` from the unembedding matrix) for steering is an **original contribution**. Prior steering methods (activation addition, CAA, RepE) learn directions from contrastive examples; using raw vocabulary embeddings eliminates the training step entirely. The key **original finding** is position specificity: identical steering vectors produce opposite effects depending on where in the sequence they are applied. The 10+ experiment variants (stream ablation, embed-init ablation, readout decomposition, position controls) provide unusually thorough empirical validation for a steering result.

## Scripts
- `run_direct_vocab_steering.py` — steering experiments across layers, positions, and strengths
- `run_direct_vocab_demo.py` — matched-pair demonstrations
- `generate_direct_vocab_memo_figures.py` — generates memo figures

## Documents (in this directory)
- `DIRECT_VOCAB_DEMO_REPORT.md` — detailed experiment report
- `memo/` — LaTeX memo with publication-quality figures

## Runs (in `runs/`)
- `direct_vocab_demo_matched_71m/`
- `direct_vocab_embedinit_ablation_xe/`, `direct_vocab_embedinit_ablation_xt/`
- `direct_vocab_large_models_cpu/`
- `direct_vocab_position_control_decision/`, `direct_vocab_position_control_token0/`
- `direct_vocab_preblock_ablation_xe/`, `direct_vocab_preblock_ablation_xt/`
- `direct_vocab_readout_c71_embed_xt_combined/`, `..._xeread/`, `..._xtread/`
- `direct_vocab_stream_ablation_xe/`, `direct_vocab_stream_ablation_xt/`

## Figures (in `figures/`)
- `fig_matched_pair_heatmaps.{pdf,png}`
- `fig_matched_pair_layer_profiles.{pdf,png}`
- `fig_matched_pair_strength_scans.{pdf,png}`
- `fig_baseline_local_gaps.{pdf,png}`

## Key findings
- Exact `W[token_A] - W[token_B]` directions in late layers cleanly flip model answers
- Effect is **position-specific**: steering at the decision-token position flips answers; identical steering at token 0 produces zero effect
- Position specificity rules out diffuse perturbation artifacts — this is real circuit-level control
- 10+ experiment variants covering stream ablation, embed-init ablation, readout decomposition
- 4.8 GB of experiment data — the most thoroughly exercised thread

## Package dependencies
`benchmarks.tasks`, `steering.*`, `backends.transformers_gpt_oss`

## Related threads
- [4-decision-trajectories](../4-decision-trajectories/) — self-supervised source of steering directions
- [11-cascade-distillation](../../theoretical/11-cascade-distillation/) — automates direction extraction
- [7-channel-probing](../../in-progress/7-channel-probing/) — per-channel analysis of steering effects
