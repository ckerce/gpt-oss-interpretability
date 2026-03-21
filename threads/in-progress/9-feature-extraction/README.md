# Thread 9: Feature Extraction

**Status**: In progress
**Narrative beat**: Measure (extended)

## Problem
Individual interpretability readouts — logit lens, attention patterns, expert routing weights — each capture one slice of what a model is doing. But interpreting a model requires combining these signals: a token might have high expert-3 activation *and* late convergence *and* a specific attention pattern, and it's the combination that defines the computational mode. Constructing a unified feature vector that captures all of these signals simultaneously enables clustering, dimensionality reduction, and geometric analysis that would be impossible on any single readout.

## Why it matters
If computational modes form distinct clusters in a high-dimensional feature space, that's direct evidence of structured computation — the model is doing categorically different things for different inputs, not computing on a smooth continuum. This connects interpretability to the representation learning literature and provides a quantitative foundation for claims like "the model handles induction differently from coreference." The feature space is also a prerequisite for the geometric framework (thread 12), which uses it for cross-model comparison.

## Contribution
The extended Tier-2 feature system is **adapted from the companion PLS preprint's** activation clustering methodology, extended for MoE architectures. The original PLS feature vector captures attention and residual-stream signals; this adaptation adds expert routing weights and MoE-specific features to produce ~7,200-dimensional vectors. The adaptation to MoE is a modest extension of prior work. This thread is in progress — the code exists but has only been exercised on 12 tokens across 3 runs, far too thin for the geometric analysis it was designed to support.

## Scripts
- `run_feature_extraction.py` — extended Tier-2 feature extraction (~7,200D for MoE)

## Runs (in `runs/`)
- `features_induction/`
- `features_recency/`
- `features_syntax/`

## Figures
None.

## Current state
- Code is implemented (`features/extractor.py`: 477 lines, `features/geometry.py`: 239 lines)
- 3 runs completed but thin — only 12 tokens analyzed, ~20 KB each
- Geometric framework document exists but empirical validation is token-level PCA on tiny samples

## Gaps
- Needs larger-scale extraction to validate the 7,200D feature space
- Geometry analysis (`features/geometry.py`) hasn't been exercised at meaningful scale
- Connection to downstream tasks (e.g., probing, steering) not yet demonstrated

## Package dependencies
`features.extractor`, `features.geometry`, `capture.activation_cache`, `backends.transformers_gpt_oss`

## Related threads
- [12-geometric-framework](../../theoretical/12-geometric-framework/) — theoretical basis for feature-space analysis
- [1-convergence-logit-lens](../../solid/1-convergence-logit-lens/) — simpler per-layer readouts
