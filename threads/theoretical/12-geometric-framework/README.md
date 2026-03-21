# Thread 12: Geometric Framework

**Status**: Theoretical
**Narrative beat**: Generalize

## Question
Can pullback metrics and dual metrics enable principled comparison of computational organization across models?

## Documents (in this directory)
- `GEOMETRIC_FRAMEWORK.md` — full theoretical treatment (33.6 KB): pullback metrics, representation manifolds, cross-model alignment

## Code status
- `gpt_oss_interp/features/geometry.py` (239 lines) — implemented metric-space analysis of feature point clouds
- `gpt_oss_interp/features/extractor.py` (477 lines) — feature extraction pipeline

## Experiment status
- Zero empirical validation of the geometric framework
- Feature extraction has been run but only at token-level PCA on tiny samples (thread 9)
- No cross-model comparison experiments

## Gaps
- Rich theory with zero empirical backing
- Pullback metric / dual metric concepts need demonstration on at least 2 models
- Depends on bridge/cross-model thread (10) for multi-model data

## Package dependencies
`features.geometry`, `features.extractor`

## Related threads
- [9-feature-extraction](../../in-progress/9-feature-extraction/) — provides the feature vectors this framework would analyze
- [10-bridge-cross-model](../../in-progress/10-bridge-cross-model/) — multi-model data needed for comparison
