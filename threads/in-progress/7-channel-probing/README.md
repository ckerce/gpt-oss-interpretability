# Thread 7: Channel-Level Probing

**Status**: In progress
**Narrative beat**: Steer (fine-grained)

## Problem
Thread 6 shows that steering works at the whole-vector level: adding a vocabulary-space direction to the full hidden state at the right layer and position flips model answers. But which dimensions of that hidden state are actually carrying the signal? A 4096-dimensional steering vector might have its effect concentrated in a handful of channels, or it might be diffusely spread across all of them. Knowing this determines whether the model has *sparse, interpretable features* at the channel level or whether meaning is encoded in a distributed, superposition-like manner within layers.

## Why it matters
Channel-level decomposition connects mechanistic interpretability to the superposition hypothesis (Elhage et al. 2022): if steering effects concentrate in a few channels, the model may have features that are more interpretable than the Hydra effect (thread 5) would suggest. If effects are distributed across hundreds of channels, that confirms the superposition picture and motivates dictionary-learning approaches. Either answer is informative. Per-channel causal analysis also provides a finer-grained intervention primitive — potentially enabling more selective steering with fewer off-target effects (see thread 8).

## Contribution
Per-channel causal intervention on steering directions is **original work** extending the whole-vector steering of thread 6. While per-neuron or per-feature probing is a well-established technique (Bau et al. 2018; Gurnee et al. 2023), applying it specifically to vocabulary-space steering directions in an MoE model and measuring per-channel causal effects is novel. This thread is in progress — the code and initial experiments exist, but broader task coverage and publication figures are needed.

## Scripts
- `run_channel_probe.py` — per-channel probing experiments
- `run_per_channel_causal.py` — per-channel causal intervention analysis

## Runs (in `runs/`)
- `channel_probe_c71_phase1/`, `channel_probe_smoke_c71/`
- `channel_probe_e2_phase1/`, `channel_probe_e2_smoke/`
- `channel_probe_ss71_phase1/`
- `per_channel_causal_e2_recency/`, `per_channel_causal_e2_recency_smoke/`

## Figures
None yet — needs figure generation.

## Current state
- Code is solid (`steering.probing`, `steering.causal`)
- 5 probe runs completed with causal experiments
- No publication figures yet

## Gaps
- Missing referenced docs: `CHANNELIZED_XT_INTERVENTION_NOTE.md`, `PER_CHANNEL_XT_INTERVENTION_PLAN.md`
- Needs figure generation to match other solid threads
- Limited to recency task family so far

## Package dependencies
`steering.probing`, `steering.causal`, `common.artifacts`, `common.io`

## Related threads
- [6-direct-vocab-steering](../../solid/6-direct-vocab-steering/) — whole-vector steering that this decomposes
- [8-selectivity](../8-selectivity/) — selectivity metrics for channel-level effects
