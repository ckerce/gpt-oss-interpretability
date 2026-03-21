# Thread 8: Selectivity

**Status**: In progress
**Narrative beat**: Steer (evaluation)

## Problem
A steering intervention that flips the target answer (thread 6) is only useful if it doesn't simultaneously break everything else. If adding a `W["small"] - W["large"]` direction changes the answer from "large" to "small" but also degrades fluency, changes unrelated predictions, or shifts the model's confidence distribution on other tokens, the intervention isn't mechanistically clean — it's more like damage that happens to produce the desired side effect. Selectivity quantifies this: does the intervention affect *only* the intended behavior?

## Why it matters
Selectivity is the quality metric for steering. Without it, a 100% steering success rate is meaningless — the intervention might succeed by disrupting the model broadly rather than by engaging a specific circuit. High selectivity supports the mechanistic interpretation; low selectivity suggests the intervention is exploiting model fragility. This distinction matters both for interpretability (are we understanding real circuits?) and for potential applications (can we deploy targeted interventions without side effects?).

## Contribution
The selectivity comparison framework — measuring channelized vs whole-vector intervention effects across multiple metrics — is an **original evaluation methodology** developed in this project. Selectivity metrics exist in the broader ML fairness and robustness literatures, but the specific application to mechanistic steering interventions and the comparison between channel-level and whole-vector granularities is novel. This thread is in progress with limited task coverage (recency family only).

## Scripts
- `run_selectivity_comparison.py` — compares channelized vs whole-vector selectivity

## Runs (in `runs/`)
- `selectivity_e2_recency/`
- `selectivity_e2_recency_full/`
- `selectivity_e2_recency_h5/`
- `selectivity_e2_recency_topheads/`

## Figures
Referenced in steering memo but no standalone publication figure.

## Current state
- Code is solid (`steering.selectivity` — 33.9 KB, the largest module)
- 3 selectivity variants compared
- Limited to recency task family

## Gaps
- Needs broader task coverage beyond recency
- No standalone figure for the main paper

## Package dependencies
`steering.selectivity`, `steering.controls`, `steering.interventions`, `common.*`

## Related threads
- [6-direct-vocab-steering](../../solid/6-direct-vocab-steering/) — the interventions being evaluated
- [7-channel-probing](../7-channel-probing/) — channel-level analysis informs selectivity
