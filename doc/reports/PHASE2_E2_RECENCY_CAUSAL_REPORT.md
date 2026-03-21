# Phase 2 E2 Recency Causal Report

## Purpose

This report is the first direct follow-up to
[PHASE1_CHANNEL_PROBE_INITIAL_REPORT.md](doc/reports/PHASE1_CHANNEL_PROBE_INITIAL_REPORT.md).

Phase 1 established two things:

1. the matched `71M` pair is not the right substrate for layerwise `x_t`
   probing because the symbolic stream is frozen there
2. `E2_independent` is the first model where layerwise symbolic probing is
   nondegenerate, with the strongest family-level signal appearing in
   `recency_bias`

This report asks the next planned question:

> do the channels promoted by Phase 1 actually become the strongest causal
> channels under matched symbolic-slice intervention?

That is the central bridge from probing to intervention.

## Artifacts

- causal artifact:
  [e2_recency_bias_causal.json](runs/per_channel_causal_e2_recency/e2_recency_bias_causal.json)
- causal report:
  [e2_recency_bias_report.md](runs/per_channel_causal_e2_recency/e2_recency_bias_report.md)
- upstream probe artifact:
  [e2_channel_probe.json](runs/channel_probe_e2_phase1/e2_channel_probe.json)

## Experimental Setup

Model:

- `E2_independent`

Family:

- `recency_bias`

Intervention:

- one symbolic head slice at a time
- `x_t` stream
- `post_block`
- scales:
  `{-8, -4, -2, -1, +1, +2, +4, +8}`

Per-channel causal score:

- for each `(layer, head)`, compute the best scale by mean directed local effect
  across the recency family

Comparisons:

- promoted channels vs low-ranked channels
- promoted channels vs random channel sample
- probe rank vs causal rank
- held-out probe accuracy vs causal effect
- readout decomposition on the strongest channels

## Main Result

The probing signal survives causally, but only weakly.

Numerically:

- probe-rank vs causal-rank Spearman: `0.221`
- held-out probe accuracy vs causal effect Spearman: `0.162`

That is not zero, but it is far from a strong monotonic alignment.

So the first honest conclusion is:

- Phase 1 is **not** useless
- but Phase 1 ranking is only a weak predictor of Phase 2 causal potency in
  this first recency run

That means the central hypothesis is not yet supported strongly enough to move
confidently into mixed-token composites.

## What Did Work

The causal interventions themselves are real and strong.

Top channels:

- `L5 H5`: `6.660`
- `L0 H5`: `6.462`
- `L4 H5`: `6.329`
- `L1 H5`: `6.312`
- `L3 H5`: `6.283`
- `L2 H5`: `6.247`

These are large effects, and they are tightly concentrated on head `H5`
throughout depth.

That is a meaningful mechanistic pattern:

- one symbolic channel family is doing most of the causal work for recency
- the effect persists across layers

So even though probe rank is only weakly aligned overall, the causal picture is
not diffuse. It is structurally concentrated.

## What Did Not Work

The promoted set did not separate sharply from controls.

Mean effect sizes:

- promoted channels: `5.373`
- low-ranked channels: `4.332`
- random-control sample: `4.942`

So the promoted channels are better on average, but not dramatically so.

That is the most important caution in the whole run.

It means the current Phase 1 features:

- sign accuracy
- raw probe score
- position sensitivity

are not yet isolating the causally dominant channels as sharply as the full
plan would want.

## Readout Decomposition

The strongest channels are still meaningfully symbolic.

Examples:

- `L5 H5` on `recency_002`
  - combined effect: `-14.710`
  - `x_t` effect: `-16.110`
  - `x_e` effect: `0.000`
  - transfer ratio: `0.000`

- `L0 H5` on `recency_002`
  - combined effect: `-15.022`
  - `x_t` effect: `-16.758`
  - `x_e` effect: `+9.173`
  - transfer ratio: `0.547`

- `L4 H5` on `recency_002`
  - combined effect: `-14.394`
  - `x_t` effect: `-16.073`
  - `x_e` effect: `+5.856`
  - transfer ratio: `0.364`

Interpretation:

- the strongest recency channels are not merely causing behavior through a
  hidden `x_e` detour
- some channels remain primarily visible in `x_t`
- others partially transfer into `x_e`, but not so strongly that the symbolic
  interpretation disappears

This is a useful positive result.

## What This Means For The Plan

The plan survives, but it needs a refinement before Phase 4.

Current status:

- Phase 1 probing is real
- Phase 2 causal effects are real
- the probing-to-causal bridge is weaker than hoped

So the next move should **not** be mixed-token composites yet.

Instead, the right next step is:

1. improve the Phase 1 feature set
2. rerun the probe-to-causal comparison
3. only then consider composites

## Most Likely Missing Ingredient

The current probe is too static.

Right now it emphasizes:

- symbolic slice alignment at the decision token
- sign consistency across cases
- crude position sensitivity

But the causal result suggests that recency potency may depend more on:

- how the same head evolves across layers
- vertical-channel structure
- or interaction with downstream readout

So the next refinement should probably add:

- vertical persistence as an actual ranking feature, not just a descriptive one
- maybe trajectory-based features over the same head across depth
- possibly a probe feature derived from `x_t` to `x_e` transfer propensity

## Bottom Line

The first Phase 2 run is a qualified positive result.

Positive:

- single-channel symbolic interventions in `E2` causally move recency behavior
- the dominant causal structure is concentrated in head `H5`
- the strongest channels remain strongly visible in `x_t`

Qualified:

- the current Phase 1 ranking is only weakly predictive of causal rank
- promoted channels outperform controls only modestly
- so the bridge from probing to compositional intervention is not yet strong
  enough to justify Phase 4

That is exactly the kind of result we needed to learn before trying mixed-token
symbolic compositions.
