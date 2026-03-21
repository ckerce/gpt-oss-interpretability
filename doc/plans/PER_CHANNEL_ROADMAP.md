# Per-Channel Research Roadmap

## Purpose

This document captures the simplified path forward after the first Phase 1 and
Phase 2 runs.

It is intentionally narrower than
[PER_CHANNEL_XT_INTERVENTION_PLAN.md](doc/plans/PER_CHANNEL_XT_INTERVENTION_PLAN.md).
That earlier document still defines the full design space. This roadmap defines
what matters *now*.

## Executive Assessment

The dual-stream idea remains promising, but the program is currently trying to
answer three different questions at once:

1. `Representation claim`
- dual-stream models preserve a symbolic channel that is more directly writable
  and interpretable than a standard residual stream

2. `Discovery claim`
- differential probing can identify the symbolic channels that actually matter

3. `Compositional control claim`
- once those channels are identified, mixed-token symbolic composites can be
  used as interpretable controls

The current evidence does not support all three equally.

Current status:

- claim 1: partially supported
- claim 2: weak partial support
- claim 3: still speculative

That means the program should *not* move to composites yet.

## What Questions Matter Most

The highest-value questions are:

1. `Can symbolic channels be causally written in a cleaner way than whole-vector activation steering?`
2. `Does probing identify causally potent channels, or only descriptive ones?`
3. `Are the strongest effects genuinely symbolic, or mostly transferred through x_e?`
4. `Does channelized intervention buy anything over whole-vector intervention?`

Lower-value questions for now:

- broad mixed-token composites
- family-wide expansion before the method stabilizes
- large role taxonomies
- immediate scaling to every available model

## What Capabilities Matter

The capabilities that matter most are:

1. `Single-channel causal potency`
- one symbolic slice can move the decision locally and predictably

2. `Selectivity`
- target behavior moves more than matched controls

3. `Symbolic visibility`
- the effect remains visible in `x_t`, not only in combined readout

4. `Rankability`
- useful channels can be identified before intervention

5. `Compositionality`
- only after the previous four hold

## What We Learned So Far

### Matched 71M pair

The matched `SS-71` / `C-71` pair is not the right substrate for layerwise
`x_t` discovery.

Why:

- the symbolic stream is frozen there for the purpose that matters here
- the layer axis becomes degenerate

These models are still useful, but for different questions:

- symbolic-write persistence
- readout decomposition
- stream-transfer analysis
- frozen-stream intervention behavior

### E2_independent

`E2_independent` is the first meaningful Phase 1 substrate.

Why:

- `x_t` changes across depth
- per-channel probing is nondegenerate

Current empirical picture:

- recency yields promoted channels
- induction and coreference do not yet clear the gate
- Phase 2 on recency gives real causal effects
- but probe rank only weakly predicts causal rank

That is the central constraint on the next step.

## Critical Assessment

### What is strong

- the scaffolded repo split was the right move
- the nulls and promotion gate are necessary
- the readout decomposition work is genuinely informative
- the decision to stop forcing the matched `71M` pair into layerwise probing was correct

### What is weak

1. `The current probe is too static`
- it mainly captures local symbolic alignment at the decision token
- that is likely too weak a summary of why a channel becomes causally dominant

2. `The current scope is too broad`
- recency is the only family that currently gives a clean enough signal
- the right move is to narrow, not broaden

3. `The real unit of mechanism may not be a single (layer, head)`
- the architecture and the causal results both suggest vertical head families
  may matter more than isolated sites

4. `The whole-vector and per-channel stories need cleaner separation`
- whole-vector steering already showed that direct-vocab control exists
- per-channel work should now focus on what is *better* or *different* about
  channelized symbolic control

## Streamlined Path Forward

The next path should be simpler than the full original plan.

### Track A: establish the correct unit of mechanism

Stay on:

- model: `E2_independent`
- family: `recency_bias`

Ask only:

> is the meaningful unit a single head at one layer, or a vertical channel
> across layers?

Experiments:

1. one `(layer, head)` symbolic write at a time
2. same head index across multiple layers
3. neighboring-head controls
4. readout decomposition on the best candidates

If vertical channels dominate, the rest of the roadmap should be updated
accordingly.

### Track B: improve Phase 1 features

Do not add a feature zoo.

Add only the features most likely to matter:

1. vertical persistence as a ranking feature
2. cross-layer trajectory consistency
3. stream-transfer propensity
4. possibly a very small supervised calibration model over channel summaries

The goal is not to explain everything. It is to improve the probe-to-causal
bridge enough to justify the next stage.

### Track C: compare against whole-vector baselines cleanly

This question matters a lot:

> does per-channel symbolic intervention give better selectivity than a
> whole-vector write?

If yes, that is already a meaningful dual-stream result, even before composites.

Canonical experiment spec:

- [CHANNELIZED_VS_WHOLEVECTOR_SELECTIVITY_EXPERIMENT.md](doc/plans/CHANNELIZED_VS_WHOLEVECTOR_SELECTIVITY_EXPERIMENT.md)

That document should be treated as the authoritative plan for the next
decision-boundary experiment. The older
[CHANNELIZED_SELECTIVITY_EXPERIMENT.md](doc/plans/CHANNELIZED_SELECTIVITY_EXPERIMENT.md)
is retained only as a superseded draft.

## What Should Be Simplified

1. `Stay on one family until the method stabilizes`
- right now that should be recency on `E2`

2. `Do not move to mixed-token composites yet`
- the current probe-to-causal bridge is too weak

3. `Reduce the main bottleneck question`
- the immediate question is not:
  - "can we build symbolic programs?"
- it is:
  - "can we identify causally potent symbolic channels better than naive baselines?"

4. `Use the matched 71M pair for the right question`
- not layerwise discovery
- yes to symbolic-write and readout studies

## Missed Opportunities

These are the main opportunities still not fully exploited.

1. `Vertical channel as the primary object`
- this may deserve promotion from a later phase to the next immediate phase

2. `Small trained probe over channel summaries`
- not a large black-box model
- just a small supervised predictor to test whether a learnable mapping does
  better than the raw inner-product heuristic

3. `Cross-family selectivity tests`
- if a recency channel is truly specific, it should stay relatively quiet on
  induction and coreference

4. `Per-channel site comparison`
- post-block vs pre-block vs embedding-level writes

5. `Whole-vector vs sparse-channel selectivity`
- this may be the cleanest near-term architecture-specific result

## Likely Outcomes

Most likely near-term outcomes:

1. single-channel symbolic interventions will remain real
2. probe-to-causal alignment will improve somewhat, but probably not become
   near-perfect
3. a small number of head families will dominate the causal story
4. vertical structure will matter more than the current probe captures
5. mixed-token composites will disappoint if attempted too early

Less likely but important:

- the correct unit may turn out to be a vertical head family rather than a
  layer-local head slice

If that happens, the roadmap should pivot around vertical channels directly.

## Concrete Next Sequence

1. Stay on `E2` + `recency_bias`
2. Add vertical-channel features to Phase 1
3. Re-run Phase 1 to Phase 2 correlation
4. Run explicit vertical-channel causal interventions
5. Compare:
- single-channel
- vertical-channel
- whole-vector
6. Only if the probe-to-causal bridge improves materially, try sparse
   same-token composites
7. Leave mixed-token composites for later

## Stop/Go Gates

### Go to sparse composites only if:

- probe-to-causal correlation improves materially over the current weak result
- promoted channels separate more clearly from controls
- readout decomposition still shows a meaningful `x_t`-visible effect

### Stop and write the qualification result if:

- probing remains only weakly predictive of causal potency
- promoted channels do not separate from controls
- or apparent `x_t` success is mostly an `x_e` transfer story

That would still be a meaningful result:

- symbolic channels are causally writable
- but naive differential probing is not sufficient to recover the right ones

## Bottom Line

The key question now is not whether composites are possible.

The key question is:

> can we make the probe-to-causal bridge work well enough to justify
> compositional symbolic interventions?

That is the bottleneck that matters most. The roadmap should stay optimized for
that question until it is either solved or clearly shown to fail.
