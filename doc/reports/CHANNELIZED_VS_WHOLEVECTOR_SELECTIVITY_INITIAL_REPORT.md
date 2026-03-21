# Channelized vs Whole-Vector Selectivity: Initial E2 Readout

## Scope

This is the first implementation pass of the canonical experiment in
[CHANNELIZED_VS_WHOLEVECTOR_SELECTIVITY_EXPERIMENT.md](doc/plans/CHANNELIZED_VS_WHOLEVECTOR_SELECTIVITY_EXPERIMENT.md).

Important qualification:

- this was **not** the full exhaustive channel search
- it was a tractable first pass restricted to the `H5` family
- reason: earlier `E2` Phase 2 results showed the strongest recency effects
  concentrated in head 5, and the first exhaustive implementation was too blunt
  for CPU

So this report should be read as:

> a real first empirical readout under the updated spec, not yet the final
> decision experiment

Artifacts:

- [report.md](runs/selectivity_e2_recency_h5/report.md)
- [selectivity_comparison.json](runs/selectivity_e2_recency_h5/selectivity_comparison.json)
- shortlist widening follow-up:
  [report.md](runs/selectivity_e2_recency_topheads/report.md)
  and
  [selectivity_comparison.json](runs/selectivity_e2_recency_topheads/selectivity_comparison.json)

## Setup

- model: `E2_independent`
- target family: `recency_bias`
- target cases: `recency_001` to `recency_004`
- off-target pool: 19 valid `induction` / `coreference` cases
- stream: `x_t`
- stage: `post_block`
- leave-one-out selection over the 4 recency cases
- compared conditions:
  - `channelized`
  - `whole_vector`
  - `random_channel` control
  - `random_direction` control

The first reduced pass searched:

- channelized candidates: all layers, `head=5`, scales `{-4,-2,-1,1,2,4}`
- whole-vector candidates: all layers, scales `{-4,-2,-1,1,2,4}`

Follow-up widening check:

- channelized candidates widened to heads `{0, 4, 5}` with the same layer/scale grid

That follow-up selected the same best row and reproduced the same held-out
summary, so the first result was not just an artifact of forbidding obvious
neighboring candidates.

## Main Result

The first readout is **mixed but encouraging**.

Headline numbers:

- mean held-out channelized selectivity: `2.6827`
- mean held-out whole-vector selectivity: `2.7087`
- channelized wins on held-out cases: `3 / 4`

That means:

- channelized does **not** yet win on the aggregate mean
- but it does win on most held-out cases
- the comparison is close enough that the result is not a negative outcome
- the architecture-specific story remains alive, but not yet proven

## Case-Level Pattern

### `recency_001`

- channelized: `4.03`
- whole-vector: `5.87`

Whole-vector clearly wins here, but it does so with a much larger raw norm:

- channelized raw norm: `0.144`
- whole-vector raw norm: `0.781`

So this is not a clean argument against channelization. It is mostly evidence
that the current comparison still needs tighter budget matching in the full
experiment.

### `recency_002`

- channelized: `3.27`
- whole-vector: `2.28`

This is the clearest case-level channelized win in the reduced pass.

Also important:

- `random_channel`: `2.49`
- `random_direction`: `-0.22`

So the effect is not just "any small one-head perturbation works" and it is
definitely not "any random token direction works."

### `recency_003`

- channelized: `2.21`
- whole-vector: `1.83`

This is another real but modest channelized win.

The same pattern holds:

- random-channel is weaker than the target channel
- random-direction is strongly wrong-sign

### `recency_004`

- channelized: `1.21`
- whole-vector: `0.85`

Again channelized wins, but only modestly.

This is also the noisiest fold because both target effect and off-target drift
are relatively small.

## What Matters Most

Three things matter here.

### 1. The sanity check passed cleanly

The selected `H5` family response was monotonic on all 4 recency cases under
the same-sign scale sweep.

That means the comparison is not being built on a flaky target channel.

### 2. The controls behaved as they should

Across folds:

- random-direction was consistently much worse than the true channelized write
- random-channel was usually weaker than the target channel

That is important. It means the observed signal is not just an artifact of
small perturbation size.

### 3. Whole-vector is still very competitive

This is the main caution.

The reduced pass does **not** support a strong claim that channelized control is
already decisively better. At this point the honest statement is:

> `H5` channelized writes are competitive with whole-vector writes on held-out
> selectivity, and often better, but the advantage is not yet clean enough to
> call decisive.

## Follow-Up Widening Check

After the initial `H5` pass, the same experiment was rerun with a wider but
still tractable shortlist:

- candidate heads: `{0, 4, 5}`
- all layers
- same leave-one-out protocol

That run selected the same best channelized row:

- `L0 H5 scale=-1.0`

and reproduced the same fold-level comparison.

Interpretation:

- the first result is not an `H5`-only coding artifact
- the nearby plausible alternatives did not displace the selected channel
- the near-tie with whole-vector is therefore more credible

This strengthens the current reading:

- the channelized result is real
- but it is still modest rather than decisive

## Critical Interpretation

This is exactly the kind of intermediate result that should change the next
step.

What it does **not** justify:

- jumping to mixed-token composites
- declaring a solved channelized-control result
- broadening immediately to more families

What it **does** justify:

1. tightening the budget-matching in the full comparison
2. rerunning with the full channelized search once the implementation is made
   cheaper
3. keeping the random controls in the final design

## Best Reading of the Result

The best reading is:

- the channelized story survived first contact with held-out evaluation
- but the decisive experiment is still unresolved
- the right next move is the **full** canonical comparison, not a rhetorical
  interpretation of this reduced pass

If the full search preserves the same pattern:

- channelized wins on most folds
- aggregate mean is near-tied or slightly favorable
- random controls remain weaker

then the likely paper claim is:

> channelized intervention buys a modest but measurable selectivity advantage,
> not a dramatic one

That would still be a good result.
