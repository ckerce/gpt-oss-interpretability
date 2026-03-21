# Channelized vs Whole-Vector Selectivity Experiment

## Purpose

This is the next decisive experiment.

It asks the question that matters most right now:

> does channelized symbolic intervention provide better selective causal control
> than a whole-vector direct-vocabulary write?

If the answer is no, then much of the per-channel program becomes an elaborate
way of doing something simpler methods already do. If the answer is yes, even
modestly, then the dual-stream/channelized architecture has justified one of
its main claims.

This experiment is therefore a decision boundary, not just another ablation.

## Why This Experiment Comes First

The current program has already established:

- whole-vector direct-vocabulary steering is real
- single-channel symbolic interventions are real
- the probe-to-causal bridge is weaker than hoped

What is still unresolved is whether channelized intervention actually buys a
capability that whole-vector steering does not.

That is the bottleneck question.

## Core Comparison

Compare two intervention families:

1. `Whole-vector intervention`
- full direct-vocabulary vector:
  `Δv = e_A - e_B`

2. `Channelized intervention`
- single head slice:
  `Δv_h = e_A,h - e_B,h`

The comparison is only meaningful if both interventions are evaluated under the
same causal budget and the same search freedom.

## Primary Hypothesis

The primary hypothesis is:

> channelized symbolic interventions can achieve similar target-local control
> with lower off-target drift than whole-vector interventions.

That is a selectivity claim, not a raw-effect-size claim.

## Experimental Scope

### Model

Primary model:

- `E2_independent`

Reason:

- `x_t` is mutable across layers
- Phase 1 probing is nondegenerate there
- recency already yields promoted channels

### Task family

Primary family:

- `recency_bias`

Reason:

- this is currently the only family with a stable promoted channel set under
  the full Phase 1 gate

### Stream and site

First pass:

- stream: `x_t`
- stage: `post_block`
- position: decision token

Hold these fixed for the initial comparison.

Do **not** mix in:

- pre-block vs post-block
- `x_e`
- embedding-init
- vertical multi-layer interventions

Those are follow-up questions, not part of the first decision experiment.

## Search Freedom

The two intervention families must be given matched search freedom.

### Whole-vector search

Allowed:

- all layers
- all scales in the allowed scale grid

Not allowed:

- multi-layer combinations
- family-specific learned vectors
- extra tuning freedom not available to the channelized case

### Channelized search

Allowed:

- all `(layer, head)` sites
- all scales in the same allowed scale grid

Not allowed:

- multi-head combinations
- vertical composites
- arbitrary mixed-token composites

This keeps the comparison fair:

- whole-vector gets to search over layer
- channelized gets to search over layer and head
- neither gets extra nonlinear optimization freedom

## Hold-Out Selection Policy

Selection and evaluation must not happen on the same four recency cases.

Because the target pool is only:

- `recency_001`
- `recency_002`
- `recency_003`
- `recency_004`

the experiment must use leave-one-out evaluation.

### Required procedure

For each held-out recency case:

1. select the best `whole-vector` row using the other 3 recency cases
2. select the best `channelized` row using the other 3 recency cases
3. evaluate both selected rows on the held-out case

Where:

- `whole-vector row = (layer, scale)`
- `channelized row = (layer, head, scale)`

The primary reported result is:

- mean held-out selectivity across the 4 leave-one-out folds

The experiment should also report:

- all 4 held-out case rows individually
- how often channelized beats whole-vector on held-out selectivity

This is the minimum needed to avoid overclaiming from a tiny target pool.

## Intervention Budget

The comparison will be confounded unless the intervention budget is matched.

### Required reporting

For every compared row, report:

1. raw intervention norm
2. target local effect
3. effect per unit norm

### Matching rule

The preferred comparison is:

- compare rows under a matched intervention norm budget

If exact norm matching is inconvenient in the first implementation, then report
both:

1. `best target effect under the sweep`
2. `best selectivity under the sweep`

with explicit raw norms shown beside each.

This avoids the common failure mode where one method is declared better simply
because it was allowed a larger effective perturbation.

### Control matching

Any control condition must also be norm-matched.

In particular:

1. `random-channel control`
- same direction
- different head
- rescaled to the target-head intervention norm

2. `random-direction control`
- same head
- different token-pair direction
- rescaled to the target-head intervention norm

Otherwise the control comparison is confounded by slice-norm differences rather
than channel identity or direction specificity.

## Metrics

### Primary metric: selectivity

For each intervention row, define:

```text
selectivity = target_local_effect / (mean_abs_off_target_local_effect + ε)
```

This is the primary comparison.

Why:

- raw target effect alone is not enough
- a larger but sloppier intervention is not better

The primary analysis should be performed on:

- held-out rows selected by leave-one-out evaluation

not on averages over all rows in the sweep.

### Required companion metrics

1. `target local effect`
- first-divergent-token logit gap shift on the target family

2. `target total effect`
- full forced-choice score shift

3. `mean absolute off-target local effect`
- measured on held-out control cases outside the recency family

4. `non-target KL`
- KL divergence at the decision token after removing the A/B choice tokens and
  renormalizing the remaining vocabulary mass
- this is the primary same-case collateral metric
- it measures how much the intervention disturbs the rest of the local
  distribution beyond the intended A/B movement

5. `tail fraction`
- to ensure the effect is local, not suffix-mediated

6. `stream-transfer ratio`
- to determine whether apparent `x_t` success is actually mediated through `x_e`

### Off-target controls

The experiment should use two classes of off-target controls.

#### Cross-family controls

Use control cases from:

- induction
- coreference

These should be measured with the same intervention row, not tuned separately.

These are useful, but they are not enough by themselves.

#### Matched intervention controls

Also include:

1. `norm-matched random-channel`
- same target-family direction
- different head
- norm-matched to the target-head write

2. `norm-matched random-direction`
- same target head
- different token-pair direction
- norm-matched to the target-head write

These controls separate:

- channel identity
- direction specificity
- simple perturbation-size effects

#### Geometric alignment check

Before interpreting cross-family drift, report:

- whole-vector cosine similarity between each recency direction and each
  cross-family case direction
- per-head-slice cosine similarity for the same pairs

If cross-family drift is high when geometric alignment is also high, that result
should be interpreted as contaminated rather than as clean evidence about
selectivity.

#### Exact control pool

The off-target case set is fixed at experiment design time:

- `induction_001`: A7 B2 C9 D4 A7 B2 C9 (D4 vs E5)
- `induction_003`: 1 2 3 4 1 2 3 4 1 2 3 (4 vs 5)
- `induction_004`: alpha beta gamma alpha beta gamma alpha beta (gamma vs delta)
- `coref_001`: Alice gave her old laptop to Bob (Alice vs Bob)
- `coref_003`: John called Mary and told her... (Mary vs John)
- `coref_004`: The mother picked up the child... (mother vs child)

Six off-target cases across two families. This set was chosen to avoid
`induction_002` (known pathology) and to sample broadly across both families
without exhaustive enumeration.

### Baseline sanity check

Before running the full comparison matrix, verify that the selected target
channel family actually produces a usable response on the target family.

Minimum sanity bar:

- the top candidate channel should show a monotonic or near-monotonic target
  local effect over the scale sweep on at least 3 of 4 recency cases

If this does not hold, the full selectivity comparison is premature.

## Stop/Go Decision Criteria

The stop/go decision is based on the leave-one-out held-out results, not
aggregate best-row numbers.

### Go: channelized is more selective

All three conditions must hold:

1. channelized held-out selectivity exceeds norm-matched whole-vector held-out
   selectivity in at least **3 of 4** leave-one-out folds
2. the median channelized selectivity advantage exceeds the interquartile
   spread across folds (the signal exceeds the noise)
3. channelized held-out selectivity exceeds both norm-matched random-channel
   and norm-matched random-direction controls in at least **3 of 4** folds

### Stop: no meaningful selectivity advantage

Either condition is sufficient:

1. channelized and norm-matched whole-vector are interleaved across folds with
   no consistent direction (channelized wins 2 folds, loses 2)
2. norm-matched random-channel controls match channelized selectivity in 3+
   folds

In the stop case:

> Architectural channelization does not provide selectivity beyond what is
> achievable with appropriately scaled whole-vector writes. The dual-stream
> architecture may still have value for inspectability and bounded cost, but the
> per-channel intervention story does not add meaningful causal modularity.

That would still be a publishable negative result.

### Ambiguous

Channelized wins 3/4 folds but the advantage is within the interquartile
spread. In this case:

- report the result honestly as suggestive but not decisive
- proceed cautiously to interchange intervention for a stronger causal test
- do not make strong selectivity claims

## What Counts as a Win

Channelized wins if it achieves either:

1. similar target-local effect with materially lower off-target drift

or

2. somewhat smaller target effect but substantially better selectivity

Whole-vector does **not** automatically win by producing the largest raw shift.

The point of the architecture is not to maximize blunt force. It is to improve
causal specificity.

## What Counts as an Unusable Comparison

This experiment should be rejected as inconclusive if any of the following
happen:

1. intervention norms are not reported
2. whole-vector and channelized searches are not comparably constrained
3. only total effect is reported without local effect
4. off-target effects are not measured
5. the comparison quietly uses different control pools for the two methods
6. selection and evaluation are performed on the same target cases
7. control rows are not norm-matched to the target intervention budget

These are all common ways of accidentally manufacturing a result.

## Minimal Output

The experiment should produce:

1. one artifact for all evaluated rows
2. one summary report
3. one compact comparison table with:
- best whole-vector row
- best channelized row
- matched random-channel row
- matched random-direction row
- target local effect
- off-target drift
- non-target KL
- selectivity
- tail fraction
- stream-transfer ratio
- raw norm

4. one short interpretation note:
- did channelized win on selectivity or not?

5. one held-out summary table with:
- held-out case id
- selected whole-vector row
- selected channelized row
- held-out selectivity for each
- whether channelized won on that held-out case

## Path After This Experiment

### If channelized clearly wins

Then:

1. run minimal interchange intervention on the top recency channel(s)
2. run vertical-channel vs single-site comparison
3. revisit improved probing only after the causal unit is better understood

### If the result is mixed

Then:

1. refine the metric or norm matching if needed
2. rerun once
3. if still mixed, treat the benefit as modest and avoid inflated claims

### If whole-vector clearly matches or beats channelized

Then:

1. stop prioritizing mixed-token composites
2. reframe the architecture story around bounded-cost interpretability and
   causal modularity, not superior control
3. reconsider whether per-channel discovery is worth further investment

## Simplifications

To keep this experiment honest and readable:

- use one model
- use one target family
- use one stream
- use one intervention stage
- do not include interchange yet
- do not include vertical composites yet
- do not include scaling yet

This is the smallest comparison that can answer the actual question.

## Bottom Line

This experiment should be treated as the highest-priority next test.

It answers the most important unresolved question in the current program:

> does channelized symbolic intervention provide a real selectivity advantage
> over whole-vector steering?

Everything else is downstream of that answer.
