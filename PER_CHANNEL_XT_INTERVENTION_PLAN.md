# Per-Channel `x_t` Intervention Plan

## Central Falsifiable Hypothesis

The central hypothesis of this plan is:

> channels ranked highly by differential probing are the channels with the
> strongest and cleanest causal effects under matched symbolic-slice
> intervention.

Everything else in the plan is either:

- a way to measure that ranking cleanly
- a gate for deciding whether a ranking is trustworthy
- or a follow-on experiment that is justified only if that hypothesis survives

If probing rank and causal rank do not line up, then:

- role labels are not trustworthy enough for composites
- Phase 4 should stop
- and the outcome should be written up as a negative or qualification result

## Why This Is The Right Next Experiment

The whole-vector direct-vocabulary experiments established something real:

- exact token directions can steer these models
- the effect can be local, bidirectional, and behaviorally meaningful

But those runs still treated the symbolic stream too much like an ordinary
residual space. They used one full `d_model` vector:

```text
Δv = W[token_A] - W[token_B]
```

and added that same object to `x_t` or `x_e`.

That is only the first layer of the intervention space. In the dual-stream
architecture, `x_t` is channelized by head. For a 6-head model:

```text
e_word = [e_word,1, e_word,2, ..., e_word,6]
```

and each slice is a different symbolic channel. That creates new choices that a
standard transformer does not expose:

- intervene on one head slice at a time
- compare head slices with different functional roles
- combine slices from different words into one composite symbolic write
- trace one vertical channel through depth instead of treating all heads as mixed

So the next question is not just:

> does `x_t` work?

It is:

> which channels in `x_t` do what, and what symbolic control do they support?

## Why One-Head-At-A-Time Is Not Enough

A one-head-at-a-time sweep is necessary, but it is not sufficient.

Suppose we write only head 3's slice of `e_Jacob - e_Natalie` and observe a
large effect. That tells us head 3 matters for this case. It does **not** yet
tell us:

- whether head 3 is an entity-binding channel, a generic answer-amplification
  channel, or a position-sensitive routing channel
- whether head 3 should be combined with head 1 from the same token or with a
  different token's slice
- whether head 3 should be paired with head 5 because those channels form a
  vertical processing pipeline

To build meaningful mixed-token symbolic interventions like

```text
[e_word1,1, e_word2,2, ..., e_word6,6]
```

we first need to know what each channel appears to do.

That is why the right workflow is:

1. differential probing
2. channel ranking and role assignment
3. one-channel causal intervention
4. channel-composition experiments

This mirrors the older ACL/LFA workflow, where heads were first ranked by
position-dependence or coreference specialization, and only then targeted by
intervention.

Concrete precedents:

- [README.md](companion-repo/acl-2026-late-fusion/README.md)
- [main.tex](companion-repo/acl-2026-late-fusion/arxiv-20260308/main.tex)
- [analyze_vertical_channels.py](companion-repo/acl-2026-late-fusion/analysis/attention/analyze_vertical_channels.py)
- [pronoun_resolution.py](companion-repo/acl-2026-late-fusion/analysis/shared/attention_analysis/analyzers/pronoun_resolution.py)

Implementation home:

- Phase 1 probing code should be written natively in
  [gpt_oss_interp/steering/probing.py](gpt_oss_interp/steering/probing.py)
- causal intervention helpers belong in
  [gpt_oss_interp/steering/interventions.py](gpt_oss_interp/steering/interventions.py)
- null baselines belong in
  [gpt_oss_interp/steering/controls.py](gpt_oss_interp/steering/controls.py)
- readout decomposition and transfer metrics belong in
  [gpt_oss_interp/steering/readouts.py](gpt_oss_interp/steering/readouts.py)
- run artifact schemas should be defined once in
  [gpt_oss_interp/common/artifacts.py](gpt_oss_interp/common/artifacts.py)

This is deliberate. New per-channel work should not start life in `scripts/`
and be reorganized later.

## What The ACL-Style Precedent Suggests

The older late-fusion / ACL work followed a consistent pattern:

1. build clean minimal pairs
2. measure per-head differential behavior on those pairs
3. rank heads by a task-relevant score
4. intervene on the top-ranked heads
5. compare against matched random or low-ranked controls

That is the right pattern here too. The difference is that the intervention
object is now a token-slice in `x_t`, not just a soft head gate.

In this repo, the dual-stream-specific analog should be:

1. build tokenizer-clean binary choice panels
2. measure per-channel symbolic preference and differential sensitivity
3. assign provisional channel roles
4. intervene on one symbolic channel at a time
5. only after that, try channel composites

## Decision Space

The full per-channel intervention design space is larger than the current
whole-vector sweep. We should make that explicit.

### Intervention object

1. Whole-vector token direction
- `Δv = e_A - e_B`

2. Single-channel token direction
- `Δv_h = e_A,h - e_B,h`

3. Single-channel token write
- `e_A,h`
- `e_B,h`

4. Multi-channel same-token write
- choose a subset of heads and write the same token's slices there

5. Mixed-token channel composite
- `[e_word1,1, e_word2,2, ..., e_wordH,H]`

### Intervention site

1. embedding initialization
2. pre-block `x_t`
3. post-block `x_t`
4. possibly direct value-path symbolic intervention later

### Intervention scope

1. one head at a time
2. top-k ranked heads
3. one vertical channel across all layers
4. one layer-head site

### Readout

1. combined
2. `x_t` only
3. `x_e` only
4. per-head slice readout diagnostics

The important point is that a standard transformer does not naturally give us
items 2-5 under "Intervention object." Those are specific to the preserved
channelized symbolic stream.

## Phase 1: Differential Probing

Before doing composite interventions, we need channel role hypotheses.

### Panel design

Use tokenizer-clean cases only. The current best families are:

- coreference
- induction
- recency

Avoid capitalization until the tokenization issue is fixed.

For each family, build small minimal-pair sets that change one factor at a
time:

- `coreference`: swap antecedent identity while preserving syntax
- `induction`: swap the repeated item while preserving pattern structure
- `recency`: swap the favored recent referent while preserving wording

### What to measure per channel

For each model, layer, head, and case:

1. `slice preference score`
- at the decision position, compare the current `x_t` slice to the token slices
  for `A` and `B`
- example:
```text
pref(ℓ,h) = <x_t[ℓ,h], e_A,h - e_B,h>
```

2. `differential preference shift`
- between matched minimal-pair conditions, how much does that score move in the
  predicted direction?

3. `stability`
- does the same head behave similarly across paraphrases in the same family?

4. `selectivity`
- does the head respond strongly in-family but weakly off-family?

5. `position sensitivity`
- is the effect strongest at the decision token and weak at unrelated tokens?

6. `vertical persistence`
- does the same head index remain informative across multiple layers?

### Output of Phase 1

For each family, produce:

- a ranked list of channels
- a provisional role label per channel
- evidence for that label

Example role labels:

- entity-binding
- answer-selector
- recent-mention tracker
- copy / pattern-completion
- generic amplifier
- diffuse / no clear role

At this stage the labels are hypotheses, not truths. The point is to constrain
the later intervention search.

### Required null baselines

Phase 1 must include explicit nulls. Otherwise high differential scores could
come from token-embedding geometry rather than functional specialization.

Required controls:

1. `shuffled-head control`
- preserve scores but randomly permute head identities

2. `matched random-direction control`
- replace `e_A,h - e_B,h` with a same-norm random token-slice direction

3. `within-family label permutation`
- shuffle which condition is treated as the positive direction within a family

These controls should be reported beside the real scores, not left to an
appendix.

### Promotion gate

A channel only advances to Phase 2 if it clears both gates below:

1. `held-out sign prediction`
- on held-out minimal pairs from the same family, the channel's differential
  preference score must predict the correct sign at least `70%` of the time

2. `null-beating requirement`
- that held-out accuracy must exceed the matched null controls by a meaningful
  margin

The threshold is fixed now to avoid post-hoc gate adjustment.

### Partial-pass policy

The plan must handle mixed outcomes cleanly.

If a 6-head model yields:

- 2 strong channels
- 2 ambiguous channels
- 2 channels that fail the gate

then Phase 2 proceeds on the 2 channels that passed.

The headline claim scales to the fraction that passed:

- if only 1-2 channels pass, the claim is sparse and local
- if most channels fail, the result is not "the model has no structure"
- it is "the probing method only identified a small reliable subset"

Ambiguous channels are not promoted to composite design.

## Phase 2: One-Channel Causal Intervention

Once channels are ranked, test whether the high-scoring channels are actually
causal.

### Main experiment

For one family at a time:

1. choose the top-ranked channel
2. intervene with only that channel slice
3. sweep layer and scale
4. compare against:
- whole-vector baseline
- random channel control
- low-ranked channel control

### Central deliverable of Phase 2

Phase 2 is not just an intervention sweep. It is the first direct test of the
central hypothesis.

Required output:

- rank correlation between Phase 1 probing rank and Phase 2 causal rank

If that correlation is weak, then Phase 1 did not identify the right channels
and Phase 4 should not proceed.

### Primary metrics

Use the same behavioral metrics as the current steering work:

- local first-divergent-token gap shift
- total choice-score shift
- tail fraction
- off-target drift

Add one channel-specific metric:

- `channel efficiency = local shift / ||intervention slice||`

That should be treated as a descriptive ratio, not as proof of architectural
importance. It must be compared against same-norm random-direction controls.

### Stream-transfer metric

Because a successful `x_t` intervention may act either:

- directly through symbolic readout
- or indirectly after transferring into `x_e`

Phase 2 must report a concrete stream-transfer metric.

For a matched intervention at layer `\ell` and head `h`, define:

```text
transfer(ℓ,h) = |effect in x_e-only readout| / max(|effect in x_t-only readout|, ε)
```

where "effect" is the local first-divergent-token gap shift at the decision
position under the same intervention.

Interpretation:

- `transfer ≈ 0`: effect remains primarily symbolic / `x_t`-visible
- `transfer ≈ 1`: effect is comparably visible in both streams
- `transfer >> 1`: apparent `x_t` success is mostly visible through `x_e`

That quantity is important because it changes the interpretation of what a
"successful `x_t` intervention" actually means.

### What success looks like

A channel is compelling if:

- it causes the predicted local shift on its own
- it outperforms a matched random channel
- it is more selective than the whole-vector baseline
- its effect aligns with the role suggested by the differential probing
- its stream-transfer profile matches the intended interpretation

### What failure looks like

Phase 2 counts as a substantive negative result if:

- promoted channels do not outperform random or low-ranked controls
- probing rank and causal rank show weak correspondence
- successful `x_t` interventions are only behaviorally visible through `x_e`
- or promoted labels fail to predict held-out behavior in the causal setting

## Phase 3: Vertical-Channel Intervention

The architecture suggests a second dual-stream-specific idea: same-index heads
may function as a vertical pipeline across depth.

So after one-channel intervention, test:

- write the same symbolic slice into head `h` at one layer
- then compare with writing it into head `h` across a vertical sequence of
  layers

This asks a different question from Phase 1. Phase 1 measures whether the same
head index carries persistent information across depth. Phase 3 asks whether
that head index is causally acting as:

- a local amplifier at isolated layers
- or a cross-layer relay / vertical pathway

That distinction matters for composite design:

- if channels behave locally, composites should be layer-local
- if channels behave like vertical relays, composites should respect the
  channel pipeline across depth

The older ACL "vertical channel" analysis is directly relevant here.

## Phase 4: Composite Symbolic Intervention

This is the experiment that originally motivated the question.

Once we have role-labeled channels, we can try composites like:

```text
[e_Jacob,entity, e_he,pronoun, e_lock,action, ...]
```

or more realistically for the current setup:

```text
[e_A, h1, e_A, h3, e_B, h4, e_A, h5]
```

where the head choices are informed by the probing results.

### Composition tests

1. `same-token sparse composite`
- only the top 1-3 channels from token `A - B`

2. `mixed-token functional composite`
- channels from different token slices chosen by hypothesized role

3. `anti-composite control`
- deliberately mismatched channel-token assignments

The central question is whether informed composites outperform:

- random composites
- full-vector writes
- one-channel-only writes

This phase is fully contingent on the earlier gates. If the probing-to-causal
link is weak, arbitrary composites should not be treated as evidence for
symbolic compositionality.

## Practical First Pass

The first concrete pass should stay narrow.

### Model

- `C-71` first
- then compare with `SS-71`

### Case families

- `coref_009`
- `induction_005`
- `recency_001`

These are already the strongest demo cases.

### First probing output

For each case family:

1. compute per-layer, per-head slice preference scores
2. compare across a small family-specific minimal-pair set
3. rank channels by differential score
4. visualize:
- head-by-layer heatmaps
- vertical channel trajectories
- per-family top-channel tables

### First causal follow-up

Then run:

1. top-1 channel slice intervention
2. random-channel control
3. whole-vector baseline

If that already shows cleaner or more efficient steering, the main claim gets
much stronger.

## What We Probably Still Have Not Tested

This new plan closes one major gap, but it also clarifies what remains missing.

Likely untested choices still include:

- direct value-path slice intervention instead of stream-state injection
- gate-path symbolic intervention
- per-head readout from the unembedding rather than only stream-level readout
- channel composites that mix role types across families
- whether channel roles transfer across models of the same architecture family

So yes: the earlier omission was real, and it probably means there are still
other architecture-specific choices left unexplored. The right response is not
to distrust the existing results, but to make the intervention space explicit
and test it systematically.

## Bottom Line

The per-channel experiment should not begin with arbitrary mixed-token
composites. That would be too unconstrained.

The right order is:

1. differential probing to identify candidate channel roles
2. one-channel-at-a-time causal writes
3. vertical-channel tests
4. informed composite symbolic interventions

That is the cleanest way to turn the dual-stream architecture's extra
intervention freedom into a real scientific result rather than an anecdotal
demo.
