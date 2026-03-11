# Embedding-Level `x_t` vs `x_e` Intervention Report

## Why This Variant Matters

If `x_t` is really the symbolic write channel, then one of the cleanest ways to test that claim is to intervene at initialization, before any blocks run at all.

That is what this experiment does.

Instead of adding the vocabulary direction before or after some later block, we add it at the start of the network:

- once to `x_t`
- once to `x_e`

Then we let the full model run forward normally.

This is a sharper version of the original question:

- does the symbolic stream look special when we intervene at the very beginning?

Important scope note:

- here too, "`x_t` intervention" means adding one whole vector to the symbolic
  stream at initialization
- it does **not** yet mean exploiting the head-sliced channel structure of
  `x_t`

That untested intervention family is now tracked in
[CHANNELIZED_XT_INTERVENTION_NOTE.md](doc/notes/CHANNELIZED_XT_INTERVENTION_NOTE.md).

## Setup

Same models:

- `SS-71`
- `C-71`

Same cases:

- `coref_009`
- `induction_005`
- `recency_001`
- `coref_006`

Same scales:

- `-8, -6, -4, -2, -1, 1, 2, 4, 6, 8`

Because the intervention happens at initialization, there is no meaningful layer sweep. So the run uses a single nominal layer index only to keep the output format consistent.

Artifacts:

- embedding-init `x_t`: [direct_vocab_steering.json](runs/direct_vocab_embedinit_ablation_xt/direct_vocab_steering.json)
- embedding-init `x_e`: [direct_vocab_steering.json](runs/direct_vocab_embedinit_ablation_xe/direct_vocab_steering.json)

## The Headline Result

This variant finally starts to tease the streams apart, but only in part.

What we saw:

- in `SS-71`, embedding-level `x_t` and embedding-level `x_e` were again effectively identical
- in `C-71`, the two streams began to separate modestly

So the clean story is not:

- "`x_t` is obviously special everywhere"

It is:

- `SS-71` is so sensitive to early intervention that `x_t` and `x_e` still collapse together
- `C-71` is where the first meaningful stream-level differences start to appear

## A Concrete Example: `recency_001`

Prompt:

`The trophy would not fit in the suitcase because the suitcase was too small. The word 'small' refers to the`

Choices:

- `A = " suitcase"`
- `B = " trophy"`

This case is useful because in the CASCADE model it separates the two streams more clearly than the earlier variants.

### Embedding-Level `x_t`

For `C-71`, embedding-level `x_t`:

- did flip the case
- best negative total gap reached `-0.191`

That is not a huge flip, but it crosses zero.

### Embedding-Level `x_e`

For `C-71`, embedding-level `x_e`:

- did **not** flip the case
- best negative total gap only reached `+0.128`

So on this case, at this intervention stage, `x_t` is stronger than `x_e`.

That is the first result in this sequence that clearly points in the direction of the original architectural intuition.

## Another Example: `induction_005`

Prompt:

`sun moon star sun moon star sun moon`

Choices:

- `A = " star"`
- `B = " cloud"`

For `C-71`:

- embedding-level `x_t`: best negative gap `-0.454`
- embedding-level `x_e`: best negative gap `-0.977`

So here both streams flip, but `x_e` is actually stronger.

This is why the result is still only partial. We are not yet seeing a clean universal rule that says embedding-level `x_t` always dominates embedding-level `x_e`.

## The `SS-71` Story

For `SS-71`, the picture is very simple:

- `coref_009`: flip under both streams
- `induction_005`: flip under both streams
- `recency_001`: flip under both streams
- `coref_006`: flip under both streams

And the numbers are identical between embedding-level `x_t` and embedding-level `x_e`.

That strongly suggests that in `SS-71`, early interventions are just generally overpowering. Once you intervene at initialization, the model seems happy to route that change through the rest of the network no matter which stream you use.

So `SS-71` is not the best model for demonstrating a clean stream distinction.

## The `C-71` Story

For `C-71`, the picture is more nuanced:

- `coref_009`
  - `x_t`: no flip, best negative `+3.236`
  - `x_e`: no flip, best negative `+3.544`
- `induction_005`
  - `x_t`: flip, best negative `-0.454`
  - `x_e`: flip, best negative `-0.977`
- `recency_001`
  - `x_t`: flip, best negative `-0.191`
  - `x_e`: no flip, best negative `+0.128`
- `coref_006`
  - `x_t`: no flip, best negative `+9.748`
  - `x_e`: no flip, best negative `+9.695`

This is not a dramatic separation, but it is the clearest one so far.

The most informative point is:

- on `recency_001`, embedding-level `x_t` flips while embedding-level `x_e` does not

That is exactly the kind of pattern we were hoping to eventually find.

## What This Means

This variant gives the first real support for the idea that `x_t` can sometimes behave like a more direct symbolic control channel.

But it is not yet strong enough to justify a sweeping claim.

The evidence at this point is:

- `SS-71`: stream distinction still largely collapses
- `C-71`: some cases begin to distinguish the streams

So the architectural intuition is starting to show up, but only weakly and only in some settings.

## Comparison To Earlier Variants

Relative to the earlier runs:

- post-block:
  - `x_t` and `x_e` were nearly identical
- pre-block:
  - timing mattered a lot, but stream separation was still weak
- embedding-level:
  - `SS-71` still shows near-collapse
  - `C-71` begins to separate the streams modestly

That progression is useful. It suggests the issue was not just “we need any sharper intervention.” It suggests the intervention has to be both:

- early enough
- and structurally close enough to the symbolic stream's role

to expose a difference.

## Bottom Line

This is the first variant that gives partial empirical support to the symbolic-stream intuition.

Not because `x_t` suddenly dominates everywhere. It does not.

But because:

- in the CASCADE model
- at initialization
- on at least one clean case

`x_t` can succeed where `x_e` does not.

That is a meaningful result, even if it is still not the full clean separation we ultimately want.
