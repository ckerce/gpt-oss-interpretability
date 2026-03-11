# Pre-Block `x_t` vs `x_e` Intervention Report

## Why This Was The Next Experiment

The first `x_t` vs `x_e` comparison used a **post-block** hook. That left an obvious concern:

- maybe intervening after a block is too late
- maybe at that point both streams are already collapsed into a similar downstream effect

So the next sharpest test was:

- keep the same token direction
- keep the same cases
- keep the same layer/scale sweep
- but intervene **before** the selected block instead of after it

If the streams really differ functionally, this should be a better place to see it.

## Setup

Same models:

- `SS-71`
- `C-71`

Same cases:

- `coref_009`
- `induction_005`
- `recency_001`
- `coref_006`

Same sweep:

- layers `0..5`
- scales `-8, -6, -4, -2, -1, 1, 2, 4, 6, 8`

Only the intervention timing changed:

- old ablation: post-block
- this ablation: pre-block

Artifacts:

- pre-block `x_e`: [direct_vocab_steering.json](runs/direct_vocab_preblock_ablation_xe/direct_vocab_steering.json)
- pre-block `x_t`: [direct_vocab_steering.json](runs/direct_vocab_preblock_ablation_xt/direct_vocab_steering.json)

## The Headline Result

Moving the intervention **before** the block mattered a lot.

But the effect was not simply:

- "`x_t` now looks trivial and `x_e` now looks hard"

Instead, the result was:

- in `SS-71`, pre-block intervention became much stronger for **both** streams
- in `C-71`, there was some stream-specific divergence, but not a clean separation

So timing clearly matters. Stream identity still matters less cleanly than the original intuition suggested.

## A Concrete Example: `coref_006`

Prompt:

`Megan called Thomas after he sent the address. The word 'he' refers to`

Choices:

- `A = " Thomas"`
- `B = " Megan"`

This is the most useful worked example because under the earlier post-block ablation it was a control case:

- the steering direction moved the margin
- but it did **not** flip the answer

### What Happened Before

Under the earlier post-block comparison:

- `SS-71`: no flip, best negative gap `+0.806`
- `C-71`: no flip, best negative gap `+5.985`

So `coref_006` acted like a “strong movement, no reversal” case.

### What Happened With Pre-Block `x_e`

Now the important change.

For `SS-71`, pre-block `x_e` at layer `0`, scale `-8.0`:

- pushed the total gap all the way to `-4.863`
- fully flipped the answer from `Thomas` to `Megan`

For `C-71`, pre-block `x_e` did **not** flip:

- best negative gap stayed at `+8.271`

So simply moving the intervention earlier in the network made the single-stream model much easier to overturn, but not the CASCADE model.

### What Happened With Pre-Block `x_t`

For `SS-71`, pre-block `x_t` looked essentially the same as pre-block `x_e`:

- best negative gap `-4.863`
- same layer and scale
- same flip behavior

For `C-71`, pre-block `x_t` also failed to flip:

- best negative gap `+8.210`

So `coref_006` tells the story clearly:

- timing mattered a lot
- stream identity still did not create a dramatic split
- the biggest separation here was between `SS-71` and `C-71`, not between `x_t` and `x_e`

## Broader Pattern

### `SS-71`

Pre-block intervention became extremely strong.

All four curated cases flipped under both streams:

- `coref_009`
- `induction_005`
- `recency_001`
- `coref_006`

And the strongest settings were concentrated at:

- layer `0`
- scale `-8.0`

That is a very strong sign that in `SS-71`, early pre-block intervention can dominate the decision process almost regardless of whether the vector is injected into `x_t` or `x_e`.

### `C-71`

The picture was more mixed.

- `coref_009`: no flip under either stream
- `coref_006`: no flip under either stream
- `induction_005`: flip under both streams, but `x_e` was stronger
- `recency_001`: flip under both streams, with `x_t` slightly stronger

This is more interesting than the first post-block ablation because it shows some nontrivial variation, but it still does not give a clean “`x_t` easy / `x_e` hard” separation.

## Side-By-Side Summary

### `SS-71`

- `coref_009`
  - pre-block `x_e`: flip, best negative gap `-2.578`
  - pre-block `x_t`: flip, best negative gap `-2.578`
- `induction_005`
  - pre-block `x_e`: flip, best negative gap `-6.184`
  - pre-block `x_t`: flip, best negative gap `-6.184`
- `recency_001`
  - pre-block `x_e`: flip, best negative gap `-7.222`
  - pre-block `x_t`: flip, best negative gap `-7.222`
- `coref_006`
  - pre-block `x_e`: flip, best negative gap `-4.863`
  - pre-block `x_t`: flip, best negative gap `-4.863`

### `C-71`

- `coref_009`
  - pre-block `x_e`: no flip, best negative gap `+2.758`
  - pre-block `x_t`: no flip, best negative gap `+2.753`
- `induction_005`
  - pre-block `x_e`: flip, best negative gap `-3.584`
  - pre-block `x_t`: flip, best negative gap `-2.555`
- `recency_001`
  - pre-block `x_e`: flip, best negative gap `-0.656`
  - pre-block `x_t`: flip, best negative gap `-0.748`
- `coref_006`
  - pre-block `x_e`: no flip, best negative gap `+8.271`
  - pre-block `x_t`: no flip, best negative gap `+8.210`

## What Changed Relative To The Post-Block Result

Three things changed.

### 1. Timing mattered a lot

This is the clearest conclusion.

Changing the intervention from post-block to pre-block produced much larger effects, especially in `SS-71`.

That means the previous post-block equivalence was not the whole story. Where in the computation you intervene matters at least as much as which stream you choose.

### 2. `SS-71` became easy to overturn

In `SS-71`, pre-block interventions at layer `0` were powerful enough to flip every curated case, including the previous control case `coref_006`.

This suggests that for `SS-71`, the earliest block is a very sensitive intervention site.

### 3. `C-71` still did not give a clean stream split

Even with pre-block interventions, the CASCADE model did not cleanly separate into:

- trivial `x_t`
- nontrivial `x_e`

There were some quantitative differences, but not the sharp qualitative split we were hoping for.

## Interpretation

The best reading of this result is:

- the **timing** of the intervention was underappreciated
- the current **stream identity** distinction is still not being isolated sharply enough

In plain terms:

- if you intervene earlier, you can change more of what the block does
- but changing the stream label from `x_t` to `x_e` still does not automatically produce a clean mechanistic separation

So this experiment strengthens one claim and weakens another:

- stronger claim: pre-block interventions are much more causally potent than post-block ones
- weaker claim: current evidence still does not support a simple empirical slogan that `x_t` intervention is obviously trivial while `x_e` intervention is obviously nontrivial

## Why This Is Still Progress

This was not a dead end. It clarified the next move.

We now know:

- post-block hooks were too late to expose much structure
- pre-block hooks are more revealing
- but even pre-block hooks are not enough by themselves to resolve the conceptual distinction fully

So the next variants should focus on:

- embedding-level `x_t` intervention
- readout decomposition
- wrong-position controls

Those are the most likely to separate “direct symbolic write” from “computational steering.”

## Bottom Line

The pre-block ablation changed the story in an important way.

It showed that:

- intervention timing is a first-order variable
- early interventions can be dramatically stronger than late ones
- but the `x_t`/`x_e` distinction is still not cleanly resolved by this intervention family alone

That means the next analyses should not just keep repeating the same sweep. They should target the remaining places where the streams can actually differ in function.
