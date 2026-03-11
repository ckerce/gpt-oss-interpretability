# `x_t` vs `x_e` Intervention Report

## Why We Ran This Experiment

The dual-stream model exposes two different internal streams:

- `x_t`: the symbolic/token-like stream
- `x_e`: the contextual/computational stream

The architectural intuition is:

- writing to `x_t` should be comparatively easy or even "trivial," because it is already close to token space
- writing to `x_e` should be the harder and therefore more interesting test, because `x_e` carries the model's evolving contextual computation

That is the conceptual story.

This experiment asked a narrower empirical question:

- if we take the same direct-vocabulary direction and inject it into `x_t` instead of `x_e`, do we actually see a different outcome?

## The Setup

We kept almost everything fixed.

Models:

- `SS-71`
- `C-71`

Cases:

- `coref_009`
- `induction_005`
- `recency_001`
- `coref_006`

Sweep:

- layers `0, 1, 2, 3, 4, 5`
- scales `-8, -6, -4, -2, -1, 1, 2, 4, 6, 8`

Intervention object:

- exact token direction `W[token_A] - W[token_B]`

Only one thing changed between the two runs:

- Run 1 added the vector to `x_e`
- Run 2 added the same vector to `x_t`

Important limitation:

- in this report, "`x_t` intervention" still means writing one whole vector
  into the symbolic stream
- we did **not** yet test the richer dual-stream-specific option of head-sliced
  or composite channelized `x_t` interventions

That larger intervention family is documented in
[CHANNELIZED_XT_INTERVENTION_NOTE.md](/mnt/c/Users/ckerc/Documents/job_applications/openai-interpretability/gpt-oss-interp/doc/notes/CHANNELIZED_XT_INTERVENTION_NOTE.md), with the concrete follow-up protocol in
[PER_CHANNEL_XT_INTERVENTION_PLAN.md](/mnt/c/Users/ckerc/Documents/job_applications/openai-interpretability/gpt-oss-interp/doc/plans/PER_CHANNEL_XT_INTERVENTION_PLAN.md).

Artifacts:

- `x_e` run: [direct_vocab_steering.json](/mnt/c/Users/ckerc/Documents/job_applications/openai-interpretability/gpt-oss-interp/runs/direct_vocab_stream_ablation_xe/direct_vocab_steering.json)
- `x_t` run: [direct_vocab_steering.json](/mnt/c/Users/ckerc/Documents/job_applications/openai-interpretability/gpt-oss-interp/runs/direct_vocab_stream_ablation_xt/direct_vocab_steering.json)

## The Result In One Sentence

With the current post-block hook, `x_t` and `x_e` behaved almost the same.

That is the central result.

It is not what the simple conceptual story would have led us to expect, and that is exactly why it matters.

## A Concrete Example: `coref_009`

Prompt:

`Natalie reminded Jacob that he needed to lock the door. The word 'he' refers to`

Choices:

- `A = " Jacob"`
- `B = " Natalie"`

Steering direction:

- `W[" Jacob"] - W[" Natalie"]`

This is a good example because it is clean, readable, and already familiar from the main demo.

### What Happened With `x_e`

For `SS-71`, the best negative `x_e` intervention:

- used layer `5`, scale `-8.0`
- flipped the model from `Jacob` to `Natalie`
- moved the total gap from `+5.242` to `-3.827`

For `C-71`, the best negative `x_e` intervention:

- used layer `5`, scale `-8.0`
- also flipped the model
- moved the total gap from `+5.967` to `-0.556`

### What Happened With `x_t`

Now the surprising part.

For the same case, same models, same direction, same sweep:

- the best negative `x_t` intervention landed at the same layer and scale
- it produced the same flip behavior
- it produced essentially the same total-gap change

For `SS-71`:

- `x_e`: best negative total gap `-3.827`
- `x_t`: best negative total gap `-3.827`

For `C-71`:

- `x_e`: best negative total gap `-0.556`
- `x_t`: best negative total gap `-0.556`

So if a reader expected:

- "`x_t` should be obviously easier and much stronger"

that is **not** what this first experiment showed.

## The Broader Pattern

The same story repeated across the other curated cases.

### Cases That Flipped Under Both Streams

- `coref_009`
- `induction_005`
- `recency_001`

### Case That Did Not Flip Under Either Stream

- `coref_006`

### Side-by-Side Summary

#### `SS-71`

- `coref_009`
  - `x_e`: flip, best negative total gap `-3.827`
  - `x_t`: flip, best negative total gap `-3.827`
- `induction_005`
  - `x_e`: flip, best negative total gap `-0.449`
  - `x_t`: flip, best negative total gap `-0.449`
- `recency_001`
  - `x_e`: flip, best negative total gap `-0.661`
  - `x_t`: flip, best negative total gap `-0.661`
- `coref_006`
  - `x_e`: no flip, best negative total gap `+0.806`
  - `x_t`: no flip, best negative total gap `+0.806`

#### `C-71`

- `coref_009`
  - `x_e`: flip, best negative total gap `-0.556`
  - `x_t`: flip, best negative total gap `-0.556`
- `induction_005`
  - `x_e`: flip, best negative total gap `-3.584`
  - `x_t`: flip, best negative total gap `-3.559`
- `recency_001`
  - `x_e`: flip, best negative total gap `-0.656`
  - `x_t`: flip, best negative total gap `-0.748`
- `coref_006`
  - `x_e`: no flip, best negative total gap `+5.985`
  - `x_t`: no flip, best negative total gap `+5.985`

These are not the kinds of differences that justify a strong claim that one stream is behaving in a fundamentally different way under the current intervention procedure.

## What This Means

The right takeaway is not:

- "the distinction between `x_t` and `x_e` was wrong"

The right takeaway is:

- the current **intervention method** does not separate the two streams sharply enough

That is a very different statement.

The current hook adds a vector **after** a block. Downstream, the model relies heavily on the combined state `x_t + x_e`. That means a post-block perturbation to `x_t` may look very similar to a post-block perturbation to `x_e`, even if the two streams have different conceptual roles.

So the experiment teaches us something important:

- the conceptual distinction exists
- but the current ablation is too blunt to expose it cleanly

## A Useful Analogy

Imagine a company with two departments:

- one department writes the official records
- the other writes the working notes

In principle, changing the official record should be very different from changing the working notes.

But if the only thing you measure is the final memo that combines both, then editing either one late in the process may produce almost the same final output.

That is where we are here.

`x_t` and `x_e` may still play different roles, but this particular intervention point mostly sees their downstream sum.

## Why This Matters For The Main Claim

This result makes the next step clearer.

If we want to argue:

- `x_t` intervention is the trivial symbolic write
- `x_e` intervention is the nontrivial computational steering result

then we need an experiment that can actually tell those apart.

This first comparison did not do that.

That is not a failure. It is a diagnostic result. It tells us the next ablation has to be sharper.

## The Next Ablations That Matter

The most useful follow-ups are:

1. `Pre-block vs post-block intervention`
- Right now we hook after a block.
- A pre-block intervention may preserve more of the functional distinction between streams.

2. `Embedding-level x_t intervention`
- If `x_t` really is the symbolic write channel, the cleanest version of that may be to intervene at initialization rather than after a block.

3. `Separate readout from x_t, x_e, and x_t + x_e`
- If the streams differ internally but collapse at the combined readout, we need to show that directly.

4. `Per-channel symbolic intervention`
- Whole-vector `x_t` writes may still be too blunt.
- The stronger dual-stream-specific test is to identify channel roles by
  differential probing and then intervene with one symbolic slice at a time.

4. `Attention-path sensitivity`
- In CASCADE, later attention uses `x_t` in a special way.
- That is a more promising place to look for a real stream-specific difference than the current post-block output hook.

5. `Wrong-position controls`
- The same vector at the wrong token position should be much weaker.
- That helps isolate position-specific causal effects from generic stream perturbation.

## Bottom Line

This experiment was worth doing because it changed the story.

Before running it, the intuitive story was:

- `x_t` should be easy
- `x_e` should be hard

After running it, the evidence says:

- with the current post-block intervention, `x_t` and `x_e` look almost the same on the clean demo cases

So the next step is not to assert the conceptual difference more strongly. The next step is to build the ablation that can actually reveal it.
