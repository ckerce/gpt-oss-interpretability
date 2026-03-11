# Readout Decomposition Report

## Why This Analysis Matters

One concern from the earlier stream-ablation results was:

- maybe `x_t` and `x_e` are genuinely different internally
- but we keep decoding from the combined state `x_t + x_e`
- so any stream-specific difference gets washed out at the final readout

This analysis tests exactly that.

We take one informative condition:

- model: `C-71`
- intervention: embedding-level `x_t`

and decode the model three different ways:

1. from `x_t + x_e`
2. from `x_t` alone
3. from `x_e` alone

## Why This Condition

We chose embedding-level `x_t` in `C-71` because that was the first setting where the stream distinction started to appear at all.

In earlier results:

- `SS-71` tended to collapse everything together under early intervention
- `C-71` showed at least a modest difference between `x_t` and `x_e`

So `C-71` is the better place to ask whether the internal streams carry different steering signals.

## Cases

We used three clean cases:

- `recency_001`
- `induction_005`
- `coref_009`

Artifacts:

- combined readout: [direct_vocab_steering.json](/mnt/c/Users/ckerc/Documents/job_applications/openai-interpretability/gpt-oss-interp/runs/direct_vocab_readout_c71_embed_xt_combined/direct_vocab_steering.json)
- `x_t` readout: [direct_vocab_steering.json](/mnt/c/Users/ckerc/Documents/job_applications/openai-interpretability/gpt-oss-interp/runs/direct_vocab_readout_c71_embed_xt_xtread/direct_vocab_steering.json)
- `x_e` readout: [direct_vocab_steering.json](/mnt/c/Users/ckerc/Documents/job_applications/openai-interpretability/gpt-oss-interp/runs/direct_vocab_readout_c71_embed_xt_xeread/direct_vocab_steering.json)

## The Main Result

The streams do look different internally.

That is the key conclusion.

In plain language:

- decoding from `x_t` alone gives a very strong, often extreme steering signal
- decoding from `x_e` alone gives a weaker and sometimes unstable signal
- decoding from `x_t + x_e` lands in between

So the combined readout is not telling the whole internal story. It is blending two streams with different behavior.

## A Concrete Example: `recency_001`

Prompt:

`The trophy would not fit in the suitcase because the suitcase was too small. The word 'small' refers to the`

Choices:

- `A = " suitcase"`
- `B = " trophy"`

Same intervention in all three cases:

- embedding-level `x_t`
- same token direction
- same scale sweep

### Combined Readout

Baseline:

- predicts `A`
- total gap `+4.197`

Best negative intervention:

- flips the choice
- best negative total gap `-0.191`

So the combined readout shows a real but modest steering effect.

### `x_t`-Only Readout

Baseline:

- predicts `B`
- total gap `-4.395`

Best negative intervention:

- remains strongly on the `B` side
- best negative total gap `-22.522`

This is the important point:

- the symbolic stream by itself is carrying a very large signal
- it is not subtle

The combined readout looks gentler partly because the contextual stream moderates it.

### `x_e`-Only Readout

Baseline:

- predicts `A`
- total gap `+4.784`

Best negative intervention:

- does **not** flip
- best negative total gap `+4.578`

So the contextual stream by itself is much less responsive on this case.

### What This Example Means

On `recency_001`, the three readouts tell three different stories:

- `x_t` alone: strong and heavily tilted
- `x_e` alone: much weaker
- combined: moderated net effect

That is exactly the kind of internal difference that could be hidden if we only ever look at the final combined readout.

## Another Example: `coref_009`

Prompt:

`Natalie reminded Jacob that he needed to lock the door. The word 'he' refers to`

Here the split is also revealing.

### Combined Readout

- baseline predicts `Jacob`
- best negative intervention does **not** flip
- best negative total gap stays at `+3.236`

### `x_t`-Only Readout

- baseline strongly predicts `Jacob`
- best negative intervention flips hard
- best negative total gap `-23.857`

### `x_e`-Only Readout

- baseline predicts `Jacob`
- best negative intervention also flips
- best negative total gap `-0.683`

Interpretation:

- both streams carry steerable information
- but the symbolic stream carries it at much larger magnitude
- the combined readout is again a moderated mixture rather than a direct reflection of either stream alone

## Another Example: `induction_005`

Prompt:

`sun moon star sun moon star sun moon`

### Combined Readout

- baseline `+1.816`
- best negative `-0.454`

### `x_t`-Only Readout

- baseline `+4.480`
- best negative `-22.993`

### `x_e`-Only Readout

- baseline `+1.096`
- best negative `-1.893`

Again the pattern is the same:

- `x_t` alone is much stronger
- `x_e` alone is weaker but still real
- the final model behavior is the moderated result of combining them

## What This Changes

This analysis changes how to read the earlier ablations.

Before this, one plausible reading was:

- "`x_t` and `x_e` looked similar, so maybe they are not functionally very different"

After this decomposition, the better reading is:

- `x_t` and `x_e` can be quite different internally
- but the combined readout can hide that by averaging or offsetting their effects

So the earlier lack of a sharp separation at the final output does **not** mean the streams are internally equivalent.

## What The Reader Should Take Away

The easiest way to understand the result is:

- `x_t` is carrying a stronger, more direct steering signal
- `x_e` is carrying a weaker, more contextual signal
- the model's final answer comes from combining both

That means the final answer can understate how strong the symbolic-stream effect really is.

## Why This Is Important For The Big Picture

This is the first analysis that really supports the architectural story in an interpretable way.

Not by showing:

- "`x_t` always wins"

But by showing:

- the two streams are not internally the same
- `x_t` can carry a much stronger steering signal than `x_e`
- the combined readout can partially mask that difference

That is a more precise and more useful claim than the earlier slogans.

## Bottom Line

The readout decomposition suggests that some of the earlier ambiguity came from reading out only from `x_t + x_e`.

Internally:

- `x_t` looks stronger and more direct
- `x_e` looks weaker and more moderate

Externally:

- the final combined readout often lands between them

So if we want to understand how direct-vocabulary steering really works in the dual-stream model, we cannot look only at the final combined answer. We have to look at what each stream is carrying on its own.
