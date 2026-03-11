# Channelized `x_t` Intervention Note

## What We Actually Tested

All direct-vocabulary steering experiments in this repo so far used the same basic intervention object:

- a **full embedding-space direction**
- usually `W[token_A] - W[token_B]`
- injected as one whole vector into either `x_e` or `x_t`

In other words, the current experiments treated the token direction as a single `d_model`-dimensional object.

That means we **did not** test the more architecture-specific intervention family that the dual-stream symbolic model makes possible:

- per-head or per-channel token-slice interventions inside `x_t`

## Why This Matters

In the dual-stream/channelized architecture, the symbolic stream is naturally partitioned by head. For a 6-head model, one token embedding can be viewed as:

\[
e_{\text{car}} =
[e_{\text{car},1}, e_{\text{car},2}, \ldots, e_{\text{car},6}]
\]

That creates intervention choices that do not exist in a standard transformer:

1. **Whole-token `x_t` write**
- add the full token vector to `x_t`

2. **Per-head token write**
- add only the slice corresponding to each head

3. **Mixed-token per-head write**
- for example:
\[
e_{\text{intervention}} =
[e_{\text{word1},1}, e_{\text{word2},2}, \ldots, e_{\text{word6},6}]
\]
- this creates a symbolic composite intervention that is native to the channelized `x_t` representation

4. **Direct symbolic-value-path intervention**
- because later attention consumes `x_t` in a special way, a head-sliced `x_t` perturbation may affect attention values differently from a whole-vector residual-style intervention

These are genuinely new degrees of freedom created by the dual-stream design.

## What We Did Not Do

We did **not**:

- intervene per head
- intervene with mixed-token headwise composites
- modify attention values or gates directly
- exploit channelized symbolic slices as first-class intervention objects

So when the reports say "`x_t` intervention" so far, they mean:

- writing one whole vector into the symbolic stream

not:

- using the full channelized intervention space that the architecture exposes

## Why This Was Easy To Miss

The current runner naturally inherited the standard steering mindset:

- choose one vector
- inject it at one site
- sweep layer and scale

That is a reasonable first pass, but it underuses the dual-stream model. It effectively treats the symbolic stream too much like an ordinary residual space.

## What This Means For The Existing Conclusions

The existing results are still useful, but they should be read more carefully.

They establish:

- whole-vector state interventions can steer the model
- `x_t` and `x_e` can behave differently depending on timing, readout, and position

They do **not** yet establish:

- how much additional control is unlocked by channelized symbolic intervention
- whether per-head `x_t` interventions are cleaner than whole-vector ones
- whether mixed-token headwise interventions reveal more modular symbolic structure

So there is still a major architecture-specific intervention family left unexplored.

## Most Important New Experimental Question

The next high-value question is:

\begin{quote}
Does head-sliced symbolic intervention in `x_t` reveal cleaner, more modular, or more compositional control than whole-vector direct-vocabulary steering?
\end{quote}

That is a sharper dual-stream-specific question than many of the earlier ablations, because it relies on a degree of freedom that simply does not exist in a standard transformer.

## Practical Next Steps

The most direct follow-up experiments are now laid out in
[PER_CHANNEL_XT_INTERVENTION_PLAN.md](/mnt/c/Users/ckerc/Documents/job_applications/openai-interpretability/gpt-oss-interp/PER_CHANNEL_XT_INTERVENTION_PLAN.md).

The short version is:

1. **Differential probing**
- do ACL-style minimal-pair analysis first
- rank channels by task-relevant differential behavior before intervening

2. **Per-head `x_t` ablation**
- inject only one head slice of `W[token_A] - W[token_B]` at a time

3. **Vertical-channel tests**
- treat the same head index across layers as one pipeline, not just isolated sites

4. **Mixed-token composite intervention**
- build interventions like
  `[e_{\text{word1},1}, e_{\text{word2},2}, \ldots]`
- but only after channel roles are partially understood

5. **Compare against whole-vector baseline**
- same case, same position, same scale budget
- whole-vector `x_t` vs per-head `x_t` vs composite `x_t`

6. **Readout decomposition under headwise interventions**
- determine whether individual heads are strong in `x_t`, `x_e`, or only in the combined readout

## Bottom Line

Yes: this was a real missing choice in the intervention space.

We did not yet use the full dual-stream-specific channelized vocabulary intervention family. The current work demonstrated whole-vector state steering. The next wave of experiments should explicitly exploit head-sliced symbolic intervention, guided by differential probing, because that is one of the clearest things this architecture gives us that a standard transformer does not.
