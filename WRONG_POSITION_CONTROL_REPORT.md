# Wrong-Position Control Report

## Why This Control Matters

One natural worry about the strong early interventions is:

- maybe they are not really answer-specific
- maybe they just inject enough signal into the network to disturb the whole completion

To test that, we kept the same intervention but moved it to the wrong place.

Specifically:

- main condition: inject at the answer decision token
- control condition: inject at token `0`, the very beginning of the prompt

If the steering effect is genuinely tied to the answer position, then moving the intervention to token `0` should make it much less behaviorally effective even if some local answer-token motion remains.

## Setup

Model family:

- `SS-71`
- `C-71`

Intervention family:

- pre-block
- `x_e`

Cases:

- `coref_009`
- `induction_005`
- `recency_001`
- `coref_006`

Artifacts:

- decision-position run: [direct_vocab_steering.json](runs/direct_vocab_position_control_decision/direct_vocab_steering.json)
- token-0 control run: [direct_vocab_steering.json](runs/direct_vocab_position_control_token0/direct_vocab_steering.json)

## The Main Result

This control worked exactly the way we would want a good positional control to work.

At the decision token:

- `SS-71` flipped all four curated cases
- `C-71` flipped two of four

At token `0`:

- neither model flipped **any** case

That is the cleanest result in this whole ablation sequence.

It tells us the strong pre-block interventions are not just generic global noise. Position matters.

## A Concrete Example: `recency_001`

Prompt:

`The trophy would not fit in the suitcase because the suitcase was too small. The word 'small' refers to the`

Choices:

- `A = " suitcase"`
- `B = " trophy"`

### Decision-Position Intervention

For `SS-71`:

- decision-position pre-block `x_e`
- best negative total gap `-7.222`
- clear flip

For `C-71`:

- decision-position pre-block `x_e`
- best negative total gap `-0.656`
- still flips

So at the right position, the intervention can reverse the answer.

### Token-0 Control

For `SS-71`:

- token-0 control
- best negative total gap `+2.104`
- no flip

For `C-71`:

- token-0 control
- best negative total gap `+3.856`
- no flip

So the same vector, same stream, same scale range, and same model stops being behaviorally effective when moved to the start of the prompt.

That is exactly what a good wrong-position control should show.

## The Subtle Part

There is an important subtlety here.

The local answer-token shift did **not** disappear in the token-0 control. In fact, the local shifts could still be large.

But the behavior changed:

- no flips
- very large `tail_fraction`

That means:

- the intervention is no longer acting as a clean local answer edit
- instead it is creating diffuse downstream effects through the rest of the sequence

This is why the wrong-position control is so useful. It separates:

- “I can move the answer-token logits somehow”

from

- “I can causally control the answer in a clean, local way”

The token-0 control fails the second test.

## A Worked Contrast: `coref_009`

Prompt:

`Natalie reminded Jacob that he needed to lock the door. The word 'he' refers to`

### Decision Position

For `SS-71`:

- best negative gap `-2.578`
- flip

For `C-71`:

- best negative gap `+2.758`
- no flip

### Token 0

For `SS-71`:

- best negative gap `+2.733`
- no flip

For `C-71`:

- best negative gap `+5.175`
- no flip

So even in the model where the decision-position intervention was strong enough to flip, the wrong-position version lost that power.

## Overall Pattern

### Decision Position

- `SS-71`
  - all four cases flipped
- `C-71`
  - `induction_005` and `recency_001` flipped
  - `coref_009` and `coref_006` did not

### Token 0

- `SS-71`
  - zero flips
- `C-71`
  - zero flips

This is an unusually clean control result.

## Why This Matters

This result rescues the interpretation of the strong pre-block interventions.

Before this control, a skeptical reader could say:

- "maybe early intervention just overwhelms the network wherever you put it"

After this control, the better reading is:

- the intervention still has to be placed near the actual decision point to produce clean behavioral reversals

So the position of the intervention is a real causal variable, not a cosmetic detail.

## The Mechanistic Lesson

The best concise summary is:

- right place: strong, local, behavior-changing
- wrong place: logit motion can remain, but behavior no longer follows cleanly

That distinction is exactly what we want if the goal is to argue that direct-vocabulary steering is acting on a specific decision rather than just scrambling the whole forward pass.

## Bottom Line

The wrong-position control is one of the strongest pieces of evidence in the whole demo package.

It shows that:

- early interventions are not enough by themselves
- the intervention also has to land at the right token position

So whatever else remains unresolved about `x_t` vs `x_e`, the demo now has a clear positional specificity result:

- answer-position steering works
- token-0 steering does not

That makes the direct-vocabulary steering story much more credible.
