# Direct Vocabulary Steering Demo Report

## Purpose

This demo is meant to show a narrow but important claim:

- in these symbolic / gated-attention models, an exact vocabulary direction
  `W[token_A] - W[token_B]`
- injected into the contextual stream at one layer
- can move, and in some cases reverse, the model's preference between `A` and `B`
- without relying on suffix rescue or diffuse tail effects

This is a stronger fit for the current models than an Anthropic-style broad semantic steering demo such as `love` vs `hate` or `Golden Gate Bridge` vs `not Golden Gate Bridge`. Those examples rely on richer open-domain concept structure than these training runs reliably support. Here, the best story is direct pairwise lexical steering on clean prompts.

## What We Tested

We used the matched 71M pair:

- `SS-71`: `dns-dns_S_G_gpt2`
- `C-71`: `dns-dns_S_G_cascade_gpt2`

The demo uses exact token-pair directions taken from the model's output embedding:

- coreference: `W[" Jacob"] - W[" Natalie"]`
- induction: `W[" star"] - W[" cloud"]`
- recency: `W[" suitcase"] - W[" trophy"]`

The intervention is applied to the contextual stream `x_e`, not to the token-identity stream. The key measurement is the local first-divergent-token gap, with total choice-score gap used as a behavioral check.

One important explicit limitation: all current interventions use a **whole-vector** vocabulary direction. We did **not** yet use the full channelized symbolic intervention space of the dual-stream architecture, where `x_t` can be treated head-by-head. That missing intervention family is documented in [CHANNELIZED_XT_INTERVENTION_NOTE.md](/mnt/c/Users/ckerc/Documents/job_applications/openai-interpretability/gpt-oss-interp/CHANNELIZED_XT_INTERVENTION_NOTE.md), and the concrete follow-up plan is in [PER_CHANNEL_XT_INTERVENTION_PLAN.md](/mnt/c/Users/ckerc/Documents/job_applications/openai-interpretability/gpt-oss-interp/PER_CHANNEL_XT_INTERVENTION_PLAN.md).

Implementation note:

- the repo now has a scaffolded split between
  [gpt_oss_interp/steering](/mnt/c/Users/ckerc/Documents/job_applications/openai-interpretability/gpt-oss-interp/gpt_oss_interp/steering),
  [gpt_oss_interp/distillation](/mnt/c/Users/ckerc/Documents/job_applications/openai-interpretability/gpt-oss-interp/gpt_oss_interp/distillation),
  and [gpt_oss_interp/common](/mnt/c/Users/ckerc/Documents/job_applications/openai-interpretability/gpt-oss-interp/gpt_oss_interp/common)
- the next per-channel probing implementation should be written directly in
  [gpt_oss_interp/steering/probing.py](/mnt/c/Users/ckerc/Documents/job_applications/openai-interpretability/gpt-oss-interp/gpt_oss_interp/steering/probing.py)
  rather than added as another first-pass script

## Intervention Decision Space

For a token-flip demo, the intervention site matters as much as the steering vector. The full decision space was larger than what appears in the final CLI.

At a high level, the choices were:

1. **What vector to inject**
- broad learned concept vector
- activation-derived direction
- exact token-pair direction from the model vocabulary

2. **What internal stream to modify**
- token-identity stream
- contextual / residual stream
- some other feature-like subspace

3. **Where in depth to intervene**
- early layers
- middle layers
- late layers
- multiple layers at once

4. **How strong to intervene**
- weak perturbation for gentle margin motion
- strong perturbation for visible flips

5. **How to judge success**
- local answer-token motion only
- final completion behavior only
- some combination of both

The final demo fixes some of these choices and sweeps others.

## How The Space Was Narrowed

We narrowed the space in a deliberately conservative way.

### 1. Use exact vocabulary directions

We chose:

- `W[token_A] - W[token_B]`

instead of a learned steering vector or a broad semantic direction.

Why:

- this is the cleanest test of the direct-vocabulary thesis
- it removes an extra degree of freedom from the demo
- it makes the intervention object fully interpretable

The demo is therefore not asking whether *some* vector can steer the model. It asks whether the literal token-space geometry exposed by the model is itself usable for steering.

It is important to be precise about what this means. In the current experiments, the intervention object is one full `d_model` vector. We did **not** yet test the stronger dual-stream-specific option of slicing the symbolic token embedding by head and intervening with those slices separately or compositionally.

### 2. Intervene in the contextual stream, not the token-identity stream

We chose to add the vector to `x_e`, the contextual stream, after a transformer block.

Why:

- the token-identity stream `x_t` is closer to an explicit symbolic / vocabulary channel than a reasoning state
- the contextual stream is the natural place to test whether the model's evolving representation can be pushed toward one token and away from another
- this is the setting most aligned with the causal claim: we want to alter the model's internal preference at the decision point, not overwrite token identity directly

This also keeps the intervention semantically modest. We are not replacing the model's token representation; we are perturbing the contextual state that determines the next-token choice.

This distinction is critical. In the dual-stream architecture, intervention on `x_t` is architecturally close to directly writing in token space. That is useful for control, but it is not the main scientific test here. The nontrivial question is whether the model can be steered by injecting a vocabulary direction into `x_e`, the contextual stream that carries accumulated computation. In a standard transformer there is no clean `x_t`/`x_e` separation, which is exactly why this architecture makes the experiment sharper.

There is an additional dual-stream-specific degree of freedom that the first demo intentionally did not use: `x_t` is channelized by head, so one can in principle intervene head-by-head with token slices such as `[e_{\text{word1},1}, e_{\text{word2},2}, \ldots]`. That is not available in a standard transformer, and it remains future work here.

### 3. Restrict the site search to single-layer interventions

We did **not** search:

- multi-layer combinations
- per-head combinations
- per-position combinations beyond the answer position

We **did** search:

- one layer at a time
- always at the first divergent answer position

Why:

- a single-layer intervention is much easier to interpret
- the answer-position intervention matches the local-mechanism hypothesis directly
- multi-layer search would explode the search space and make the demo much less defensible

This was a deliberate choice to keep the demo causal and readable rather than maximally optimized.

The next step is therefore not "search over more combinations." It is:

1. use differential probing to learn which symbolic channels appear to carry
   entity, recency, induction, or answer-selection structure
2. intervene on one channel at a time
3. only then build mixed-token channel composites

The intended code path for that work is now explicit:

- probing in `gpt_oss_interp/steering/probing.py`
- causal head-slice interventions in `gpt_oss_interp/steering/interventions.py`
- null baselines in `gpt_oss_interp/steering/controls.py`
- stream-transfer and readout decomposition in `gpt_oss_interp/steering/readouts.py`

That is the only defensible way to use the extra intervention freedom that the
dual-stream architecture exposes.

### 4. Turn the remaining choices into a linear parameter sweep

Once the vector and stream were fixed, the remaining free parameters were:

- layer index
- signed intervention scale

That gives a simple rectangular sweep:

- layers `0..5`
- scales from negative to positive values

For the memo-style experiments, we used a moderate sweep:

- `{-2, -1, -0.5, 0.5, 1, 2}`

For the live demo, we widened it to:

- `{-8, -6, -4, -2, -1, 1, 2, 4, 6, 8}`

Why:

- the moderate sweep is better for characterizing margin movement without overdriving the model
- the live demo needs visible flips, so it uses a stronger range

This is why the demo presets use larger scales than the memo figures.

## How The Sweep Was Scored

Each `(layer, scale)` pair was evaluated with three linked quantities:

1. `local_gap`
- the logit gap between the two competing answer tokens at the first divergent token

2. `total_gap`
- the full forced-choice score difference between completion `A` and completion `B`

3. `tail_fraction`
- the fraction of the total change that cannot be explained by the local answer-token change

Why these three:

- `local_gap` tells us whether the answer token itself moved
- `total_gap` tells us whether the model's actual choice changed
- `tail_fraction` tells us whether the effect is local or whether it leaked into suffix behavior

This prevents a misleading success criterion. A demo that changed only the tail while leaving the answer token untouched would not count as evidence for the direct-vocabulary claim.

## How The Final Demo Presets Were Chosen

The live CLI does not guess the layer. It uses presets that came out of the sweep.

The selection rule was:

1. keep cases that are tokenizer-clean and baseline-correct
2. run the single-layer layer/scale sweep
3. prefer rows that:
- move the local gap in the expected direction
- produce a full flip when possible
- keep `tail_fraction` near zero
4. if several rows work, prefer the simplest strong setting

In practice this meant:

- `coref_009` used layer `5`
- `induction_005` used layer `4` or `5` depending on model
- `recency_001` used layer `2` or `3` depending on model

Those choices were empirical outcomes of the sweep, not architecture priors.

## Why We Did Not Search More Aggressively

There are many ways to get a flip if the only goal is optimization. We intentionally did not do that.

We did not search over:

- multiple simultaneous layers
- arbitrary positions
- learned nonlinear steering rules
- separate vectors for each case

Why:

- each extra degree of freedom makes the demo less interpretable
- the more we optimize, the easier it is to accidentally build a stunt rather than a scientific demonstration
- the current demo is stronger if it succeeds under a small, transparent decision space

So the important methodological point is:

- the token direction was fixed by the vocabulary
- the intervention stream was fixed by the architectural claim
- the remaining search was reduced to a simple linear sweep over layer and scale
- the selected presets were the clean winners of that sweep

That is why the resulting examples are meaningful rather than merely tuned.

## Why `x_t` Was Not The Demo Target

It is worth stating this plainly because it is easy to misunderstand.

If we intervene on `x_t`, we are acting directly on the model's symbolic/token-like stream. In this architecture, that is close to writing directly in vocabulary space. For some purposes that is a feature, not a bug. It is part of the motivation for the architecture.

But for the present demo, `x_t` intervention is too easy to carry the main claim. The interesting claim is not:

- "if we directly modify the symbolic stream, can we change the answer token?"

That would be expected.

The interesting claim is:

- "if we inject the exact same vocabulary direction into the contextual stream `x_e`, can we still causally move or even flip the model's answer?"

That is why the demo uses `x_e`.

So the logic is:

- `x_t` intervention is trivial in the intended sense: it directly edits the symbolic channel
- `x_e` intervention is nontrivial: it asks whether token-space geometry remains usable inside the computation stream
- success on `x_e` is therefore the stronger result

This is also the key contrast with a standard transformer. A standard transformer does not expose a separate symbolic stream whose role can be isolated this way. The dual-stream model does, which turns the steering question into a much cleaner mechanistic experiment.

## Why These Examples

These examples were chosen because they satisfy the demo requirements:

- tokenizer-clean under GPT-2
- baseline-correct on both matched models
- strong local steering response
- visible full choice flips under a wider demo-only scale sweep

The default demo cases are:

1. `coref_009`
2. `induction_005`
3. `recency_001`

We also keep `coref_006` as a useful non-flip control: it shows large local movement, but the baseline preference is strong enough that the tested sweep does not reverse it.

## Main Story

The core result is simple:

- positive steering increases the model's preference for choice `A`
- negative steering decreases that preference
- in the strongest demo cases, negative steering crosses zero and makes the model choose `B`
- the effect is local: `tail_fraction` is effectively zero in these cases

That means the intervention is not merely changing later suffix probabilities. It is acting directly at the first decision token.

## Worked Examples

### Example 1: Coreference

Prompt:

`Natalie reminded Jacob that he needed to lock the door. The word 'he' refers to`

Choices:

- `A = " Jacob"`
- `B = " Natalie"`

Why it works well:

- both tokens are clean single-token contrasts
- both matched models start out correct
- both matched models flip under negative steering at layer 5, scale `-8.0`

Observed behavior:

- `SS-71`: baseline total gap `+5.242` -> steered total gap `-3.827`
- `C-71`: baseline total gap `+5.967` -> steered total gap `-0.556`

Interpretation:

- the exact name-pair direction is enough to make the model reverse a pronoun-resolution decision
- the effect is sharp and local rather than tail-mediated

#### Reading The Live Demo Output

If you run:

```bash
python3 scripts/live_direct_vocab_demo.py --model SS-71 --case-id coref_009
```

you will see:

```text
== SS-71 :: coref_009 ==
Natalie reminded Jacob that he needed to lock the door. The word 'he' refers to
A:  Jacob  |  B:  Natalie
direction = W[' Jacob'] - W[' Natalie']
baseline: pred=A local_gap=+5.242 total_gap=+5.242
positive: layer=5 scale=+8.0 pred=A local_shift=+8.664 total_gap=+13.906 tail_fraction=0.000
negative: layer=5 scale=-8.0 pred=B local_shift=-9.069 total_gap=-3.827 tail_fraction=0.000
```

This means:

- `== SS-71 :: coref_009 ==`
  - we are using the matched 71M single-stream model on the curated `coref_009` case.

- `A: Jacob | B: Natalie`
  - the model is evaluated as a forced choice between those two completions.
  - `A` is the benchmark-correct answer.

- `direction = W[' Jacob'] - W[' Natalie']`
  - the steering vector is not learned separately.
  - it is the exact difference between the output embedding row for `" Jacob"` and the output embedding row for `" Natalie"`.
  - positive steering pushes the contextual state toward `" Jacob"` and away from `" Natalie"`.
  - negative steering does the opposite.

- `baseline: pred=A local_gap=+5.242 total_gap=+5.242`
  - with no intervention, the model prefers `Jacob`.
  - `pred=A` means the model chooses the correct answer.
  - `local_gap=+5.242` is the logit gap at the first divergent token only:
    `logit(" Jacob") - logit(" Natalie")`.
  - `total_gap=+5.242` is the full choice-score gap over the completion.
  - here they are equal because this example is a single-token contrast with no suffix difference after the decision token.

- `positive: layer=5 scale=+8.0 pred=A local_shift=+8.664 total_gap=+13.906 tail_fraction=0.000`
  - the exact same vector is injected into layer 5 of the contextual stream with positive sign and scale `+8.0`.
  - `pred=A` means the model still chooses `Jacob`.
  - `local_shift=+8.664` means the first-token preference for `Jacob` increased by `8.664` logit units relative to baseline.
  - `total_gap=+13.906` means the model now favors `Jacob` much more strongly overall.
  - `tail_fraction=0.000` means the effect is entirely local under this decomposition. It is not coming from later tokens.

- `negative: layer=5 scale=-8.0 pred=B local_shift=-9.069 total_gap=-3.827 tail_fraction=0.000`
  - now the same direction is applied with the opposite sign.
  - `pred=B` means the model has flipped and now chooses `Natalie`.
  - `local_shift=-9.069` means the intervention reduced the `Jacob - Natalie` token gap by a little over nine logit units.
  - that is enough to move the total choice score from `+5.242` to `-3.827`, crossing zero and reversing the decision.
  - again `tail_fraction=0.000`, so the flip is not caused by downstream suffix effects. The intervention changes the answer token directly.

The important point is that this is not a fuzzy semantic control vector. It is a literal token-pair direction, applied at one layer, and it is enough to reverse a coreference decision in a clean, local way.

### Example 2: Induction

Prompt:

`sun moon star sun moon star sun moon`

Choices:

- `A = " star"`
- `B = " cloud"`

Why it works well:

- easy to read live
- clean divergent tokens
- the intervention produces strong movement in both models

Observed behavior:

- `SS-71`: baseline total gap `+4.640` -> steered total gap `-0.449`
- `C-71`: baseline total gap `+1.816` -> steered total gap `-3.584`

Interpretation:

- the direct vocab direction can override a simple copying / continuation preference
- the CASCADE model is especially easy to flip here because the baseline margin is smaller and the steering effect is strong

### Example 3: Recency / Reference Resolution

Prompt:

`The trophy would not fit in the suitcase because the suitcase was too small. The word 'small' refers to the`

Choices:

- `A = " suitcase"`
- `B = " trophy"`

Why it works well:

- semantically intuitive
- tokenizer-clean
- flips on both matched models

Observed behavior:

- `SS-71`: baseline total gap `+2.400` -> steered total gap `-0.661`
- `C-71`: baseline total gap `+4.197` -> steered total gap `-0.656`

Interpretation:

- even in a more semantic reference-resolution setting, the exact lexical direction still has enough causal leverage to reverse the model's choice

### Control Example: Strong Movement Without Flip

Prompt:

`Megan called Thomas after he sent the address. The word 'he' refers to`

Choices:

- `A = " Thomas"`
- `B = " Megan"`

Observed behavior:

- `SS-71`: baseline total gap `+8.520` -> strongest tested negative gap `+0.806`
- `C-71`: baseline total gap `+11.464` -> strongest tested negative gap `+5.985`

Interpretation:

- the direction clearly works
- but the baseline preference is strong enough that this sweep does not cross zero
- this is useful in the demo because it shows that the intervention is not a trivial always-flip hack

## What Not To Demo

Avoid using these as headline examples:

- most capitalization cases
  - many split into dirty sub-token contrasts under GPT-2 tokenization
- syntax agreement
  - these runs are weak on the syntax cases
- broad open-domain concept steering
  - not well matched to the training regime here

## How To Run The Demo

Generated demo report:

- [runs/direct_vocab_demo_matched_71m/report.md](/mnt/c/Users/ckerc/Documents/job_applications/openai-interpretability/gpt-oss-interp/runs/direct_vocab_demo_matched_71m/report.md)

Live CLI:

```bash
python3 scripts/live_direct_vocab_demo.py --model SS-71 --case-id coref_009
python3 scripts/live_direct_vocab_demo.py --model C-71 --case-id induction_005
python3 scripts/live_direct_vocab_demo.py --model SS-71 --all
```

Rebuild the curated demo artifact:

```bash
python3 scripts/run_direct_vocab_demo.py --output runs/direct_vocab_demo_matched_71m
```

## Bottom Line

The best demo story is not “we found a broad semantic concept vector.” It is:

- these models expose useful token-space directions directly
- those directions can be injected into the contextual stream
- the intervention moves the local decision token in the expected direction
- on clean examples, the intervention is strong enough to reverse the model's decision

That is exactly the kind of evidence needed for the direct-vocabulary steering thesis.
