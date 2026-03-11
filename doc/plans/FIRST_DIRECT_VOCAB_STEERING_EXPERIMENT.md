# First Direct Vocabulary Steering Experiment

## Purpose

Run the first direct-vocabulary steering experiment on a small trained symbolic
/ factored architecture that is already available on disk.

This experiment is designed to answer the core thesis question directly:

- can a small symbolic-style model be steered by an exact vocabulary direction?
- does the effect appear locally at the first divergent answer token?
- does the effect remain cleaner than a generic activation-space perturbation?

## Chosen Model

Primary target:
- `/mnt/d/data/neuro-symb-v2/backup-data/experiments/cog-sys-paper-1/series_e_25K_test/E2_independent/checkpoint_epoch_2.pt`

Why this model:
- small enough to iterate on quickly
- approximately `19.0M` parameters
- `6` layers, `6` heads, `384` hidden size
- more relevant than the `single_stream` `S*` runs for the direct-vocabulary /
  symbolic steering question
- less aggressive than the `kronecker` compression variants, which makes it a
  better first testbed

Comparison baseline:
- `/mnt/d/data/neuro-symb-v2/backup-data/experiments/cog-sys-paper-1/series_e_25K_test/E1_dense/checkpoint_epoch_2.pt`

Optional smaller follow-up:
- `/mnt/d/data/neuro-symb-v2/backup-data/experiments/cog-sys-paper-1/series_e_25K_test/E3_kronecker/checkpoint_epoch_2.pt`

## Three-Case Evaluation Panel

Use one clean case from each retained family.

### 1. Capitalization

- `caps_005`
- Prompt: `Complete the US state in headline style: south`
- Choices:
  - `A = " Dakota"`
  - `B = " dakota"`

Why:
- accepted by the smaller-model bridge-candidate screen
- strong local-support signal
- simple lexical contrast

### 2. Induction

- `induction_009`
- Prompt: `cat dog bird cat dog bird cat dog`
- Choices:
  - `A = " bird"`
  - `B = " fish"`

Why:
- accepted by the smaller-model bridge-candidate screen
- single-token semantic answer
- cleaner than the multi-token induction alternatives

### 3. Coreference

- `coref_010`
- Prompt: `Lucas thanked Emma because she shared the notes. The word 'she' refers to`
- Choices:
  - `A = " Emma"`
  - `B = " Lucas"`

Why:
- accepted by the smaller-model bridge-candidate screen
- unambiguous reference
- locally supported under the smaller-model audit

## Intervention Object

### Core steering direction

For each case, define the exact vocabulary steering direction:

```text
Δv = W[token_A] - W[token_B]
```

where:
- `W` is the tied embedding / unembedding matrix in the model's vocabulary space
- `token_A` and `token_B` are the first divergent answer tokens for the chosen
  binary completion pair

This is the direct-vocabulary object under test.

Scope note: this first experiment treats `Δv` as one whole `d_model`
vector. It does **not** yet use the full dual-stream-specific intervention
family in which `x_t` is channelized by head and token embeddings can be
written head-by-head or composed across heads. That omission is now tracked in
[CHANNELIZED_XT_INTERVENTION_NOTE.md](doc/notes/CHANNELIZED_XT_INTERVENTION_NOTE.md).

That was intentional for the first pass. The point of this experiment was to
establish that direct vocabulary steering exists at all before opening the much
larger intervention space. The proper follow-up is now specified in
[PER_CHANNEL_XT_INTERVENTION_PLAN.md](doc/plans/PER_CHANNEL_XT_INTERVENTION_PLAN.md):

- differential probing to identify candidate channel roles
- one-channel symbolic interventions
- vertical-channel tests
- only then mixed-token channel composites

### Steering sign test

Run both directions:

```text
+Δv  : steer toward A
-Δv  : steer toward B
```

This is important. A one-sided result is weaker than a bidirectional result.

## Intervention Site

### Primary site

Inject the steering direction into the **contextual contribution**, not the raw
token embedding.

Operational target:
- the narrowest available representation corresponding to the model's
  contextual update before final readout
- preferred:
  - explicit `x_e` / contextual stream, if the architecture exposes it
- fallback:
  - the post-FFN or post-block contextual delta right before it is merged into
    the running representation

### What not to do

- do not inject into the frozen token-identity stream if the architecture
  separates it
- do not begin with a broad residual-stream perturbation if a narrower
  contextual site is available

## Layer Sweep

Run the intervention at every layer:
- layers `0-5`

Scales:
- `α ∈ {-2.0, -1.0, -0.5, +0.5, +1.0, +2.0}`

This is the first layerwise steering sweep. The goal is not just to show an
effect, but to localize where the exact vocabulary direction is actionable.

## Measurements

### Primary metric

At the first divergent answer token, compute:

```text
local_gap = logit(token_A) - logit(token_B)
```

For each intervention:

```text
local_shift(α, ℓ) = local_gap_with_intervention - local_gap_baseline
```

This is the main success metric.

### Secondary metric

Compute the benchmark-aligned total choice-score difference:

```text
total_gap = score(choice_A) - score(choice_B)
```

and:

```text
total_shift(α, ℓ) = total_gap_with_intervention - total_gap_baseline
```

### Tail contamination metric

Compute:

```text
tail_shift = total_shift - local_shift
tail_fraction = |tail_shift| / max(|total_shift|, ε)
```

This determines whether an apparent steering success is local or suffix-driven.

### Off-target drift metric

For each intervention found to be effective on the target case, measure the
same intervention on the other two panel cases.

Report:
- change in local gap
- change in total gap
- whether the off-target effect is same-direction, opposite-direction, or noise

## Success Criteria

The first experiment is successful if all of the following hold on at least
`2` of the `3` panel cases:

1. `Sign correctness`
   - `+Δv` increases the local A-vs-B gap
   - `-Δv` decreases the local A-vs-B gap

2. `Bidirectionality`
   - the response is not one-sided
   - both signs produce the predicted directional change

3. `Locality`
   - a substantial fraction of the total steering effect is explained by the
     local first-divergent-token shift
   - preferred threshold:
     - `tail_fraction < 0.5`

4. `Layer specificity`
   - the effect is materially stronger at some layers than others
   - if all layers respond equally, the intervention is too blunt to support a
     mechanism claim

5. `Collateral discipline`
   - off-target drift on the other two panel cases is smaller than the on-target
     effect

## Failure Criteria

The first experiment counts as a failure if any of the following dominate:

- only total completion score moves, not local answer-token preference
- steering works in one sign only
- the effect is entirely or mostly tail-mediated
- every layer behaves similarly, suggesting a global blunt perturbation
- off-target drift is as large as the on-target effect

## Baseline Comparison

After the primary run on `E2_independent`, rerun the same protocol on:
- `E1_dense`

Question:
- does the more symbolic / factored model produce a cleaner, more local, or more
  layer-specific response than the dense baseline?

This is the first architecture-sensitive comparison.

## Required Engineering Work

### Loader

Implement a loader for the checkpoint format:
- `.pt` checkpoint
- `config` dict embedded in the checkpoint
- `model_state_dict` embedded in the checkpoint

### Model interface

Expose:
- embedding / unembedding matrix `W`
- first-divergent-token ids for each case
- the chosen contextual intervention site at each layer
- per-layer hook or intervention function

### Report script

Create one script that:
- loads the checkpoint
- runs the panel
- sweeps layers and scales
- writes:
  - JSON artifact
  - Markdown report

Suggested name:
- `scripts/run_direct_vocab_steering.py`

### Artifact schema

Per run, save:
- model identifier
- case identifier
- first divergent token ids and strings
- layer
- scale `α`
- baseline local gap
- intervened local gap
- local shift
- baseline total gap
- intervened total gap
- total shift
- tail shift
- tail fraction

## Recommended Execution Order

1. Implement the checkpoint loader for `E2_independent`.
2. Verify the three panel cases score correctly at baseline.
3. Implement local-gap extraction at the first divergent token.
4. Implement contextual-stream intervention for one layer.
5. Expand to the full `6`-layer sweep.
6. Add `E1_dense` as the baseline comparison.

## Expected Outcomes

### Positive outcome

- direct vocabulary steering is bidirectional
- strongest at a narrow layer band
- mostly local at the first divergent token
- cleaner on `E2_independent` than on `E1_dense`

### Negative but informative outcome

- no local response, only total-score response
- strong tail contamination
- no architecture advantage over `E1_dense`

If that happens, the core thesis needs tightening before any Gemma or gpt-oss
distillation story.
