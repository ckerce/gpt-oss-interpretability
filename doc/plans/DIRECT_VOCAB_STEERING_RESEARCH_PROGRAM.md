# Direct Vocabulary Steering Research Program

## Goal

Validate the core claim that symbolic / dual-stream / CASCADE-style models can
be steered directly in vocabulary space because key attention-side computations
live in an intrinsically token-interpretable representation.

Then push this claim toward larger teachers by distilling a dense teacher into a
CASCADE student and asking whether:
- the student reaches usable teacher-relative performance
- direct vocabulary steering still works after distillation
- the resulting student is easier to steer and interpret than the teacher

## Scope

Primary teachers:
- `Gemma 3 1B` as the first practical teacher
- `gpt-oss-20b` as the larger production-style teacher once runtime permits

Primary student:
- a CASCADE / symbolic-transformer student trained by distillation

## Repo Structure

The repo now has a scaffolded split between the two major work streams:

- `gpt_oss_interp/steering/`
- `gpt_oss_interp/distillation/`
- `gpt_oss_interp/common/`

Working rule:

- new intervention, probing, readout, and steering-control code belongs in
  `steering/`
- new teacher-student and CASCADE-target code belongs in `distillation/`
- only artifact schemas, shared run types, shared JSON I/O, and shared model
  metadata interfaces belong in `common/`

This split is intentional. It is meant to prevent direct-vocabulary /
mechanistic steering work from being mixed together with distillation training
in an undifferentiated scripts layer.

## Headline Questions

### Scientific questions

1. Is direct vocabulary steering causally real in the symbolic / CASCADE setting?
2. Does it work locally at the answer token rather than through tail effects?
3. Does the effect survive scaling to larger teacher architectures?
4. Can a distilled CASCADE student preserve enough teacher behavior to make the
   steering result meaningful rather than toy-scale?
5. Is the distilled student at least as steerable, and ideally more steerable,
   than the teacher it came from?

### Engineering questions

1. What is the minimal teacher artifact set needed for CASCADE distillation?
2. How do we make teacher extraction reliable on the available hardware?
3. What is the smallest student architecture that still supports the claim?
4. What evaluation pipeline distinguishes local steering from suffix-mediated
   behavior?
5. What training and reporting artifacts need to be standardized?
6. How do we identify the roles of individual symbolic channels before trying
   channel-composite interventions in `x_t`?
7. How do we keep steering and distillation codepaths separated while
   preserving shared artifact schemas?

## Research Program

### Phase 0: Lock the evaluation substrate

Purpose:
- Prevent the steering and distillation work from floating on noisy or
  pathologically scored cases.

Questions to answer:
- Which benchmark cases are clean enough to support direct-vocabulary steering
- Does whole-vector vocabulary steering miss a dual-stream-specific intervention family
  based on head-sliced symbolic writes in `x_t`
  claims?
- Which cases should be excluded because they are tail-rescued or otherwise
  mechanistically dirty?

Current status:
- The local-support / tail-rescue workflow exists.
- A provisional 14-case bridge pool exists in
  [PROVISIONAL_BRIDGE_POOL.md](doc/plans/PROVISIONAL_BRIDGE_POOL.md).

Deliverables:
- one canonical checked-in clean pool
- one canonical exclusion list
- one report stating why excluded cases are excluded

Decision gate:
- Do not make strong steering claims on any case outside the checked-in clean
  pool.

### Phase 1: Direct vocabulary steering on a symbolic / CASCADE model

Purpose:
- Validate the core claim on the architecture where it should be strongest,
  before involving large-model distillation.

Questions to answer:
- Can we define a direct vocabulary steering intervention that changes behavior
  in the predicted direction?
- Does it work at the first divergent answer token?
- Does it preserve nearby behavior better than more generic activation steering?

Experiment set:

1. `Local token-direction steering`
   - Choose a clean binary decision case with one semantic first-divergent token.
   - Construct the exact vocabulary direction:
- `W[token_A] - W[token_B]`
- whole-vector `W[token_A] - W[token_B]`
- head-sliced symbolic interventions in `x_t`, where token embeddings are split by
  head/channel and intervened on per slice or compositionally
   - Apply this direction inside the student at the candidate steering site.
   - Measure:
     - local answer-token logit shift
     - total choice-score shift
     - off-target drift on nearby cases

2. `Layerwise steering sweep`
   - Apply the same vocabulary direction at multiple student layers.
   - Measure where steering has the best effect-size / collateral-damage tradeoff.

3. `Token-local vs tail-mediated test`
   - For every positive steering result, decompose total choice shift into:
     - local first-divergent-token shift
     - shared-tail contribution
   - Reject any result whose apparent success is mostly tail-mediated.

4. `Channelized symbolic follow-up`
   - For dual-stream models, treat `x_t` as head-sliced symbolic state rather
     than only as one `d_model` vector.
   - First run differential probing to rank channels by task-relevant behavior.
   - Then test:
     - one-channel symbolic writes
     - vertical same-channel interventions across layers
     - informed mixed-token composites
   - Only make strong compositional claims if composite interventions are built
     from role-labeled channels rather than arbitrary head mixes.

Success criteria:
- positive steering on a nontrivial subset of clean cases
- effect visible locally at the answer token
- off-target degradation materially lower than a naive activation-space baseline

Failure criteria:
- effects appear only in total completion score, not local token preference
- results collapse once tail-mediated cases are removed

Engineering requirements:
- symbolic-transformer / CASCADE model with inspectable internal streams
- exact vocabulary-direction intervention interface
- token-local decomposition report
- channelwise probing interface for `x_t`
- vertical-channel analysis artifact set
- shared artifact schema in `gpt_oss_interp/common/artifacts.py`
- steering-native implementation in `gpt_oss_interp/steering/`, not new logic
  in `scripts/`

### Phase 2: First teacher-scale validation on Gemma 3 1B

Purpose:
- Move to a larger teacher without inheriting the full runtime complexity of
  gpt-oss-20b.

Questions to answer:
- Which clean candidate behaviors survive when screened on a larger dense model?
- Can Gemma provide stable teacher distributions and layerwise artifacts for
  distillation?

Experiment set:

1. `Teacher-side screening`
   - Use Gemma to screen candidate cases for:
     - baseline correctness
     - late stability
     - local support vs tail rescue
   - Keep only accepted local-support cases.

2. `Teacher-side decision-direction extraction`
   - For accepted cases, compute:
     - first-divergent-token vocabulary direction
     - layerwise convergence profile
     - candidate steering-relevant layers

3. `Teacher-side bridge preflight`
   - Reuse the CASCADE bridge logic on Gemma to ensure:
     - layerwise CASCADE metrics vary enough to be informative
     - benchmark-side damage is not trivially flat

Success criteria:
- enough accepted local-support cases to justify using Gemma as the first
  distillation teacher
- no evidence that the evaluation is dominated by formatting tails

Current status:
- Gemma candidate screen is running path is implemented and has already accepted
  six new candidates.
- the repo scaffold now supports a clean separation between steering analysis
  and distillation code, but the actual per-channel probing implementation has
  not been written yet

Deliverables:
- accepted-case artifact set for Gemma
- teacher-layer shortlist per case family
- one note on what Gemma validates and what it does not

### Phase 3: CASCADE distillation from Gemma 3 1B

Purpose:
- Produce the first teacher-derived CASCADE student.

Primary question:
- Can a CASCADE student distilled from Gemma recover useful behavior while
  retaining the direct-vocabulary steering interface?

Student-design questions:
- What student size is large enough to be meaningful but small enough to train
  repeatedly?
- How many layers and heads are needed?
- Is final-output supervision enough, or do we need intermediate constraints?

Recommended first student:
- `12-layer`, `12-head`, `d_model ≈ 768` CASCADE student

Reason:
- large enough to be credible
- much cheaper than teacher
- still small enough to iterate on

Distillation questions to answer:

1. `Target definition`
   - Are we supervising:
     - final teacher distribution only
     - closed-form CASCADE `x_e*`
     - both
   - Recommendation:
     - regress to `x_e*` first, then refine with KL

2. `Vocabulary compatibility`
   - Is teacher and student vocabulary shared?
   - If not, what projection is used?
   - Recommendation:
     - first experiment should keep vocabulary handling as simple as possible
     - prefer same or near-compatible tokenization if the student code allows it

3. `Intermediate supervision`
   - Do teacher intermediate layers help?
   - Recommendation:
     - first run: output-only CASCADE target
     - second run: add one intermediate constraint family
     - do not start with a fully layered loss

Training stages:

1. `Artifact generation`
   - teacher final distributions
   - optional teacher layerwise readouts
   - fixed-position metadata for the prediction site

2. `Closed-form target construction`
   - compute `x_e*` where defined by the CASCADE target

3. `Regression warmup`
   - train student to predict `x_e*`

4. `KL refinement`
   - refine with teacher-vs-student distribution loss

5. `Steering evaluation`
   - evaluate direct vocabulary steering on the student

Implementation note:
- new distillation code should be written directly into
  `gpt_oss_interp/distillation/`
- avoid building new teacher-student logic in `scripts/` first and moving it
  later

Success criteria:
- student reaches acceptable teacher-relative benchmark accuracy
- direct vocabulary steering remains causally effective
- steering is at least as local and clean as on the teacher

Failure criteria:
- student behavior is acceptable but steering disappears
- steering works only through tails
- student becomes less interpretable than the teacher

### Phase 4: Mechanistic comparison, teacher vs student

Purpose:
- Show that distillation preserved not just outputs, but useful causal and
  interpretability structure.

Questions to answer:
- Are the same decisions localized to similar layer ranges?
- Does the student preserve the same local token-preference structure?
- Is the student easier to steer or easier to causally localize?

Experiment set:

1. `Teacher vs student local-direction comparison`
   - Compare the same vocabulary direction on:
     - local answer token
     - total choice score
     - tail contribution

2. `Residual-delta damage comparison`
   - Compare layerwise ablation damage in teacher and student.

3. `Bridge comparison`
   - Compare whether CASCADE quality tracks intervention sensitivity in both.

4. `Collateral damage comparison`
   - Compare off-target degradation under steering.

Success criteria:
- student preserves at least the teacher’s clean local steering behavior
- student is at least as modular or more predictable under intervention

### Phase 5: Larger-teacher replication on gpt-oss-20b

Purpose:
- Upgrade the claim from “works on Gemma-scale teacher” to “works on a larger,
  production-style MoE teacher.”

Questions to answer:
- Can the same screening and distillation logic survive a more complex teacher?
- Are MoE-specific teacher pathologies fatal, or merely expensive?

Dependencies:
- working gpt-oss runtime that supports repeated forward passes
- reliable teacher artifact extraction without the current offload failure path

Recommended sequencing:
- do not begin this phase until the Gemma teacher-to-student pipeline has one
  positive end-to-end result

## Workstreams

### Workstream A: Steering science

Questions:
- what is the exact intervention object?
- where should it be applied?
- how local is the effect?

Outputs:
- steering protocols
- local-vs-tail steering reports
- baseline comparisons

### Workstream B: Distillation science

Questions:
- what target object should the student learn?
- what intermediate data, if any, matter?
- what counts as faithful transfer?

Outputs:
- distillation losses
- student variants
- teacher-vs-student comparisons

### Workstream C: Evaluation methodology

Questions:
- which cases are admissible?
- how do we reject suffix artifacts?
- how do we report positive and negative results honestly?

Outputs:
- checked-in pools
- screening scripts
- bridge analysis artifacts

### Workstream D: Runtime and infra

Questions:
- how do we reliably run teachers locally?
- how do we cache artifacts?
- how do we keep expensive model passes from blocking research?

Outputs:
- local snapshot loading
- smaller-model screening path
- teacher artifact format

## Immediate Action Queue

### Priority 1

1. Freeze the provisional 14-case bridge pool as the current evaluation default.
2. Define the first direct-vocabulary steering intervention on the symbolic /
   CASCADE model.
3. Select `2-3` clean cases for first steering validation:
   - one capitalization
   - one induction
   - one coreference

### Priority 2

4. Design the first Gemma teacher artifact format for distillation:
   - prompt metadata
   - target position metadata
   - final distribution
   - optional layerwise distribution
5. Define the first student architecture and training recipe.
6. Implement the closed-form `x_e*` target generation path for Gemma outputs.

### Priority 3

7. Train the first Gemma-derived CASCADE student.
8. Evaluate direct vocabulary steering on the student.
9. Compare teacher vs student on the same clean cases.

## Minimum Viable Papers

### Paper path A: Steering-first

Claim:
- symbolic / CASCADE models support direct vocabulary steering that is cleaner
  and more local than generic activation steering

Requires:
- strong Phase 1 result
- moderate teacher-scale evidence

### Paper path B: Distillation-first

Claim:
- a CASCADE student distilled from a larger teacher preserves useful behavior
  and remains directly steerable in vocabulary space

Requires:
- strong Phase 3 result

### Paper path C: Bridge / methodology-first

Claim:
- clean mechanism evaluation requires local-support filtering and
  first-divergent-token decomposition; naive benchmark correctness is not enough

Requires:
- current benchmark thread plus screening path

## What Not To Do

- Do not start with gpt-oss distillation before the Gemma teacher path works.
- Do not treat total choice-score shifts as evidence of local steering.
- Do not add many intermediate-layer losses before an output-only baseline
  exists.
- Do not let the bridge pool drift informally through notebook entries.
- Do not treat a screened smaller-model acceptance as final confirmation for
  gpt-oss.

## Recommended Next Decision

Choose the first direct-vocabulary steering experiment on the symbolic /
CASCADE model:
- exact intervention site
- exact three-case evaluation panel
- exact local success metric

That is the highest-value unanswered question because it tests the core thesis
directly, before further teacher complexity or distillation overhead.
