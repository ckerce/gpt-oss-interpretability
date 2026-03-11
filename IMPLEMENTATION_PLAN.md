# Implementation Plan

## Goal
Build a companion repository for `gpt-oss-20b` intervention and introspection work that preserves the configurability and modularity of `symbolic-transformer`, while staying separate from the architecture repo.

## Guiding decision
Do not optimize first.

The first successful version should be:
- correct enough to trust
- easy to instrument
- easy to reconfigure
- easy to extend to new behaviors and interventions

Only after those conditions are met should backend-level optimization matter.

## Phase 0: Repository boundary
Keep this repo separate from `symbolic-transformer`.

Reason:
- the architecture repo stays intellectually clean
- transfer to external models becomes its own research artifact
- backend-specific code, caching, and prompting logic do not muddy the architectural work

## Phase 1: Correctness-first baseline
Build a minimal path that can:
- construct a harmony-correct prompt payload
- score prompt completions or choices
- produce baseline outputs on a small task set
- run through one benchmark harness end-to-end

Deliverable:
- one real backend
- one dry-run backend
- one benchmark config
- one report output directory

## Phase 2: Activation capture
Add support for capturing:
- layer outputs / residual stream states
- per-layer logits where available
- attention summaries
- MoE router logits and selected experts

Deliverable:
- activation cache abstraction
- router capture abstraction
- saved artifacts for one prompt family

## Phase 3: Readouts
Build human-legible readouts first.

Priority order:
- layerwise token probability trajectories
- top-k token tables by layer
- attention summary tables
- expert routing summary tables

Deliverable:
- one markdown report showing layerwise and routerwise evolution on a narrow task

## Phase 4: Interventions
Start with simple, robust interventions.

Priority order:
- layer output scaling
- head downweighting / masking
- expert downweighting / masking
- router perturbation
- attention temperature scaling if accessible

Deliverable:
- one sweep on one task family with before/after report

## Phase 5: Benchmark suite
Turn isolated probes into a reusable harness.

Initial task families:
- recency bias
- simple coreference
- capitalization / formatting
- induction-style copying

Metrics:
- baseline accuracy
- expected-vs-competitor margin
- intervention delta
- collateral damage on nearby tasks
- monotonicity of response to intervention scale

Deliverable:
- config-driven benchmark runner with CSV / JSON / Markdown outputs

## Phase 6: Feature work
Once capture and interventions are stable, add:
- sparse autoencoders on cached activations
- feature cards
- feature-level intervention support
- comparisons between architectural readouts and post-hoc feature discovery

Deliverable:
- one SAE-backed analysis notebook and one feature-card style report

## Best initial technical path
Use a hookable backend first.

Recommended order:
1. transformers-compatible backend if hook access is clean enough
2. fallback to a more explicit reference backend if needed for router or expert visibility
3. leave fast inference backends for later

## Risks to manage
- backend drift as upstream model wrappers change
- over-coupling benchmark logic to one backend
- mixing prompting, inference, and analysis code in the same modules
- chasing speed too early

## Definition of success
A good first milestone is reached when the repo can do this reliably:
- load config
- run baseline on a narrow task set
- apply one intervention sweep
- emit report
- leave clean extension points for activation capture and SAE work
