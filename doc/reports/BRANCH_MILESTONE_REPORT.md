# Branch Milestone Report

This report explains why each current branch exists, what milestone it captured, and what research or engineering outcome it preserved.

## Branch Summary

### `main`
- Head: `def32ea`
- Purpose: stable shared branch containing the direct-vocabulary steering milestone.
- Why it was merged: the direct-vocab thread reached a coherent endpoint with code, artifacts, figures, and a compiled memo. It represented a self-contained result rather than an intermediate scratch state.
- Outcome captured:
  - a reproducible direct-vocabulary steering evaluation pipeline
  - larger-model CPU evaluation artifacts
  - a literature-style memo with figures under `doc/memo/`
- Meaning of the milestone: this is the first checked-in result showing that exact vocabulary-space steering works cleanly on larger symbolic / gated-attention models, with the matched `71M` single-stream vs CASCADE comparison as the central result.

### `direct-vocab-steering-memo`
- Head: `10689df`
- Purpose: original working branch for the direct-vocabulary steering thread.
- Why it was created: to isolate the steering experiment, memo drafting, and figure generation from the ongoing benchmark / bridge work.
- Outcome captured:
  - `scripts/run_direct_vocab_steering.py`
  - `scripts/generate_direct_vocab_memo_figures.py`
  - `runs/direct_vocab_large_models_cpu/`
  - `doc/memo/direct_vocab_steering_memo_20260311.tex`
  - `doc/memo/figs/`
- Meaning of the milestone: this branch marked the point where the direct-vocab thread became a complete research package rather than an exploratory notebook-style effort.
- Current role: historical checkpoint. Its contents were promoted to `main`, so it should be treated as the original branch of record for that milestone, not the place for new work.

### `integrate-direct-vocab-memo`
- Head: `def32ea`
- Purpose: clean integration branch used to move the direct-vocab milestone onto `main` without bringing along unrelated bridge work.
- Why it was created: the direct-vocab result and the bridge thread were mixed in the broader working tree. This branch allowed a clean cherry-pick of just the direct-vocab package.
- Outcome captured:
  - exact same direct-vocab milestone as `main`
  - clean integration history showing that only the direct-vocab package was merged
- Meaning of the milestone: this is an engineering hygiene branch, not a separate research result. It documents the controlled promotion path from exploratory branch to stable branch.
- Current role: effectively superseded by `main`.

### `bridge-work`
- Head: `ca99eaa`
- Purpose: active branch for the benchmark-to-CASCADE bridge thread.
- Why it was created: after the direct-vocab milestone was merged to `main`, the remaining bridge and screening work needed to continue on its own branch instead of being mixed into the stable line.
- Outcome captured:
  - `BRIDGE_CASE_GENERATION_SPEC.md`
  - `PROVISIONAL_BRIDGE_POOL.md`
  - `gpt_oss_interp/benchmarks/pools.py`
  - `scripts/bridge_cascade_intervention.py`
  - `scripts/screen_bridge_candidates.py`
  - expanded task library and soft-main config cleanup
  - bridge candidate screening artifacts under `runs/bridge_candidate_screen/`
  - package scaffold for `gpt_oss_interp/steering/`, `gpt_oss_interp/distillation/`, and `gpt_oss_interp/common/`
- Meaning of the milestone: this branch marks the transition from benchmark cleanup to a concrete bridge experiment program. The key achievement is not a final scientific claim, but a cleaner experimental substrate:
  - a provisional 14-case bridge pool
  - a candidate-screening workflow using a smaller model
  - a bridge experiment script ready for the main CASCADE-vs-causal-sensitivity test
  - a repo structure that now separates steering/intervention work from distillation/training work
- Current role: active development branch for the unresolved CASCADE bridge question.

## Milestone Narrative

The branch structure reflects two distinct research threads:

1. Direct-vocabulary steering
- Goal: test the core symbolic-transformer claim that exact vocabulary-space directions can causally steer model behavior.
- Milestone reached: yes.
- Captured in: `direct-vocab-steering-memo`, then promoted through `integrate-direct-vocab-memo` into `main`.

2. Benchmark-to-CASCADE bridge
- Goal: build a defensible clean-case pool and evaluate whether CASCADE quality tracks causal intervention sensitivity.
- Milestone reached: partial but meaningful.
- Captured in: `bridge-work`.
- Status: ready as an engineering and methodology checkpoint, but not yet a final positive empirical result.

3. Repo architecture split
- Goal: separate steering/intervention code from teacher-student distillation code before the next wave of per-channel work lands.
- Milestone reached: initial scaffold only.
- Captured in: `bridge-work`.
- Status: package scaffold exists; workflow code migration is still in front of us.

## Why These Branches Matter

These branches were worth checking in because each one preserved a real decision boundary:

- `direct-vocab-steering-memo` preserved the point where the steering hypothesis moved from code exploration to a documented, reproducible result.
- `integrate-direct-vocab-memo` preserved a clean promotion path onto `main`.
- `bridge-work` preserved the point where benchmark cleanup became a structured bridge experiment program with a defined pool, screening pipeline, and execution plan.
- `bridge-work` also now preserves the point where the repo stopped being implicitly single-threaded and gained a documented package split for steering vs distillation.

That separation is useful because the two threads are related, but they do not have the same evidentiary status. The direct-vocab thread has a stable memo-quality result. The bridge thread has a stable experimental foundation, but still needs the main gpt-oss/CASCADE run to answer its headline question.

## Recommended Usage

- Use `main` when you want the stable direct-vocab steering result.
- Use `bridge-work` when you want to continue the CASCADE bridge program.
- Use the scaffold under `bridge-work` as the starting point for new steering and distillation code, rather than adding more first-pass logic to `scripts/`.
- Keep `direct-vocab-steering-memo` and `integrate-direct-vocab-memo` as historical provenance branches unless there is a specific need to revisit that integration history.
