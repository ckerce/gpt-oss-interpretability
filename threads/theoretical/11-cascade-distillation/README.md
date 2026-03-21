# Thread 11: CASCADE Distillation

**Status**: Theoretical (major gap between framework and implementation)
**Narrative beat**: Automate

## Question
Can we automatically extract steering directions from the model's decision trajectories via matrix factorization?

## Documents (in this directory)
- `CASCADE_DISTILLATION.md` — full mathematical framework (gauge-safe pseudoinverse, LSA connection, training stages)

## Code status
The `gpt_oss_interp/distillation/` package exists but is **stubs only**:
- `train.py` — 7 lines, docstring only
- `evaluate.py` — 9 lines, docstring only
- `cascade_targets.py` — 5 lines, docstring only
- `config.py` and `teacher_artifacts.py` — configuration and I/O scaffolding

## Experiment status
- CASCADE feasibility validated: gauge-safe pseudoinverse reconstruction achieves relative residual ~0.003 and KL divergence < 1e-4 on 3 tasks
- 3 reference runs exist in `runs/cascade_reference_*/` but these are target computation, not student training

## Gaps
- **No working implementation** — this is the biggest disconnect between ambition and delivery
- The theoretical framework is well-specified but there's zero training code
- Student distillation pipeline needs to be built from scratch

## Package dependencies (planned)
`distillation.*`, `capture.activation_cache`, `backends.transformers_gpt_oss`

## Related threads
- [4-decision-trajectories](../../solid/4-decision-trajectories/) — the self-supervised directions CASCADE would automate
- [6-direct-vocab-steering](../../solid/6-direct-vocab-steering/) — manual version of what CASCADE would do automatically
- [10-bridge-cross-model](../../in-progress/10-bridge-cross-model/) — bridge experiments use CASCADE reference targets
