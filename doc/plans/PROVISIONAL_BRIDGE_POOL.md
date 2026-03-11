# Provisional Bridge Pool

This file defines the current working bridge pool for CASCADE-vs-intervention
analysis.

It is intentionally more conservative than `main_analysis_soft`.

## Construction

The pool is:
- the original `gpt-oss` locally supported clean cases
- plus the Gemma 3 1B-screened bridge additions
- minus known pathologies such as `induction_002`

## Canonical IDs

Legacy gpt-oss-supported local-support cases:
- `caps_002`
- `caps_003`
- `coref_002`
- `coref_003`
- `coref_004`
- `induction_001`
- `induction_003`
- `induction_004`

Smaller-model screened additions:
- `caps_005`
- `caps_006`
- `caps_007`
- `coref_010`
- `induction_008`
- `induction_009`

Explicit exclusions:
- `induction_002`

## Current Pool Size

- `14` cases total
- capitalization: `5`
- coreference: `4`
- induction: `5`

## Status

- The first `8` cases are already supported by the gpt-oss benchmark thread.
- The additional `6` cases are provisional:
  - accepted by the Gemma 3 1B screening pipeline
  - not yet re-confirmed on gpt-oss-20b

## Source of Truth

The checked-in source of truth is:
- [gpt_oss_interp/benchmarks/pools.py](gpt_oss_interp/benchmarks/pools.py)

The smaller-model screening artifacts live under:
- [runs/bridge_candidate_screen/dcc83ea841ab6100d6b47a070329e1ba4cf78752/screening_summary.json](runs/bridge_candidate_screen/dcc83ea841ab6100d6b47a070329e1ba4cf78752/screening_summary.json)
