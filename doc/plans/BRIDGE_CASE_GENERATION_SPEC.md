# Bridge Case Generation Spec

## Purpose

This document turns Entry 20 in the lab notebook into an operational spec for
expanding the clean benchmark pool used by the CASCADE-intervention bridge
experiment.

The bridge experiment should not expand the dataset opportunistically after
seeing results. New cases should be generated and admitted using the criteria
below, then passed through the existing audit pipeline before inclusion.

## Current Starting Pool

The current locally supported clean pool is the intersection of:
- `main_analysis_soft` from [runs/analysis_set_stratification/analysis_set_stratification.json](runs/analysis_set_stratification/analysis_set_stratification.json)
- `classification == "local_support"` from [runs/retained_case_decision_audit/decision_audit.json](runs/retained_case_decision_audit/decision_audit.json)

Current pool:
- `caps_002`
- `caps_003`
- `coref_002`
- `coref_003`
- `coref_004`
- `induction_001`
- `induction_003`
- `induction_004`

Special-case exclusion:
- `induction_002`
  - keep as a stress-test / pathology case
  - do not include in the main bridge pool

## Target Composition

Minimum audited clean cases:
- `22`

Per-family target ranges:
- `induction`: `10-14`
- `coreference`: `8-10`
- `capitalization`: `4-6`

Capitalization is included for diversity, but it should never carry the
headline claim by itself. Report pooled bridge results both with and without
capitalization cases.

## Admission Criteria

A case is eligible for the bridge pool only if all of the following hold.

### Baseline criteria

- Final-layer baseline prediction is correct.
- The case satisfies the soft late-commitment rule:
  - expected choice wins for the last `4` consecutive layers.

### Local-support criteria

- Final-layer first-divergent-token preference is locally supportive:
  - `local A-B > 0`
  - `total A-B > 0`
- The first divergent token is the semantic answer token, not a formatting or
  control artifact.

### Completion-shape criteria

- Shared suffix after the first divergence is minimal:
  - preferred: `<= 2` tokens
- Cases with long or semantically meaningful shared tails should be rejected or
  at minimum tagged for sensitivity analysis.

### Exclusion criteria

- `tail_rescued` cases:
  - `local A-B <= 0` but `total A-B > 0`
- Cases whose first divergence is not the semantic answer token
- Cases with weak answer semantics that depend mainly on Harmony tail effects

## Warning Tags

These are not hard exclusions unless promoted later by calibration.

- `suffix_fraction > 0.80`
  - where `suffix_fraction = |tail contribution| / |total A-B|`
- multi-token answer with strong tail dependence after the first divergence
- family with unstable baseline behavior across nearby variants

## Generation Templates

### Induction

Goal:
- Repeated-pattern continuation where the answer token is the obvious next item.

Preferred forms:
- short alphanumeric cycles
- repeated word triplets
- repeated numeric sequences

Good examples:
- `K1 M4 T7 K1 M4 T7 K1 M4 -> T7`
- `amber teal silver amber teal silver amber teal -> silver`
- `2 5 8 2 5 8 2 5 -> 8`

Avoid:
- choices where both completions are plausible stylistic continuations
- cases where the answer is mainly carried by a later token in a multi-token
  completion

### Coreference

Goal:
- Unambiguous pronoun/reference resolution with a single semantic answer token.

Preferred forms:
- short two-entity examples
- strong semantic cues
- low ambiguity about gender/role/reference

Good examples:
- `Sarah thanked Emma because she had stayed late. 'she' refers to -> Sarah`
- `The doctor reassured the patient after he reviewed the scan. 'he' refers to the -> doctor`

Avoid:
- examples where world knowledge or pragmatic inference dominates
- examples with multiple compatible antecedents
- examples where the answer token is split into a misleading first divergence

### Capitalization

Goal:
- Surface-form completion where the divergent token is exactly the capitalization
  decision under audit.

Preferred forms:
- single-token proper nouns where lowercase is a clean competitor
- completions with minimal suffix after divergence

Good examples:
- `Complete the city name: san -> Francisco`
- `Write the state in headline style: south -> Dakota`

Avoid:
- completions where the lowercase competitor tokenizes into a different prefix
  plus tail that dominates the score
- cases with many plausible named entities

## Required Vetting Workflow

Every candidate batch should go through this exact pipeline before any bridge
measurement is run on it.

1. Add candidate cases to the task library.
2. Re-run choice-relative convergence calibration:
   - `python3 scripts/calibrate_convergence.py`
3. Re-run analysis-set stratification:
   - `python3 scripts/stratify_analysis_set.py --tail-length 4`
4. Re-run retained-case local-vs-tail audit:
   - `python3 scripts/audit_retained_case_decision_decomposition.py`
5. Admit only the cases that satisfy the admission criteria above.

## Stop Conditions

Do not expand further once:
- the pool reaches at least `22` audited clean cases
- family targets are satisfied
- capitalization remains a minority family

At that point, freeze the bridge pool and move to:
- `python3 scripts/bridge_cascade_intervention.py`

## Non-Goals

- Do not broaden the pool with recency or syntax cases just to increase `N`.
- Do not include `induction_002`-style Harmony-tail cases in the main pool.
- Do not change the admission rules after inspecting bridge correlations.
