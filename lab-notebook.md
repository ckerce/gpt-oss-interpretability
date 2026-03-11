# Lab Notebook

## Purpose

This notebook records research steps in a form that is auditable and decision-relevant.
Each entry should make it possible to recover:
- the question being asked
- the operational definition used
- the commands run
- the artifacts produced
- the minimal warranted conclusion
- the decision taken afterward

## Entry Template

Each entry should include, when applicable:
- `Question`
- `Motivation`
- `Why now`
- `Hypothesis`
- `Falsifiable prediction`
- `Assumptions`
- `Method`
- `Commands`
- `Artifacts`
- `Results`
- `Interpretation`
- `Alternative explanations / risks`
- `Decision`
- `Next step`

## Project-Level Assumptions

- `gpt-oss-20b` is being treated as the teacher model for benchmark analysis and same-model CASCADE feasibility checks.
- For CASCADE, the relevant target object is a centered least-squares surrogate, not a uniquely identified latent state.
- For benchmark analysis, convergence should be defined relative to the benchmark objective, not raw vocabulary top-1 behavior, unless explicitly stated otherwise.

## Project-Level Open Questions

- Does the gauge-safe CASCADE target remain strong outside the same-model / same-vocab setting?
- Which benchmark families are reliable enough for mechanistic claims without confounding failure analysis?
- Do intermediate-layer signals add value beyond final-output distillation targets?

## 2026-03-10

### Entry 1: CASCADE Framing Tightening

`Question`
- What mathematical object should CASCADE actually target?

`Motivation`
- CASCADE is unusual enough that vague derivations will create implementation drift and false confidence later.

`Why now`
- The project had already accumulated implementation ideas that depended on a not-yet-sharp target definition.

`Hypothesis`
- Recasting CASCADE in centered-logit least-squares form will remove ambiguity and make later evaluation criteria explicit.

`Falsifiable prediction`
- If the reframing is correct, later CASCADE steps can be stated in terms of identifiability, conditioning, and behavioral adequacy rather than raw “teacher logits.”

`Assumptions`
- Only the teacher distribution is directly identified.
- Softmax gauge freedom must be fixed explicitly.

`Method`
- Rewrite the planning docs to define:
  - `C = I - (1/V)11^T`
  - `A = C W`
  - `b = C(log p_teacher - W x_t)`
  - `x_e* = A^+ b`

`Artifacts`
- [EXECUTION_PLAN.md](EXECUTION_PLAN.md)
- [CASCADE_DISTILLATION.md](CASCADE_DISTILLATION.md)

`Results`
- Added explicit CASCADE prerequisites:
  - richer logit-lens artifacts
  - gauge-safe target definition
  - convergence calibration across families
- Reframed CASCADE as a centered least-squares problem.

`Interpretation`
- The roadmap is now constrained by a precise target object rather than an ill-defined notion of teacher logits.

`Alternative explanations / risks`
- A mathematically cleaner target is not yet evidence that the target is behaviorally useful.

`Decision`
- Proceed, but require empirical checks before treating the target as validated.

`Next step`
- Upgrade artifacts so the analyses the docs call for are actually computable.

---

### Entry 2: Logit-Lens Artifact Upgrade

`Question`
- What is the minimum artifact format needed to support convergence and CASCADE analyses?

`Motivation`
- Top-k-only logit-lens output is enough for qualitative inspection but not enough for stable tracked-token trajectories.

`Why now`
- Downstream steps depended on target-token behavior across layers.

`Hypothesis`
- A fixed-token trajectory format plus top-k summaries will be enough for convergence analysis without the storage cost of full dense logits.

`Falsifiable prediction`
- After the upgrade, the canonical logit-lens runs should contain tracked-token trajectories for each position.

`Assumptions`
- Final-layer top-1 at a position is a reasonable tracked token for qualitative trajectory work.

`Method`
- Extend `LayerPrediction` to store:
  - `top_token_ids`
  - `target_token_id`
  - `target_token`
  - `target_rank`
  - `target_logprob`
- Add `tracked_positions` to the JSON output.

`Commands`
```bash
python3 -m py_compile \
  gpt_oss_interp/readouts/logit_lens.py \
  scripts/run_logit_lens.py \
  scripts/reference_cascade_target.py
```

`Artifacts`
- [gpt_oss_interp/readouts/logit_lens.py](gpt_oss_interp/readouts/logit_lens.py)
- [scripts/run_logit_lens.py](scripts/run_logit_lens.py)

`Results`
- Syntax/compile check passed.
- Fixed-token trajectory support is implemented.

`Interpretation`
- Steps depending on tracked-token probabilities are now supported.

`Alternative explanations / risks`
- Full vocab-scale `Δz` analysis still requires denser artifacts or live model access.

`Decision`
- Regenerate the canonical prompt runs.

`Next step`
- Rebuild the three canonical logit-lens runs with the new schema.

---

### Entry 3: Canonical Logit-Lens Reruns

`Question`
- Do the upgraded artifacts work on the three canonical prompts?

`Motivation`
- The new schema is only useful if it appears in the saved runs, not just in code.

`Why now`
- These runs are the basis for later convergence and CASCADE work.

`Hypothesis`
- The reruns will produce valid `tracked_positions` outputs and preserve the earlier qualitative convergence picture.

`Commands`
```bash
python3 scripts/run_logit_lens.py \
  --model openai/gpt-oss-20b \
  --prompt "The trophy would not fit in the suitcase because the suitcase was too" \
  --top_k 10 \
  --output runs/logit_lens_recency

python3 scripts/run_logit_lens.py \
  --model openai/gpt-oss-20b \
  --prompt "The keys to the cabinet are on the table, so they" \
  --top_k 10 \
  --output runs/logit_lens_syntax

python3 scripts/run_logit_lens.py \
  --model openai/gpt-oss-20b \
  --prompt "A7 B2 C9 D4 A7 B2 C9" \
  --top_k 10 \
  --output runs/logit_lens_induction
```

`Schema check command`
```bash
python3 - <<'PY'
import json
from pathlib import Path
for name in ['logit_lens_recency','logit_lens_syntax','logit_lens_induction']:
    path = Path('runs') / name / 'logit_lens_data.json'
    data = json.loads(path.read_text())
    first = data['tracked_positions'][0]
    print(name)
    print('  tracked_positions:', len(data.get('tracked_positions', [])))
    print('  first keys:', sorted(first.keys()))
    print('  trajectory keys:', sorted(first['trajectory'][0].keys()))
PY
```

`Artifacts`
- [runs/logit_lens_recency/logit_lens_data.json](runs/logit_lens_recency/logit_lens_data.json)
- [runs/logit_lens_syntax/logit_lens_data.json](runs/logit_lens_syntax/logit_lens_data.json)
- [runs/logit_lens_induction/logit_lens_data.json](runs/logit_lens_induction/logit_lens_data.json)

`Results`
- `tracked_positions` is present in all three runs.
- Example observed convergence:
  - recency: `' small'` at position 12 converges at layer 18
  - syntax: multiple completion positions converge between layers 18 and 20
  - induction: relevant positions converge between layers 19 and 21

`Interpretation`
- The artifact upgrade worked.
- A single family convergence layer is already suspect for syntax and induction.

`Alternative explanations / risks`
- These are still hand-picked prompts, not family-level summaries.

`Decision`
- Proceed to same-model CASCADE feasibility checks.

`Next step`
- Implement and run a gauge-safe CASCADE reference calculation.

---

### Entry 4: Gauge-Safe CASCADE Reference Implementation

`Question`
- Can the centered least-squares CASCADE target be computed concretely on one prompt position?

`Motivation`
- The derivation needed a reference implementation before further claims.

`Hypothesis`
- The same-model / same-vocab least-squares target will be numerically stable enough to measure.

`Method`
- Compute final-layer teacher log-probs at one position.
- Get `x_t` from the embedding layer.
- Center the vocab dimension:
  - `W_centered = W - mean_vocab(W)`
  - `centered_teacher = log_probs - mean(log_probs)`
  - `centered_xt = W x_t - mean(W x_t)`
- Solve:
  - `x_e* = argmin_x ||W_centered x - (centered_teacher - centered_xt)||_2^2`

`Artifacts`
- [scripts/reference_cascade_target.py](scripts/reference_cascade_target.py)

`Results`
- Reference implementation added and compiled.

`Interpretation`
- The derivation now has a runnable object tied to it.

`Alternative explanations / risks`
- Same-model feasibility does not imply cross-vocabulary or student-distillation feasibility.

`Decision`
- Test it on the canonical recency, syntax, and induction examples.

`Next step`
- Run the reference script on three canonical decision positions.

---

### Entry 5: CASCADE Reference Metrics on Three Canonical Examples

`Question`
- Does the gauge-safe CASCADE target reconstruct the teacher distribution well in the same-model / same-vocab setting?

`Motivation`
- This is the first empirical test of whether the target is numerically meaningful rather than only mathematically tidy.

`Hypothesis`
- For canonical recency, syntax, and induction positions, the relative residual and KL should both be very small.

`Falsifiable prediction`
- If CASCADE is a poor surrogate even in the cleanest setting, the residual and KL should be materially large.

`Commands`
```bash
python3 scripts/reference_cascade_target.py \
  --model openai/gpt-oss-20b \
  --prompt "The trophy would not fit in the suitcase because the suitcase was too" \
  --position 12 \
  --output runs/cascade_reference_recency

python3 scripts/reference_cascade_target.py \
  --model openai/gpt-oss-20b \
  --prompt "The keys to the cabinet are on the table, so they" \
  --position 11 \
  --output runs/cascade_reference_syntax

python3 scripts/reference_cascade_target.py \
  --model openai/gpt-oss-20b \
  --prompt "A7 B2 C9 D4 A7 B2 C9" \
  --position 13 \
  --output runs/cascade_reference_induction
```

`Artifacts`
- [runs/cascade_reference_recency/cascade_reference.json](runs/cascade_reference_recency/cascade_reference.json)
- [runs/cascade_reference_syntax/cascade_reference.json](runs/cascade_reference_syntax/cascade_reference.json)
- [runs/cascade_reference_induction/cascade_reference.json](runs/cascade_reference_induction/cascade_reference.json)

`Results`
- Recency (`' small'`):
  - relative residual: `0.0029417266204564644`
  - KL: `1.1872602044604719e-05`
- Syntax (`' can'`):
  - relative residual: `0.002944351194247872`
  - KL: `8.864954725140706e-05`
- Induction (`' D'`):
  - relative residual: `0.002950425037122956`
  - KL: `0.0001179423852590844`

`Interpretation`
- In the same-model / same-vocab setting, the gauge-safe least-squares target is numerically very strong on the tested examples.

`Alternative explanations / risks`
- The examples are selected and canonical.
- This is not yet evidence about student vocabularies or student architectures.

`Decision`
- Treat same-model CASCADE feasibility as empirically supported.

`Next step`
- Calibrate convergence at the benchmark level before selecting ablation layers.

---

### Entry 6: Raw Token-Level Convergence Calibration Attempt

`Question`
- Can convergence be estimated benchmark-wide by asking when the expected token becomes globally top-ranked?

`Motivation`
- Family-level ablations need benchmark-wide, not hand-picked, layer estimates.

`Hypothesis`
- Raw token-level convergence may provide a first family-level approximation.

`Method`
- Added an initial calibration script based on token-level tracked-target convergence.

`Artifacts`
- [scripts/calibrate_convergence.py](scripts/calibrate_convergence.py)
- intermediate outputs were written to [runs/convergence_calibration](runs/convergence_calibration)

`Results`
- First version had an off-by-one causal-position bug and was corrected.
- After correction, expected-token convergence remained `None` for all cases.
- Final-top convergence still produced family-level signals.

`Interpretation`
- The bug fix mattered, but the core failure remained.
- For choice-style benchmarks, “when does the expected token become vocab-rank 0?” is the wrong object.

`Alternative explanations / risks`
- The benchmark prompts may not align with raw token-level next-token behavior.
- Choice evaluation and vocab-top-1 evaluation are not equivalent.

`Decision`
- Do not use raw token-level convergence as the primary benchmark metric.

`Next step`
- Replace it with choice-relative convergence.

---

### Entry 7: Choice-Relative Convergence Calibration

`Question`
- When does the model converge under the same objective used by the benchmark itself?

`Motivation`
- The previous token-level convergence definition was misaligned with the benchmark.

`Why now`
- Ablation-layer selection should be based on the benchmark objective, not an auxiliary proxy that already failed.

`Hypothesis`
- Choice-relative convergence will produce meaningful family-level signals.

`Falsifiable prediction`
- If the definition is appropriate, case-level convergence should be finite and interpretable for most cases.

`Assumptions`
- Teacher-forced per-layer choice logprobs are the right layerwise analogue of the benchmark’s final scoring rule.

`Method`
- Added per-layer choice scoring in the backend:
  - full forward pass on `prompt + choice`
  - capture block outputs
  - project each block output through final norm + `lm_head`
  - sum teacher-forced completion logprobs at each layer
- Defined:
  - `expected_choice_convergence` = earliest layer where the expected choice wins
  - `final_choice_convergence` = earliest layer where the final-layer winner already wins

`Commands`
```bash
python3 scripts/calibrate_convergence.py \
  --model openai/gpt-oss-20b
```

`Artifacts`
- [gpt_oss_interp/backends/transformers_gpt_oss.py](gpt_oss_interp/backends/transformers_gpt_oss.py)
- [scripts/calibrate_convergence.py](scripts/calibrate_convergence.py)
- [runs/convergence_calibration/convergence_calibration.json](runs/convergence_calibration/convergence_calibration.json)
- [runs/convergence_calibration/convergence_calibration.md](runs/convergence_calibration/convergence_calibration.md)

`Results`
- Family summary:
  - `capitalization`
    - final correct rate: `0.50`
    - expected convergence mean: `7.67`
    - final-winner convergence mean: `1.00`
  - `coreference`
    - final correct rate: `1.00`
    - expected/final convergence mean: `5.00`
  - `induction`
    - final correct rate: `1.00`
    - expected/final convergence mean: `4.25`
  - `recency_bias`
    - final correct rate: `0.25`
    - expected convergence mean: `0.00`
    - final-winner convergence mean: `7.00`
  - `syntax_agreement`
    - final correct rate: `0.50`
    - expected convergence mean: `9.50`
    - final-winner convergence mean: `4.75`

`Interpretation`
- Choice-relative convergence is the correct benchmark-aligned definition.
- It also reveals that some task families are not solved reliably enough to support clean mechanistic claims in their current form.

`Alternative explanations / risks`
- A family with poor final accuracy may still contain useful correctly solved cases.
- Family means can hide mixture structure between easy and hard cases.
- “Early expected convergence” can occur even when the model later flips to a wrong answer.

`Decision`
- Do not use all benchmark cases indiscriminately for mechanistic claims.
- Restrict future benchmark-side interpretability analyses to:
  - reliable families, or
  - correctly solved cases within each family

`Next step`
- Filter the benchmark by baseline correctness and recompute convergence summaries on the retained subset.

---

### Entry 8: Analysis-Set Stratification Protocol

`Question`
- How should benchmark cases be filtered without silently collapsing distinct behavioral regimes?

`Motivation`
- A binary keep/drop rule would hide whether failures arise because:
  - the model never prefers the expected answer
  - the model prefers it early and then loses it
  - the model ends up correct but with unstable winner trajectories

`Why now`
- The convergence calibration step established that not all benchmark families are solved reliably enough for direct mechanistic claims.

`Hypothesis`
- A stratified protocol is better than a binary filter because it preserves failure structure and distinguishes clean successes from unstable successes.

`Falsifiable prediction`
- If the protocol is useful, it should separate the benchmark into interpretable buckets rather than a single heterogeneous “correct” set.

`Assumptions`
- Choice-relative winner trajectories are the relevant object for benchmark-side filtering.
- A conservative definition of stability is:
  - once the expected choice first wins, it never loses again

`Method`
- Added a stratification script that reads choice-relative convergence outputs and classifies each case into:
  - `correct_stable`
  - `correct_unstable`
  - `incorrect_early_expected`
  - `incorrect_never_expected`
- Definitions:
  - `correct_stable`
    - final winner is correct
    - expected choice wins at some layer
    - after first winning, expected choice never loses again
  - `correct_unstable`
    - final winner is correct
    - expected choice wins at some layer
    - but loses again before the end
  - `incorrect_early_expected`
    - final winner is wrong
    - expected choice wins at some earlier layer
  - `incorrect_never_expected`
    - expected choice never wins

`Commands`
```bash
python3 -m py_compile scripts/stratify_analysis_set.py
python3 scripts/stratify_analysis_set.py
```

`Artifacts`
- [scripts/stratify_analysis_set.py](scripts/stratify_analysis_set.py)
- [runs/analysis_set_stratification/analysis_set_stratification.json](runs/analysis_set_stratification/analysis_set_stratification.json)
- [runs/analysis_set_stratification/analysis_set_stratification.md](runs/analysis_set_stratification/analysis_set_stratification.md)

`Results`
- Overall counts:
  - `correct_stable`: `4`
  - `correct_unstable`: `9`
  - `incorrect_early_expected`: `6`
  - `incorrect_never_expected`: `1`
- Main analysis set under the strict definition:
  - `caps_002`
  - `caps_003`
  - `induction_002`
  - `coref_003`
- Secondary analysis set:
  - `recency_004`
  - `induction_001`
  - `induction_003`
  - `induction_004`
  - `coref_001`
  - `coref_002`
  - `coref_004`
  - `syntax_001`
  - `syntax_002`

`Interpretation`
- The protocol is useful: it reveals that “correct” is not a single regime.
- But the conservative stability definition is severe; it leaves only four main-analysis cases.
- Many apparently successful cases are actually winner-trajectory unstable.

`Alternative explanations / risks`
- The stability definition may be too strict for realistic transformer trajectories.
- Some late flips may be shallow and not mechanistically meaningful.
- A stricter definition reduces confounding but risks over-pruning the benchmark.

`Decision`
- Keep the four-way stratification.
- Do not yet commit to `correct_stable` as the only main analysis set without considering a softened stability notion.

`Next step`
- Decide whether to keep the current conservative stability rule or replace it with a softer one, for example:
  - stable after some late layer cutoff
  - stable within a margin threshold
  - stable on a smoothed winner trajectory rather than exact winner identity

---

### Entry 9: Soft Stability Rule Comparison

`Question`
- Can a softer stability rule enlarge the main analysis set without collapsing meaningfully unstable cases into the success set?

`Motivation`
- The conservative rule left only 4 cases in the main analysis set, which is too small for most benchmark-side interpretability work.

`Why now`
- Before redesigning prompts, it is worth checking whether a better stability definition recovers a usable subset from the existing benchmark.

`Hypothesis`
- A late-commitment rule will keep cases with genuine late stabilization while still excluding clear failures and chronically unstable winner trajectories.

`Falsifiable prediction`
- If the softer rule is useful, it should materially expand the main analysis set while leaving clearly bad families like recency mostly excluded.

`Assumptions`
- For interpretability purposes, late commitment may matter more than early oscillation.
- Requiring the expected choice to win for the last `N` consecutive layers is a reasonable operationalization of late commitment.

`Method`
- Extended the stratification script to compute two rules:
  - `strict`
    - once the expected choice first wins, it never loses again
  - `soft_tail_4`
    - final winner is correct
    - expected choice wins for the last 4 consecutive layers

`Commands`
```bash
python3 -m py_compile scripts/stratify_analysis_set.py
python3 scripts/stratify_analysis_set.py --tail-length 4
```

`Artifacts`
- [scripts/stratify_analysis_set.py](scripts/stratify_analysis_set.py)
- [runs/analysis_set_stratification/analysis_set_stratification.json](runs/analysis_set_stratification/analysis_set_stratification.json)
- [runs/analysis_set_stratification/analysis_set_stratification.md](runs/analysis_set_stratification/analysis_set_stratification.md)

`Results`
- Strict counts:
  - `correct_stable`: `4`
  - `correct_unstable`: `9`
  - `incorrect_early_expected`: `6`
  - `incorrect_never_expected`: `1`
- Soft counts:
  - `correct_late_stable`: `9`
  - `correct_late_unstable`: `4`
  - `incorrect_early_expected`: `6`
  - `incorrect_never_expected`: `1`
- Main analysis set size:
  - strict: `4`
  - soft: `9`
- The soft rule mainly recovers:
  - all induction cases
  - additional coreference cases
- It does **not** recover the weak recency cases, and syntax remains problematic.

`Interpretation`
- The soft rule is substantially more usable than the strict rule.
- It expands the main set in a way that looks structurally sensible:
  - induction and coreference often stabilize late after early oscillation
  - recency failures remain failures
  - syntax still looks mixed

`Alternative explanations / risks`
- A tail-of-4 rule is still arbitrary.
- Some recovered cases may have late stabilization that is too shallow or brittle for intervention work.
- The rule may be overfit to a 24-layer model; a different checkpoint might require a different tail length.

`Decision`
- Prefer the soft late-commitment rule over the strict rule for the current benchmark.
- Keep the strict rule as a sensitivity analysis, not as the primary analysis-set definition.

`Next step`
- Use the soft late-commitment analysis set for future benchmark-side experiments, but keep weak families (`recency_bias`, parts of `syntax_agreement`) flagged for redesign or exclusion.

---

### Entry 10: First Benchmark-Side Analysis on the Soft Main Set

`Question`
- Does the soft main-analysis set actually produce cleaner benchmark-side signal on existing intervention results, or is it just a smaller subset by definition?

`Motivation`
- Before launching new heavy runs, it is cheaper to test whether the filtered set improves signal quality on the benchmark sweep that already exists.

`Why now`
- The soft analysis-set definition is now available, and the existing sweep already covers all benchmark cases.

`Hypothesis`
- If the soft main-analysis set is meaningful, it should show substantially cleaner baseline performance and margins than the unfiltered benchmark and the failure set.

`Falsifiable prediction`
- If the soft filter is not doing useful work, baseline accuracy and margin should look similar to the full benchmark.\n- If it is useful, the soft main set should be much cleaner than both the full benchmark and the failure set.

`Assumptions`
- The existing `gpt_oss_20b_sweep` is good enough for a first comparison, even though its interventions were not designed from the new filtering protocol.

`Method`
- Analyze the saved benchmark case results on multiple subsets:
  - all cases
  - strict main-analysis set
  - soft main-analysis set
  - soft secondary set
  - failure set
- Compare baseline (`early_heads_L2@0`) accuracy and mean margin.

`Commands`
```bash
python3 -m py_compile scripts/analyze_filtered_benchmark.py
python3 scripts/analyze_filtered_benchmark.py
```

`Artifacts`
- [scripts/analyze_filtered_benchmark.py](scripts/analyze_filtered_benchmark.py)
- [runs/filtered_benchmark_analysis/filtered_benchmark_analysis.json](runs/filtered_benchmark_analysis/filtered_benchmark_analysis.json)
- [runs/filtered_benchmark_analysis/filtered_benchmark_analysis.md](runs/filtered_benchmark_analysis/filtered_benchmark_analysis.md)

`Results`
- Baseline (`early_heads_L2@0`) comparison:
  - all cases:
    - count: `20`
    - accuracy: `0.650`
    - mean margin: `2.883`
  - strict main set:
    - count: `4`
    - accuracy: `1.000`
    - mean margin: `4.792`
  - soft main set:
    - count: `9`
    - accuracy: `1.000`
    - mean margin: `6.260`
  - soft secondary set:
    - count: `4`
    - accuracy: `0.500`
    - mean margin: `3.248`
  - failure set:
    - count: `7`
    - accuracy: `0.286`
    - mean margin: `-1.667`

`Interpretation`
- The soft main-analysis set is not just smaller; it is substantially cleaner.
- It preserves a nontrivial number of cases (`9`) while achieving:
  - perfect baseline accuracy in the existing sweep
  - much larger positive margin than the full benchmark
- The failure set is clearly separated, with negative mean margin.

`Alternative explanations / risks`
- This is still using an existing sweep whose interventions were not chosen from the filtered analysis protocol.
- Some improvement is expected from filtering by success; the real question is whether the retained set is still large and diverse enough to be useful.

`Decision`
- Use the soft main-analysis set as the default cohort for subsequent benchmark-side interpretability analyses.

`Next step`
- Base the next intervention-focused analysis on the soft main-analysis set rather than the full benchmark.

---

### Entry 11: Ranking Existing Interventions on the Soft Main Set

`Question`
- Which interventions in the existing sweep actually matter on the retained soft main-analysis set?

`Motivation`
- Before designing new runs, use the existing sweep to identify where the current intervention surface already shows strong signal.

`Why now`
- The soft main-analysis set is now defined and validated against the existing benchmark sweep.

`Hypothesis`
- If the retained cases reflect real successful computation, the existing sweep should reveal at least one intervention family with a clear disruptive effect.

`Falsifiable prediction`
- If no intervention stands out on the soft main set, the current sweep is too weakly targeted to guide a focused follow-up.

`Method`
- Rank all saved runs by change in:
  - accuracy
  - mean margin
- Compare them against the baseline run `early_heads_L2@0` on the `main_analysis_soft` subset.

`Commands`
```bash
python3 -m py_compile scripts/rank_soft_main_interventions.py
python3 scripts/rank_soft_main_interventions.py
```

`Artifacts`
- [scripts/rank_soft_main_interventions.py](scripts/rank_soft_main_interventions.py)
- [runs/soft_main_intervention_ranking/soft_main_intervention_ranking.json](runs/soft_main_intervention_ranking/soft_main_intervention_ranking.json)
- [runs/soft_main_intervention_ranking/soft_main_intervention_ranking.md](runs/soft_main_intervention_ranking/soft_main_intervention_ranking.md)

`Results`
- The one clearly disruptive intervention is:
  - `layer_scale_L20@0`
    - accuracy: `0.444`
    - mean margin: `0.945`
    - delta accuracy: `-0.556`
    - delta margin: `-5.314`
- Everything else in the existing sweep leaves accuracy at `1.000` on the soft main set.
- Many nonzero interventions slightly increase mean margin rather than degrading it.

`Interpretation`
- The current sweep already contains one strong and useful signal:
  - full ablation of layer 20 substantially damages the retained cases
- The head/expert sweeps in their current coarse form are not targeted enough to create comparably strong effects on this filtered subset.

`Alternative explanations / risks`
- `layer_scale_L20@0` is a coarse intervention; it shows late-layer importance but not which mechanism inside that layer matters.
- Lack of signal in the current head/expert sweeps may reflect poor targeting rather than true irrelevance.

`Decision`
- Use late-layer sensitivity as the basis for the next focused benchmark-side run.

`Next step`
- Run a focused late-layer ablation sweep on the soft main-analysis set to localize which late layers matter most.

---

### Entry 12: Focused Late-Layer Ablation on the Soft Main Set

`Question`
- Which late layers actually matter on the retained soft main-analysis cases?

`Motivation`
- The existing sweep identified `layer_scale_L20@0` as the strongest disruptive intervention on the soft main set, but it only tested one late layer.

`Why now`
- Before moving to finer-grained interventions, localize the sensitivity across the full late stack.

`Hypothesis`
- The soft main-analysis cases should depend on a narrow band of late layers rather than the whole tail of the network.

`Falsifiable prediction`
- If late-layer sensitivity is diffuse, then ablating layers 18-23 should produce broadly similar damage.
- If it is localized, one or two layers should stand out sharply.

`Assumptions`
- Whole-block `layer_scale` is a meaningful first-pass localization tool.
- The soft main-analysis set is clean enough that layer-level differences will not be washed out by benchmark noise.

`Method`
- Filter the benchmark tasks to `main_analysis_soft`.
- Run six separate full-ablation interventions:
  - layers 18, 19, 20, 21, 22, 23
- Compare accuracy and mean margin across the nine retained cases.
- Inspect failures by behavior family, not just global averages.

`Commands`
```bash
python3 -m py_compile configs/soft_main_late_layer_sweep.py
python3 scripts/run_benchmark.py --config configs/soft_main_late_layer_sweep.py
```

`Artifacts`
- [configs/soft_main_late_layer_sweep.py](configs/soft_main_late_layer_sweep.py)
- [runs/soft_main_late_layer_sweep/case_results.csv](runs/soft_main_late_layer_sweep/case_results.csv)
- [runs/soft_main_late_layer_sweep/summary.json](runs/soft_main_late_layer_sweep/summary.json)
- [runs/soft_main_late_layer_sweep/report.md](runs/soft_main_late_layer_sweep/report.md)

`Results`
- Late-layer summary:
  - `L18@0`: accuracy `0.667`, mean margin `2.126`
  - `L19@0`: accuracy `0.444`, mean margin `0.726`
  - `L20@0`: accuracy `0.444`, mean margin `0.945`
  - `L21@0`: accuracy `0.444`, mean margin `0.607`
  - `L22@0`: accuracy `0.333`, mean margin `1.258`
  - `L23@0`: accuracy `1.000`, mean margin `1.357`
- Family-level failure pattern:
  - `L20@0` breaks induction completely (`0/4` correct) while leaving capitalization intact and only partially hurting coreference
  - `L22@0` breaks coreference completely (`0/3` correct) while partially hurting induction and capitalization
  - `L23@0` leaves accuracy untouched on all retained cases

`Interpretation`
- The late-layer effect is real and localized.
- It is not one single “late reasoning layer”; different retained behaviors appear to depend on different late layers:
  - induction is most sensitive to layer 20
  - coreference is most sensitive to layer 22
- Layer 23 appears largely dispensable under this coarse whole-block intervention.

`Alternative explanations / risks`
- Whole-block `layer_scale` may be too coarse to map cleanly onto architectural subcomponents.
- The behavior split could reflect intervention semantics rather than true mechanism location.

`Decision`
- Follow the whole-block result with a component decomposition at the implicated layers instead of jumping directly to per-head or per-expert sweeps.

`Next step`
- Separate attention-path and MoE-path interventions at layers 20 and 22.

---

### Entry 13: Component Decomposition at Layers 20 and 22

`Question`
- Is the strong whole-block late-layer effect coming from attention, from the MoE/MLP path, or from something else about the block-level intervention?

`Motivation`
- The late-layer sweep implicated layers 20 and 22, but whole-block ablation does not tell us which subcomponent matters.

`Why now`
- A cheap attention-vs-MoE split is the right next test before any expensive per-head or per-expert sweep.

`Hypothesis`
- If the whole-block effect is mechanistically attributable to a major subcomponent, then zeroing all heads or all experts at the implicated layer should reproduce a substantial part of the damage.

`Falsifiable prediction`
- If the late-layer signal is really an attention-path or MoE-path effect, then at least one of:
  - all-head ablation at layer 20 or 22
  - all-expert ablation at layer 20 or 22
  should significantly reduce accuracy or margin on the soft main-analysis set.

`Assumptions`
- `HEAD_MASK` on all 64 heads is a reasonable proxy for suppressing the attention contribution at a layer.
- `EXPERT_MASK` on all 32 experts is a reasonable proxy for suppressing the MoE contribution at a layer.
- The intervention hooks act on the parts of the block we think they act on.

`Method`
- Reuse the soft main-analysis set.
- Run four interventions:
  - all heads at layer 20 scaled to `0`
  - all experts at layer 20 scaled to `0`
  - all heads at layer 22 scaled to `0`
  - all experts at layer 22 scaled to `0`
- Compare against the soft-main baseline from the earlier filtered analysis:
  - accuracy `1.000`
  - mean margin `6.260`
- Inspect any individual failures.

`Commands`
```bash
python3 -m py_compile configs/soft_main_component_decomposition.py
python3 scripts/run_benchmark.py --config configs/soft_main_component_decomposition.py
```

`Artifacts`
- [configs/soft_main_component_decomposition.py](configs/soft_main_component_decomposition.py)
- [runs/soft_main_component_decomposition/case_results.csv](runs/soft_main_component_decomposition/case_results.csv)
- [runs/soft_main_component_decomposition/summary.json](runs/soft_main_component_decomposition/summary.json)
- [runs/soft_main_component_decomposition/report.md](runs/soft_main_component_decomposition/report.md)

`Results`
- First attempt under the activated virtualenv failed because `accelerate` was missing.
- Rerunning with the previously working `python3` environment succeeded.
- Summary:
  - `all_heads_L20@0`: accuracy `1.000`, mean margin `6.787`
  - `all_experts_L20@0`: accuracy `0.889`, mean margin `6.665`
  - `all_heads_L22@0`: accuracy `1.000`, mean margin `6.926`
  - `all_experts_L22@0`: accuracy `1.000`, mean margin `6.555`
- Only one case failed:
  - `induction_002` under `all_experts_L20@0`

`Interpretation`
- The strong whole-block late-layer effect does not decompose cleanly into “attention path” or “MoE path” under the current hook semantics.
- That means the earlier whole-block result cannot yet be read as evidence that a specific subcomponent at layer 20 or 22 carries the critical computation.
- The more likely explanation is intervention mismatch:
  - whole-block `layer_scale` hits a different object than the current head/expert hooks do
  - especially because block-level scaling can remove the full block output, while the head/expert hooks act on submodule outputs only

`Alternative explanations / risks`
- The head hook is an approximation on the post-projection attention output, not a guaranteed faithful head-wise intervention.
- The expert hook explicitly scales the full MLP contribution proportionally rather than selectively removing routed experts.
- If the block output includes residual structure not represented by the submodule hooks, whole-block ablation is not a valid decomposition target.

`Decision`
- Do not proceed yet to per-head or per-expert sweeps.
- First resolve the intervention semantics mismatch.

`Next step`
- Audit the intervention semantics and implement a cleaner residual-delta-style decomposition before finer-grained late-layer sweeps.

---

### Entry 14: Auditing and Correcting Layer-Scale Semantics

`Question`
- Why does whole-block `layer_scale@0` produce large late-layer failures while the head/expert follow-up does not?

`Motivation`
- The previous decomposition result suggested that the intervention surface itself might be inconsistent.

`Why now`
- Any finer-grained late-layer analysis is untrustworthy until the layer-level intervention means what we think it means.

`Hypothesis`
- The old `layer_scale` hook is too destructive because it scales the absolute block output rather than the block delta relative to the incoming residual stream.

`Falsifiable prediction`
- If this diagnosis is correct, then replacing
  - `output <- scale * output`
  with
  - `output <- input + scale * (output - input)`
  should sharply reduce the apparent late-layer sensitivity.

`Assumptions`
- The `GptOssDecoderLayer` forward path in `transformers` is the authoritative semantics for the model block.
- Its output can be treated as:
  - input residual stream
  - plus attention delta
  - plus MLP delta

`Method`
- Inspect the upstream `transformers` implementation of:
  - `GptOssDecoderLayer`
  - `GptOssAttention`
  - `GptOssMLP`
- Confirm the residual structure of the block.
- Change the backend `LAYER_SCALE` hook so that, by default, it preserves the residual stream and scales only the block delta.
- Rerun the focused late-layer sweep on the soft main-analysis set under the corrected semantics.

`Commands`
```bash
python3 -m py_compile gpt_oss_interp/backends/transformers_gpt_oss.py configs/soft_main_late_layer_delta_sweep.py
python3 scripts/run_benchmark.py --config configs/soft_main_late_layer_delta_sweep.py
```

`Artifacts`
- [gpt_oss_interp/backends/transformers_gpt_oss.py](gpt_oss_interp/backends/transformers_gpt_oss.py)
- [configs/soft_main_late_layer_delta_sweep.py](configs/soft_main_late_layer_delta_sweep.py)
- [runs/soft_main_late_layer_delta_sweep/case_results.csv](runs/soft_main_late_layer_delta_sweep/case_results.csv)
- [runs/soft_main_late_layer_delta_sweep/summary.json](runs/soft_main_late_layer_delta_sweep/summary.json)
- [runs/soft_main_late_layer_delta_sweep/report.md](runs/soft_main_late_layer_delta_sweep/report.md)

`Results`
- Upstream decoder-layer semantics:
  - `hidden <- residual + self_attn(norm(hidden))`
  - `hidden <- residual + mlp(norm(hidden))`
- Old interpretation error:
  - legacy `layer_scale@0` zeroed the **entire block output**
  - this deletes the residual passthrough, not just the block computation
- Corrected residual-delta sweep:
  - `L18`: accuracy `1.000`, mean margin `7.314`
  - `L19`: accuracy `1.000`, mean margin `6.273`
  - `L20`: accuracy `0.889`, mean margin `6.018`
  - `L21`: accuracy `0.889`, mean margin `4.726`
  - `L22`: accuracy `1.000`, mean margin `6.306`
  - `L23`: accuracy `1.000`, mean margin `6.157`
- Only one case failed under the corrected semantics:
  - `induction_002` at layers 20 and 21

`Interpretation`
- The old late-layer sweep was largely measuring residual-stream deletion, not clean block skipping.
- After correcting the semantics, late-layer sensitivity remains but is much narrower and weaker:
  - the only consistent residual-delta sensitivity is on one induction case at layers 20-21
- This resolves the earlier mismatch with the head/expert follow-up:
  - the old whole-block intervention was simply not comparable to the submodule hooks

`Alternative explanations / risks`
- The corrected `LAYER_SCALE` hook is now semantically cleaner, but subset head masking is still approximate because it acts after the mixed `o_proj`.
- The expert hook is still only a coarse proxy for selective routed-expert suppression under fused MoE execution.

`Decision`
- Treat the legacy whole-output late-layer sweep as a coarse stress test, not as a mechanistic localization result.
- Use residual-delta `LAYER_SCALE` as the default layer-level intervention going forward.

`Next step`
- Re-run the component decomposition against the corrected residual-delta baseline or move directly to a cleaner block-delta decomposition that compares:
  - attention delta suppression
  - MLP delta suppression
  - full block-delta suppression

---

### Entry 15: Residual-Delta Component Decomposition at Layers 20 and 21

`Question`
- On the corrected intervention surface, do the remaining late-layer effects align more with the attention path or the MLP/MoE path?

`Motivation`
- After fixing `LAYER_SCALE`, only layers 20 and 21 still showed any residual-delta sensitivity, and only on one induction case.

`Why now`
- This is the first point where a component decomposition is meaningful again, because the block-level control now preserves the residual stream.

`Hypothesis`
- If the corrected late-layer effect reflects a real block-internal computation, then the failing residual-delta case should align with one coarse full-component suppression at the same layer.

`Falsifiable prediction`
- If the remaining block-delta failures are not attributable to a major block component, then:
  - full attention suppression
  - and full expert/MLP suppression
  should both leave the corrected failing case intact.

`Assumptions`
- Full-head suppression is a reasonable coarse proxy for attention-delta suppression at a layer.
- Full-expert suppression is a reasonable coarse proxy for MLP-delta suppression at a layer.
- The corrected residual-delta block ablation is the right reference intervention.

`Method`
- Reuse the soft main-analysis set.
- Compare, at layers 20 and 21:
  - full block-delta suppression
  - full-head suppression
  - full-expert suppression
- Inspect whether the same case fails under the block-level and component-level interventions.

`Commands`
```bash
python3 -m py_compile configs/soft_main_component_decomposition_delta.py
python3 scripts/run_benchmark.py --config configs/soft_main_component_decomposition_delta.py
```

`Artifacts`
- [configs/soft_main_component_decomposition_delta.py](configs/soft_main_component_decomposition_delta.py)
- [runs/soft_main_component_decomposition_delta/case_results.csv](runs/soft_main_component_decomposition_delta/case_results.csv)
- [runs/soft_main_component_decomposition_delta/summary.json](runs/soft_main_component_decomposition_delta/summary.json)
- [runs/soft_main_component_decomposition_delta/report.md](runs/soft_main_component_decomposition_delta/report.md)

`Results`
- Layer 20:
  - `late_delta_L20@0`: accuracy `0.889`, mean margin `6.018`
  - `all_heads_L20@0`: accuracy `1.000`, mean margin `6.787`
  - `all_experts_L20@0`: accuracy `0.889`, mean margin `6.665`
- Layer 21:
  - `late_delta_L21@0`: accuracy `0.889`, mean margin `4.726`
  - `all_heads_L21@0`: accuracy `0.889`, mean margin `4.936`
  - `all_experts_L21@0`: accuracy `1.000`, mean margin `6.684`
- In every failing run, the failing case is the same:
  - `induction_002`

`Interpretation`
- The corrected component split is now informative.
- At layer 20, the residual-delta failure aligns with the MLP/MoE-side intervention:
  - block-delta suppression and full-expert suppression both flip `induction_002`
  - full-head suppression does not
- At layer 21, the residual-delta failure aligns with the attention-side intervention:
  - block-delta suppression and full-head suppression both flip `induction_002`
  - full-expert suppression does not
- The current best coarse interpretation is:
  - layer 20 contributes through the MLP/MoE path
  - layer 21 contributes through the attention path
  - both matter only for a narrow induction-style behavior, not the whole retained set

`Alternative explanations / risks`
- The head hook still acts on a post-projection mixed representation, so “attention path” here is only a coarse operational label.
- The expert hook still suppresses the full MLP contribution proportionally rather than selectively removing routed experts.
- Because only one case fails, this result is suggestive but narrow.

`Decision`
- The corrected intervention surface is now good enough to support a cautious, case-specific mechanistic claim.
- Future late-layer work should focus on `induction_002`, not on the whole soft main-analysis set.

`Next step`
- Trace `induction_002` more directly across layers 20 and 21 using:
  - per-layer choice margins
  - activation capture
  - and, if useful, a single-case targeted intervention script

---

### Entry 16: Targeted Single-Case Analysis for `induction_002`

`Question`
- For the one corrected failing case, how do the choice margins evolve across layers under the implicated interventions?

`Motivation`
- The sweep-level summaries established that `induction_002` is the only case that consistently flips under the corrected late-layer surface.

`Why now`
- The right next step after the corrected decomposition is to stop averaging and inspect the single failing case directly.

`Hypothesis`
- The affected interventions should show distinguishable layer-by-layer trajectories rather than only final-output flips.

`Falsifiable prediction`
- If the earlier corrected decomposition is real, then:
  - layer-20 MLP-side interventions
  - and layer-21 attention-side interventions
  should show their strongest divergence from baseline near their respective intervention layers.

`Assumptions`
- Per-layer choice logprob scoring is the correct benchmark-aligned diagnostic for this case.
- The default six-intervention surface for `induction_002` is the right first targeted panel.

`Method`
- Build a single-case analysis script that:
  - loads one benchmark case
  - computes baseline final choice scores
  - computes baseline per-layer choice scores
  - reruns the same for the six corrected interventions:
    - `late_delta_L20`
    - `all_experts_L20`
    - `all_heads_L20`
    - `late_delta_L21`
    - `all_heads_L21`
    - `all_experts_L21`
- Save both JSON and Markdown reports.

`Commands`
```bash
python3 -m py_compile scripts/analyze_single_case.py
python3 scripts/analyze_single_case.py --case_id induction_002 --output runs/single_case_induction_002
```

`Artifacts`
- [scripts/analyze_single_case.py](scripts/analyze_single_case.py)
- [runs/single_case_induction_002/single_case_analysis.json](runs/single_case_induction_002/single_case_analysis.json)
- [runs/single_case_induction_002/single_case_analysis.md](runs/single_case_induction_002/single_case_analysis.md)

`Results`
- Baseline:
  - final prediction `A`
  - final margin `2.829`
- Final-output intervention summary:
  - `late_delta_L20@0`: flips to `B`, margin `-1.338`
  - `all_experts_L20@0`: flips to `B`, margin `-0.206`
  - `all_heads_L20@0`: stays `A`, margin `2.626`
  - `late_delta_L21@0`: flips to `B`, margin `-2.040`
  - `all_heads_L21@0`: flips to `B`, margin `-0.148`
  - `all_experts_L21@0`: stays `A`, margin `0.969`
- Layer-by-layer turning point:
  - baseline becomes correct at layer 18 and stays correct through layer 23
  - every failing intervention is still correct through layer 22
  - the flip happens only at layer 23, the final layer

`Interpretation`
- The coarse component split is confirmed at the single-case level:
  - layer 20 aligns with the expert/MLP-side hook
  - layer 21 aligns with the attention-side hook
- But the strongest new fact is temporal:
  - the implicated interventions do **not** immediately reverse the decision at their intervention layer
  - instead, they perturb the trajectory so that the case flips only at the final layer
- That suggests the relevant effect is downstream accumulation or final-layer re-weighting, not a local one-layer decision reversal at 20 or 21.

`Alternative explanations / risks`
- This result is still limited to one case.
- Because the component hooks are coarse, the delayed flip could reflect broad downstream compensation rather than a precise mechanistic handoff.
- The current script is still benchmark-aligned only; it does not yet inspect internal activation geometry directly.

`Decision`
- Keep `analyze_single_case.py` as the default path for narrow case-specific follow-up.
- Use `induction_002` as the first case for direct activation comparison under clean and intervened runs.

`Next step`
- Capture and compare activations for `induction_002` under:
  - baseline
  - `late_delta_L20`
  - `all_experts_L20`
  - `late_delta_L21`
  - `all_heads_L21`
  focusing on layers 20-23.

---

### Entry 17: Harmony-Aligned Activation Capture for `induction_002`

`Question`
- Under the corrected failing interventions, where do the final-token residual-stream activations diverge most from baseline across layers 20-23?

`Motivation`
- The single-case choice-margin analysis showed delayed final-layer flips, but not whether the residual stream was already substantially perturbed before the last layer.

`Why now`
- This is the first activation-level follow-up after identifying a narrow corrected late-layer signal.

`Hypothesis`
- The failing interventions should produce large residual-stream deviations at or after their intervention layer, with the strongest divergence appearing by the final layer.

`Falsifiable prediction`
- If the delayed flip is not supported by real hidden-state drift, then cosine similarity to baseline should stay near 1.0 and delta norms should stay small through layers 20-23.

`Assumptions`
- Harmony-formatted prompt capture is the right activation path for benchmark alignment.
- For this task, the useful activation object is the final-token residual vector at the assistant-generation boundary.
- Because Harmony control tokens dominate the local decoded top-1 readout, vector movement is more trustworthy here than top-token identity.

`Method`
- Build a Harmony-aligned activation-capture script for one case.
- Capture block outputs at layers 20-23 for:
  - baseline
  - `late_delta_L20`
  - `all_experts_L20`
  - `late_delta_L21`
  - `all_heads_L21`
- At the final prompt position, compare each intervention vector to baseline using:
  - cosine similarity
  - delta norm
  - max / mean absolute difference
- Save both raw summaries and a Markdown report.

`Commands`
```bash
python3 -m py_compile scripts/capture_single_case_activations.py
python3 scripts/capture_single_case_activations.py --case_id induction_002 --output runs/single_case_induction_002_activations
```

`Artifacts`
- [scripts/capture_single_case_activations.py](scripts/capture_single_case_activations.py)
- [runs/single_case_induction_002_activations/activation_comparison.json](runs/single_case_induction_002_activations/activation_comparison.json)
- [runs/single_case_induction_002_activations/activation_comparison.md](runs/single_case_induction_002_activations/activation_comparison.md)

`Results`
- Layer-20 interventions:
  - `late_delta_L20`
    - cosine to baseline: `0.959` → `0.931` → `0.946` → `0.891` across layers 20-23
    - delta norm: `5344` → `6950` → `7902` → `10535`
  - `all_experts_L20`
    - cosine to baseline: `0.965` → `0.952` → `0.964` → `0.930`
    - delta norm: `4823` → `5927` → `6528` → `9064`
- Layer-21 interventions:
  - `late_delta_L21`
    - cosine to baseline: `1.000` at layer 20, then `0.901` → `0.936` → `0.871`
    - delta norm: `0` at layer 20, then `8159` → `8643` → `11447`
  - `all_heads_L21`
    - cosine to baseline: `1.000` at layer 20, then `0.919` → `0.943` → `0.880`
    - delta norm: `0` at layer 20, then `7539` → `8175` → `11043`
- The strongest divergence is consistently at layer 23 for all four failing interventions.

`Interpretation`
- The activation-level result matches the choice-margin result:
  - the intervention effect is not local-only
  - it propagates downstream and is largest at the final layer
- The corrected component split is also visible in activation space:
  - layer-20 block and expert-side interventions produce similar drift patterns
  - layer-21 block and attention-side interventions produce similar drift patterns
- The delayed final-layer flip is therefore supported by real residual-stream movement, not just a quirk of the choice scorer.

`Alternative explanations / risks`
- Because Harmony control tokens dominate the decoded top-1 token at the assistant boundary, the token readout itself is not the right semantic object here.
- The activation comparison is still single-case and final-position-only.
- These are coarse full-component interventions, not selective feature-level edits.

`Decision`
- Treat residual-stream distance growth toward layer 23 as a supported feature of the `induction_002` phenomenon.
- Use this activation workflow as the default follow-up when a benchmark-side effect collapses to one or two cases.

`Next step`
- Connect the activation drift to benchmark-relevant choice readouts more directly by:
  - projecting the captured vectors onto the `green` vs `red` logit difference direction
  - or computing direct logit differences from the captured hidden states at layers 20-23.

---

### Entry 18: Benchmark-Aligned `green` vs `red` Decision Direction

`Question`
- Does the corrected late-layer effect directly change the local `green` vs `red` token preference, or does it act through later shared-suffix tokens in the Harmony completion?

`Motivation`
- The Harmony-aligned activation capture showed large residual-stream drift, but the assistant-boundary top token was dominated by control tokens and could not answer the benchmark-relevant question directly.

`Why now`
- The activation workflow needed to be tied back to the exact benchmark scoring object, not just to generic hidden-state movement.

`Hypothesis`
- The failing interventions should reduce the benchmark-relevant `A-B` choice difference at the exact completion position where ` green` and ` red` diverge.

`Falsifiable prediction`
- If the benchmark choice score is driven mainly by the local divergent token, then the sign of:
  - local `green-red` logit difference
  should match the sign of:
  - total `A-B` choice score difference
  at the same layer.

`Assumptions`
- The correct decision position is the first divergent token in the Harmony-formatted completion, not the assistant boundary.
- For `induction_002`, the two completions share:
  - completion prefix: `<|channel|> final <|message|>`
  - shared suffix: `<|return|>`
- Therefore total `A-B` difference can be decomposed into:
  - local divergent-token contribution
  - shared-suffix contribution

`Method`
- Find the first divergent token between Harmony-formatted completions for:
  - `A = " green"`
  - `B = " red"`
- Capture the shared-prefix hidden state at the position that predicts that divergent token.
- Project each layer’s hidden state onto the exact unembedding direction:
  - `W_green - W_red`
- Separately compute the full per-layer total choice-score difference `A-B`.
- Define suffix contribution as:
  - total choice difference minus local divergent-token difference

`Commands`
```bash
python3 -m py_compile scripts/project_single_case_decision_direction.py
python3 scripts/project_single_case_decision_direction.py --case_id induction_002 --output runs/single_case_induction_002_decision_direction
```

`Artifacts`
- [scripts/project_single_case_decision_direction.py](scripts/project_single_case_decision_direction.py)
- [runs/single_case_induction_002_decision_direction/decision_direction.json](runs/single_case_induction_002_decision_direction/decision_direction.json)
- [runs/single_case_induction_002_decision_direction/decision_direction.md](runs/single_case_induction_002_decision_direction/decision_direction.md)

`Results`
- Harmony decomposition:
  - shared completion prefix: `<|channel|> final <|message|>`
  - divergent tokens: ` green` vs ` red`
  - shared suffix: `<|return|>`
- Baseline:
  - local `green-red` difference is negative at every tested layer:
    - L20: `-4.413`
    - L21: `-5.875`
    - L22: `-3.923`
    - L23: `-7.985`
  - but total choice difference is positive at every tested layer:
    - L20: `+3.741`
    - L21: `+2.626`
    - L22: `+5.034`
    - L23: `+2.829`
  - therefore the shared-suffix contribution is strongly positive:
    - between `+8.154` and `+10.814`
- Under failing interventions, the local `green-red` difference stays negative throughout, but the positive suffix contribution shrinks enough at layer 23 to flip the total choice score negative:
  - `late_delta_L20@0` at L23:
    - local `-8.318`
    - total `-1.338`
    - suffix `+6.980`
  - `all_experts_L20@0` at L23:
    - local `-8.471`
    - total `-0.206`
    - suffix `+8.265`
  - `late_delta_L21@0` at L23:
    - local `-9.529`
    - total `-2.040`
    - suffix `+7.489`
  - `all_heads_L21@0` at L23:
    - local `-9.034`
    - total `-0.148`
    - suffix `+8.887`

`Interpretation`
- This is the most important methodological finding so far for `induction_002`:
  - the benchmark-favored answer `A` is **not** winning locally at the divergent `green` token
  - it wins only because the shared `<|return|>` suffix contribution is even more strongly favorable to `A`
- The failing interventions do not reverse a locally positive `green-red` direction.
- Instead, they weaken the positive suffix contribution enough that the already-negative local `green-red` preference dominates the total choice score at the final layer.
- So the benchmark behavior for this case is not a simple “the model prefers `green` over `red` at the answer token.”
- It is a more complicated Harmony-conditioned completion effect.

`Alternative explanations / risks`
- This may be specific to Harmony formatting and to this case.
- The shared-suffix effect is still part of the model’s real completion behavior, but it is weaker evidence of a clean semantic induction mechanism than a locally positive answer-token preference would be.
- Generalizing this result to the rest of the benchmark would be premature.

`Decision`
- Treat `induction_002` as a useful mechanistic debugging case, not as a clean exemplar of local answer-token induction.
- Raise the evidentiary bar for benchmark cases used in strong mechanistic claims:
  - total choice correctness alone is not enough
  - local divergent-token preference should also be checked when possible

`Next step`
- Audit the other retained benchmark cases for the same decomposition:
  - local divergent-token preference
  - shared-suffix contribution
- especially before treating any case as a clean mechanism exemplar.

---

### Entry 19: Auditing the Retained Soft-Main Cases

`Question`
- Is `induction_002` an isolated Harmony-tail artifact, or are the other retained soft-main cases also winning mainly through tail contribution after the first divergence?

`Motivation`
- After Entry 18, the retained-set benchmark could no longer be treated as uniformly clean without checking the other cases.

`Why now`
- This is the minimum audit needed before using the retained set as evidence for broader mechanistic claims.

`Hypothesis`
- `induction_002` is an outlier and most retained cases will show locally correct first-divergent-token preference at the final layer.

`Falsifiable prediction`
- If the retained set is not clean, then several of the soft-main cases should fall into the same `tail_rescued` bucket as `induction_002`.

`Assumptions`
- A final-layer audit is sufficient for this first-pass retained-set classification.
- First-divergence decomposition is still informative even for multi-token cases such as:
  - `caps_002`
  - `induction_001`
  because the tail term can absorb later divergent-token effects.

`Method`
- Use the `main_analysis_soft` case IDs from the stratification artifact.
- For each retained case:
  - find the first divergent completion token under Harmony formatting
  - capture the shared-prefix hidden state at the corresponding decision position
  - compute local first-divergent-token `A-B` preference at the final layer
  - compute total final-layer choice-score difference `A-B`
  - define tail contribution as `total - local`
- Classify each case as:
  - `local_support`: local `A-B > 0` and total `A-B > 0`
  - `tail_rescued`: local `A-B <= 0` but total `A-B > 0`

`Commands`
```bash
python3 -m py_compile scripts/audit_retained_case_decision_decomposition.py
python3 scripts/audit_retained_case_decision_decomposition.py --output runs/retained_case_decision_audit
```

`Artifacts`
- [scripts/audit_retained_case_decision_decomposition.py](scripts/audit_retained_case_decision_decomposition.py)
- [runs/retained_case_decision_audit/decision_audit.json](runs/retained_case_decision_audit/decision_audit.json)
- [runs/retained_case_decision_audit/decision_audit.md](runs/retained_case_decision_audit/decision_audit.md)

`Results`
- Summary on `main_analysis_soft` at layer 23:
  - `local_support`: `8`
  - `tail_rescued`: `1`
- The one `tail_rescued` case is:
  - `induction_002`
- All other retained cases are locally supported at the first divergent token:
  - `caps_002`
  - `caps_003`
  - `coref_002`
  - `coref_003`
  - `coref_004`
  - `induction_001`
  - `induction_003`
  - `induction_004`
- Representative values:
  - `coref_003`
    - local `+0.290`
    - total `+12.919`
  - `induction_004`
    - local `+5.684`
    - total `+14.694`
  - `induction_002`
    - local `-7.985`
    - total `+2.829`

`Interpretation`
- `induction_002` is an outlier, not the dominant pattern in the retained set.
- The retained soft-main cohort is much cleaner than the single-case failure mode initially suggested.
- The right conclusion is now:
  - use the retained set for mechanism work
  - but explicitly exclude or separately tag `induction_002` as a tail-rescued Harmony case

`Alternative explanations / risks`
- This is still a final-layer-only audit.
- Some `local_support` cases still have large positive tail contributions; they are cleaner than `induction_002`, but not necessarily “pure.”
- The multi-token cases (`caps_002`, `induction_001`) use a first-divergence-plus-tail decomposition rather than a full token-by-token decomposition.

`Decision`
- Keep `main_analysis_soft` as the working cohort, but mark `induction_002` as a special-case stress test rather than a clean exemplar.
- Treat the other eight retained cases as the primary clean-mechanism pool.

`Next step`
- Prefer the eight locally supported retained cases for broader interpretability claims.
- Keep `induction_002` for debugging Harmony-conditioned and tail-mediated effects.

---

### Entry 20: CASCADE-Intervention Bridge Experiment Design

`Question`
- Do CASCADE-derived quality signals track causal intervention sensitivity on clean benchmark cases?

`Motivation`
- CASCADE has strong same-model numerical feasibility (Entry 5), and the benchmark thread has produced a corrected intervention surface with clean cases (Entries 14-19).
- These two threads have not yet shared an experimental result.
- If CASCADE quality at a layer predicts how much that layer's ablation damages benchmark performance, CASCADE is mechanistically grounded, not just a good least-squares fit.
- If there is no link, CASCADE is a distributional readout tool at best.

`Why now`
- The clean-case pool exists but is too small (`N=8`).
- Designing the bridge experiment first determines what the expanded pool must look like.

`Hypothesis`
- Layers where CASCADE reconstruction quality degrades fastest are layers where residual-delta ablation causes the most benchmark damage.

`Falsifiable prediction`
- If CASCADE quality and intervention sensitivity are linked, per-case rank correlations between layer-over-layer CASCADE quality change and layer-level margin damage should be predominantly positive across cases and families.
- If they are unlinked, per-case correlations should be centered near zero with no family-level consistency.

`Preflight Check`
- Mandatory before pool expansion.
- On `2-3` existing clean cases (for example `induction_004`, `coref_003`, `caps_002`):
  - compute layerwise `reconstruction_KL(l, c)` using intermediate hidden states projected through the final unembedding head
  - compute `delta_KL(l, c) = reconstruction_KL(l, c) - reconstruction_KL(l-1, c)`
  - inspect whether `delta_KL` has nontrivial variance across layers
- Gate:
  - if `delta_KL` is nearly flat (for example coefficient of variation `< 0.1` across the measurement range; provisional threshold subject to revision after inspecting the first three curves), the CASCADE signal must be revised before generating more cases
  - candidate revisions:
    - use a different quality metric such as top-k overlap or entropy difference
    - use absolute quality rather than layer-over-layer change
    - use a nonlinear transformation of the quality curve such as second derivative or inflection-point features
  - if `delta_KL` shows clear layer-dependent structure, proceed to pool expansion
- Also inspect whether the shape of the quality curve differs across the preflight cases, even if absolute variance is modest. Between-case shape differences could support the bridge correlation even when within-case variance is small.

`Primary Statistic`
- For each clean case `c` and each layer `l` in the measurement range:
  - CASCADE signal:
    - primary metric: `reconstruction_KL(l, c)` = KL divergence between the CASCADE-reconstructed distribution and the teacher distribution
    - secondary metric: `relative_residual(l, c)` = relative residual norm of the centered least-squares fit
  - intervention signal:
    - primary metric: `margin_damage(l, c)` = baseline margin minus intervened margin on case `c` when layer `l` block-delta is suppressed
    - secondary metric: `correctness_flip(l, c)` = binary indicator of whether the case flips from correct to incorrect
  - layer-level CASCADE quality change:
    - `delta_KL(l, c) = reconstruction_KL(l, c) - reconstruction_KL(l-1, c)`
- Primary test:
  - for each case `c`, compute Spearman rank correlation across layers between `delta_KL(l, c)` and `margin_damage(l, c)`
  - summarize across cases:
    - median per-case correlation
    - proportion of cases with positive correlation
    - sign-flip permutation test on the per-case correlation signs
  - separately report:
    - pooled Spearman `rho` across all `(layer, case)` pairs
    - case-level permutation `p`-value on the pooled statistic
    - bootstrap `95%` confidence interval on pooled `rho`
- Rationale:
  - per-case correlation respects case as the experimental unit and avoids over-crediting within-case layer autocorrelation
  - pooled statistics are secondary and serve as a check, not the headline
- Layer range:
  - measure across layers `4-22` inclusive
  - exclude layer `23` because CASCADE is trivially perfect there in the same-model setting and intervention damage is known to be near-zero
  - exclude layers `0-3` because CASCADE quality is expected to be uniformly poor in early layers, contributing noise without signal

`Positive / Negative / Ambiguous Criteria`
- Positive result:
  - median per-case correlation is clearly positive (for example `> 0.2`)
  - proportion of positive per-case correlations is substantially above `0.5` (for example `> 0.7`)
  - pooled Spearman `rho` is positive with bootstrap CI excluding zero or strong permutation support
  - at least `2` of `3` families show the effect in the predicted direction when stratified
  - the effect is not driven solely by a floor/ceiling artifact
- Negative result:
  - median per-case correlation is near zero (for example `|median| < 0.1`)
  - proportion of positive per-case correlations is near `0.5`
  - no family shows a consistent effect individually
- Ambiguous result:
  - median per-case correlation between `0.1` and `0.2`, or significant in one family only
  - or: the effect is present but driven by a confound such as layer depth, family membership, or a single outlier case
  - or: bootstrap CI on pooled `rho` is wide and includes zero despite a nominally positive point estimate

`Required Dataset Composition`
- Minimum audited clean cases: `22`
- Per-family targets:
  - induction: `10-14`
  - coreference: `8-10`
  - capitalization: `4-6`
- Capitalization's role:
  - included for diversity and statistical balance
  - capitalization is a surface-form family, not a semantic-choice family
  - therefore:
    - always report results with and without capitalization
    - capitalization-only significance never carries the headline claim
    - capitalization contributes to pooled statistics but is never the sole basis for a positive result
- Per-case requirements:
  - final-layer correct under baseline
  - soft late-commitment stable (expected choice wins for the last `4` consecutive layers)
  - locally supported at the first divergent token (`local A-B > 0` at layer `23`)
  - first divergent token is the semantic answer token
  - minimal shared suffix after divergence (`<= 2` tokens)
- Per-case disqualifiers:
  - `tail_rescued` classification (`local A-B <= 0` at the final layer)
  - suffix contribution exceeding `80%` of total choice-score difference at the final layer:
    - provisional disqualifier pending calibration on the existing `8` clean cases
    - use as a warning tag initially
    - promote to hard exclusion only after verifying the distribution on the current clean pool
    - if several existing clean cases are near the boundary, revise the threshold or replace it with a continuous covariate in the analysis

`Family-Stratified Reporting Plan`
- Report results at three levels:
  - per-case then summarized:
    - Spearman correlation within each case across layers, then median / proportion / sign-flip across cases
    - this is primary for the headline claim
  - family-stratified:
    - separate per-case summary within each family
    - primary for robustness
    - minimum: at least `2` families individually showing the effect in the predicted direction for a positive headline
  - family-controlled:
    - pooled correlation after regressing out family fixed effects
    - guards against between-family confounds
- Additionally:
  - report the layer-depth confound check: partial correlation between `delta_KL` and `margin_damage` controlling for layer index
  - report per-family scatterplots of `delta_KL` vs `margin_damage` with layer identity marked
  - report all headline results with and without capitalization cases

`Sensitivity Analyses`
- Alternate CASCADE metric:
  - repeat with `relative_residual` instead of `reconstruction_KL`
- Alternate intervention metric:
  - repeat with `correctness_flip` instead of `margin_damage`
- Tail-length sensitivity:
  - repeat the case filter with tail-`3` and tail-`5` to confirm the clean pool is not an artifact of the tail-`4` rule
- Capitalization inclusion:
  - report pooled results with and without capitalization cases
- Suffix threshold:
  - if the `80%` suffix disqualifier is promoted to hard exclusion, report results under the original and revised pools

`Assumptions`
- The same-model CASCADE target is the right object to correlate against intervention effects. Cross-model CASCADE quality is a separate question.
- Residual-delta `LAYER_SCALE` is the correct intervention for this comparison, not the legacy whole-output version.
- Layer-by-layer CASCADE quality is well-defined using intermediate hidden states projected through the final unembedding head.
- The measurement range (`4-22`) is wide enough to capture both the CASCADE quality transition and the intervention sensitivity transition, and narrow enough to avoid floor/ceiling artifacts.

`Method`
- Phase `0`: preflight
  - on `2-3` existing clean cases, compute layerwise `reconstruction_KL` and `delta_KL`
  - inspect variance
  - if degenerate, revise signal before proceeding
- Phase `1`: pool expansion
  - generate new cases per the case-generation spec
  - run the local-support audit on all new cases immediately
  - confirm the expanded pool meets composition requirements
- Phase `2`: bridge measurement
  - for each audited clean case `c`:
    - run a full forward pass and capture hidden states at layers `4-22`
    - at each layer `l`, compute the CASCADE surrogate:
      - project hidden state through centered unembedding
      - solve the least-squares target
      - compute `reconstruction_KL(l, c)` and `relative_residual(l, c)`
    - run residual-delta ablation at each layer `l` in range:
      - compute `margin_damage(l, c)` and `correctness_flip(l, c)`
    - compute `delta_KL(l, c)` as the layer-over-layer change in CASCADE quality
- Phase `3`: analysis
  - compute per-case Spearman correlations across layers
  - summarize across cases (median, proportion positive, sign-flip test)
  - compute pooled statistics with bootstrap CIs
  - run family-stratified and family-controlled analyses
  - run all sensitivity analyses
  - produce scatterplots and summary report

`Artifacts`
- Planned:
  - [preflight_cascade_variance.py](scripts/preflight_cascade_variance.py)
  - [bridge_cascade_intervention.py](scripts/bridge_cascade_intervention.py)
  - [bridge_preflight](runs/bridge_preflight)
  - [bridge_experiment](runs/bridge_experiment)

`Risks Specific To This Experiment`
- `delta_KL` may be degenerate.
  - If same-model CASCADE quality is uniformly high across intermediate layers, there is no variance to correlate against.
  - The preflight check is designed to catch this before pool expansion.
- `N` is still modest.
  - Even at `22+` cases with `19` layers each, the effective degrees of freedom are limited by within-case autocorrelation across layers.
  - The per-case correlation approach mitigates this by treating each case as one observation in the summary statistics.
  - Bootstrap CIs complement the permutation test where `p`-values are coarse.
- Intervention damage may be concentrated in `1-2` layers per case.
  - This is a feature if it is real, but a risk if it is coincidental.
  - Family-stratified analysis partially addresses this.
- Capitalization cases may behave differently from induction and coreference because the divergent token is a surface-form variant.
- The `80%` suffix threshold is uncalibrated.
  - It is included provisionally and should not be treated as a hard boundary until validated against the existing clean pool.

`Decision`
- This entry defines the bridge experiment.
- Phase `0` (preflight) runs immediately on existing clean cases.
- Pool expansion (Phase `1`) proceeds only after preflight passes.
- The bridge measurement (Phase `2`) runs only after the expanded pool meets composition requirements.

`Next step`
- Run the preflight variance check on `induction_004`, `coref_003`, and `caps_002`.
- If preflight passes: draft the case-generation spec.
- If preflight fails: revise the CASCADE signal before expanding the pool.

---

## Current Belief State

### Supported
- The richer logit-lens artifact format is necessary and implemented for fixed-token trajectories.
- The gauge-safe CASCADE target is numerically strong in the same-model / same-vocab setting on tested recency, syntax, and induction examples.
- Choice-relative convergence is the right benchmark-aligned notion of convergence for these tasks.
- A stratified filtering protocol is better than a binary keep/drop rule for this benchmark.
- A soft late-commitment stability rule is more usable than the strict “never lose again after first win” rule.
- The soft main-analysis set produces substantially cleaner signal on the existing benchmark sweep than the unfiltered benchmark.
- The strongest signal in the existing intervention sweep on the soft main set is late-layer ablation, especially `layer_scale_L20@0`.
- The legacy whole-output `layer_scale` intervention was overly destructive because it deleted the residual passthrough.
- Under corrected residual-delta semantics, late-layer sensitivity is much weaker and currently concentrated on one induction case at layers 20-21.
- On the corrected intervention surface:
  - layer 20 aligns with the coarse MLP/MoE-side intervention
  - layer 21 aligns with the coarse attention-side intervention
  - both effects are currently specific to `induction_002`
- For `induction_002`, the failing interventions remain correct through layer 22 and flip only at the final layer.
- For `induction_002`, the failing interventions also produce large Harmony-aligned residual-stream drift that grows toward layer 23.
- For `induction_002`, the local divergent-token direction (`green` vs `red`) favors `B` at every tested layer; benchmark success comes from an even larger positive shared-suffix contribution.
- Within `main_analysis_soft`, `induction_002` is the only tail-rescued outlier; the other eight retained cases are locally supported at the first divergent token.

### Not Yet Supported
- Cross-vocabulary CASCADE quality.
- Student distillation quality.
- The usefulness of intermediate-layer supervision.
- Mechanistic claims on weak benchmark families without filtering or redesign.
- The current conservative stability rule as the final analysis-set definition.
- A clean component-level explanation of the late-layer whole-block effect.
- Clean subset-head interventions below the full-attention-delta level.
- Clean selective-expert interventions below the full-MLP-delta level.
- Generalization of the layer-20/layer-21 split beyond `induction_002`.
- Whether the delayed final-layer flip is driven by a specific activation-direction change or only by broad downstream drift.
- A direct activation-space connection to the `green` vs `red` decision direction.
- Whether the retained benchmark cases generally reflect local answer-token preference or Harmony-suffix effects.
- A full token-by-token decomposition for the multi-token retained cases.
- Whether CASCADE-derived layerwise quality signals track causal intervention sensitivity on clean cases.

## Current Risks

- Selection effects if filtering to only correctly solved cases.
- Overinterpreting same-model CASCADE feasibility as a result about cross-architecture distillation.
- Using benchmark families whose baseline accuracy is too low for clean interpretability claims.
- Overcommitting to an arbitrary late-stability threshold without sensitivity analysis.
- Trusting approximate subset-head or subset-expert hooks beyond what they actually intervene on.
- Confusing the legacy whole-output `layer_scale` results with the corrected residual-delta `layer_scale` results.
- Overgeneralizing from a one-case late-layer signal.
- Mistaking a final-layer readout effect for an immediate local decision reversal at the intervention layer.
- Overinterpreting decoded Harmony control-token readouts as if they were the benchmark choice object.
- Treating total choice correctness as evidence of local answer-token preference without checking divergent-token decomposition.
- Treating all retained cases as equally clean without separating `induction_002` from the eight locally supported cases.
- Scaling the bridge dataset before confirming that `delta_KL` has usable layerwise variance.

## Strategic Realignment

### Why We Are Changing Direction
- The benchmark/intervention thread has now done its job:
  - it exposed the residual-stream deletion bug
  - it exposed the Harmony tail-rescue confound
  - it produced a usable clean-case filter
- Further benchmark refinement is now likely to have diminishing returns.
- The highest-leverage unresolved question is no longer benchmark hygiene.
- It is whether CASCADE connects to causal intervention structure at all.

### New Main Arc
- Primary research arc:
  - `CASCADE + mechanistic grounding`
- Supporting arc:
  - benchmark/intervention methodology as validation and trust-building
- De-emphasized arc:
  - further benchmark polishing as a standalone objective

### New Research Questions
1. `Bridge question`
   - Do CASCADE-derived quantities track or predict causal intervention sensitivity on clean cases?
2. `Transfer question`
   - Does the gauge-safe CASCADE target survive cross-vocabulary / cross-tokenizer mismatch?
3. `Boundary question`
   - If CASCADE fails outside the same-model setting, is the failure due to:
     - alignment
     - gauge choice
     - or a more fundamental limitation?

### What The Benchmark Thread Is For Now
- Provide clean cases for bridge experiments.
- Provide causal reference points for evaluating CASCADE quantities.
- Provide methodology sections that justify why later CASCADE claims should be trusted.

### What The Benchmark Thread Is Not For Now
- It is not the primary novelty path.
- It is not the main place to invest additional infrastructure unless that work directly supports CASCADE.

## Next Required Decision

Choose the next CASCADE-centered experiment:
- run Entry 20 Phase `0` preflight on the existing clean cases
- or, only if preflight fails, revise the CASCADE signal before pool expansion

## Path Forward

### Immediate Priority
1. `Bridge experiment`
   - Defined in Entry 20.
   - Immediate action:
     - run Phase `0` preflight before pool expansion
   - Purpose:
     - determine whether CASCADE is mechanistically grounded or merely numerically good

2. `Cross-vocabulary CASCADE feasibility`
   - After Entry 20 preflight and bridge measurement, test one carefully chosen model pair with a different tokenizer.
   - Keep this minimal at first:
     - one source model
     - one target model
     - one alignment strategy
   - Purpose:
     - determine whether CASCADE has real transfer potential
   - Candidate alignment strategies to choose from:
     - byte-level / string-level alignment of completion targets
     - learned linear projection on shared prompt states
     - token-overlap or token-frequency-weighted lexical mapping
   - Requirement:
     - choose the first strategy before running, not after seeing outcomes

### Deprioritized Work
- More generic benchmark-family cleanup
- More broad intervention sweeps on the current tiny clean set
- More hook refinement that is not directly tied to a CASCADE-facing hypothesis

### Conditional Branches
- If the bridge experiment shows a real link between CASCADE quality and intervention sensitivity:
  - double down on CASCADE as the main story
  - treat the benchmark thread as validation scaffolding
- If the bridge experiment shows no link:
  - investigate whether CASCADE is:
    - numerically valid but mechanistically irrelevant
    - mismeasured
    - or only valid in a narrower setting
  - fallback reframing:
    - test whether CASCADE predicts distributional properties instead of causal sensitivity, for example:
      - entropy
      - top-k concentration
      - logit-mass concentration
- If cross-vocabulary CASCADE works:
  - characterize failure modes and boundary conditions
- If cross-vocabulary CASCADE fails:
  - diagnose whether the problem is:
    - alignment
    - gauge handling
    - or a fundamental incompatibility

## Pause State

### Where We Stopped
- The methodological cleanup phase is complete enough to stop being the main focus.
- The benchmark thread has produced:
  - corrected intervention semantics
  - a clean-case filter
  - a benchmark-evaluation confound analysis
- The main unresolved work is now CASCADE-centered.

### Working Sets
- Clean bridge-experiment pool:
  - `caps_002`
  - `caps_003`
  - `coref_002`
  - `coref_003`
  - `coref_004`
  - `induction_001`
  - `induction_003`
  - `induction_004`
- Special-case benchmark pathology:
  - `induction_002`

### Recommended Restart Order
1. Run Entry 20 Phase `0` preflight on `induction_004`, `coref_003`, and `caps_002`.
2. If preflight passes, draft the case-generation spec and expand the clean pool.
3. Then run the full bridge experiment.
4. Then run one minimal cross-vocabulary CASCADE feasibility test.

### Concrete Next Actions
- `Option A: Bridge experiment`
  - execute Entry 20 as staged
  - this is the default next move
  - before running:
    - use the pre-registered bridge criteria already defined in Entry 20
- `Option B: Cross-vocabulary CASCADE`
  - test one model pair with different tokenization
  - only after or alongside the bridge experiment
  - before running:
    - choose one explicit alignment strategy from the candidate list above
- `Option C: Supporting benchmark work`
  - only do more benchmark-side work if it directly helps interpret a CASCADE result

### Files To Reopen First
- [lab-notebook.md](lab-notebook.md)
- [CASCADE_DISTILLATION.md](CASCADE_DISTILLATION.md)
- [EXECUTION_PLAN.md](EXECUTION_PLAN.md)
- [decision_audit.md](runs/retained_case_decision_audit/decision_audit.md)
- [summary.json](runs/soft_main_component_decomposition_delta/summary.json)

### Current Default
- Treat benchmark/intervention work as supporting methodology unless it answers a CASCADE-facing question.
- Run the Entry 20 preflight before expanding the bridge dataset.
- Keep `induction_002` as a stress-test pathology, not as a headline mechanism case.
