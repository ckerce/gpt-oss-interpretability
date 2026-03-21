# Execution Plan: Making gpt-oss-interp a Compelling Portfolio Piece

**Created**: 2026-03-10
**Supersedes**: TODO.md (intervention experiments), parts of IMPLEMENTATION_PLAN.md (Phases 1-5 are DONE)
**Context**: OpenAI Interpretability Researcher application. The repo needs to demonstrate original insight, not just infrastructure.

## Current State

### What works
- Hook-based intervention system on gpt-oss-20b (HEAD_MASK, EXPERT_MASK, LAYER_SCALE, TEMPERATURE_SCALE)
- Logit-lens with per-layer convergence detection
- Benchmark harness (5 task families, 20 cases, config-driven, CSV/JSON/MD output)
- Activation cache and router capture
- Dry-run backend for testing
- Extended Tier-2 feature extractor (7,200D for MoE, implemented but never run)
- Geometric analysis module (clustering, dimensionality, inspectability)

### What exists in runs/
- `dry_run_full/` — smoke test (320 cases, synthetic data)
- `gpt_oss_20b_sweep/` — real benchmark (320 cases, head/layer/expert ablations)
- `logit_lens_recency/` — "The trophy would not fit..." → converges at layer 18 to "small"
- `logit_lens_syntax/` — "The keys to the cabinet..." → converges at layer 20 to "are"
- `logit_lens_induction/` — "A7 B2 C9 D4 A7 B2 C9..." → converges at layer 21 to "D"

### What's missing (the problem)
- No demonstrated findings beyond "convergence happens at different layers"
- No tests
- No figures or visualizations
- Feature extraction never run on real data
- CASCADE distillation insight is in markdown, not code
- ~80KB of theory docs, ~3KB of findings — ratio is inverted
- Nothing a reviewer couldn't replicate by running logit-lens on any open model

---

## CASCADE Prerequisites

Before any CASCADE steering or distillation experiments, three prerequisites need to be explicit.

### 1. Artifact prerequisite: current logit-lens outputs are too lossy

The existing logit-lens JSON files store top-k tokens and top-k log-probs. That is enough for qualitative inspection, but not enough for:
- vocab-scale layer-to-layer decision vectors
- stable probability-vs-layer trajectories for a fixed target token when that token drops out of top-k
- pseudoinverse target construction

Required upgrade:
- add a richer logit-lens artifact format before more CASCADE analysis
- minimum acceptable format:
  - final predicted token id per tracked position
  - probability/log-probability of that fixed token at every layer
  - top-k token ids/log-probs for qualitative inspection
- preferred format:
  - full logits or full log-probabilities for tracked positions

### 2. Mathematical prerequisite: define the CASCADE target unambiguously

The teacher gives a probability distribution `p`, not a unique logit vector. CASCADE must therefore be defined on a canonical representative of the teacher distribution.

Working definition:
- let `C = I - (1/V)11^T` be the centering operator
- let `z_hat = C log p_teacher`
- define `A = C W`
- define `b = z_hat - C W x_t`
- define the CASCADE target as the minimum-norm least-squares solution:
  - `x_e* = argmin_x ||A x - b||_2^2`
  - equivalently `x_e* = A^+ b`

This is the object that is mathematically justified. It is not exact recovery of a "true" internal state.

### 3. Empirical prerequisite: calibrate convergence layers across families

The current layer numbers `L18`, `L20`, `L21` come from single exemplar prompts. Before using them to drive family-level ablation studies, estimate the convergence-layer distribution across each task family.

Required output:
- mean and variance of convergence layer by task family
- whether a fixed family-level ablation layer is valid
- if not, whether ablations should target per-case convergence layers instead

---

## Priority-Ordered Implementation Steps

### TIER 0: Prerequisites (must be completed first)

#### Step 0.1: Upgrade logit-lens artifacts
**What**: Extend the logit-lens output format so later CASCADE analyses are actually supported by saved data.
**Why**: The current top-k-only JSON is insufficient for decision-vector extraction, pseudoinverse targets, and stable trajectory plots.
**How**:
1. Update `scripts/run_logit_lens.py` and the underlying readout code to save one of:
   - full logits/log-probs for tracked positions, or
   - a compact tracked-token format with fixed-token trajectories plus top-k tables
2. Re-run the three existing prompts with the richer artifact format
3. Keep the current markdown report, but treat it as presentation output rather than the canonical analysis artifact
**Output**: revised `runs/logit_lens_*/logit_lens_data.json` with enough information for downstream analysis

#### Step 0.2: Implement a gauge-safe CASCADE target note and reference code
**What**: Write the CASCADE target in one place using a centering operator and least-squares language.
**Why**: This prevents later code from silently optimizing the wrong object.
**How**:
1. Define `C = I - (1/V)11^T`
2. Define `A = C W`
3. Define `b = C(log p_teacher - W x_t)`
4. Define `x_e* = A^+ b`
5. Add a small reference implementation that computes this target for one position and reports residual norms
**Output**: a reference script and a short mathematical note linked from the README

#### Step 0.3: Family-level convergence calibration
**What**: Run logit-lens over the existing benchmark cases and estimate convergence-layer distributions by task family.
**Why**: The current fixed-layer ablation plan assumes exemplar prompts are representative.
**How**:
1. Add a script that computes convergence layers for all benchmark cases
2. Summarize mean, std, and histogram by family
3. Decide whether later ablations should use:
   - fixed family-level layers, or
   - per-case convergence layers
**Output**: `runs/convergence_calibration/` plus one summary figure

### TIER 1: Concrete findings from existing code (days, after prerequisites)

#### Step 1: Head Ablation Sweep → Hydra Measurement
**What**: Ablate each of the 64 heads at layer 20 individually. Compute the variance of ablation effects across heads.
**Why**: This is a single number that tests the Hydra hypothesis (distributed redundancy) at production scale. The PLS paper's core claim is that PLS breaks the Hydra effect (measured by ablation-effect variance). Measuring this on gpt-oss-20b provides a negative control at scale.
**How**:
1. Create config `configs/head_ablation_L20.py`:
   - 64 interventions: `HEAD_MASK` at layer 20, one head at a time (heads 0-63), scale=0.0
   - Run all 5 task families
2. Run: `python scripts/run_benchmark.py --config configs/head_ablation_L20.py`
3. Analyze: compute σ² of margin-change across the 64 heads
4. Compare to PLS paper Table 2 (PLS σ=0.47 vs control σ=0.08)
**Output**: `runs/head_ablation_L20/` with case_results.csv, summary.json, report.md
**Expected result**: Tight variance (Hydra active) — gpt-oss-20b is standard-trained, so it should match the paper's control condition.
**Time**: ~2 hours (64 forward passes × 20 cases = 1,280 forward passes)

#### Step 2: Decision Transition Extraction (CASCADE Steering Concept)
**What**: Extract layer-to-layer decision transitions from the richer logit-lens artifacts. Distinguish between qualitative transition summaries and quantitative CASCADE targets.
**Why**: This demonstrates the core insight without pretending the current artifacts determine more than they do.
**Contrast with standard steering (CAA)**: Standard CAA requires collecting 100+ positive and 100+ negative examples, running forward passes on all of them, and taking the mean activation difference at a chosen layer — with no guarantee the resulting vector captures the intended concept. CASCADE steering reads the decision direction directly from the model's own computation trajectory. The model tells you what decision it made, at which layer, and the CASCADE architecture gives a closed-form projection of that decision into the student's x_e steering space. This is a fundamentally different paradigm: introspective steering from computation trajectories rather than contrastive steering from curated example pairs.
**How**:
1. Create script `scripts/extract_decision_vectors.py` that:
   a. Loads the upgraded logit-lens artifacts from `runs/logit_lens_*/logit_lens_data.json`
   b. For each position, computes:
      - qualitative top-k transition summaries
      - quantitative centered-logit differences `C(z^(l+1) - z^(l))` when dense data is available
   c. Identifies "decision layers" where the top-1 prediction changes or tracked probability mass shifts > 0.1
   d. For each decision layer ℓ→ℓ+1:
      - Reports which tokens gained probability (top 5 gainers)
      - Reports which tokens lost probability (top 5 losers)
      - Computes the centered decision direction when possible
      - Labels the semantic content of the decision (e.g., "noise→content", "content→adjective", "adjective→'small'")
   e. For a given embedding matrix `W`, computes the gauge-safe CASCADE projection:
      - `v_steer = (C W)^+ C Δz`
      - Reports the norm and top-k nearest vocabulary items to v_steer
   f. Outputs a markdown report and a JSON file with all decision vectors
2. Run on all 3 existing logit-lens prompts
**Output**: `runs/decision_vectors/` with per-prompt reports showing the semantic decision trajectory
**Key insight to demonstrate**: The recency prompt at position 12 shows:
- Layers 0-10: noise (random tokens)
- Layers 10-16: endoftext dominates (padding detection)
- Layer 16→17: endoftext→"too" (content token recognition)
- Layer 17→18: →"small" (adjective selection — THE key decision)
- Layer 18→21: "small" confidence increases (refinement, not new decision)
Each transition is a candidate steering direction. In CASCADE mode, the quantitative object is the centered least-squares projection, not a claim of exact latent-state recovery.
**Time**: ~2 hours coding, no GPU needed (analysis of existing JSON files)

#### Step 3: Convergence Trajectory Visualization
**What**: Create a figure showing probability-of-correct-token vs. layer for all 3 tasks.
**Why**: A single figure showing three different convergence curves is worth more than 10 pages of framework text. It's the visual proof that different behaviors converge at different layers.
**How**:
1. Create script `scripts/plot_convergence_trajectories.py` that:
   a. Loads the upgraded logit-lens JSON for each prompt
   b. Extracts the target token probability at each layer from the fixed-token trajectory data
   c. Plots 3 curves on one axis: P(target) vs. layer
   d. Marks the convergence layer for each with a vertical line or marker
   e. Saves as PDF and PNG
2. Use matplotlib, keep it clean (no seaborn flourishes)
**Output**: `figures/convergence_trajectories.{pdf,png}`
**Time**: ~1 hour

#### Step 4: Add Tests
**What**: Add pytest tests for the dry-run backend, feature extractor (on synthetic data), benchmark harness, and logit-lens readout.
**Why**: Absence of tests signals "prototype, not engineering." Tests also ensure future changes don't break working code.
**How**:
1. Create `tests/` directory with `__init__.py`
2. `tests/test_dry_run.py` — run dry-run config, check output format and expected results
3. `tests/test_feature_extractor.py` — create synthetic attention/logit tensors, verify feature dimensions and invariances (same computation at different positions → same features)
4. `tests/test_benchmark_runner.py` — verify config parsing, intervention expansion, output generation
5. `tests/test_logit_lens.py` — verify convergence detection on synthetic layer predictions
6. Add missing `__init__.py` files in: `benchmarks/`, `capture/`, `harmony/`, `interventions/`, `readouts/`, `reports/`
**Output**: All tests pass under `pytest tests/`
**Time**: ~3 hours

### TIER 2: Original insight (days to a week)

#### Step 5: Ablation-Effect Variance Across Tasks and Layers
**What**: Run the 64-head ablation sweep at calibrated convergence layers for each task family. Show that different heads matter for different tasks.
**Why**: This directly tests whether the model has discoverable computational modes — the feature framework's core prediction. Different tasks should activate different heads at their respective convergence layers.
**How**:
1. Use the convergence-calibration output from Step 0.3 to choose:
   - fixed family-level layers, or
   - per-case layers
2. Create configs for the chosen policy
3. Run each: approximately 64 heads × 20 cases × 3 layers = 3,840 forward passes
3. For each task × layer combination, compute:
   - Which heads have the largest ablation effect
   - Whether the "important" heads differ by task
   - The ablation-effect variance at each layer
4. Create a heatmap: heads (x-axis) × tasks (y-axis), colored by ablation effect
**Output**: `runs/head_ablation_L18/`, `runs/head_ablation_L21/`, plus `figures/ablation_heatmap.{pdf,png}`
**Expected result**: Different heads are important for different tasks near their convergence layers. If all heads matter equally → Hydra is total, model is not modular. If specific heads dominate per task → modular computation exists.
**Time**: ~6 hours (GPU-bound)

#### Step 6: CASCADE Pseudoinverse Feasibility Test
**What**: Compute gauge-safe least-squares CASCADE targets and characterize identifiability, conditioning, and behavioral adequacy.
**Why**: Before building a full pipeline, verify that the target is numerically stable and behaviorally useful. This test does not ask whether the teacher logits are exactly reconstructable; it asks whether the centered least-squares target is a useful surrogate.
**How**:
1. Create script `scripts/test_cascade_pseudoinverse.py` that:
   a. Loads the gpt-oss-20b model (just need embedding matrix and lm_head weights)
   b. For a given prompt, gets the teacher's final-layer probability distribution
   c. Looks up x_t (token embedding for the predicted position)
   d. Computes W = lm_head weight matrix [V, d]
   e. Defines the centering operator `C` and the system matrix `A = C W`
   f. Computes condition information for `A`
   g. Computes `b = C(log p_teacher - W·x_t)`
   h. Computes `x_e* = A⁺ · b` (using `torch.linalg.lstsq` or `pinv`)
   i. Measures residual norm: `‖A x_e* - b‖ / ‖b‖`
   j. Measures distribution match after reconstruction
   k. Reports nullspace size / effective rank and whether multiple internal targets are observationally equivalent
2. Run on all 3 logit-lens prompts, all positions
**Key question**: For gpt-oss-20b with V=201,088 and d=2,880:
- W is 201088×2880 — extremely overdetermined (201K equations, 2880 unknowns)
- the CASCADE target is a least-squares projection in centered-logit space
- exact reconstruction occurs only if `b` lies in the column space of `C W`
- low residual and low KL support CASCADE as a useful surrogate
- high residual means the vocab-space target is not well captured by the student's bottleneck
**Note on vocabulary mismatch**: This test uses the teacher's OWN W and vocabulary (201K). A student with GPT-2 BPE (50K) would need a vocabulary projection first. This test validates the approach in the cleanest setting.
**Output**: `runs/cascade_feasibility/` with per-prompt reconstruction errors and condition numbers
**Time**: ~3 hours (mostly coding; computation is fast — one SVD of a 201K×2880 matrix)

#### Step 7: Feature Extraction on Existing Prompts
**What**: Run the extended Tier-2 feature extractor on the 3 logit-lens prompts.
**Why**: The feature extractor is implemented but never demonstrated. Running it shows the methodology works on gpt-oss-20b.
**How**:
1. Run: `python scripts/run_feature_extraction.py --model openai/gpt-oss-20b --prompt "The trophy would not fit..." --output runs/features_recency/`
2. Repeat for induction and syntax prompts
3. If attention hooks aren't working (MXFP4 limitation), run with `--no-attention` for Components A+B only (trajectory + stability)
4. Create a summary showing: convergence layer (from features) matches convergence layer (from logit-lens)
**Output**: `runs/features_*/` with feature tensors and geometric analysis reports
**Possible blocker**: Attention hooks may not work with MXFP4 quantization. If so, Components A+B (trajectory, stability — 73 dims) still demonstrate the methodology. Document the MXFP4 limitation explicitly.
**Time**: ~2 hours

### TIER 3: Unique contribution (week+)

#### Step 8: Feature-Predicted vs. Actual Ablation Importance
**What**: Correlate head activation features (Component C values) with actual ablation effect sizes from Step 5.
**Why**: This tests the core hypothesis: does the feature system predict causal structure without ablation? If feature-predicted importance correlates with actual ablation importance → the feature system is a cheap proxy for expensive causal analysis. Nobody has published this test.
**How**:
1. From Step 5: for each (task, layer, head), we have the ablation effect size
2. From Step 7: for each (token, layer, head), we have the feature activation value
3. Compute rank correlation (Spearman) between feature activation and ablation effect per task
4. Report: does high feature activation predict high causal importance?
**Depends on**: Steps 5 and 7 both completed
**Output**: Correlation coefficients and scatter plots
**Time**: ~2 hours (analysis only, uses existing results)

#### Step 9: Minimal CASCADE Distillation Prototype
**What**: If Step 6 shows well-conditioned solutions, train a tiny CASCADE student on x_e* regression targets.
**Why**: Demonstrates the CASCADE distillation concept end-to-end.
**How**:
1. Use the symbolic transformer codebase (PLS paper) for the student architecture
2. Configure a 2-layer, 4-head CASCADE model (tiny — for proof of concept)
3. Compute x_e* targets from teacher logit-lens data (using the approach from Step 6, projected to GPT-2 50K vocab)
4. Train the student FFN to predict x_e* (regression warmup)
5. Fine-tune with KL on teacher's final distribution
6. Evaluate: does the student's output distribution approximate the teacher's?
**Depends on**: Step 6 shows feasibility, plus access to symbolic transformer training code
**Output**: Trained student checkpoint + reconstruction quality metrics
**Time**: ~1 week

#### Step 10: Computational Mode Discovery on Larger Corpus
**What**: Run logit-lens on 100+ diverse prompts, cluster by convergence layer and feature vectors, show that clusters correspond to task types.
**Why**: Demonstrates that the model has discoverable computational modes at scale, not just on 3 handpicked prompts.
**Depends on**: Steps 3 and 7 working
**Time**: ~1 week (GPU-bound for 100+ forward passes with full instrumentation)

---

## Document Consolidation

The repo currently has 6 heavyweight research documents. For a portfolio piece, consolidate:

### Keep as-is
- `README.md` — expand with results sections as Steps 1-7 complete
- `GEOMETRIC_FRAMEWORK.md` — tightened in this session, now mathematically precise
- `STYLE_GUIDE.md` — coding conventions

### Merge into one document
Combine these three into `RESEARCH_DIRECTIONS.md`:
- `STEERING_AND_DISTILLATION_PLAN.md` — steering experiments + distillation pipeline
- `CASCADE_DISTILLATION.md` — matrix factorization derivation + staged training
- `INTERMEDIATE_LAYERS_ANALYSIS.md` — honest assessment of intermediate layer value

### Keep but deprioritize
- `RESEARCH_CONNECTIONS.md` — useful context but not for a reviewer's first impression
- `TODO.md` — superseded by this document for new work; keep old intervention experiments as reference
- `IMPLEMENTATION_PLAN.md` — Phases 1-5 are done; Phase 6 is superseded by this plan

### Signal-to-noise principle
A reviewer scanning the repo should see: **results first, code second, theory third.**
- `runs/` and `figures/` should be the most prominent directories
- README.md should lead with findings, not architecture descriptions
- Framework documents should be linked from findings, not vice versa

---

## Dependencies and Ordering

```
Step 1 (Head ablation L20)  ──────────────────────────────┐
Step 2 (Decision vectors)   ──────────────────────────────┤
Step 3 (Convergence figure) ──────────────────────────────┤──→ Update README with results
Step 4 (Tests)              ──────────────────────────────┘

Step 5 (Multi-layer ablation) ← depends on Step 1 working
Step 6 (Pseudoinverse test)   ← independent
Step 7 (Feature extraction)   ← independent

Step 8 (Feature vs ablation)  ← depends on Steps 5 AND 7
Step 9 (CASCADE prototype)    ← depends on Step 6
Step 10 (Large corpus modes)  ← depends on Steps 3 and 7
```

Steps 0.1, 0.2, 0.3 should be completed before any CASCADE-heavy work.
Steps 1 and 4 are independent of CASCADE and can run immediately.
Steps 2 and 3 depend on Step 0.1.
Step 5 depends on Step 0.3.
Steps 6 and 9 depend on Step 0.2.
Step 7 is independent of CASCADE but useful background.
Steps 8, 9, 10 have explicit dependencies.

---

## Existing Data Available for Reuse

### Logit-lens JSON files (no GPU needed to analyze)
- `runs/logit_lens_recency/logit_lens_data.json` (175 KB)
  - Prompt: "The trophy would not fit in the suitcase because the suitcase was too"
  - 24 layers × 13 positions × top-10 tokens with logprobs
  - Key finding: position 12 converges at layer 18 to "small"
  - Decision trajectory: noise → endoftext → content → "small"

- `runs/logit_lens_induction/logit_lens_data.json` (110 KB)
  - Prompt: "A7 B2 C9 D4 A7 B2 C9"
  - Key finding: converges at layer 21 to "D" (induction/copying)

- `runs/logit_lens_syntax/logit_lens_data.json` (41 KB)
  - Prompt: "The keys to the cabinet"
  - Key finding: converges at layer 20 to "are" (subject-verb agreement)

### Benchmark results
- `runs/gpt_oss_20b_sweep/case_results.csv` (52 KB)
  - 320 test cases with head/layer/expert ablation results
  - Key finding: Layer 20 ablation → 89% margin collapse (3.114 → 0.359)
  - Key finding: L12 head ablation → margin INCREASES (3.178 vs 3.114)

### Model architecture info
- gpt-oss-20b: L=24, H_q=64, H_kv=8, d=2880, V=201088, E=32, top-4 MoE
- MXFP4 quantization blocks router hooks and possibly attention hooks
- Fits on single 4090 (24GB VRAM)

---

## PLS Paper Code Available for Reference

Located at: `companion-repo/neurips-2026-activation-clustering/`

### Directly reusable
- `src/activation_clustering/features/feature_extractor.py` — TensorFeatureExtractor (reference implementation)
- `info/paper/generate_paper_figures.py` — matplotlib figure generation patterns
- `src/activation_clustering/analysis/cluster_probes.py` — HDBSCAN clustering pipeline
- `src/activation_clustering/data/builder.py` — batch feature extraction with progress tracking

### Reference for architecture
- `src/activation_clustering/cli/causal_steering.py` — comprehensive intervention experiments
- `src/activation_clustering/annotations/` — attention visualization

---

## What "Done" Looks Like

### Minimum viable portfolio (Steps 1-4 complete)
- README leads with 3 concrete findings (Hydra measurement, decision vectors, convergence trajectories)
- One figure showing convergence curves
- One table showing ablation-effect variance
- One report showing semantic decision vectors with CASCADE projection
- Tests pass
- A reviewer thinks: "This person can work with production models and extract insight"

### Strong portfolio (Steps 1-7 complete)
- All of the above, plus:
- Heatmap showing task-specific head importance
- CASCADE feasibility validated (condition number, reconstruction error)
- Feature extraction demonstrated on gpt-oss-20b
- A reviewer thinks: "This person has original ideas that could work at scale"

### Exceptional portfolio (Steps 1-10 complete)
- All of the above, plus:
- Feature-predicted importance correlates with actual ablation importance
- CASCADE distillation prototype producing reasonable outputs
- Computational modes discovered on a real corpus
- A reviewer thinks: "This is publishable work"
