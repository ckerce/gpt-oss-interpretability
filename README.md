# gpt-oss-interp

## Context

This repository is the **production-scale validation** half of a two-part research program on mechanistic interpretability.

The first part, developed across three preprints ([inspectable internals via stream separation](https://arxiv.org/abs/2603.07461), [head specialization via per-layer supervision](https://arxiv.org/abs/2603.18029), [preserving specialization into late layers via delayed integration](https://arxiv.org/abs/2603.07482)), demonstrates at controlled scale (22M–53M parameters) that architectural constraints — dual-stream decomposition, per-layer supervision, gated attention — make transformer internals causally inspectable. The core finding is that per-layer supervision breaks the Hydra effect, exposing 5–23x larger ablation effects and enabling 4x greater steering control than standard-trained models.

This repository takes the **same inspection toolkit** — causal intervention benchmarks, per-layer logit-lens readouts, direct-vocabulary steering, activation capture — and applies it to OpenAI's **gpt-oss-20b** (21B params, 3.6B active), a production MoE transformer trained without any interpretability constraints. The goal is to answer: do the measurement methods transfer, and what do they reveal about a standard model at scale?

## Architecture Target

**gpt-oss-20b** (`GptOssForCausalLM`): 24-layer MoE transformer.

| Component | Detail |
|-----------|--------|
| Attention | 64 GQA query heads, 8 KV heads, head_dim=64 |
| Pattern | Alternating sliding (128-token window) / full attention |
| MoE | 32 experts, top-4 routing, SwiGLU |
| Position | RoPE with YaRN scaling (131K context) |
| Vocab | 201,088 tokens (o200k_harmony BPE) |
| Quantization | MXFP4 on expert weights; attention and router in bf16 |

Two aspects are particularly relevant to the measurement strategy: the alternating sliding/full attention pattern constrains which layers can attend globally, and MXFP4 quantization on expert weights limits hook-based router introspection (see Addenda).

## Measurement Approach

Four instruments are used throughout: **logit-lens readouts** track the model's per-layer top-1 prediction to locate when each task is resolved; **causal ablation** confirms which layers are critical rather than merely correlated; **direct-vocabulary steering** applies exact unembedding directions to flip model answers; and **hook-based activation capture** records hidden states and routing decisions for offline analysis. All measurements are applied through PyTorch forward hooks without modifying model weights, across five task families: capitalization, coreference, induction, recency bias, and syntax agreement.

## Analysis Set Stratification

Not every benchmark case supports a clean mechanistic claim. Before presenting any findings, we separate cases by convergence stability — whether the model's top-1 prediction converges reliably and holds through the final layer.

![Stratification](figures/fig3_analysis_set_stratification.png)

A 4-way stratification yields:

| Category | Cases | Use |
|----------|------:|-----|
| Correct, late-stable | 9 / 20 | Main analysis set — supports causal claims |
| Correct, unstable | 4 / 20 | Convergence too noisy for mechanistic inference |
| Incorrect | 4 / 20 | Model fails; recency bias and syntax agreement largely here |
| Ambiguous | 3 / 20 | Top-1 oscillates near the decision layer |

Only 9/20 cases (45%) pass the filter. This is a feature, not a bug: it establishes which behaviors are robust enough for downstream intervention claims. Every finding below is scoped to this main analysis set unless otherwise noted. This approach — systematic stratification before claiming mechanism — is not standard in the interpretability literature; we are not aware of a prior explicit methodology for it.

## Key Findings

The nine solid threads form a four-phase argument.

---

### Phase 1 — Where does gpt-oss-20b compute each task?

**Finding 1 — Task-dependent convergence: different behaviors resolve at different depths (Thread 1)**

Per-layer logit-lens readouts with choice-relative convergence show that the depth at which the model commits to an answer is not uniform — it is task-specific:

![Convergence trajectories](figures/fig1_convergence_trajectories.png)

- **Capitalization**: converges early (L1–2), minimal depth processing
- **Coreference**: converges mid-depth (L5), requires semantic resolution
- **Induction**: converges late (L17+), consistent with induction heads as a late-layer phenomenon

The induction result is particularly interpretable. Standard induction benchmarks use sequences like `sun moon star sun moon star sun moon ___` — but these could be retrieved from training data. We test with arbitrary tokens that cannot plausibly be memorized:

**Clean structural induction**: `D 5 Z 7 B 2 D 5 Z 7 B 2 D 5 Z 7 B ___`
- Core pattern: `D 5 Z 7 B 2` repeats 3 times. After "B ", the model predicts **"2"** (converges at L19).

**Noisy structural induction**: `D 5 Z 7 B 2 D 5 A 7 B 2 D 5 W 7 B 2 D 5 X 7 B 2 D 5 Y 7 B 2 D 5 Q 7 B 2 D 5 R ___`
- Core pattern: `7 B 2 D 5 [letter]` — the letter changes each cycle (Z, A, W, X, Y, Q, R) while the rest repeats. After "R ", the model predicts **"7"** (converges at L20).

The model extracts the stable structural regularity `7 B 2 D 5` despite the varying letter — genuine in-context pattern recognition, not memorized sequence retrieval. The noisy case requires one additional layer (L20 vs L19), reflecting the extra computation needed to distinguish structure from noise.

This connects two lines of work. Olsson et al. (2022) identified **induction heads** as specific attention circuits implementing match-and-copy behavior through a phase transition in middle-to-late layers — consistent with our observation that structural induction requires deep processing (L19–L20) while simpler tasks resolve at L1–2. DeepSeek's **Engram module** (2025) replaces early FFN layers with O(1) hash-based n-gram lookup with no loss in model quality, validating that early layers can be externalized as non-parametric memory. The contrast is precise: the early-layer tasks that Engram can replace (pattern recall, factual retrieval) converge at L1–5 in our logit-lens analysis, while the structural computation that *cannot* be replaced by lookup (noisy induction) converges at L19–L20.

**Finding 2 — Late-layer ablation reveals that L19–21 are causally critical (Thread 2)**

Logit-lens convergence is correlational — it shows where the model's prediction stabilizes, not which layers produce that stability. Layer ablation answers the causal question.

Ablating individual layers on the 9-case main analysis set shows that layers 19–21 are where gpt-oss-20b resolves task-relevant behavior:

![Late-layer ablation](figures/fig2_late_layer_ablation.png)

Layers 19–21 ablation drops accuracy from 100% to 44% and collapses margin by 85–90%. Layer 23 ablation preserves accuracy (100%) despite margin loss, suggesting it refines rather than decides. This narrows the interpretability target from 24 layers to 3.

---

### Phase 2 — How do steering directions arise, and why is head-level intervention hard?

**Finding 3 — Decision trajectories reveal self-supervised steering directions (Thread 4)**

Having located the critical layers, the next question is what to steer toward. Each layer where the model's top-1 prediction changes is a "decision point." The logit-space difference at decision layers is a self-supervised steering direction — the model tells you what decision it made, and at which layer:

![Decision trajectories](figures/fig4_decision_trajectories.png)

Example decision arc for recency-bias position 12 ("suitcase was too ___"):
```
L0: noise → L8: '‑' → L13: 'pack' → L14: <eos> → L16: '(' → L17: 'too' → L18: 'small'
```

This is fundamentally different from contrastive activation addition (CAA), which requires 100+ curated positive/negative example pairs. CASCADE reads steering directions directly from the model's own computation, with no external annotation.

**Finding 4 — Per-head Hydra measurement confirms why head-level intervention fails (Thread 5)**

Having identified candidate steering layers (L19–21) and candidate directions, the natural intervention target is individual attention heads. But ablating each of 64 heads individually at L20 produces near-identical margins (σ = 0.042) — the model barely notices losing any single head:

![Hydra variance](figures/fig5_hydra_variance.png)

gpt-oss-20b's σ = 0.042 is **half** the PLS-paper control (σ = 0.08) and **11× smaller** than PLS-trained models (σ = 0.47). This directly validates the Hydra hypothesis at production scale: standard training produces extreme distributed redundancy. There is no individual attention head to target — the computation is spread uniformly across all 64. This explains why head-level circuit-finding is difficult in standard models.

**Finding 5 — Direct-vocabulary steering works despite Hydra, with positional specificity (Thread 6)**

Despite the absence of specialized heads, exact vocabulary-space directions (`W[token_A] - W[token_B]`) applied in the contextual stream at late layers cleanly flip model answers:

![Steering heatmaps](figures/fig_matched_pair_heatmaps.png)

Crucially, the effect is **position-specific**: steering at the decision-token position flips answers; identical steering at token 0 produces zero effect. This rules out diffuse perturbation artifacts. The vocabulary direction, applied at the right layer and position, is sufficient to override the model's answer — even though no individual head is responsible for producing it.

---

### Phase 3 — How selective and channel-specific is the steering?

Threads 7 and 8 require channel-level analysis that is tractable only in architecturally structured companion models, where stream separation makes individual dimensions interpretable. These experiments use three matched 71M–22M DST models from the companion preprint work:

| Label | Architecture | Mixing | Params | Description |
|-------|-------------|--------|-------:|-------------|
| DST-baseline | Single-stream dense | dns-dns/dns-dns | 71M | Standard transformer baseline (no stream separation) |
| DST-cascade | CASCADE dual-stream | dns-dns/dns-dns + gated attention | 71M | Dual-stream with frozen symbolic stream; attention output cascades into FFN |
| DST-independent | Fully independent channels | ind-ind/ind-ind | 22M | Complete channel isolation — maximum interpretability |

The mixing signature `attn-attn/ffn-ffn` describes how heads share information at four mixing points (attention value, attention output, FFN up-projection, FFN down-projection): `dns` = dense (standard), `ind` = independent (no cross-head mixing).

**Finding 6 — Probing and causal intervention dissociate: they identify different channels (Thread 7)**

Given a steering direction, which hidden-state dimensions carry the signal? The naive approach is to probe each channel for answer-predictive content, then intervene on high-probe-weight channels. Thread 7 tests whether this is valid.

Per-channel causal analysis on DST-cascade reveals that the channels that probe well for the answer (H4) are systematically *not* the channels whose intervention most changes the output (H2, H5). Spearman correlation between probe weight and causal effect = **−0.363** on the induction task.

This is the probe-causation dissociation: probing identifies channels whose activation is *correlated* with the model's readout; intervention identifies channels whose activation *drives* the model's computation. These are different things. The practical implication is that probing alone cannot identify intervention targets — causal confirmation is required.

**Finding 7 — Steering selectivity is task-dependent: recency concentrates, induction distributes (Thread 8)**

Given that steering affects specific channels (not all of them), how broadly does the effect spread across the model? Thread 8 measures selectivity as the ratio of within-target-task effect to cross-task effect.

Cross-family comparison on DST-cascade:

| Task | Channelized/whole ratio | Interpretation |
|------|------------------------:|----------------|
| Recency | 0.80 | Steering concentrates in a few channels |
| Induction | 0.60 | Steering is distributed across many channels |

Recency steering is spatially concentrated; induction steering is spatially distributed. The right intervention granularity is task-dependent — a single steering strategy does not generalize across task families.

---

### Phase 4 — Why does linear steering work geometrically?

Phases 2 and 3 establish that vocabulary-space directions cleanly flip answers at some layers and for some tasks, but not uniformly. Phase 4 provides the geometric explanation.

**Finding 8 — Standard transformers have low effective rank at intermediate layers; stream separation improves conditioning up to 22x (Thread 14)**

Linear steering methods implicitly assume that the hidden-state geometry is well-conditioned — that a direction in embedding space corresponds to a meaningful direction in the model's internal representation. Thread 14 tests whether this assumption holds.

Using a 2×2 factorial design (stream separation × auxiliary loss) across four matched 45.4M transformers, we extend Park et al.'s output-layer Bregman analysis inward to intermediate layers:

- **Standard single-stream transformers**: effective rank 8 in 516 dimensions at intermediate layers. Linear methods operate in **2% of the available geometry**.
- **Stream-separated models**: conditioning improves up to **22x** — not by changing the training objective, but by architectural constraint.

A **cosine diagnostic** predicts which layers are safe for linear intervention: when the cosine similarity between the steering direction and the leading eigenvectors of the layer's Hessian is low, the intervention is operating in a nearly-null subspace and will be ineffective or unpredictable.

This explains the pattern observed in Phases 2–3 (measured on companion models; inferred for gpt-oss-20b): steering succeeds at late layers (L19–21) because those layers have relatively better-conditioned geometry, suggesting the decision region is where the model's representational manifold aligns with vocabulary space. It also explains task-dependent selectivity: recency concentrates because the recency direction aligns well with the dominant geometry; induction distributes because the induction direction is spread across many low-eigenvalue components.

---

### Summary

The nine solid threads form a coherent argument:

**Phase 1** locates computation by task type (convergence) and confirms causal criticality (ablation), narrowing the interpretability target from 24 layers to 3.

**Phase 2** shows that the model's own decision transitions yield self-supervised steering directions (no curated contrast pairs needed), that despite extreme head redundancy (Hydra), vocabulary-space directions applied at the right layer and position cleanly flip answers with positional specificity.

**Phase 3** reveals that probing and causation identify different channels, and that steering selectivity is task-dependent — meaning the right intervention granularity must be determined empirically per task.

**Phase 4** provides the geometric foundation: standard transformers have near-degenerate geometry at intermediate layers (rank 8 in 516 dimensions), which explains both where steering succeeds and why selectivity varies. Stream separation improves conditioning up to 22x, making the companion DST models tractable for channel-level analysis.

---

## Research Threads

This project is organized as 14 research threads at three maturity levels. See **[THREAD_MAP.md](THREAD_MAP.md)** for the full index with links, run counts, and figures.

### Solid — completed results

| # | Thread | Model | Problem | Contribution | Impact |
|---|--------|-------|---------|--------------|--------|
| 1 | [Convergence](threads/solid/1-convergence-logit-lens/) | gpt-oss-20b | Where in the network does each task get resolved? | Applies logit lens (prior work) to a production MoE; measures task-dependent convergence depth | Locates computation by task type — prerequisite for targeted intervention |
| 2 | [Late-layer ablation](threads/solid/2-late-layer-ablation/) | gpt-oss-20b | Which layers are causally critical, not just correlated? | Standard ablation methodology applied at production MoE scale with attention-vs-MoE decomposition | Narrows the interpretability target from 24 layers to 3 (L19–21) |
| 3 | [Analysis set filtering](threads/solid/3-analysis-set-filtering/) | gpt-oss-20b | Which test cases support honest mechanistic claims? | Systematic 4-way stratification by convergence stability; not aware of prior systematic methodology for this in interpretability | Only 45% of cases pass — sets defensible scope for all downstream threads |
| 4 | [Decision trajectories](threads/solid/4-decision-trajectories/) | gpt-oss-20b | Can the model's own prediction changes serve as steering directions? | Extracts logit-space directions from prediction-transition layers; not aware of prior work using these as self-supervised steering signals | Provides steering directions without curated contrastive pairs; empirical basis for CASCADE |
| 5 | [Hydra / head redundancy](threads/solid/5-hydra-head-redundancy/) | gpt-oss-20b | Are individual attention heads specialized or redundant? | Measures the Hydra effect (from companion PLS preprint) at 21B-param production scale | σ=0.042 confirms extreme redundancy — explains why circuit-level interpretation is hard in standard models |
| 6 | [Direct vocab steering](threads/solid/6-direct-vocab-steering/) | DST-baseline, DST-cascade | Can exact vocabulary directions flip model answers with positional precision? | Uses raw `W[A]−W[B]` unembedding directions for steering; not aware of prior work using exact vocabulary differences (vs learned directions) | Position specificity distinguishes targeted intervention from diffuse perturbation |
| 7 | [Channel probing](threads/solid/7-channel-probing/) | DST-cascade | Which hidden-state dimensions carry the steering signal? | Per-channel causal analysis: probe-promoted channels (H4) do not predict causal importance (H2, H5); Spearman = -0.363 on induction | Probing identifies readout-correlated channels; causal intervention reveals computation-driving channels — these are different |
| 8 | [Selectivity](threads/solid/8-selectivity/) | DST-cascade | Does steering affect only the target behavior? | Cross-family comparison: recency channelized/whole ratio 0.80 (recency) vs 0.60 (induction) on same model | Recency steering signal is concentrated in few channels; induction is distributed — task-dependent steering granularity |
| 14 | [Bregman geometry](threads/solid/14-bregman-geometry/) | 4 matched 45.4M transformers | Are linear interpretability methods geometrically valid at intermediate layers? | Extends Park et al.'s output-layer Bregman analysis inward; 2x2 factorial (stream separation x aux. loss) | Standard transformers have effective rank 8/516 at intermediate layers — linear methods operate in 2% of the geometry; cosine diagnostic predicts steering validity |

### In progress — code and initial experiments exist

| # | Thread | Model | Problem | Contribution | Impact |
|---|--------|-------|---------|--------------|--------|
| 9 | [Feature extraction](threads/in-progress/9-feature-extraction/) | gpt-oss-20b | Can computational modes be captured as unified feature vectors? | 6,425D features across 555 tokens / 5 families; intrinsic dimension ranges from 20 (syntax) to 68 (coreference) | Task-dependent dimensionality — syntax is structured, coreference is distributed |
| 10 | [Bridge / cross-model](threads/in-progress/10-bridge-cross-model/) | gpt-oss-20b, Gemma-3-1B | Do these findings generalize beyond gpt-oss-20b? | Screening pipeline to evaluate new models for interpretability compatibility | Early infrastructure — one model (Gemma-3-1B) screened so far |

### Theoretical — framework documented, not yet implemented

Threads 11–13 (CASCADE distillation, geometric framework, attention path sensitivity) have theoretical specifications but incomplete or stub implementations. See [THREAD_MAP.md](THREAD_MAP.md) for details.

---

## Addenda

### CASCADE feasibility

The gauge-safe pseudoinverse CASCADE target (`x_e* = (CW)⁺ · C(log p - Wx_t)`) reconstructs teacher distributions with:

| Prompt | Relative residual | KL divergence |
|--------|------------------:|-------------:|
| Recency ("small") | 0.0029 | 1.2e-5 |
| Syntax ("can") | 0.0029 | 8.9e-5 |
| Induction ("D") | 0.0030 | 1.2e-4 |

In the same-model setting, the centered least-squares target is numerically excellent. This validates the mathematical machinery before attempting cross-vocabulary distillation.

### MXFP4 quantization and the interpretability tradeoff

MXFP4 fused kernels bypass Python-level forward hooks on the router module. Router introspection is opaque under quantization — expert masking operates at the MLP output level, not gate-level. This is a concrete instance of a general tradeoff: compression techniques that fuse operations reduce the surface area for mechanistic inspection.

---

## Quick Start

```bash
# Install
pip install -e .
pip install kernels   # MXFP4 support

# Download model (~13 GB)
huggingface-cli download openai/gpt-oss-20b

# Smoke test (no GPU needed)
python scripts/run_benchmark.py --config configs/dry_run_recency.py

# Run tests
pytest tests/

# Intervention benchmark on real model
python scripts/run_benchmark.py --config configs/head_ablation_L20.py

# Logit-lens analysis
python threads/solid/1-convergence-logit-lens/run_logit_lens.py \
    --model openai/gpt-oss-20b \
    --prompt "The trophy would not fit in the suitcase because the suitcase was too" \
    --output runs/logit_lens_demo/

# Feature extraction
python threads/in-progress/9-feature-extraction/run_feature_extraction.py \
    --model openai/gpt-oss-20b \
    --prompt "The trophy would not fit in the suitcase because the suitcase was too" \
    --output runs/features_demo/

# Generate figures from existing data (no GPU needed)
python scripts/generate_phase1_figures.py
python threads/solid/4-decision-trajectories/generate_decision_figure.py
```

## Repository Structure

```
THREAD_MAP.md                    # Index of all 14 research threads
threads/                         # Thread-specific scripts, docs, and READMEs
  solid/                         # 9 publication-ready threads
  in-progress/                   # 2 threads with code but thin experiments
  theoretical/                   # 3 threads with frameworks but no implementation

gpt_oss_interp/                  # Shared Python package (~40 modules)
├── config.py                    # Dataclasses: tasks, interventions, benchmarks
├── backends/                    # Model execution (dry_run + real gpt-oss-20b)
├── benchmarks/                  # Task library (5 families, 36 cases), runner, pools
├── capture/                     # Activation and routing capture via hooks
├── features/                    # Feature extraction + geometry
├── readouts/                    # Per-layer logit-lens readouts
├── interventions/               # Intervention sweep expansion
├── steering/                    # Probing, causal analysis, selectivity
├── distillation/                # CASCADE workflows (stubs)
├── common/                      # Shared utilities (artifacts, I/O)
└── reports/                     # CSV, JSON, Markdown output

scripts/                         # Shared CLI tools (4 cross-thread scripts + one_off/)
configs/                         # 7 benchmark configurations
runs/                            # 52 experiment output directories
figures/                         # 16 publication-quality figures (PDF + PNG)
tests/                           # pytest suite
doc/references/                  # Literature reviews + academic papers
```

## Intervention Types

| Kind | Target | Mechanism |
|------|--------|-----------|
| `HEAD_MASK` | Attention heads | Scale specific GQA head outputs |
| `EXPERT_MASK` | MoE experts | Proportional MLP output scaling |
| `LAYER_SCALE` | Transformer blocks | Scale block delta (residual-preserving) |
| `TEMPERATURE_SCALE` | Attention | Scale attention logits |

## Design Principles

- **Results first**: `figures/` and `runs/` are the most prominent directories
- **Backend-agnostic benchmarks**: benchmark code never sees model internals
- **Config-driven experiments**: Python config files, not CLI flags
- **Hook-based inspection**: PyTorch forward hooks for capture and intervention
- **Document what doesn't work**: MXFP4 limitations are findings, not failures
- **Honest analysis sets**: filter cases by convergence stability before claiming mechanism

## Companion Work

This toolkit validates at production scale the same ideas demonstrated at controlled scale in three preprints:

- [**The Dual-Stream Transformer**](https://arxiv.org/abs/2603.07461) — interpretability through stream separation (2.5% loss cost for full decomposition)
- [**Engineering Verifiable Modularity via Per-Layer Supervision**](https://arxiv.org/abs/2603.18029) — PLS + gated attention yields 5-23x larger ablation effects, exposing hidden modularity
- [**Interpretable-by-Design Transformers via Architectural Stream Independence**](https://arxiv.org/abs/2603.07482) — delayed integration concentrates coreference resolution in L3–L4 vs. distributed across layers in standard transformers; targeted head suppression produces 4.25x less collateral damage (Cohen's d: -0.158 vs. -0.672)
- **Stream Separation Improves Bregman Conditioning in Transformers** (in preparation) — geometric account of why stream separation aids steerability: standard single-stream transformers have effective rank 8 in 516 dimensions at intermediate layers (linear methods operate in 2% of the geometry); stream separation improves conditioning up to 22x without changing the training objective. Provides a cosine diagnostic for predicting which layers are safe for linear intervention.
