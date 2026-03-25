# gpt-oss-interp

Production-scale causal analysis of `gpt-oss-20b` for mechanistic interpretability.

## Scope

This repository brings together standard mechanistic interpretability techniques — logit-lens convergence mapping, causal ablation, direct-vocabulary steering, hook-based activation capture — and applies them to `gpt-oss-20b` as a coherent measurement program. `gpt-oss-20b` presents properties that complicate standard analysis — MoE routing with MXFP4-quantized expert weights that bypass hook-based router introspection, alternating sliding/full attention, and limited public documentation — making it a harder test of methodology than analyzing a well-documented dense model. It also serves as the production-scale validation ground for a parallel line of work on architectural modifications designed to make mechanistic interpretability lower-friction. Four companion papers develop those techniques:

- [Engineering Verifiable Modularity in Transformers via Per-Layer Supervision](https://arxiv.org/abs/2603.18029)
- [The Dual-Stream Transformer: Channelized Architecture for Interpretable Language Modeling](https://arxiv.org/abs/2603.07461)
- [Interpretable-by-Design Transformers via Architectural Stream Independence](https://arxiv.org/abs/2603.07482)
- [Stream Separation Improves Bregman Conditioning in Transformers](https://arxiv.org/abs/2603.21317)

Those papers establish what is possible when architectural constraints, particularly dual-stream decomposition and the frozen symbolic-stream (CASCADE) variant, make individual channels directly accessible to causal intervention. This repository shows what the same toolkit reveals in a standard production model without those constraints: where the methods hold, where they break, and what that implies for interpretability practice at scale.

## Overview

Applied to `gpt-oss-20b` without modification, the inspection toolkit that revealed mechanistic structure in small controlled models exposes a clear computational geography in a production 21B-parameter MoE: task resolution is depth-stratified, causally concentrated in three late layers, and amenable to targeted intervention via vocabulary-space directions — despite the model exhibiting extreme head-level redundancy that makes circuit-level analysis difficult in standard transformers.

The findings unfold across four phases of investigation:

- **Phase 1 — Where does computation happen?** Per-layer logit-lens readouts show task-specific convergence depths (L1–2 for capitalization, L5 for coreference, L17+ for induction); layer ablation confirms that L19–21 are causally critical, narrowing the interpretability target from 24 layers to 3.
- **Phase 2 — How do steering directions arise, and why is head-level intervention hard?** The model's own prediction transitions at decision layers yield self-supervised steering directions. Direct-vocabulary steering succeeds with positional specificity despite the Hydra effect making individual attention heads non-targetable.
- **Phase 3 — How selective is the steering?** Probing and causal intervention identify different channels — they dissociate. Steering selectivity is task-dependent: recency concentrates in a few channels; induction distributes across many.
- **Phase 4 — Why does linear steering work geometrically?** Standard transformers have effective rank 8 in 516 dimensions at intermediate layers — linear methods operate in 2% of the available geometry. A cosine diagnostic predicts which layers are safe for linear intervention; stream separation improves conditioning up to 22×.

Full phase narratives, figures, and the induction examples are in [FINDINGS.md](FINDINGS.md).

## What This Repository Contributes

Using `gpt-oss-20b`, this repository shows:

- computation is task-dependent in depth and causally concentrated in `L19-L21`
- honest mechanistic claims require explicit analysis-set stratification; only `9/20` benchmark cases support stable causal interpretation
- head-level interventions are ineffective because the model exhibits extreme Hydra-style redundancy (`sigma = 0.042`)
- exact vocabulary-space directions can still flip answers when applied at the right layer and position
- probe-promoted channels need not be the causally important channels, so probing alone is not a sufficient basis for intervention

## Relationship To Companion Work

| Artifact | Main contribution | Role in the overall program |
|---|---|---|
| `gpt-oss-interp` repo | Production-scale causal analysis on `gpt-oss-20b` | Tests transfer of the toolkit to a real MoE model |
| `Engineering Verifiable Modularity` (`arXiv:2603.18029`) | PLS / modularity / Hydra-breaking in controlled models | Shows how training-time constraints create inspectable internals |
| Dual-Stream Transformer (`arXiv:2603.07461`) | Channelized architecture with frozen symbolic stream | Shows how stream factorization enables surgical channel-level intervention |
| Stream Independence (`arXiv:2603.07482`) | Late-fusion architecture; coreference concentrated in L3–L4 | Shows how positional/semantic disentanglement reduces intervention collateral damage |
| `Stream Separation Improves Bregman Conditioning` (`arXiv:2603.21317`) | Geometric validity of linear interventions at intermediate layers | Explains when steering, probing, and related linear methods should be trusted |

## Results At A Glance

| Result | Measurement | Why it matters |
|---|---:|---|
| Task-dependent convergence | capitalization `L1-L2`, coreference `L5`, induction `L17+` | Different behaviors are resolved at different depths; there is no single interpretability layer |
| Late-layer causal bottleneck | `L19-L21` ablation drops accuracy `100% -> 44%` on the main analysis set | Narrows the interpretability target from 24 layers to 3 |
| Honest claim filtering | `9/20` cases retained after convergence-stability stratification | Mechanistic claims should be scoped to stable cases, not averaged over noisy ones |
| Hydra measurement at scale | head-ablation variance `sigma = 0.042` at `L20` | Explains why single-head targeting is ineffective in standard-trained production models |
| Positional steering validity | exact vocabulary directions flip answers only at decision-relevant positions | Distinguishes targeted intervention from diffuse perturbation |
| Probe-causation dissociation | induction-channel rank correlation `Spearman = -0.363` | Decodable channels are not necessarily the channels driving the computation |
| Intermediate-layer geometry | effective rank `8 / 516` in standard models; up to `22x` conditioning improvement with stream separation | Explains when linear interventions are likely to work or fail |

## Repo-Native Contributions vs. Companion-Paper Dependencies

The repository should be read as an empirical validation artifact, not as a claim that every result originated here.

### Repo-native contributions

- Production-scale causal analysis on `gpt-oss-20b`
- Task-family convergence mapping
- Late-layer causal bottleneck identification
- Analysis-set stratification for stable mechanistic claims
- Hydra measurement at production scale
- Decision-trajectory extraction on the production model

### Companion-paper-supported contributions

- Channel-level analysis in structured DST companion models
- Per-layer-supervision and modularity claims from `Engineering Verifiable Modularity`
- Intermediate-layer Bregman-geometry analysis (`arXiv:2603.21317`)

The point of grouping them together is not to blur attribution. It is to show a linked research program in which controlled-model results, geometric theory, and production-scale validation inform one another.

## Read This First

For a fast orientation:

1. [FINDINGS.md](FINDINGS.md) — full phase narratives with figures and induction examples
2. [THREAD_MAP.md](THREAD_MAP.md) — thread inventory, maturity levels, run counts
3. `threads/solid/1-convergence-logit-lens/` — task-dependent convergence depth
4. `threads/solid/2-late-layer-ablation/` — causal bottleneck identification
5. `threads/solid/3-analysis-set-filtering/` — claim scoping and honest-case selection
6. `threads/solid/5-hydra-head-redundancy/` — production-scale redundancy result
7. `threads/solid/14-bregman-geometry/` — geometric link to `arXiv:2603.21317`

Key figures:

- `figures/fig1_convergence_trajectories.png`
- `figures/fig2_late_layer_ablation.png`
- `figures/fig3_analysis_set_stratification.png`
- `figures/fig4_decision_trajectories.png`
- `figures/fig5_hydra_variance.png`
- `figures/fig_matched_pair_heatmaps.png`

## Architecture Target

**gpt-oss-20b** (`GptOssForCausalLM`): 24-layer MoE transformer.

| Component | Detail |
|---|---|
| Attention | 64 GQA query heads, 8 KV heads, `head_dim = 64` |
| Pattern | Alternating sliding (`128` token window) / full attention |
| MoE | 32 experts, top-4 routing, SwiGLU |
| Position | RoPE with YaRN scaling (`131K` context) |
| Vocab | 201,088 tokens (`o200k_harmony` BPE) |
| Quantization | MXFP4 on expert weights; attention and router in `bf16` |

Two properties matter directly for the measurement strategy:

- alternating sliding / full attention constrains which layers can attend globally
- MXFP4 quantization on expert weights limits router-level hook introspection

## Measurement Approach

Four instruments are used throughout:

- **logit-lens readouts** track the model's per-layer top-1 prediction to locate when each task is resolved
- **causal ablation** confirms which layers are critical rather than merely correlated
- **direct-vocabulary steering** applies exact unembedding directions to flip model answers
- **hook-based activation capture** records hidden states and routing decisions for offline analysis

All measurements are applied through PyTorch forward hooks without modifying model weights. The benchmark families are capitalization, coreference, induction, recency bias, and syntax agreement.

## Analysis Set Stratification

Not every benchmark case supports a clean mechanistic claim. Before presenting any downstream finding, cases are separated by convergence stability: whether the model's top-1 prediction converges reliably and holds through the final layer.

| Category | Cases | Use |
|---|---:|---|
| Correct, late-stable | `9 / 20` | Main analysis set supporting causal claims |
| Correct, unstable | `4 / 20` | Convergence too noisy for mechanistic inference |
| Incorrect | `4 / 20` | Model failure cases |
| Ambiguous | `3 / 20` | Top-1 oscillates near the decision layer |

Only `9/20` cases pass the filter. This is a methodological feature, not a bug: all mechanistic claims should be scoped to cases stable enough to support them.

## Research Threads

This project is organized as 14 research threads at three maturity levels. See `THREAD_MAP.md` for the full index with links, run counts, and figures.

### Solid

| # | Thread | Model | Problem | Contribution | Impact |
|---|---|---|---|---|---|
| 1 | `Convergence` | `gpt-oss-20b` | Where in the network does each task get resolved? | Measures task-dependent convergence depth on a production MoE | Locates computation by task type |
| 2 | `Late-layer ablation` | `gpt-oss-20b` | Which layers are causally critical? | Production-scale ablation with attention-vs-MoE decomposition | Narrows the target from 24 layers to 3 |
| 3 | `Analysis set filtering` | `gpt-oss-20b` | Which cases support honest mechanistic claims? | Explicit 4-way stratification by convergence stability | Sets defensible scope for downstream claims |
| 4 | `Decision trajectories` | `gpt-oss-20b` | Can prediction changes serve as steering directions? | Extracts self-supervised logit-space directions | Removes dependence on curated contrastive pairs |
| 5 | `Hydra / head redundancy` | `gpt-oss-20b` | Are individual heads specialized or redundant? | Measures Hydra at production scale | Explains why head-level targeting is ineffective |
| 6 | `Direct vocab steering` | `DST-baseline`, `DST-cascade` | Can exact vocabulary directions flip answers precisely? | Uses raw `W[A] - W[B]` directions | Shows position-specific intervention rather than diffuse perturbation |
| 7 | `Channel probing` | `DST-cascade` | Which channels carry the steering signal? | Per-channel causal analysis and probe comparison | Shows probing and causation identify different channels |
| 8 | `Selectivity` | `DST-cascade` | Does steering stay within the target behavior? | Cross-family concentration analysis | Shows intervention granularity is task-dependent |
| 14 | `Bregman geometry` | 2×2 factorial design (stream separation × per-layer supervision), 4 matched `45.4M`-parameter transformers | Are linear methods geometrically valid at intermediate layers? | Measures the Hessian metric at every intermediate layer; cosine diagnostic predicts safe intervention layers | Explains where linear methods are trustworthy and provides a deployable validity check |

### In Progress

| # | Thread | Model | Problem | Contribution | Impact |
|---|---|---|---|---|---|
| 9 | `Feature extraction` | `gpt-oss-20b` | Can computational modes be captured as unified feature vectors? | `6,425D` feature extraction across task families | Suggests task-dependent intrinsic dimensionality |
| 10 | `Bridge / cross-model` | `gpt-oss-20b`, `Gemma-3-1B` | Do the findings transfer beyond one model? | Screening pipeline for new-model compatibility | Early transfer infrastructure |

### Theoretical

Threads 11-13 are framework-level threads with documented theory but incomplete implementations. See `THREAD_MAP.md` for details.

## Addenda

### CASCADE feasibility

The gauge-safe pseudoinverse CASCADE target reconstructs teacher distributions with very low residual and KL divergence in the same-model setting, validating the mathematical machinery before cross-vocabulary distillation.

### MXFP4 quantization and interpretability

MXFP4 fused kernels bypass Python-level forward hooks on the router module. Router introspection is therefore opaque under quantization. This is not just an implementation detail; it is a concrete example of a broader tradeoff between compression and mechanistic inspectability.

## Quick Start

```bash
# Install
pip install -e .
pip install kernels

# Download model (~13 GB)
huggingface-cli download openai/gpt-oss-20b

# Smoke test
python scripts/run_benchmark.py --config configs/dry_run_recency.py

# Run tests
pytest tests/

# Intervention benchmark
python scripts/run_benchmark.py --config configs/head_ablation_L20.py
```

## Repository Structure

```
THREAD_MAP.md                    # Index of all 14 research threads
FINDINGS.md                      # Full phase narratives and figures
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

## Summary

`gpt-oss-interp` applies causal intervention, convergence mapping, direct-vocabulary steering, and activation analysis to `gpt-oss-20b` to determine which mechanistic interpretability methods transfer to a real production MoE model, where they break, and what that implies for trustworthy intervention at scale. The repository is most relevant in combination with the four companion papers (`arXiv:2603.18029`, `arXiv:2603.07461`, `arXiv:2603.07482`, `arXiv:2603.21317`): those works show how architectural constraints and geometric structure make internal mechanisms more tractable, while this repository shows what remains measurable in a standard model without those constraints.
