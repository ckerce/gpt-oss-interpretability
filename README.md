# gpt-oss-interp

Mechanistic interpretability toolkit for OpenAI's gpt-oss-20b. Provides causal intervention benchmarks, per-layer logit-lens readouts, and activation capture via PyTorch forward hooks.

Companion repository to `symbolic-transformer`. This repo demonstrates that inspection and intervention methods developed for interpretable-by-design architectures transfer to external production models.

Documentation is organized under `doc/`. Start with
[DOC_INDEX.md](/mnt/c/Users/ckerc/Documents/job_applications/openai-interpretability/gpt-oss-interp/DOC_INDEX.md)
for the current map.

The repo is now in the middle of a light structural refactor:

- `gpt_oss_interp/steering/` is the intended home for intervention, probing,
  readout, and steering-control code
- `gpt_oss_interp/distillation/` is the intended home for teacher-student and
  CASCADE-target code
- `gpt_oss_interp/common/` is the intended home for shared artifact schemas and
  minimal shared infrastructure

Legacy workflow code still exists in `scripts/` and older package locations.
New code should prefer the new package split.

## Architecture target

**gpt-oss-20b** (`GptOssForCausalLM`): 24-layer MoE transformer, 21B total / 3.6B active parameters per token.

| Component | Detail |
| --- | --- |
| Attention | 64 GQA query heads, 8 KV heads, head_dim=64 |
| Attention pattern | Alternating `sliding_attention` (128-token window) / `full_attention` |
| MoE | 32 experts, top-4 routing via `GptOssTopKRouter`, SwiGLU (clamp 7.0) |
| Normalization | RMSNorm (pre-LN), eps=1e-5 |
| Position | RoPE with YaRN scaling (theta=150k, factor=32) |
| Vocab | 201,088 tokens (o200k_harmony BPE) |
| Context | 131,072 tokens |
| Quantization | MXFP4 on expert weights; attention and router in bf16 |

## Results

### Intervention benchmark (legacy 320-case sweep, original 20-case benchmark)

Baseline accuracy at scale=1.0 is 0.650 across the original 20-case benchmark
(5 tasks x 4 cases each). The current task library is larger because it now
includes extra bridge-experiment candidates for capitalization, induction, and
coreference.

| Run | Accuracy | Mean Margin | Signal |
| --- | ---: | ---: | --- |
| layer_scale_L20@0 (ablate) | 0.500 | 0.359 | **Strong** — margin collapses 89% |
| layer_scale_L20@2 (amplify) | 0.600 | 2.617 | Moderate — accuracy and margin both degrade |
| early_heads_L2@0 (ablate) | 0.650 | 2.883 | Margin drops 7%; accuracy unchanged |
| mid_heads_L12@0 (ablate) | 0.650 | 3.178 | Margin *increases* — these heads add noise |

Layer 20 ablation produces the strongest causal signal: zeroing its output causes 89% margin collapse while accuracy drops to chance on several tasks.

### Logit-lens convergence

Per-layer readouts show when the model "knows" the answer:

| Prompt | Final prediction | Converges at layer | Layers total |
| --- | --- | ---: | ---: |
| "...suitcase was too" | " small" | 18 | 24 |
| "The keys to the cabinet" | " are" | 20 | 24 |
| "A7 B2 C9 D4 A7 B2 C9" | " D" (induction) | 21 | 24 |

**Recency bias**: The model transitions from noise (layers 0-10) through `<endoftext>` (layers 11-16) to "small" at layer 18 — a sharp phase transition.

**Induction**: No copying signal until layer 19; full induction pattern locks in at layer 21 with logprobs near 0 for all continuation tokens (" B", "2", " C", "9", " D"). Consistent with induction heads being a late-layer phenomenon.

**Syntax agreement**: "are" (correct plural verb) first appears at layer 20 — agreement across the prepositional-phrase attractor is resolved very late.

### MXFP4 interpretability finding

The MXFP4 fused kernel for MoE expert computation bypasses Python-level forward hooks on the router module and does not populate `output_router_logits`. This means:

- **Router introspection is opaque** under MXFP4 quantization
- Expert masking operates at the MLP output level (proportional scaling) rather than gate-level suppression
- Full router capture requires a non-quantized (bf16) checkpoint

This is a concrete example of the **quantization–interpretability tradeoff**: model compression techniques that fuse operations for performance simultaneously reduce the surface area available for mechanistic inspection.

## Quick start

```bash
# Install (requires torch, transformers, accelerate, safetensors, kernels)
pip install -e .
pip install kernels   # required for MXFP4 support

# Download model weights (~13 GB)
huggingface-cli download openai/gpt-oss-20b

# Smoke test (no GPU needed)
python scripts/run_benchmark.py --config configs/dry_run_recency.py

# Inspect model structure
python scripts/inspect_model.py --model openai/gpt-oss-20b

# Run intervention benchmark on real model
python scripts/run_benchmark.py --config configs/gpt_oss_20b_template.py

# Run logit-lens analysis
python scripts/run_logit_lens.py \
    --model openai/gpt-oss-20b \
    --prompt "The trophy would not fit in the suitcase because the suitcase was too" \
    --output runs/logit_lens_demo/
```

## Layout

```
gpt_oss_interp/
├── config.py                    # Dataclasses: tasks, interventions, benchmarks
├── common/                      # Shared artifact schemas and shared run types
├── backends/
│   ├── base.py                  # Backend contract (score, intervene, clear)
│   ├── dry_run.py               # Synthetic backend for smoke testing
│   └── transformers_gpt_oss.py  # Real gpt-oss-20b backend with hooks
├── capture/
│   ├── activation_cache.py      # Hidden-state capture via forward hooks
│   └── router_capture.py        # MoE routing decision capture
├── features/
│   ├── extractor.py             # Extended Tier-2 feature extraction for MoE
│   └── geometry.py              # Metric-space analysis of feature point clouds
├── harmony/
│   └── prompting.py             # Harmony chat-template formatting
├── readouts/
│   └── logit_lens.py            # Per-layer token prediction readouts
├── benchmarks/
│   ├── tasks.py                 # Task library (5 families; expanded bridge-candidate pool)
│   └── runner.py                # Benchmark orchestration and scoring
├── interventions/
│   └── specs.py                 # Intervention sweep expansion
├── steering/                    # New home for steering / probing / readouts
├── distillation/                # New home for teacher-student workflows
└── reports/
    └── writers.py               # CSV, JSON, Markdown output

scripts/
├── run_benchmark.py             # CLI: run full intervention benchmark
├── run_feature_extraction.py    # CLI: computational mode feature extraction
├── inspect_model.py             # CLI: dump model structure for hook planning
├── run_logit_lens.py            # CLI: per-layer prediction analysis
└── capture_routing.py           # CLI: MoE expert selection analysis

configs/
├── dry_run_recency.py           # Smoke-test config (no model needed)
└── gpt_oss_20b_template.py      # Full sweep on real model
```

## Refactor Note

The `steering/`, `distillation/`, and `common/` packages are currently a
scaffolded target structure.

Short-term rules:

- write new per-channel probing logic in `gpt_oss_interp/steering/probing.py`
- write new steering readout/control logic in `gpt_oss_interp/steering/`
- write new CASCADE distillation logic in `gpt_oss_interp/distillation/`
- keep shared JSON artifact schemas in `gpt_oss_interp/common/artifacts.py`
- avoid adding major new first-pass workflow logic directly to `scripts/`

The old locations are still valid during migration, but they are no longer the
intended place for new work.

## Intervention types

| Kind | Target | Mechanism | Status |
| --- | --- | --- | --- |
| `HEAD_MASK` | Attention heads | Scale specific GQA head outputs | Working |
| `EXPERT_MASK` | MoE experts | Proportional MLP output scaling | Working (MLP-level) |
| `LAYER_SCALE` | Transformer blocks | Scale entire layer output | Working |
| `TEMPERATURE_SCALE` | Attention | Scale attention output | Working |

## Task families

Current built-in task counts after bridge-pool expansion:

| Task | Cases | Behavior | Baseline |
| --- | ---: | --- | --- |
| recency_bias | 4 | Attributive adjective resolution | Partial |
| capitalization | 8 | Proper noun formatting | Strong |
| induction | 10 | Pattern copying | Strong |
| coreference | 10 | Pronoun resolution | Strong |
| syntax_agreement | 4 | Subject-verb agreement across attractors | Partial |

## Feature extraction and geometric analysis

The `features/` module extends the 163-dimensional Tier-2 feature system from the NeurIPS 2026 activation clustering paper to MoE architectures. For gpt-oss-20b, the feature space expands to ~7,200 dimensions:

| Component | Dims | What it captures |
| --- | ---: | --- |
| A. Trajectory | 71 | Per-layer probability, margin, confidence drops |
| B. Stability | 2 | Convergence layer k*, max consecutive correct κ |
| C. Head Activation | 3,072 | Peak attention at stability + final layers (64 heads × 24 layers × 2) |
| D. Head Entropy | 3,072 | Attention sharpness at stability + final layers |
| E. Expert Routing | 160 | Weighted routing vectors at stability + final layers, sorted top-4 weights |
| F. Routing Entropy | 24 | Per-layer routing diversity |
| G. Attention Scale | 24 | Sliding-window vs. full-attention local fraction |

The feature map φ: Tokens-in-Context → ℝ^D simultaneously structures both the data (tokens processed similarly are "close") and the model (components that co-activate are "similar"). This dual metric enables architecture-agnostic comparison of computational modes across models of different scale and training objective.

See `GEOMETRIC_FRAMEWORK.md` for the mathematical exposition and `RESEARCH_CONNECTIONS.md` for experimental plans.

```bash
# Extract features and run geometric analysis
python scripts/run_feature_extraction.py \
    --model openai/gpt-oss-20b \
    --prompt "The trophy would not fit in the suitcase because the suitcase was too" \
    --output runs/features_demo/
```

## Design principles

- **Backend-agnostic benchmarks**: benchmark code never sees model internals
- **Config-driven experiments**: Python config files, not CLI flags
- **Correctness before optimization**: reference implementations first
- **Hook-based inspection**: PyTorch forward hooks for capture and intervention
- **Document what doesn't work**: MXFP4 limitations are findings, not failures
