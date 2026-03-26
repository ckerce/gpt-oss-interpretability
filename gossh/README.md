# gossh — gpt-oss inspectability harness

An installable mechanistic interpretability toolkit for `gpt-oss-20b` and `gpt-oss-120b`.

```bash
pip install -e .
```

---

## Why gossh exists

`gpt-oss-20b` has three properties that make standard interpretability tooling break:

1. **MXFP4 fused kernels bypass Python hooks on the router gate.** The fused kernel executes token-to-expert routing and expert dispatch as a single native operation. `register_forward_hook` on the gate module never fires. `output_router_logits=True` returns `None`. Routing is opaque.

2. **Alternating sliding / full attention** (128-token window on even layers, full on odd) means locality is structural, not learned — but naive activation capture does not distinguish the two.

3. **64 GQA query heads with 8 KV heads** means per-head intervention does not decompose cleanly: zeroing one query head redistributes attention across all heads sharing the same KV projection, making the Hydra effect worse than in standard MHA.

gossh addresses each of these directly and exposes four inspection instruments as a coherent API:

| Instrument | Module | What it measures |
|---|---|---|
| Logit-lens readouts | `gossh.readouts` | Per-layer top-1 prediction → task convergence depth |
| Causal ablation | `gossh.backends` | Layer/head/expert scaling → causal bottleneck identification |
| MoE router sidecar | `gossh.sidecar` | MXFP4-safe routing capture via bf16 gate clone subprocess |
| Bregman geometry | `gossh.steering` | Softmax Hessian geometry → validity of linear intervention |

---

## Installation

```bash
# Core package (no GPU required for tests and dry-run)
pip install -e .

# MXFP4 kernel support (optional; only needed to run gpt-oss-20b with quantization)
pip install kernels
```

Python ≥ 3.10, PyTorch ≥ 2.0.

---

## The MXFP4 routing opacity problem and the sidecar solution

Under MXFP4 quantization, the MoE router is invisible to Python. The sidecar works around this without modifying the model:

```
┌─────────────────────────────────────────────────┐
│  Main process (GPU)                             │
│                                                 │
│  InputCapture                                   │
│  register_forward_pre_hook(mlp_modules)         │
│  ↓  fires BEFORE the fused kernel               │
│  hidden_states captured at MoE input            │
│                                                 │
│  model(input_ids)  ← fused MXFP4 kernel runs   │
│  (router gate never fires a Python hook)        │
└────────────────────┬────────────────────────────┘
                     │  hidden_states (IPC, Unix socket)
                     ▼
┌─────────────────────────────────────────────────┐
│  Sidecar subprocess (CPU)                       │
│                                                 │
│  RouterSidecarModel                             │
│  bf16 router weight clone (never quantized)     │
│  logits = gate_weight @ hidden_states           │
│  experts = topk(softmax(logits), k=top_k)       │
│                                                 │
│  Returns: RouterDecision per layer              │
└─────────────────────────────────────────────────┘
```

The router weights stay in bf16 even in MXFP4 checkpoints because routing is a control-flow decision that cannot tolerate low precision. `RouterWeightExtractor` retrieves them directly from `named_parameters()` without triggering dequantization of expert weights.

```python
from gossh.backends.gpt_oss import GPTOSSTransformersBackend
from gossh.sidecar import MoeSidecar, RouterWeightExtractor

backend = GPTOSSTransformersBackend("openai/gpt-oss-20b")

weights = RouterWeightExtractor().extract(backend.model, backend.structure)
with MoeSidecar(weights, top_k=backend._arch.top_k) as sidecar:
    backend.attach_sidecar(sidecar)
    decisions = backend.capture_routing("The trophy didn't fit in the suitcase")
    backend.detach_sidecar()

for d in decisions:
    print(f"Layer {d.layer_idx:2d}: top experts = {d.selected_experts[0].tolist()}")
```

---

## Four interesting properties of gpt-oss-20b

### 1. Task resolution is depth-stratified

Different task families converge to their final answer at very different layers:

| Task family | Convergence | Mechanism |
|---|---|---|
| Capitalization | L1–2 | Surface token statistics — resolved in embedding + first block |
| Coreference | ~L5 | Entity binding — mid-network association |
| Syntax agreement | L8–12 | Structural composition |
| Induction | L17+ | In-context retrieval — requires deep propagation |

The logit-lens shows this directly: project each layer's hidden state through the final norm + unembedding, read off the top-1 token.

```python
result = backend.run_logit_lens("The trophy didn't fit in the suitcase because it was too small.", top_k=3)
for pos in [-1]:   # last token position
    print(f"Convergence layer: {result.convergence_layer(pos)}")
    for pred in result.layer_slice(0):   # layer 0
        print(f"  L0 top-1: {pred.top_tokens[0]!r}  logp={pred.top_logprobs[0]:.2f}")
```

See: `examples/howto_logit_lens.py`

---

### 2. Layers 19–21 are the causal bottleneck

Ablating layers L19–21 (scaling their residual contribution toward zero) drops task accuracy from ~100% to ~44% on the main analysis set. Layers 0–18 are largely redundant for final answer selection, despite being essential for building the representations that late layers read.

This is measured via `LAYER_SCALE` interventions with `preserve_residual=True`, which scales only the block's learned delta without zeroing the bypass:

```python
from gossh.config import InterventionSpec, InterventionKind, InterventionTarget, TargetUnit

spec = InterventionSpec(
    name="ablate_L19_21",
    kind=InterventionKind.LAYER_SCALE,
    target=InterventionTarget(
        unit=TargetUnit.LAYER,
        layer_indices=(19, 20, 21),
    ),
    scales=(0.0,),      # scale=0 zeros the block delta, preserving residual
)
backend.apply_intervention(spec, scale=0.0)
score_ablated = backend.score_case(case)
backend.clear_interventions()
```

See: `examples/howto_layer_ablation.py`

---

### 3. Head ablation is ineffective (Hydra at production scale)

Ablating individual attention heads at L20 produces near-zero variance in task accuracy across all 64 heads (σ ≈ 0.042). The model compensates immediately — the remaining heads redistribute the load. This is the Hydra effect at production scale.

For comparison: models trained with per-layer supervision (PLS) show σ ≈ 0.47; standard untrained controls show σ ≈ 0.08. `gpt-oss-20b` sits at the floor.

```python
import statistics

margins = []
for head_idx in range(64):
    spec = InterventionSpec(
        name=f"head_L20_H{head_idx}",
        kind=InterventionKind.HEAD_MASK,
        target=InterventionTarget(
            unit=TargetUnit.HEAD,
            layer_indices=(20,),
            head_indices=(head_idx,),
        ),
        scales=(0.0,),
    )
    backend.apply_intervention(spec, scale=0.0)
    score = backend.score_case(case)
    backend.clear_interventions()
    a, b = score.choice_logprobs["A"], score.choice_logprobs["B"]
    margins.append(a - b)

print(f"Hydra σ: {statistics.stdev(margins):.4f}")   # expect ~0.042
```

See: `examples/howto_layer_ablation.py`

---

### 4. Linear steering only works where the Hessian says it should

The softmax Hessian `H(h) = Cov[w_y | h]` in hidden-state coordinates measures how curved the loss landscape is at each layer. A low effective rank means the model's probability mass is concentrated in a low-dimensional subspace — linear interventions in other directions mostly miss.

`gpt-oss-20b` effective rank at intermediate layers (L8–16): ~8 out of 4096 dimensions (0.2%). At decision layers (L19–21): rises to ~40–60. The cosine between `W[A] - W[B]` and `H(W[A] - W[B])` predicts whether a vocabulary-space steering direction will work at a given layer.

```python
from gossh.capture import ActivationCache
from gossh.steering import (
    analyze_bregman_state, unembedding_direction, format_bregman_summary,
    summarize_bregman_metrics,
)
import torch
from collections import defaultdict

lm_head_weight = backend.structure.lm_head.weight.detach().float().cpu()
direction = unembedding_direction(lm_head_weight, token_a_id=12345, token_b_id=67890)

cache = ActivationCache(detach=True, to_cpu=True)
handles = cache.register(backend.model, backend.structure.block_names)
with torch.no_grad():
    backend.model(input_ids)
for h in handles:
    h.remove()

norm_device = next(backend.structure.final_norm.parameters()).device
metrics_by_layer = defaultdict(list)
for layer_idx, block_name in enumerate(backend.structure.block_names):
    record = cache.last(block_name)
    if record is None:
        continue
    normed = backend.structure.final_norm(record.tensor.to(norm_device))[0].cpu()
    m = analyze_bregman_state(normed[-1], lm_head_weight, direction=direction)
    metrics_by_layer[layer_idx].append(m)

summaries = summarize_bregman_metrics(metrics_by_layer, top_k_vocab=2048)
print(format_bregman_summary(summaries))
```

See: `examples/howto_bregman_conditioning.py`

---

## API surface

### `gossh.backends`

| Class | Description |
|---|---|
| `GPTOSSTransformersBackend` | Full HuggingFace backend for gpt-oss-20b/120b |
| `DryRunBackend` | No-GPU stub for CI and config validation |
| `ModelStructure` | Block/attn/mlp/gate name discovery from a loaded model |

Key methods on `GPTOSSTransformersBackend`:

```python
backend.score_case(case)                   # → BackendScore
backend.score_case_by_layer(case)          # → {layer_idx: {label: logp}}
backend.run_logit_lens(prompt, top_k=5)    # → LogitLensResult
backend.capture_activations(prompt)        # → list[ActivationRecord]
backend.capture_routing(prompt)            # → list[RouterDecision]  (4-path priority)
backend.apply_intervention(spec, scale)    # registers a hook
backend.clear_interventions()
backend.attach_sidecar(sidecar)            # enables path-0 MXFP4 routing
backend.detach_sidecar()
```

`capture_routing` path priority:
0. **Sidecar** (attach first — MXFP4-safe)
1. `output_router_logits=True` (non-quantized models)
2. Gate forward hooks (non-quantized, hook-visible gate modules)
3. Unavailable — emits informative message

### `gossh.sidecar`

```python
MoeSidecar(router_weights, top_k)          # context manager or manual start/stop
MoeSidecar.from_model(model, structure, arch_spec)
sidecar.route(layer_hidden)                # → list[RouterDecision]
sidecar.ping()                             # → bool
RouterWeightExtractor().extract(model, structure)  # → {layer_idx: Tensor}
RouterSidecarModel(weights, top_k)         # local (in-process) routing
run_full_validation(sidecar, ref, hidden)  # 3-layer validation report
```

### `gossh.readouts`

```python
run_logit_lens(model, input_ids, tokenizer, final_norm, lm_head, blocks, top_k)
# → LogitLensResult

result.convergence_layer(position)         # earliest layer matching final prediction
result.target_convergence_layer(position)  # earliest layer where tracked token is rank-0
result.layer_slice(layer_idx)
result.position_slice(position)
format_logit_lens_table(result)
```

### `gossh.features`

```python
MoEFeatureExtractor(config).extract(
    layer_logits,       # [L, T, V]
    layer_attentions,   # [L, H, T, T] or None
    expert_routing,     # [L, T, E] or None (MXFP4 fallback: zeros)
)  # → FeatureResult ([T, D] feature vectors)

extract_features_from_backend(backend, prompt)   # end-to-end convenience

analyze_geometry(features, processing_depth)     # → GeometricSummary
```

### `gossh.steering`

```python
analyze_bregman_state(hidden, lm_head_weight, direction=None)  # → BregmanMetrics
softmax_hessian_from_hidden(hidden, weight, top_k_vocab=2048)  # → (H, mass_covered)
unembedding_direction(weight, token_a_id, token_b_id)          # → W[a] - W[b]
summarize_bregman_metrics(metrics_by_layer, top_k_vocab)       # → list[LayerBregmanSummary]
format_bregman_summary(summaries)                               # → markdown table
```

### `gossh.config`

```python
PromptCase(case_id, prompt, choices, expected_label, task_name)
InterventionSpec(name, kind, target, scales)
BenchmarkConfig(backend_kind, tasks, interventions, ...)
BenchmarkConfig.from_yaml(path)
BenchmarkConfig.to_yaml(path)
```

### `gossh.benchmarks`

```python
BenchmarkRunner(config).run()    # returns list[RunResult], writes CSV/JSON/Markdown
gossh-benchmark --config configs/dry_run_recency.py   # CLI entry point
```

---

## Tests

```bash
pytest tests/gossh/          # 90 tests, no GPU required (~7s system Python / ~45s venv)
```

Test coverage:
- Config serialization and YAML round-trip
- DryRunBackend scoring and intervention behavior
- All four hook factories (head_mask, expert_mask, layer_scale, temperature_scale)
- BenchmarkRunner end-to-end with DryRunBackend
- ModelArchSpec registry
- Full sidecar subsystem: protocol, routing, lifecycle, validation (subprocess tests)
- Backend sidecar integration (attach/detach/route)

---

## Design principles

- **baukit**: hook factories are pure functions returning closures; no global state
- **pyvene**: config types are serializable; YAML ↔ dataclass round-trip
- **No hook spaghetti**: every hook is registered explicitly and removed in `finally` blocks
- **Lazy GPU imports**: `import gossh` does not touch torch or the GPU; backends are loaded on demand
- **MXFP4 honesty**: routing capture failure under MXFP4 is surfaced explicitly, not silently swallowed
