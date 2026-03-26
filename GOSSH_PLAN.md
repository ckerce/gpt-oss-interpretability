# GOSSH Implementation Plan
## gpt-oss-inspectability-harness

A pip-installable, HuggingFace-distributable interpretability harness for
`gpt-oss-20b` and `gpt-oss-120b`, with a new MoE sidecar process that resolves
the MXFP4 router opacity problem.

---

## Background and Motivation

The existing `gpt-oss-interp` package is a research repo with solid interpretability
infrastructure: hook-based activation capture, `RouterCapture`, `ActivationCache`,
the full intervention framework (HEAD_MASK, EXPERT_MASK, LAYER_SCALE,
TEMPERATURE_SCALE), logit-lens readouts, and feature extraction with MoE-specific
components. What it lacks is:

1. **MoE router transparency.** Under MXFP4 quantization, `output_router_logits=True`
   returns `None` and Python-level forward hooks on the gate module fire after the
   fused kernel has already executed. The current workaround (`_expert_output_scale_hook`)
   scales expert outputs by a fraction — it does not provide actual token-to-expert
   routing decisions.

2. **Public distribution.** No HuggingFace Hub integration, no stable public API
   surface, no example notebooks, no model registry for multi-model support.

3. **120b support.** Architecture constants are hardcoded for 20b only.

GOSSH addresses all three.

---

## What Already Exists (carry-over inventory)

The following modules carry over with minimal changes:

| Module | Status |
|---|---|
| `capture/activation_cache.py` | Clean carry-over |
| `capture/router_capture.py` | Carry over + extend with sidecar path |
| `readouts/logit_lens.py` | Clean carry-over |
| `features/extractor.py` | Clean carry-over (expert_routing features already present) |
| `features/geometry.py` | Clean carry-over |
| `steering/bregman.py` | Clean carry-over |
| `steering/selectivity.py` | Clean carry-over |
| `interventions/specs.py` | Clean carry-over |
| `benchmarks/tasks.py` | Clean carry-over |
| `benchmarks/pools.py` | Clean carry-over |
| `common/artifacts.py` | Carry over, bump SCHEMA_VERSION |
| `common/io.py` | Clean carry-over |
| `config.py` | Carry over, add ModelKind enum |

Modules requiring rework:

| Module | Changes Required |
|---|---|
| `backends/transformers_gpt_oss.py` | Extract hook factories; replace hardcoded arch constants with registry; add sidecar as fourth routing-capture path |
| `benchmarks/runner.py` | Add YAML/dict config path alongside Python file import |
| `steering/probing.py` | Factor out DST-specific imports; keep generic linear probe fitting |
| `steering/causal.py` | Factor out DST-specific imports; keep generic direction ablation |

---

## Package Structure

The public package is renamed `gossh`. Internal research scaffolding stays in
`gpt-oss-interp/` and is not distributed.

```
gossh/
├── __init__.py                  # Stable public API (Tier 1: types, Tier 2: entry points)
├── py.typed                     # PEP 561 marker
├── config.py                    # Carry over; add ModelKind enum
├── model_registry.py            # NEW: ModelArchSpec, list_supported_models()
├── backends/
│   ├── base.py                  # Carry over
│   ├── dry_run.py               # Carry over
│   ├── gpt_oss.py               # Refactored unified 20b/120b backend
│   └── structure.py             # NEW: extracted ModelStructure discovery
├── capture/
│   ├── activation_cache.py      # Carry over
│   ├── input_cache.py           # NEW: InputCapture using pre-hooks (feeds sidecar)
│   └── router_capture.py        # Extend: sidecar as fourth path
├── interventions/
│   ├── specs.py                 # Carry over
│   └── hooks.py                 # NEW: extracted hook factories
├── readouts/
│   └── logit_lens.py            # Carry over
├── steering/
│   ├── probing.py               # Refactored: generic only
│   ├── causal.py                # Refactored: generic only
│   ├── selectivity.py           # Carry over
│   └── bregman.py               # Carry over
├── features/
│   ├── extractor.py             # Carry over
│   └── geometry.py              # Carry over
├── benchmarks/
│   ├── runner.py                # Refactored: YAML/dict config support
│   ├── tasks.py                 # Carry over
│   └── pools.py                 # Carry over
├── sidecar/                     # NEW: MoE sidecar subsystem
│   ├── __init__.py
│   ├── process.py               # MoESidecar context manager
│   ├── protocol.py              # IPC wire format (msgpack over Unix socket)
│   ├── worker.py                # Worker process entry point
│   └── dequant.py               # Router weight extraction + RouterSidecarModel
└── common/
    ├── artifacts.py             # Carry over; bump SCHEMA_VERSION to 2.0.0
    └── io.py                    # Carry over
```

### Public API Surface (`gossh/__init__.py`)

```python
# Tier 1: Types (stable, semver-guaranteed)
from gossh.config import (
    PromptCase, PromptTask, InterventionSpec, InterventionTarget,
    InterventionKind, TargetUnit, BenchmarkConfig, ModelKind,
)
from gossh.model_registry import list_supported_models, get_arch_spec

# Tier 2: Functional entry points
from gossh.backends.gpt_oss import GPTOSSBackend
from gossh.benchmarks.runner import BenchmarkRunner
from gossh.readouts.logit_lens import run_logit_lens, LogitLensResult
from gossh.sidecar import MoESidecar   # conditional on platform
```

---

## The MoE Sidecar

### The Problem

MXFP4 fused kernels execute router + expert dispatch as a single native operation.
Python forward hooks on the gate module fire after the kernel has completed —
`gate_logits` in the hook output are not populated. `output_router_logits=True`
also returns `None`. The current `_expert_output_scale_hook` approximation is:

```
fraction = len(expert_indices) / num_experts
```

This is wrong for any analysis requiring actual token-to-expert assignment.

### Key Insight: Router Weights Are bf16

Only the *expert weights* are MXFP4. The router/gate is a standard bf16 linear
layer. Its parameters are accessible via `model.named_parameters()` without
dequantization. The sidecar runs the router projection in Python using these
weights, on CPU, and returns exact routing decisions.

### Sidecar Architecture

```
Main process                           Sidecar worker process
──────────────────────────────         ──────────────────────────────
GPTOSSBackend.forward()                RouterSidecarModel
  │                                      24 x nn.Linear(4096, 32)
  ├─ InputCapture pre-hooks              loaded from gate.weight tensors
  │    capture residual stream           running on CPU / secondary GPU
  │    at each MoE layer input
  │
  ├─ serialize layer inputs
  │    (shared memory or msgpack)
  │
  └─→ [Unix socket] ──────────────→  worker receives layer inputs
                                       runs linear projection per layer
                                       computes softmax + topk
                                       returns RouterDecision list
       [Unix socket] ←──────────────  worker sends RouterDecision objects
  │
  └─ RouterDecision objects available
     for all downstream analysis
```

### Implementation: `sidecar/dequant.py`

```python
class RouterWeightExtractor:
    """Extract bf16 gate/router weights from a quantized checkpoint."""

    def extract(self, model, structure: ModelStructure) -> dict[int, Tensor]:
        # Walk model.named_parameters()
        # Match gate patterns from structure.gate_patterns
        # Return {layer_idx: weight_tensor} — already bf16, no dequant needed

class RouterSidecarModel(nn.Module):
    """Minimal model: one nn.Linear per layer for the gate projection."""

    def route(self, layer_idx: int, hidden_state: Tensor) -> RouterDecision:
        logits = self.gates[layer_idx](hidden_state)   # [seq, n_experts]
        weights, indices = logits.softmax(-1).topk(self.top_k, dim=-1)
        return RouterDecision(
            layer_idx=layer_idx,
            selected_experts=indices.cpu().numpy(),
            expert_weights=weights.cpu().numpy(),
            gate_logits=logits.cpu().numpy(),
            token_count=hidden_state.shape[0],
        )
```

### Implementation: `capture/input_cache.py`

```python
class InputCapture:
    """Capture residual stream inputs to MoE layers using pre-hooks."""

    def register(self, model, structure: ModelStructure) -> None:
        for layer_idx, module in structure.moe_modules():
            module.register_forward_pre_hook(
                self._make_hook(layer_idx)
            )

    def _make_hook(self, layer_idx: int):
        def hook(module, args):
            self._cache[layer_idx] = args[0].detach()
        return hook
```

### IPC Protocol (`sidecar/protocol.py`)

Each forward pass:
1. Main process captures `{layer_idx: hidden_state}` via `InputCapture`
2. Serializes to msgpack (float32, shape metadata) or writes to shared memory
3. Sends over Unix domain socket
4. Worker receives, runs `RouterSidecarModel.route()` per layer
5. Returns list of `RouterDecision` objects

Memory estimate for gpt-oss-20b (24 layers, seq_len=128, hidden_dim=4096):
- 24 × 128 × 4096 × 4 bytes = **~48MB per forward pass**
- With `multiprocessing.shared_memory`: near-zero copy overhead

Router weight memory (24 × 4096 × 32 × 2 bytes bf16):
- **~6MB total** — negligible

### Integration into `capture/router_capture.py`

The existing `RouterCapture` has three paths:
1. `output_router_logits=True` (works for non-MXFP4)
2. Forward hook on gate module (fires but gate_logits empty under MXFP4)
3. Fraction-scaling approximation (current fallback)

GOSSH adds a fourth path, tried first for MXFP4 models:

```python
def capture_routing(self, ...):
    if self._sidecar is not None:
        return self._sidecar_routing(...)   # Path 0: sidecar (MXFP4-safe)
    elif self._try_router_logits():
        return self._native_routing(...)    # Path 1: native
    elif self._try_hook():
        return self._hook_routing(...)      # Path 2: hook
    else:
        return self._fraction_routing(...)  # Path 3: approximation
```

### Validation Strategy

Validation runs at three layers, forming a `gossh.sidecar.validation` module.
All three are required CI tests before Phase 2 ships.

**Layer 1 — Numerical tap-off comparison.**
Load identical prompts through both a bf16 model (`dtype=torch.bfloat16`,
`output_router_logits=True`) and an MXFP4 model with the sidecar active.
Compare routing decisions at every layer and report:

- Expert selection exact-match % per layer (selected_experts index agreement)
- Expert weight MAE per layer
- Gate logit KL divergence per layer
- Overall mismatch rate across all layers and tokens

Any layer exceeding a configurable mismatch threshold (default: 5% of tokens
routed differently) is flagged as a validation failure.

**Layer 2 — Semantic vocabulary validation.**
Use the existing `run_logit_lens()` at decision layers (layers where top-1
prediction changes) for both bf16 and MXFP4 runs. Compare:

- Top-1 vocabulary prediction agreement % per layer
- Top-5 vocabulary prediction agreement % per layer
- Convergence layer agreement: does `LogitLensResult.target_convergence_layer()`
  return the same layer for both runs?

This validates semantic functionality independently of numerical precision:
even if gate logits drift slightly, the model's semantic predictions should
remain stable if routing is robust.

**Layer 3 — Routing sensitivity analysis.**
Cross-reference layers where numerical mismatch exceeds threshold against
semantic agreement:

- *Robust routing*: numerical mismatch present but top-1 vocabulary agrees.
  Sidecar is valid for interpretability use; note quantization tolerance.
- *Sensitive routing*: numerical mismatch correlates with vocabulary divergence.
  Flag these layers as unreliable for sidecar-based analysis; fall back to
  fraction-scaling approximation with a warning.

Report output: `SidecarValidationReport` dataclass with per-layer statistics,
overall pass/fail, and a list of flagged layers.

**Rotation matrix caveat.**
MXFP4 online rotation applies a rotation matrix to activations before
quantization. If the `kernels` package folds this rotation into the gate
projection, the sidecar's bf16 router clone will diverge unless the same
rotation is applied. Layer 1 validation will detect this as systematic
mismatch. Mitigation: inspect whether `gate.weight` in `model.named_parameters()`
reflects the post-rotation weights (stored) or pre-rotation weights (requiring
runtime rotation). If the latter, extract the rotation matrix and apply it in
`RouterSidecarModel.route()` before the linear projection.

---

## Model Registry

```python
@dataclass(frozen=True)
class ModelArchSpec:
    model_id: str
    num_layers: int
    hidden_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    num_experts: int
    top_k: int
    quantization: str        # "mxfp4" | "bf16" | "none"
    vocab_size: int
    max_seq_len: int

    @classmethod
    def from_hf_config(cls, model_id: str) -> "ModelArchSpec":
        """Read architecture constants from HuggingFace config.json at runtime."""
        ...

REGISTRY = {
    "openai/gpt-oss-20b": ModelArchSpec(
        model_id="openai/gpt-oss-20b",
        num_layers=24,
        hidden_dim=4096,
        num_heads=64,
        num_kv_heads=8,
        head_dim=64,
        num_experts=32,
        top_k=2,
        quantization="mxfp4",
        vocab_size=200019,
        max_seq_len=8192,
    ),
    "openai/gpt-oss-120b": ModelArchSpec(
        model_id="openai/gpt-oss-120b",
        # To be filled from config.json at first use
        quantization="mxfp4",
    ),
}
```

---

## Phased Roadmap

### Phase 1 — Stable Installable Core
**Goal:** `pip install gossh` works; load gpt-oss-20b, run benchmark, get logit-lens.
**No sidecar yet.**

- [ ] Rename package to `gossh`, update `pyproject.toml` (name, version 1.0.0a1)
- [ ] Implement `model_registry.py` with `ModelArchSpec` for gpt-oss-20b
- [ ] Extract `ModelStructure` → `backends/structure.py`
- [ ] Extract hook factories → `interventions/hooks.py`
- [ ] Refactor `backends/transformers_gpt_oss.py` to use registry for arch constants
- [ ] Add YAML/dict config support to `benchmarks/runner.py`
- [ ] Write `gossh/__init__.py` Tier 1 + Tier 2 exports
- [ ] Port existing test suite to new package structure
- [ ] Validate: `pip install -e . && python -c "from gossh import BenchmarkRunner"`
- [ ] Write hardware requirements doc (VRAM for 20b/120b, CPU fallback)

**Deliverable:** External user can `huggingface-cli download openai/gpt-oss-20b`
then run the benchmark in 5 lines of Python.

### Phase 2 — MoE Sidecar
**Goal:** `RouterDecision` objects populated for MXFP4 models.

- [ ] Implement `capture/input_cache.py` (`InputCapture` using pre-hooks)
- [ ] Implement `sidecar/dequant.py` (`RouterWeightExtractor`, `RouterSidecarModel`)
- [ ] Implement `sidecar/protocol.py` (msgpack + shared memory IPC)
- [ ] Implement `sidecar/worker.py` (worker entry point)
- [ ] Implement `sidecar/process.py` (`MoESidecar` context manager)
- [ ] Integrate sidecar as Path 0 in `capture/router_capture.py`
- [ ] Repair `EXPERT_MASK` intervention to use sidecar-supplied indices
- [ ] Implement `sidecar/validation.py`: `SidecarValidationReport`, three-layer validation suite
- [ ] Layer 1: numerical tap-off comparison (expert selection match %, weight MAE, gate logit KL)
- [ ] Layer 2: semantic vocabulary validation via `run_logit_lens()` (top-1/top-5 agreement, convergence layer agreement)
- [ ] Layer 3: routing sensitivity cross-reference (robust vs. sensitive routing classification per layer)
- [ ] Rotation matrix inspection: determine whether `gate.weight` is pre- or post-rotation; apply correction if needed
- [ ] Document sidecar: what it does, memory requirements, launch options, validation report interpretation

**Deliverable:** `backend.capture_routing(prompt)` returns populated
`RouterDecision` for MXFP4 gpt-oss-20b. Expert specialization analysis
across task families is a new research thread.

### Phase 3 — HuggingFace Distribution
**Goal:** Package is discoverable and usable without the research repo.

- [ ] Example notebooks (load model, benchmark, logit-lens, MoE routing, steering)
- [ ] HuggingFace model card linking to PyPI package + hardware requirements
- [ ] GitHub Actions CI: DryRunBackend tests (no model download), GPU integration tests
- [ ] Gradio Space demo (DryRunBackend with pre-computed results; links to Colab for real runs)
- [ ] Tag `v1.0.0` on PyPI

**Deliverable:** Discoverable on HuggingFace; Colab notebook runs end-to-end.

### Phase 4 — gpt-oss-120b and Cross-Model Bridge
**Goal:** 120b backend works; cross-model screening is first-class.

- [ ] Fill 120b `ModelArchSpec` from `config.json` (requires checkpoint access)
- [ ] Validate sidecar with 120b router weights
- [ ] Port thread-10 cross-model bridge into `gossh.bridge`
- [ ] Abstract `harmony/prompting.py` into `ChatTemplate` interface for non-gpt-oss models

---

## Risks and Open Questions

**Risk 1: Router weight isolation assumption.**
Plan assumes `gate.weight` is bf16 and separate from MXFP4 expert weights.
README states this explicitly. Validate before trusting sidecar outputs by
comparing against a bf16 model load.

**Risk 2: MXFP4 kernel availability for external users.**
The `kernels` package is not standard PyPI. Mark as optional:
`pip install gossh[mxfp4]`. `DryRunBackend` must be fully functional
as a no-download fallback.

**Risk 3: gpt-oss-120b architecture unknown.**
Use `ModelArchSpec.from_hf_config()` to read constants at runtime from
HuggingFace `config.json` rather than hardcoding.

**Risk 4: Sidecar IPC latency at scale.**
For 120b (80+ layers, longer sequences), 48MB/forward-pass becomes larger.
Use `multiprocessing.shared_memory` for zero-copy transfer; support
batched requests in the protocol.

**Risk 5: DST coupling in steering modules.**
`steering/probing.py` and `steering/causal.py` import DST-specific functions.
Factor generic analysis primitives (linear probe fitting, direction ablation)
into `gossh.analysis` before Phase 1 ships; gate DST code behind `gossh._dst`.

**Open Questions:**
1. Is a non-quantized gpt-oss-20b checkpoint publicly available? (Needed for sidecar validation)
2. Does `output_router_logits=True` work for a bf16 load of gpt-oss-20b?
3. What GPU tier does HuggingFace Spaces require for gpt-oss-20b? (~13GB VRAM for MXFP4)
4. Does gpt-oss-120b use model parallelism / tensor parallelism in its checkpoint?

---

## New Research Thread: MoE Expert Specialization

Once the sidecar is operational, a new research thread becomes available:

**Thread 15 — MoE Expert Specialization**
- Expert load distribution across task families (induction, recency, syntax, coreference, capitalization)
- Per-expert activation patterns: which experts preferentially activate for which task types
- Expert routing geometry: PCA/Bregman analysis of gate weight matrix to measure expert separation in weight space
- Interaction with layer criticality: do L19-21 critical layers show more or less expert specialization?
- Expert ablation using accurate sidecar indices (replacing fraction-scaling approximation)

This thread connects directly to the Hydra effect findings: if head redundancy is
extreme (sigma=0.042), do experts show analogous redundancy, or does the MoE
architecture enforce specialization by a different mechanism?

---

## Design Principles (from ecosystem lessons)

Two lessons from the existing interpretability tooling landscape apply directly
to Phase 1 implementation decisions:

**1. Minimal hook management (baukit principle).**
The existing `ActivationCache` is already close to baukit's design: a thin
context-manager wrapper around `register_forward_hook` that clears itself on
exit. Resist adding abstraction layers. The public API should expose clean
types and entry points, not framework machinery.

**2. Serializable interventions (pyvene principle).**
`InterventionSpec` objects should be serializable to JSON/YAML so that
experiments can be saved, shared, and reproduced without Python files.
This costs almost nothing in Phase 1 and pays off immediately for
HuggingFace distribution: a researcher can share a YAML file that
fully reproduces a benchmark run.

---

## Future Directions

### Neuronpedia Compatibility

**Aspiration:** Export GOSSH activation and routing data in Neuronpedia-compatible
formats so results are browsable via the community's standard interpretability
platform.

**Why deferred:** Neuronpedia is designed around SAE feature dictionaries and
attribution graphs. GOSSH's primary outputs — routing decisions, causal ablation
results, logit-lens trajectories — do not map cleanly to that format without
first training SAEs on gpt-oss-20b, which is a separate research project.
The mismatch is structural, not incidental: GOSSH captures *mechanisms*;
Neuronpedia displays *features*. Compatibility becomes natural once SAE
training on gpt-oss-20b is complete.

**Path forward:** Once Phase 2 (sidecar) is operational and expert-level
activation data is available, SAE training on gpt-oss-20b activations becomes
tractable. At that point, `gossh.export.neuronpedia` can wrap the SAE features
and routing data in the platform's expected format.

### Circuit Tracer Integration

**Aspiration:** Use Anthropic's open-sourced Circuit Tracer
(`github.com/decoderesearch/circuit-tracer`) to generate attribution graphs
on gpt-oss-20b and gpt-oss-120b, enabling circuit-level analysis of the
production models.

**What makes this significant:** Circuit Tracer operates via cross-layer
transcoders (CLTs) — sparse linear maps trained to decompose MLP activations
into interpretable features across layers. CLTs do not yet exist for
gpt-oss-20b. Training them would be a first for a production-scale MXFP4 MoE
model and would directly extend Anthropic's interpretability program to the
OpenAI model family.

**The compute requirement is the gating factor.** CLT training at the scale
of gpt-oss-20b/120b requires distributed training infrastructure:
activation collection across a large and diverse prompt corpus, sharded
training across many GPUs, and validation of feature quality via downstream
probing and circuit reconstruction. This is not a solo workstation task.

**Why it's worth planning for:** GOSSH's sidecar and activation capture
infrastructure (Phases 1–2) directly supports CLT training — the activation
data pipeline is the same. Phase 3 (HuggingFace distribution) makes
collecting diverse activations from community prompts tractable. If compute
becomes available, the path from GOSSH to full Circuit Tracer integration
on gpt-oss-20b is:

1. Collect large-scale activations via GOSSH (Phase 2 infrastructure)
2. Train CLTs on collected activations (distributed training job)
3. Validate CLT feature quality via GOSSH probing tools
4. Run Circuit Tracer attribution graphs using trained CLTs
5. Publish CLTs to HuggingFace for community use

**MoE compatibility caveat:** Circuit Tracer's CLT design targets dense MLP
layers. gpt-oss-20b's expert-gated MLP requires per-expert CLTs or a
routing-aware CLT variant. This is an open research question — and a
potentially publishable contribution in its own right.
