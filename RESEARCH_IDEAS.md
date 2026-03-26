# Research Ideas — gpt-oss-20b

Collected experiment proposals and feature-system extensions, organized by type.
Ideas here are candidates for future threads; the Thread Map is the authoritative status tracker.

---

## Part I — Feature System Extensions (Thread 9 follow-up)

The current `MoEFeatureExtractor` produces 6,425D vectors from components A–G.
`ExpertCapture` (added in Thread 15) gives access to individual expert output tensors,
enabling three new components that weren't previously computable.

### Current components

| ID | Name | Dimensions | What it captures |
|----|------|:----------:|-----------------|
| A | Trajectory | `3L − 1 = 71` | Per-layer logit prob, margin over 2nd-best, inter-layer drops for the final-prediction token |
| B | Stability | `2` | k* (first stable layer) and κ (max consecutive stable layers) |
| C | Head Activation | `2LH = 6144` | Peak attention weight per head at stable layer and at final layer |
| D | Head Entropy | `2LH = 6144` | Attention entropy per head at stable + final layer (entropy-gated) |
| E | Expert Routing | `2E + L·top_k = 160` | Soft routing weights at stable + final layer; sorted top-k weights at every layer |
| F | Routing Entropy | `L = 24` | Shannon entropy of routing distribution per layer |
| G | Attention Scale | `L = 24` | Fraction of attention mass in the sliding window per layer (local vs global contrast) |
| — | **Total** | **6,425** | |

### Proposed new components

#### H — Expert Output Norm  `L · top_k = 96D`

**What it is**: For each of the top-k experts that fire at each layer, the L2 norm of the
expert's output tensor before gating (`‖expert_output‖₂`).

**Why it matters**: Routing weights (Component E) tell you *which* experts are selected;
output norms tell you *how much* each expert writes to the residual stream. A highly-weighted
expert with a small output norm is doing something different from a lightly-weighted expert with
a large norm. The two are only proportional if expert output norms are constant — they aren't.

**How to compute**: Requires `ExpertCapture` (Thread 15). For each token position and each
top-k expert slot, capture `expert_output.norm(dim=-1)` at the time of capture.

**Dimensionality**: `L × top_k = 24 × 4 = 96D`. Sorted in descending routing-weight order
(same ordering as Component E's top-k weights) so the vector is permutation-invariant w.r.t.
expert identity.

---

#### I — Expert Vocabulary Alignment  `2 · top_k = 8D`  (or `2E = 64D` dense variant)

**What it is**: For each expert that fires at the stable layer and the final layer,
the cosine similarity between the expert's output (projected through `final_norm + lm_head`)
and the unembedding direction of the model's final prediction:

```
alignment(expert_e, layer_l) = cos( lm_head(final_norm(expert_e_output)), W_U[:, predicted_token] )
```

**Why it matters**: This is the most direct measure of whether an expert is *contributing to
the answer* vs. doing something orthogonal. An expert at L20 with alignment ≈ 1 is writing
the answer into the residual stream. An expert with alignment ≈ 0 is doing something else
(syntax, positional bookkeeping, etc.). Combined with routing weights, this gives a per-expert
decomposition of the model's answer.

**How to compute**: Requires `ExpertCapture` + `lm_head`. Project each captured expert output
tensor through `final_norm` and `lm_head`, take the top-1 token direction, compute cosine.
Only feasible on non-quantized checkpoints (MXFP4 fuses expert dispatch).

**Dimensionality (sparse variant)**: `2 × top_k = 2 × 4 = 8D` — alignment scores for the
top-4 experts at stable and final layers only. Sorted by routing weight order.

**Dimensionality (dense variant)**: `2 × n_experts = 2 × 32 = 64D` — alignment score for
all 32 experts (zero for non-selected experts). Richer but requires all expert outputs.

---

#### J — Logit Delta  `L = 24D`

**What it is**: Per-layer change in log-probability of the final prediction:

```
Δ log p_l = log p_l(predicted_token) − log p_{l−1}(predicted_token)
```

**Why it matters**: Component A captures the *cumulative* probability trajectory.
Logit delta is its *derivative* — it shows which layers *promote* vs. *suppress* the
eventual answer token. This is directly the quantity studied in Thread 15 (Measurement 2)
and connects the feature vector to the layer logit-delta analysis. A token with a single
large positive delta at L20 has a very different computational profile from one with a
smooth ramp.

**How to compute**: Derived from Component A's `traj_probs` tensor — no additional forward
pass required. `Δlog_p[l] = log(traj_probs[l]) - log(traj_probs[l-1])`, with
`Δlog_p[0] = 0` (or relative to the embedding layer output).

**Dimensionality**: `L = 24D`. Can be added at near-zero cost during extraction.

---

#### K — Expert Co-activation Signature  `L · (top_k choose 2) = 144D`

**What it is**: For each layer, a binary indicator of which *pairs* of experts co-fired
(out of the `C(top_k, 2) = 6` possible pairs when top-k = 4). Across all layers: `L × 6 = 144D`.

**Why it matters**: Component E records which experts fire and how strongly. The co-activation
pattern records *which experts fire together* — a different structural signal. If experts
{3, 17} always co-fire on syntax-agreement tokens and {3, 8} always co-fire on induction
tokens, the co-activation signature discriminates them even if individual routing weights
are similar.

**How to compute**: From the top-k expert indices (already available in routing data).
For each layer, enumerate all pairs from the top-k set, set the corresponding bit. Sorted
expert indices ensure canonical ordering (permutation-invariant).

**Dimensionality**: `L × C(top_k, 2) = 24 × 6 = 144D`.

---

#### L — Expert Routing Persistence  `L − 1 = 23D`

**What it is**: For each consecutive layer pair `(l, l+1)`, the Jaccard similarity of the
top-k expert sets:

```
persistence(l) = |selected_experts(l) ∩ selected_experts(l+1)| / |selected_experts(l) ∪ selected_experts(l+1)|
```

**Why it matters**: If the same token is routed to the same 4 experts across all 24 layers,
routing is "sticky" — the token's representation stays in a fixed expert subspace as it
propagates. If routing changes completely at every layer, each layer is making an independent
routing decision. Persistence links the per-layer routing signal into a depth-trajectory
feature. This also connects to the question of whether MXFP4 rounding distorts routing
relative to bf16 (Thread 22 below).

**How to compute**: From top-k expert index arrays. No additional forward pass.

**Dimensionality**: `L − 1 = 23D`.

---

### Updated feature dimensionality

| ID | Component | Dims | Status |
|----|-----------|:----:|--------|
| A | Trajectory | 71 | Implemented |
| B | Stability | 2 | Implemented |
| C | Head Activation | 6144 | Implemented |
| D | Head Entropy | 6144 | Implemented |
| E | Expert Routing | 160 | Implemented |
| F | Routing Entropy | 24 | Implemented |
| G | Attention Scale | 24 | Implemented |
| H | Expert Output Norm | 96 | **Proposed** — needs ExpertCapture |
| I | Expert Vocab Alignment | 8 (sparse) or 64 (dense) | **Proposed** — non-quantized only |
| J | Logit Delta | 24 | **Proposed** — free from Component A |
| K | Expert Co-activation | 144 | **Proposed** — from routing indices |
| L | Expert Routing Persistence | 23 | **Proposed** — from routing indices |
| — | **New total (sparse I)** | **~6,700** | |
| — | **New total (dense I)** | **~6,756** | |

Components J, K, L are zero-cost (derived from existing data). Components H and I require
`ExpertCapture` and are MXFP4-constrained (non-quantized checkpoint only).

---

## Part II — Proposed New Experiments

### Near-term (tools exist, new measurement)

#### Thread 16 — Sliding vs. Full Attention Specialization

**Model**: gpt-oss-20b
**Question**: Do MoE layers operating under sliding-window attention compute differently
from those under global attention?

gpt-oss-20b alternates layers: even layers use a 128-token sliding window; odd layers
use full attention. Component G (attention scale) measures local fraction but doesn't
directly compare what MoE experts *write* across these two regimes.

**Measurement**:
- Run logit-delta (Measurement 2, Thread 15) separately grouping sliding vs. full-attention layers
- Compare routing entropy (Component F) between the two groups
- Test whether expert utilization skew (Thread 15 Measurement 1) differs systematically
  by attention type

**Hypothesis**: Full-attention layers contribute more to coreference and induction (require
global context); sliding-attention layers dominate local syntax and capitalization.

**Tooling**: `run_logit_lens`, `RouterCapture` or sidecar, `ExpertCapture`
**Expected effort**: ~1 day (analysis script + 1 figure)

---

#### Thread 17 — Token-Type Routing Bias

**Model**: gpt-oss-20b
**Question**: Do specific experts systematically over-select certain BPE token categories
(punctuation, digits, function words, CJK characters, rare tokens)?

**Measurement**:
- Collect sidecar routing decisions across a diverse corpus (~1K tokens spanning ASCII,
  Unicode, rare BPE tokens)
- Categorise tokens by surface type: punctuation, alphabetic, numeric, whitespace, CJK,
  vocabulary rank (top-1K vs. long tail)
- Compute mutual information `MI(expert_selected, token_category)` per layer

**Hypothesis**: Early-layer experts (L0–8) have higher MI with token surface form than
late-layer experts; late-layer experts are more conditioned on semantic context than
token surface.

**Connection to feature system**: MI scores at each layer would be a natural addition to
the feature vector as a *corpus-level* routing bias signal — different from the per-token
routing weight.

**Tooling**: MoE sidecar (MXFP4-safe), diverse tokenizer corpus
**Expected effort**: ~2 days (corpus collection + analysis)

---

#### Thread 18 — Late-Layer Expert Identity for Analysis Set

**Model**: gpt-oss-20b
**Question**: For the 9 clean analysis-set cases (Thread 3), which specific experts within
L19–21 are selected? Is the set consistent within task families?

**Measurement**:
- Run sidecar routing for all 9 clean cases across all 5 task families
- For L19, L20, L21: record the exact 4-of-32 expert combination selected at the
  decision token position
- Compute: (a) within-family expert overlap, (b) cross-family expert overlap,
  (c) whether any single expert appears in all clean cases

**Hypothesis**: Within a task family, the same 2–3 experts are selected consistently
at L20 (the causal bottleneck layer). Across families, expert sets diverge.
If true, this is the strongest evidence yet for expert-level functional specialization.

**Tooling**: MoE sidecar, Thread 3 analysis-set prompts
**Expected effort**: ~1 day (routing capture + table)

---

#### Thread 19 — Attention Sink Characterization

**Model**: gpt-oss-20b
**Question**: Do attention sinks (disproportionate attention mass at position 0) form
differently in sliding vs. full-attention layers?

Many transformers route large attention weight to position 0 regardless of content.
gpt-oss-20b's alternating pattern creates a natural comparison:
- Sliding layers: sinks are constrained to the 128-token window
- Full layers: sinks can form at position 0 regardless of sequence length

**Measurement**:
- Capture attention weights at all layers for sequences of length 32, 64, 256, 512
- For each layer, measure: fraction of attention mass at position 0 (sink score)
- Plot sink score as a function of depth, stratified by layer type (sliding/full)

**Hypothesis**: Sinks appear earlier in full-attention layers. Sliding-attention layers
may exhibit "local sinks" (attention concentrated at the window boundary) that don't
appear in full-attention layers.

**Tooling**: `ActivationCache` or `output_attentions=True`
**Expected effort**: ~1 day (no routing needed)

---

#### Thread 20 — Induction Head Localization Under GQA

**Model**: gpt-oss-20b
**Question**: Where are induction heads, and does GQA (64 query / 8 KV heads) prevent
their localization?

Standard induction-head detection: measure copy-prefix score per head on a `[A][B]...[A]`
pattern. The Hydra result (σ = 0.042 at L20) predicts induction heads are distributed,
not localized. This test makes that prediction precise.

**Measurement**:
- Generate copy-prefix prompts: repeat a 10-token random sequence 5×
- Measure per-head induction score: `attn_weight[h, pos_A2, pos_B1]` where B1 follows A1
- Plot induction score by head and layer; compare variance to σ = 0.042 Hydra baseline

**Hypothesis**: No single head achieves high induction score in isolation; the signal
distributes across many heads in L12–20, consistent with the Hydra result.

**Tooling**: `output_attentions=True`
**Expected effort**: ~1 day

---

#### Thread 21 — Residual Stream Rank Trajectory on gpt-oss-20b

**Model**: gpt-oss-20b
**Question**: What is the effective rank of hidden states at each layer in the production
model? How does this compare to the DST models' 8/516 effective rank?

Thread 14 (Bregman geometry) measures this for DST companion models (45.4M parameters,
stream separation up to 22× conditioning improvement). Applying `analyze_geometry` to
gpt-oss-20b's actual hidden states gives the production-model equivalent.

**Measurement**:
- Capture hidden states at all 24 layers for the 9 clean analysis-set cases
- Compute effective rank (90% variance explained) and intrinsic dimension at each layer
- Plot rank trajectory vs. depth; mark the L19–21 causal bottleneck

**Hypothesis**: Effective rank drops in late layers (L19–21) as the model concentrates
computation for the answer — low rank is expected at the decision layer. Early layers
may have higher rank (more distributed representations).

**Tooling**: `ActivationCache`, `analyze_geometry`, `compute_inspectability`
**Expected effort**: ~1 day (analysis script + 1 figure)

---

### Longer-horizon (new measurement infrastructure needed)

#### Thread 22 — MXFP4 as an Interpretability Perturbation

**Model**: gpt-oss-20b (quantized vs. reference)
**Question**: How much does MXFP4 quantization distort the routing decisions relative to
bf16, and what does that mean for interpretability claims made via the sidecar?

The sidecar already runs bf16 gate clones and validates against the quantized model.
This thread makes the comparison systematic:

**Measurement**:
- For a held-out set of prompts, compare sidecar routing (bf16) vs. any available
  non-quantized checkpoint routing
- Compute: routing agreement rate per layer, Jensen-Shannon divergence between routing
  distributions, fraction of tokens where the top-1 expert changes

**Hypothesis**: Agreement rate > 90% overall; divergence concentrates at tokens where
the routing logits are near-tied (close to the top-4 boundary). MXFP4 distortion is
measurable but not the dominant source of routing variance.

**Significance**: If confirmed, this validates the sidecar as a reliable proxy.
If not, quantization is a confound for all routing-based interpretability claims.

---

#### Thread 23 — Cross-Layer Routing Correlation Matrix

**Model**: gpt-oss-20b
**Question**: Is expert routing consistent across layers for a given token (sticky),
or does each layer make independent decisions?

**Measurement**:
- For each prompt and each token, collect the top-k expert set at every layer
- Compute a 24×24 matrix of pairwise Jaccard similarity across layers (per token)
- Average over the corpus; visualise as a heatmap

**Hypothesis**: High correlation between consecutive layers (especially L18–21) for
tokens in the stable analysis set; near-zero correlation between early and late layers.
The correlation matrix is task-dependent — induction tokens may show different cross-layer
consistency than capitalization tokens.

**Connection to feature system**: The persistence vector (Component L) is the diagonal
of this matrix. This thread studies the full off-diagonal structure.

---

### Theoretical extensions

#### Thread 24 — SAE Features as Feature-Vector Components

**Question**: Can sparse autoencoder (SAE) features (Cunningham et al. 2023, Bricken et al. 2023)
be incorporated as additional components in the MoE feature vector?

The current feature vector captures *computational structure* (how the model processes a token).
SAE features capture *semantic structure* (what the model represents). The two are complementary.

**Proposal**:
- Train a small SAE on gpt-oss-20b's L20 hidden states (the causal bottleneck layer)
- Use SAE feature activations as an additional component (Component M) in the feature vector
- Re-run the Thread 9 PCA/clustering and compare intrinsic dimensions to the current 6,425D result

**Expected finding**: Adding SAE features at L20 should reduce intrinsic dimension for
task families whose computation is concentrated at the bottleneck layer (capitalization,
induction) while leaving coreference intrinsic dimension high (coreference computation
is distributed across many layers, not just L20).

---

## Part III — Summary Table

| Thread | Title | Type | Effort | Priority |
|--------|-------|------|--------|----------|
| 9 ext | Feature components H–L | Feature system | Low (J/K/L free; H/I need ExpertCapture) | **High** |
| 16 | Sliding vs. full attention specialization | Measurement | 1 day | **High** |
| 17 | Token-type routing bias | Measurement | 2 days | High |
| 18 | Late-layer expert identity for analysis set | Measurement | 1 day | **High** |
| 19 | Attention sink characterization | Measurement | 1 day | Medium |
| 20 | Induction head localization under GQA | Measurement | 1 day | Medium |
| 21 | Residual stream rank trajectory | Measurement | 1 day | Medium |
| 22 | MXFP4 as interpretability perturbation | Validation | 2 days | Medium |
| 23 | Cross-layer routing correlation matrix | Measurement | 2 days | Medium |
| 24 | SAE features as feature-vector components | Theoretical | — | Low |

**Immediate recommendations**:
1. Add Component J (logit delta) to `gossh/features/extractor.py` — it's free and directly connects Thread 9 to Thread 15
2. Add Components K and L (co-activation + persistence) — also free from existing routing data
3. Run Thread 18 next — uses only the sidecar + existing analysis-set prompts, directly strengthens the Thread 2 causal bottleneck result
4. Then Thread 16 — sliding/full attention specialization directly tests the most structurally distinctive property of gpt-oss-20b's architecture
