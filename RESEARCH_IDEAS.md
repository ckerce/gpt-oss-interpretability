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

## Part III — Information-Theoretic Routing Analysis

### Conceptual grounding

Every information-theoretic measure answers a slightly different question about the routing
distribution.  Routing at layer l produces a probability vector `p = [p_0, …, p_31]` over
32 experts.  The measures below decompose what that vector reveals.

---

### The four core interpretations

#### 1. Entropy H — "How much do you need to know?"

```
H(routing_l) = −Σ_e  p_e · log(p_e)
```

**Interpretation**: the minimum average nats needed to describe which expert fires at
layer l.  Uniform routing = log(32) ≈ 3.47 nats.  Any structure below this baseline
represents a compression of the routing decision — the model has "already decided"
something that reduces uncertainty.

**Already measured** in Thread 15.  Shows monotonic concentration in L17–23.

---

#### 2. KL Divergence — "How much do you waste using the wrong codebook?"

```
D_KL(P ‖ Q) = Σ_e  p_e · log(p_e / q_e)
```

Asymmetric.  D_KL(P‖Q) = extra nats burned if you encode samples from P using a code
optimised for Q.  Three distinct uses for routing:

**2a — Specialization gain**: `D_KL(routing_l ‖ uniform) = log(32) − H(routing_l)`

The nats saved by knowing the true routing distribution vs. assuming uniform.  Equivalent
to entropy but framed as *information value*: how much does knowing the routing policy
compress routing descriptions?  Expected profile: near-zero in early layers (load
balancing succeeds), rising toward the causal bottleneck.

**2b — Routing transfer cost**: `D_KL(routing_{task_A, l} ‖ routing_{task_B, l})`

Extra nats burned if task A's routing decisions are encoded using task B's routing
"codebook."  Asymmetric on purpose: `D_KL(induction ‖ coreference)` answers "how
surprised is the coreference routing policy by induction tokens?"  High asymmetry
between two tasks indicates that one task's routing is a *special case* of the other's,
not that they simply differ.

**2c — Routing velocity**: `D_KL(routing_l ‖ routing_{l−1})`

How fast does the routing distribution change across depth?  A spike in velocity identifies
a routing *phase transition* — the layer where the model's routing policy reorganises most
sharply.  Hypothesis: velocity peaks near L17, at the onset of the causal bottleneck, not
within it.  The model commits to a routing regime before the bottleneck, not during.

---

#### 3. Jensen-Shannon Divergence — "How easily can you tell them apart?"

```
JSD(P, Q) = ½ D_KL(P ‖ M) + ½ D_KL(Q ‖ M),   M = ½(P + Q)
```

Symmetric, bounded in [0, ln 2] nats ≈ [0, 0.693 nats].  Interpretation: the Bayes-optimal
cost of determining whether a routing decision was drawn from P or Q.  JSD = 0 means the
distributions are identical; JSD = ln 2 means they have disjoint support (a perfect detector
exists).  Equivalent to the mutual information in a symmetric hypothesis test.

**3a — Task routing distinguishability**: `JSD(routing_{task_A, l}, routing_{task_B, l})`

At each layer, compute the 5×5 symmetric matrix of pairwise task JSD values.  This gives:
- Which task pairs have distinguishable routing at which depths
- The layer at which each task pair's routing "separates" (JSD rises above threshold)
- Whether the separation order matches the convergence order from Thread 1

Hypothesis: capitalization and coreference separate earliest (L1–5); induction and recency
separate latest (L17+).  If routing separation tracks convergence depth, routing *encodes*
the task resolution timeline.

**3b — MXFP4 routing distortion**: `JSD(routing_{bf16}, routing_{MXFP4})`

The symmetric counterpart to Thread 22's KL analysis.  JSD is more interpretable here:
a JSD of 0.1 nats means a classifier that observes one routing decision can identify the
quantization regime with probability ≈ 1 − exp(−0.1) ≈ 9.5%.  This calibrates how much
MXFP4 "moves" the routing distribution in a detector-theoretically meaningful way.

---

#### 4. Mutual Information — "How much does one thing tell you about another?"

```
I(X; Y) = H(X) + H(Y) − H(X, Y) = H(X) − H(X | Y)
```

The reduction in uncertainty about X after observing Y.  Symmetric.

**4a — Task routing alignment**: `I(expert_l; task_family)` per layer

How many nats of task identity are encoded in the routing decision at layer l?
- Compute the joint distribution `p(expert_e, task_t)` from the routing data
- `I = H(task) − H(task | expert)` = nats of task information "readable" from the
  routing decision
- Expected profile: near-zero in early layers (routing is token-driven, not task-driven),
  rising to a peak at L19–21 if routing encodes task structure

This is the most direct measure of whether expert selection is *task-specific* at any
given depth.

**4b — Expert–prediction alignment**: `I(expert_l; predicted_token)` per layer

How much does knowing which expert fired tell you about the model's final output token?
- Compute `p(expert_e, output_token_v)` from routing + logit-lens data
- Expected: near-zero in early layers, rising sharply at the causal bottleneck

**4c — Cross-layer routing persistence**: `I(expert_l; expert_{l+1})` for all layer pairs

A 24×24 MI matrix measuring how "sticky" routing is across depth.  High off-diagonal MI
between nearby layers indicates that routing decisions propagate — the same token stays
in the same expert subspace as it passes through depth.  Low MI indicates each layer
routes independently.  This is the information-theoretic formalisation of Component L
(routing persistence) in the feature system.

**4d — Token surface vs. context routing**: `I(expert; token_surface) vs. I(expert; task)`

Is routing driven by the surface form of the token (BPE identity) or by the broader
task context?  At early layers, `I(expert; token_surface)` should dominate.  At late
layers, `I(expert; task)` should dominate.  The crossover depth is a direct measure of
where the model transitions from surface-form to semantic processing.

---

### Channel capacity framing

Each routing decision selects a combination of 4 from 32 experts:

```
routing capacity = log₂(C(32, 4)) = log₂(35,960) ≈ 15.1 bits per token per layer
```

The actual task mutual information `I(expert; task)` is likely < 1 bit at any layer.
The *routing utilisation efficiency*:

```
η_l = I(expert_l; task) / log₂(C(32, 4))
```

is almost certainly < 7% at the bottleneck layer.  This is itself an interpretability
finding: the routing mechanism has enormous combinatorial capacity, but the vast majority
of it is not used for task-discriminating computation.  The remainder is used for:
- Token-surface processing (I(expert; token_surface))
- Load balancing (the load-balancing loss forces routing toward uniform, consuming capacity
  "uselessly" from a task-information standpoint)
- Context-dependent processing not captured by the 5-family task taxonomy

The three components partition the routing capacity:
```
log₂(C(32,4)) ≈ I(expert; task) + I(expert; token_surface) + residual
```

Measuring each component gives a budget decomposition of what routing capacity is spent on.

---

### Total Variation Distance — the non-specialist version

```
TV(P, Q) = ½ · Σ_e |p_e − q_e|   ∈ [0, 1]
```

Directly interpretable: TV(P, Q) = the maximum probability that any detector can
distinguish a sample drawn from P vs. one drawn from Q.  Equivalent to JSD in the
large-sample limit but easier to explain to non-specialists.

For routing: `TV(routing_{task_A}, routing_{task_B})` is the probability that an optimal
Neyman-Pearson classifier — given a single routing decision — can identify which task
generated it.  A TV of 0.15 at L20 would mean: observing a single routing vector at L20
allows you to identify the task with 15% better-than-chance accuracy.

---

### Proposed experiments (Threads 25–29)

#### Thread 25 — Routing Specialization Gain Trajectory

**Question**: How does `D_KL(routing_l ‖ uniform)` evolve across depth?

**Measurement**: From the sidecar routing data (Thread 15 Measurement 1), compute the
KL from uniform at each layer.  This is `log(32) − H(routing_l)` — directly derivable
from Thread 15's `routing_entropy.json` at zero additional cost.

**Expected figure**: A bar chart identical in structure to Thread 15's entropy figure
but y-axis in "specialization nats."  Y-axis now reads as "nats gained by knowing the
routing policy" rather than "nats of uncertainty."  The two figures are numerically
related by a reflection and a constant offset, but the framing is different and more
actionable.

**Novelty**: Zero new computation needed.  Pure reframing of existing data.  The
specialization gain at the bottleneck layer (expected ~0.3 nats at L23) quantifies how
much routing "knows" beyond uniform at the moment of task resolution.

---

#### Thread 26 — Cross-Task Routing Distinguishability (JSD Matrix)

**Question**: Which task pairs have distinguishable routing, and at which depth?

**Measurement**:
1. For each task family × each layer, collect the empirical routing distribution from
   sidecar data across all prompts in that family
2. Compute the 5×5 symmetric JSD matrix at each of 24 layers
3. Plot: (a) JSD heatmap at L0, L8, L17, L20, L23; (b) max JSD vs. depth per task pair

**Expected figure**: At L0, all task pairs are near-indistinguishable (JSD ≈ 0 everywhere).
By L20, capitalization and induction should be highly distinguishable.  Coreference and
recency may remain similar if their routing profiles overlap.

**Key result**: The depth at which each task pair's JSD crosses a threshold (e.g., 0.05
nats) gives a routing-based convergence depth.  If this tracks the logit-lens convergence
depth from Thread 1, routing and representation share the same computational timeline.
If not — routing separates tasks at a different depth than representations do — that
dissociation is scientifically meaningful.

---

#### Thread 27 — Task Routing Mutual Information Curve

**Question**: How many nats of task identity are encoded in the routing decision at each layer?

**Measurement**:
1. From sidecar routing: collect `(layer, expert_set, task_family)` tuples
2. Compute `I(expert_l; task_family)` using the empirical joint distribution
3. Plot MI vs. depth

**Expected shape**: Near-zero in L0–8, rising in L9–16, peaking at L19–21 if routing
is task-specific at the bottleneck.  If MI peaks earlier than L19, routing encodes task
identity before the representation does — routing leads the computation.  If MI is flat
or near-zero everywhere, routing is task-agnostic (purely token-surface driven) and
the functional specialisation must come from within-expert computation, not selection.

**Connection to feature system**: `I(expert_l; task_family)` is the *ground truth* label
for Component E's routing features — it tells you how much predictive value those features
have for task identification.

---

#### Thread 28 — Routing Velocity and Phase Transitions

**Question**: Where does the routing distribution change fastest across depth?

**Measurement**:
1. From sidecar routing: compute `D_KL(routing_l ‖ routing_{l−1})` for each consecutive
   layer pair across all prompts
2. Average over prompts; plot routing velocity vs. depth
3. Stratify by task family

**Expected finding**: Routing velocity peaks near L16–17 — the onset of the causal
bottleneck — not within it (L19–21).  The model "decides" its routing regime one to two
layers before the causal bottleneck becomes active, consistent with the bottleneck being
the execution site of a routing decision made upstream.

**Asymmetry check**: Also compute `D_KL(routing_{l−1} ‖ routing_l)` and check for
asymmetry.  A large asymmetry at a specific layer means routing at that layer is
"surprising" to the previous layer but not vice versa — indicative of a directional
information flow rather than a smooth transition.

---

#### Thread 29 — Routing Capacity Budget Decomposition

**Question**: What fraction of the 15.1-bit routing capacity is used for task-discriminating
computation vs. token-surface processing vs. load-balancing overhead?

**Measurement**:
1. `I(expert_l; task_family)` — task component (Thread 27)
2. `I(expert_l; token_BPE_id)` — token-surface component
3. `H(routing_l)` — total uncertainty used
4. `log₂(C(32,4)) − H(routing_l)` — load-balancing overhead (nats "consumed" by
   concentration below maximum capacity)
5. Residual = routing capacity − task − token-surface − overhead

Plot the budget stacked bar chart across all 24 layers.

**Expected finding**: In early layers, token-surface dominates.  In late layers, task
and token components both grow but the total is still << 15.1 bits.  The residual
(context-dependent computation not captured by either) may be large — this is the
"unknown unknown" in the routing budget.

---

### Summary additions to Part III table

| Thread | Title | Type | Effort | Priority |
|--------|-------|------|--------|----------|
| 25 | Routing specialization gain trajectory | Analysis (free) | 0.5 day | **High** |
| 26 | Cross-task JSD distinguishability matrix | Measurement | 1 day | **High** |
| 27 | Task routing mutual information curve | Measurement | 1 day | **High** |
| 28 | Routing velocity and phase transitions | Measurement | 1 day | High |
| 29 | Routing capacity budget decomposition | Measurement | 2 days | Medium |

Threads 25–28 require only sidecar routing data already collected in Thread 15.
Thread 25 is literally free — it's a relabelling of Thread 15's `routing_entropy.json`.

---

## Part V — Routing as a Data-Organization Lens

*Grounded in the lit review synthesised March 2026.*

The literature establishes three tiers of evidence that MoE routing is fundamentally a
data-clustering mechanism, not merely a load-balancer:

**Foundational (1991–1995)**: Jacobs et al. (1991), Jordan & Jacobs (1994), and Xu et al.
(1995) show that gating networks perform competitive learning over the input distribution —
mathematically identical to EM-based Gaussian mixture clustering.  Routing an input to an
expert is assigning it to a cluster.

**Theoretical proof (2022–2025)**: Chen et al. (NeurIPS 2022) proved that under gradient
descent, MoE routers learn cluster-center features.  Dikkala et al. (EMNLP 2023) proved
routers recover Gaussian mixture assignments.  Kawata et al. (ICML 2025) proved MoE detects
latent cluster structure that dense networks *provably cannot* — the router is the mechanism
by which MoE outperforms dense models on clustered input distributions.

**Empirical exploitation (2021–2025)**: V-MoE (2021) shows routing recovers ImageNet
semantic classes in deep layers.  ST-MoE (2022) and OpenMoE (2024) find token-surface
categories (punctuation, digits, function words) emerge as routing clusters.  HMOE (2022)
uses routing for label-free domain discovery, finding learned clusters more coherent than
human labels.  Li & Zhou (ICLR 2025) show routing-weight vectors are off-the-shelf
embeddings competitive on MTEB.  Nikolic et al. (2025) show unsupervised routing discovers
sub-categorical structure *beyond* human-defined class boundaries.

### How our empirical results fit

| Our finding | Literature connection | Interpretation |
|---|---|---|
| MI(expert; task) peaks at L12, not L19–21 | OpenMoE #11: "routing determined early, stable thereafter" | Routing encodes cluster membership mid-network; late layers execute, not decide |
| Code JSD 0.39–0.51 vs. natural language | ST-MoE, OpenMoE: token-surface drives early routing | Code's distinct BPE surface creates a near-disjoint cluster from the start |
| Analogy/coreference/recency cluster (JSD 0.04–0.09) | HMOE: clusters more coherent than human labels | Router groups by shared computation (antecedent resolution), not surface domain |
| 4.4% task capacity budget | Li & Zhou: routing = high-level semantics; rest = surface | Task identity is a thin semantic slice of the full routing capacity |
| Routing sub-uniform all layers | Chen et al.: router recovers cluster centers under gradient descent | Load-balancing suppresses but cannot eliminate specialization |

### Three unexploited analyses on existing data

All three can be run against `runs/expert_readouts/routing_patterns.json`
*today* — no new model forward passes needed.

---

#### Thread 30 — Routing-Weight Embeddings and Task Geometry

**Core reference**: Li & Zhou, ICLR 2025 ("Your MoE LLM Is Secretly an Embedding Model For Free")

**Question**: Does the routing-weight vector across layers encode task structure that is
(a) discriminative without any fine-tuning, and (b) complementary to hidden-state embeddings?

**What the routing-weight vector is**:
For each prompt, the sidecar captures per-layer routing decisions.  Aggregating across the
24 MoE layers, we can form a vector `r ∈ ℝ^{24 × 32}` where `r[l, e]` is the fraction of
tokens routed to expert `e` at layer `l`.  This is a 768-dimensional embedding derived
purely from routing decisions — no hidden states required.

**Measurements**:

1. **UMAP/t-SNE projection** (Nikolic 2025 methodology):
   - Project 135 routing-weight vectors (one per prompt) into 2D
   - Color by task family
   - Expected: code_completion separates first; analogy/coreference/recency cluster together
   - Key question: does unsupervised layout recover the JSD dendrogram?

2. **Routing-weight vs. hidden-state similarity** (Li & Zhou methodology):
   - For each prompt pair, compute:
     - `sim_routing(i, j) = cosine(r_i, r_j)` — routing-weight similarity
     - `sim_hidden(i, j) = cosine(h_i, h_j)` — last-token hidden-state similarity at L20
   - Compare the two similarity matrices via Mantel test (correlation of pairwise distances)
   - Expected: high within-family similarity in both; routing-weight similarity decorrelates
     more across certain task pairs (code vs. NL) while hidden-state similarity decorrelates
     more for others (e.g., factual recall prompts with shared surface forms)

3. **Hierarchical clustering dendrogram** (HMOE methodology):
   - Use the empirical JSD matrix (`routing_jsd_matrix.json`, already computed) as a
     distance metric
   - Produce a dendrogram over 9 task families
   - Compare to: (a) human intuitions about task relatedness, (b) NLP benchmark correlation
     matrices from SuperGLUE / BIG-Bench
   - Expected dendrogram: `{code} | {arithmetic | factual_recall} | {analogy | coreference |
     recency | syntax | capitalization | induction}` — three broad clusters

4. **Routing clusters vs. human labels** (Nikolic 2025 key result):
   - Run k-means (k = 5, 9, 15) on the 135 routing-weight vectors
   - Compute adjusted Rand index between k-means assignments and true task labels
   - Expected: ARI is highest at k ≈ 9 (matching our 9 families), but the unsupervised
     clusters at k = 9 may not perfectly align with task labels — code/arithmetic may split,
     while analogy/coreference/recency may merge into one cluster

**Tooling**: `routing_patterns.json` (on disk), `sklearn.manifold.UMAP`, `scipy.cluster.hierarchy`
**New model runs required**: None
**Expected effort**: 1 day (analysis script + 3–4 figures)

**Connection to feature system**: The 768D routing-weight vector is a new candidate feature
component (Component M) — complementary to the existing 160D Component E (which captures
per-token routing weights, not prompt-aggregate statistics).  If Li & Zhou's finding holds
for gpt-oss-20b, Component M would be the single most semantically informative feature in
the entire 6,700D vector.

---

#### Thread 31 — Sub-Categorical Structure Discovery

**Core reference**: Nikolic et al. 2025 ("unsupervised routing consistently achieves superior
reconstruction performance… experts learn sub-categorical structures that transcend human-defined
class boundaries")

**Question**: Are there meaningful sub-categories *within* our human-defined task families that
routing discovers but we haven't labelled?

**Motivation**: Our 9-family taxonomy is human-imposed.  The HMOE and Nikolic results suggest
routing may identify finer-grained structure — e.g., within "arithmetic," routing might separate
word-problem arithmetic from symbolic arithmetic; within "coreference," it might separate
subject-role ambiguity from object-role ambiguity.

**Measurement**:
1. Run k-means on routing-weight vectors with k > 9 (try k = 12, 15, 20)
2. Inspect which prompts fall into "extra" clusters that split a single task family
3. For each extra cluster, compute the routing-weight centroid — what is distinctive about
   this sub-group's routing pattern?
4. Run HMOE's "interpretability check": for each cluster, what is the most human-interpretable
   description of the shared property?

**Expected finding**: Within code_completion, functional-style code (no class/state) and
OOP-style code (class definitions, self.x) may route differently.  Within arithmetic, digit
tokens route differently from word-form arithmetic tokens.  Within analogy, relational analogies
("King : Queen :: Man : ?") may route differently from categorical analogies
("Paris : France :: Rome : ?").

**Tooling**: `routing_patterns.json`, `sklearn`, manual annotation of cluster members
**New model runs required**: None (for k-means); optional (more diverse prompts to populate
sub-categories)
**Expected effort**: 1–2 days

---

#### Thread 32 — Token-Surface vs. Semantic Routing Crossover

**Core reference**: OpenMoE (2024): routing is "predominantly based on token IDs, with minimal
context relevance"; also Section 4d of Part III above.

**Question**: At what depth does the routing signal transition from token-surface-driven
to task-context-driven?  Our MI(expert; task) data partially answers this, but we haven't
measured the complementary quantity `I(expert; token_BPE_id)`.

**Measurement**:
1. For each token in our 135 prompts, record: (a) BPE token id, (b) task family,
   (c) routing decision at each layer
2. Compute `I(expert_l; token_BPE_id)` — how much does knowing the BPE token predict
   the routing decision at each layer?
3. Compute `I(expert_l; task_family)` — already in `routing_mi_task.json`
4. Plot both curves on the same axis across 24 layers
5. Identify the crossover depth: `min l s.t. I(expert_l; task) > I(expert_l; token_surface)`

**Expected finding (from OpenMoE and our L12 MI result)**: `I(expert; token_surface)` is
high in L0–8 and declines slowly.  `I(expert; task)` is low in L0–8 and peaks at L12.
The crossover (if it exists) is likely in L8–12 — this would be the depth at which routing
shifts from surface-form partitioning to semantic partitioning.  If no crossover exists —
if `I(expert; token_surface) > I(expert; task)` at all layers — routing is predominantly
a surface tokenizer throughout the network, which is the OpenMoE conclusion and implies
the L12 task MI peak is driven by surface-form correlates of task identity (e.g., question
marks for arithmetic prompts, colons for syntax prompts).

**Tooling**: `routing_patterns.json`, `tokenizer.encode()` to recover BPE ids
**New model runs required**: None (routing data on disk; BPE ids from tokenizer)
**Expected effort**: 1 day

---

### Literature citations to add to Thread 15 README

The following papers are directly relevant to Thread 15 findings and should be cited:

| Result | Paper | Key claim |
|---|---|---|
| Routing encodes task structure | Li & Zhou, ICLR 2025 | Routing weights are off-the-shelf semantic embeddings |
| L12 MI peak (early routing commitment) | Xue et al. (OpenMoE), ICML 2024 | "Early Routing Learning": assignments fixed early in training |
| Code cluster separation | Zoph et al. (ST-MoE), 2022 | Experts specialize by syntactic/surface token category |
| Analogy/coref/recency grouping | Qu et al. (HMOE), 2022 | Routing discovers clusters more coherent than human labels |
| Sub-uniform routing all layers | Chen et al., NeurIPS 2022 | Gradient descent causes routers to recover cluster-center features |
| 4.4% capacity budget | Kawata et al., ICML 2025 | MoE detects latent structure that dense networks cannot |

---

## Part IV — Full Summary Table

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

| 25 | Routing specialization gain trajectory | Analysis (free) | 0.5 day | **High** |
| 26 | Cross-task JSD distinguishability matrix | Measurement | 1 day | **High** |
| 27 | Task routing mutual information curve | Measurement | 1 day | **High** |
| 28 | Routing velocity and phase transitions | Measurement | 1 day | High |
| 29 | Routing capacity budget decomposition | Measurement | 2 days | Medium |
| 30 | Routing-weight embeddings and task geometry | Analysis (free) | 1 day | **High** |
| 31 | Sub-categorical structure discovery | Analysis (free) | 1–2 days | High |
| 32 | Token-surface vs. semantic routing crossover | Analysis (free) | 1 day | **High** |

**Immediate recommendations**:
1. Add Component J (logit delta) to `gossh/features/extractor.py` — it's free and directly connects Thread 9 to Thread 15
2. Add Components K and L (co-activation + persistence) — also free from existing routing data
3. Run Thread 30 next — UMAP + dendrogram on `routing_patterns.json` requires zero new model runs, produces publication-quality figures, and directly connects to the Li & Zhou ICLR 2025 result
4. Run Thread 32 in parallel — BPE-id MI vs. task-family MI crossover depth directly tests the OpenMoE "context-independent routing" claim on gpt-oss-20b
5. Run Thread 18 — late-layer expert identity for the clean analysis set, directly strengthens the Thread 2 causal bottleneck result
