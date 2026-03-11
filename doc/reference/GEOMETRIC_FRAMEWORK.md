# Geometric Framework: Feature Maps as Dual Structure on Models and Data

## 1. The Core Mathematical Object

The NeurIPS paper's Tier-2 feature extraction defines a map:

```
φ_M : Tokens-in-Context → ℝ^D
```

For a fixed model M, each token t (in its context) maps to a D-dimensional vector capturing *how* the model processes it. Three invariances govern the feature design:

1. **Token-identity invariance**: Features capture how the model processed, not what it processed. Two predictions with identical computational signatures cluster together whether the context contains "cat" or "dog."

2. **Position permutation invariance**: Features don't depend on absolute sequence position. Achieved via max-pooling over positions (discards which position achieved max attention), entropy (aggregates the full distribution), and relative distances (attention span, not absolute indices).

3. **Topology preservation**: Layer ordering is preserved as ordinal structure. The stability layer k* is a scalar, not one-hot, so layer 1 is "closer" to layer 2 than to layer 5.

**A critical distinction**: The paper achieves invariance over *token positions* — the same computation at different positions produces the same feature vector. It does NOT achieve invariance over *head labels* — Components C and D store per-head values indexed by `i = layer × H + head`. Head 3's activation occupies a specific coordinate. This asymmetry has consequences for within-model vs. cross-model analysis (see §1.2, §1.3).

**But this map has richer structure than the paper exploits.** It is not just a featurization for clustering — it is a geometric morphism that simultaneously equips both the data and the model's components with natural metric structure.

### 1.1 The Pullback Metric on Data

For a fixed model M, the feature map induces a pseudometric on the input:

```
d_M(t₁, t₂) = ‖φ_M(t₁) - φ_M(t₂)‖
```

Two tokens are "close" when the model processes them the same way — regardless of what they are. This is the ARI=0.008 result: the pullback metric is orthogonal to the vocabulary metric. The computational metric and the semantic metric are (approximately) independent.

**Key property**: This metric is invariant to head relabeling. If you simultaneously permute all head indices (swap head 3 and head 5 throughout), both φ(t₁) and φ(t₂) undergo the same coordinate permutation, and the L2 distance is unchanged. Position permutation invariance (from the feature design) and head-label invariance (from L2) together make d_M a well-defined metric on *computational equivalence classes* — tokens with identical attention patterns and logit trajectories are at distance zero, regardless of where they appear or how heads are labeled.

This pullback metric turns the token stream into a **metric space whose geometry reflects the model's computation**. Clustering is one way to analyze this metric space — but it's the coarsest. The metric itself carries finer information:
- **Local density**: Regions of feature space where many tokens cluster tightly = stereotyped, well-practiced computation
- **Sparse regions**: Isolated points in feature space = unusual computational patterns, candidates for novel modes
- **Geodesic structure**: Continuous paths through feature space = computational "phase transitions" as context changes

### 1.2 The Dual Metric on Model Components

The same feature vectors also induce structure on the model's components. Consider the head activation components (C, D in the paper). For each head h within a fixed model M, define:

```
ψ_h : Corpus → ℝ^(2·|Corpus|)    [activation and entropy across all tokens]
```

Now define:

```
d(h₁, h₂) = ‖ψ_{h₁} - ψ_{h₂}‖    [over a reference corpus]
```

This puts a metric on attention heads *within a single model*. Heads that are "close" activate similarly across diverse inputs — they perform the same *computational role*.

**Important caveat**: This is an *observational* metric — it captures co-activation patterns, not causal importance. Two heads could activate identically yet serve different causal roles (one might be compensated by other heads under ablation, the other might not). The dual metric generates *hypotheses* about functional equivalence that ablation experiments can test, but it does not replace them. The paper's ablation-effect variance measurements (§3d in the experimental plan) provide the interventional complement.

**For MoE experts, this becomes especially powerful**: each expert has a routing pattern across the corpus, and experts that route similarly form natural functional groups. The expert dual metric is well-defined because expert labels within a model are consistent (though, like head labels, they are arbitrary and carry no meaning across models).

### 1.3 Cross-Model Comparison: What Is and Isn't Shared

The feature map is parameterized by the model:

```
φ_M : Tokens-in-Context → ℝ^{D(M)}
```

where D(M) depends on the model's architecture (L, H, and whether it has MoE). A 6-layer, 6-head model produces 163D vectors; a 24-layer, 64-head MoE model produces ~7,200D vectors. **The feature vectors from different models do not live in the same space.** Head 3 of model A bears no relation to head 3 of model B.

**What IS comparable across models is the pullback metric on shared data.** Both models process the same tokens, and each induces its own metric d_M on those tokens. Cross-model comparison operates at the level of these metric spaces:

- **Gromov-Hausdorff**: Do the two models induce similar metric structures on the same data? d_{M₁}(t₁, t₂) ≈ d_{M₂}(t₁, t₂)?
- **Cluster agreement (ARI)**: Do the two models' feature clusterings partition the same data similarly?
- **Rank correlation**: Do the two models agree on which token pairs are computationally similar vs. dissimilar?

None of these require a shared coordinate system. They compare *metric structures* and *partition structures*, not feature vectors.

This is the mathematical reason the same feature *methodology* (not the same feature *space*) can compare PLS-trained symbolic transformers with standard-trained gpt-oss-20b: **both models induce metrics on a shared dataset, and those metrics are comparable even though the feature coordinates are model-specific.**

Two models are "similar" if they induce similar metrics on the same data. Two data points are "similar" if a family of models processes them alike. This is a Gromov-Hausdorff perspective, not an RKHS one — the comparison is between metric spaces, not between points in a shared Hilbert space.

---

## 2. Decomposition of the Feature Space

The 163-dimensional feature vector is not a monolithic object. It decomposes into functionally distinct subspaces:

```
ℝ^D = ℝ^(3L-1) × ℝ^2 × ℝ^(2LH) × ℝ^(2LH)
        │           │       │            │
    Trajectory   Stability  Activation   Entropy
    (prediction  (depth)    (what fires) (how sharply)
     dynamics)
```

Each subspace captures a different aspect of computation:

| Subspace | Geometry | What it measures |
| --- | --- | --- |
| Trajectory (A) | Probability simplex paths | How confidence evolves across layers |
| Stability (B) | Ordinal line segment [0, L] | When computation "commits" |
| Activation (C) | Product of [0,1] intervals | Which heads contribute |
| Entropy (D) | Positive reals | How broadly attention distributes |

**Key insight**: These subspaces are not fully independent. The stability layer k* determines *which* head features are populated in Components C and D. The feature layout stores head activation/entropy at two layers: the stability layer k* and the final layer L-1. Head features at other layers are zero. This means k* acts as a selector: it determines which slice of the LH-dimensional head-activation space carries signal.

This creates a **stratified structure** (not a fiber bundle in the strict sense, since the feature vector includes both k*-dependent and k*-independent components):

```
Stability layer k* ∈ {0, 1, ..., L}     [stratification variable]
        │
        ▼
  Head features at k* AND at L-1         [populated subspace depends on k*]
  Trajectory features (all layers)        [independent of k*]
```

The effect is that tokens at the same depth k* have feature vectors that are structurally comparable in the head-activation components (both populate the same coordinates), while tokens at different depths populate different coordinates and are therefore naturally separated in Euclidean distance — even if their head patterns are qualitatively similar.

**Why this matters for clustering**: HDBSCAN exploits this stratification because the density structure reflects it. Tokens at the same depth and with similar head patterns cluster tightly; tokens at different depths are geometrically distant by construction. This is a design consequence of the feature layout, not a discovered property — it's built into the coordinates.

---

## 3. Extension to gpt-oss-20b

### 3.1 Expanded Feature Space

gpt-oss-20b (L=24, H_q=64 query heads, H_kv=8 KV heads, E=32 experts, top-4 routing) expands each component:

| Component | Symbolic (L=6, H=6) | gpt-oss-20b | Notes |
| --- | ---: | ---: | --- |
| A. Trajectory | 17 | 71 | 3×24 - 1 |
| B. Stability | 2 | 2 | Same |
| C. Head Activation | 72 | 3,072 | 2 × 24 × 64 |
| D. Head Entropy | 72 | 3,072 | 2 × 24 × 64 |
| **Subtotal** | **163** | **6,217** | |
| **E. Expert Routing** | — | ~960 | New: see below |
| **F. Routing Entropy** | — | 24 | New: per-layer routing diversity |
| **G. Attention Scale** | — | 48 | New: sliding vs. full contrast |
| **Total** | **163** | **~7,249** | Before dimensionality reduction |

### 3.2 New Components for MoE

**Component E: Expert Routing Features**

For each MoE layer ℓ, the top-4 router selects 4 of 32 experts with weights:

```
r^(ℓ) = (e₁, w₁, e₂, w₂, e₃, w₃, e₄, w₄)    [4 expert indices + 4 weights]
```

But expert indices are categorical, not ordinal. We can't use them as raw features (expert 17 is not "between" expert 16 and 18). Instead:

**Option A: Binary routing indicator (32 dims per layer)**
```
R^(ℓ) ∈ {0,1}^32    [R^(ℓ)_e = 1 if expert e is selected]
```
This gives 24 × 32 = 768 dimensions. Sparse (4/32 nonzero per layer). Permutation-invariant over expert ordering.

**Option B: Weighted routing vector (32 dims per layer)**
```
R^(ℓ) ∈ [0,1]^32    [R^(ℓ)_e = weight if expert e selected, else 0]
```
Same dimensionality, but preserves weight information. The gate weights live on a simplex (they sum to 1 within each layer), adding a constraint geometry.

**Option C: Routing at stability and final layers only (64 dims)**
```
R_stable, R_final ∈ [0,1]^32
```
Mirrors the head activation/entropy design: capture routing at the two most informative layers.

**Recommendation**: Option B at stability + final layers (64 dims), plus sorted top-4 weights at all layers (24 × 4 = 96 dims). Total: 160 dims for routing. This balances expressiveness with dimensionality.

**Component F: Routing Entropy**

Per-layer routing entropy measures how "unusual" the routing is:

```
H_route^(ℓ) = -Σ_e w_e log(w_e)    [entropy of the top-4 weights]
```

High entropy = weights spread evenly across 4 experts (model "uncertain" about routing).
Low entropy = one expert dominates (specialized computation).

This is 24 dimensions and analogous to head entropy.

**Component G: Attention Scale Features**

gpt-oss-20b alternates sliding-window (128 tokens) and full attention. This creates a natural contrast:

```
local_fraction^(ℓ) = attention mass in [t-128, t] / total attention mass
```

For sliding layers, this is always 1.0 (by construction). For full layers, it measures how much the model looks beyond the local window. The contrast between adjacent layers captures whether the model uses the global context.

This is 24 dimensions but only 12 are informative (full-attention layers). In practice, store only the 12 full-attention values + 12 sliding-attention values to capture the local/global balance.

### 3.3 The MoE Routing Geometry

The expert routing introduces structure not present in the symbolic transformer: a **discrete combinatorial layer** on top of the continuous feature space.

For each token, the router selects C(32,4) = 35,960 possible expert combinations. This defines a combinatorial structure:

```
Routing space = Δ₃ × C(32,4)
                │        │
          Weight     Expert
          simplex    selection
```

The weight simplex Δ₃ is a continuous 3-dimensional manifold. The expert selection is a discrete set. Together they form a **stratified space** where each stratum (expert combination) carries a simplex of possible weight configurations.

**Implication for computational modes**: Two tokens that select the same 4 experts are in the same stratum — they share a computational pathway. The within-stratum variation (different weights on the same experts) represents fine-grained modulation. The between-stratum variation (different experts) represents qualitatively different computation.

This stratification is *exactly* the kind of structure PLS is designed to expose. In a standard-trained model (like gpt-oss-20b), the Hydra effect may smear routing decisions across many strata, making the stratification indistinct. In a PLS-trained model, we'd expect routing to concentrate into fewer, sharper strata.

---

## 4. Inspectability as Geometric Property

### 4.1 Definition

A model is **inspectable** at input x if:
1. The feature vector φ_M(x) lies in a well-defined cluster (high local density)
2. The cluster has low intrinsic dimension (consistent computational mode)
3. The cluster members respond coherently to interventions (causal consistency)

These three conditions are ordered by cost. Conditions (1) and (2) are computable from feature extraction alone (single forward pass per input). Condition (3) requires ablation experiments — it's the interventional complement to the observational features.

**Operationally**: Conditions (1) and (2) define a *candidate inspectability score* that can be computed cheaply. Condition (3) validates whether the observational score predicts actual causal regularity. The key empirical question is whether (1)+(2) correlate with (3) — whether tokens in dense, low-dimensional clusters actually respond coherently to ablation. If they do, the cheap score becomes a reliable proxy for the expensive one.

We deliberately avoid combining these into a single formula (e.g., multiplying density × 1/dim × coherence). The functional relationship between these quantities is an empirical question, not something to be assumed.

### 4.2 Inspectability Corpus Refinement

The convergence layer from logit-lens is a 1D projection of inspectability:
- Early convergence → high density in "shallow processing" region → likely inspectable
- Late convergence → may be in a sparse, high-dimensional region → harder to inspect

But the full feature vector provides a much richer signal. The protocol:

1. Run feature extraction on large corpus (thousands of prompts)
2. Compute per-token feature vectors
3. Fit HDBSCAN — tokens assigned to clusters are "inspectable"
4. Rank clusters by (a) within-cluster variance, (b) cluster size, (c) ablation coherence
5. Select representative tokens from top clusters as the inspectability corpus

**This is methodologically novel**: instead of choosing prompts by heuristic (Winograd, IOI, etc.), you let the model's own computational structure select the prompts where it's most transparent.

### 4.3 Uninspectable Regions

Equally valuable: identify tokens where φ_M(x) is an outlier (HDBSCAN noise point). These are tokens the model processes in an unusual way — candidates for novel computational modes, or for pathological behavior.

**For gpt-oss-20b**: The MoE routing adds an extra dimension of uninspectability. A token that routes to an unusual expert combination (low probability under the corpus-wide routing distribution) is computationally anomalous even if its attention pattern is normal.

---

## 5. Distillation as Metric Preservation

### 5.1 The Core Idea

Standard distillation matches outputs: P_student(y|x) ≈ P_teacher(y|x).

**Computational mode distillation** matches the *geometry*:

```
d_{M_student}(t₁, t₂) ≈ d_{M_teacher}(t₁, t₂)    for all token pairs
```

This is a Gromov-Hausdorff condition: the student model must induce approximately the same metric on the data as the teacher.

### 5.2 Why Naive Layer Matching Fails

The obvious approach — match teacher layer ℓ to student layer π(ℓ) via KL divergence — does not work across architectures. The literature (OFA-KD NeurIPS 2024, MOHAWK NeurIPS 2024, TMLR 2025 survey) confirms:
- CKA analysis shows significant feature divergence between heterogeneous architectures
- Proportional layer mapping (teacher layer 18/24 ≈ student layer 4.5/6) assumes a correspondence that doesn't exist
- Even learned projection/stitching layers have limited benefit when information flow patterns differ fundamentally
- The student may need to organize computation completely differently to fit in fewer layers

See INTERMEDIATE_LAYERS_ANALYSIS.md and references/DISTILLATION_LITERATURE.md for the full argument.

### 5.3 CASCADE Distillation: The Linear Structure

In the symbolic transformer's CASCADE mode, x_t (token embedding) is frozen and W (lm_head) is known. The teacher's output distribution defines a closed-form regression target:

```
x_e* = W⁺ · (z_teacher - W · x_t)
```

This reduces distillation to structured matrix factorization — the teacher's word-context probability matrix factors through the d-dimensional x_e bottleneck. This is LSA / topic modeling in a modern context: the computational modes are the latent topics. See CASCADE_DISTILLATION.md for the full derivation.

### 5.4 Using Intermediate Layers Without Layer Correspondence

The teacher's per-layer logit-lens readouts provide useful information beyond the final output, but NOT as layer-matched supervision. Three defensible uses:

1. **Hard negatives**: Words considered at intermediate layers but rejected by the final layer
2. **Decision subspace**: PCA of layer-to-layer contrast vectors → the directions of semantic decisions
3. **Unordered decision constraints**: Each contrast is a semantic decision the student must make somewhere

Whether these help beyond output-only distillation is an empirical question (estimated 50/50). See INTERMEDIATE_LAYERS_ANALYSIS.md.

### 5.5 Metric Preservation Loss

The contrastive formulation preserves the teacher's computational mode structure without layer correspondence. The teacher's mode assignments (from clustering the teacher's features) define which tokens should be "close" or "far" in the student's *own* feature space:

- **Positive pairs**: Tokens in the same teacher mode should remain close in the student's pullback metric
- **Negative pairs**: Tokens in different teacher modes should remain far in the student's pullback metric

```
L_contrastive = Σ triplet(anchor, positive, negative) with margin
```

where distances are computed in the student's feature space, but positive/negative sampling uses the teacher's cluster assignments. This is computationally feasible via within/across cluster sampling from the teacher's Tier-2 feature clustering. Note that the student's features have different dimensionality than the teacher's — the contrastive loss operates within the student's space using the teacher's partition as supervision.

### 5.6 Evaluation

A model is well-distilled when the teacher's and student's feature point clouds (on the same corpus) have similar structure:
- High ARI between teacher and student computational mode clusterings
- Similar intrinsic dimensionality
- Steering vectors extracted from the teacher's computation trajectory also work on the student

---

## 6. Concrete Experimental Plan for gpt-oss-20b

### Phase 1: Build the Extended Feature Extractor (days)

**Goal**: Port TensorFeatureExtractor to consume gpt-oss-interp's activation cache and logit-lens outputs.

**What exists**:
- `gpt_oss_interp.capture.activation_cache.ActivationCache` — captures per-layer hidden states
- `gpt_oss_interp.readouts.logit_lens.run_logit_lens` — per-layer token predictions
- `TensorFeatureExtractor` — computes Tier-2 features from layer logits and layer attentions

**What's needed**:
1. Adapter: Convert `ActivationCache` records + `LogitLensResult` into the tensor format TensorFeatureExtractor expects
2. Attention capture: Add attention-weight hooks to the backend (currently not captured)
3. Config adaptation: Set n_layers=24, n_heads=64 (query heads for GQA)
4. GQA handling: 64 query heads share 8 KV heads — decide whether to use query-head attention (64D) or KV-head attention (8D) or both

**GQA Decision**: Use query-head attention patterns (64 heads). The query heads contain the fine-grained attention information; KV heads are the compressed version. For feature extraction, we want the richest signal.

**Deliverable**: `gpt_oss_interp/features/extractor.py` that takes a prompt and returns a feature vector per token.

### Phase 2: Corpus Feature Extraction (days, needs GPU)

**Goal**: Extract feature vectors for a diverse corpus and analyze the resulting geometry.

**Corpus**: 200-500 prompts covering:
- Simple factual completions (low depth expected)
- Syntactic agreement across attractors (medium depth)
- Coreference resolution (variable depth)
- Induction/copying (late depth)
- Ambiguous/adversarial prompts (unknown depth)

**Analysis**:
1. PCA of the ~7,000D feature space → identify dominant variance directions
2. UMAP visualization colored by (a) processing depth, (b) expert routing cluster, (c) attention pattern type
3. HDBSCAN clustering → discover computational modes
4. Compare mode structure to the symbolic transformer's modes via ARI on shared prompts (different feature spaces, same data — see §1.3)

### Phase 3: Geometric Analysis (weeks)

**Goal**: Characterize the metric structure that φ induces on gpt-oss-20b's inputs.

**Experiments**:

**3a. Metric dimensionality**: What is the intrinsic dimensionality of the feature point cloud? Use PCA eigenvalue decay, correlation dimension, or persistent homology. If it's much lower than 7,000 → the model has a low-dimensional computational structure despite its high parameter count.

**3b. Stratification by routing**: Does expert routing create discrete strata in feature space? Compute within-stratum vs. between-stratum variance. If the stratification is clean → MoE routing creates genuinely distinct computational pathways.

**3c. Sliding vs. full attention geometry**: Do sliding-attention and full-attention layers create distinct geometric features? Compute mutual information between attention-scale features (Component G) and computational mode assignment.

**3d. Hydra effect as metric property**: The Hydra effect (tight ablation variance) corresponds to a metric property: under ablation, the point cloud *barely moves* in feature space. Measure:
```
Δ_ablation = ‖φ_M(x) - φ_{M\setminus h}(x)‖    [feature distance under head ablation]
```
If Δ_ablation is uniformly small across all heads → Hydra is active (distributed redundancy). If Δ_ablation is large for specific heads → those heads are causally important and detectable without exhaustive ablation.

### Phase 4: Distillation Prototype (weeks)

**Goal**: Train a symbolic transformer that preserves gpt-oss-20b's computational geometry.

**Architecture**:
- 12-layer, 12-head dual-stream symbolic transformer (~80-126M params)
- CASCADE mode (frozen x_t), gated attention, PLS
- Vocabulary projection from teacher (201K) to student (50K) vocabulary

**Training** (staged, see CASCADE_DISTILLATION.md):
1. Compute closed-form x_e targets from teacher's final distribution via pseudoinverse
2. Regression warmup: train FFN pathway to predict x_e* targets
3. KL refinement: fine-tune with actual distributional loss (handles softmax nonlinearity)
4. Optional: add hard-negative contrastive loss from intermediate layers; add decision subspace regularization
5. Optional: add contrastive mode-preservation loss (L_contrastive on Tier-2 features)

**Evaluation**:
- Feature-space ARI between teacher and student computational mode clusterings
- Ablation-effect variance (should be higher than same architecture trained from scratch, due to mode-preserving distillation + PLS)
- Whether steering vectors from teacher's computation trajectory transfer to student
- Ordering of prediction difficulty preserved (inputs the teacher finds easy should be easy for the student)

---

## 7. What This Framework Reveals That Current Methods Don't

### 7.1 vs. Probing Classifiers

Probing asks: "Can we decode property X from layer ℓ?" This is a supervised question — you need to know what X is in advance. The feature map asks: "What computational structure exists?" This is unsupervised — it discovers X.

### 7.2 vs. Sparse Autoencoders (SAEs)

SAEs decompose activations into interpretable directions. They operate in the residual stream, which is dominated by token embeddings (the ARI=0.008 result shows that clustering raw activations recovers vocabulary structure, not computational modes).

The feature map's relationship to vocabulary is more nuanced than "orthogonal." Component A (trajectory) is defined on the probability simplex over vocabulary — it directly involves token probabilities. Components C and D (head activation, head entropy) are genuinely orthogonal to vocabulary: they measure attention pattern properties that are independent of what tokens are being attended to. The ARI=0.008 result reflects the dominance of C and D in the feature space (144 of 163 dimensions), not a blanket orthogonality claim.

SAE features and computational-mode features should be complementary. SAEs find *what* the model represents (interpretable directions in activation space); computational modes find *how* it processes (attention patterns, convergence dynamics, routing). A full picture needs both.

### 7.3 vs. Circuit Discovery (ACDC, etc.)

Circuit discovery finds minimal subgraphs that implement a behavior. This requires (a) specifying the behavior in advance and (b) exhaustive ablation search. The feature map discovers behaviors from the model's own computation — and the cluster structure *predicts* which components matter, reducing the ablation search space.

**Concrete speedup (hypothesis)**: Instead of ablating every head on every task, ablate only the heads that are distinctive in the cluster's feature vector. If a cluster has high activation at heads [L20H3, L20H17, L21H42], those are the candidates for causal importance. Whether high feature-activation predicts high causal importance is testable: compare feature-predicted importance rankings to actual ablation-effect rankings on the same inputs.

### 7.4 The Unique Contribution

No existing method provides **unsupervised, geometric comparison** of how different models compute on the same data. The feature methodology does — not by placing models in a shared coordinate system, but by comparing the metric structures they induce on shared inputs (§1.3). This enables:
- Comparing PLS-trained vs. standard-trained models via ARI and GH distance on shared corpora
- Comparing models of radically different scale (6-layer symbolic vs. 24-layer MoE) at the level of computational mode partitions
- Discovering that two architectures implement the "same" computation differently (same cluster structure, different head allocation)
- Guiding distillation by matching computational structure, not just outputs

**Scope**: The feature methodology applies to any model that produces per-layer logits (or logit-lens readouts) and attention weights. Components A-D require only these. Components E-G (routing, routing entropy, attention scale) are MoE-specific — they extend the methodology to MoE architectures but are not architecture-agnostic.

---

## 8. Connection to Broader Mathematical Frameworks

### 8.1 Information Geometry

The per-layer probability trajectories (Component A) trace paths on the probability simplex Δ^(V-1). The Fisher-Rao metric on this simplex provides a natural geometry for comparing trajectories. Two tokens whose probability paths diverge early (in Fisher-Rao distance) are making fundamentally different computational moves.

### 8.2 Persistent Homology

The feature point cloud can be analyzed via persistent homology to detect topological features (loops, voids) in computational mode space. The persistence diagram summarizes the multi-scale structure: short-lived features are noise, long-lived features are robust topological properties. This is a standard TDA technique for point cloud analysis and requires no transformer-specific interpretation — it simply characterizes the shape of the feature distribution. Whether any persistent topological features have interpretable meaning for the model's computation is an empirical question.

### 8.3 Comparing Feature Distributions Across Models

Since teacher and student feature vectors live in different spaces (§1.3), pointwise comparison ‖φ_T(x) - φ_S(x)‖ is not meaningful. Two approaches to comparing feature *distributions*:

**Approach 1: Gromov-Hausdorff on pullback metrics.** Compare the metric spaces (X, d_{M_T}) and (X, d_{M_S}) induced on shared data X. This captures whether the two models agree on which inputs are computationally similar.

**Approach 2: Compare cluster-level statistics.** Cluster each model's features independently, then compare via ARI (do the partitions agree?), cluster count, within-cluster variance, and cluster size distributions. This is coarser than GH but more robust and directly interpretable.

Optimal transport (Wasserstein distance) would be applicable if the feature spaces were shared or if a correspondence between them were established. In the current setting — different D, different coordinate meanings — OT requires either dimensionality reduction to a common space or working through the pullback metrics.

### 8.4 Representation Stability

The feature map defines a function from model parameters to metric spaces on data. For the analysis to be robust, small parameter perturbations should produce small changes in the induced metric. Some feature components satisfy this naturally — head activation (Component C) is a continuous function of attention weights. Others do not: the stability layer k* is a discrete argmax, so a small weight change can shift k* from layer 18 to 17, causing a discontinuous jump in the feature vector and changing which coordinates are populated in Components C and D.

This means the pullback metric d_M is continuous in model parameters *except at boundaries where k* changes*. At those boundaries, tokens can jump between clusters. This is not a flaw — it reflects genuine computational phase transitions — but it means stability guarantees apply generically (almost everywhere in parameter space), not universally.

---

## 9. Why This Matters for OpenAI's Interpretability Agenda

OpenAI's core challenge is understanding frontier models (GPT-4, o-series) that are too large for exhaustive circuit analysis. The geometric framework offers:

1. **Scalable mode discovery**: Feature extraction is a single forward pass per prompt. The observational analysis (clustering, metric structure, intrinsic dimensionality) requires no ablation. Causal validation of discovered modes does require ablation, but the feature analysis *focuses* the ablation search — instead of testing all heads on all inputs, test only the heads that are distinctive for each cluster (§7.3). Cost: O(corpus × sequence_length) for discovery, O(clusters × distinctive_heads) for validation.

2. **Cross-model comparison via shared data**: The same feature *methodology* (not the same feature space) works on any model with per-layer logits and attention weights. Models produce feature vectors of different dimensionality, but the pullback metrics and cluster partitions on shared data are directly comparable (§1.3). This enables studying how computational mode structure evolves with scale — does a larger model have more modes, sharper modes, or qualitatively different modes?

3. **Distillation for interpretability**: If you can distill a frontier model into a smaller one that preserves computational geometry (measured by ARI and GH distance on shared corpora), you can study the small model to understand the large one. The metric comparison tells you *which aspects of computation* were preserved and which were lost.

4. **Quantitative interpretability metrics**: Inspectability decomposes into an observational component (cluster density, intrinsic dimension — cheap) and a causal component (ablation coherence — expensive). Whether the cheap component predicts the expensive one is a testable hypothesis (§4.1). If it does, inspectability becomes a scalable, quantitative metric.

5. **Depth-stratified analysis**: The feature layout stratifies tokens by processing depth (§2). Within each stratum, head activation patterns characterize the computational mode. Understanding how shallow-processing modes differ from deep-processing modes gives a natural decomposition of the model's computational repertoire.
