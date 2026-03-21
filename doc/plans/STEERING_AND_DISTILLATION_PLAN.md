# Steering Vectors in the Symbolic Transformer + Correct Distillation from gpt-oss-20b

## Part 1: Steering Vectors in the Symbolic Transformer

### 1.1 Why the Symbolic Transformer Is Uniquely Suited for Steering Research

The standard steering vector story (CAA, RepE) operates on the residual stream — add a direction, shift behavior. The failure modes are severe: ~50% anti-steerability on many concepts (Durmus et al., NeurIPS 2024), bimodal response distributions, capability degradation up to 53%.

The symbolic transformer has three properties that could fundamentally change this:

**Property 1: Dual-stream decomposition separates steerable from non-steerable computation.**

In a standard transformer, the residual stream mixes everything — token identity, context, syntax, semantics. A steering vector applied there inevitably corrupts multiple computational channels. In the symbolic transformer:
- **x_t** (token stream) carries token-identity structure — stable, position-independent
- **x_e** (context stream) carries accumulated contextual computation — dynamic, context-dependent

A steering vector applied to x_e modulates *how* the model contextualizes, without corrupting *what* tokens it's looking at. Applied to x_t, it would shift the token-identity landscape. These are categorically different interventions.

**Hypothesis**: Steering x_e alone should have higher steerability rates and lower anti-steerability than steering the combined residual stream, because x_e is the stream that carries behavioral decisions while x_t provides stable reference structure.

**Property 2: Gated attention provides per-head steering control.**

The gated attention mechanism computes:
```
gate_logits = einsum('bhtd,hde->bhte', gate_input, gate_weight)
gate = sigmoid(gate_logits)
y = y * gate
```

This is already a *learned modulation* of information flow per head. A steering vector that targets the gate input space can amplify or suppress specific heads without touching the attention computation itself. This is more surgical than residual-stream steering.

**Concrete mechanism**: Instead of adding a vector to the residual stream, *multiply* the gate values by a steering mask:
```
gate_steered = gate * (1 + α · steering_mask)
```
where `steering_mask ∈ ℝ^(n_head × head_dim)` is computed from contrast pairs via the gate-input projections. This steers *which heads contribute* rather than *what the residual stream contains*.

**Property 3: PLS makes steering effects predictable.**

In a standard model (Hydra active), steering at one layer gets compensated by redundant computation elsewhere. That's why anti-steerability exists — the model routes around the perturbation. In a PLS-trained model:
- Computation is modular (5-23× larger ablation effects)
- Head specialization is task-specific (diagonal sparsity in the ablation matrix)
- There's no redundant pathway to compensate

**Prediction**: PLS-trained models should show higher steerability AND lower anti-steerability than standard models, because there's less compensatory circuitry.

### 1.2 Concrete Steering Experiments

**Experiment 1: Stream-selective steering**

Compute steering vectors the standard way (mean-difference on contrast pairs), but apply them selectively:
- (a) Add to combined stream `x_t + x_e` (baseline — equivalent to standard steering)
- (b) Add to `x_e` only (contextual steering)
- (c) Add to `x_t` only (token-identity steering)
- (d) Add to `x_e` with CASCADE mode active (x_t frozen — purest contextual steering)

Measure: steerability rate, anti-steerability rate, off-target capability degradation.

**Experiment 2: Gate-mediated steering**

Compute contrast pairs in gate-input space:
1. For each head h at layer ℓ, collect gate activations on positive and negative examples
2. Compute mean gate difference: `Δg_h = mean(gate_pos) - mean(gate_neg)`
3. At inference, multiply gate values: `gate_h *= (1 + α · Δg_h / ||Δg_h||)`

This steers by modulating which heads contribute, preserving the attention computation within each head.

Compare to residual-stream steering on the same tasks.

**Experiment 3: PLS vs. standard training and steerability**

Train two identical architectures:
- (a) Standard training (single final-layer CE loss)
- (b) PLS training (per-layer supervision)

Apply identical steering vectors to both. Measure steerability, anti-steerability, and the key prediction: PLS should be more steerable because it lacks compensatory circuits.

**Experiment 4: Feature-guided steering**

Use the Tier-2 feature extraction to identify computational modes. For each mode:
1. Collect the tokens/inputs that belong to that mode
2. Identify which heads are distinctive for that mode (high feature values in Components C, D)
3. Compute a mode-specific steering vector using only the distinctive heads' representations
4. Apply steering that targets this specific mode

This connects the feature geometry to steering: the computational mode decomposition tells you *where* to steer, not just *what direction*.

### 1.3 What This Contributes

No one has studied how architectural choices (dual-stream, gated attention, per-layer supervision) affect steerability. This is new territory:
- If stream-selective steering reduces anti-steerability → dual-stream architectures are "steering-aware" by design
- If gate-mediated steering outperforms residual-stream steering → gated attention provides a natural interface for behavioral control
- If PLS improves steerability → modular training objectives create models that are not just more interpretable but more *controllable*

This directly addresses the Durmus et al. failure modes: the ~50% anti-steerability problem may be an artifact of standard architectures, not a fundamental limitation.

---

## Part 2: Correct Distillation from gpt-oss-20b to Symbolic Transformer

### 2.1 Why Naive Approaches Fail

You're right that simple paired-output matching doesn't work. The literature confirms this empirically:

1. **CKA analysis** (OFA-KD, IJCAI 2024) shows significant feature divergence between heterogeneous architectures — the representation spaces are not aligned, so L2/KL losses on raw features are meaningless.
2. **FitNets** (Romero 2015) requires at minimum a learned projection (stitching) layer between teacher and student representations, and this only works within the same architecture family.
3. **MOHAWK** (NeurIPS 2024) — the most successful cross-architecture distillation — uses a three-stage pipeline with learned matrix alignment, not direct feature matching.

The gpt-oss-20b → symbolic transformer case is particularly hard:
- **Scale mismatch**: 24 layers → 6 or 12 layers. 64 heads → 6 or 12 heads. 21B params → 3-126M params.
- **Architecture mismatch**: MoE routing, GQA, alternating sliding/full attention, RMSNorm → dense, standard MHA, gated attention, ChannelLayerNorm.
- **Information flow mismatch**: Single residual stream with MoE experts → dual-stream with frozen token pathway.
- **Representation mismatch**: The internal representations live in completely different spaces with different dimensionalities.

### 2.2 What the Literature Recommends

The most successful approaches share common principles:

1. **Projection layers are mandatory** — never match raw features directly across architectures.
2. **Match in a common space** — project both teacher and student into a shared representation (logits space, or a learned latent space).
3. **Stage the distillation** — bottom-up (MOHAWK) or curriculum-based. Don't try to match everything simultaneously.
4. **Transfer what you can directly** — embeddings, LM heads, any shared components. Initialize, don't learn from scratch.
5. **Use CKA to identify layer correspondences** — don't assume proportional mapping (teacher layer 12 ≠ student layer 3).
6. **Contrastive losses work but are expensive** — CRD (ICLR 2020) captures mutual information better than MSE, at 2-3× the cost.
7. **Feature distillation > logit distillation** for preserving structural information.

### 2.3 Proposed Architecture: 12-Layer, 12-Head Symbolic Transformer Student

The paper's experiments use 6-layer, 6-head models (~3-29M params). For distillation from a 24-layer, 64-head teacher, a larger student is appropriate:

**Option A: 12-layer, 12-head (targeting ~80-126M params)**
```
n_layer = 12, n_head = 12, head_dim = 64, n_embd = 768
FFN: 768 × 4 = 3072
Vocab: 50257 (GPT-2 BPE — shared with gpt-oss via mapping)
Gated attention: per-head gates
PLS: per-layer supervision from distillation targets
Dual-stream: CASCADE mode
```

This is a 2:1 layer compression (24 → 12) and ~5:1 head compression per layer (64 → 12). Manageable ratios.

**Option B: 6-layer, 6-head (targeting ~3-29M params)**
- More extreme compression (4:1 layers, ~10:1 heads)
- Useful as a second stage: first distill 24→12, then distill 12→6
- The PLS paper's existing models are this size — direct comparison possible

**Recommendation**: Start with Option A (12/12). Once that works, try progressive distillation to Option B (6/6).

### 2.4 Proposed Distillation Pipeline (CASCADE + Staged Refinement)

CASCADE mode makes this problem fundamentally different from generic cross-architecture distillation. Because x_t is frozen and lm_head (W) is known, the teacher's output distribution defines a **closed-form regression target** for x_e. This reduces distillation from function matching to structured matrix factorization. See CASCADE_DISTILLATION.md for the full derivation.

**Stage 0: Vocabulary Projection and Initialization**

The vocabulary mismatch (gpt-oss: 201K o200k_harmony vs. student: 50K GPT-2 BPE) must be handled first:
- Build a vocabulary projection matrix P via token string matching and subword overlap
- Project the teacher's output distributions into the student's vocabulary space: P_student(w) = Σ_{w' maps to w} P_teacher(w')
- Initialize student's embedding and lm_head from projected teacher weights where possible
- Everything else randomly initialized (architectures too different for weight transfer)

**Stage 1: Closed-Form x_e Targets (No Gradient Descent)**

For each training example:
1. Run teacher forward pass → get P_teacher (projected to student vocab)
2. Compute logit targets: z = log P_teacher (temperature-scaled)
3. Compute residual: b = z - W · x_t (subtract known token-identity contribution)
4. Solve: x_e* = W⁺ · b (pseudoinverse — closed-form least-squares)

This produces a dataset of (context, x_e*) pairs — the optimal contextual corrections.

**Stage 2: Regression Warmup**

Train the student's FFN pathway to predict x_e* targets:
```
L_regression = ||f_θ(context) - x_e*||²
```
This is a pure regression problem. No softmax, no KL, no entropy. Just predict the right point in d-dimensional space. This puts the student in the right basin before distributional refinement.

**Stage 3: KL Refinement**

Fine-tune with the actual distributional loss to handle softmax nonlinearity:
```
L_KL = KL(P_teacher || softmax(W · (x_t + f_θ(context)) / τ))
```
Optionally add ground-truth CE loss and contrastive mode-preservation loss.

**Stage 4: Intermediate Layer Information (Optional — Test Empirically)**

Whether teacher intermediate layers help is an open question (estimated 50/50). Three approaches to test incrementally:

1. **Hard negatives** (cheapest): Words considered at intermediate teacher layers but rejected by the final layer. Penalize the student for predicting these.
2. **Decision subspace regularization**: PCA of teacher layer-to-layer contrast vectors → constrain student's x_e trajectory to lie in this subspace.
3. **Unordered decision constraints**: Each teacher layer contrast = a semantic decision direction. Student must traverse all of them, but we don't prescribe which student layer handles which.

**Ablation protocol**: Compare (a) output-only, (b) + hard negatives, (c) + decision subspace, (d) + unordered constraints, (e) shuffled intermediate layers as control. If (e) matches (b)-(d), intermediate layers provide noise, not signal.

See INTERMEDIATE_LAYERS_ANALYSIS.md for the full analysis.

### 2.5 Why CASCADE Mode Simplifies Everything

In CASCADE mode, x_t is frozen — the student has no choice about how to represent token identity. All contextual computation flows through x_e. This eliminates the "which stream gets which supervision" question entirely:
- x_t = the token embedding (fixed, no supervision needed)
- x_e = the only learnable representation (all supervision targets x_e)
- The combined stream x_t + x_e is what gets projected through lm_head

No CKA alignment between streams is needed. No stream-aware distillation. The closed-form x_e* targets are the complete specification of what the student should learn.

### 2.6 Handling MoE → Dense Compression

The teacher uses top-4-of-32 MoE routing. The student is dense. The literature says MoE→dense retains 30-40% of sparsity gains. To do better:

**Approach 1: Routing-informed data selection**
1. Cluster teacher's routing patterns across training data
2. Identify which inputs activate which expert clusters
3. Balance training data across routing clusters
4. This ensures the dense student sees the full range of teacher computation, not just the most common expert pathway

**Approach 2: Multi-teacher from experts**
- Treat each expert cluster as a separate "teacher"
- Train the student to match each expert cluster's computation on the inputs that cluster routes to
- OneS (MoE distillation) shows this can recover 60-90% of MoE benefits

**Approach 3: Small student MoE**
- Instead of a dense student, use a small MoE student (e.g., 8 experts, top-2)
- The student's routing learns to approximate the teacher's routing in compressed form
- This preserves the routing-as-discrete-geometry structure from GEOMETRIC_FRAMEWORK.md

### 2.7 Training Budget and Data

Drawing on MOHAWK's result that 3B tokens (~0.1% of pre-training data) suffices:

**Distillation corpus**:
- 1-3B tokens of diverse text (the grade-school corpus from the PLS paper, augmented with broader text)
- Teacher forward passes generate logit-lens targets, attention patterns, and routing decisions
- These are cached once and reused across student training runs

**Training cost**:
- Teacher inference: ~1 forward pass per training example (cached)
- Student training: ~3-5x standard training (multi-stage, contrastive loss)
- Total: feasible on a single 4090 for Option B (6/6), needs multi-GPU for Option A (12/12)

### 2.8 Evaluation: What "Success" Means

Standard metrics (perplexity, downstream accuracy) matter but aren't the point. The interpretability-specific metrics are:

1. **Ablation-effect variance**: Does the student have higher ablation-effect variance than a same-architecture model trained from scratch? If yes → distillation preserved the teacher's computation structure, and PLS + distillation together produce more modular models than PLS alone.

2. **Feature-space ARI**: Do the student's computational modes (Tier-2 feature clusters) align with the teacher's? High ARI = the student implements similar computational strategies.

3. **Steering controllability**: Is the distilled student more steerable than the same architecture trained from scratch? If yes → computational structure from the teacher makes the student more controllable.

4. **Convergence profile**: Does the student develop a clean convergence trajectory (early layers uncertain, late layers committed)? The student's convergence layer need not be proportional to the teacher's — the student may organize computation differently — but the *ordering* of prediction difficulty should be preserved (inputs the teacher finds easy should also be easy for the student).

---

## Part 3: Connecting the Two Projects

The steering and distillation projects reinforce each other:

### Steering validates distillation quality
If the distilled student is steerable in the same ways as the teacher (same steering vectors work, same concepts are steerable/unsteerable), that's evidence the distillation preserved computational structure — a much richer test than perplexity alone.

### Distillation enables steering research
The symbolic transformer at 6-layer, 6-head is small enough for exhaustive steering experiments (sweep all layers, all coefficients, all contrast sets). But it's only interesting if it implements real computation — distillation from gpt-oss-20b ensures this.

### Feature geometry connects both
The Tier-2 feature space is the common language:
- Steering vectors that move tokens in feature space = behaviorally meaningful steering
- Steering vectors that don't change feature vectors = cosmetic changes (token-level)
- Distillation that preserves feature geometry = structurally faithful distillation
- The feature map tells you *which* modes to steer and *whether* distillation preserved them

### The research arc

```
gpt-oss-20b (24L, 64H, MoE)
    │
    │ Feature extraction (Tier-2 extended for MoE)
    │ Logit-lens convergence profiles
    │ CKA layer correspondence analysis
    │
    ▼
Distillation (MOHAWK-inspired, 4 stages)
    │ Stage 0: Weight transfer (vocab projection)
    │ Stage 1: CKA alignment analysis
    │ Stage 2: Block-level + PLS logit matching
    │ Stage 3: E2E + contrastive mode matching
    │
    ▼
Symbolic Transformer Student (12L, 12H or 6L, 6H)
    │ CASCADE mode, gated attention, PLS
    │
    │ Evaluate: ablation variance, feature ARI,
    │   convergence profiles, perplexity
    │
    ▼
Steering Vector Experiments
    │ Stream-selective steering (x_t vs x_e)
    │ Gate-mediated steering
    │ PLS vs. standard steerability comparison
    │ Feature-guided mode-specific steering
    │
    ▼
Results → Paper: "Interpretable Distillation and Controllable Steering
          via Dual-Stream Architectures with Per-Layer Supervision"
```

---

## References

See `references/STEERING_VECTORS_LITERATURE.md` and `references/DISTILLATION_LITERATURE.md` for full citations.

Key papers for this plan:
- MOHAWK (NeurIPS 2024): 3-stage cross-architecture distillation
- CAA (Rimsky et al., ACL 2024): Mean-difference steering vectors
- Durmus et al. (NeurIPS 2024): Steering failure modes (~50% anti-steerability)
- OFA-KD (NeurIPS 2024): CKA-based layer alignment for cross-architecture KD
- CRD (Tian et al., ICLR 2020): Contrastive representation distillation
- TMLR 2025 Survey: Practical lessons on what actually works
