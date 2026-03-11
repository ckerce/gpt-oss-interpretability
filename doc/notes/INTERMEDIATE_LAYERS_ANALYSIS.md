# Do the Teacher's Intermediate Layers Help? An Honest Analysis

## The Question

gpt-oss-20b produces a next-token distribution at EVERY layer (via logit-lens). The symbolic transformer student has PLS, so it CAN accept per-layer supervision. But SHOULD it? Are the teacher's intermediate distributions useful training signal, or is the final output sufficient?

**Prior**: 50/50. The arguments are genuinely balanced.

---

## The Case for "Just Outputs"

### 1. The architectures are too different for layer matching

The teacher (24-layer MoE, single residual stream, GQA, alternating attention) and the student (6-layer dual-stream, gated attention, frozen x_t) solve problems in completely different ways. The teacher's layer 12 decision corresponds to... nothing specific in the student.

MOHAWK (NeurIPS 2024) found that even with a structured 3-stage distillation, the intermediate matching stages mainly serve as *initialization* — the final end-to-end KD stage does the real work. And MOHAWK operates on architecturally similar models (Transformer → SSM) that at least share embedding dimensions.

### 2. The student may need to organize differently

A 6-layer student can't afford to replicate the teacher's 24-layer computation trajectory. It needs to make different tradeoffs:
- Compress multiple teacher decisions into single layers
- Skip intermediate states the teacher passes through
- Use the dual-stream structure for parallelism the teacher doesn't have

Layer-matched supervision could prevent the student from finding its own optimal organization.

### 3. Output KD is already very strong

Standard knowledge distillation (soft labels from the teacher's final distribution) captures most of what matters. The "dark knowledge" (Hinton 2015) in the teacher's distribution — the relative probabilities of wrong answers — already encodes the similarity structure of the vocabulary. Temperature-scaled KL divergence is hard to beat.

### 4. CASCADE mode makes the final layer special

In CASCADE mode, x_t is frozen. The entire computation is the trajectory of x_e from zero to its final value. The final x_e is what matters for the prediction. Intermediate x_e values are artifacts of the student's architecture, not meaningful targets.

**Verdict for "just outputs"**: If pressed, I'd implement the closed-form cascade distillation from CASCADE_DISTILLATION.md using only the teacher's final distribution. This gives the regression target x_e* = W⁺(z_teacher - Wx_t) plus KL refinement. Simple, principled, testable.

---

## The Case for "Intermediate Layers Add Value" — But Not as Layer Supervision

The teacher's per-layer logit-lens data contains information that the final output doesn't: the **trajectory of computation**. The question is how to extract that information without imposing a layer correspondence.

### What information is in the intermediate layers?

#### 1. Semantic Decisions (Layer Contrasts)

Between consecutive teacher layers, some words gain probability and others lose it. Each transition encodes a semantic decision:

```
Δ^(ℓ) = Σ_w (p^(ℓ+1)(w) - p^(ℓ)(w)) · embedding(w)
```

This is a vector in d-dimensional embedding space pointing in the direction of "what the teacher decided at layer ℓ." For the recency bias prompt:

- Layers 0-10: noise → no clear decision direction
- Layers 11-16: <endoftext> → early tokens. Direction: "away from padding, toward content"
- Layers 16-18: content → "small." Direction: "the adjective direction" — the semantic axis distinguishing the specific adjective from general content

These decision directions are **architecture-independent**. The student needs to traverse the same semantic axes, regardless of how many layers it takes.

#### 2. Decision Difficulty (Convergence Layer)

The teacher's convergence layer tells you how "hard" each prediction is:
- Layer 18 convergence (recency bias): moderately hard
- Layer 20 convergence (syntax agreement): hard
- Layer 21 convergence (induction): very hard

This is useful as **curriculum weighting** — train the student harder on predictions the teacher found difficult. Easy predictions need less supervision; hard predictions need more.

#### 3. Rejected Alternatives (Transient Hypotheses)

Words that appear in intermediate distributions but vanish by the final layer represent **paths not taken**. The teacher considered "big" at layer 14 but rejected it by layer 18. The student should ALSO not predict "big" — but standard output KD doesn't tell the student what to avoid, only what to match.

A **contrastive signal** extracted from intermediate layers:
- Positive: words in the final distribution (match these)
- Hard negatives: words in intermediate distributions but NOT final (avoid these specifically)
- Easy negatives: words never in any distribution (ignore these)

The hard negatives from intermediate layers provide a curriculum of "plausible but wrong" answers — richer than random negative sampling.

#### 4. Semantic Subspace Identification

Across a corpus, the layer-to-layer decision vectors Δ^(ℓ) span a low-dimensional subspace of embedding space. This subspace contains the "directions that matter" for the teacher's computation. The student should also organize its computation along these directions.

This doesn't require layer matching. It's a **subspace constraint**: the student's per-layer changes to x_e should lie (mostly) in the subspace spanned by the teacher's decision vectors.

---

## Three Concrete Ways to Use Intermediate Layers Without Layer Matching

### Approach 1: Unordered Decision Constraints

Extract the set of semantic decisions from the teacher's trajectory:

```python
decisions = []
for ell in range(n_teacher_layers - 1):
    delta_probs = p_teacher[ell+1] - p_teacher[ell]  # [V]
    # Only keep significant transitions
    if delta_probs.abs().max() > threshold:
        # Direction in embedding space
        direction = delta_probs @ embedding_matrix  # [d]
        decisions.append(direction / direction.norm())
```

These decisions are UNORDERED. The loss says: "the student's computation, from first layer to last, must traverse all of these directions somewhere."

```python
# Student's total computation vector (sum of per-layer x_e changes)
student_trajectory = [x_e[ell+1] - x_e[ell] for ell in range(n_student_layers - 1)]

# Each teacher decision must be "covered" by some student layer
L_coverage = 0
for decision in teacher_decisions:
    # How well does any student layer align with this decision?
    alignments = [F.cosine_similarity(step, decision, dim=-1) for step in student_trajectory]
    best_alignment = torch.stack(alignments).max(dim=0).values
    L_coverage += (1 - best_alignment).mean()
```

This is a **set matching** loss: each teacher decision needs a student layer that implements it, but we don't prescribe which layer.

**Cost**: O(teacher_decisions × student_layers) per example. Modest.

### Approach 2: Hard-Negative Contrastive Loss

Extract "considered but rejected" words from intermediate layers:

```python
# Words the teacher considered (appeared in top-k at any intermediate layer)
considered = set()
for ell in range(n_teacher_layers):
    topk = p_teacher[ell].topk(k=50).indices
    considered.update(topk.tolist())

# Words in the final output
final_words = set(p_teacher[-1].topk(k=50).indices.tolist())

# Hard negatives: considered but rejected
hard_negatives = considered - final_words
```

Add a contrastive term to the loss:

```python
student_logits = (x_t + x_e) @ W.T
# Push down hard negatives more than easy negatives
for neg_idx in hard_negatives:
    L_contrast += F.relu(student_logits[:, neg_idx] - margin)
```

**Cost**: Negligible. Just modifies the existing loss with targeted penalties.

### Approach 3: Decision Subspace Regularization

Compute the principal subspace of teacher decision vectors:

```python
# Collect all layer-to-layer decision vectors across a corpus
all_decisions = []  # List of [d]-dimensional vectors
for example in corpus:
    for ell in range(n_teacher_layers - 1):
        delta = p_teacher[ell+1] - p_teacher[ell]
        direction = delta @ embedding_matrix
        if direction.norm() > threshold:
            all_decisions.append(direction)

# PCA to find the k-dimensional decision subspace
decision_matrix = torch.stack(all_decisions)  # [N_decisions, d]
U, S, V = torch.svd(decision_matrix)
decision_subspace = V[:, :k]  # [d, k] — the principal decision directions
```

Regularize the student to compute in this subspace:

```python
# Project student's x_e changes onto the decision subspace
for ell in range(n_student_layers - 1):
    delta_x_e = x_e[ell+1] - x_e[ell]  # [batch, seq, d]
    projection = delta_x_e @ decision_subspace @ decision_subspace.T
    residual = delta_x_e - projection
    L_subspace += residual.norm(dim=-1).mean()  # Penalize off-subspace computation
```

This says: "your computation should mostly move in directions the teacher found useful." It doesn't say which directions at which layers — just that the overall subspace should match.

**Cost**: One SVD precomputation. Per-batch cost is O(d × k) — cheap.

---

## My Actual Recommendation

**Start with just outputs.** The closed-form CASCADE distillation (x_e* from the pseudoinverse) plus KL refinement on the final distribution is clean, principled, and fast to implement. It might be 80% of the answer.

**Then test Approach 2 (hard-negative contrastive) as the first addition.** It's cheap, doesn't require layer correspondence, and uses intermediate layers in the most defensible way — identifying "plausible but wrong" predictions. If this improves over output-only distillation, the intermediate layers contain useful information.

**If Approach 2 helps, try Approach 3 (decision subspace regularization).** This is the deepest use of intermediate layers — it constrains the student's computation geometry to align with the teacher's semantic decision structure.

**Approach 1 (unordered decision constraints) is the most theoretically appealing but hardest to tune.** The set-matching loss is non-trivial to optimize. Save this for when the simpler approaches have been tested.

**What I'd actually measure**:
1. Output-only distillation: x_e* regression + KL on final distribution
2. + Hard-negative contrastive from intermediate layers
3. + Decision subspace regularization
4. + Unordered decision constraints
5. Ablation: random intermediate layers (shuffled) as a control

If (5) performs as well as (2)-(4), the intermediate layers are providing noise, not signal. If (2)-(4) beat (5), the semantic content matters.

---

## The Connection to Steering Vectors

The teacher's layer-to-layer decision vectors Δ^(ℓ) are literally **steering vectors computed from the teacher's own computation**. The teacher steers itself from "could be big or small" to "definitely small" between layers 16 and 18. That steering direction lives in embedding space — the same space as x_t and x_e.

This creates a bridge between the distillation and steering projects:
- **Distillation** extracts the teacher's steering directions (what decisions does it make?)
- **Steering** applies new directions to the student (can we override its decisions?)
- The quality of distillation can be measured by whether the teacher's steering directions also work on the student
- If the student's decision subspace aligns with the teacher's → distillation preserved computational structure

The Tier-2 features capture exactly this: the processing depth and head activation patterns encode which decisions happen where. If the student's Tier-2 features cluster similarly to the teacher's → the decision trajectories are preserved.
