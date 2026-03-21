# CASCADE Distillation: Matrix Factorization with Known Factors

## The Core Observation

In CASCADE mode:
- **x_t** is frozen after the embedding layer — it IS the token embedding
- **x_e** is learned by the FFN pathway — it's the contextual correction
- **lm_head** (W) projects back to vocabulary — it's a known linear map
- The teacher (gpt-oss-20b) provides a full probability distribution over vocabulary at each position

The prediction is:

```
P_student(word | context) = softmax(W · (x_t + x_e))
```

Since x_t and W are known, the problem reduces to: **find x_e such that W · x_e approximates the teacher's distribution, corrected for the known token identity W · x_t.**

```
W · x_e ≈ logit_teacher - W · x_t
```

This is a linear system with V ≈ 200K equations and d ≈ 768 unknowns.

## What Is Actually Identified

The teacher supplies a probability distribution `p_teacher`, not a unique logit vector. If `z` is a valid logit vector, then `z + c1` gives the same distribution for any constant `c`.

That means CASCADE cannot be defined on raw logits without fixing a gauge.

The clean way to do this is to work in centered-logit space:

```
C = I - (1/V)11^T
z_hat = C log p_teacher
A = C W
b = z_hat - C W x_t
```

The CASCADE target is then defined as the minimum-norm least-squares solution:

```
x_e* = argmin_x ||A x - b||_2^2 = A^+ b
```

This is the mathematically justified object.

Important consequences:
- `x_e*` is a least-squares surrogate, not exact recovery of the teacher's hidden state
- if `b` is not in the column space of `A`, exact reconstruction is impossible
- any component in `null(A)` is observationally invisible in centered-logit space
- therefore the identified object is an equivalence class in representation space, with `x_e*` the minimum-norm representative

## Why This Is LSA / Matrix Factorization

For a single context, the linear system has a pseudoinverse solution. But across a corpus of N contexts, we're factoring:

```
R[i, j] = P_teacher(word_j | context_i)     [N × V matrix]
```

The student's model is:

```
R ≈ softmax((X_t + f_θ(contexts)) · W^T)
```

where X_t ∈ ℝ^(N × d) is the matrix of frozen token embeddings and f_θ is the student's FFN pathway parameterized by θ. The only free parameter is θ, which determines X_e = f_θ(contexts).

**Without the softmax**, and after fixing the logit gauge, each example satisfies:

```
z_hat^(i) ≈ C W (x_t^(i) + x_e^(i))
```
```
b^(i) = z_hat^(i) - C W x_t^(i) ≈ C W x_e^(i)
```

Across a corpus, stacking these row by row gives the matrix relation:

```
B ≈ X_e · (C W)^T
```

This is a **matrix equation with known right factor**. The minimum-norm solution for each row is:

```
x_e^(i) = B[i, :] · (C W) · ((C W)^T (C W))^{-1} = B[i, :] · (C W)⁺ᵀ
```

This is classical least-squares, applied row by row. The student doesn't need gradient descent to find the minimum-norm target for each context.

**But** the student must *generalize*: f_θ must produce good x_e for unseen contexts, not just memorize the training set. The closed-form solutions for the training contexts become **regression targets** for f_θ.

## The Training Procedure

### Stage 1: Compute Optimal x_e Targets (Closed Form)

For each training context i:
1. Run teacher forward pass → get `P_teacher(· | context_i)` for the predicted position
2. Compute the centered log-probability target: `z_hat_i = C log P_teacher`
3. Look up `x_t^(i)` (the token embedding at the predicted position)
4. Compute residual: `b_i = z_hat_i - C W · x_t^(i)`
5. Solve for the minimum-norm target: `x_e^*(i) = (C W)^+ · b_i`

This gives a **dataset of (context_i, x_e^*(i)) pairs** — the minimum-norm contextual corrections that best match the teacher in centered-logit space.

### Stage 2: Train f_θ to Predict x_e*

Train the student's FFN pathway (the x_e-producing network) to regress:

```
L_regression = Σ_i ||f_θ(context_i) - x_e^*(i)||^2
```

This is a pure regression problem. No softmax, no KL divergence, no entropy. Just predict the least-squares target in d-dimensional space.

**But this is too simple** — it ignores the nonlinearity of softmax and treats all errors equally. The actual loss should weight errors by their impact on the distribution:

### Stage 3: Refine with Distribution Matching

After the regression warmup, fine-tune with the actual distributional loss:

```
L_distill = KL(P_teacher || P_student) = Σ_j P_teacher(j) · log(P_teacher(j) / P_student(j))
```

where P_student(j) = softmax(W · (x_t + f_θ(context)))[j].

The regression initialization (Stage 2) puts the student in the right basin. The KL refinement (Stage 3) handles the softmax nonlinearity.

## Three Questions Every CASCADE Experiment Must Answer

Before treating `x_e*` as a useful supervision target, evaluate three things:

1. `Identifiability`
   - What part of `x_e` is actually visible through `C W`?
   - What is the effective rank of `C W`?

2. `Numerical stability`
   - Is solving with `(C W)^+` well-conditioned in practice?
   - Do truncated SVD or ridge-regularized variants materially change the target?

3. `Behavioral adequacy`
   - Does reconstructing with `x_t + x_e*` preserve the teacher's behavior well enough to justify the surrogate?
   - Residual norms matter, but the decisive metric is induced distributional / behavioral match

## Connection to Topic Modeling

The teacher's distributions across a corpus define a **term-context probability matrix** R. Factoring this matrix reveals latent structure:

- **LSA** (Deerwester et al., 1990): SVD of the term-document matrix discovers latent "topics" — directions in word space that capture co-occurrence patterns
- **pLSA** (Hofmann, 1999): Probabilistic version — each context is a mixture of topics, each topic is a distribution over words
- **LDA** (Blei, Ng, Jordan, 2003): Bayesian version with Dirichlet priors on topic mixtures

In CASCADE distillation:
- The **topics** are the computational modes (Tier-2 feature clusters)
- Each context's x_e determines a **topic mixture** (which modes are active)
- The lm_head weights W define the **topic-word distributions** (how each mode maps to vocabulary)

The key insight: **the computational modes discovered by Tier-2 feature clustering on the teacher are exactly the latent topics in the matrix factorization**. They're the same structure seen from two perspectives:
- Feature clustering: discovered from forward-pass dynamics (how the model processes)
- Matrix factorization: discovered from input-output behavior (what the model predicts)

If these two perspectives give the same decomposition → the mode structure is real, not an artifact.

## Vocabulary Mismatch

The closed-form approach assumes the same `W` (lm_head) for both teacher and student. When vocabularies differ (gpt-oss: 201K o200k_harmony vs. student: 50K GPT-2 BPE), the teacher's distribution must be projected into the student's vocabulary space first:

1. Build a mapping M between teacher and student tokens via string matching and subword overlap
2. Project: P_student(w) = Σ_{w' ∈ M(w)} P_teacher(w') (aggregate probability for equivalent tokens)
3. Compute z_teacher in the student's vocabulary space
4. Then apply the pseudoinverse with the student's own W

This is a necessary preprocessing step, but it is also another source of approximation error. The projected CASCADE target is only as good as the vocabulary projection.

## Incorporating PLS (Per-Layer Supervision) — Open Question

PLS naturally fits this framework: at each student layer ℓ, the student produces an x_e^(ℓ) that could in principle target a teacher logit-lens readout:

```
logit_lens(ℓ) = W · (x_t + x_e^(ℓ)) ≈ teacher_logit_lens(ℓ_T)
```

**However, this per-layer approach has a critical caveat**: we don't know the teacher-student layer correspondence, and the literature strongly suggests that proportional mapping is wrong across heterogeneous architectures. Even EM-style alternating optimization (discover correspondence, then match) is questionable when the architectures organize computation fundamentally differently.

**The recommended approach** (see INTERMEDIATE_LAYERS_ANALYSIS.md):
- The final-layer x_e* target is well-grounded and should definitely be used
- Per-layer PLS targets from teacher logit-lens are speculative — whether they help is an empirical question (estimated 50/50)
- If per-layer targets are used, prefer the EM approach below over fixed correspondences
- More defensible uses of intermediate layer data: hard negatives, decision subspace regularization, unordered decision constraints (see INTERMEDIATE_LAYERS_ANALYSIS.md)

### Per-layer EM (if intermediate layers prove useful)

If per-layer targets ARE used, the correspondence should be discovered, not assumed:

**E-step**: For each student layer ℓ_S, find the teacher layer ℓ_T with lowest regression error:
```
        correspondence(ℓ_S) = argmin_{ℓ_T} ||x_e^(ℓ_S) - (C W)^+ · C(z_teacher^(ℓ_T) - W · x_t)||^2
```

**M-step**: Given correspondence, update θ to minimize per-layer regression loss.

This alternating optimization discovers the layer correspondence and trains the student simultaneously. It is more principled than fixed proportional mapping, but still uncertain whether it adds value over output-only distillation.

## Practical Loss Function

Combining everything:

```python
def cascade_distillation_loss(
    student_x_e,           # [batch, seq, d] — student's contextual stream at each layer
    x_t,                   # [batch, seq, d] — frozen token embeddings
    lm_head_weight,        # [V, d] — known lm_head matrix
    teacher_log_probs,     # [batch, seq, V] — teacher's log-probability distribution
    teacher_layer_logits,  # [n_teacher_layers, batch, seq, V] — teacher logit-lens
    student_layer_x_e,     # [n_student_layers, batch, seq, d] — student x_e per layer
    layer_correspondence,  # [n_student_layers] → teacher layer indices
    temperature=2.0,
    lambda_reg=1.0,
    lambda_kl=1.0,
    lambda_pls=0.5,
):
    W = lm_head_weight  # [V, d]
    # Do not materialize C explicitly at vocabulary scale.
    # Center the vocab dimension by subtracting means.
    W_centered = W - W.mean(dim=0, keepdim=True)
    A_pinv = torch.linalg.pinv(W_centered)  # [d, V]

    # --- Stage A: Regression loss (closed-form targets) ---
    centered_teacher = teacher_log_probs - teacher_log_probs.mean(dim=-1, keepdim=True)
    centered_xt = (x_t @ W.T)
    centered_xt = centered_xt - centered_xt.mean(dim=-1, keepdim=True)
    residual = centered_teacher - centered_xt  # [batch, seq, V]
    x_e_target = residual @ A_pinv.T  # [batch, seq, d]

    L_reg = ((student_x_e - x_e_target) ** 2).mean()

    # --- Stage B: KL divergence (handles softmax nonlinearity) ---
    student_logits = (x_t + student_x_e) @ W.T  # [batch, seq, V]
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_log_probs / temperature, dim=-1)

    L_kl = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')

    # --- Stage C: Per-layer supervision (trajectory matching) ---
    L_pls = 0.0
    for ell_S in range(len(student_layer_x_e)):
        ell_T = layer_correspondence[ell_S]
        teacher_layer_lp = F.log_softmax(teacher_layer_logits[ell_T], dim=-1)
        teacher_layer_lp = teacher_layer_lp - teacher_layer_lp.mean(dim=-1, keepdim=True)
        residual_ell = teacher_layer_lp - centered_xt
        x_e_target_ell = residual_ell @ A_pinv.T

        # Layer weight (linear decay — later layers matter more)
        w_ell = (ell_S + 1) / sum(range(1, len(student_layer_x_e) + 1))
        L_pls += w_ell * ((student_layer_x_e[ell_S] - x_e_target_ell) ** 2).mean()

    return lambda_reg * L_reg + lambda_kl * L_kl + lambda_pls * L_pls
```

## The Geometric Picture

In d-dimensional embedding space (d ≈ 768):
- x_t is a known point (the current token's embedding)
- The teacher's distribution defines a **target region** — the set of x_e values that produce distributions close to the teacher's
- This target region is an affine subspace (for the linear part) deformed by softmax (for the nonlinear part)
- The closed-form `x_e^*` is the **minimum-norm representative** of the equivalence class that best matches the teacher in centered-logit space
- The PLS targets {x_e^*(ℓ)} trace a **geodesic** from origin to x_e^* through the layers

The Tier-2 feature vectors measure properties of this geodesic (convergence layer = when x_e reaches the target, head activation = which directions are traversed, entropy = how sharply the trajectory commits). The computational modes are clusters of geodesics that follow similar paths.

## What Makes This Different from Standard Distillation

Standard distillation: minimize KL(P_teacher || P_student). This is a black-box matching — the student figures out its own internal organization.

CASCADE distillation with closed-form targets: the student is given **explicit least-squares targets in its own representation space**. The lm_head provides the bridge — it translates between the teacher's vocabulary-space distribution and the student's d-dimensional representation.

This is why CASCADE mode is special: **the known, shared embedding geometry makes the distillation problem a structured linear system rather than a generic function approximation**. You're not just matching outputs — you're solving for the student's internal state.

## Relation to Classical Problems

| Classical Problem | Modern Incarnation |
| --- | --- |
| Least squares (Gauss, 1795) | Closed-form x_e from W^+ · residual |
| SVD / matrix factorization (Eckart-Young, 1936) | Factoring the teacher's word-context matrix through x_e bottleneck |
| Latent Semantic Analysis (Deerwester, 1990) | Discovering computational modes from teacher distributions |
| pLSA / Topic Models (Hofmann, 1999) | Each context is a mixture of computational modes |
| EM algorithm (Dempster et al., 1977) | Alternating layer-correspondence and parameter updates |
| Optimal transport (Kantorovich, 1942) | Wasserstein loss respecting embedding geometry |

The "classical problem in a modern context" is all of these simultaneously: **matrix factorization with side information** (known x_t, known W) **using EM to handle latent correspondences** (which teacher layers map to which student layers) **in a geometric space** (the shared embedding) **where the factors have physical meaning** (computational modes = topics).

## Next Steps

1. **Implement the gauge-safe x_e target computation** — this requires teacher log-probs, the embedding matrix, the lm_head weights, and an explicit centering convention.

2. **Verify on upgraded logit-lens data** — compute `x_e^*` for each prompt and check:
   - residual norm in centered-logit space
   - effective rank / conditioning of `C W`
   - KL after reconstruction

3. **Train a 6-layer CASCADE student** with the staged loss (regression warmup → KL refinement). Treat PLS trajectory matching as optional until the final-layer target is validated.

4. **Extract Tier-2 features from the student** and compare mode structure to the teacher's. Do the topics match?

5. **Measure ablation-effect variance** on the student. If the mode-preserving distillation works, the student should be MORE modular than a student trained from scratch, because the training signal explicitly separates modes.
