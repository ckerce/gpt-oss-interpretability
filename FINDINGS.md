# Key Findings

Full narrative for the nine solid research threads, organized by the four-phase investigation structure. For the compact summary see [README.md](README.md); for the thread index and maturity levels see [THREAD_MAP.md](THREAD_MAP.md).

---

## Phase 1 — Where does gpt-oss-20b compute each task?

**Finding 1 — Task-dependent convergence: different behaviors resolve at different depths (Thread 1)**

Per-layer logit-lens readouts with choice-relative convergence show that the depth at which the model commits to an answer is not uniform — it is task-specific:

![Convergence trajectories](figures/fig1_convergence_trajectories.png)

- **Capitalization**: converges early (L1–2), minimal depth processing
- **Coreference**: converges mid-depth (L5), requires semantic resolution
- **Induction**: converges late (L17+), consistent with induction heads as a late-layer phenomenon

The induction result is particularly interpretable. Standard induction benchmarks use sequences like `sun moon star sun moon star sun moon ___` — but these could be retrieved from training data. We test with arbitrary tokens that cannot plausibly be memorized:

**Clean structural induction**: `D 5 Z 7 B 2 D 5 Z 7 B 2 D 5 Z 7 B ___`
- Core pattern: `D 5 Z 7 B 2` repeats 3 times. After "B ", the model predicts **"2"** (converges at L19).

**Noisy structural induction**: `D 5 Z 7 B 2 D 5 A 7 B 2 D 5 W 7 B 2 D 5 X 7 B 2 D 5 Y 7 B 2 D 5 Q 7 B 2 D 5 R ___`
- Core pattern: `7 B 2 D 5 [letter]` — the letter changes each cycle (Z, A, W, X, Y, Q, R) while the rest repeats. After "R ", the model predicts **"7"** (converges at L20).

The model extracts the stable structural regularity `7 B 2 D 5` despite the varying letter — genuine in-context pattern recognition, not memorized sequence retrieval. The noisy case requires one additional layer (L20 vs L19), reflecting the extra computation needed to distinguish structure from noise.

This connects two lines of work. Olsson et al. (2022) identified **induction heads** as specific attention circuits implementing match-and-copy behavior through a phase transition in middle-to-late layers — consistent with our observation that structural induction requires deep processing (L19–L20) while simpler tasks resolve at L1–2. DeepSeek's **Engram module** (2025) replaces early FFN layers with O(1) hash-based n-gram lookup with no loss in model quality, validating that early layers can be externalized as non-parametric memory. The contrast is precise: the early-layer tasks that Engram can replace (pattern recall, factual retrieval) converge at L1–5 in our logit-lens analysis, while the structural computation that *cannot* be replaced by lookup (noisy induction) converges at L19–L20.

---

**Finding 2 — Late-layer ablation reveals that L19–21 are causally critical (Thread 2)**

Logit-lens convergence is correlational — it shows where the model's prediction stabilizes, not which layers produce that stability. Layer ablation answers the causal question.

Ablating individual layers on the 9-case main analysis set shows that layers 19–21 are where gpt-oss-20b resolves task-relevant behavior:

![Late-layer ablation](figures/fig2_late_layer_ablation.png)

Layers 19–21 ablation drops accuracy from 100% to 44% and collapses margin by 85–90%. Layer 23 ablation preserves accuracy (100%) despite margin loss, suggesting it refines rather than decides. This narrows the interpretability target from 24 layers to 3.

---

## Phase 2 — How do steering directions arise, and why is head-level intervention hard?

**Finding 3 — Decision trajectories reveal self-supervised steering directions (Thread 4)**

Having located the critical layers, the next question is what to steer toward. Each layer where the model's top-1 prediction changes is a "decision point." The logit-space difference at decision layers is a self-supervised steering direction — the model tells you what decision it made, and at which layer:

![Decision trajectories](figures/fig4_decision_trajectories.png)

Example decision arc for recency-bias position 12 ("suitcase was too ___"):
```
L0: noise → L8: '‑' → L13: 'pack' → L14: <eos> → L16: '(' → L17: 'too' → L18: 'small'
```

This is fundamentally different from contrastive activation addition (CAA), which requires 100+ curated positive/negative example pairs. CASCADE reads steering directions directly from the model's own computation, with no external annotation.

---

**Finding 4 — Per-head Hydra measurement confirms why head-level intervention fails (Thread 5)**

Having identified candidate steering layers (L19–21) and candidate directions, the natural intervention target is individual attention heads. But ablating each of 64 heads individually at L20 produces near-identical margins (σ = 0.042) — the model barely notices losing any single head:

![Hydra variance](figures/fig5_hydra_variance.png)

gpt-oss-20b's σ = 0.042 is **half** the PLS-paper control (σ = 0.08) and **11× smaller** than PLS-trained models (σ = 0.47). This directly validates the Hydra hypothesis at production scale: standard training produces extreme distributed redundancy. There is no individual attention head to target — the computation is spread uniformly across all 64. This explains why head-level circuit-finding is difficult in standard models.

---

**Finding 5 — Direct-vocabulary steering works despite Hydra, with positional specificity (Thread 6)**

Despite the absence of specialized heads, exact vocabulary-space directions (`W[token_A] - W[token_B]`) applied in the contextual stream at late layers cleanly flip model answers:

![Steering heatmaps](figures/fig_matched_pair_heatmaps.png)

Crucially, the effect is **position-specific**: steering at the decision-token position flips answers; identical steering at token 0 produces zero effect. This rules out diffuse perturbation artifacts. The vocabulary direction, applied at the right layer and position, is sufficient to override the model's answer — even though no individual head is responsible for producing it.

---

## Phase 3 — How selective and channel-specific is the steering?

Channel-level analysis of this kind is tractable only in architecturally structured companion models, where stream separation makes individual dimensions interpretable. These experiments use three matched DST models from the companion preprint work:

| Label | Architecture | Mixing | Params | Description |
|-------|-------------|--------|-------:|-------------|
| DST-baseline | Single-stream dense | dns-dns/dns-dns | 71M | Standard transformer baseline (no stream separation) |
| DST-cascade | CASCADE dual-stream | dns-dns/dns-dns + gated attention | 71M | Dual-stream with frozen symbolic stream; attention output cascades into FFN |
| DST-independent | Fully independent channels | ind-ind/ind-ind | 22M | Complete channel isolation — maximum interpretability |

The mixing signature `attn-attn/ffn-ffn` describes how heads share information at four mixing points (attention value, attention output, FFN up-projection, FFN down-projection): `dns` = dense (standard), `ind` = independent (no cross-head mixing).

---

**Finding 6 — Probing and causal intervention dissociate: they identify different channels (Thread 7)**

Given a steering direction, which hidden-state dimensions carry the signal? The naive approach is to probe each channel for answer-predictive content, then intervene on high-probe-weight channels. We tested this directly.

Per-channel causal analysis on DST-cascade reveals that the channels that probe well for the answer (H4) are systematically *not* the channels whose intervention most changes the output (H2, H5). Spearman correlation between probe weight and causal effect = **−0.363** on the induction task.

This is the probe-causation dissociation: probing identifies channels whose activation is *correlated* with the model's readout; intervention identifies channels whose activation *drives* the model's computation. These are different things. The practical implication is that probing alone cannot identify intervention targets — causal confirmation is required.

---

**Finding 7 — Steering selectivity is task-dependent: recency concentrates, induction distributes (Thread 8)**

Given that steering affects specific channels (not all of them), how broadly does the effect spread across the model? Selectivity is measured as the ratio of within-target-task effect to cross-task effect.

Cross-family comparison on DST-cascade:

| Task | Channelized/whole ratio | Interpretation |
|------|------------------------:|----------------|
| Recency | 0.80 | Steering concentrates in a few channels |
| Induction | 0.60 | Steering is distributed across many channels |

Recency steering is spatially concentrated; induction steering is spatially distributed. The right intervention granularity is task-dependent — a single steering strategy does not generalize across task families.

---

## Phase 4 — Why does linear steering work geometrically?

Vocabulary-space directions cleanly flip answers at some layers and for some tasks, but not uniformly. The underlying reason is geometric.

**Finding 8 — Standard transformers have low effective rank at intermediate layers; stream separation improves conditioning up to 22× (Thread 14)**

Linear steering methods implicitly assume that the hidden-state geometry is well-conditioned — that a direction in embedding space corresponds to a meaningful direction in the model's internal representation. We tested whether this holds.

Using a 2×2 factorial design (stream separation × per-layer supervision) across four matched 45.4M-parameter transformers, we extend Park et al.'s output-layer Bregman analysis inward to intermediate layers (`arXiv:2603.21317`):

- **Standard single-stream transformers**: effective rank 8 in 516 dimensions at intermediate layers. Linear methods operate in **2% of the available geometry**.
- **Stream-separated models**: conditioning improves up to **22×** — not by changing the training objective, but by architectural constraint.

A **cosine diagnostic** predicts which layers are safe for linear intervention: when the cosine similarity between the steering direction and the leading eigenvectors of the layer's Hessian is low, the intervention is operating in a nearly-null subspace and will be ineffective or unpredictable.

This explains the pattern observed in Phases 2–3: steering succeeds at late layers (L19–21) because those layers have relatively better-conditioned geometry, suggesting the decision region is where the model's representational manifold aligns with vocabulary space. It also explains task-dependent selectivity: recency concentrates because the recency direction aligns well with the dominant geometry; induction distributes because the induction direction is spread across many low-eigenvalue components.

---

## Summary

Across the four phases:

**Phase 1** locates computation by task type (convergence) and confirms causal criticality (ablation), narrowing the interpretability target from 24 layers to 3.

**Phase 2** shows that the model's own decision transitions yield self-supervised steering directions (no curated contrast pairs needed), and that despite extreme head redundancy (Hydra), vocabulary-space directions applied at the right layer and position cleanly flip answers with positional specificity.

**Phase 3** reveals that probing and causation identify different channels, and that steering selectivity is task-dependent — meaning the right intervention granularity must be determined empirically per task.

**Phase 4** provides the geometric foundation: standard transformers have near-degenerate geometry at intermediate layers (rank 8 in 516 dimensions), which explains both where steering succeeds and why selectivity varies. Stream separation improves conditioning up to 22×, making the companion DST models tractable for channel-level analysis.
