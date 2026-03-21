# Research Connections: gpt-oss-interp ↔ Activation Clustering / PLS Paper

## Overview

The PLS paper ("Engineering Verifiable Modularity in Transformers via Per-Layer Supervision") demonstrates that per-layer supervision (PLS) breaks the Hydra effect and enables mechanistic interpretability on custom architectures at 29-126M scale. The gpt-oss-interp toolkit provides forward-pass instrumentation on a production 21B MoE model.

The connection between these is not superficial. The same conceptual apparatus — layer-wise readouts, ablation-effect variance, feature-based computational mode discovery — can be applied to gpt-oss-20b to answer questions the paper currently cannot.

---

## 1. Scale validation of PLS claims

### The paper's acknowledged weakness

The paper's results hold at 6-12 layers, 29-126M parameters. The main open question — stated explicitly — is whether the findings generalize to frontier scale. Reviewers will ask this.

### What gpt-oss-interp provides

The logit-lens results on gpt-oss-20b already measure the quantity PLS is designed to optimize: **per-layer lm_head compatibility**. The logit-lens convergence layer is a direct readout of when intermediate representations become decodable through the final unembedding.

**Concrete experiment**: Run the 163-dimensional Tier-2 feature extraction on gpt-oss-20b activations. Compare the resulting computational mode clusters to those found in the PLS-trained symbolic transformer. If the same feature space discovers meaningful clusters in both architectures, the feature engineering methodology is architecture-general — not an artifact of the custom training setup.

**What this would show**: The invariance-constrained feature design (token-identity invariance, permutation invariance, topology preservation) transfers from controlled-scale to production-scale MoE models. This is a much stronger validation than scaling the custom architecture itself.

---

## 2. Hydra effect measurement on gpt-oss-20b

### The paper's core claim

The Hydra effect (distributed redundancy that compensates for ablation) is measured by ablation-effect variance: tight variance = Hydra active, wide variance = exposed circuitry.

### What gpt-oss-interp already measures

The intervention benchmark computes exactly this: per-case accuracy and margin under head/layer ablation. The current results show:

- Layer 20 ablation: margin collapses 89% (from 3.114 to 0.359)
- Head ablation at L2: margin drops 7%
- Head ablation at L12: margin *increases*

These are Hydra-relevant signals. The paper could include a section: "Does gpt-oss-20b exhibit the Hydra effect?" Compute the ablation-effect variance across all 64 heads at a single layer and compare the σ to the PLS vs. control σ values.

**Prediction**: A standard-trained model like gpt-oss-20b should show tight ablation-effect variance (Hydra active), matching the paper's control condition. This is a powerful negative-control demonstration at scale.

---

## 3. Graph representation from forward-pass features

### What the activation clustering paper does

The Tier-2 feature extraction generates a 163-dimensional vector per token from the forward pass:
- Processing depth (convergence layer)
- Confidence trajectory across layers
- Head activation patterns (per layer, per head)
- Head entropy (per layer, per head)

These features define a graph over the input: tokens are nodes, similarity in feature space defines edges. Clustering this graph reveals **computational modes** — groups of tokens that the model processes similarly regardless of surface identity.

### Application to gpt-oss-20b

The gpt-oss-interp activation cache and logit-lens modules capture exactly the raw data needed to compute Tier-2 features:
- `ActivationCache` captures per-layer hidden states → processing depth, confidence
- `run_logit_lens` captures per-layer top-k predictions → convergence trajectory
- Attention hooks would capture head activations → head activation and entropy features

**New capability**: For gpt-oss-20b, the Tier-2 feature space expands significantly because of GQA (64 query heads, 8 KV heads) and MoE (32 experts). The feature vector would include:
- Head activation and entropy: 2 × 24 × 64 = 3,072 dimensions (vs. 2 × 12 × 12 = 288 in the paper)
- Expert routing features: 24 × 32 = 768 dimensions (new — not in the paper)
- Processing depth, confidence, trajectory: ~73 dimensions

This gives a ~3,900-dimensional feature space, which after PCA/dimensionality reduction could reveal computational modes specific to MoE routing.

---

## 4. Inspectability corpus refinement

### The idea

Not all inputs are equally interpretable. Some prompts produce clean, localized computational patterns; others produce smeared, distributed ones. A "corpus refinement" pass would:

1. Run the Tier-2 feature extraction on a large corpus of prompts
2. Cluster the resulting feature vectors
3. Identify clusters with high within-cluster homogeneity and low noise
4. Select representative prompts from clean clusters
5. Use these as the inspectability corpus for deeper analysis

### Why this matters

Current interpretability research picks prompts ad hoc (Winograd sentences, IOI templates, etc.). A principled corpus selection method that maximizes interpretability signal-to-noise would be methodologically significant.

### What gpt-oss-interp enables

The logit-lens convergence layer is a fast proxy for "processing cleanness":
- **Early convergence** (layer 15-18): The model resolves the prediction quickly — likely a clean, well-structured computation
- **Late convergence** (layer 22-24): The model struggles — may involve complex multi-step reasoning or distributional uncertainty
- **No convergence**: The final-layer prediction differs from all intermediate layers — maximally distributed computation

Running logit-lens on thousands of prompts and sorting by convergence layer gives a principled ordering from "most interpretable" to "most opaque."

---

## 5. Efficient probing for undiscovered effects

### The idea

The 163-dimensional feature space is a compressed fingerprint of "how the model processes this token." Outliers in this space — tokens whose feature vectors are far from any cluster centroid — are candidates for **novel computational modes** that haven't been characterized yet.

### Probing protocol

1. Extract Tier-2 features on a large diverse corpus
2. Fit a clustering model (HDBSCAN or similar)
3. Identify noise points (tokens not assigned to any cluster)
4. Examine what's special about these tokens: unusual attention patterns? Rare expert routing? Conflicting per-layer predictions?
5. These outliers become candidates for new interpretability findings

### Application to MoE

For gpt-oss-20b, expert routing adds a new dimension to "unusual processing." A token that activates an unusual set of experts (low cosine similarity to the average routing pattern at that layer) is computationally anomalous and worth investigating.

---

## 6. Distillation from gpt-oss-20b to symbolic transformer

### The core idea

Use gpt-oss-20b's forward-pass data to train a smaller symbolic transformer that preserves computational structure. The key insight: in CASCADE mode, x_t is frozen and lm_head is known, so the teacher's output distribution defines a **closed-form regression target** for the student's contextual stream x_e. This reduces distillation from a generic function-matching problem to structured matrix factorization (see CASCADE_DISTILLATION.md).

### CASCADE mode makes distillation a linear problem

In CASCADE mode, x_t (token embedding) is frozen and W (lm_head) is known. The teacher provides P_teacher(word | context). Since `logits = W · (x_t + x_e)`, we can solve for the optimal x_e directly:

```
x_e* = W⁺ · (z_teacher - W · x_t)
```

where z_teacher ≈ log P_teacher (temperature-scaled). This is Gauss's method of least squares — the student's FFN pathway then learns to generalize these closed-form solutions across contexts. Across a corpus, this is matrix factorization / latent semantic analysis: the teacher's word-context probability matrix factors through the d-dimensional x_e bottleneck.

**Important**: Naive per-layer KL matching (L = Σ_ℓ KL(teacher_layer_ℓ || student_layer_π(ℓ))) does NOT work well across architectures. The literature (OFA-KD, MOHAWK) confirms that architecturally different models organize computation differently — proportional layer mapping is wrong, and even learned stitching layers have limited benefit. See INTERMEDIATE_LAYERS_ANALYSIS.md for the full argument.

### How intermediate layers help (without layer matching)

The teacher's per-layer logit-lens readouts contain useful information, but not as layer-matched supervision. Three defensible uses:

1. **Hard negatives**: Words the teacher considered at intermediate layers but rejected by the final layer provide "plausible but wrong" contrastive signal — richer than random negatives.
2. **Decision subspace**: The PCA of layer-to-layer contrast vectors identifies the low-dimensional subspace of "semantic directions that matter." Regularize the student to compute within this subspace.
3. **Unordered decision constraints**: Each layer-to-layer contrast encodes a semantic decision (e.g., "not big, but small" as a direction in embedding space). The student must traverse all of them somewhere, without prescribing where.

Whether intermediate layers help at all beyond output-only distillation is an empirical question (estimated 50/50). The recommendation is: start with output-only CASCADE distillation, then test hard negatives, then decision subspace regularization.

### Why symbolic transformer is the right student

CASCADE mode makes x_t a fixed reference frame:
- **x_t** is the frozen token embedding — provides stable structure in embedding space
- **x_e** is the only learnable representation — all contextual computation flows through it
- **lm_head** (shared embedding geometry) provides the bridge between teacher and student vocabularies

This means the distillation operates in a shared geometric space. The computational modes discovered by Tier-2 clustering are the latent topics in the matrix factorization — the same structure seen from two perspectives.

### Routing-aware architecture design

gpt-oss-20b's MoE routing patterns (if capturable — requires non-quantized checkpoint) reveal how many "computational strategies" the teacher uses:
1. Cluster the teacher's routing patterns across a corpus
2. If routing clusters into K groups → the student may need K computational pathways
3. K=2: The dual-stream architecture may already suffice
4. K>2: Consider a small student MoE, or accept that the dense student will compress multiple strategies

### Vocabulary mismatch

gpt-oss uses o200k_harmony (201K vocab), the symbolic transformer uses GPT-2 BPE (50K vocab). The CASCADE closed-form approach requires projecting the teacher's distribution into the student's vocabulary space first. This can be done via token-string matching (many subwords overlap) plus a learned residual projection for tokens without direct correspondence.

---

## 7. New paper possibilities

### Option A: Extend the existing PLS paper

Add a section: "Feature extraction at scale: gpt-oss-20b"
- Apply Tier-2 features to gpt-oss-20b
- Show that the same feature space discovers meaningful computational modes
- Measure Hydra-effect variance on gpt-oss-20b as a negative control
- Compare logit-lens convergence profiles between PLS and standard training

This directly addresses the scale limitation without requiring PLS training at scale.

### Option B: New companion paper

"Mechanistic Interpretability of MoE Transformers via Invariance-Constrained Feature Extraction"

**Contributions:**
1. Extend the 163D feature space to MoE architectures (add expert routing features)
2. Discover computational modes in gpt-oss-20b using the extended features
3. Show that MXFP4 quantization collapses interpretability surface area (router opacity)
4. Demonstrate inspectability corpus refinement using convergence-layer sorting
5. Compare computational modes between gpt-oss-20b (standard training) and PLS-trained symbolic transformer — same feature space, different training objectives

### Option C: Distillation paper

"CASCADE Distillation: Structured Matrix Factorization for Interpretable Model Compression"

**Contributions:**
1. CASCADE mode reduces cross-architecture distillation to a linear problem (closed-form x_e targets via pseudoinverse)
2. Teacher's per-layer logit-lens data provides semantic decision directions — usable as hard negatives and subspace constraints without layer correspondence
3. Connection to classical matrix factorization / LSA / topic modeling
4. Mode-distilled students are more interpretable (higher ablation-effect variance) than output-distilled students
5. Steering vectors extracted from teacher's computation trajectory transfer to distilled student

---

## 8. Immediate next steps

### Quick wins (hours, not days)

1. **Port the Tier-2 feature extractor to work with gpt-oss-interp's activation cache**
   - The feature extractor currently assumes symbolic-transformer checkpoints
   - Adapt it to consume `ActivationCache` records and `LogitLensResult` objects
   - This is a significant methodological extension: the feature space expands from 163D to ~7,200D with three new MoE-specific components (expert routing, routing entropy, attention scale). See `gpt_oss_interp/features/extractor.py` (already implemented) and `GEOMETRIC_FRAMEWORK.md` §3

2. **Run Tier-2 features on the logit-lens prompts already tested**
   - "The trophy would not fit in the suitcase because the suitcase was too"
   - "A7 B2 C9 D4 A7 B2 C9"
   - "The keys to the cabinet"
   - Compare the feature vectors across these prompts

3. **Compute ablation-effect variance across all heads at one layer**
   - Pick layer 12 (where we already know there's an interesting signal)
   - Run HEAD_MASK at scale=0.0 for each of the 64 heads individually
   - Compute σ of the ablation effect across heads
   - Compare to the paper's PLS vs. control σ values

### Medium-term (days)

4. **Build a convergence-layer corpus map**
   - Run logit-lens on 100-500 diverse prompts
   - Record convergence layer for the final token of each
   - Visualize the distribution
   - Select the cleanest-converging prompts as the inspectability corpus

5. **Implement CASCADE distillation prototype**
   - Compute closed-form x_e targets from teacher's final distribution via pseudoinverse
   - Train a 6-layer CASCADE symbolic transformer with regression warmup + KL refinement
   - Test whether hard negatives from intermediate layers improve over output-only distillation
   - Compare modularity (ablation-effect variance) to standard training

### Longer-term (weeks)

6. **Full Tier-2 feature extraction on gpt-oss-20b**
   - Extended feature space with MoE routing dimensions
   - Clustering and computational mode discovery
   - Comparison to symbolic-transformer modes

7. **Quantization-interpretability study**
   - Compare feature extraction on MXFP4 vs. bf16 gpt-oss-20b
   - Measure how much interpretability surface area quantization removes
   - Propose quantization-aware interpretability techniques

---

## 9. How this strengthens the OpenAI application

This connection transforms the narrative from:
- "I built custom interpretable architectures" (potentially adjacent)

To:
- "I have a unified methodology for mechanistic interpretability that works on both custom and production models, and I can use production model introspection to guide architectural design"

The gpt-oss-interp work is no longer a standalone demo; it's the **scale validation** arm of a coherent research program that spans architecture design, training objectives, feature engineering, and causal analysis.
