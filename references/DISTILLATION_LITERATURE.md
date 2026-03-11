# Cross-Architecture Distillation — Literature Review

## Why Naive Layer Matching Fails

Simple loss functions that match teacher layer outputs to student layer outputs don't work well across architectures because:
1. Layers learn different representations at different depths
2. Information flow patterns differ between architectures
3. CKA analysis confirms significant feature divergence between heterogeneous architectures
4. At minimum a learned stitching/projection layer is required, but information flow patterns are more sophisticated than that

## FitNets and the Hint Layer Lineage

### FitNets — Romero et al., 2015 (Foundation)
- **Method**: Uses intermediate teacher representations as "hints" to guide student training. Adds a regressor (linear projection) at the guided layer to match teacher width.
- **What worked**: Deeper, thinner students could outperform shallower, wider ones.
- **What didn't**: Assumes same model family. CKA analysis shows significant feature divergence between heterogeneous architectures, making naive hint matching ineffective across architectures.
- **Cost**: ~1.5x standard training (additional forward passes through teacher + regression loss).

## CKA-Based Alignment

### One-for-All Knowledge Distillation (OFA-KD) — NeurIPS 2024
- **Paper**: [OFA-KD](https://arxiv.org/abs/2310.19444)
- **Method**: Projects intermediate features into an aligned latent space (e.g., logits space) where architecture-specific information is discarded. Uses CKA to identify which teacher/student layer pairs are most similar. Introduces adaptive target enhancement to filter irrelevant information.
- **What worked**: Gains of up to **8.0% on CIFAR-100** and **0.7% on ImageNet-1K** across CNN, Transformer, and MLP architectures. Works bidirectionally (Transformer-to-CNN and CNN-to-Transformer).
- **What didn't**: Raw hint-based matching between heterogeneous architectures fails badly (CKA confirms the feature spaces are too divergent).
- **Cost**: Moderate overhead for projection layers + CKA computation.

### Feature-based OFA (FOFA-KD) — 2025 extension
- **Paper**: [FOFA-KD](https://arxiv.org/html/2501.08885v1)
- Adds prompt tuning blocks incorporating student feedback, plus region-aware attention to handle view mismatch between architectures.

### Cross-Architecture Knowledge Distillation — IJCV 2024
- **Paper**: [Cross-Architecture KD](https://link.springer.com/article/10.1007/s11263-024-02002-0)

### Rethinking CKA in Knowledge Distillation — IJCAI 2024
- **Paper**: [Rethinking CKA in KD](https://www.ijcai.org/proceedings/2024/0628.pdf)
- Greater CKA similarity between teacher-student features correlates with better distillation outcomes.
- Adding a strong projector to students increases teacher-student CKA similarity and improves classification.
- However, CKA's effectiveness as a *loss function* (vs. an analysis tool) remains under-explored.

## Contrastive Representation Distillation (CRD)

### CRD — Tian et al. (ICLR 2020)
- **Paper**: [CRD](https://arxiv.org/abs/1910.10699)
- **Method**: Formulates distillation as contrastive learning using InfoNCE loss. Student learns to capture mutual information between teacher representations and data.
- **What worked**: Outperforms standard KD on single model compression, ensemble distillation, and cross-modal transfer.
- **What didn't**: Requires a massive memory buffer of negative samples, significantly increasing training time and memory. Can inadvertently push same-class samples apart in feature space.
- **Cost**: 2-3x standard KD due to memory buffer and contrastive loss computation.

### 2025 Improvements
- *Contrast Enhanced Representation Normalization*: Addresses information redundancy from excessive negative pairs.
- *Multi-Scale Feature Decoupling*: Budget-saving contrastive distillation that reduces memory requirements. [Paper](https://arxiv.org/html/2502.05835v1)
- *Multi-Level KD*: Extends CRD to multiple feature layers rather than just the penultimate one. [Paper](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cit2.70036)

## Attention Transfer Distillation

### Attention-guided Feature Distillation (AttnFD, 2024)
- **Paper**: [AttnFD](https://arxiv.org/abs/2403.05451)
- Uses CBAM (channel + spatial attention) to refine feature maps before matching.
- Multi-stage feature fusion with cross-stage attention mechanisms.

### Cross-attention Bridge Modules (2025)
- Compress transformer token-pairwise softmax interactions into supervision for Mamba or linear-recurrent student models.
- Key enabler for Transformer-to-SSM distillation.

## MoE-to-Dense Distillation

### Key Finding
Distilling MoE back to dense retains only **30-40% of sparsity gains** in general.

### OneS (Multi-teacher from MoE experts)
- Dense student preserves **61.7% of MoE benefits on ImageNet** (78.4% top-1 with only 15M params), **88.2% on NLP**, with **3.7x inference speedup**.

### MoDE — AAAI 2024
- **Paper**: [MoDE: Mixture-of-Experts with Mutual Distillation](https://arxiv.org/abs/2402.00893)
- Uses mutual distillation *among* experts during MoE training, improving per-expert task awareness.

### OFA-KD with MoE
- Leverages Mixture-of-Experts *as the bridge mechanism* for cross-architecture distillation (not just MoE as the teacher).

## MOHAWK: Distilling Computational Structure (Most Relevant)

### MOHAWK — NeurIPS 2024
- **Paper**: [MOHAWK Framework](https://goombalab.github.io/blog/2024/distillation-part1-mohawk/) | [Transformers to SSMs](https://arxiv.org/abs/2408.10189)
- **Method**: Three-stage bottom-up distillation from Transformer to SSM (Mamba-2):
  1. **Matrix Orientation**: Aligns the matrix mixer (attention matrix vs. SSM recurrence matrix) using Frobenius norm loss.
  2. **Hidden-State Alignment**: Matches block-level outputs using L2 loss.
  3. **Weight Transfer + End-to-End KD**: Transfers MLP/embedding/LM-head weights, then fine-tunes with cross-entropy + KD loss.
- **What worked**: Phi-Mamba distilled using only **3B tokens** (~0.1% of typical pre-training data). Outperforms all prior open-source non-Transformer models despite massive data efficiency. Freezing MLPs after weight transfer reduces trainable params by 50%+ with minimal performance loss.
- **What didn't**: Requires architecture compatibility (teacher and student must share everything except the mixer). Convolutions initialized to identity and Mamba gates set to 1 to enable weight transfer.
- **Cost**: Dramatically cheaper than training from scratch (< 1% of training data). Higher complexity than standard KD due to three-stage pipeline.

### Llamba — 2025
- **Paper**: [Llamba](https://arxiv.org/html/2502.14458)
- Extends MOHAWK, achieving strong results with < 0.1% of typical training data.

## Redundancy Suppression

### Redundancy Suppression Distillation (RSD, 2025)
- **Paper**: [RSD](https://arxiv.org/html/2507.21844v1)
- Casts cross-architecture KD as redundant information suppression.
- Works as a strong logit distiller, outperforming existing logit distillation objectives individually.

### PCA Projectors
- Map CNN feature maps into Transformer-like query/key/value spaces, mimicking global dependencies.
- Combined with contrastive objectives (InfoNCE) after spatial smoothing to handle spatial misalignment between CNNs and Transformers.

## Practical Lessons: What Actually Works

From the comprehensive 2025 survey (TMLR): [A Comprehensive Survey on Knowledge Distillation](https://arxiv.org/pdf/2503.12067)

1. **Feature distillation > logit distillation** for cross-architecture scenarios, because logits discard structural information.
2. **Projection layers are essential** when architectures differ — raw feature matching fails (CKA confirms divergence).
3. **Mid-layer matching** outperforms early or late layer matching; early layers are too generic, late layers too task-specific.
4. **Staged/progressive distillation** (MOHAWK-style) outperforms single-stage approaches.
5. **Weight transfer where possible** provides a strong initialization — if teacher and student share any components (MLPs, embeddings), reuse them.
6. **Dual/multi-teacher strategies** yield up to +5.95% improvement when architectures differ (e.g., using both a ViT and CNN teacher).
7. **CKA as analysis tool > CKA as loss**: CKA helps identify which layers to align but is less well-validated as a training objective.
8. **Contrastive losses help but are expensive**: CRD-style approaches improve quality but 2-3x the cost. Recent budget-saving variants reduce this.
9. **Cross-architecture gap is real but closeable**: With the right techniques, dense students can recover 60-90% of teacher performance even across architecture families.

## Summary Table

| Method | Mechanism | Gain vs Baseline | Cost vs Standard KD | Key Limitation |
|--------|-----------|-----------------|---------------------|----------------|
| FitNets | Linear projection at hint layer | ~2-5% | ~1.5x | Same-family only |
| OFA-KD | Logit-space projection + adaptive targets | Up to 8% (CIFAR-100) | ~1.5-2x | Requires CKA layer selection |
| CRD | Contrastive (InfoNCE) on representations | 2-4% over KD | 2-3x | Memory-hungry, can push same-class apart |
| MOHAWK | 3-stage: mixer alignment, block matching, E2E | Matches scratch training with <1% data | Higher complexity but far less data | Requires shared non-mixer components |
| RSD | Redundancy suppression formulation | Outperforms individual logit distillers | ~1.5x | Newer, less validated |
| PCA Projectors | Q/K/V projection + InfoNCE | Bridges CNN-Transformer gap | ~2x | Spatial smoothing needed |
