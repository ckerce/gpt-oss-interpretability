# Steering Vectors in Transformers — Literature Review

## Foundational Approaches

### Activation Addition (ActAdd) — Turner et al., 2024
- **Paper**: [Activation Addition: Steering Language Models Without Optimization](https://arxiv.org/abs/2308.10248)
- **Method**: Compute a "steering vector" by contrasting intermediate activations on prompt pairs (e.g., "Love" vs. "Hate"), then add this vector to the residual stream during inference. No weight modification or optimization required.
- **What worked**: State-of-the-art results on sentiment shift and detoxification on LLaMA-3 and OPT. Preserves performance on off-target tasks when the layer and scaling coefficient are well-chosen.
- **What didn't**: Requires read/write access to intermediate activations (not available via commercial APIs). Optimal layer and coefficient selection is non-trivial.
- **Cost**: Negligible at inference time (single vector addition). Training cost is just forward passes over contrast pairs.

### Contrastive Activation Addition (CAA) — Rimsky et al., 2024 (ACL)
- **Paper**: [Steering Llama 2 via Contrastive Activation Addition](https://aclanthology.org/2024.acl-long.828/)
- **Method**: Extends ActAdd by averaging residual stream activation differences over hundreds/thousands of contrast pairs (multiple-choice questions with appended answer letters indicating desired vs. undesired behavior). Uses mean-difference rather than single-pair vectors.
- **What worked**: Mean-difference method consistently outperforms PCA-based and classifier-based alternatives by a large margin. Theorem 3.1 in a 2025 unified evaluation paper proves mean-difference minimizes ||h+ - h- - v||^2.
- **What didn't**: PCA of differences (RepE-style) performed poorly because positive/negative embeddings vary along a direction nearly orthogonal to the steering vector in many cases.
- **Cost**: Slightly higher than ActAdd (need ~hundreds of contrast pairs), but still no optimization — just forward passes + averaging.

### Representation Engineering (RepE) — Zou et al., 2023
- **Paper**: [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405)
- **Method**: Draws on cognitive neuroscience. Uses Linear Artificial Tomography (LAT) to identify directions in representation space that correlate with cognitive functions (honesty, harmlessness, power-seeking). Intervenes by adding/subtracting fixed directions from activations.
- **What worked**: Truthfulness steering increased TruthfulQA accuracy by up to 30 percentage points. Circuit breakers (a RepE application) proved effective for safety.
- **What didn't**: PCA-based vector extraction is theoretically less grounded than mean-difference (see unified evaluation above).
- **Cost**: Similar to CAA — forward passes over contrastive prompts, no gradient-based optimization.

## How Steering Vectors Are Computed

The dominant methodology:
1. **Contrast pairs**: Construct prompt pairs where one ends with desired behavior, the other with undesired. Hundreds to thousands of pairs reduce noise.
2. **Forward pass**: Run both prompts through the model, extract activations at a chosen layer (typically middle layers — e.g., layer 13 for a 32-layer model).
3. **Mean difference**: Compute the average activation for positive examples minus the average for negative examples. This is the steering vector.
4. **PCA variant**: Alternatively, compute pairwise differences and take the first principal component. However, the unified evaluation shows this underperforms mean-difference.
5. **Application**: Add the scaled vector to the residual stream at the chosen layer during inference.

## Application to Small/Custom Transformers

- The **steering-vectors** PyPI library provides a Pytorch/HuggingFace-compatible framework that works with any transformer model, not just frontier models. [steering-vectors on PyPI](https://pypi.org/project/steering-vectors/0.4.0/) | [GitHub](https://github.com/steering-vectors/steering-vectors)
- **EasySteer** (2025) benchmarked steering on models as small as **DeepSeek-R1-Distill-Qwen-1.5B** and **Qwen2.5-1.5B-Instruct**, demonstrating that steering vectors work on small models with 5.5-11.4x speedup over prior frameworks. [EasySteer](https://arxiv.org/abs/2509.25175)
- The linear representation hypothesis underlying steering vectors is architecture-agnostic — it should apply to any transformer with a residual stream.
- No papers found specifically applying steering vectors to custom/toy transformers for research purposes, though the existing libraries and frameworks make this straightforward.

## When Steering Vectors Work vs. Fail

### Failure Modes (Durmus et al., NeurIPS 2024)
- **Paper**: [Analysing the Generalisation and Reliability of Steering Vectors](https://proceedings.neurips.cc/paper_files/paper/2024/hash/fb3ad59a84799bfb8d700e56d19c231b-Abstract-Conference.html)
- **Anti-steerability**: For several concepts, nearly **50% of inputs are anti-steerable** — the steering vector produces the *opposite* of the intended effect. This makes reliability deeply problematic.
- **Bimodal steerability**: Per-sample steerability distributions are often bimodal, not Gaussian. Some inputs are strongly positively steerable, others strongly negatively.
- **Spurious biases**: Token/position biases (e.g., preference for answer "A" vs "B") contribute substantially to measured steerability, but most variance remains unexplained.
- **Capability degradation**: Steering decreases accuracy by up to **53.25%** on examples that were already correctly aligned. In some cases, degradation is equivalent to halving pre-training compute.
- **Many behaviors are unsteerable**: Even sweeping across all layers and strengths, some concepts cannot be reliably steered.

### What Generalizes
- In-distribution and out-of-distribution steerability correlate (rho=0.891 for Llama, rho=0.694 for Qwen), but the correlation is imperfect.
- Concepts with clear linear separability in activation space (verifiable via logistic regression on PCA-projected activations) tend to be more steerable.

### Practical Recommendations
- **Paper**: [A Unified Understanding and Evaluation of Steering Methods (2025)](https://arxiv.org/html/2502.02716v1)
- Use mean-difference, not PCA.
- Apply at middle layers in the residual stream.
- Avoid blanket application; selective/controlled steering improves outcomes.
- Consider **Low-Rank Representation Steering (LoReSt)** (NeurIPS 2024) as an alternative that intervenes via a low-rank projection matrix rather than a single vector, overcoming some brittleness. [LoReSt](https://neurips.cc/virtual/2024/104073)

## Connection to Gated Attention

No direct paper connecting steering vectors to gated attention mechanisms. Two relevant threads:

- **Gated Attention (NeurIPS 2025 Best Paper)**: Introduces a gating projection that determines how much attention output to preserve. The gating mechanism controls information flow in a way conceptually related to steering (both modulate what flows through the residual stream), but no one has explicitly combined them. [Gated Attention for LLMs](https://arxiviq.substack.com/p/neurips-2025-gated-attention-for)
- **Composable steering via gating**: EasyEdit2 and Steer-MoE use dimensionwise gating (e.g., TIES merging) to compose multiple steering vectors, which is a form of learned gating over steering directions.
- **Gated Linear Attention Transformers**: [Paper](https://arxiv.org/abs/2312.06635)

## Additional References

- [A Sober Look at Steering Vectors for LLMs](https://www.alignmentforum.org/posts/QQP4nq7TXg89CJGBh/a-sober-look-at-steering-vectors-for-llms)
- [Representation Engineering Survey (2025)](https://arxiv.org/html/2502.17601v1)
- [Shifting Perspectives: Steering for Bias Mitigation](https://arxiv.org/html/2503.05371v2)
- [Personalized Steering via Bi-directional Preference Optimization (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/58cbe393b4254da8966780a40d023c0b-Abstract-Conference.html)

## Summary Table

| Method | Computation | Best For | Failure Mode | Cost |
|--------|------------|----------|--------------|------|
| ActAdd (Turner) | Single contrast pair difference | Quick proof-of-concept | Noisy, single-pair dependent | Minimal |
| CAA (Rimsky) | Mean difference over hundreds of pairs | Production steering | ~50% anti-steerable inputs for some concepts | Low (forward passes only) |
| RepE (Zou) | PCA of contrast activations | Safety interventions | PCA direction misaligns with behavior direction | Low |
| LoReSt | Low-rank projection matrix | Overcoming single-vector brittleness | Less studied at scale | Moderate (requires optimization) |
| BiPO | Bi-directional preference optimization | Personalized steering | Newer, less validated | Moderate |
