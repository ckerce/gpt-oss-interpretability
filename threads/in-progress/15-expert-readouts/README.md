# Thread 15: MoE Expert Readouts

**Status**: in-progress
**Model**: `gpt-oss-20b`
**Question**: What do the MoE experts in gpt-oss-20b specialize in?

---

## Scientific question

`gpt-oss-20b` has 32 experts per MoE layer across 24 layers — 768 expert modules in total. At each token, 4 are selected. Do the experts divide up the work along interpretable axes (syntax vs. semantics, early vs. late convergence, specific token types)? Or is expert selection diffuse and uninterpretable?

Three measurements address this at increasing resolution:

| Measurement | Tool | MXFP4-safe? |
|---|---|---|
| Routing patterns (utilization, entropy) | Sidecar | Yes |
| Layer logit-delta (what each MoE block writes) | ActivationCache | Yes |
| Expert vocabulary profiles (what each expert predicts) | ExpertCapture | Non-quantized only |

---

## Measurement 1: Routing patterns

The sidecar recovers routing decisions even when MXFP4 fuses the router kernel. For each prompt and each token position, we record which 4 of 32 experts were selected at each layer. Across a corpus of prompts, this gives:

- **Expert utilization**: frequency each expert is selected. Uniform = generalist. Concentrated = specialist.
- **Routing entropy** per layer: `H = -sum(p_e * log p_e)` where `p_e = fraction of tokens routed to expert e`. Low entropy = routing is concentrated; high entropy = routing is diffuse.
- **Task-family routing correlation**: do different task families (capitalization, induction, coreference) systematically prefer different experts?

A uniform router would give entropy `log(32) ≈ 3.47` nats. Load balancing losses push toward this. Any structured deviation reveals specialization.

---

## Measurement 2: Layer logit-delta

Each transformer block adds a delta to the residual stream:

```
h_l = h_{l-1} + attn_delta_l + moe_delta_l
```

The logit-lens projects each `h_l` through `final_norm + lm_head` to recover a vocabulary distribution `p_l`. The per-layer **logit-delta** is:

```
Δ log p_l(pos) = log p_l(pos) - log p_{l-1}(pos)
```

This shows which tokens each layer **promotes** or **suppresses** at each token position — the vocabulary signature of what that layer contributes, not what it has accumulated. For MoE layers specifically, `Δ log p_l` reflects the combined contribution of the selected experts after gating.

Key questions:
- Do early MoE layers (L1–8) contribute surface-level token adjustments (capitalization, punctuation)?
- Do late MoE layers (L17–21) contribute task-relevant promotions?
- Is the delta concentrated on a few tokens (targeted) or diffuse (distributed)?

---

## Measurement 3: Expert vocabulary profiles (non-quantized)

For non-quantized checkpoints where individual expert modules are Python-visible, `ExpertCapture` hooks each expert's forward call and captures its output tensor before gating. These outputs are projected through `final_norm + lm_head` to recover a vocabulary distribution for each expert activation.

Averaged across all activations of expert `e` at layer `l`:

```
profile(l, e) = E_{tokens routed to e}[ softmax( lm_head( final_norm( expert_e_output ) ) ) ]
```

This is the expert's **vocabulary profile** — the probability distribution it produces over the vocabulary when it fires.

Expected findings based on similar analyses in smaller MoE models:
- Early-layer experts may specialize by token surface form (punctuation, casing, function words vs. content words)
- Mid-layer experts may specialize by syntactic role (verbs, nouns, prepositions)
- Late-layer experts (L17–21, the causal bottleneck identified in Thread 2) may have the most semantically meaningful profiles

---

## Outputs

```
runs/expert_readouts/
├── routing_patterns.json          # {layer: {expert: token_count}} per task family
├── routing_entropy.json           # routing entropy per layer
├── layer_logit_delta.json         # top-promoted tokens per layer per position
├── expert_vocab_profiles.json     # {layer: {expert: [(token, logp), ...]}}  (non-quantized)
├── expert_readout_report.md       # Markdown summary
└── figures/
    ├── fig_routing_heatmap.png    # Layer × expert utilization heatmap
    ├── fig_routing_entropy.png    # Entropy per layer (bar chart)
    ├── fig_logit_delta.png        # Top-token delta per layer
    └── fig_expert_profiles.png    # Expert vocabulary profile matrix (non-quantized)
```

---

## Key relationship to other threads

- **Thread 1 (logit-lens)**: Measurement 2 is the *derivative* of Thread 1's logit-lens — per-layer change rather than cumulative state.
- **Thread 2 (late-layer ablation)**: Measurement 1 and 2 should show qualitatively different routing and delta patterns at L19–21 vs. L0–8.
- **Sidecar (Phase 2 engineering)**: Thread 15 is the primary consumer of `gossh.sidecar`'s routing API in a research context.

---

## Limitations

- **MXFP4**: Measurement 3 (expert vocabulary profiles) requires a non-quantized checkpoint. The sidecar provides routing but not individual expert outputs.
- **Load balancing**: The model was trained with auxiliary load-balancing losses. Expert utilization will be closer to uniform than in models without this constraint. Deviations from uniform are therefore more informative, not less.
- **Token selection bias**: Expert profiles are conditioned on which tokens were routed there, which is itself a function of the model's routing policy. Expert A looking like it "knows about punctuation" could mean it causes punctuation-related activations, or that punctuation-like hidden states are routed to it, or both.
