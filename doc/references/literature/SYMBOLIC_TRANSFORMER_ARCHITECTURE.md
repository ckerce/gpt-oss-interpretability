# Symbolic Transformer Architecture Reference

## Source Code Locations

| Aspect | File Path |
|--------|-----------|
| Main transformer | `companion-repo/neuro-symb-v2/src/symbolic_transformer/model/transformer.py` |
| Attention (gated) | `companion-repo/neuro-symb-v2/src/symbolic_transformer/model/attention.py` |
| FFN & PLS | `companion-repo/neuro-symb-v2/src/symbolic_transformer/model/ffn.py` |
| Loss computation | `companion-repo/neuro-symb-v2/src/symbolic_transformer/model/loss.py` |
| Mixing strategies | `companion-repo/neuro-symb-v2/src/symbolic_transformer/model/mixing.py` |
| Configs | `companion-repo/neuro-symb-v2/src/symbolic_transformer/model/config.py` |
| Training configs | `companion-repo/neurips-2026-activation-clustering/configs/training/series_gated_attention/` |
| Training script | `companion-repo/neurips-2026-activation-clustering/training/train_grade_school.py` |
| Trainer class | `companion-repo/neuro-symb-v2/src/trainer.py` |

## Dual-Stream Architecture (x_t and x_e)

### Stream Definitions
- **x_t (symbolic/token stream)**: Initialized from vocabulary embeddings at the first embedding layer. Represents the "token-space structure" and is updated by attention operations.
- **x_e (contextual/embedding stream)**: Initialized as zeros at the model input. Updated only by FFN operations. Represents context accumulated from feed-forward transformations.

### Information Flow Between Streams

```python
# In SymbolicTransformerBlock.forward():
# 1. ATTENTION PATH
combined = x_t + x_e              # Combine at input normalization
attn_out = attention(ln1(combined), xt_for_v)
x_t_updated = x_t + attn_out      # x_t receives attention updates

# 2. FFN PATH
combined = x_t_updated + x_e      # Combine again
ffn_out, interp = ffn(ln2(combined), ...)
x_e_updated = x_e + ffn_out       # x_e receives FFN updates
```

**Key interaction point**: Both streams observe the combined signal `x_t + x_e` via layer normalization, but updates are segregated: attention writes to x_t, FFN writes to x_e. The final output combines them: `x = ln_f(x_t + x_e)`.

## CASCADE Training Mode

### What Gets Frozen
- The symbolic stream (x_t) is **frozen after the embedding layer** when `stream_update_mode = StreamUpdateMode.CASCADE`
- x_t remains at its initial embedding value throughout all transformer blocks
- No attention updates propagate into x_t in CASCADE mode

### What Trains
- The contextual stream (x_e) trains normally and receives all FFN updates
- All attention, FFN, and normalization parameters train
- The embedding layer itself remains trainable (unless explicitly frozen with `freeze_vocab_embeddings`)

### Effect on Information Flow

```python
if self.stream_mode == StreamUpdateMode.CASCADE:
    x_t_updated = x_t  # Frozen - no update
    x_e = x_e + attn_out  # Attention updates go to x_e instead
```

The attention output instead updates **x_e** in CASCADE mode, meaning the frozen symbolic structure is "contextualized" by the embedding stream rather than modified.

### Optional CLN at Init
When `cascade_cln_at_init=True`, x_t receives a single channel-wise layer normalization at initialization, providing normalized token-space structure.

## Gated Attention Mechanism

### Architecture (G1 position — SDPA output gating)

Located in `attention.py` (lines 103-125, 297-324):

```python
# Gate parameters: (n_head, head_dim, head_dim)
self.gate_weight = nn.Parameter(torch.zeros(self.n_head, self.head_dim, self.head_dim))

# Gate input selection (based on config.gated_attention_input):
if gated_attention_input == "combined":
    gate_input = q  # From CLN(x_t + x_e) via Q projection
else:  # "symbolic"
    gate_input = v_input  # From CLN(x_t)

# Vectorized gating via einsum:
gate_logits = torch.einsum('bhtd,hde->bhte', gate_input, self.gate_weight)
gate = torch.sigmoid(gate_logits)
y = y * gate  # Element-wise modulation of attention output
```

### Gate Learning
- Gate weights initialized to zeros by default (`gated_attention_init="zeros"`)
- This means `sigmoid(0) ≈ 0.5`, providing mild initial modulation
- Can be initialized to ones for stronger initial gating
- Gates are **learned per-head** with shape `(n_head, head_dim, head_dim)` per layer

### Gate Input Options
- `"combined"`: Uses query projections from normalized combined stream (more standard)
- `"symbolic"`: Uses only the symbolic stream for gate computation (for interpretability ablations)

### Parameter Count
With 12 layers and 12 heads × 64 head_dim, gated attention adds `12 × 12 × 64 × 64 ≈ 590K` parameters per layer.

## Per-Layer Supervision (PLS) Implementation

### Auxiliary Head Placement
- No separate heads are created; instead, the **single shared `lm_head`** is reused at each layer
- Intermediate representations at layer i are projected: `logits_i = lm_head(ln_f(x_t + x_e))`
- This is memory-efficient and tests whether intermediate states are "language-model-head compatible"

### Layer Loss Combination

From `loss.py`:

```python
def get_layer_weight(layer_idx, n_layers, decay, base_weight, final_layer_weight=None):
    if decay == "linear":
        normalizer = n_layers * (n_layers + 1) / 2
        layer_weight = (layer_idx + 1) / normalizer * base_weight
    elif decay == "exponential":
        alpha = 2.0
        ...
    elif decay == "linear_final":
        if layer_idx == n_layers - 1:
            layer_weight = final_layer_weight
        else:
            normalizer = (n_layers - 1) * n_layers / 2
            layer_weight = (1.0 - final_layer_weight) * (layer_idx + 1) / normalizer
```

### Configuration Options
- `per_layer_supervision: bool` — Enable/disable
- `per_layer_loss_weight: float` (0-1) — Total weight allocated to all PLS losses
- `per_layer_loss_decay: str` — Strategy: "linear", "exponential", "uniform", or "linear_final"
- `per_layer_decode_source: str` — Which representation to use: "auto" (x_t+x_e), "x_e", "combined", or "decode_output"
- `use_decode_pathway: bool` — Whether to use separate FFN decode projection

### Memory-Efficient Mode
When targets are provided to `forward()`, per-layer CE losses are computed immediately and discarded, avoiding OOM from storing all layer logits.

## Model Dimensions and Configurations

### Core Parameters
```python
vocab_size: int = 50257           # Vocabulary size
n_layer: int = 12                 # Number of transformer blocks
n_head: int = 12                  # Number of attention heads
n_embd: int = 768                 # Total embedding dimension (= n_head × head_dim)
head_dim: int = n_embd / n_head   # Per-head dimension (computed in __post_init__)
```

### Derived Parameters
- `ffn_hidden_dim = n_embd × ffn_hidden_multiplier` (default 4.0, so 3072 for 768 base)
- `ffn_hidden_dim_per_head = ffn_hidden_dim / n_head` (256 for 12 heads × 768 embd)

### Example Configurations

| Config | n_head | n_layer | head_dim | n_embd | Parameters | Notes |
|--------|--------|---------|----------|--------|------------|-------|
| S1 (SINGLE_STREAM) | 6 | 6 | 86 | 516 | ~3.2M | Dense baseline |
| A2 (CASCADE) | 6 | 6 | 92 | 552 | ~3.5M | Independent attention |
| B1 (CASCADE) | 6 | 6 | 92 | 552 | ~3.5M | Kronecker attention |
| 80M (Codelion) | 12 | 12 | 50 | 600 | ~82M | Large-scale |

### Additional Hyperparameters
- `max_seq_len: int = 2048` — Maximum sequence length
- `dropout: float = 0.1` — Dropout rate
- `bias: bool = True` — Use bias in linear layers
- `ffn_hidden_multiplier: float = 4.0` — FFN expansion ratio
- `positional_encoding: str = "rope"` — RoPE or ALiBi
- `causal: bool = True` — Causal (language modeling) vs bidirectional

## Training Loop and Losses

### Loss Components

1. **Main Loss**: Final layer cross-entropy
   ```python
   main_loss = F.cross_entropy(logits, targets, ignore_index=-100)
   ```

2. **Per-Layer Supervision Loss** (if enabled):
   ```python
   for layer_interp in interpretations:
       layer_loss = F.cross_entropy(layer_logits, targets, ignore_index=-100)
       weight = get_layer_weight(i, n_layers, decay, base_weight)
       per_layer_loss += weight * layer_loss
   ```

3. **Auxiliary Losses** (from vocabulary projection, if enabled):
   - `vocab_l2_regularization`: L2 distance from constrained vocab projection
   - `vocab_entropy_loss`: Entropy of attention over vocabulary

4. **Total Loss**:
   ```python
   total_loss = main_loss + per_layer_loss + l2_loss + entropy_loss
   ```

### Training Schedule
- **Optimizer**: AdamW with `betas=(0.9, 0.95)`, `weight_decay=0.1`
- **LR Schedule**: Cosine annealing with linear warmup
- **Mixed Precision**: Optional AMP with GradScaler
- **Gradient Clipping**: Norm clipping at 1.0
- **Gradient Accumulation**: Configurable steps before optimizer step

## Residual Stream Structure

### Pre-Norm Design

```python
# Attention block
combined = x_t + x_e
ln_out = self.ln1(combined)           # ChannelLayerNorm per head
attn_out = self.attn(ln_out, xt_for_v=...)
x_t = x_t + attn_out                  # Residual connection

# FFN block
combined = x_t + x_e
ln_out = self.ln2(combined)
ffn_out, interp = self.ffn(ln_out, ...)
x_e = x_e + ffn_out                   # Residual connection
```

### ChannelLayerNorm (Special Normalization)

```python
class ChannelLayerNorm(nn.Module):
    """Applies LayerNorm independently per attention head."""
    def forward(self, x):  # x: (B, T, H, head_dim)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        x_norm = (x - mean) / sqrt(var + eps)
        return x_norm * self.weight + self.bias  # Per-head affine
```

### Final Output
```python
x = self.ln_f(x_t + x_e)  # Final combination
logits = self.lm_head(x)   # Decode to vocabulary
```

## Architecture Nomenclature

Config names follow the pattern: `{attn}-{ffn}_{stream}[_{gate}]`

- **attn**: Attention mixing — `dns` (dense), `ind` (independent), `kron` (kronecker)
- **ffn**: FFN mixing — `dns` (dense), `ind` (independent)
- **stream**: `S` (single), `C` (cascade), `T` (token-factor)
- **gate** (optional): `G` (combined gate), `Gs` (symbolic gate)

Example: `dns-dns_S_G_cascade_pls_xe` = Dense attention, Dense FFN, Single stream with combined gating, Cascade mode, Per-layer supervision using x_e.

## No Existing Distillation Code

No distillation-specific code exists in the repository. The codebase focuses on per-layer supervision, optional decode pathways, vocabulary projection regularization, and standard KV caching for inference.
