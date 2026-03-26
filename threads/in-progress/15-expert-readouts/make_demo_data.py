"""Generate synthetic but realistic demo outputs for Thread 15.

Produces JSON files that generate_expert_figures.py and write_report() can
consume — no model or GPU required.  The synthetic data is calibrated to
match the statistical properties expected from gpt-oss-20b:

- 24 layers, 32 experts, top-4 routing
- Routing entropy slightly below uniform log(32) ≈ 3.47 nats (load balancing
  keeps it close but deviations mark specialist experts)
- Late-layer routing more concentrated (lower entropy) than early layers
- Three "specialist" experts per layer with 2–4× above-uniform utilization
- Logit-delta pattern: early layers adjust surface tokens; late layers (L19-21)
  promote content/task-specific tokens
- Expert vocab profiles show token-surface specialisation in early layers,
  semantic specialisation in late layers

Usage::

    python make_demo_data.py [--output runs/expert_readouts_demo]
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


SEED = 42
N_LAYERS = 24
N_EXPERTS = 32
TOP_K = 4
TASK_FAMILIES = ["induction", "coreference", "capitalization", "recency", "syntax"]

# Tokens we'll use to make expert profiles readable
SURFACE_TOKENS = [",", ".", "!", "?", ";", ":", "'", '"', "(", ")", "-", "\n", " "]
FUNCTION_WORDS = ["the", "a", "an", "is", "was", "be", "to", "of", "and", "in", "that", "it"]
CONTENT_WORDS = ["lion", "trophy", "cabinet", "Paris", "doctor", "key", "bank", "letter"]
SYNTAX_TOKENS = ["which", "who", "whom", "that", "where", "when", "because", "although"]
ANSWER_TOKENS = ["large", "small", "true", "false", "yes", "no", "A", "B", "I", "We"]
DIGIT_TOKENS = ["1", "2", "3", "10", "100", "2023", "42"]


def _entropy(weights: list[float]) -> float:
    return -sum(w * math.log(w + 1e-12) for w in weights if w > 0)


def make_routing_patterns(rng: random.Random) -> dict:
    """Routing patterns: {family: {layer: {expert: token_count}}}."""
    uniform_share = 100  # tokens per expert under perfect load balancing

    # Assign 3 "specialist" experts per layer — same across families with small variation
    specialists: dict[int, list[int]] = {}
    for li in range(N_LAYERS):
        specialists[li] = rng.sample(range(N_EXPERTS), 3)

    out = {}
    for family in TASK_FAMILIES:
        layer_data: dict[str, dict[str, int]] = {}
        for li in range(N_LAYERS):
            expert_counts: dict[str, int] = {}
            # Distribute ~3200 tokens across 32 experts
            base = [uniform_share] * N_EXPERTS
            for si in specialists[li]:
                boost = rng.randint(80, 160)  # 1.8–2.6× boost
                base[si] += boost
                # Compensate by reducing neighbours
                for _ in range(boost):
                    victim = rng.randint(0, N_EXPERTS - 1)
                    if base[victim] > 10:
                        base[victim] -= 1
            # Family-specific noise
            noise = [max(0, b + rng.randint(-20, 20)) for b in base]
            expert_counts = {str(ei): noise[ei] for ei in range(N_EXPERTS)}
            layer_data[str(li)] = expert_counts
        out[family] = layer_data
    return out


def make_routing_entropy(routing_patterns: dict) -> dict:
    """Per-layer routing entropy averaged across families."""
    entropy_sum = [0.0] * N_LAYERS
    for family_data in routing_patterns.values():
        for li in range(N_LAYERS):
            counts = family_data[str(li)]
            total = sum(counts.values()) or 1
            weights = [v / total for v in counts.values()]
            entropy_sum[li] += _entropy(weights)
    n_fam = len(routing_patterns)
    # Late layers (L17-23) get a concentration penalty: routing is slightly
    # more specialised near the causal bottleneck
    out = {}
    for li in range(N_LAYERS):
        avg = entropy_sum[li] / n_fam
        late_penalty = max(0.0, (li - 16) * 0.04)  # up to ~0.28 nats lower
        out[str(li)] = round(avg - late_penalty, 4)
    return out


def make_layer_logit_delta(rng: random.Random) -> dict:
    """Top-promoted / suppressed tokens per layer per position.

    Schema: {layer_str: {pos_str: {"promoted": [(tok, delta), ...],
                                    "suppressed": [(tok, delta), ...]}}}
    """
    n_positions = 12  # typical prompt length for demo

    # Early layers promote surface tokens; late layers promote semantic tokens
    early_vocab = SURFACE_TOKENS + FUNCTION_WORDS
    mid_vocab = FUNCTION_WORDS + SYNTAX_TOKENS
    late_vocab = CONTENT_WORDS + ANSWER_TOKENS

    out: dict = {}
    for li in range(N_LAYERS):
        pos_data: dict = {}
        progress = li / (N_LAYERS - 1)  # 0.0 → 1.0
        vocab = (
            early_vocab if progress < 0.35
            else late_vocab if progress > 0.70
            else mid_vocab
        )
        for pi in range(n_positions):
            # Delta magnitude grows toward the decision layer (L19-21)
            peak_delta = 0.3 + 1.8 * math.exp(-((li - 20) ** 2) / 8)
            promoted = []
            suppressed = []
            for _ in range(5):
                tok = rng.choice(vocab)
                delta = round(rng.uniform(0.05, peak_delta), 3)
                promoted.append([tok, delta])
            for _ in range(5):
                tok = rng.choice(early_vocab)
                delta = round(rng.uniform(-peak_delta, -0.05), 3)
                suppressed.append([tok, delta])
            pos_data[str(pi)] = {
                "promoted": sorted(promoted, key=lambda x: -x[1]),
                "suppressed": sorted(suppressed, key=lambda x: x[1]),
            }
        out[str(li)] = pos_data
    return out


def make_expert_vocab_profiles(rng: random.Random) -> dict:
    """Per-expert vocabulary profiles.

    Schema: {layer_str: {expert_str: [(token, logp), ...]}}

    Early-layer experts: surface/punctuation tokens dominate
    Mid-layer experts: syntax tokens dominate
    Late-layer experts (esp. L19-21): semantic/answer tokens dominate
    """
    # Token pools by layer zone
    def token_pool(li: int) -> list[str]:
        progress = li / (N_LAYERS - 1)
        if progress < 0.30:
            return SURFACE_TOKENS + FUNCTION_WORDS
        elif progress < 0.65:
            return FUNCTION_WORDS + SYNTAX_TOKENS
        else:
            return CONTENT_WORDS + ANSWER_TOKENS

    # Each expert has a "speciality" — a token it strongly prefers
    expert_specialty: dict[tuple[int, int], str] = {}
    for li in range(N_LAYERS):
        pool = token_pool(li)
        for ei in range(N_EXPERTS):
            expert_specialty[(li, ei)] = rng.choice(pool)

    out: dict = {}
    for li in range(N_LAYERS):
        pool = token_pool(li)
        layer_data: dict = {}
        for ei in range(N_EXPERTS):
            specialty = expert_specialty[(li, ei)]
            # Build a logp profile: specialty token gets the highest logp
            tokens = [specialty] + rng.sample([t for t in pool if t != specialty],
                                               min(9, len(pool) - 1))
            # Assign descending logps with specialty at top
            base_logp = rng.uniform(-2.0, -0.5)
            profile = []
            for rank, tok in enumerate(tokens):
                logp = round(base_logp - rank * rng.uniform(0.3, 0.8), 3)
                profile.append([tok, logp])
            layer_data[str(ei)] = sorted(profile, key=lambda x: -x[1])
        out[str(li)] = layer_data
    return out


def write_report(out_dir: Path, routing_entropy: dict, layer_logit_delta: dict) -> None:
    """Write a sample markdown report."""
    # Find the layer with lowest entropy (most concentrated routing)
    min_ent_layer = min(routing_entropy, key=lambda k: routing_entropy[k])
    min_ent = routing_entropy[min_ent_layer]
    uniform_ent = round(math.log(N_EXPERTS), 3)

    # Find the layer with largest logit delta magnitude
    max_delta_layer = "20"  # hardcoded to match synthetic peak

    report = f"""\
# Thread 15 — MoE Expert Readout Report (DEMO)

**Note**: This report was generated from synthetic data calibrated to gpt-oss-20b
statistics. Run `run_expert_analysis.py` against a real model checkpoint for
empirical results.

---

## Measurement 1: Routing patterns

- **Uniform entropy baseline**: log(32) ≈ {uniform_ent} nats
- **Observed range**: {min(routing_entropy.values()):.3f} – {max(routing_entropy.values()):.3f} nats
- **Most concentrated layer**: L{min_ent_layer} (entropy = {min_ent:.3f}, {100*(1 - min_ent/uniform_ent):.1f}% below uniform)

Routing is consistently sub-uniform across all layers, indicating the load-balancing
loss does not fully suppress specialisation. Concentration increases in late layers
(L17–21), consistent with the causal bottleneck identified in Thread 2.

### ASCII routing entropy by layer

```
"""
    for li in range(N_LAYERS):
        ent = routing_entropy[str(li)]
        bar_len = int(20 * ent / uniform_ent)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        report += f"  L{li:02d}  [{bar}]  {ent:.3f} nats\n"

    report += f"""\
```

---

## Measurement 2: Layer logit-delta

The logit-delta peaks at **L{max_delta_layer}**, consistent with the L19–21 causal bottleneck.

### Top promoted tokens at key layers

| Layer | Top-promoted tokens | Interpretation |
|-------|---------------------|----------------|
| L01 | {_fmt_top_tokens(layer_logit_delta, "1")} | Surface adjustment |
| L08 | {_fmt_top_tokens(layer_logit_delta, "8")} | Syntactic structure |
| L17 | {_fmt_top_tokens(layer_logit_delta, "17")} | Semantic refinement |
| L20 | {_fmt_top_tokens(layer_logit_delta, "20")} | Task answer promotion |
| L23 | {_fmt_top_tokens(layer_logit_delta, "23")} | Final confidence |

Early layers (L0–8) show surface-token adjustments (punctuation, function words).
Late layers (L17–21) show content-token and answer-token promotion, consistent with
the causal bottleneck being the locus of task resolution.

---

## Measurement 3: Expert vocabulary profiles

Expert vocabulary profiles reveal depth-stratified specialisation:

- **Early experts (L0–8)**: strong preference for surface tokens (punctuation,
  whitespace, function words). Expert co-activation signatures are diverse —
  different experts handle different surface forms.

- **Mid-layer experts (L9–16)**: profiles shift toward syntactic tokens (relative
  clauses, prepositions). This is the layer range where Thread 1 shows coreference
  beginning to resolve.

- **Late experts (L17–23)**: profiles concentrate on content words and answer tokens.
  At L20 (the causal bottleneck), the highest-utilisation experts show the strongest
  alignment with the final prediction token.

---

## Summary

| Measurement | Finding |
|-------------|---------|
| Routing entropy | Sub-uniform at all layers; concentrates in L17–21 |
| Layer logit-delta | Peaks at L20 with task-relevant content tokens |
| Expert vocab profiles | Depth-stratified: surface → syntax → semantic |

These patterns are consistent with the Thread 2 late-layer causal bottleneck and
suggest that expert specialisation is interpretably organised along the depth axis.
"""

    with open(out_dir / "expert_readout_report.md", "w") as f:
        f.write(report)
    print(f"[make_demo_data] report → {out_dir}/expert_readout_report.md")


def _fmt_top_tokens(layer_logit_delta: dict, layer_str: str) -> str:
    entry = layer_logit_delta.get(layer_str, {})
    promoted = entry.get("0", {}).get("promoted", [])
    if not promoted:
        return "—"
    return ", ".join(f'`{t}`' for t, _ in promoted[:3])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate Thread 15 demo outputs (no model needed).")
    parser.add_argument("--output", default="runs/expert_readouts_demo", help="Output directory")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[make_demo_data] Generating synthetic routing patterns...")
    routing_patterns = make_routing_patterns(rng)
    with open(out_dir / "routing_patterns.json", "w") as f:
        json.dump(routing_patterns, f)

    print("[make_demo_data] Computing routing entropy...")
    routing_entropy = make_routing_entropy(routing_patterns)
    with open(out_dir / "routing_entropy.json", "w") as f:
        json.dump(routing_entropy, f)

    print("[make_demo_data] Generating logit-delta data...")
    layer_logit_delta = make_layer_logit_delta(rng)
    with open(out_dir / "layer_logit_delta.json", "w") as f:
        json.dump(layer_logit_delta, f)

    print("[make_demo_data] Generating expert vocab profiles...")
    expert_vocab_profiles = make_expert_vocab_profiles(rng)
    with open(out_dir / "expert_vocab_profiles.json", "w") as f:
        json.dump(expert_vocab_profiles, f)

    print("[make_demo_data] Computing information-theoretic quantities...")
    kl_from_uniform = {
        str(li): round(math.log(N_EXPERTS) - routing_entropy[str(li)], 4)
        for li in range(N_LAYERS)
    }
    with open(out_dir / "routing_kl_from_uniform.json", "w") as f:
        json.dump(kl_from_uniform, f)

    # Synthetic routing velocity — peaks near L16
    routing_velocity = {}
    for li in range(1, N_LAYERS):
        vel = 0.001 + 0.018 * math.exp(-((li - 16) ** 2) / 4)
        routing_velocity[str(li)] = round(vel + rng.uniform(-0.001, 0.001), 5)
    with open(out_dir / "routing_velocity.json", "w") as f:
        json.dump(routing_velocity, f)

    # Synthetic MI(expert; task) — rises in late layers, peaks at L20
    mi_task = {}
    for li in range(N_LAYERS):
        mi = 0.0002 + 0.008 * math.exp(-((li - 20) ** 2) / 6)
        mi_task[str(li)] = round(max(0.0, mi + rng.uniform(-0.0001, 0.0001)), 6)
    with open(out_dir / "routing_mi_task.json", "w") as f:
        json.dump(mi_task, f)

    # Synthetic JSD matrix at a few layers (5 task families)
    task_names = ["capitalization", "coreference", "induction", "recency", "syntax"]
    jsd_matrix = {}
    for li in range(N_LAYERS):
        progress = li / (N_LAYERS - 1)
        layer_mat = {}
        for ta in task_names:
            layer_mat[ta] = {}
            for tb in task_names:
                if ta == tb:
                    layer_mat[ta][tb] = 0.0
                else:
                    # JSD grows with depth; some pairs differentiate earlier
                    base = 0.005 + 0.08 * progress
                    layer_mat[ta][tb] = round(min(0.693, base + rng.uniform(0, 0.02)), 4)
        jsd_matrix[str(li)] = layer_mat
    with open(out_dir / "routing_jsd_matrix.json", "w") as f:
        json.dump(jsd_matrix, f)

    print("[make_demo_data] Writing report...")
    write_report(out_dir, routing_entropy, layer_logit_delta)

    print(f"\n[make_demo_data] Done. Now run:")
    print(f"  python threads/in-progress/15-expert-readouts/generate_expert_figures.py \\")
    print(f"    --run-dir {out_dir}")


if __name__ == "__main__":
    main()
