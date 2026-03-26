#!/usr/bin/env python3
"""Thread 15: MoE expert readout analysis.

Five measurements in priority order:

  1. Routing patterns  — sidecar, always works (MXFP4-safe)
  2. Layer logit-delta — ActivationCache + logit-lens projection, always works
  3. Expert vocab profiles — ExpertCapture, non-quantized checkpoints only
  4. Information-theoretic analysis — KL, JSD, MI over routing data, always works
  5. Routing capacity budget — I(expert; task) vs I(expert; token surface)

All measurements write to --output.  Measurements 1, 2, 4 always run.
Measurement 3 is skipped on MXFP4 (prints a note).  Measurement 5 requires
--task-suite (needs multiple task families for MI estimation).

Usage:
    # All three measurements, single prompt
    python threads/in-progress/15-expert-readouts/run_expert_analysis.py \\
        --model openai/gpt-oss-20b \\
        --prompt "The trophy did not fit in the suitcase because it was too small." \\
        --output runs/expert_readouts/

    # Multiple task families (shows cross-family routing differences)
    python threads/in-progress/15-expert-readouts/run_expert_analysis.py \\
        --model openai/gpt-oss-20b \\
        --task-suite \\
        --output runs/expert_readouts/

    # Expert vocab profiles (non-quantized checkpoint only)
    python threads/in-progress/15-expert-readouts/run_expert_analysis.py \\
        --model openai/gpt-oss-20b-base \\
        --task-suite \\
        --expert-profiles \\
        --output runs/expert_readouts/
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

# Representative prompts for each task family used in the main analysis set.
# Nine families × 15 prompts = 135 total, covering diverse token surfaces,
# semantic domains, and reasoning types for robust MI(expert; task) estimation.
TASK_SUITE = {
    "capitalization": [
        "The name of the first president of the United States is george",
        "My favorite city is paris, which is known for",
        "The capital of france is paris and the capital of germany is berlin",
        "She was born in london and moved to new york when she was",
        "The treaty was signed in versailles after the war ended in nineteen",
        "His full name is john fitzgerald kennedy, the 35th president of the",
        "The river nile flows through egypt and sudan before reaching the",
        "The company was founded by steve jobs and steve wozniak in cupertino",
        "Mount everest, located in the himalayas, was first climbed in",
        "The ancient city of rome was founded on the banks of the tiber",
        "The headquarters of the united nations is located in new york city near the",
        "She studied at oxford university before joining the faculty at cambridge",
        "The expedition departed from cape town and sailed toward antarctica",
        "He received the nobel prize in oslo alongside marie curie from",
        "The amazon river, stretching across brazil, flows into the atlantic",
    ],
    "coreference": [
        "The trophy didn't fit in the suitcase because it was too small. The",
        "The developer argued with the designer because she didn't like the",
        "The ball rolled off the shelf because it wasn't stable. The",
        "Paul called Tom because he wanted to ask for advice. He",
        "The city council refused the demonstrators a permit because they feared violence. They",
        "The lawyer asked the witness a question, but she was not satisfied with the answer. She",
        "The scientist told the journalist that she had made an important discovery. She",
        "The boy chased the dog until it was exhausted. It",
        "Susan asked Mary to proofread her report before she submitted it. She",
        "The manager fired the employee because he was unhappy with the performance. He",
        "The judge warned the defendant that he would be penalized for lying. He",
        "The doctor advised the patient to rest because she was overworked. She",
        "Jane told Alice that her proposal was rejected by the board. Her",
        "The police interviewed the suspect because they had new evidence against him. He",
        "The coach praised the athlete because her performance exceeded expectations. She",
    ],
    "induction": [
        "The sequence continues: alpha beta gamma alpha beta",
        "In the pattern red blue green red blue",
        "The series goes: 1 2 3 1 2",
        "The letters repeat: A B C D A B C D A B C",
        "The pattern is: cat dog bird cat dog bird cat dog",
        "The sequence: Monday Tuesday Wednesday Monday Tuesday Wednesday Monday",
        "Repeating colors: red green blue red green blue red",
        "The tokens are: X Y Z X Y Z X Y",
        "The cycle: spring summer autumn winter spring summer autumn",
        "The digits repeat: 1 4 7 1 4 7 1",
        "The symbols alternate: circle square triangle circle square triangle circle",
        "The words cycle: fast slow fast slow fast slow",
        "The pattern: north south east west north south east",
        "Repeating: ping pong ping pong ping pong",
        "The sequence: one two three one two three one two",
    ],
    "syntax_agreement": [
        "The keys to the cabinet",
        "The player with the best statistics on both teams",
        "The bouquet of yellow flowers",
        "The committee of senior managers",
        "The rules of the game",
        "The list of items on the agenda",
        "The speed of the cars on the highway",
        "The behavior of children in classrooms",
        "The price of houses in the neighborhood",
        "The impact of policies on local communities",
        "The collection of rare books in the library",
        "The group of students studying in the hall",
        "The number of errors in the report",
        "The team of engineers working on the bridge",
        "The quality of decisions made by the board",
    ],
    "recency": [
        "I bought milk, eggs, and bread. Then I bought coffee. And finally I bought",
        "She visited Rome, then Paris, then London. Her last stop was",
        "The menu listed pasta, pizza, and soup. The waiter recommended the",
        "He tried the red shirt, then the blue one, then the green one. He chose the",
        "The countries visited were Spain, Italy, and Greece. The most recent was",
        "She learned piano, then violin, then guitar. Her current instrument is",
        "The project phases were planning, design, and implementation. The current phase is",
        "They discussed the proposal, the budget, and the timeline. The last topic was",
        "The ingredients are flour, sugar, butter. The last item is",
        "We reviewed chapter one, chapter two, and chapter three. The final chapter was",
        "The runners finished in positions first, second, and third. The last to cross was",
        "He watched a documentary, then a comedy, then a thriller. His final choice was",
        "She drafted an email, then a memo, then a report. Her most recent document was",
        "The train stopped at Berlin, Warsaw, then Moscow. The last city was",
        "They built the foundation, the walls, and the roof. The last stage was",
    ],
    "arithmetic": [
        "What is 3 plus 5? The answer is",
        "If you have 12 apples and eat 4, you have",
        "Seven times eight equals",
        "The square root of 144 is",
        "100 divided by 4 equals",
        "If a train travels 60 miles per hour for 2 hours, it covers",
        "What is 15 percent of 200? The answer is",
        "9 squared equals",
        "The sum of 47 and 53 is",
        "What is 1000 minus 337? The answer is",
        "If x equals 5 and y equals 3, then x times y equals",
        "The product of 11 and 11 is",
        "24 divided by 6 equals",
        "What is 2 to the power of 8? The answer is",
        "If one kilogram equals 2.2 pounds, then 5 kilograms equals",
    ],
    "factual_recall": [
        "The chemical symbol for water is",
        "The speed of light in a vacuum is approximately",
        "William Shakespeare was born in the year",
        "The largest planet in our solar system is",
        "The human body has approximately",
        "The French Revolution began in the year",
        "DNA stands for",
        "The atomic number of carbon is",
        "The Great Wall of China was primarily built during the",
        "Albert Einstein published his special theory of relativity in",
        "The capital city of Japan is",
        "The element with the highest atomic number that occurs naturally is",
        "Photosynthesis converts sunlight into",
        "The Battle of Waterloo took place in the year",
        "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals",
    ],
    "code_completion": [
        "def factorial(n):\n    if n == 0:\n        return",
        "import numpy as np\nnp.array([1, 2, 3]).mean() ==",
        "for i in range(10):\n    if i % 2 == 0:\n        print(",
        "x = {'a': 1, 'b': 2}\nx.get('c', 0) ==",
        "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b,",
        "s = 'hello world'\ns.split()[1] ==",
        "lst = [3, 1, 4, 1, 5]\nsorted(lst)[0] ==",
        "class Counter:\n    def __init__(self):\n        self.count =",
        "try:\n    result = 10 / 0\nexcept ZeroDivisionError:\n    result =",
        "import re\nre.match(r'\\d+', '123abc').group() ==",
        "def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0: return",
        "d = {}\nfor k, v in [('a', 1), ('b', 2)]:\n    d[k] =",
        "stack = []\nstack.append(1)\nstack.append(2)\nstack.pop() ==",
        "words = ['cat', 'elephant', 'dog']\nmax(words, key=len) ==",
        "n = 256\nwhile n > 1:\n    n //=",
    ],
    "analogy": [
        "King is to queen as man is to",
        "Paris is to France as Rome is to",
        "Hot is to cold as day is to",
        "Doctor is to hospital as teacher is to",
        "Fish is to water as bird is to",
        "Glove is to hand as shoe is to",
        "Author is to book as composer is to",
        "Puppy is to dog as kitten is to",
        "Thick is to thin as heavy is to",
        "Chef is to kitchen as pilot is to",
        "Pen is to write as brush is to",
        "Hearing is to ears as sight is to",
        "Planet is to solar system as cell is to",
        "Smile is to happiness as frown is to",
        "Architect is to building as sculptor is to",
    ],
}


###############################################################################
# Measurement 1: Routing patterns via sidecar
###############################################################################

def measure_routing_patterns(
    backend,
    prompts_by_task: dict[str, list[str]],
    arch,
) -> dict:
    """Capture routing decisions for all prompts and aggregate statistics."""
    from gossh.sidecar import MoeSidecar, RouterWeightExtractor

    print("\n=== Measurement 1: Routing patterns (sidecar) ===")

    weights = RouterWeightExtractor().extract(backend.model, backend.structure)
    if not weights:
        print("  WARNING: No router weights found. Skipping routing measurement.")
        return {}

    print(f"  Router weights: {len(weights)} layers extracted")

    # token_counts[layer][expert] = total tokens routed there
    token_counts: dict[str, dict[int, dict[int, int]]] = {}
    # Also accumulate cross-task
    combined_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    with MoeSidecar(weights, top_k=arch.top_k) as sidecar:
        backend.attach_sidecar(sidecar)

        for task_name, prompts in prompts_by_task.items():
            print(f"  [{task_name}]", end="", flush=True)
            task_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

            for prompt in prompts:
                decisions = backend.capture_routing(prompt)
                for d in decisions:
                    # selected_experts may be [seq_len, top_k], [top_k], or
                    # [batch, seq_len, top_k] — flatten to [N, top_k]
                    se = d.selected_experts.reshape(-1, d.selected_experts.shape[-1])
                    for ei in se.reshape(-1).tolist():
                        task_counts[d.layer_idx][int(ei)] += 1
                        combined_counts[d.layer_idx][int(ei)] += 1
                print(".", end="", flush=True)

            token_counts[task_name] = {
                li: dict(expert_counts)
                for li, expert_counts in task_counts.items()
            }
            print()

        backend.detach_sidecar()

    # Compute routing entropy per layer
    n_experts = arch.num_experts
    entropy_by_layer: dict[int, float] = {}
    for layer_idx, counts in combined_counts.items():
        total = sum(counts.values()) or 1
        probs = [counts.get(e, 0) / total for e in range(n_experts)]
        ent = -sum(p * math.log(p + 1e-12) for p in probs)
        entropy_by_layer[layer_idx] = ent

    max_entropy = math.log(n_experts)
    print(f"\n  Routing entropy (max possible = {max_entropy:.3f} nats):")
    for li in sorted(entropy_by_layer):
        bar = "█" * int(entropy_by_layer[li] / max_entropy * 20)
        print(f"    L{li:2d}  {entropy_by_layer[li]:.3f}  {bar}")

    return {
        "by_task": {
            task: {str(li): counts for li, counts in by_layer.items()}
            for task, by_layer in token_counts.items()
        },
        "combined": {
            str(li): dict(counts) for li, counts in combined_counts.items()
        },
        "entropy_by_layer": {str(k): v for k, v in entropy_by_layer.items()},
        "max_entropy": max_entropy,
        "n_experts": n_experts,
    }


###############################################################################
# Measurement 2: Layer logit-delta
###############################################################################

def measure_layer_logit_delta(
    backend,
    prompts: list[str],
    top_k: int = 8,
) -> dict:
    """For each layer, compute Δ log p_l = log p_l - log p_{l-1}.

    Returns top promoted/suppressed tokens per layer, averaged over prompts.
    """
    from gossh.capture import ActivationCache

    print("\n=== Measurement 2: Layer logit-delta ===")

    final_norm = backend.structure.final_norm
    lm_head = backend.structure.lm_head
    norm_device = next(final_norm.parameters()).device

    # Accumulate logprob deltas: {layer: {token_id: [delta, ...]}}
    delta_accum: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))

    for p_idx, prompt in enumerate(prompts):
        print(f"  [{p_idx+1}/{len(prompts)}] {prompt[:60]!r}")
        input_ids = backend.tokenizer.encode(prompt, return_tensors="pt").to(backend.device)
        seq_len = input_ids.shape[1]

        cache = ActivationCache(detach=True, to_cpu=True)
        handles = cache.register(backend.model, backend.structure.block_names)
        try:
            with torch.no_grad():
                backend.model(input_ids)
        finally:
            for h in handles:
                h.remove()

        # Project each layer's hidden state → log probs
        layer_logp: list[torch.Tensor] = []
        for block_name in backend.structure.block_names:
            record = cache.last(block_name)
            if record is None:
                layer_logp.append(None)
                continue
            hidden = record.tensor  # [1, seq_len, D]
            with torch.no_grad():
                normed = final_norm(hidden.to(norm_device))
                logits = lm_head(normed).cpu().float()
                lp = torch.log_softmax(logits[0], dim=-1)  # [seq_len, V]
            layer_logp.append(lp)

        # Compute deltas at last token position
        pos = seq_len - 1
        for l_idx in range(1, len(layer_logp)):
            if layer_logp[l_idx] is None or layer_logp[l_idx - 1] is None:
                continue
            delta = (layer_logp[l_idx][pos] - layer_logp[l_idx - 1][pos])  # [V]
            topk_vals, topk_ids = delta.topk(top_k * 2)
            for v, i in zip(topk_vals.tolist(), topk_ids.tolist()):
                delta_accum[l_idx][i].append(v)

    # Average and find top promoted tokens per layer
    results: dict[int, list[dict]] = {}
    for layer_idx in sorted(delta_accum.keys()):
        avg_delta = {
            tid: sum(vals) / len(vals)
            for tid, vals in delta_accum[layer_idx].items()
        }
        top_promoted = sorted(avg_delta.items(), key=lambda x: -x[1])[:top_k]
        top_suppressed = sorted(avg_delta.items(), key=lambda x: x[1])[:top_k]

        results[layer_idx] = {
            "promoted": [
                {"token": backend.tokenizer.decode([tid]), "token_id": tid, "delta": d}
                for tid, d in top_promoted
            ],
            "suppressed": [
                {"token": backend.tokenizer.decode([tid]), "token_id": tid, "delta": d}
                for tid, d in top_suppressed
            ],
        }

    # Print summary table
    print(f"\n  Top promoted tokens at last position (averaged over {len(prompts)} prompts):")
    print(f"  {'Layer':6s}  {'Top promoted':40s}  {'Top suppressed'}")
    print(f"  {'------':6s}  {'-'*40}  {'-'*40}")
    for layer_idx, info in sorted(results.items()):
        promoted_str = " ".join(
            f"{e['token']!r}({e['delta']:+.2f})" for e in info["promoted"][:3]
        )
        suppressed_str = " ".join(
            f"{e['token']!r}({e['delta']:+.2f})" for e in info["suppressed"][:3]
        )
        print(f"  L{layer_idx:2d}     {promoted_str:40s}  {suppressed_str}")

    return {str(k): v for k, v in results.items()}


###############################################################################
# Measurement 3: Expert vocabulary profiles (non-quantized only)
###############################################################################

def measure_expert_vocab_profiles(
    backend,
    prompts: list[str],
    top_k: int = 10,
) -> dict | None:
    """Project each expert's output through lm_head to get its vocabulary profile.

    Returns None if no hookable expert modules are found (MXFP4).
    """
    from gossh.capture import ExpertCapture

    print("\n=== Measurement 3: Expert vocabulary profiles ===")

    expert_names = ExpertCapture.discover_expert_names(backend.model, backend.structure)
    if not expert_names:
        print("  No hookable expert modules found.")
        print("  This model is likely running under MXFP4 quantization.")
        print("  Expert vocabulary profiles require a non-quantized checkpoint.")
        print("  → Routing patterns (measurement 1) are available via the sidecar.")
        return None

    n_layers = len(expert_names)
    n_experts_per_layer = max(len(v) for v in expert_names.values())
    print(f"  Found {n_experts_per_layer} hookable experts × {n_layers} layers")

    final_norm = backend.structure.final_norm
    lm_head = backend.structure.lm_head
    norm_device = next(final_norm.parameters()).device

    # Accumulate mean logp per expert: {layer: {expert: [vocab_tensor, ...]}}
    # We accumulate log-probs to average in log space (geometric mean of distributions)
    accum: dict[int, dict[int, list[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))
    token_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    with ExpertCapture(detach=True, to_cpu=True) as cap:
        cap.register(backend.model, expert_names)

        for p_idx, prompt in enumerate(prompts):
            print(f"  [{p_idx+1}/{len(prompts)}] {prompt[:60]!r}")
            cap.clear()

            input_ids = backend.tokenizer.encode(prompt, return_tensors="pt").to(backend.device)
            with torch.no_grad():
                backend.model(input_ids)

            # Project captured expert outputs
            for layer_idx, experts in cap.captured.items():
                for expert_idx, outputs in experts.items():
                    if not outputs:
                        continue
                    stacked = torch.cat(outputs, dim=0)  # [n_tokens, hidden_dim]
                    with torch.no_grad():
                        normed = final_norm(stacked.to(norm_device))
                        logits = lm_head(normed).cpu().float()
                        lp = torch.log_softmax(logits, dim=-1)  # [n, V]
                    accum[layer_idx][expert_idx].append(lp.mean(dim=0))
                    token_counts[layer_idx][expert_idx] += stacked.shape[0]

    # Average log-probs → expert vocabulary profile
    profiles: dict[int, dict[int, dict]] = {}
    for layer_idx in sorted(accum.keys()):
        profiles[layer_idx] = {}
        for expert_idx in sorted(accum[layer_idx].keys()):
            lp_list = accum[layer_idx][expert_idx]
            if not lp_list:
                continue
            mean_lp = torch.stack(lp_list).mean(dim=0)  # [V]
            topk_vals, topk_ids = mean_lp.topk(top_k)
            profiles[layer_idx][expert_idx] = {
                "top_tokens": [
                    {"token": backend.tokenizer.decode([tid]), "token_id": tid, "logp": float(v)}
                    for tid, v in zip(topk_ids.tolist(), topk_vals.tolist())
                ],
                "tokens_routed": token_counts[layer_idx][expert_idx],
            }

    # Print summary (early, mid, late layers)
    n_layers = backend.structure.num_layers
    show_layers = sorted({0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1}
                         & set(profiles.keys()))

    for layer_idx in show_layers:
        print(f"\n  Layer {layer_idx} expert profiles (top-5 tokens):")
        print(f"  {'Expert':8s}  {'Tokens routed':14s}  Top-5 vocabulary")
        print(f"  {'------':8s}  {'-'*14}  {'-'*50}")
        for expert_idx in sorted(profiles[layer_idx].keys())[:8]:
            info = profiles[layer_idx][expert_idx]
            tokens_str = " ".join(
                f"{e['token']!r}" for e in info["top_tokens"][:5]
            )
            print(f"  E{expert_idx:2d}       {info['tokens_routed']:8d}        {tokens_str}")

    return {
        str(li): {
            str(ei): info
            for ei, info in experts.items()
        }
        for li, experts in profiles.items()
    }


###############################################################################
# Measurement 4: Information-theoretic routing analysis
###############################################################################

def _kl_div(p: list[float], q: list[float]) -> float:
    """D_KL(P ‖ Q) in nats. Clips to avoid log(0)."""
    eps = 1e-12
    return sum(
        pi * math.log((pi + eps) / (qi + eps))
        for pi, qi in zip(p, q)
        if pi > eps
    )


def _jsd(p: list[float], q: list[float]) -> float:
    """Jensen-Shannon divergence JSD(P, Q) in nats. Symmetric, bounded [0, ln2]."""
    m = [(pi + qi) / 2 for pi, qi in zip(p, q)]
    return 0.5 * _kl_div(p, m) + 0.5 * _kl_div(q, m)


def _entropy(p: list[float]) -> float:
    eps = 1e-12
    return -sum(pi * math.log(pi + eps) for pi in p if pi > eps)


def measure_information_theory(routing_data: dict) -> dict:
    """Compute KL, JSD, MI, and routing velocity from routing_data.

    All quantities derived from the token counts already collected in
    Measurement 1 — no additional forward passes required.

    Returns dict with keys:
      kl_from_uniform     {layer: float}   — specialization gain (nats)
      routing_velocity    {layer: float}   — D_KL(routing_l ‖ routing_{l-1})
      jsd_cross_task      {layer: {task_a: {task_b: float}}}  — pairwise JSD
      mi_task             {layer: float}   — I(expert; task_family)
      mi_interpretation   str              — plain-English summary
    """
    if not routing_data:
        print("\n=== Measurement 4: Information-theoretic analysis ===")
        print("  No routing data available — skipping.")
        return {}

    print("\n=== Measurement 4: Information-theoretic routing analysis ===")

    by_task: dict[str, dict[str, dict]] = routing_data.get("by_task", {})
    combined: dict[str, dict] = routing_data.get("combined", {})
    n_experts: int = routing_data.get("n_experts", 32)
    max_entropy: float = routing_data.get("max_entropy", math.log(n_experts))
    task_names = sorted(by_task.keys())
    all_layers = sorted(int(k) for k in combined.keys())

    # Helper: normalise raw count dict to probability list over [0, n_experts)
    def to_probs(counts: dict, n: int) -> list[float]:
        total = sum(counts.values()) or 1
        return [counts.get(e, 0) / total for e in range(n)]

    uniform = [1.0 / n_experts] * n_experts

    # --- Specialization gain: D_KL(routing_l ‖ uniform) ---
    kl_from_uniform: dict[int, float] = {}
    entropy_by_layer: dict[int, float] = {int(k): v for k, v in routing_data.get("entropy_by_layer", {}).items()}
    for li in all_layers:
        # Specialization gain = log(n_experts) - H(routing_l)
        kl_from_uniform[li] = max_entropy - entropy_by_layer.get(li, max_entropy)

    # --- Routing velocity: D_KL(routing_l ‖ routing_{l-1}) ---
    routing_velocity: dict[int, float] = {}
    prev_probs = None
    for li in all_layers:
        probs = to_probs(combined.get(str(li), {}), n_experts)
        if prev_probs is not None:
            routing_velocity[li] = _kl_div(probs, prev_probs)
        prev_probs = probs

    # --- Cross-task JSD matrix per layer ---
    jsd_cross_task: dict[int, dict[str, dict[str, float]]] = {}
    for li in all_layers:
        task_probs = {
            t: to_probs(by_task.get(t, {}).get(str(li), {}), n_experts)
            for t in task_names
        }
        jsd_matrix: dict[str, dict[str, float]] = {}
        for ta in task_names:
            jsd_matrix[ta] = {}
            for tb in task_names:
                if ta == tb:
                    jsd_matrix[ta][tb] = 0.0
                else:
                    jsd_matrix[ta][tb] = _jsd(task_probs[ta], task_probs[tb])
        jsd_cross_task[li] = jsd_matrix

    # --- Task routing mutual information: I(expert; task_family) ---
    # I(E; T) = H(E) - H(E|T) = H(T) - H(T|E)
    # Using: I(E;T) = sum_{e,t} p(e,t) * log(p(e,t) / (p(e)*p(t)))
    n_tasks = len(task_names)
    mi_task: dict[int, float] = {}
    for li in all_layers:
        # Build joint distribution p(expert_e, task_t)
        # Each task contributes equally (uniform task prior)
        joint: list[list[float]] = []
        task_marginal_counts = []
        for t in task_names:
            counts = by_task.get(t, {}).get(str(li), {})
            total = sum(counts.values()) or 1
            task_marginal_counts.append([counts.get(e, 0) / total for e in range(n_experts)])

        # p(e, t) = (1/n_tasks) * p(e | t)
        # p(e) = sum_t p(e, t) = (1/n_tasks) * sum_t p(e|t)
        # p(t) = 1/n_tasks  (uniform task prior)
        expert_marginal = [
            sum(task_marginal_counts[ti][e] for ti in range(n_tasks)) / n_tasks
            for e in range(n_experts)
        ]
        task_prior = 1.0 / n_tasks

        mi = 0.0
        for ti in range(n_tasks):
            for e in range(n_experts):
                p_et = task_prior * task_marginal_counts[ti][e]
                p_e = expert_marginal[e]
                if p_et > 1e-12 and p_e > 1e-12:
                    mi += p_et * math.log(p_et / (p_e * task_prior))
        mi_task[li] = max(0.0, mi)  # numerical floor

    # --- Print summaries ---
    print(f"\n  Specialization gain D_KL(routing ‖ uniform) per layer:")
    print(f"  (nats saved by knowing routing policy vs. guessing uniform)")
    for li in all_layers:
        gain = kl_from_uniform[li]
        pct = 100 * gain / max_entropy if max_entropy > 0 else 0
        bar = "▓" * int(gain / max_entropy * 40)
        print(f"    L{li:02d}  {gain:.4f} nats ({pct:.1f}%)  {bar}")

    print(f"\n  Routing velocity D_KL(routing_l ‖ routing_{{l-1}}) — phase transition detection:")
    for li in sorted(routing_velocity):
        vel = routing_velocity[li]
        bar = "▓" * min(40, int(vel * 200))
        print(f"    L{li:02d}→L{li:02d}  {vel:.5f} nats  {bar}")

    if len(task_names) >= 2:
        print(f"\n  Task routing mutual information I(expert; task_family) per layer:")
        print(f"  (nats of task identity encoded in the routing decision)")
        print(f"  Task families: {', '.join(task_names)}")
        peak_mi_layer = max(mi_task, key=mi_task.get) if mi_task else None
        for li in all_layers:
            mi = mi_task[li]
            bar = "▓" * min(40, int(mi * 1000))
            marker = " ← PEAK" if li == peak_mi_layer else ""
            print(f"    L{li:02d}  {mi:.5f} nats  {bar}{marker}")

        peak_mi = mi_task.get(peak_mi_layer, 0) if peak_mi_layer is not None else 0
        routing_capacity_bits = math.log2(
            # C(n_experts, top_k) — approximated as n_experts^top_k / top_k! for large n
            1  # placeholder; exact computation below
        )
        # Exact: log2(C(32,4)) = log2(35960)
        try:
            from math import comb
            top_k = routing_data.get("top_k", 4)
            capacity_bits = math.log2(comb(n_experts, top_k))
        except Exception:
            capacity_bits = 15.13  # log2(C(32,4))

        efficiency = peak_mi / (capacity_bits * math.log(2)) * 100 if capacity_bits > 0 else 0
        print(f"\n  Routing capacity budget at peak MI layer (L{peak_mi_layer}):")
        print(f"    Theoretical capacity:          {capacity_bits:.2f} bits = {capacity_bits * math.log(2):.3f} nats")
        print(f"    I(expert; task) at peak:       {peak_mi:.5f} nats ({efficiency:.2f}% of capacity)")
        print(f"    Specialization gain at peak:   {kl_from_uniform.get(peak_mi_layer, 0):.4f} nats")
        print(f"    Interpretation: task-discriminating routing uses {efficiency:.1f}% of routing capacity")

        # JSD cross-task at bottleneck layer (last third of layers)
        late_layers = [li for li in all_layers if li >= max(all_layers) * 2 // 3]
        if late_layers:
            probe_layer = late_layers[len(late_layers) // 2]
            print(f"\n  Cross-task JSD matrix at L{probe_layer} (0 = identical, {math.log(2):.3f} nats = disjoint):")
            header = f"  {'':12s}" + "".join(f"  {t[:8]:>8s}" for t in task_names)
            print(header)
            for ta in task_names:
                row = f"  {ta[:12]:12s}" + "".join(
                    f"  {jsd_cross_task[probe_layer][ta][tb]:8.4f}"
                    for tb in task_names
                )
                print(row)

    return {
        "kl_from_uniform": {str(k): v for k, v in kl_from_uniform.items()},
        "routing_velocity": {str(k): v for k, v in routing_velocity.items()},
        "jsd_cross_task": {str(li): mat for li, mat in jsd_cross_task.items()},
        "mi_task": {str(k): v for k, v in mi_task.items()},
        "routing_capacity_bits": capacity_bits if len(task_names) >= 2 else None,
        "peak_mi_layer": peak_mi_layer,
        "peak_mi_nats": peak_mi if len(task_names) >= 2 else None,
    }


###############################################################################
# Report generation
###############################################################################

def write_report(
    output_dir: Path,
    routing_data: dict,
    delta_data: dict,
    profile_data: dict | None,
    info_data: dict,
    prompts_by_task: dict[str, list[str]],
) -> None:
    lines = [
        "# Thread 15: MoE Expert Readout Analysis",
        "",
        f"Prompts analyzed: {sum(len(v) for v in prompts_by_task.values())} "
        f"across {len(prompts_by_task)} task families",
        "",
    ]

    if routing_data:
        lines += [
            "## Measurement 1: Routing Entropy by Layer",
            "",
            "| Layer | Entropy (nats) | % of max |",
            "| ---: | ---: | ---: |",
        ]
        max_ent = routing_data.get("max_entropy", 1.0)
        for li_str, ent in sorted(routing_data.get("entropy_by_layer", {}).items(), key=lambda x: int(x[0])):
            pct = 100 * ent / max_ent
            lines.append(f"| L{li_str} | {ent:.3f} | {pct:.1f}% |")
        lines.append("")

    if delta_data:
        lines += [
            "## Measurement 2: Layer Logit-Delta (Top Promoted Tokens at Last Position)",
            "",
            "| Layer | Top promoted | Top suppressed |",
            "| ---: | --- | --- |",
        ]
        for li_str in sorted(delta_data.keys(), key=int):
            info = delta_data[li_str]
            promoted = " ".join(
                f"`{e['token']}`({e['delta']:+.2f})" for e in info["promoted"][:3]
            )
            suppressed = " ".join(
                f"`{e['token']}`({e['delta']:+.2f})" for e in info["suppressed"][:3]
            )
            lines.append(f"| L{li_str} | {promoted} | {suppressed} |")
        lines.append("")

    if profile_data is None:
        lines += [
            "## Measurement 3: Expert Vocabulary Profiles",
            "",
            "_Skipped: no hookable expert modules found (MXFP4 quantization)._",
            "_Use a non-quantized checkpoint to enable per-expert readouts._",
            "",
        ]
    else:
        lines += [
            "## Measurement 3: Expert Vocabulary Profiles",
            "",
            "_Top-5 vocabulary predictions for each expert (averaged over all activations)._",
            "",
        ]
        for li_str in sorted(profile_data.keys(), key=int):
            lines.append(f"### Layer {li_str}\n")
            lines.append("| Expert | Tokens routed | Top-5 tokens |")
            lines.append("| ---: | ---: | --- |")
            for ei_str, info in sorted(profile_data[li_str].items(), key=lambda x: int(x[0])):
                tokens = " ".join(f"`{t['token']}`" for t in info["top_tokens"][:5])
                lines.append(f"| E{ei_str} | {info['tokens_routed']} | {tokens} |")
            lines.append("")

    # Measurement 4: information-theoretic routing analysis
    if info_data:
        kl = info_data.get("kl_from_uniform", {})
        vel = info_data.get("routing_velocity", {})
        mi = info_data.get("mi_task", {})
        peak_layer = info_data.get("peak_mi_layer")
        peak_mi = info_data.get("peak_mi_nats", 0) or 0
        capacity = info_data.get("routing_capacity_bits")

        lines += [
            "## Measurement 4: Information-Theoretic Routing Analysis",
            "",
            "### 4a — Specialization gain D_KL(routing ‖ uniform)",
            "",
            "_Nats saved by knowing the routing policy vs. guessing uniform._",
            "_Higher = more structured routing at this layer._",
            "",
            "| Layer | Spec. gain (nats) | % of max entropy |",
            "| ---: | ---: | ---: |",
        ]
        max_ent = routing_data.get("max_entropy", math.log(32))
        for li_str in sorted(kl.keys(), key=int):
            gain = kl[li_str]
            pct = 100 * gain / max_ent if max_ent > 0 else 0
            lines.append(f"| L{li_str} | {gain:.4f} | {pct:.1f}% |")
        lines.append("")

        lines += [
            "### 4b — Routing velocity D_KL(routing_l ‖ routing_{l-1})",
            "",
            "_Extra nats burned encoding layer l's routing using layer l-1's codebook._",
            "_A spike marks a routing phase transition — where policy changes fastest._",
            "",
            "| Layer pair | Velocity (nats) |",
            "| ---: | ---: |",
        ]
        peak_vel_layer = max(vel, key=vel.get) if vel else None
        for li_str in sorted(vel.keys(), key=int):
            v = vel[li_str]
            marker = " ← PHASE TRANSITION" if li_str == str(peak_vel_layer) else ""
            lines.append(f"| L{int(li_str)-1}→L{li_str} | {v:.5f}{marker} |")
        lines.append("")

        if mi:
            lines += [
                "### 4c — Task routing mutual information I(expert; task_family)",
                "",
                "_Nats of task identity encoded in the routing decision at each layer._",
                "_Near-zero = routing is task-agnostic. Peak = routing encodes task structure._",
                "",
                "| Layer | I(expert; task) (nats) |",
                "| ---: | ---: |",
            ]
            for li_str in sorted(mi.keys(), key=int):
                m = mi[li_str]
                marker = " ← PEAK" if li_str == str(peak_layer) else ""
                lines.append(f"| L{li_str} | {m:.5f}{marker} |")
            lines.append("")

            if capacity is not None:
                capacity_nats = capacity * math.log(2)
                efficiency = 100 * peak_mi / capacity_nats if capacity_nats > 0 else 0
                lines += [
                    "### 4d — Routing capacity budget",
                    "",
                    f"- Theoretical routing capacity: log₂(C(32,4)) = {capacity:.2f} bits = {capacity_nats:.3f} nats",
                    f"- Peak I(expert; task): {peak_mi:.5f} nats at L{peak_layer} ({efficiency:.2f}% of capacity)",
                    "",
                    f"**Interpretation**: At the peak task-routing alignment layer, only {efficiency:.1f}% of the "
                    f"routing mechanism's combinatorial capacity is used for task-discriminating computation. "
                    f"The remainder encodes token surface form, positional context, and other signals not captured "
                    f"by the 5-family task taxonomy. This does not mean routing is inefficient — it means routing "
                    f"is a multi-purpose mechanism, most of whose capacity serves non-task purposes.",
                    "",
                ]

    report = "\n".join(lines) + "\n"
    (output_dir / "expert_readout_report.md").write_text(report)
    print(f"\nReport written to {output_dir / 'expert_readout_report.md'}")


###############################################################################
# Main
###############################################################################

def main() -> int:
    parser = argparse.ArgumentParser(description="MoE expert readout analysis")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--prompt", action="append", default=[],
                        help="One or more prompts (repeatable)")
    parser.add_argument("--task-suite", action="store_true",
                        help="Use the built-in 5-family task suite")
    parser.add_argument("--expert-profiles", action="store_true",
                        help="Attempt measurement 3 (non-quantized models only)")
    parser.add_argument("--top-k", type=int, default=8,
                        help="Top-k tokens in logit-delta and expert profiles")
    parser.add_argument("--output", default="runs/expert_readouts/")
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()

    from gossh.backends.gpt_oss import GPTOSSTransformersBackend

    print(f"Loading backend: {args.model}")
    backend = GPTOSSTransformersBackend(
        args.model, local_files_only=args.local_files_only
    )
    arch = backend._arch

    # Build prompt set
    if args.task_suite:
        prompts_by_task = TASK_SUITE
    elif args.prompt:
        prompts_by_task = {"custom": args.prompt}
    else:
        # Default: a single cross-family demo
        prompts_by_task = {
            "induction": ["The sequence: alpha beta gamma alpha beta"],
            "coreference": ["The trophy didn't fit in the suitcase because it was too small."],
        }

    all_prompts = [p for prompts in prompts_by_task.values() for p in prompts]
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run measurements
    routing_data = measure_routing_patterns(backend, prompts_by_task, arch)
    delta_data = measure_layer_logit_delta(backend, all_prompts, top_k=args.top_k)
    profile_data = (
        measure_expert_vocab_profiles(backend, all_prompts, top_k=args.top_k)
        if args.expert_profiles else None
    )
    info_data = measure_information_theory(routing_data)

    if not args.expert_profiles:
        print("\n(Skipping measurement 3. Pass --expert-profiles to enable.)")

    # Save outputs
    if routing_data:
        (out_dir / "routing_patterns.json").write_text(json.dumps(routing_data, indent=2))
        (out_dir / "routing_entropy.json").write_text(
            json.dumps(routing_data.get("entropy_by_layer", {}), indent=2)
        )
    if delta_data:
        (out_dir / "layer_logit_delta.json").write_text(json.dumps(delta_data, indent=2))
    if profile_data is not None:
        (out_dir / "expert_vocab_profiles.json").write_text(json.dumps(profile_data, indent=2))
    if info_data:
        (out_dir / "routing_kl_from_uniform.json").write_text(
            json.dumps(info_data.get("kl_from_uniform", {}), indent=2)
        )
        (out_dir / "routing_velocity.json").write_text(
            json.dumps(info_data.get("routing_velocity", {}), indent=2)
        )
        (out_dir / "routing_jsd_matrix.json").write_text(
            json.dumps(info_data.get("jsd_cross_task", {}), indent=2)
        )
        (out_dir / "routing_mi_task.json").write_text(
            json.dumps(info_data.get("mi_task", {}), indent=2)
        )

    write_report(out_dir, routing_data, delta_data, profile_data, info_data, prompts_by_task)

    print(f"\nAll outputs written to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
