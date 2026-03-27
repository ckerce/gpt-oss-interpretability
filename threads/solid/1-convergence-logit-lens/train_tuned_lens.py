#!/usr/bin/env python3
"""Train tuned-lens translators for gpt-oss-20b.

Trains a per-layer low-rank affine translator T_l that maps hidden states
h_l to a geometry where ``final_norm + lm_head`` produces a meaningful
vocabulary distribution at every depth.  Without these translators, the
logit lens is only reliable from L21 onward (see runs/unembedding_validation/).

Training objective: minimise KL(P_l || P_L) where P_l is the translator's
prediction at layer l and P_L is the final-layer distribution (frozen target).

Usage::

    # Train on Thread 15 task suite (default, fast, ~5 min)
    python threads/solid/1-convergence-logit-lens/train_tuned_lens.py \\
        --model openai/gpt-oss-20b \\
        --output runs/tuned_lens/translators.pt

    # Train on a larger custom corpus for better generalisation
    python threads/solid/1-convergence-logit-lens/train_tuned_lens.py \\
        --model openai/gpt-oss-20b \\
        --corpus path/to/corpus.txt \\
        --n-epochs 5 \\
        --output runs/tuned_lens/translators.pt

    # Measure translation gap (raw vs tuned KL per layer) after training
    python threads/solid/1-convergence-logit-lens/train_tuned_lens.py \\
        --model openai/gpt-oss-20b \\
        --load runs/tuned_lens/translators.pt \\
        --measure-gap-only

After training, pass ``--tuned-lens runs/tuned_lens/translators.pt`` to
``run_logit_lens.py`` and ``calibrate_convergence.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# Default corpus: Thread 15 task suite (135 diverse prompts, on disk)
DEFAULT_CORPUS_PROMPTS = [
    # capitalization
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
    # coreference
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
    # induction
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
    # recency
    "I bought milk, eggs, and bread. Then I bought coffee. And finally I bought",
    "She visited Rome, then Paris, then London. Her last stop was",
    "The menu listed pasta, pizza, and soup. The waiter recommended the",
    "He tried the red shirt, then the blue one, then the green one. He chose the",
    "The countries visited were Spain, Italy, and Greece. The most recent was",
    # arithmetic
    "What is 3 plus 5? The answer is",
    "Seven times eight equals",
    "The square root of 144 is",
    "100 divided by 4 equals",
    "What is 15 percent of 200? The answer is",
    # factual recall
    "The chemical symbol for water is",
    "The speed of light in a vacuum is approximately",
    "William Shakespeare was born in the year",
    "The largest planet in our solar system is",
    "DNA stands for",
    # code
    "def factorial(n):\n    if n == 0:\n        return",
    "for i in range(10):\n    if i % 2 == 0:\n        print(",
    "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b,",
    # analogy
    "King is to queen as man is to",
    "Paris is to France as Rome is to",
    "Hot is to cold as day is to",
    "Doctor is to hospital as teacher is to",
    "Fish is to water as bird is to",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train tuned-lens translators")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument(
        "--corpus",
        default=None,
        help="Path to a plain-text file with one prompt per line. "
             "Defaults to the built-in 60-prompt task suite.",
    )
    parser.add_argument("--rank", type=int, default=32,
                        help="Low-rank bottleneck (default 32 ≈ 4.5M params)")
    parser.add_argument("--n-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--output", default="runs/tuned_lens/translators.pt")
    parser.add_argument(
        "--load",
        default=None,
        help="Path to pre-trained translators (skip training, go to gap measurement).",
    )
    parser.add_argument(
        "--base-translators",
        default=None,
        help="Path to frozen base translators to apply before training new ones. "
             "Trains a residual correction on top of the base — use to test whether "
             "the KL floor is rank-limited or genuinely nonlinear.",
    )
    parser.add_argument(
        "--measure-gap-only",
        action="store_true",
        help="Skip training; only measure KL gap (requires --load).",
    )
    parser.add_argument("--gap-output", default=None,
                        help="JSON file for KL gap results.")
    args = parser.parse_args(argv)

    from gpt_oss_interp.backends.transformers_gpt_oss import GPTOSSTransformersBackend
    from gpt_oss_interp.readouts.tuned_lens import (
        TunedLensTranslators,
        measure_translation_gap,
        train_tuned_lens,
    )

    print(f"Loading backend: {args.model}")
    backend = GPTOSSTransformersBackend(model_name=args.model, local_files_only=True)

    # ── corpus ────────────────────────────────────────────────────────────
    if args.corpus:
        prompts = [
            line.strip() for line in Path(args.corpus).read_text().splitlines()
            if line.strip()
        ]
        print(f"Loaded {len(prompts)} prompts from {args.corpus}")
    else:
        prompts = DEFAULT_CORPUS_PROMPTS
        print(f"Using built-in corpus: {len(prompts)} prompts")

    # ── load base translators (for residual experiment) ────────────────────
    base_translators = None
    if args.base_translators:
        print(f"Loading base translators from {args.base_translators}")
        base_translators = TunedLensTranslators.load(args.base_translators)

    # ── load or train ─────────────────────────────────────────────────────
    if args.load:
        print(f"Loading translators from {args.load}")
        translators = TunedLensTranslators.load(args.load)
    elif args.measure_gap_only:
        parser.error("--measure-gap-only requires --load")
        return 1
    else:
        if base_translators is not None:
            print(
                f"\nTraining residual translators on top of frozen base "
                f"(rank={args.rank}, epochs={args.n_epochs}, lr={args.lr})"
            )
            print("Objective: KL(T2_l(T1_l(h_l)) || h_L) — T1 frozen.\n")
        else:
            print(
                f"\nTraining tuned-lens translators "
                f"(rank={args.rank}, epochs={args.n_epochs}, lr={args.lr})"
            )
            print("Training objective: KL(T_l(h_l) || h_L) minimised per layer.\n")
        translators = train_tuned_lens(
            backend,
            prompts,
            rank=args.rank,
            n_epochs=args.n_epochs,
            lr=args.lr,
            base_translators=base_translators,
            verbose=True,
        )
        out_path = Path(args.output)
        translators.save(out_path)
        print(f"\nTranslators saved to {out_path}")

    # ── measure gap ───────────────────────────────────────────────────────
    # If base_translators provided, measure the full chain T1→T2
    measure_translators = translators
    if base_translators is not None:
        from gpt_oss_interp.readouts.tuned_lens import make_chained_translator
        measure_translators = make_chained_translator(base_translators, translators)
        print("\nMeasuring KL gap (raw / T1 / T1+T2 chained) per layer ...")
    else:
        print("\nMeasuring KL gap (raw logit-lens vs tuned-lens) per layer ...")

    gap = measure_translation_gap(backend, prompts[:20], translators=measure_translators)
    gap_base = (
        measure_translation_gap(backend, prompts[:20], translators=base_translators)
        if base_translators is not None else None
    )

    if gap_base is not None:
        print(
            f"\n{'Layer':>6}  {'Raw KL':>10}  {'T1 KL':>10}  {'T1+T2 KL':>10}  {'Extra reduction':>15}"
        )
        print("-" * 60)
        for l, (raw, t1, t2) in enumerate(zip(
            gap["raw"], gap_base.get("tuned", []), gap.get("tuned", [])
        )):
            extra = (t1 - t2) / (t1 + 1e-12) * 100
            marker = " ◀" if raw > 1.0 else ""
            print(f"  L{l:02d}  {raw:10.4f}  {t1:10.4f}  {t2:10.4f}  {extra:14.1f}%{marker}")
    else:
        print(
            f"\n{'Layer':>6}  {'Raw KL':>10}  {'Tuned KL':>10}  {'Reduction':>10}"
        )
        print("-" * 44)
        for l, (raw, tuned) in enumerate(zip(gap["raw"], gap.get("tuned", []))):
            reduction = (raw - tuned) / (raw + 1e-12) * 100
            marker = " ◀" if raw > 1.0 else ""
            print(f"  L{l:02d}  {raw:10.4f}  {tuned:10.4f}  {reduction:9.1f}%{marker}")

    if args.gap_output:
        gap_path = Path(args.gap_output)
        gap_path.parent.mkdir(parents=True, exist_ok=True)
        gap_path.write_text(json.dumps(gap, indent=2) + "\n")
        print(f"\nKL gap data saved to {gap_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
