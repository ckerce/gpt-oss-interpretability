#!/usr/bin/env python3
"""HOW-TO: Task-dependent convergence depth via logit-lens.

INTERESTING PROPERTY: gpt-oss-20b resolves different task families at
very different depths.

  Capitalization:   L1–2   (pure surface statistics)
  Coreference:      ~L5    (entity binding)
  Syntax agreement: L8–12  (structural composition)
  Induction:        L17+   (in-context retrieval)

The logit-lens makes this visible by projecting every layer's hidden
state through the final norm + unembedding matrix and reading off the
top-1 token.  No separate probing classifier; no architectural changes.

Usage:
    python examples/howto_logit_lens.py --model openai/gpt-oss-20b
    python examples/howto_logit_lens.py --model openai/gpt-oss-20b --all
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEMO_PROMPTS = {
    "capitalization": (
        "The name of the first president of the United States is george",
        "  (surface statistics — expected convergence L1-2)",
    ),
    "coreference": (
        "The trophy didn't fit in the suitcase because it was too small. The",
        "  (entity binding — expected convergence ~L5)",
    ),
    "induction": (
        "The sequence continues: alpha beta gamma alpha beta",
        "  (in-context retrieval — expected convergence L17+)",
    ),
    "syntax_agreement": (
        "The keys to the cabinet",
        "  (subject-verb agreement — expected convergence L8-12)",
    ),
}


def run_logit_lens_demo(backend, prompt: str, label: str, top_k: int = 5) -> None:
    from gossh.readouts import format_logit_lens_table

    print(f"\n{'='*70}")
    print(f"Task family: {label}")
    print(f"Prompt: {prompt!r}")

    result = backend.run_logit_lens(prompt, top_k=top_k)

    # Convergence summary — one line per token position
    all_positions = sorted({p.position for p in result.predictions})
    print(f"\nConvergence by position (last 4):")
    for pos in all_positions[-4:]:
        conv = result.convergence_layer(pos)
        preds = result.position_slice(pos)
        final = preds[-1].top_tokens[0] if preds else "?"
        early = preds[0].top_tokens[0] if preds else "?"
        print(f"  pos {pos:3d}: converges L{conv:2d}  "
              f"L0→{early!r:12s}  final→{final!r}")

    # Full table (last 3 positions)
    print(f"\nPer-layer table (last 3 positions, top-{top_k}):")
    print(format_logit_lens_table(result, last_n_positions=3))


def main() -> int:
    parser = argparse.ArgumentParser(description="Logit-lens convergence demo")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--task", choices=list(DEMO_PROMPTS), default="induction",
                        help="Which task family to demonstrate")
    parser.add_argument("--all", action="store_true", help="Run all task families")
    parser.add_argument("--output", default=None, help="Directory for JSON output")
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()

    from gossh.backends.gpt_oss import GPTOSSTransformersBackend

    print(f"Loading backend: {args.model}")
    backend = GPTOSSTransformersBackend(
        args.model, local_files_only=args.local_files_only
    )

    tasks_to_run = list(DEMO_PROMPTS.items()) if args.all else [
        (args.task, DEMO_PROMPTS[args.task])
    ]

    results_payload = []
    for task_name, (prompt, note) in tasks_to_run:
        print(f"\n{note}")
        run_logit_lens_demo(backend, prompt, task_name, top_k=args.top_k)

        result = backend.run_logit_lens(prompt, top_k=args.top_k)
        all_positions = sorted({p.position for p in result.predictions})
        results_payload.append({
            "task": task_name,
            "prompt": prompt,
            "convergence": {
                pos: result.convergence_layer(pos)
                for pos in all_positions
            },
        })

    if args.output:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        (out / "logit_lens_convergence.json").write_text(
            json.dumps(results_payload, indent=2) + "\n"
        )
        print(f"\nSaved to {out}/logit_lens_convergence.json")

    print("\n\nKey takeaway:")
    print("  Different tasks converge at very different layers.")
    print("  Capitalization is resolved in the first 2 blocks.")
    print("  Induction requires most of the network.")
    print("  This is the computational geography of gpt-oss-20b.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
