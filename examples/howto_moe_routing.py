#!/usr/bin/env python3
"""HOW-TO: MoE routing capture under MXFP4 quantization.

INTERESTING PROPERTY: gpt-oss-20b uses MXFP4 fused kernels that make the
router gate completely invisible to standard Python hooks.  The sidecar
solves this by maintaining bf16 router weight clones in a separate subprocess
and routing via hidden states captured BEFORE the fused kernel fires.

There are four capture paths, tried in priority order:

  Path 0 (sidecar)             — attach a MoeSidecar; always works
  Path 1 (output_router_logits) — works for non-quantized checkpoints
  Path 2 (hook-based)          — works if gate is a Python-visible module
  Path 3 (unavailable)         — MXFP4 fused — emits informative message

This script demonstrates:
  - Why paths 1 and 2 fail under MXFP4
  - How the sidecar works (InputCapture pre-hook → IPC → RouterSidecarModel)
  - How to validate that sidecar routing matches expected bf16 computation
  - What routing patterns look like across layers and task families

Usage:
    python examples/howto_moe_routing.py --model openai/gpt-oss-20b
    python examples/howto_moe_routing.py --model openai/gpt-oss-20b --validate
"""
from __future__ import annotations

import argparse

DEMO_PROMPTS = {
    "capitalization": "The name of the first president is george",
    "induction": "The sequence: alpha beta gamma alpha beta",
    "coreference": "The trophy didn't fit in the suitcase because it was too small.",
}


def show_routing_decisions(decisions, num_experts: int, top_k: int) -> None:
    """Print a compact routing table."""
    print(f"\n  {'Layer':5s}  {'Tokens':6s}  {'Top-{} experts (first token)'.format(top_k):30s}  "
          f"{'Top weights':20s}")
    print(f"  {'-----':5s}  {'------':6s}  {'-'*30}  {'-'*20}")
    for d in decisions:
        experts = d.selected_experts[0].tolist()   # first token
        weights = [f"{w:.3f}" for w in d.expert_weights[0].tolist()]
        print(f"  L{d.layer_idx:2d}     {d.token_count:4d}    "
              f"{str(experts):30s}  {str(weights)}")


def demo_routing_paths(backend) -> None:
    """Demonstrate what each capture path does (and why some fail under MXFP4)."""
    print("\n" + "="*60)
    print("Routing capture path priority")
    print("="*60)

    prompt = DEMO_PROMPTS["induction"]
    print(f"\nPrompt: {prompt!r}")

    # Path 1/2 attempt (no sidecar attached)
    print("\nAttempting path 1 (output_router_logits) + path 2 (hooks) ...")
    decisions = backend.capture_routing(prompt)
    if decisions:
        print(f"  Got {len(decisions)} routing decisions via path 1 or 2.")
        show_routing_decisions(decisions, backend._arch.num_experts, backend._arch.top_k)
    else:
        print("  Both paths returned empty — model is running under MXFP4.")
        print("  → This is expected. Use the sidecar (path 0).")


def demo_sidecar(backend) -> None:
    """Demonstrate the sidecar workflow end-to-end."""
    from gossh.sidecar import MoeSidecar, RouterWeightExtractor

    print("\n" + "="*60)
    print("Sidecar routing (path 0 — MXFP4-safe)")
    print("="*60)

    # Extract router weights (bf16, from named_parameters — not dequantized)
    print("\nExtracting bf16 router weights from model...")
    weights = RouterWeightExtractor().extract(backend.model, backend.structure)
    if not weights:
        print("ERROR: No router weights found. Check structure.gate_names.")
        return
    print(f"  Found weights for {len(weights)} MoE layers.")
    first_layer = sorted(weights.keys())[0]
    w = weights[first_layer]
    print(f"  Layer {first_layer}: shape {tuple(w.shape)}  dtype={w.dtype}")
    print(f"  (num_experts={w.shape[0]}, hidden_dim={w.shape[1]})")

    arch = backend._arch

    with MoeSidecar(weights, top_k=arch.top_k) as sidecar:
        print(f"\nSidecar started. is_running={sidecar.is_running()}  "
              f"ping={sidecar.ping()}")

        backend.attach_sidecar(sidecar)
        print("Sidecar attached to backend (path 0 now active).")

        for task_name, prompt in DEMO_PROMPTS.items():
            print(f"\n--- {task_name} ---")
            print(f"  Prompt: {prompt!r}")
            decisions = backend.capture_routing(prompt)
            print(f"  Got {len(decisions)} routing decisions.")
            if decisions:
                show_routing_decisions(decisions, arch.num_experts, arch.top_k)

        backend.detach_sidecar()

    print(f"\nSidecar stopped. is_running={sidecar.is_running()}")


def demo_validation(backend) -> None:
    """Run the 3-layer sidecar validation pipeline."""
    import torch
    from gossh.sidecar import (
        MoeSidecar, RouterWeightExtractor, RouterSidecarModel,
        run_full_validation, format_validation_report,
    )

    print("\n" + "="*60)
    print("Sidecar validation (numerical + sensitivity + cross-reference)")
    print("="*60)

    weights = RouterWeightExtractor().extract(backend.model, backend.structure)
    if not weights:
        print("No router weights available.")
        return

    arch = backend._arch
    prompt = DEMO_PROMPTS["induction"]

    # Capture hidden states for validation
    from gossh.capture import InputCapture
    capture = InputCapture()
    capture.register(backend.model, backend.structure.mlp_names)

    input_ids = backend.tokenizer.encode(prompt, return_tensors="pt").to(backend.device)
    with torch.no_grad():
        backend.model(input_ids)
    capture.remove_hooks()

    hidden = capture.captured
    print(f"\nCaptured hidden states for {len(hidden)} layers.")

    # Local reference routing (in-process)
    local_model = RouterSidecarModel(weights, top_k=arch.top_k)
    ref_decisions = local_model.route_all(hidden)

    # Validate sidecar against local reference
    with MoeSidecar(weights, top_k=arch.top_k) as sidecar:
        report = run_full_validation(sidecar, ref_decisions, hidden)

    print(format_validation_report(report))
    print(f"\nValidation passed: {report.passed}")
    if report.flagged_layers:
        print(f"Flagged layers: {report.flagged_layers}")


def main() -> int:
    parser = argparse.ArgumentParser(description="MoE routing capture demo")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--validate", action="store_true",
                        help="Also run the 3-layer sidecar validation pipeline")
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()

    from gossh.backends.gpt_oss import GPTOSSTransformersBackend

    print(f"Loading backend: {args.model}")
    backend = GPTOSSTransformersBackend(
        args.model, local_files_only=args.local_files_only
    )

    demo_routing_paths(backend)
    demo_sidecar(backend)

    if args.validate:
        demo_validation(backend)

    print("\n\nKey takeaway:")
    print("  MXFP4 kernels make the router invisible to Python hooks.")
    print("  The sidecar holds bf16 gate clones in a separate process.")
    print("  InputCapture pre-hooks fire BEFORE the fused kernel runs.")
    print("  Together they recover routing decisions for any MXFP4 model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
