#!/usr/bin/env python3
"""Inspect gpt-oss model structure for hook registration.

Prints all named modules, their types, and parameter shapes so that
the backend can register hooks at the right locations.

Usage:
    python scripts/inspect_model.py --model openai/gpt-oss-20b
    python scripts/inspect_model.py --model openai/gpt-oss-20b --layer 0 --verbose
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def inspect_model(model_name: str, layer: int | None, verbose: bool) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"  vocab size  : {tokenizer.vocab_size}")
    print(f"  model_max_length: {tokenizer.model_max_length}")
    print()

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto",
    )
    model.eval()
    print()

    # Model config
    config = model.config
    print("=" * 72)
    print("MODEL CONFIG")
    print("=" * 72)
    for key in sorted(vars(config)):
        if key.startswith("_"):
            continue
        val = getattr(config, key)
        if not callable(val):
            print(f"  {key}: {val}")
    print()

    # Named modules
    print("=" * 72)
    print("NAMED MODULES")
    print("=" * 72)
    for name, module in model.named_modules():
        type_name = type(module).__name__
        if layer is not None:
            # Only show modules for this layer
            if f".{layer}." not in name and not name.endswith(f".{layer}"):
                if name and not any(name == prefix for prefix in ["", "model", "model.model"]):
                    continue

        param_count = sum(p.numel() for p in module.parameters(recurse=False))
        indent = "  " * name.count(".")
        suffix = f"  ({param_count:,} params)" if param_count > 0 else ""
        print(f"{indent}{name}: {type_name}{suffix}")

        if verbose and param_count > 0:
            for pname, param in module.named_parameters(recurse=False):
                print(f"{indent}  .{pname}: {tuple(param.shape)} {param.dtype}")
    print()

    # Discover structure using our utility
    print("=" * 72)
    print("STRUCTURE DISCOVERY")
    print("=" * 72)
    from gpt_oss_interp.backends.transformers_gpt_oss import ModelStructure
    try:
        structure = ModelStructure(model)
        print(structure.summary())
    except RuntimeError as e:
        print(f"Discovery failed: {e}")
        print("Check the module listing above to identify the block pattern.")
    print()

    # Quick smoke test: tokenize and forward
    print("=" * 72)
    print("SMOKE TEST")
    print("=" * 72)
    test_prompt = "The capital of France is"
    inputs = tokenizer(test_prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    print(f"  prompt  : {test_prompt!r}")
    print(f"  tokens  : {inputs['input_ids'].shape}")

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    print(f"  logits  : {logits.shape}")

    # Top predictions for last token
    probs = torch.softmax(logits[0, -1].float(), dim=-1)
    top5 = torch.topk(probs, k=5)
    print("  top-5 next tokens:")
    for i in range(5):
        token = tokenizer.decode([top5.indices[i].item()])
        prob = top5.values[i].item()
        print(f"    {token!r:20s}  p={prob:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect gpt-oss model structure")
    parser.add_argument("--model", default="openai/gpt-oss-20b", help="Model name or path")
    parser.add_argument("--layer", type=int, default=None, help="Show only this layer index")
    parser.add_argument("--verbose", action="store_true", help="Show parameter shapes")
    args = parser.parse_args()
    inspect_model(args.model, args.layer, args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
