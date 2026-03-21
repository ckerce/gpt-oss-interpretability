#!/usr/bin/env python3
"""Capture and compare activations for a single benchmark case.

Focuses on the final prompt position and a small layer window so the
result stays interpretable and aligned with the benchmark decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gpt_oss_interp.capture.activation_cache import ActivationCache
from gpt_oss_interp.benchmarks.tasks import all_tasks
from gpt_oss_interp.config import (
    InterventionKind,
    InterventionSpec,
    InterventionTarget,
    TargetUnit,
)
from gpt_oss_interp.harmony.prompting import encode_prompt


###############################################################################
#
# Case and intervention lookup
#
###############################################################################

def _find_case(case_id: str):
    for task in all_tasks():
        for case in task.cases:
            if case.case_id == case_id:
                return task, case
    raise KeyError(f"Unknown case_id: {case_id}")


def _default_interventions_for_case(case_id: str) -> list[InterventionSpec]:
    if case_id == "induction_002":
        return [
            InterventionSpec(
                name="late_delta_L20",
                kind=InterventionKind.LAYER_SCALE,
                target=InterventionTarget(unit=TargetUnit.LAYER, layer_indices=(20,)),
                scales=(0.0,),
                description="Residual-preserving block-delta suppression at layer 20",
                params={"preserve_residual": True},
            ),
            InterventionSpec(
                name="all_experts_L20",
                kind=InterventionKind.EXPERT_MASK,
                target=InterventionTarget(
                    unit=TargetUnit.EXPERT,
                    layer_indices=(20,),
                    expert_indices=tuple(range(32)),
                ),
                scales=(0.0,),
                description="Full expert-side suppression at layer 20",
            ),
            InterventionSpec(
                name="late_delta_L21",
                kind=InterventionKind.LAYER_SCALE,
                target=InterventionTarget(unit=TargetUnit.LAYER, layer_indices=(21,)),
                scales=(0.0,),
                description="Residual-preserving block-delta suppression at layer 21",
                params={"preserve_residual": True},
            ),
            InterventionSpec(
                name="all_heads_L21",
                kind=InterventionKind.HEAD_MASK,
                target=InterventionTarget(
                    unit=TargetUnit.HEAD,
                    layer_indices=(21,),
                    head_indices=tuple(range(64)),
                ),
                scales=(0.0,),
                description="Full attention-side suppression at layer 21",
            ),
        ]
    raise ValueError(f"No default activation panel defined for case {case_id}")


###############################################################################
#
# Tensor summaries
#
###############################################################################

def _next_token_summary(backend, hidden_last_token: torch.Tensor) -> dict[str, object]:
    with torch.no_grad():
        norm_param = next(backend.structure.final_norm.parameters())
        lm_head_param = next(backend.structure.lm_head.parameters())
        device = norm_param.device
        normed = backend.structure.final_norm(hidden_last_token.unsqueeze(0).to(device=device, dtype=norm_param.dtype))
        logits = backend.structure.lm_head(normed.to(dtype=lm_head_param.dtype)).float().cpu()[0]
        log_probs = torch.log_softmax(logits, dim=-1)
        top_vals, top_ids = torch.topk(log_probs, k=5)
    return {
        "top_token_ids": top_ids.tolist(),
        "top_tokens": [backend.tokenizer.decode([tid]) for tid in top_ids.tolist()],
        "top_logprobs": [float(v) for v in top_vals.tolist()],
    }


def _vector_metrics(vec: torch.Tensor, baseline: torch.Tensor) -> dict[str, float]:
    diff = vec - baseline
    cosine = torch.nn.functional.cosine_similarity(
        vec.unsqueeze(0), baseline.unsqueeze(0), dim=-1
    ).item()
    return {
        "norm": float(torch.norm(vec).item()),
        "baseline_norm": float(torch.norm(baseline).item()),
        "delta_norm": float(torch.norm(diff).item()),
        "cosine_to_baseline": float(cosine),
        "max_abs_delta": float(diff.abs().max().item()),
        "mean_abs_delta": float(diff.abs().mean().item()),
    }


def _capture_harmony_prompt_activations(backend, prompt: str, layer_indices: list[int]):
    """Capture block outputs on the Harmony-formatted prompt path."""
    block_names = [backend.structure.block_names[i] for i in layer_indices]
    cache = ActivationCache(detach=True, to_cpu=True)
    handles = cache.register(backend.model, block_names)
    input_ids = torch.tensor([encode_prompt(backend.tokenizer, prompt)], device=backend.device)
    try:
        with torch.no_grad():
            backend.model(input_ids)
    finally:
        for h in handles:
            h.remove()
    return cache.records


###############################################################################
#
# Reporting
#
###############################################################################

def _write_report(path: Path, payload: dict[str, object]) -> None:
    lines = [
        f"# Activation Comparison: {payload['case']['case_id']}",
        "",
        f"Task: `{payload['task']['name']}`",
        f"Behavior: `{payload['task']['behavior']}`",
        f"Layers analyzed: {payload['layers']}",
        "",
        "## Prompt",
        "",
        f"`{payload['case']['prompt']}`",
        "",
        "## Baseline Final-Token Readout",
        "",
    ]

    baseline = payload["baseline"]
    for layer in payload["layers"]:
        info = baseline["layers"][str(layer)]
        top1 = info["next_token"]["top_tokens"][0]
        lp = info["next_token"]["top_logprobs"][0]
        lines.append(f"- Layer {layer}: top-1 `{top1}` ({lp:.3f})")

    lines.extend(
        [
            "",
            "## Intervention Comparison",
            "",
            "| Intervention | Layer | Cosine to Baseline | Delta Norm | Top-1 Token | Top-1 Logprob |",
            "| --- | ---: | ---: | ---: | --- | ---: |",
        ]
    )

    for item in payload["interventions"]:
        for layer in payload["layers"]:
            info = item["layers"][str(layer)]
            top1 = info["next_token"]["top_tokens"][0]
            lp = info["next_token"]["top_logprobs"][0]
            lines.append(
                f"| {item['name']} | {layer} | {info['metrics']['cosine_to_baseline']:.6f} | "
                f"{info['metrics']['delta_norm']:.6f} | `{top1}` | {lp:.3f} |"
            )

    lines.extend(["", "## Detailed Layer Summaries"])
    for item in [{"name": "baseline", "layers": baseline["layers"]}] + payload["interventions"]:
        lines.extend(["", f"### {item['name']}", ""])
        for layer in payload["layers"]:
            info = item["layers"][str(layer)]
            lines.append(f"- Layer {layer}")
            lines.append(f"  - top tokens: {info['next_token']['top_tokens']}")
            lines.append(f"  - top logprobs: {[round(v, 3) for v in info['next_token']['top_logprobs']]}")
            if item["name"] != "baseline":
                m = info["metrics"]
                lines.append(
                    f"  - metrics: cosine={m['cosine_to_baseline']:.6f}, "
                    f"delta_norm={m['delta_norm']:.6f}, mean_abs_delta={m['mean_abs_delta']:.6f}"
                )

    path.write_text("\n".join(lines) + "\n")


###############################################################################
#
# Main
#
###############################################################################

def main() -> int:
    parser = argparse.ArgumentParser(description="Capture single-case activations")
    parser.add_argument("--case_id", default="induction_002")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--layers", default="20,21,22,23")
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory; defaults to runs/single_case_<case_id>_activations",
    )
    args = parser.parse_args()

    from gpt_oss_interp.backends.transformers_gpt_oss import GPTOSSTransformersBackend

    layer_indices = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    task, case = _find_case(args.case_id)
    interventions = _default_interventions_for_case(args.case_id)
    output_dir = Path(args.output or f"runs/single_case_{args.case_id}_activations")
    output_dir.mkdir(parents=True, exist_ok=True)

    backend = GPTOSSTransformersBackend(model_name=args.model)

    # Baseline capture on the Harmony-formatted prompt path.  This keeps the
    # activation view aligned with the benchmark scoring path.
    baseline_records = _capture_harmony_prompt_activations(backend, case.prompt, layer_indices)
    baseline_layers: dict[str, dict[str, object]] = {}
    baseline_vectors: dict[int, torch.Tensor] = {}
    for li in layer_indices:
        name = backend.structure.block_names[li]
        record = next(r for r in baseline_records if r.layer_name == name)
        vec = record.tensor[0, -1].float()
        baseline_vectors[li] = vec
        baseline_layers[str(li)] = {
            "final_token_vector": vec.tolist(),
            "next_token": _next_token_summary(backend, vec),
        }

    payload = {
        "model": args.model,
        "task": {
            "name": task.name,
            "behavior": task.behavior,
            "description": task.description,
        },
        "case": {
            "case_id": case.case_id,
            "prompt": case.prompt,
            "choices": case.choices,
            "expected_label": case.expected_label,
            "metadata": case.metadata,
        },
        "layers": layer_indices,
        "baseline": {"layers": baseline_layers},
        "interventions": [],
        }

    for spec in interventions:
        backend.clear_interventions()
        backend.apply_intervention(spec, spec.scales[0])
        records = _capture_harmony_prompt_activations(backend, case.prompt, layer_indices)
        backend.clear_interventions()

        layer_payload: dict[str, dict[str, object]] = {}
        for li in layer_indices:
            name = backend.structure.block_names[li]
            record = next(r for r in records if r.layer_name == name)
            vec = record.tensor[0, -1].float()
            layer_payload[str(li)] = {
                "final_token_vector": vec.tolist(),
                "next_token": _next_token_summary(backend, vec),
                "metrics": _vector_metrics(vec, baseline_vectors[li]),
            }

        payload["interventions"].append(
            {
                "name": f"{spec.name}@{spec.scales[0]:g}",
                "spec": {
                    "kind": spec.kind.value,
                    "description": spec.description,
                    "params": dict(spec.params),
                },
                "layers": layer_payload,
            }
        )

    (output_dir / "activation_comparison.json").write_text(json.dumps(payload, indent=2) + "\n")
    _write_report(output_dir / "activation_comparison.md", payload)

    print(f"Activation capture complete: {case.case_id}")
    for item in payload["interventions"]:
        print(f"  {item['name']}")
        for li in layer_indices:
            m = item["layers"][str(li)]["metrics"]
            print(
                f"    L{li}: cosine={m['cosine_to_baseline']:.6f} "
                f"delta_norm={m['delta_norm']:.6f}"
            )
    print(f"Reports written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
