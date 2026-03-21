#!/usr/bin/env python3
"""Project a single case onto its benchmark-relevant decision direction.

For Harmony-formatted choices, the first predicted assistant token is often
a control token rather than the semantic answer token.  This script finds the
first divergent completion token between the two answer choices, captures the
shared-prefix hidden state at that prediction position, and projects it onto
the exact unembedding direction for the divergent tokens.
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
from gpt_oss_interp.harmony.prompting import encode_prompt_with_completion


###############################################################################
#
# Case and interventions
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
    raise ValueError(f"No default intervention panel defined for case {case_id}")


###############################################################################
#
# Shared-prefix decision position
#
###############################################################################

def _decision_prefix(tokenizer, prompt: str, choice_a: str, choice_b: str) -> dict[str, object]:
    full_a, start_a = encode_prompt_with_completion(tokenizer, prompt, choice_a)
    full_b, start_b = encode_prompt_with_completion(tokenizer, prompt, choice_b)
    if start_a != start_b:
        raise ValueError("Choice completions do not share the same prompt prefix length")

    comp_a = full_a[start_a:]
    comp_b = full_b[start_b:]
    diff_idx = None
    for i, (ta, tb) in enumerate(zip(comp_a, comp_b)):
        if ta != tb:
            diff_idx = i
            break
    if diff_idx is None:
        raise ValueError("Choices do not diverge in tokenization")

    shared_completion_prefix = comp_a[:diff_idx]
    prefix_ids = full_a[: start_a + diff_idx]
    target_a = comp_a[diff_idx]
    target_b = comp_b[diff_idx]
    suffix_a = comp_a[diff_idx + 1 :]
    suffix_b = comp_b[diff_idx + 1 :]
    if suffix_a != suffix_b:
        raise ValueError("Choices diverge in more than one place; suffix decomposition assumes shared suffix")
    return {
        "full_a": full_a,
        "full_b": full_b,
        "prompt_start": start_a,
        "shared_completion_prefix_ids": shared_completion_prefix,
        "decision_input_ids": prefix_ids,
        "decision_position_index": len(prefix_ids) - 1,
        "choice_token_ids": {"A": target_a, "B": target_b},
        "shared_suffix_ids": suffix_a,
    }


###############################################################################
#
# Capture and projection
#
###############################################################################

def _capture_prefix_hidden_states(backend, input_ids: list[int], layer_indices: list[int]) -> dict[int, torch.Tensor]:
    block_names = [backend.structure.block_names[i] for i in layer_indices]
    cache = ActivationCache(detach=True, to_cpu=True)
    handles = cache.register(backend.model, block_names)
    tensor_ids = torch.tensor([input_ids], device=backend.device)
    try:
        with torch.no_grad():
            backend.model(tensor_ids)
    finally:
        for h in handles:
            h.remove()

    outputs: dict[int, torch.Tensor] = {}
    for li in layer_indices:
        name = backend.structure.block_names[li]
        record = next(r for r in cache.records if r.layer_name == name)
        outputs[li] = record.tensor[0, -1].float()
    return outputs


def _decision_projection(backend, hidden_vec: torch.Tensor, token_a: int, token_b: int) -> dict[str, float | str]:
    norm_param = next(backend.structure.final_norm.parameters())
    lm_head_weight = backend.structure.lm_head.weight.detach().cpu().float()
    with torch.no_grad():
        normed = backend.structure.final_norm(
            hidden_vec.unsqueeze(0).to(device=norm_param.device, dtype=norm_param.dtype)
        ).cpu().float()[0]

    direction = lm_head_weight[token_a] - lm_head_weight[token_b]
    logit_a = torch.dot(normed, lm_head_weight[token_a]).item()
    logit_b = torch.dot(normed, lm_head_weight[token_b]).item()
    return {
        "logit_A": float(logit_a),
        "logit_B": float(logit_b),
        "logit_diff_A_minus_B": float(logit_a - logit_b),
        "direction_norm": float(torch.norm(direction).item()),
        "projection_on_direction": float(torch.dot(normed, direction).item()),
    }


def _total_choice_logprob_by_layer(backend, prompt: str, choice_text: str) -> dict[int, float]:
    full_ids, choice_start = encode_prompt_with_completion(backend.tokenizer, prompt, choice_text)
    input_ids = torch.tensor([full_ids], device=backend.device)

    cache = ActivationCache(detach=True, to_cpu=True)
    handles = cache.register(backend.model, backend.structure.block_names)
    try:
        with torch.no_grad():
            backend.model(input_ids)
    finally:
        for h in handles:
            h.remove()

    scores: dict[int, float] = {}
    norm_device = next(backend.structure.final_norm.parameters()).device
    for layer_idx, block_name in enumerate(backend.structure.block_names):
        record = cache.last(block_name)
        if record is None:
            continue
        hidden = record.tensor
        with torch.no_grad():
            normed = backend.structure.final_norm(hidden.to(norm_device))
            logits = backend.structure.lm_head(normed).cpu().float()
            log_probs = torch.log_softmax(logits[0], dim=-1)
        total = 0.0
        for i in range(choice_start, len(full_ids)):
            total += log_probs[i - 1, full_ids[i]].item()
        scores[layer_idx] = total
    return scores


###############################################################################
#
# Report
#
###############################################################################

def _write_report(path: Path, payload: dict[str, object]) -> None:
    lines = [
        f"# Decision-Direction Projection: {payload['case']['case_id']}",
        "",
        f"Task: `{payload['task']['name']}`",
        f"Behavior: `{payload['task']['behavior']}`",
        "",
        "## Prompt",
        "",
        f"`{payload['case']['prompt']}`",
        "",
        "## Decision Tokenization",
        "",
        f"- Shared completion prefix tokens: {payload['decision_prefix']['shared_completion_prefix_tokens']}",
        f"- Divergent token A: `{payload['decision_prefix']['choice_tokens']['A']}`",
        f"- Divergent token B: `{payload['decision_prefix']['choice_tokens']['B']}`",
        f"- Decision position index: `{payload['decision_prefix']['decision_position_index']}`",
        "",
        "## Layerwise Decision Direction",
        "",
        "| Condition | Layer | Local A-B | Total A-B | Suffix Contribution | Local Pred | Total Pred |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]

    def add_rows(name: str, rows: dict[str, object]):
        for layer in payload["layers"]:
            info = rows[str(layer)]
            local_pred = "A" if info["local_logit_diff_A_minus_B"] > 0 else "B"
            total_pred = "A" if info["total_choice_diff_A_minus_B"] > 0 else "B"
            lines.append(
                f"| {name} | {layer} | {info['local_logit_diff_A_minus_B']:.3f} | "
                f"{info['total_choice_diff_A_minus_B']:.3f} | {info['suffix_contribution']:.3f} | "
                f"`{local_pred}` | `{total_pred}` |"
            )

    add_rows("baseline", payload["baseline"]["layers"])
    for item in payload["interventions"]:
        add_rows(item["name"], item["layers"])

    path.write_text("\n".join(lines) + "\n")


###############################################################################
#
# Main
#
###############################################################################

def main() -> int:
    parser = argparse.ArgumentParser(description="Project a single case onto its decision direction")
    parser.add_argument("--case_id", default="induction_002")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--layers", default="20,21,22,23")
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory; defaults to runs/single_case_<case_id>_decision_direction",
    )
    args = parser.parse_args()

    from gpt_oss_interp.backends.transformers_gpt_oss import GPTOSSTransformersBackend

    layer_indices = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    task, case = _find_case(args.case_id)
    output_dir = Path(args.output or f"runs/single_case_{args.case_id}_decision_direction")
    output_dir.mkdir(parents=True, exist_ok=True)

    backend = GPTOSSTransformersBackend(model_name=args.model)
    choice_a = case.choices["A"]
    choice_b = case.choices["B"]
    decision = _decision_prefix(backend.tokenizer, case.prompt, choice_a, choice_b)
    token_a = decision["choice_token_ids"]["A"]
    token_b = decision["choice_token_ids"]["B"]
    decision_input_ids = decision["decision_input_ids"]
    total_a = _total_choice_logprob_by_layer(backend, case.prompt, choice_a)
    total_b = _total_choice_logprob_by_layer(backend, case.prompt, choice_b)

    baseline_states = _capture_prefix_hidden_states(backend, decision_input_ids, layer_indices)
    baseline_payload = {}
    for li in layer_indices:
        local = _decision_projection(backend, baseline_states[li], token_a, token_b)
        total_diff = total_a[li] - total_b[li]
        baseline_payload[str(li)] = {
            **local,
            "local_logit_diff_A_minus_B": local["logit_diff_A_minus_B"],
            "total_choice_diff_A_minus_B": float(total_diff),
            "suffix_contribution": float(total_diff - local["logit_diff_A_minus_B"]),
        }

    interventions = []
    for spec in _default_interventions_for_case(args.case_id):
        backend.clear_interventions()
        backend.apply_intervention(spec, spec.scales[0])
        states = _capture_prefix_hidden_states(backend, decision_input_ids, layer_indices)
        total_a = _total_choice_logprob_by_layer(backend, case.prompt, choice_a)
        total_b = _total_choice_logprob_by_layer(backend, case.prompt, choice_b)
        backend.clear_interventions()
        layer_payload = {}
        for li in layer_indices:
            local = _decision_projection(backend, states[li], token_a, token_b)
            total_diff = total_a[li] - total_b[li]
            layer_payload[str(li)] = {
                **local,
                "local_logit_diff_A_minus_B": local["logit_diff_A_minus_B"],
                "total_choice_diff_A_minus_B": float(total_diff),
                "suffix_contribution": float(total_diff - local["logit_diff_A_minus_B"]),
            }
        interventions.append(
            {
                "name": f"{spec.name}@{spec.scales[0]:g}",
                "layers": layer_payload,
            }
        )

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
        },
        "layers": layer_indices,
        "decision_prefix": {
            "shared_completion_prefix_ids": decision["shared_completion_prefix_ids"],
            "shared_completion_prefix_tokens": [
                backend.tokenizer.decode([tid]) for tid in decision["shared_completion_prefix_ids"]
            ],
            "choice_token_ids": decision["choice_token_ids"],
            "choice_tokens": {
                "A": backend.tokenizer.decode([token_a]),
                "B": backend.tokenizer.decode([token_b]),
            },
            "shared_suffix_ids": decision["shared_suffix_ids"],
            "shared_suffix_tokens": [
                backend.tokenizer.decode([tid]) for tid in decision["shared_suffix_ids"]
            ],
            "decision_position_index": decision["decision_position_index"],
        },
        "baseline": {"layers": baseline_payload},
        "interventions": interventions,
    }

    (output_dir / "decision_direction.json").write_text(json.dumps(payload, indent=2) + "\n")
    _write_report(output_dir / "decision_direction.md", payload)

    print(f"Decision-direction projection complete: {case.case_id}")
    for li in layer_indices:
        info = baseline_payload[str(li)]
        print(
            f"  baseline L{li}: local={info['local_logit_diff_A_minus_B']:.3f} "
            f"total={info['total_choice_diff_A_minus_B']:.3f}"
        )
    for item in interventions:
        print(f"  {item['name']}")
        for li in layer_indices:
            info = item["layers"][str(li)]
            print(
                f"    L{li}: local={info['local_logit_diff_A_minus_B']:.3f} "
                f"total={info['total_choice_diff_A_minus_B']:.3f}"
            )
    print(f"Reports written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
