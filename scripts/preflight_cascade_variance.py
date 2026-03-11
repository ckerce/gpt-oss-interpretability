#!/usr/bin/env python3
"""Run the Entry 20 CASCADE variance preflight on audited clean cases.

For each case, this script:
1. finds the first divergent answer token
2. captures the block output at that decision position for layers 4-22
3. projects each layer through the final unembedding head to obtain a layerwise
   teacher distribution
4. solves the centered least-squares CASCADE target for that teacher
   distribution
5. records reconstruction KL, relative residual, and layer-over-layer delta_KL

The main purpose is to check whether the same-model CASCADE signal has enough
layerwise variation to support the bridge experiment defined in Entry 20.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpt_oss_interp.benchmarks.tasks import all_tasks
from gpt_oss_interp.capture.activation_cache import ActivationCache
from gpt_oss_interp.harmony.prompting import encode_prompt_with_completion


###############################################################################
#
# Case lookup and divergence
#
###############################################################################


def _find_case(case_id: str):
    for task in all_tasks():
        for case in task.cases:
            if case.case_id == case_id:
                return task, case
    raise KeyError(f"Unknown case_id: {case_id}")


def _first_divergence(tokenizer, prompt: str, choice_a: str, choice_b: str) -> dict[str, object]:
    full_a, start_a = encode_prompt_with_completion(tokenizer, prompt, choice_a)
    full_b, start_b = encode_prompt_with_completion(tokenizer, prompt, choice_b)
    if start_a != start_b:
        raise ValueError("Prompt prefix lengths differ")

    comp_a = full_a[start_a:]
    comp_b = full_b[start_b:]
    max_len = min(len(comp_a), len(comp_b))
    diff_positions = [i for i in range(max_len) if comp_a[i] != comp_b[i]]
    if not diff_positions:
        raise ValueError("Choices do not diverge")

    first = diff_positions[0]
    return {
        "decision_input_ids": full_a[: start_a + first],
        "decision_position_index": len(full_a[: start_a + first]) - 1,
        "shared_prefix_tokens": [tokenizer.decode([tid]) for tid in comp_a[:first]],
        "token_a": comp_a[first],
        "token_b": comp_b[first],
        "token_a_text": tokenizer.decode([comp_a[first]]),
        "token_b_text": tokenizer.decode([comp_b[first]]),
    }


###############################################################################
#
# Capture and CASCADE metrics
#
###############################################################################


def _capture_last_token_hiddens(backend, input_ids: list[int], layer_indices: list[int]) -> dict[int, torch.Tensor]:
    block_names = [backend.structure.block_names[i] for i in layer_indices]
    cache = ActivationCache(detach=True, to_cpu=True)
    handles = cache.register(backend.model, block_names)
    tensor_ids = torch.tensor([input_ids], device=backend.device)
    try:
        with torch.no_grad():
            backend.model(tensor_ids)
    finally:
        for handle in handles:
            handle.remove()

    records: dict[int, torch.Tensor] = {}
    for layer_idx in layer_indices:
        record = cache.last(backend.structure.block_names[layer_idx])
        if record is None:
            raise RuntimeError(f"No activation captured for layer {layer_idx}")
        records[layer_idx] = record.tensor[0, -1].float()
    return records


def _layer_log_probs(backend, hidden_vec: torch.Tensor) -> torch.Tensor:
    norm_module = backend.structure.final_norm
    lm_head = backend.structure.lm_head
    if norm_module is None or lm_head is None:
        raise RuntimeError("Model structure is missing final_norm or lm_head")

    norm_param = next(norm_module.parameters())
    lm_head_param = next(lm_head.parameters())
    normed = norm_module(hidden_vec.unsqueeze(0).to(device=norm_param.device, dtype=norm_param.dtype))
    logits = lm_head(normed.to(device=lm_head_param.device, dtype=lm_head_param.dtype)).float().cpu()[0]
    return torch.log_softmax(logits, dim=-1)


def _distribution_kl(teacher_log_probs: torch.Tensor, reconstructed_centered_logits: torch.Tensor) -> float:
    teacher_probs = teacher_log_probs.exp()
    recon_log_probs = torch.log_softmax(reconstructed_centered_logits, dim=-1)
    return float(torch.sum(teacher_probs * (teacher_log_probs - recon_log_probs)).item())


def _relative_residual(matrix: torch.Tensor, solution: torch.Tensor, rhs: torch.Tensor) -> float:
    residual = matrix @ solution - rhs
    denom = rhs.norm().item()
    if denom == 0.0:
        return float(residual.norm().item())
    return float(residual.norm().item() / denom)


def _coefficient_of_variation(values: list[float]) -> float | None:
    if not values:
        return None
    mean_value = statistics.fmean(values)
    if math.isclose(mean_value, 0.0, abs_tol=1e-12):
        return None
    if len(values) == 1:
        return 0.0
    return statistics.pstdev(values) / abs(mean_value)


def _case_payload(backend, case_id: str, layer_indices: list[int]) -> dict[str, object]:
    task, case = _find_case(case_id)
    divergence = _first_divergence(backend.tokenizer, case.prompt, case.choices["A"], case.choices["B"])
    decision_input_ids = divergence["decision_input_ids"]
    hiddens = _capture_last_token_hiddens(backend, decision_input_ids, layer_indices)

    if backend.structure.embed is None or backend.structure.lm_head is None:
        raise RuntimeError("Model structure is missing embed or lm_head")

    input_tensor = torch.tensor([decision_input_ids], device=backend.device)
    with torch.no_grad():
        embed_out = backend.structure.embed(input_tensor)
        if isinstance(embed_out, tuple):
            embed_out = embed_out[0]
        x_t = embed_out[0, -1].float().cpu()

    # Use the centered unembedding directly. Softmax is invariant to the global
    # logit shift introduced by centering, which avoids carrying both W and C W.
    unembedding = backend.structure.lm_head.weight.detach().float().cpu()
    centered_unembedding = unembedding
    centered_unembedding -= centered_unembedding.mean(dim=0, keepdim=True)
    centered_xt = centered_unembedding @ x_t

    teacher_log_probs = []
    for layer_idx in layer_indices:
        teacher_log_probs.append(_layer_log_probs(backend, hiddens[layer_idx]))
    teacher_matrix = torch.stack(teacher_log_probs, dim=1)  # [V, n_layers]
    centered_teacher = teacher_matrix - teacher_matrix.mean(dim=0, keepdim=True)
    rhs_matrix = centered_teacher - centered_xt.unsqueeze(1)

    solutions = torch.linalg.lstsq(centered_unembedding, rhs_matrix).solution
    reconstructed_logits = centered_unembedding @ (x_t.unsqueeze(1) + solutions)

    layer_rows = []
    prev_kl: float | None = None
    delta_values: list[float] = []
    abs_delta_values: list[float] = []
    kl_values: list[float] = []

    for column, layer_idx in enumerate(layer_indices):
        teacher_lp = teacher_matrix[:, column]
        rhs = rhs_matrix[:, column]
        solution = solutions[:, column]
        reconstructed = reconstructed_logits[:, column]
        reconstruction_kl = _distribution_kl(teacher_lp, reconstructed)
        relative_residual = _relative_residual(centered_unembedding, solution, rhs)
        delta_kl = None if prev_kl is None else reconstruction_kl - prev_kl
        if delta_kl is not None:
            delta_values.append(delta_kl)
            abs_delta_values.append(abs(delta_kl))
        kl_values.append(reconstruction_kl)
        top_id = int(torch.argmax(teacher_lp).item())
        layer_rows.append(
            {
                "layer": layer_idx,
                "reconstruction_kl": reconstruction_kl,
                "relative_residual": relative_residual,
                "delta_kl": delta_kl,
                "teacher_top_token_id": top_id,
                "teacher_top_token": backend.tokenizer.decode([top_id]),
                "teacher_top_logprob": float(teacher_lp[top_id].item()),
            }
        )
        prev_kl = reconstruction_kl

    return {
        "case_id": case.case_id,
        "task_name": task.name,
        "behavior": task.behavior,
        "prompt": case.prompt,
        "expected_label": case.expected_label,
        "choices": case.choices,
        "decision": {
            "position_index": divergence["decision_position_index"],
            "shared_prefix_tokens": divergence["shared_prefix_tokens"],
            "token_a": divergence["token_a_text"],
            "token_b": divergence["token_b_text"],
        },
        "layers": layer_rows,
        "summary": {
            "min_reconstruction_kl": min(kl_values),
            "max_reconstruction_kl": max(kl_values),
            "mean_reconstruction_kl": statistics.fmean(kl_values),
            "reconstruction_kl_cv": _coefficient_of_variation(kl_values),
            "delta_kl_mean": statistics.fmean(delta_values) if delta_values else None,
            "delta_kl_abs_mean": statistics.fmean(abs_delta_values) if abs_delta_values else None,
            "delta_kl_cv": _coefficient_of_variation(delta_values),
            "delta_kl_abs_cv": _coefficient_of_variation(abs_delta_values),
        },
    }


###############################################################################
#
# Reporting
#
###############################################################################


def _write_report(path: Path, payload: dict[str, object]) -> None:
    lines = [
        "# CASCADE Preflight Variance",
        "",
        f"Model: `{payload['model']}`",
        f"Layers: `{payload['layer_start']}-{payload['layer_end']}`",
        "",
        "This report checks whether same-model layerwise CASCADE reconstruction",
        "has enough variation to support the bridge experiment preflight.",
        "",
    ]

    for case in payload["cases"]:
        summary = case["summary"]
        lines.extend(
            [
                f"## {case['case_id']}",
                "",
                f"Task: `{case['task_name']}`",
                f"Behavior: `{case['behavior']}`",
                f"Decision token A/B: `{case['decision']['token_a']}` vs `{case['decision']['token_b']}`",
                f"Decision position index: `{case['decision']['position_index']}`",
                "",
                "Summary:",
                f"- mean reconstruction_KL: `{summary['mean_reconstruction_kl']:.8f}`",
                f"- reconstruction_KL CV: `{summary['reconstruction_kl_cv']}`",
                f"- mean delta_KL: `{summary['delta_kl_mean']}`",
                f"- mean |delta_KL|: `{summary['delta_kl_abs_mean']}`",
                f"- delta_KL CV: `{summary['delta_kl_cv']}`",
                f"- |delta_KL| CV: `{summary['delta_kl_abs_cv']}`",
                "",
                "| Layer | reconstruction_KL | relative_residual | delta_KL | top token |",
                "| ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in case["layers"]:
            delta_value = "NA" if row["delta_kl"] is None else f"{row['delta_kl']:.8f}"
            lines.append(
                f"| {row['layer']} | {row['reconstruction_kl']:.8f} | "
                f"{row['relative_residual']:.8f} | {delta_value} | "
                f"`{row['teacher_top_token']}` |"
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n")


###############################################################################
#
# Main
#
###############################################################################


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Entry 20 CASCADE variance preflight")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--cases", default="induction_004,coref_003,caps_002")
    parser.add_argument("--layer_start", type=int, default=4)
    parser.add_argument("--layer_end", type=int, default=22)
    parser.add_argument("--output", default="runs/bridge_preflight")
    args = parser.parse_args()

    from gpt_oss_interp.backends.transformers_gpt_oss import GPTOSSTransformersBackend

    case_ids = [part.strip() for part in args.cases.split(",") if part.strip()]
    layer_indices = list(range(args.layer_start, args.layer_end + 1))

    backend = GPTOSSTransformersBackend(model_name=args.model)
    case_payloads = [_case_payload(backend, case_id, layer_indices) for case_id in case_ids]

    payload = {
        "model": args.model,
        "layer_start": args.layer_start,
        "layer_end": args.layer_end,
        "cases": case_payloads,
    }

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "preflight_cascade_variance.json").write_text(json.dumps(payload, indent=2) + "\n")
    _write_report(out_dir / "preflight_cascade_variance.md", payload)

    print(f"CASCADE preflight complete for {len(case_payloads)} cases.")
    print(f"Wrote {out_dir / 'preflight_cascade_variance.json'}")
    print(f"Wrote {out_dir / 'preflight_cascade_variance.md'}")
    for case in case_payloads:
        summary = case["summary"]
        print(
            f"  {case['case_id']}: mean_KL={summary['mean_reconstruction_kl']:.8f} "
            f"delta_abs_cv={summary['delta_kl_abs_cv']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
