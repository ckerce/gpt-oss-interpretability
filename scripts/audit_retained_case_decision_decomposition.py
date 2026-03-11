#!/usr/bin/env python3
"""Audit retained benchmark cases by local-vs-tail decision decomposition.

For each retained case, compare:
1. the local first-divergent-token preference
2. the total benchmark choice preference

This identifies which cases are clean local answer-token exemplars and which
are being rescued by later Harmony-conditioned tail contributions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpt_oss_interp.benchmarks.tasks import all_tasks
from gpt_oss_interp.harmony.prompting import encode_prompt_with_completion
from gpt_oss_interp.capture.activation_cache import ActivationCache


###############################################################################
#
# Helpers
#
###############################################################################

def _all_cases():
    for task in all_tasks():
        for case in task.cases:
            yield task, case


def _retained_case_ids(stratification_json: Path, set_name: str) -> list[str]:
    payload = json.loads(stratification_json.read_text())
    return list(payload["recommended_sets"][set_name])


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
        "completion_tokens_a": [tokenizer.decode([tid]) for tid in comp_a],
        "completion_tokens_b": [tokenizer.decode([tid]) for tid in comp_b],
        "diff_positions": diff_positions,
        "num_diff_positions": len(diff_positions),
        "same_len": len(comp_a) == len(comp_b),
        "shared_prefix_ids": comp_a[:first],
        "shared_prefix_tokens": [tokenizer.decode([tid]) for tid in comp_a[:first]],
        "decision_input_ids": full_a[: start_a + first],
        "decision_position_index": len(full_a[: start_a + first]) - 1,
        "token_a": comp_a[first],
        "token_b": comp_b[first],
        "token_a_text": tokenizer.decode([comp_a[first]]),
        "token_b_text": tokenizer.decode([comp_b[first]]),
        "tail_a_ids": comp_a[first + 1 :],
        "tail_b_ids": comp_b[first + 1 :],
        "tail_a_tokens": [tokenizer.decode([tid]) for tid in comp_a[first + 1 :]],
        "tail_b_tokens": [tokenizer.decode([tid]) for tid in comp_b[first + 1 :]],
    }


def _capture_prefix_hidden(backend, input_ids: list[int], layer_idx: int) -> torch.Tensor:
    block_name = backend.structure.block_names[layer_idx]
    cache = ActivationCache(detach=True, to_cpu=True)
    handles = cache.register(backend.model, [block_name])
    tensor_ids = torch.tensor([input_ids], device=backend.device)
    try:
        with torch.no_grad():
            backend.model(tensor_ids)
    finally:
        for h in handles:
            h.remove()
    record = cache.last(block_name)
    if record is None:
        raise RuntimeError(f"No activation captured for layer {layer_idx}")
    return record.tensor[0, -1].float()


def _local_diff(backend, hidden_vec: torch.Tensor, token_a: int, token_b: int) -> float:
    norm_param = next(backend.structure.final_norm.parameters())
    lm_head_weight = backend.structure.lm_head.weight.detach().cpu().float()
    with torch.no_grad():
        normed = backend.structure.final_norm(
            hidden_vec.unsqueeze(0).to(device=norm_param.device, dtype=norm_param.dtype)
        ).cpu().float()[0]
    return float(torch.dot(normed, lm_head_weight[token_a] - lm_head_weight[token_b]).item())


def _classification(local_diff: float, total_diff: float) -> str:
    if total_diff > 0 and local_diff > 0:
        return "local_support"
    if total_diff > 0 and local_diff <= 0:
        return "tail_rescued"
    if total_diff <= 0 and local_diff > 0:
        return "tail_reversed"
    return "local_and_total_negative"


###############################################################################
#
# Main
#
###############################################################################

def main() -> int:
    parser = argparse.ArgumentParser(description="Audit retained cases by local-vs-tail decomposition")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument(
        "--stratification_json",
        default="gpt-oss-interp/runs/analysis_set_stratification/analysis_set_stratification.json",
    )
    parser.add_argument("--set_name", default="main_analysis_soft")
    parser.add_argument("--layer", type=int, default=23, help="Layer to audit (default: final layer 23)")
    parser.add_argument(
        "--output",
        default="gpt-oss-interp/runs/retained_case_decision_audit",
        help="Output directory",
    )
    args = parser.parse_args()

    from gpt_oss_interp.backends.transformers_gpt_oss import GPTOSSTransformersBackend

    strat_path = Path(args.stratification_json)
    retained_ids = set(_retained_case_ids(strat_path, args.set_name))

    backend = GPTOSSTransformersBackend(model_name=args.model)

    rows = []
    by_id = {case.case_id: (task, case) for task, case in _all_cases()}
    for case_id in sorted(retained_ids):
        task, case = by_id[case_id]
        divergence = _first_divergence(backend.tokenizer, case.prompt, case.choices["A"], case.choices["B"])
        hidden_vec = _capture_prefix_hidden(backend, divergence["decision_input_ids"], args.layer)
        local_diff = _local_diff(backend, hidden_vec, divergence["token_a"], divergence["token_b"])
        total_scores = backend.score_case(case).choice_logprobs
        total_diff = total_scores["A"] - total_scores["B"]
        tail_contribution = total_diff - local_diff
        rows.append(
            {
                "case_id": case.case_id,
                "task_name": task.name,
                "behavior": task.behavior,
                "expected_text_A": case.choices["A"],
                "expected_text_B": case.choices["B"],
                "layer": args.layer,
                "num_diff_positions": divergence["num_diff_positions"],
                "same_len": divergence["same_len"],
                "shared_prefix_tokens": divergence["shared_prefix_tokens"],
                "local_token_A": divergence["token_a_text"],
                "local_token_B": divergence["token_b_text"],
                "tail_a_tokens": divergence["tail_a_tokens"],
                "tail_b_tokens": divergence["tail_b_tokens"],
                "local_diff_A_minus_B": float(local_diff),
                "total_diff_A_minus_B": float(total_diff),
                "tail_contribution": float(tail_contribution),
                "classification": _classification(local_diff, total_diff),
            }
        )

    summary = {}
    for row in rows:
        summary[row["classification"]] = summary.get(row["classification"], 0) + 1

    payload = {
        "model": args.model,
        "set_name": args.set_name,
        "layer": args.layer,
        "summary": summary,
        "rows": rows,
    }

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "decision_audit.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        f"# Retained Case Decision Audit ({args.set_name}, layer {args.layer})",
        "",
        "## Summary",
        "",
    ]
    for key, value in sorted(summary.items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend(
        [
            "",
            "| Case | Task | Diff Pos | Local A-B | Total A-B | Tail Contribution | Class |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['task_name']} | {row['num_diff_positions']} | "
            f"{row['local_diff_A_minus_B']:.3f} | {row['total_diff_A_minus_B']:.3f} | "
            f"{row['tail_contribution']:.3f} | `{row['classification']}` |"
        )
    (out_dir / "decision_audit.md").write_text("\n".join(lines) + "\n")

    print(f"Retained-case decision audit complete: {args.set_name}")
    for row in rows:
        print(
            f"  {row['case_id']}: local={row['local_diff_A_minus_B']:.3f} "
            f"total={row['total_diff_A_minus_B']:.3f} class={row['classification']}"
        )
    print(f"Reports written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
