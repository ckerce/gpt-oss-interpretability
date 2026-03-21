#!/usr/bin/env python3
"""Targeted single-case analysis for benchmark cases.

This script is the bridge between broad benchmark sweeps and deeper
mechanistic analysis.  It keeps the scoring aligned with the benchmark
objective while narrowing attention to one case and a small set of
interventions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpt_oss_interp.benchmarks.tasks import all_tasks
from gpt_oss_interp.config import (
    InterventionKind,
    InterventionSpec,
    InterventionTarget,
    TargetUnit,
)


###############################################################################
#
# Case lookup
#
###############################################################################

def _find_case(case_id: str):
    for task in all_tasks():
        for case in task.cases:
            if case.case_id == case_id:
                return task, case
    raise KeyError(f"Unknown case_id: {case_id}")


###############################################################################
#
# Scoring helpers
#
###############################################################################

def _summarize_choice_logprobs(choice_logprobs: dict[str, float], expected_label: str) -> dict[str, object]:
    predicted_label = max(choice_logprobs, key=choice_logprobs.get)
    best_logprob = choice_logprobs[predicted_label]
    expected_logprob = choice_logprobs[expected_label]
    sorted_values = sorted(choice_logprobs.values(), reverse=True)
    runner_up = sorted_values[1] if len(sorted_values) > 1 else best_logprob
    if predicted_label == expected_label:
        margin = best_logprob - runner_up
    else:
        margin = expected_logprob - best_logprob
    return {
        "predicted_label": predicted_label,
        "expected_label": expected_label,
        "correct": int(predicted_label == expected_label),
        "best_logprob": best_logprob,
        "expected_logprob": expected_logprob,
        "margin": margin,
        "choice_logprobs": choice_logprobs,
    }


def _layer_margin_summary(layer_scores: dict[int, dict[str, float]], expected_label: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for layer_idx in sorted(layer_scores):
        choice_logprobs = layer_scores[layer_idx]
        rows.append({"layer": layer_idx, **_summarize_choice_logprobs(choice_logprobs, expected_label)})
    return rows


###############################################################################
#
# Intervention surface
#
###############################################################################

def _default_interventions_for_case(case_id: str) -> list[InterventionSpec]:
    if case_id == "induction_002":
        return [
            InterventionSpec(
                name="late_delta_L20",
                kind=InterventionKind.LAYER_SCALE,
                target=InterventionTarget(unit=TargetUnit.LAYER, layer_indices=(20,), note="Residual-delta block skip"),
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
                    note="Full MLP/MoE-side suppression at layer 20",
                ),
                scales=(0.0,),
                description="Full expert-side suppression at layer 20",
            ),
            InterventionSpec(
                name="all_heads_L20",
                kind=InterventionKind.HEAD_MASK,
                target=InterventionTarget(
                    unit=TargetUnit.HEAD,
                    layer_indices=(20,),
                    head_indices=tuple(range(64)),
                    note="Full attention-side suppression at layer 20",
                ),
                scales=(0.0,),
                description="Full attention-side suppression at layer 20",
            ),
            InterventionSpec(
                name="late_delta_L21",
                kind=InterventionKind.LAYER_SCALE,
                target=InterventionTarget(unit=TargetUnit.LAYER, layer_indices=(21,), note="Residual-delta block skip"),
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
                    note="Full attention-side suppression at layer 21",
                ),
                scales=(0.0,),
                description="Full attention-side suppression at layer 21",
            ),
            InterventionSpec(
                name="all_experts_L21",
                kind=InterventionKind.EXPERT_MASK,
                target=InterventionTarget(
                    unit=TargetUnit.EXPERT,
                    layer_indices=(21,),
                    expert_indices=tuple(range(32)),
                    note="Full MLP/MoE-side suppression at layer 21",
                ),
                scales=(0.0,),
                description="Full expert-side suppression at layer 21",
            ),
        ]
    raise ValueError(f"No default intervention surface defined for case {case_id}")


###############################################################################
#
# Reporting
#
###############################################################################

def _format_choice_scores(choice_logprobs: dict[str, float]) -> str:
    parts = [f"{label}={value:.3f}" for label, value in sorted(choice_logprobs.items())]
    return ", ".join(parts)


def _write_report(path: Path, payload: dict[str, object]) -> None:
    baseline = payload["baseline"]
    interventions = payload["interventions"]
    lines = [
        f"# Single-Case Analysis: {payload['case']['case_id']}",
        "",
        f"Task: `{payload['task']['name']}`",
        f"Behavior: `{payload['task']['behavior']}`",
        "",
        "## Prompt",
        "",
        f"`{payload['case']['prompt']}`",
        "",
        "Choices:",
    ]
    for label, text in payload["case"]["choices"].items():
        marker = " (expected)" if label == payload["case"]["expected_label"] else ""
        lines.append(f"- `{label}`: `{text}`{marker}")

    lines.extend(
        [
            "",
            "## Baseline",
            "",
            f"- Predicted label: `{baseline['predicted_label']}`",
            f"- Correct: `{baseline['correct']}`",
            f"- Margin: `{baseline['margin']:.3f}`",
            f"- Choice logprobs: {_format_choice_scores(baseline['choice_logprobs'])}",
            "",
            "## Intervention Summary",
            "",
            "| Intervention | Pred | Correct | Margin | Choice Logprobs |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )

    for item in interventions:
        final = item["final"]
        lines.append(
            f"| {item['name']} | `{final['predicted_label']}` | {final['correct']} | "
            f"{final['margin']:.3f} | {_format_choice_scores(final['choice_logprobs'])} |"
        )

    lines.extend(
        [
            "",
            "## Per-Layer Choice Margins",
            "",
            "Baseline and intervention trajectories over layers.",
        ]
    )

    for item in [{"name": "baseline", "rows": baseline["layer_rows"]}] + [
        {"name": it["name"], "rows": it["layer_rows"]} for it in interventions
    ]:
        lines.extend(
            [
                "",
                f"### {item['name']}",
                "",
                "| Layer | Pred | Correct | Margin | Choice Logprobs |",
                "| ---: | --- | ---: | ---: | --- |",
            ]
        )
        for row in item["rows"]:
            lines.append(
                f"| {row['layer']} | `{row['predicted_label']}` | {row['correct']} | "
                f"{row['margin']:.3f} | {_format_choice_scores(row['choice_logprobs'])} |"
            )

    path.write_text("\n".join(lines) + "\n")


###############################################################################
#
# Main
#
###############################################################################

def main() -> int:
    parser = argparse.ArgumentParser(description="Targeted single-case analysis")
    parser.add_argument("--case_id", default="induction_002")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory; defaults to runs/single_case_<case_id>",
    )
    args = parser.parse_args()

    from gpt_oss_interp.backends.transformers_gpt_oss import GPTOSSTransformersBackend

    task, case = _find_case(args.case_id)
    interventions = _default_interventions_for_case(args.case_id)

    output_dir = Path(args.output or f"runs/single_case_{args.case_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    backend = GPTOSSTransformersBackend(model_name=args.model)

    baseline_score = backend.score_case(case)
    baseline_layers = backend.score_case_by_layer(case)
    baseline = _summarize_choice_logprobs(baseline_score.choice_logprobs, case.expected_label)
    baseline["layer_rows"] = _layer_margin_summary(baseline_layers, case.expected_label)

    intervention_payloads: list[dict[str, object]] = []
    for spec in interventions:
        scale = spec.scales[0]
        backend.clear_interventions()
        backend.apply_intervention(spec, scale)
        final_score = backend.score_case(case)
        layer_scores = backend.score_case_by_layer(case)
        backend.clear_interventions()

        final_summary = _summarize_choice_logprobs(final_score.choice_logprobs, case.expected_label)
        intervention_payloads.append(
            {
                "name": f"{spec.name}@{scale:g}",
                "spec": {
                    "name": spec.name,
                    "kind": spec.kind.value,
                    "description": spec.description,
                    "target": {
                        "unit": spec.target.unit.value,
                        "layer_indices": list(spec.target.layer_indices),
                        "head_indices": list(spec.target.head_indices),
                        "expert_indices": list(spec.target.expert_indices),
                        "note": spec.target.note,
                    },
                    "params": dict(spec.params),
                },
                "final": final_summary,
                "layer_rows": _layer_margin_summary(layer_scores, case.expected_label),
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
            "metadata": case.metadata,
        },
        "baseline": baseline,
        "interventions": intervention_payloads,
    }

    (output_dir / "single_case_analysis.json").write_text(json.dumps(payload, indent=2) + "\n")
    _write_report(output_dir / "single_case_analysis.md", payload)

    print(f"Single-case analysis complete: {case.case_id}")
    print(f"  baseline: pred={baseline['predicted_label']} correct={baseline['correct']} margin={baseline['margin']:.3f}")
    for item in intervention_payloads:
        final = item["final"]
        print(
            f"  {item['name']}: pred={final['predicted_label']} "
            f"correct={final['correct']} margin={final['margin']:.3f}"
        )
    print(f"Reports written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
