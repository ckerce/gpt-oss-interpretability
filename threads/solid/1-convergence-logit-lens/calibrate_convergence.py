#!/usr/bin/env python3
"""Calibrate choice-relative convergence across benchmark task families."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path



###############################################################################
#
# Helpers
#
###############################################################################


def _mean(values: list[int | float]) -> float | None:
    if not values:
        return None
    return float(statistics.mean(values))


def _std(values: list[int | float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    return float(statistics.pstdev(values))


def _safe_min(values: list[int | float]) -> int | float | None:
    return min(values) if values else None


def _safe_max(values: list[int | float]) -> int | float | None:
    return max(values) if values else None


def _format_float(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "NA"
    return f"{value:.2f}"


def _layer_winner(choice_scores: dict[str, float]) -> str | None:
    if not choice_scores:
        return None
    return max(choice_scores, key=choice_scores.get)


###############################################################################
#
# Main
#
###############################################################################


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate choice-relative convergence")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--output", default="runs/convergence_calibration")
    parser.add_argument(
        "--task-names",
        default=None,
        help="Optional comma-separated task names to include (default: all tasks)",
    )
    parser.add_argument(
        "--case-ids",
        default=None,
        help="Optional comma-separated case IDs to include (default: all cases in selected tasks)",
    )
    args = parser.parse_args()

    from gpt_oss_interp.backends.transformers_gpt_oss import GPTOSSTransformersBackend
    from gpt_oss_interp.benchmarks.tasks import all_tasks

    selected_task_names = None
    if args.task_names:
        selected_task_names = {
            name.strip() for name in args.task_names.split(",") if name.strip()
        }
    selected_case_ids = None
    if args.case_ids:
        selected_case_ids = {
            case_id.strip() for case_id in args.case_ids.split(",") if case_id.strip()
        }

    print(f"Initializing backend: {args.model}", flush=True)
    backend = GPTOSSTransformersBackend(model_name=args.model)

    rows: list[dict[str, object]] = []

    for task in all_tasks():
        if selected_task_names is not None and task.name not in selected_task_names:
            continue
        print(f"\nTask family: {task.name}", flush=True)
        for case in task.cases:
            if selected_case_ids is not None and case.case_id not in selected_case_ids:
                continue
            layer_scores = backend.score_case_by_layer(case)
            ordered_layers = sorted(layer_scores)
            if not ordered_layers:
                raise RuntimeError(f"No layer scores produced for {case.case_id}")

            layer_rows = []
            for layer_idx in ordered_layers:
                choice_scores = layer_scores[layer_idx]
                winner = _layer_winner(choice_scores)
                layer_rows.append(
                    {
                        "layer": layer_idx,
                        "winner": winner,
                        "expected_logprob": choice_scores[case.expected_label],
                        "choice_logprobs": choice_scores,
                        "margin_vs_runner_up": (
                            max(choice_scores.values()) - sorted(choice_scores.values(), reverse=True)[1]
                            if len(choice_scores) > 1 else 0.0
                        ),
                    }
                )

            final_winner = layer_rows[-1]["winner"]
            expected_convergence = next(
                (row["layer"] for row in layer_rows if row["winner"] == case.expected_label),
                None,
            )
            final_choice_convergence = next(
                (row["layer"] for row in layer_rows if row["winner"] == final_winner),
                None,
            )

            row = {
                "task_name": task.name,
                "behavior": task.behavior,
                "case_id": case.case_id,
                "expected_label": case.expected_label,
                "expected_text": case.choices[case.expected_label],
                "final_winner": final_winner,
                "final_correct": int(final_winner == case.expected_label),
                "expected_choice_convergence": expected_convergence,
                "final_choice_convergence": final_choice_convergence,
                "layer_details": layer_rows,
            }
            rows.append(row)

            print(
                f"  {case.case_id}: expected={expected_convergence} "
                f"final={final_choice_convergence} final_winner={final_winner}",
                flush=True,
            )

    family_summary: dict[str, dict[str, object]] = {}
    for task_name in sorted({row["task_name"] for row in rows}):
        family_rows = [row for row in rows if row["task_name"] == task_name]
        expected_layers = [
            row["expected_choice_convergence"]
            for row in family_rows
            if row["expected_choice_convergence"] is not None
        ]
        final_layers = [
            row["final_choice_convergence"]
            for row in family_rows
            if row["final_choice_convergence"] is not None
        ]
        family_summary[task_name] = {
            "num_cases": len(family_rows),
            "final_correct_rate": _mean([row["final_correct"] for row in family_rows]),
            "expected_choice_mean": _mean(expected_layers),
            "expected_choice_std": _std(expected_layers),
            "expected_choice_min": _safe_min(expected_layers),
            "expected_choice_max": _safe_max(expected_layers),
            "final_choice_mean": _mean(final_layers),
            "final_choice_std": _std(final_layers),
            "final_choice_min": _safe_min(final_layers),
            "final_choice_max": _safe_max(final_layers),
        }

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": args.model,
        "task_names": sorted(selected_task_names) if selected_task_names is not None else None,
        "case_ids": sorted(selected_case_ids) if selected_case_ids is not None else None,
        "family_summary": family_summary,
        "cases": rows,
    }
    (output_dir / "convergence_calibration.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = ["# Choice-Relative Convergence Calibration\n"]
    lines.append("| Task | Cases | Final Correct Rate | Expected Mean | Expected Std | Expected Range | Final Mean | Final Std | Final Range |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |")
    for task_name, summary in sorted(family_summary.items()):
        lines.append(
            "| "
            f"{task_name} | "
            f"{summary['num_cases']} | "
            f"{_format_float(summary['final_correct_rate'])} | "
            f"{_format_float(summary['expected_choice_mean'])} | "
            f"{_format_float(summary['expected_choice_std'])} | "
            f"{summary['expected_choice_min']} - {summary['expected_choice_max']} | "
            f"{_format_float(summary['final_choice_mean'])} | "
            f"{_format_float(summary['final_choice_std'])} | "
            f"{summary['final_choice_min']} - {summary['final_choice_max']} |"
        )

    lines.append("\n## Case Details\n")
    lines.append("| Task | Case | Expected | Final Winner | Expected Conv | Final Conv |")
    lines.append("| --- | --- | --- | --- | ---: | ---: |")
    for row in rows:
        lines.append(
            "| "
            f"{row['task_name']} | "
            f"{row['case_id']} | "
            f"`{row['expected_text']}` | "
            f"{row['final_winner']} | "
            f"{row['expected_choice_convergence']} | "
            f"{row['final_choice_convergence']} |"
        )

    (output_dir / "convergence_calibration.md").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {output_dir / 'convergence_calibration.json'}", flush=True)
    print(f"Wrote {output_dir / 'convergence_calibration.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
