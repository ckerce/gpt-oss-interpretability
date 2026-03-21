#!/usr/bin/env python3
"""Stratify benchmark cases into analysis buckets from convergence outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


###############################################################################
#
# Classification
#
###############################################################################


def _strict_stratum(first_expected: int | None, final_correct: bool, stable: bool) -> str:
    if first_expected is None:
        return "incorrect_never_expected"
    if final_correct and stable:
        return "correct_stable"
    if final_correct and not stable:
        return "correct_unstable"
    if not final_correct:
        return "incorrect_early_expected"
    return "unclassified"


def _soft_stratum(first_expected: int | None, final_correct: bool, late_stable: bool) -> str:
    if first_expected is None:
        return "incorrect_never_expected"
    if final_correct and late_stable:
        return "correct_late_stable"
    if final_correct and not late_stable:
        return "correct_late_unstable"
    if not final_correct:
        return "incorrect_early_expected"
    return "unclassified"


def classify_case(case: dict, tail_length: int) -> dict:
    layer_details = case["layer_details"]
    expected = case["expected_label"]
    winners = [row["winner"] for row in layer_details]
    first_expected = next((row["layer"] for row in layer_details if row["winner"] == expected), None)
    last_expected = max((row["layer"] for row in layer_details if row["winner"] == expected), default=None)
    final_correct = bool(case["final_correct"])

    if first_expected is None:
        stable_after_first = False
    else:
        after_first = [row["winner"] for row in layer_details if row["layer"] >= first_expected]
        stable_after_first = all(winner == expected for winner in after_first)

    final_expected_streak = 0
    for winner in reversed(winners):
        if winner == expected:
            final_expected_streak += 1
        else:
            break

    strict_stratum = _strict_stratum(first_expected, final_correct, stable_after_first)
    late_stable = final_correct and final_expected_streak >= tail_length
    soft_stratum = _soft_stratum(first_expected, final_correct, late_stable)

    num_winner_flips = sum(
        1 for idx in range(1, len(winners))
        if winners[idx] != winners[idx - 1]
    )

    return {
        "case_id": case["case_id"],
        "task_name": case["task_name"],
        "expected_label": expected,
        "expected_text": case["expected_text"],
        "final_winner": case["final_winner"],
        "final_correct": int(final_correct),
        "expected_choice_convergence": case["expected_choice_convergence"],
        "final_choice_convergence": case["final_choice_convergence"],
        "first_expected_layer": first_expected,
        "last_expected_layer": last_expected,
        "num_winner_flips": num_winner_flips,
        "stable_after_first_expected": int(stable_after_first),
        "final_expected_streak": final_expected_streak,
        "tail_length": tail_length,
        "late_stable": int(late_stable),
        "strict_stratum": strict_stratum,
        "soft_stratum": soft_stratum,
    }


###############################################################################
#
# Main
#
###############################################################################


def main() -> int:
    parser = argparse.ArgumentParser(description="Stratify benchmark cases into analysis buckets")
    parser.add_argument(
        "--input",
        default="runs/convergence_calibration/convergence_calibration.json",
        help="Choice-relative convergence calibration JSON",
    )
    parser.add_argument(
        "--output",
        default="runs/analysis_set_stratification",
        help="Output directory for stratification artifacts",
    )
    parser.add_argument(
        "--tail-length",
        type=int,
        default=4,
        help="Consecutive final layers required for late-stable classification",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = Path(__file__).resolve().parent.parent / input_path
    data = json.loads(input_path.read_text())

    classified = [classify_case(case, tail_length=args.tail_length) for case in data["cases"]]

    strict_counts = Counter(case["strict_stratum"] for case in classified)
    soft_counts = Counter(case["soft_stratum"] for case in classified)
    by_task_strict: dict[str, Counter] = defaultdict(Counter)
    by_task_soft: dict[str, Counter] = defaultdict(Counter)
    for case in classified:
        by_task_strict[case["task_name"]][case["strict_stratum"]] += 1
        by_task_soft[case["task_name"]][case["soft_stratum"]] += 1

    payload = {
        "source": str(input_path),
        "tail_length": args.tail_length,
        "overall_counts": {
            "strict": dict(strict_counts),
            "soft": dict(soft_counts),
        },
        "by_task": {
            "strict": {task: dict(counter) for task, counter in sorted(by_task_strict.items())},
            "soft": {task: dict(counter) for task, counter in sorted(by_task_soft.items())},
        },
        "cases": classified,
        "recommended_sets": {
            "main_analysis_strict": [
                case["case_id"] for case in classified if case["strict_stratum"] == "correct_stable"
            ],
            "main_analysis_soft": [
                case["case_id"] for case in classified if case["soft_stratum"] == "correct_late_stable"
            ],
            "secondary_analysis_strict": [
                case["case_id"] for case in classified if case["strict_stratum"] == "correct_unstable"
            ],
            "secondary_analysis_soft": [
                case["case_id"] for case in classified if case["soft_stratum"] == "correct_late_unstable"
            ],
            "failure_analysis": [
                case["case_id"] for case in classified
                if case["strict_stratum"] in {"incorrect_early_expected", "incorrect_never_expected"}
            ],
        },
    }

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent.parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "analysis_set_stratification.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = ["# Analysis Set Stratification\n"]
    lines.append("## Overall Counts\n")
    lines.append("| Rule | Stratum | Count |")
    lines.append("| --- | --- | ---: |")
    for stratum, count in sorted(strict_counts.items()):
        lines.append(f"| strict | {stratum} | {count} |")
    for stratum, count in sorted(soft_counts.items()):
        lines.append(f"| soft_tail_{args.tail_length} | {stratum} | {count} |")

    lines.append("\n## By Task: Strict Rule\n")
    lines.append("| Task | correct_stable | correct_unstable | incorrect_early_expected | incorrect_never_expected |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for task, counts in sorted(by_task_strict.items()):
        lines.append(
            f"| {task} | "
            f"{counts.get('correct_stable', 0)} | "
            f"{counts.get('correct_unstable', 0)} | "
            f"{counts.get('incorrect_early_expected', 0)} | "
            f"{counts.get('incorrect_never_expected', 0)} |"
        )

    lines.append(f"\n## By Task: Soft Rule (final streak >= {args.tail_length})\n")
    lines.append("| Task | correct_late_stable | correct_late_unstable | incorrect_early_expected | incorrect_never_expected |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for task, counts in sorted(by_task_soft.items()):
        lines.append(
            f"| {task} | "
            f"{counts.get('correct_late_stable', 0)} | "
            f"{counts.get('correct_late_unstable', 0)} | "
            f"{counts.get('incorrect_early_expected', 0)} | "
            f"{counts.get('incorrect_never_expected', 0)} |"
        )

    lines.append("\n## Case Details\n")
    lines.append("| Task | Case | Strict | Soft | Final Winner | First Expected | Last Expected | Final Streak | Flips |")
    lines.append("| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |")
    for case in classified:
        lines.append(
            f"| {case['task_name']} | {case['case_id']} | {case['strict_stratum']} | {case['soft_stratum']} | "
            f"{case['final_winner']} | {case['first_expected_layer']} | "
            f"{case['last_expected_layer']} | {case['final_expected_streak']} | {case['num_winner_flips']} |"
        )

    (output_dir / "analysis_set_stratification.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {output_dir / 'analysis_set_stratification.json'}")
    print(f"Wrote {output_dir / 'analysis_set_stratification.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
