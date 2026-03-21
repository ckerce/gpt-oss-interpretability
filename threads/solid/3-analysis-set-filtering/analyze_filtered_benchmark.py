#!/usr/bin/env python3
"""Analyze existing benchmark results on filtered analysis sets."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


###############################################################################
#
# Helpers
#
###############################################################################


def _summarize(rows: list[dict]) -> dict[str, float | int]:
    if not rows:
        return {"count": 0, "accuracy": 0.0, "mean_margin": 0.0}
    return {
        "count": len(rows),
        "accuracy": sum(int(row["correct"]) for row in rows) / len(rows),
        "mean_margin": sum(float(row["margin"]) for row in rows) / len(rows),
    }


###############################################################################
#
# Main
#
###############################################################################


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze benchmark results on filtered sets")
    parser.add_argument(
        "--case-results",
        default="runs/gpt_oss_20b_sweep/case_results.csv",
        help="Benchmark case-results CSV",
    )
    parser.add_argument(
        "--stratification",
        default="runs/analysis_set_stratification/analysis_set_stratification.json",
        help="Analysis-set stratification JSON",
    )
    parser.add_argument(
        "--output",
        default="runs/filtered_benchmark_analysis",
        help="Output directory",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    case_results_path = Path(args.case_results)
    if not case_results_path.is_absolute():
        case_results_path = repo_root / case_results_path

    strat_path = Path(args.stratification)
    if not strat_path.is_absolute():
        strat_path = repo_root / strat_path

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with case_results_path.open() as f:
        rows = list(csv.DictReader(f))

    strat = json.loads(strat_path.read_text())
    sets = strat["recommended_sets"]
    set_membership = {
        "all_cases": None,
        "main_analysis_strict": set(sets["main_analysis_strict"]),
        "main_analysis_soft": set(sets["main_analysis_soft"]),
        "secondary_analysis_soft": set(sets["secondary_analysis_soft"]),
        "failure_analysis": set(sets["failure_analysis"]),
    }

    summary_by_set: dict[str, dict[str, dict]] = {}
    for set_name, members in set_membership.items():
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            if members is not None and row["case_id"] not in members:
                continue
            grouped[row["run_name"]].append(row)
        summary_by_set[set_name] = {
            run_name: _summarize(run_rows)
            for run_name, run_rows in sorted(grouped.items())
        }

    baseline_run = "early_heads_L2@0"
    payload = {
        "source_case_results": str(case_results_path),
        "source_stratification": str(strat_path),
        "baseline_run": baseline_run,
        "summary_by_set": summary_by_set,
    }
    (output_dir / "filtered_benchmark_analysis.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = ["# Filtered Benchmark Analysis\n"]
    lines.append("## Baseline Comparison\n")
    lines.append("| Set | Count | Accuracy | Mean Margin |")
    lines.append("| --- | ---: | ---: | ---: |")
    for set_name in ["all_cases", "main_analysis_strict", "main_analysis_soft", "secondary_analysis_soft", "failure_analysis"]:
        stats = summary_by_set[set_name].get(baseline_run, {"count": 0, "accuracy": 0.0, "mean_margin": 0.0})
        lines.append(
            f"| {set_name} | {stats['count']} | {stats['accuracy']:.3f} | {stats['mean_margin']:.3f} |"
        )

    lines.append("\n## All Runs by Set\n")
    lines.append("| Set | Run | Count | Accuracy | Mean Margin |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for set_name, run_map in summary_by_set.items():
        for run_name, stats in run_map.items():
            lines.append(
                f"| {set_name} | {run_name} | {stats['count']} | {stats['accuracy']:.3f} | {stats['mean_margin']:.3f} |"
            )

    (output_dir / "filtered_benchmark_analysis.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {output_dir / 'filtered_benchmark_analysis.json'}")
    print(f"Wrote {output_dir / 'filtered_benchmark_analysis.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
