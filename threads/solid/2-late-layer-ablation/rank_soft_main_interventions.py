#!/usr/bin/env python3
"""Rank existing interventions by effect on the soft main-analysis set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank interventions on the soft main-analysis set")
    parser.add_argument(
        "--input",
        default="runs/filtered_benchmark_analysis/filtered_benchmark_analysis.json",
        help="Filtered benchmark analysis JSON",
    )
    parser.add_argument(
        "--set-name",
        default="main_analysis_soft",
        help="Subset to analyze",
    )
    parser.add_argument(
        "--baseline-run",
        default="early_heads_L2@0",
        help="Baseline run name for delta computation",
    )
    parser.add_argument(
        "--output",
        default="runs/soft_main_intervention_ranking",
        help="Output directory",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = repo_root / input_path

    data = json.loads(input_path.read_text())
    run_map = data["summary_by_set"][args.set_name]
    baseline = run_map[args.baseline_run]

    ranking = []
    for run_name, stats in sorted(run_map.items()):
        ranking.append(
            {
                "run_name": run_name,
                "count": stats["count"],
                "accuracy": stats["accuracy"],
                "mean_margin": stats["mean_margin"],
                "delta_accuracy": stats["accuracy"] - baseline["accuracy"],
                "delta_margin": stats["mean_margin"] - baseline["mean_margin"],
            }
        )

    ranking_by_margin_drop = sorted(ranking, key=lambda row: (row["delta_margin"], row["delta_accuracy"]))
    ranking_by_accuracy_drop = sorted(ranking, key=lambda row: (row["delta_accuracy"], row["delta_margin"]))

    payload = {
        "source": str(input_path),
        "set_name": args.set_name,
        "baseline_run": args.baseline_run,
        "baseline_stats": baseline,
        "ranking_by_margin_drop": ranking_by_margin_drop,
        "ranking_by_accuracy_drop": ranking_by_accuracy_drop,
    }

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "soft_main_intervention_ranking.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = ["# Soft Main Intervention Ranking\n"]
    lines.append(f"Set: `{args.set_name}`\n")
    lines.append(f"Baseline: `{args.baseline_run}`\n")
    lines.append("| Run | Accuracy | Mean Margin | Delta Accuracy | Delta Margin |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for row in ranking_by_margin_drop:
        lines.append(
            f"| {row['run_name']} | {row['accuracy']:.3f} | {row['mean_margin']:.3f} | "
            f"{row['delta_accuracy']:+.3f} | {row['delta_margin']:+.3f} |"
        )

    (output_dir / "soft_main_intervention_ranking.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {output_dir / 'soft_main_intervention_ranking.json'}")
    print(f"Wrote {output_dir / 'soft_main_intervention_ranking.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
