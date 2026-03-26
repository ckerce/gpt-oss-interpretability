"""Report writers for GOSSH benchmark output."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_case_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["run_name"], []).append(row)

    summary: dict[str, Any] = {}
    for run_name, run_rows in grouped.items():
        summary[run_name] = {
            "accuracy": mean(float(r["correct"]) for r in run_rows),
            "mean_margin": mean(float(r["margin"]) for r in run_rows),
            "count": len(run_rows),
            "task_names": sorted({r["task_name"] for r in run_rows}),
        }
    return summary


def write_markdown(path: str | Path, experiment_name: str, summary: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Benchmark Report: {experiment_name}",
        "",
        "| Run | Accuracy | Mean Margin | Cases |",
        "| --- | ---: | ---: | ---: |",
    ]
    for run_name, stats in sorted(summary.items()):
        lines.append(
            f"| {run_name} | {stats['accuracy']:.3f} | {stats['mean_margin']:.3f} | {stats['count']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
