#!/usr/bin/env python3
"""Screen new bridge-candidate cases on a smaller local model.

This script is the practical fallback path when gpt-oss-20b is not runnable in
the current environment. It:

1. collects cases marked ``bridge_candidate`` in the task library
2. runs choice-relative convergence calibration on just those cases
3. stratifies the resulting candidate pool
4. audits the retained soft-main set for local support vs tail rescue
5. writes a compact summary of which candidates survive the screen
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpt_oss_interp.benchmarks.tasks import all_tasks


###############################################################################
#
# Candidate discovery
#
###############################################################################


def _bridge_candidates() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for task in all_tasks():
        for case in task.cases:
            if case.metadata.get("bridge_candidate"):
                rows.append(
                    {
                        "case_id": case.case_id,
                        "task_name": task.name,
                        "behavior": task.behavior,
                    }
                )
    return rows


def _slugify_model_path(model: str) -> str:
    slug = model.rstrip("/").split("/")[-1]
    slug = slug.replace(".", "_").replace("-", "_")
    return slug


###############################################################################
#
# Helpers
#
###############################################################################


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def _write_report(path: Path, payload: dict[str, object]) -> None:
    lines = [
        "# Bridge Candidate Screening",
        "",
        f"Model: `{payload['model']}`",
        f"Candidates screened: `{payload['num_candidates']}`",
        "",
        "## Outcome Counts",
        "",
    ]
    for key, value in sorted(payload["counts"].items()):
        lines.append(f"- `{key}`: {value}")

    lines.extend(
        [
            "",
            "## Accepted Candidates",
            "",
            "| Case | Task | Screening Outcome |",
            "| --- | --- | --- |",
        ]
    )
    for row in payload["accepted_candidates"]:
        lines.append(
            f"| {row['case_id']} | {row['task_name']} | `{row['screening_outcome']}` |"
        )

    lines.extend(
        [
            "",
            "## Rejected / Tagged Candidates",
            "",
            "| Case | Task | Screening Outcome |",
            "| --- | --- | --- |",
        ]
    )
    for row in payload["rejected_or_tagged_candidates"]:
        lines.append(
            f"| {row['case_id']} | {row['task_name']} | `{row['screening_outcome']}` |"
        )

    path.write_text("\n".join(lines) + "\n")


###############################################################################
#
# Main
#
###############################################################################


def main() -> int:
    parser = argparse.ArgumentParser(description="Screen new bridge candidates on a smaller model")
    parser.add_argument("--model", required=True, help="Local model path or HF model id")
    parser.add_argument(
        "--output-root",
        default="runs/bridge_candidate_screen",
        help="Output root directory",
    )
    args = parser.parse_args()

    candidates = _bridge_candidates()
    if not candidates:
        raise RuntimeError("No cases marked bridge_candidate were found")

    model_slug = _slugify_model_path(args.model)
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    out_dir = output_root / model_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_case_ids = [row["case_id"] for row in candidates]
    task_names = sorted({row["task_name"] for row in candidates})
    case_ids_arg = ",".join(candidate_case_ids)
    task_names_arg = ",".join(task_names)

    calibration_dir = out_dir / "convergence_calibration"
    strat_dir = out_dir / "analysis_set_stratification"
    audit_dir = out_dir / "decision_audit"

    _run(
        [
            sys.executable,
            "threads/solid/1-convergence-logit-lens/calibrate_convergence.py",
            "--model",
            args.model,
            "--task-names",
            task_names_arg,
            "--case-ids",
            case_ids_arg,
            "--output",
            str(calibration_dir),
        ]
    )
    _run(
        [
            sys.executable,
            "threads/solid/3-analysis-set-filtering/stratify_analysis_set.py",
            "--input",
            str(calibration_dir / "convergence_calibration.json"),
            "--output",
            str(strat_dir),
            "--tail-length",
            "4",
        ]
    )
    _run(
        [
            sys.executable,
            "scripts/audit_retained_case_decision_decomposition.py",
            "--model",
            args.model,
            "--stratification_json",
            str(strat_dir / "analysis_set_stratification.json"),
            "--set_name",
            "main_analysis_soft",
            "--output",
            str(audit_dir),
        ]
    )

    strat_payload = json.loads((strat_dir / "analysis_set_stratification.json").read_text())
    audit_payload = json.loads((audit_dir / "decision_audit.json").read_text())

    soft_main_ids = set(strat_payload["recommended_sets"]["main_analysis_soft"])
    audit_class = {
        row["case_id"]: row["classification"]
        for row in audit_payload["rows"]
    }

    candidate_rows = []
    by_case_task = {row["case_id"]: row["task_name"] for row in candidates}
    for case_id in candidate_case_ids:
        if case_id not in soft_main_ids:
            outcome = "rejected_pre_audit"
        else:
            classification = audit_class.get(case_id)
            if classification == "local_support":
                outcome = "accepted_local_support"
            elif classification == "tail_rescued":
                outcome = "rejected_tail_rescued"
            else:
                outcome = f"tagged_{classification or 'unknown'}"
        candidate_rows.append(
            {
                "case_id": case_id,
                "task_name": by_case_task[case_id],
                "screening_outcome": outcome,
            }
        )

    counts = Counter(row["screening_outcome"] for row in candidate_rows)
    accepted = [row for row in candidate_rows if row["screening_outcome"] == "accepted_local_support"]
    rejected_or_tagged = [row for row in candidate_rows if row["screening_outcome"] != "accepted_local_support"]

    accepted_by_task: dict[str, list[str]] = defaultdict(list)
    for row in accepted:
        accepted_by_task[row["task_name"]].append(row["case_id"])

    summary = {
        "model": args.model,
        "num_candidates": len(candidate_case_ids),
        "task_names": task_names,
        "counts": dict(sorted(counts.items())),
        "accepted_candidates": accepted,
        "accepted_by_task": {task: ids for task, ids in sorted(accepted_by_task.items())},
        "rejected_or_tagged_candidates": rejected_or_tagged,
        "artifacts": {
            "convergence_calibration": str(calibration_dir / "convergence_calibration.json"),
            "analysis_set_stratification": str(strat_dir / "analysis_set_stratification.json"),
            "decision_audit": str(audit_dir / "decision_audit.json"),
        },
    }

    (out_dir / "screening_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_report(out_dir / "screening_summary.md", summary)
    print(f"Wrote {out_dir / 'screening_summary.json'}", flush=True)
    print(f"Wrote {out_dir / 'screening_summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
