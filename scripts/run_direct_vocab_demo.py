#!/usr/bin/env python3
"""Curated direct-vocabulary steering demo for matched 71M symbolic models."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER_PATH = REPO_ROOT / "scripts" / "run_direct_vocab_steering.py"

DEFAULT_MODELS = (
    "SS-71=/mnt/d/mechanistic_interpretability/results/neurips-2026/training/"
    "gated_attention/dns-dns_S_G_gpt2/checkpoint_epoch_1.pt",
    "C-71=/mnt/d/mechanistic_interpretability/results/neurips-2026/training/"
    "gated_attention/dns-dns_S_G_cascade_gpt2/checkpoint_epoch_1.pt",
)
DEFAULT_CASE_IDS = (
    "coref_009",
    "induction_005",
    "recency_001",
    "coref_006",
)
DEFAULT_LAYERS = (0, 1, 2, 3, 4, 5)
DEFAULT_SCALES = (-8.0, -6.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 6.0, 8.0)
DEFAULT_BASE_TOKENIZER = (
    "/home/ckerce/.cache/huggingface/hub/models--gpt2/snapshots/"
    "607a30d783dfa663caf39e06633721c8d4cfcd7e"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location("direct_vocab_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load runner module from {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _parse_csv(values: str, cast):
    return [cast(item.strip()) for item in values.split(",") if item.strip()]


def _parse_models(entries: list[str]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Model entry must be label=path, got: {entry}")
        label, path = entry.split("=", 1)
        parsed.append((label.strip(), path.strip()))
    return parsed


def _predicted_label(choice_scores: dict[str, float]) -> str:
    return "A" if float(choice_scores["A"]) >= float(choice_scores["B"]) else "B"


def _best_rows(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    positive = [row for row in rows if float(row["scale"]) > 0]
    negative = [row for row in rows if float(row["scale"]) < 0]
    return {
        "best_positive": max(positive, key=lambda row: float(row["local_logit_shift"])),
        "best_negative": min(negative, key=lambda row: float(row["local_logit_shift"])),
        "strongest_abs": max(rows, key=lambda row: abs(float(row["local_logit_shift"]))),
    }


def _flip_row(rows: list[dict[str, object]], target_label: str) -> dict[str, object] | None:
    flipped = [
        row
        for row in rows
        if _predicted_label(row["intervened_choice_scores"]) == target_label
    ]
    if not flipped:
        return None
    return max(flipped, key=lambda row: abs(float(row["total_shift"])))


def _render_row(row: dict[str, object], baseline_total_gap: float) -> str:
    choice_scores = row["intervened_choice_scores"]
    pred = _predicted_label(choice_scores)
    return (
        f"layer {row['layer']}, scale {row['scale']:+.1f}, "
        f"local shift {row['local_logit_shift']:+.3f}, "
        f"total gap {row['intervened_total_gap']:+.3f} "
        f"(baseline {baseline_total_gap:+.3f}), pred={pred}, "
        f"tail_fraction={row['tail_fraction']:.3f}"
    )


def _write_report(output_dir: Path, payload: dict[str, object]) -> None:
    lines = [
        "# Direct Vocabulary Steering Demo",
        "",
        f"Output root: `{output_dir}`",
        "",
        "This demo uses exact vocabulary directions `W[token_A] - W[token_B]` on matched 71M symbolic models.",
        "",
        "## Recommended Examples",
        "",
    ]

    recommended = payload["recommended_examples"]
    for example in recommended:
        lines.append(
            f"- `{example['case_id']}` on `{example['model']}`: "
            f"{example['choice_a']} vs {example['choice_b']}; "
            f"baseline `{example['baseline_prediction']}` -> negative-steered `{example['negative_prediction']}`"
        )
    lines.append("")

    for model_result in payload["models"]:
        lines.append(f"## {model_result['label']}")
        lines.append("")
        for case in model_result["cases"]:
            lines.append(f"### {case['case_id']}")
            lines.append("")
            lines.append(f"Prompt: `{case['prompt']}`")
            lines.append("")
            lines.append(
                f"Choices: `A={case['choice_a']}` vs `B={case['choice_b']}`; "
                f"divergent tokens `{case['token_a']}` vs `{case['token_b']}`; "
                f"clean={case['clean_tokenization']}"
            )
            lines.append("")
            lines.append(
                f"Baseline: pred={case['baseline']['predicted_label']} "
                f"(expected {case['baseline']['expected_label']}), "
                f"local_gap={case['baseline']['local_gap']:+.3f}, "
                f"total_gap={case['baseline']['total_gap']:+.3f}"
            )
            lines.append("")
            lines.append(f"- Best positive: {_render_row(case['best_positive'], case['baseline']['total_gap'])}")
            lines.append(f"- Best negative: {_render_row(case['best_negative'], case['baseline']['total_gap'])}")
            if case["flip_to_b"] is not None:
                lines.append(f"- Flip to B: {_render_row(case['flip_to_b'], case['baseline']['total_gap'])}")
            else:
                lines.append("- Flip to B: none in tested layer/scale sweep")
            lines.append("")

    lines.extend(
        [
            "## Notes",
            "",
            "- The best live demo cases are the ones that are tokenizer-clean and baseline-correct on both matched models.",
            "- This demo uses a wider steering sweep than the memo figures because visible choice flips require stronger intervention than simple margin shifts.",
            "- Capitalization was not used here because most proper-noun examples split into dirty sub-token contrasts under GPT-2 tokenization.",
            "- Syntax agreement and most recency cases are weak in these training runs, so they are less reliable as demos.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a curated direct-vocabulary steering demo")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--case-ids", default=",".join(DEFAULT_CASE_IDS))
    parser.add_argument("--layers", default=",".join(str(x) for x in DEFAULT_LAYERS))
    parser.add_argument("--scales", default=",".join(str(x) for x in DEFAULT_SCALES))
    parser.add_argument("--base-tokenizer", default=DEFAULT_BASE_TOKENIZER)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="runs/direct_vocab_demo_matched_71m")
    args = parser.parse_args()

    runner = _load_runner()
    device = torch.device(args.device)
    model_specs = _parse_models(args.models)
    case_ids = _parse_csv(args.case_ids, str)
    layers = _parse_csv(args.layers, int)
    scales = _parse_csv(args.scales, float)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, object] = {
        "config": {
            "models": args.models,
            "case_ids": case_ids,
            "layers": layers,
            "scales": scales,
            "base_tokenizer": args.base_tokenizer,
            "device": args.device,
        },
        "models": [],
        "recommended_examples": [],
    }

    for model_label, checkpoint_path in model_specs:
        print(f"[direct-vocab-demo] loading {model_label} from {checkpoint_path}", flush=True)
        model, checkpoint = runner._load_model(checkpoint_path, device)
        tokenizer = runner._build_tokenizer("gpt2", "", args.base_tokenizer, checkpoint["config"])
        model_result = {
            "label": model_label,
            "checkpoint": checkpoint_path,
            "cases": [],
        }

        for case_id in case_ids:
            task, case = runner._find_case(case_id)
            prefix = runner._choice_prefix(
                tokenizer,
                case.case_id,
                case.prompt,
                case.choices["A"],
                case.choices["B"],
            )
            baseline = runner._baseline_case_summary(
                model=model,
                prefix=prefix,
                expected_label=str(case.expected_label),
                tokenizer=tokenizer,
                device=device,
            )
            rows = runner._steering_rows_for_pair(
                model=model,
                tokenizer=tokenizer,
                model_label=model_label,
                source_prefix=prefix,
                target_prefix=prefix,
                baseline_target=baseline,
                layers=layers,
                scales=scales,
                device=device,
            )
            best = _best_rows(rows)
            flip_to_b = _flip_row(rows, "B")
            comp_a = prefix.full_a[len(prefix.prompt_ids):]
            comp_b = prefix.full_b[len(prefix.prompt_ids):]

            case_result = {
                "case_id": case.case_id,
                "task": task.name,
                "prompt": case.prompt,
                "choice_a": case.choices["A"],
                "choice_b": case.choices["B"],
                "token_a": tokenizer.decode([prefix.choice_a_token]),
                "token_b": tokenizer.decode([prefix.choice_b_token]),
                "clean_tokenization": len(comp_a) == 1 and len(comp_b) == 1,
                "baseline": {
                    "predicted_label": baseline["predicted_label"],
                    "expected_label": baseline["expected_label"],
                    "correct": baseline["correct"],
                    "local_gap": baseline["local"]["local_logit_gap"],
                    "total_gap": baseline["total_gap"],
                },
                "best_positive": {
                    **best["best_positive"],
                    "predicted_label": _predicted_label(best["best_positive"]["intervened_choice_scores"]),
                },
                "best_negative": {
                    **best["best_negative"],
                    "predicted_label": _predicted_label(best["best_negative"]["intervened_choice_scores"]),
                },
                "flip_to_b": (
                    {
                        **flip_to_b,
                        "predicted_label": _predicted_label(flip_to_b["intervened_choice_scores"]),
                    }
                    if flip_to_b is not None
                    else None
                ),
            }
            model_result["cases"].append(case_result)

            if case_result["clean_tokenization"] and baseline["correct"] and flip_to_b is not None:
                payload["recommended_examples"].append(
                    {
                        "model": model_label,
                        "case_id": case.case_id,
                        "choice_a": case.choices["A"],
                        "choice_b": case.choices["B"],
                        "baseline_prediction": baseline["predicted_label"],
                        "negative_prediction": "B",
                        "flip_layer": flip_to_b["layer"],
                        "flip_scale": flip_to_b["scale"],
                        "flip_total_gap": flip_to_b["intervened_total_gap"],
                    }
                )

        payload["models"].append(model_result)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload["recommended_examples"] = sorted(
        payload["recommended_examples"],
        key=lambda item: abs(float(item["flip_total_gap"])),
        reverse=True,
    )
    (output_dir / "direct_vocab_demo.json").write_text(json.dumps(payload, indent=2))
    _write_report(output_dir, payload)
    print(f"[direct-vocab-demo] wrote {output_dir / 'direct_vocab_demo.json'}", flush=True)
    print(f"[direct-vocab-demo] wrote {output_dir / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
