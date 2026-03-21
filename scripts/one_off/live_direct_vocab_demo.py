#!/usr/bin/env python3
"""Small live CLI for the curated direct-vocabulary steering examples."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch


RUNNER_PATH = Path(__file__).resolve().parent.parent.parent / "threads" / "solid" / "6-direct-vocab-steering" / "run_direct_vocab_steering.py"
BASE_TOKENIZER = (
    "hf-cache/models--gpt2/snapshots/"
    "607a30d783dfa663caf39e06633721c8d4cfcd7e"
)

MODEL_PATHS = {
    "SS-71": (
        "/mnt/d/mechanistic_interpretability/results/neurips-2026/training/"
        "gated_attention/dns-dns_S_G_gpt2/checkpoint_epoch_1.pt"
    ),
    "C-71": (
        "/mnt/d/mechanistic_interpretability/results/neurips-2026/training/"
        "gated_attention/dns-dns_S_G_cascade_gpt2/checkpoint_epoch_1.pt"
    ),
}

PRESETS = {
    "coref_009": {
        "SS-71": {"positive": (5, 8.0), "negative": (5, -8.0)},
        "C-71": {"positive": (5, 8.0), "negative": (5, -8.0)},
    },
    "induction_005": {
        "SS-71": {"positive": (4, 8.0), "negative": (5, -8.0)},
        "C-71": {"positive": (4, 8.0), "negative": (4, -8.0)},
    },
    "recency_001": {
        "SS-71": {"positive": (3, 8.0), "negative": (3, -8.0)},
        "C-71": {"positive": (3, 8.0), "negative": (2, -8.0)},
    },
    "coref_006": {
        "SS-71": {"positive": (5, 8.0), "negative": (5, -8.0)},
        "C-71": {"positive": (5, 8.0), "negative": (5, -8.0)},
    },
}


def _load_runner():
    spec = importlib.util.spec_from_file_location("direct_vocab_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load runner module from {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _predicted_label(choice_scores: dict[str, float]) -> str:
    return "A" if float(choice_scores["A"]) >= float(choice_scores["B"]) else "B"


def _row_summary(row: dict[str, object]) -> str:
    pred = _predicted_label(row["intervened_choice_scores"])
    return (
        f"layer={row['layer']} scale={row['scale']:+.1f} "
        f"pred={pred} local_shift={row['local_logit_shift']:+.3f} "
        f"total_gap={row['intervened_total_gap']:+.3f} "
        f"tail_fraction={row['tail_fraction']:.3f}"
    )


def _find_row(rows: list[dict[str, object]], layer: int, scale: float) -> dict[str, object]:
    for row in rows:
        if int(row["layer"]) == layer and float(row["scale"]) == float(scale):
            return row
    raise KeyError(f"Missing row for layer={layer} scale={scale}")


def _print_case(mod, model, tokenizer, case_id: str, model_label: str, device: torch.device) -> None:
    task, case = mod._find_case(case_id)
    prefix = mod._choice_prefix(
        tokenizer,
        case.case_id,
        case.prompt,
        case.choices["A"],
        case.choices["B"],
    )
    baseline = mod._baseline_case_summary(
        model=model,
        prefix=prefix,
        expected_label=str(case.expected_label),
        tokenizer=tokenizer,
        device=device,
    )
    preset = PRESETS[case_id][model_label]
    layers = sorted({preset["positive"][0], preset["negative"][0]})
    scales = sorted({preset["positive"][1], preset["negative"][1]})
    rows = mod._steering_rows_for_pair(
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
    positive = _find_row(rows, *preset["positive"])
    negative = _find_row(rows, *preset["negative"])

    print(f"\n== {model_label} :: {case_id} ==")
    print(case.prompt)
    print(f"A: {case.choices['A']}  |  B: {case.choices['B']}")
    print(
        f"direction = W[{prefix.choice_a_token_text if hasattr(prefix, 'choice_a_token_text') else tokenizer.decode([prefix.choice_a_token])!r}] "
        f"- W[{tokenizer.decode([prefix.choice_b_token])!r}]"
    )
    print(
        f"baseline: pred={baseline['predicted_label']} "
        f"local_gap={baseline['local']['local_logit_gap']:+.3f} "
        f"total_gap={baseline['total_gap']:+.3f}"
    )
    print(f"positive: {_row_summary(positive)}")
    print(f"negative: {_row_summary(negative)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Live direct-vocabulary steering demo")
    parser.add_argument("--model", choices=sorted(MODEL_PATHS), default="SS-71")
    parser.add_argument(
        "--case-id",
        choices=sorted(PRESETS),
        default="coref_009",
        help="Curated example to run",
    )
    parser.add_argument("--all", action="store_true", help="Run all curated examples for the selected model")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    mod = _load_runner()
    device = torch.device(args.device)
    model, checkpoint = mod._load_model(MODEL_PATHS[args.model], device=device)
    tokenizer = mod._build_tokenizer("gpt2", "", BASE_TOKENIZER, checkpoint["config"])

    case_ids = list(PRESETS) if args.all else [args.case_id]
    for case_id in case_ids:
        _print_case(mod, model, tokenizer, case_id, args.model, device)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
