#!/usr/bin/env python3
"""Measure attention-pattern sensitivity under x_t vs x_e direct-vocab interventions."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER_PATH = REPO_ROOT / "scripts" / "run_direct_vocab_steering.py"
MODEL_PATH = (
    "/mnt/d/mechanistic_interpretability/results/neurips-2026/training/"
    "gated_attention/dns-dns_S_G_cascade_gpt2/checkpoint_epoch_1.pt"
)
BASE_TOKENIZER = (
    "hf-cache/models--gpt2/snapshots/"
    "607a30d783dfa663caf39e06633721c8d4cfcd7e"
)
CASE_IDS = ("recency_001", "induction_005", "coref_009")
SCALE = -8.0
INTERVENTION_STAGE = "embedding_init"


def _load_runner():
    spec = importlib.util.spec_from_file_location("direct_vocab_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load runner module from {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _capture_attention(mod, model, prefix, device, *, intervention_stream: str | None) -> list[torch.Tensor]:
    for block in model.blocks:
        block.attn._store_attn = True
        block.attn.last_attn = None

    weight = model.lm_head.weight.detach().cpu().float()
    vector = None
    if intervention_stream is not None:
        vector = weight[prefix.choice_a_token] - weight[prefix.choice_b_token]
        vector = vector * SCALE

    mod._forward_logits(
        model,
        prefix.prefix_ids,
        device=device,
        layer_idx=0,
        position_idx=prefix.decision_position_index,
        vector=vector,
        intervention_stream=intervention_stream or "x_e",
        intervention_stage=INTERVENTION_STAGE,
        readout_source="combined",
    )

    snapshots: list[torch.Tensor] = []
    for block in model.blocks:
        attn = block.attn.last_attn
        if attn is None:
            raise RuntimeError("Attention capture failed; last_attn is None")
        snapshots.append(attn.clone())
        block.attn._store_attn = False
        block.attn.last_attn = None
    return snapshots


def _decision_query_delta(baseline: torch.Tensor, intervened: torch.Tensor) -> float:
    # last_attn shape: (B, H, T_q, T_k). Evaluate the final query position only.
    base_slice = baseline[0, :, -1, :]
    int_slice = intervened[0, :, -1, :]
    return float((int_slice - base_slice).abs().mean().item())


def _write_report(output_dir: Path, payload: dict[str, object]) -> None:
    lines = [
        "# Attention Path Sensitivity Report",
        "",
        "This analysis measures how much the attention pattern changes under embedding-level direct-vocabulary interventions in `C-71`.",
        "",
        f"Intervention stage: `{INTERVENTION_STAGE}`",
        f"Scale: `{SCALE}`",
        "",
        "We report mean absolute change in the attention distribution for the final query position at each layer.",
        "",
    ]

    for case in payload["cases"]:
        lines.append(f"## {case['case_id']}")
        lines.append("")
        lines.append(f"Prompt: `{case['prompt']}`")
        lines.append("")
        lines.append(f"Direction: `W[{case['choice_a_token_text']}] - W[{case['choice_b_token_text']}]`")
        lines.append("")
        lines.append("Layerwise decision-query attention change:")
        lines.append("")
        for layer in case["layers"]:
            lines.append(
                f"- layer {layer['layer']}: "
                f"`x_t` delta={layer['x_t_delta']:.4f}, "
                f"`x_e` delta={layer['x_e_delta']:.4f}"
            )
        lines.append("")
        lines.append(
            f"Summary: mean layer delta `x_t`={case['mean_x_t_delta']:.4f}, "
            f"`x_e`={case['mean_x_e_delta']:.4f}"
        )
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "- Larger values mean the intervention changed the attention routing more strongly at the answer query.",
            "- This does not by itself prove stronger causal influence, but it does show whether the two streams propagate differently into attention.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    mod = _load_runner()
    device = torch.device("cpu")
    model, checkpoint = mod._load_model(MODEL_PATH, device=device)
    tokenizer = mod._build_tokenizer("gpt2", "", BASE_TOKENIZER, checkpoint["config"])

    output_dir = Path("runs/attention_path_sensitivity")
    output_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, object] = {
        "model": "C-71",
        "checkpoint": MODEL_PATH,
        "intervention_stage": INTERVENTION_STAGE,
        "scale": SCALE,
        "cases": [],
    }

    for case_id in CASE_IDS:
        task, case = mod._find_case(case_id)
        prefix = mod._choice_prefix(
            tokenizer,
            case.case_id,
            case.prompt,
            case.choices["A"],
            case.choices["B"],
        )
        baseline = _capture_attention(mod, model, prefix, device, intervention_stream=None)
        xt = _capture_attention(mod, model, prefix, device, intervention_stream="x_t")
        xe = _capture_attention(mod, model, prefix, device, intervention_stream="x_e")

        layer_rows = []
        for layer_idx, (base, xt_attn, xe_attn) in enumerate(zip(baseline, xt, xe)):
            layer_rows.append(
                {
                    "layer": layer_idx,
                    "x_t_delta": _decision_query_delta(base, xt_attn),
                    "x_e_delta": _decision_query_delta(base, xe_attn),
                }
            )

        payload["cases"].append(
            {
                "case_id": case.case_id,
                "task": task.name,
                "prompt": case.prompt,
                "choice_a_token_text": tokenizer.decode([prefix.choice_a_token]),
                "choice_b_token_text": tokenizer.decode([prefix.choice_b_token]),
                "layers": layer_rows,
                "mean_x_t_delta": sum(row["x_t_delta"] for row in layer_rows) / len(layer_rows),
                "mean_x_e_delta": sum(row["x_e_delta"] for row in layer_rows) / len(layer_rows),
            }
        )

    (output_dir / "attention_path_sensitivity.json").write_text(json.dumps(payload, indent=2))
    _write_report(output_dir, payload)
    print(f"[attention-sensitivity] wrote {output_dir / 'attention_path_sensitivity.json'}", flush=True)
    print(f"[attention-sensitivity] wrote {output_dir / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
