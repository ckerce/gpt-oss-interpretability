"""Phase 2 causal evaluation for promoted per-channel symbolic interventions."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

import torch

from gpt_oss_interp.common.artifacts import RunArtifact
from gpt_oss_interp.common.io import save_json
from gpt_oss_interp.steering.interventions import (
    DEFAULT_BASE_TOKENIZER,
    ChoicePrefix,
    build_tokenizer,
    choice_gap,
    direct_gap_direction_slices,
    local_gap,
    load_symbolic_model,
    output_embedding_by_head,
    parse_model_specs,
    task_case_prefixes,
)
from gpt_oss_interp.steering.readouts import decompose_readout, json_ready as readout_json_ready


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    rx = _average_ranks(xs)
    ry = _average_ranks(ys)
    mean_x = sum(rx) / len(rx)
    mean_y = sum(ry) / len(ry)
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    den_x = math.sqrt(sum((a - mean_x) ** 2 for a in rx))
    den_y = math.sqrt(sum((b - mean_y) ** 2 for b in ry))
    if den_x == 0 or den_y == 0:
        return float("nan")
    return num / (den_x * den_y)


def _family_prefixes(task_names: set[str], tokenizer, family: str) -> list[ChoicePrefix]:
    prefixes, invalid = task_case_prefixes(task_names, tokenizer, skip_invalid=True)
    kept = [prefix for prefix in prefixes if prefix.task_name == family]
    if not kept:
        raise ValueError(f"No valid cases found for family={family}; invalid={invalid}")
    return kept


def _baseline_gaps(model, prefixes: list[ChoicePrefix], device: torch.device) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for prefix in prefixes:
        result[prefix.case_id] = {
            "local_gap": local_gap(
                model,
                prefix,
                device,
                readout_source="combined",
            ),
            "total_gap": choice_gap(
                model,
                prefix,
                device,
                readout_source="combined",
            ),
        }
    return result


def _evaluate_slice_row(
    model,
    prefixes: list[ChoicePrefix],
    device: torch.device,
    vocab_by_head: torch.Tensor,
    *,
    layer: int,
    head: int,
    scales: list[float],
    stream: str = "x_t",
    stage: str = "post_block",
) -> dict[str, object]:
    case_rows = []
    mean_by_scale = []
    for scale in scales:
        directed_local_effects = []
        directed_total_effects = []
        tail_fractions = []
        for prefix in prefixes:
            direction = direct_gap_direction_slices(prefix, vocab_by_head)
            vector_slice = direction[head] * scale
            full_vector = direction.reshape(-1) * scale
            baseline_local = local_gap(model, prefix, device, readout_source="combined")
            baseline_total = choice_gap(model, prefix, device, readout_source="combined")

            local = local_gap(
                model,
                prefix,
                device,
                layer_idx=layer,
                head_idx=head,
                vector_slice=vector_slice,
                stream=stream,
                stage=stage,
                readout_source="combined",
            )
            total = choice_gap(
                model,
                prefix,
                device,
                layer_idx=layer,
                head_idx=head,
                vector_slice=vector_slice,
                stream=stream,
                stage=stage,
                readout_source="combined",
            )
            whole_total = choice_gap(
                model,
                prefix,
                device,
                layer_idx=layer,
                vector=full_vector,
                stream=stream,
                stage=stage,
                readout_source="combined",
            )
            local_shift = local - baseline_local
            total_shift = total - baseline_total
            tail_fraction = abs(total_shift - local_shift) / max(abs(total_shift), 1e-8)
            directed_local = math.copysign(abs(local_shift), scale * local_shift)
            directed_total = math.copysign(abs(total_shift), scale * total_shift)
            directed_local_effects.append(directed_local)
            directed_total_effects.append(directed_total)
            tail_fractions.append(tail_fraction)
            case_rows.append(
                {
                    "case_id": prefix.case_id,
                    "layer": layer,
                    "head": head,
                    "scale": scale,
                    "local_shift": local_shift,
                    "total_shift": total_shift,
                    "tail_fraction": tail_fraction,
                    "whole_vector_total_gap": whole_total,
                }
            )
        mean_by_scale.append(
            {
                "scale": scale,
                "mean_directed_local_effect": sum(directed_local_effects) / len(directed_local_effects),
                "mean_directed_total_effect": sum(directed_total_effects) / len(directed_total_effects),
                "mean_tail_fraction": sum(tail_fractions) / len(tail_fractions),
            }
        )
    best_scale_row = max(mean_by_scale, key=lambda row: row["mean_directed_local_effect"])
    return {
        "layer": layer,
        "head": head,
        "best_scale": best_scale_row["scale"],
        "best_mean_directed_local_effect": best_scale_row["mean_directed_local_effect"],
        "best_mean_directed_total_effect": best_scale_row["mean_directed_total_effect"],
        "best_mean_tail_fraction": best_scale_row["mean_tail_fraction"],
        "scale_summaries": mean_by_scale,
        "case_rows": [row for row in case_rows if row["scale"] == best_scale_row["scale"]],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase 2 per-channel causal evaluation")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "E2=/mnt/d/data/neuro-symb-v2/backup-data/experiments/"
            "cog-sys-paper-1/series_e_25K_test/E2_independent/checkpoint_epoch_2.pt"
        ],
    )
    parser.add_argument("--family", default="recency_bias")
    parser.add_argument("--task-names", default="coreference,induction,recency_bias")
    parser.add_argument("--tokenizer-mode", default="reduced_gpt2", choices=("auto", "gpt2", "reduced_gpt2"))
    parser.add_argument("--base-tokenizer", default=DEFAULT_BASE_TOKENIZER)
    parser.add_argument("--vocab-file", default="companion-repo/neuro-symb-v2/neurips-2026/data/grade_school_vocab.pkl")
    parser.add_argument("--probe-artifact", default="runs/channel_probe_e2_phase1/e2_channel_probe.json")
    parser.add_argument("--scales", default="-8,-4,-2,-1,1,2,4,8")
    parser.add_argument("--stream", default="x_t", choices=("x_t", "x_e"))
    parser.add_argument("--stage", default="post_block", choices=("post_block", "pre_block"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--output", default="runs/per_channel_causal_e2_recency")
    args = parser.parse_args(argv)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    scales = [float(item.strip()) for item in args.scales.split(",") if item.strip()]
    device = torch.device(args.device)
    task_names = {name.strip() for name in args.task_names.split(",") if name.strip()}
    probe_payload = json.loads(Path(args.probe_artifact).read_text())
    family_rows = probe_payload["ranking"][args.family]["rows"]

    for model_label, checkpoint_path in parse_model_specs(args.models):
        model, checkpoint = load_symbolic_model(checkpoint_path, device=device)
        tokenizer = build_tokenizer(
            tokenizer_mode=args.tokenizer_mode,
            base_tokenizer_path=args.base_tokenizer,
            vocab_file=args.vocab_file,
            checkpoint_config=checkpoint["config"],
        )
        prefixes = _family_prefixes(task_names, tokenizer, args.family)
        vocab_by_head = output_embedding_by_head(model)

        causal_rows = []
        for row in family_rows:
            causal = _evaluate_slice_row(
                model,
                prefixes,
                device,
                vocab_by_head,
                layer=int(row["layer"]),
                head=int(row["head"]),
                scales=scales,
                stream=args.stream,
                stage=args.stage,
            )
            causal_rows.append(
                {
                    "family": args.family,
                    "layer": int(row["layer"]),
                    "head": int(row["head"]),
                    "probe_heldout_sign_accuracy": float(row["heldout_sign_accuracy"]),
                    "probe_mean_score": float(row["mean_score"]),
                    "probe_passes_gate": bool(row["passes_gate"]),
                    **causal,
                }
            )

        causal_rows.sort(key=lambda row: row["best_mean_directed_local_effect"], reverse=True)
        for idx, row in enumerate(causal_rows):
            row["causal_rank"] = idx + 1

        probe_rank_lookup = {
            (int(row["layer"]), int(row["head"])): idx + 1
            for idx, row in enumerate(family_rows)
        }
        probe_ranks = [probe_rank_lookup[(row["layer"], row["head"])] for row in causal_rows]
        causal_ranks = [row["causal_rank"] for row in causal_rows]
        heldout_accs = [float(row["probe_heldout_sign_accuracy"]) for row in causal_rows]
        best_effects = [float(row["best_mean_directed_local_effect"]) for row in causal_rows]
        rank_corr = _spearman(probe_ranks, causal_ranks)
        acc_corr = _spearman(heldout_accs, best_effects)

        promoted = [row for row in causal_rows if row["probe_passes_gate"]]
        low_ranked = causal_rows[-max(1, len(promoted)) :]
        rng = random.Random(args.random_seed)
        random_controls = rng.sample(causal_rows, min(len(promoted), len(causal_rows)))

        top_readouts = []
        for row in promoted[: min(3, len(promoted))]:
            best_case = max(row["case_rows"], key=lambda item: abs(float(item["local_shift"])))
            prefix = next(prefix for prefix in prefixes if prefix.case_id == best_case["case_id"])
            direction = direct_gap_direction_slices(prefix, vocab_by_head)
            vector_slice = direction[row["head"]] * float(row["best_scale"])
            decomp = decompose_readout(
                model,
                prefix,
                device,
                layer_idx=row["layer"],
                head_idx=row["head"],
                vector_slice=vector_slice,
                stream=args.stream,
                stage=args.stage,
            )
            top_readouts.append(
                {
                    "layer": row["layer"],
                    "head": row["head"],
                    "case_id": prefix.case_id,
                    **readout_json_ready(decomp),
                }
            )

        artifact = RunArtifact(
            run_id=output_dir.name,
            model_id=model_label,
            phase="steering",
            results=causal_rows,
            metadata={
                "phase_name": "per_channel_causal_phase2",
                "family": args.family,
                "probe_artifact": args.probe_artifact,
                "stream": args.stream,
                "stage": args.stage,
                "scales": scales,
                "rank_correlation_probe_vs_causal": rank_corr,
                "heldout_accuracy_vs_effect_correlation": acc_corr,
                "n_promoted": len(promoted),
                "promoted_mean_effect": sum(row["best_mean_directed_local_effect"] for row in promoted) / max(len(promoted), 1),
                "low_rank_mean_effect": sum(row["best_mean_directed_local_effect"] for row in low_ranked) / max(len(low_ranked), 1),
                "random_control_mean_effect": sum(row["best_mean_directed_local_effect"] for row in random_controls) / max(len(random_controls), 1),
                "top_readout_decompositions": top_readouts,
            },
        )
        payload = {"artifact": asdict(artifact)}
        save_json(payload, output_dir / f"{model_label.lower()}_{args.family}_causal.json")

        lines = [
            "# Per-Channel Causal Report",
            "",
            f"Model: `{model_label}`",
            f"Family: `{args.family}`",
            "",
            "## Summary",
            "",
            f"- probe vs causal rank Spearman: `{rank_corr:.3f}`",
            f"- held-out accuracy vs causal effect Spearman: `{acc_corr:.3f}`",
            f"- promoted channels: `{len(promoted)}`",
            f"- promoted mean effect: `{artifact.metadata['promoted_mean_effect']:.3f}`",
            f"- low-rank mean effect: `{artifact.metadata['low_rank_mean_effect']:.3f}`",
            f"- random-control mean effect: `{artifact.metadata['random_control_mean_effect']:.3f}`",
            "",
            "## Top Channels",
            "",
        ]
        for row in causal_rows[:10]:
            lines.append(
                f"- L{row['layer']} H{row['head']}: effect={row['best_mean_directed_local_effect']:.3f}, "
                f"best_scale={row['best_scale']:+.1f}, probe_acc={row['probe_heldout_sign_accuracy']:.3f}, "
                f"promoted={row['probe_passes_gate']}"
            )
        lines.extend(["", "## Readout Decomposition", ""])
        for item in top_readouts:
            lines.append(
                f"- L{item['layer']} H{item['head']} on `{item['case_id']}`: "
                f"combined={item['combined_effect']:+.3f}, "
                f"x_t={item['xt_effect']:+.3f}, x_e={item['xe_effect']:+.3f}, "
                f"transfer={item['transfer_ratio']:.3f}"
            )
        (output_dir / f"{model_label.lower()}_{args.family}_report.md").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
