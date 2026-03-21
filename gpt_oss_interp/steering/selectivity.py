"""Selectivity comparison for channelized vs whole-vector symbolic writes."""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import asdict
from pathlib import Path

import torch

from gpt_oss_interp.common.artifacts import RunArtifact
from gpt_oss_interp.common.io import save_json
from gpt_oss_interp.steering.controls import make_generator, random_direction_slices_like
from gpt_oss_interp.steering.interventions import (
    DEFAULT_BASE_TOKENIZER,
    ChoicePrefix,
    build_tokenizer,
    choice_gap,
    direct_gap_direction_slices,
    forward_logits,
    load_symbolic_model,
    output_embedding_by_head,
    parse_csv,
    parse_model_specs,
    task_case_prefixes,
)
from gpt_oss_interp.steering.readouts import decompose_readout, json_ready as readout_json_ready

EPS = 1e-8


def _norm(tensor: torch.Tensor) -> float:
    return float(torch.norm(tensor).item())


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.reshape(-1).float()
    b_flat = b.reshape(-1).float()
    denom = torch.norm(a_flat) * torch.norm(b_flat)
    if float(denom.item()) <= EPS:
        return float("nan")
    return float(torch.dot(a_flat, b_flat).item() / denom.item())


def _local_gap_from_logits(prefix: ChoicePrefix, logits: torch.Tensor) -> float:
    decision_logits = logits[prefix.decision_position_index]
    return float((decision_logits[prefix.choice_a_token] - decision_logits[prefix.choice_b_token]).item())


def _non_target_kl(prefix: ChoicePrefix, baseline_logits: torch.Tensor, intervened_logits: torch.Tensor) -> float:
    decision_idx = prefix.decision_position_index
    keep = torch.ones(baseline_logits.shape[-1], dtype=torch.bool)
    keep[prefix.choice_a_token] = False
    keep[prefix.choice_b_token] = False
    base = baseline_logits[decision_idx, keep]
    intr = intervened_logits[decision_idx, keep]
    base_log_probs = torch.log_softmax(base, dim=-1)
    intr_log_probs = torch.log_softmax(intr, dim=-1)
    intr_probs = torch.exp(intr_log_probs)
    return float(torch.sum(intr_probs * (intr_log_probs - base_log_probs)).item())


def _tail_fraction(local_shift: float, total_shift: float) -> float:
    return abs(total_shift - local_shift) / max(abs(total_shift), EPS)


def _intervened_logits(
    model,
    prefix: ChoicePrefix,
    device: torch.device,
    *,
    layer_idx: int,
    scale: float,
    direction_slices: torch.Tensor,
    condition: str,
    head_idx: int | None = None,
    control_head_idx: int | None = None,
    random_direction_slices: torch.Tensor | None = None,
    stream: str = "x_t",
    stage: str = "post_block",
) -> tuple[torch.Tensor, float, int | None]:
    if condition == "channelized":
        if head_idx is None:
            raise ValueError("channelized condition requires head_idx")
        vector_slice = direction_slices[head_idx] * scale
        logits = forward_logits(
            model,
            prefix.prefix_ids,
            device,
            layer_idx=layer_idx,
            position_idx=prefix.decision_position_index,
            head_idx=head_idx,
            vector_slice=vector_slice,
            stream=stream,
            stage=stage,
            readout_source="combined",
        )
        return logits, _norm(vector_slice), head_idx

    if condition == "whole_vector":
        vector = direction_slices.reshape(-1) * scale
        logits = forward_logits(
            model,
            prefix.prefix_ids,
            device,
            layer_idx=layer_idx,
            position_idx=prefix.decision_position_index,
            vector=vector,
            stream=stream,
            stage=stage,
            readout_source="combined",
        )
        return logits, _norm(vector), None

    if condition == "random_channel":
        if head_idx is None or control_head_idx is None:
            raise ValueError("random_channel requires head_idx and control_head_idx")
        reference_norm = torch.norm(direction_slices[head_idx]).clamp_min(EPS)
        control_slice = direction_slices[control_head_idx]
        control_norm = torch.norm(control_slice).clamp_min(EPS)
        vector_slice = control_slice * (reference_norm / control_norm) * scale
        logits = forward_logits(
            model,
            prefix.prefix_ids,
            device,
            layer_idx=layer_idx,
            position_idx=prefix.decision_position_index,
            head_idx=control_head_idx,
            vector_slice=vector_slice,
            stream=stream,
            stage=stage,
            readout_source="combined",
        )
        return logits, _norm(vector_slice), control_head_idx

    if condition == "random_direction":
        if head_idx is None or random_direction_slices is None:
            raise ValueError("random_direction requires head_idx and random_direction_slices")
        vector_slice = random_direction_slices[head_idx] * scale
        logits = forward_logits(
            model,
            prefix.prefix_ids,
            device,
            layer_idx=layer_idx,
            position_idx=prefix.decision_position_index,
            head_idx=head_idx,
            vector_slice=vector_slice,
            stream=stream,
            stage=stage,
            readout_source="combined",
        )
        return logits, _norm(vector_slice), head_idx

    raise ValueError(f"Unsupported condition: {condition}")


def _target_case_metrics(
    model,
    prefix: ChoicePrefix,
    device: torch.device,
    *,
    baseline_local: float,
    baseline_total: float,
    baseline_logits: torch.Tensor,
    layer_idx: int,
    scale: float,
    direction_slices: torch.Tensor,
    condition: str,
    head_idx: int | None = None,
    control_head_idx: int | None = None,
    random_direction_slices: torch.Tensor | None = None,
    stream: str = "x_t",
    stage: str = "post_block",
) -> dict[str, float | int | None]:
    intervened_logits, raw_norm, actual_head = _intervened_logits(
        model,
        prefix,
        device,
        layer_idx=layer_idx,
        scale=scale,
        direction_slices=direction_slices,
        condition=condition,
        head_idx=head_idx,
        control_head_idx=control_head_idx,
        random_direction_slices=random_direction_slices,
        stream=stream,
        stage=stage,
    )
    local = _local_gap_from_logits(prefix, intervened_logits)
    total = choice_gap(
        model,
        prefix,
        device,
        layer_idx=layer_idx,
        head_idx=actual_head if condition != "whole_vector" else None,
        vector_slice=(direction_slices[actual_head] * scale) if condition == "channelized" and actual_head is not None else (
            random_direction_slices[head_idx] * scale if condition == "random_direction" and head_idx is not None and random_direction_slices is not None else (
                direction_slices[control_head_idx] * (
                    torch.norm(direction_slices[head_idx]).clamp_min(EPS)
                    / torch.norm(direction_slices[control_head_idx]).clamp_min(EPS)
                ) * scale if condition == "random_channel" and head_idx is not None and control_head_idx is not None else None
            )
        ),
        vector=direction_slices.reshape(-1) * scale if condition == "whole_vector" else None,
        stream=stream,
        stage=stage,
        readout_source="combined",
    )
    local_shift = local - baseline_local
    total_shift = total - baseline_total
    return {
        "local_shift": local_shift,
        "directional_local_effect": math.copysign(abs(local_shift), scale * local_shift),
        "total_shift": total_shift,
        "tail_fraction": _tail_fraction(local_shift, total_shift),
        "non_target_kl": _non_target_kl(prefix, baseline_logits, intervened_logits),
        "raw_norm": raw_norm,
        "head_idx": actual_head,
    }


def _cross_family_local_shift(
    model,
    source_prefix: ChoicePrefix,
    off_target_prefix: ChoicePrefix,
    device: torch.device,
    *,
    baseline_local: float,
    layer_idx: int,
    scale: float,
    direction_slices: torch.Tensor,
    condition: str,
    head_idx: int | None = None,
    control_head_idx: int | None = None,
    random_direction_slices: torch.Tensor | None = None,
    stream: str = "x_t",
    stage: str = "post_block",
) -> float:
    intervened_logits, _norm_value, _actual_head = _intervened_logits(
        model,
        off_target_prefix,
        device,
        layer_idx=layer_idx,
        scale=scale,
        direction_slices=direction_slices,
        condition=condition,
        head_idx=head_idx,
        control_head_idx=control_head_idx,
        random_direction_slices=random_direction_slices,
        stream=stream,
        stage=stage,
    )
    local = _local_gap_from_logits(off_target_prefix, intervened_logits)
    return local - baseline_local


def _build_prefix_sets(task_names: set[str], tokenizer, target_family: str) -> tuple[list[ChoicePrefix], list[ChoicePrefix], list[dict[str, str]]]:
    prefixes, invalid = task_case_prefixes(task_names, tokenizer, skip_invalid=True)
    target = [prefix for prefix in prefixes if prefix.task_name == target_family]
    controls = [prefix for prefix in prefixes if prefix.task_name != target_family]
    if len(target) < 2:
        raise ValueError(f"Need at least two valid target cases for family={target_family}")
    if not controls:
        raise ValueError("Need at least one off-target case")
    return target, controls, invalid


def _baseline_cache(model, prefixes: list[ChoicePrefix], device: torch.device) -> dict[str, dict[str, object]]:
    cache: dict[str, dict[str, object]] = {}
    for prefix in prefixes:
        logits = forward_logits(model, prefix.prefix_ids, device, readout_source="combined")
        cache[prefix.case_id] = {
            "local_gap": _local_gap_from_logits(prefix, logits),
            "total_gap": choice_gap(model, prefix, device, readout_source="combined"),
            "logits": logits,
        }
    return cache


def _candidate_rows(
    n_layers: int,
    n_heads: int,
    scales: list[float],
    *,
    candidate_layers: list[int] | None = None,
    candidate_heads: list[int] | None = None,
    whole_layers: list[int] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    layer_pool = candidate_layers if candidate_layers is not None else list(range(n_layers))
    head_pool = candidate_heads if candidate_heads is not None else list(range(n_heads))
    whole_layer_pool = whole_layers if whole_layers is not None else list(range(n_layers))
    channelized = []
    for layer in layer_pool:
        for head in head_pool:
            for scale in scales:
                channelized.append({"condition": "channelized", "layer": layer, "head": head, "scale": scale})
    whole = []
    for layer in whole_layer_pool:
        for scale in scales:
            whole.append({"condition": "whole_vector", "layer": layer, "scale": scale})
    return channelized, whole


def _row_training_summary(
    model,
    train_prefixes: list[ChoicePrefix],
    off_target_prefixes: list[ChoicePrefix],
    baseline_cache: dict[str, dict[str, object]],
    device: torch.device,
    vocab_by_head: torch.Tensor,
    *,
    row: dict[str, object],
    stream: str,
    stage: str,
) -> dict[str, object]:
    target_effects = []
    total_effects = []
    non_target_kls = []
    raw_norms = []
    tail_fractions = []
    off_target_drifts = []

    for prefix in train_prefixes:
        direction = direct_gap_direction_slices(prefix, vocab_by_head)
        metrics = _target_case_metrics(
            model,
            prefix,
            device,
            baseline_local=float(baseline_cache[prefix.case_id]["local_gap"]),
            baseline_total=float(baseline_cache[prefix.case_id]["total_gap"]),
            baseline_logits=baseline_cache[prefix.case_id]["logits"],
            layer_idx=int(row["layer"]),
            scale=float(row["scale"]),
            direction_slices=direction,
            condition=str(row["condition"]),
            head_idx=int(row["head"]) if "head" in row else None,
            stream=stream,
            stage=stage,
        )
        target_effects.append(float(metrics["directional_local_effect"]))
        total_effects.append(math.copysign(abs(float(metrics["total_shift"])), float(row["scale"]) * float(metrics["total_shift"])))
        non_target_kls.append(float(metrics["non_target_kl"]))
        raw_norms.append(float(metrics["raw_norm"]))
        tail_fractions.append(float(metrics["tail_fraction"]))

        for off_prefix in off_target_prefixes:
            off_shift = _cross_family_local_shift(
                model,
                prefix,
                off_prefix,
                device,
                baseline_local=float(baseline_cache[off_prefix.case_id]["local_gap"]),
                layer_idx=int(row["layer"]),
                scale=float(row["scale"]),
                direction_slices=direction,
                condition=str(row["condition"]),
                head_idx=int(row["head"]) if "head" in row else None,
                stream=stream,
                stage=stage,
            )
            off_target_drifts.append(abs(off_shift))

    mean_target = sum(target_effects) / len(target_effects)
    mean_off_target = sum(off_target_drifts) / max(len(off_target_drifts), 1)
    return {
        **row,
        "train_mean_target_local_effect": mean_target,
        "train_mean_target_total_effect": sum(total_effects) / len(total_effects),
        "train_mean_off_target_local_effect": mean_off_target,
        "train_mean_non_target_kl": sum(non_target_kls) / len(non_target_kls),
        "train_mean_tail_fraction": sum(tail_fractions) / len(tail_fractions),
        "train_mean_raw_norm": sum(raw_norms) / len(raw_norms),
        "train_selectivity": mean_target / max(mean_off_target, EPS) if mean_target > 0 else float("-inf"),
        "train_effect_per_unit_norm": mean_target / max(sum(raw_norms) / len(raw_norms), EPS),
    }


def _select_best_row(
    model,
    train_prefixes: list[ChoicePrefix],
    off_target_prefixes: list[ChoicePrefix],
    baseline_cache: dict[str, dict[str, object]],
    device: torch.device,
    vocab_by_head: torch.Tensor,
    *,
    rows: list[dict[str, object]],
    stream: str,
    stage: str,
) -> dict[str, object]:
    summaries = [
        _row_training_summary(
            model,
            train_prefixes,
            off_target_prefixes,
            baseline_cache,
            device,
            vocab_by_head,
            row=row,
            stream=stream,
            stage=stage,
        )
        for row in rows
    ]
    positive = [row for row in summaries if row["train_mean_target_local_effect"] > 0]
    pool = positive if positive else summaries
    return max(pool, key=lambda row: (float(row["train_selectivity"]), float(row["train_mean_target_local_effect"])))


def _pick_random_control_head(head_idx: int, n_heads: int, rng: random.Random) -> int:
    options = [candidate for candidate in range(n_heads) if candidate != head_idx]
    if not options:
        raise ValueError("Need at least two heads for random-channel control")
    return options[rng.randrange(len(options))]


def _alignment_report(source_prefix: ChoicePrefix, off_target_prefixes: list[ChoicePrefix], vocab_by_head: torch.Tensor, target_head: int) -> list[dict[str, float | str]]:
    source_direction = direct_gap_direction_slices(source_prefix, vocab_by_head)
    rows = []
    for off_prefix in off_target_prefixes:
        off_direction = direct_gap_direction_slices(off_prefix, vocab_by_head)
        rows.append(
            {
                "source_case_id": source_prefix.case_id,
                "off_target_case_id": off_prefix.case_id,
                "whole_vector_cosine": _cosine(source_direction, off_direction),
                "target_head_cosine": _cosine(source_direction[target_head], off_direction[target_head]),
            }
        )
    return rows


def _evaluate_fold(
    model,
    target_prefixes: list[ChoicePrefix],
    off_target_prefixes: list[ChoicePrefix],
    baseline_cache: dict[str, dict[str, object]],
    device: torch.device,
    vocab_by_head: torch.Tensor,
    *,
    heldout_idx: int,
    channel_rows: list[dict[str, object]],
    whole_rows: list[dict[str, object]],
    stream: str,
    stage: str,
    rng: random.Random,
    random_generator: torch.Generator,
) -> dict[str, object]:
    heldout = target_prefixes[heldout_idx]
    train = [prefix for idx, prefix in enumerate(target_prefixes) if idx != heldout_idx]

    best_channel = _select_best_row(
        model,
        train,
        off_target_prefixes,
        baseline_cache,
        device,
        vocab_by_head,
        rows=channel_rows,
        stream=stream,
        stage=stage,
    )
    best_whole = _select_best_row(
        model,
        train,
        off_target_prefixes,
        baseline_cache,
        device,
        vocab_by_head,
        rows=whole_rows,
        stream=stream,
        stage=stage,
    )

    heldout_direction = direct_gap_direction_slices(heldout, vocab_by_head)
    random_head = _pick_random_control_head(int(best_channel["head"]), model.config.n_head, rng)
    random_direction = random_direction_slices_like(
        vocab_by_head,
        heldout_direction.unsqueeze(0),
        random_generator,
        exclude_token_ids={heldout.choice_a_token, heldout.choice_b_token},
    )[0]

    def evaluate_condition(condition: str, row: dict[str, object]) -> dict[str, object]:
        target_metrics = _target_case_metrics(
            model,
            heldout,
            device,
            baseline_local=float(baseline_cache[heldout.case_id]["local_gap"]),
            baseline_total=float(baseline_cache[heldout.case_id]["total_gap"]),
            baseline_logits=baseline_cache[heldout.case_id]["logits"],
            layer_idx=int(row["layer"]),
            scale=float(row["scale"]),
            direction_slices=heldout_direction,
            condition=condition,
            head_idx=int(row["head"]) if "head" in row else int(best_channel["head"]),
            control_head_idx=random_head,
            random_direction_slices=random_direction,
            stream=stream,
            stage=stage,
        )
        cross_family = []
        for off_prefix in off_target_prefixes:
            off_shift = _cross_family_local_shift(
                model,
                heldout,
                off_prefix,
                device,
                baseline_local=float(baseline_cache[off_prefix.case_id]["local_gap"]),
                layer_idx=int(row["layer"]),
                scale=float(row["scale"]),
                direction_slices=heldout_direction,
                condition=condition,
                head_idx=int(row["head"]) if "head" in row else int(best_channel["head"]),
                control_head_idx=random_head,
                random_direction_slices=random_direction,
                stream=stream,
                stage=stage,
            )
            cross_family.append({"case_id": off_prefix.case_id, "abs_local_shift": abs(off_shift)})
        selectivity = float(target_metrics["directional_local_effect"]) / max(
            sum(item["abs_local_shift"] for item in cross_family) / max(len(cross_family), 1),
            EPS,
        )
        payload = {
            "condition": condition,
            "layer": int(row["layer"]),
            "scale": float(row["scale"]),
            "head": int(row["head"]) if "head" in row else None,
            "control_head": random_head if condition == "random_channel" else None,
            "heldout_case_id": heldout.case_id,
            "target_local_effect": float(target_metrics["directional_local_effect"]),
            "target_total_effect": math.copysign(abs(float(target_metrics["total_shift"])), float(row["scale"]) * float(target_metrics["total_shift"])),
            "mean_abs_off_target_local_effect": sum(item["abs_local_shift"] for item in cross_family) / max(len(cross_family), 1),
            "non_target_kl": float(target_metrics["non_target_kl"]),
            "tail_fraction": float(target_metrics["tail_fraction"]),
            "raw_norm": float(target_metrics["raw_norm"]),
            "effect_per_unit_norm": float(target_metrics["directional_local_effect"]) / max(float(target_metrics["raw_norm"]), EPS),
            "selectivity": selectivity,
            "cross_family_rows": cross_family,
        }
        if condition == "channelized":
            vector_slice = heldout_direction[int(row["head"])] * float(row["scale"])
            payload["readout_decomposition"] = readout_json_ready(
                decompose_readout(
                    model,
                    heldout,
                    device,
                    layer_idx=int(row["layer"]),
                    head_idx=int(row["head"]),
                    vector_slice=vector_slice,
                    stream=stream,
                    stage=stage,
                )
            )
        return payload

    channel_eval = evaluate_condition("channelized", best_channel)
    whole_eval = evaluate_condition("whole_vector", best_whole)
    random_channel_eval = evaluate_condition("random_channel", best_channel)
    random_direction_eval = evaluate_condition("random_direction", best_channel)

    return {
        "heldout_case_id": heldout.case_id,
        "selected_channelized_row": {
            "layer": int(best_channel["layer"]),
            "head": int(best_channel["head"]),
            "scale": float(best_channel["scale"]),
            "train_selectivity": float(best_channel["train_selectivity"]),
            "train_mean_target_local_effect": float(best_channel["train_mean_target_local_effect"]),
            "train_mean_off_target_local_effect": float(best_channel["train_mean_off_target_local_effect"]),
            "train_mean_raw_norm": float(best_channel["train_mean_raw_norm"]),
        },
        "selected_whole_vector_row": {
            "layer": int(best_whole["layer"]),
            "scale": float(best_whole["scale"]),
            "train_selectivity": float(best_whole["train_selectivity"]),
            "train_mean_target_local_effect": float(best_whole["train_mean_target_local_effect"]),
            "train_mean_off_target_local_effect": float(best_whole["train_mean_off_target_local_effect"]),
            "train_mean_raw_norm": float(best_whole["train_mean_raw_norm"]),
        },
        "channelized": channel_eval,
        "whole_vector": whole_eval,
        "random_channel": random_channel_eval,
        "random_direction": random_direction_eval,
        "channelized_beats_whole_vector": bool(channel_eval["selectivity"] > whole_eval["selectivity"]),
        "alignment_checks": _alignment_report(heldout, off_target_prefixes, vocab_by_head, int(best_channel["head"])),
    }


def _monotonic_sanity_check(
    model,
    target_prefixes: list[ChoicePrefix],
    baseline_cache: dict[str, dict[str, object]],
    device: torch.device,
    vocab_by_head: torch.Tensor,
    *,
    row: dict[str, object],
    stream: str,
    stage: str,
    scales: list[float],
) -> dict[str, object]:
    sign = 1.0 if float(row["scale"]) >= 0 else -1.0
    relevant_scales = sorted([scale for scale in scales if math.copysign(1.0, scale) == sign], key=abs)
    per_case = []
    n_monotonic = 0
    for prefix in target_prefixes:
        direction = direct_gap_direction_slices(prefix, vocab_by_head)
        series = []
        for scale in relevant_scales:
            metrics = _target_case_metrics(
                model,
                prefix,
                device,
                baseline_local=float(baseline_cache[prefix.case_id]["local_gap"]),
                baseline_total=float(baseline_cache[prefix.case_id]["total_gap"]),
                baseline_logits=baseline_cache[prefix.case_id]["logits"],
                layer_idx=int(row["layer"]),
                scale=scale,
                direction_slices=direction,
                condition="channelized",
                head_idx=int(row["head"]),
                stream=stream,
                stage=stage,
            )
            series.append(float(metrics["directional_local_effect"]))
        monotonic = all(series[idx + 1] + 1e-6 >= series[idx] for idx in range(len(series) - 1))
        if monotonic:
            n_monotonic += 1
        per_case.append({"case_id": prefix.case_id, "scales": relevant_scales, "directed_local_effects": series, "monotonic": monotonic})
    return {"n_monotonic_cases": n_monotonic, "passes_minimum_bar": n_monotonic >= 3, "per_case": per_case}


def _report_text(model_label: str, artifact: RunArtifact) -> str:
    metadata = artifact.metadata
    lines = [
        f"# Selectivity Comparison Report — {model_label}",
        "",
        "## Summary",
        "",
        f"- Target family: `{metadata['target_family']}`",
        f"- Off-target families: `{', '.join(metadata['off_target_families'])}`",
        f"- Valid target cases: `{', '.join(metadata['target_case_ids'])}`",
        f"- Valid off-target cases ({metadata['n_off_target_cases']}): `{', '.join(metadata['off_target_case_ids'])}`",
        f"- Mean held-out channelized selectivity: `{metadata['mean_heldout_channelized_selectivity']:.4f}`",
        f"- Mean held-out whole-vector selectivity: `{metadata['mean_heldout_whole_vector_selectivity']:.4f}`",
        f"- Channelized wins on held-out cases: `{metadata['channelized_wins']} / {metadata['n_folds']}`",
        f"- Sanity check passes: `{metadata['sanity_check']['passes_minimum_bar']}`",
        "",
        "## Held-Out Folds",
        "",
    ]
    for fold in artifact.results:
        lines.extend(
            [
                f"### {fold['heldout_case_id']}",
                "",
                f"- Channelized row: `L{fold['selected_channelized_row']['layer']} H{fold['selected_channelized_row']['head']} scale={fold['selected_channelized_row']['scale']}`",
                f"- Whole-vector row: `L{fold['selected_whole_vector_row']['layer']} scale={fold['selected_whole_vector_row']['scale']}`",
                f"- Channelized selectivity: `{fold['channelized']['selectivity']:.4f}`",
                f"- Whole-vector selectivity: `{fold['whole_vector']['selectivity']:.4f}`",
                f"- Random-channel selectivity: `{fold['random_channel']['selectivity']:.4f}`",
                f"- Random-direction selectivity: `{fold['random_direction']['selectivity']:.4f}`",
                f"- Channelized beats whole-vector: `{fold['channelized_beats_whole_vector']}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run channelized vs whole-vector selectivity comparison")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "E2=/mnt/d/data/neuro-symb-v2/backup-data/experiments/"
            "cog-sys-paper-1/series_e_25K_test/E2_independent/checkpoint_epoch_2.pt"
        ],
    )
    parser.add_argument("--target-family", default="recency_bias")
    parser.add_argument("--off-target-families", default="coreference,induction")
    parser.add_argument("--task-names", default="coreference,induction,recency_bias")
    parser.add_argument("--tokenizer-mode", default="reduced_gpt2", choices=("auto", "gpt2", "reduced_gpt2"))
    parser.add_argument("--base-tokenizer", default=DEFAULT_BASE_TOKENIZER)
    parser.add_argument("--vocab-file", default="companion-repo/neuro-symb-v2/neurips-2026/data/grade_school_vocab.pkl")
    parser.add_argument("--scales", default="-8,-4,-2,-1,1,2,4,8")
    parser.add_argument("--stream", default="x_t", choices=("x_t", "x_e"))
    parser.add_argument("--stage", default="post_block", choices=("post_block", "pre_block"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--candidate-layers", default="")
    parser.add_argument("--candidate-heads", default="")
    parser.add_argument("--whole-layers", default="")
    parser.add_argument("--output", default="runs/selectivity_e2_recency")
    args = parser.parse_args(argv)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)
    scales = [float(item.strip()) for item in args.scales.split(",") if item.strip()]
    device = torch.device(args.device)
    task_names = {name.strip() for name in args.task_names.split(",") if name.strip()}
    off_target_families = [name.strip() for name in args.off_target_families.split(",") if name.strip()]
    candidate_layers = parse_csv(args.candidate_layers, int) if args.candidate_layers else None
    candidate_heads = parse_csv(args.candidate_heads, int) if args.candidate_heads else None
    whole_layers = parse_csv(args.whole_layers, int) if args.whole_layers else None

    for model_label, checkpoint_path in parse_model_specs(args.models):
        model, checkpoint = load_symbolic_model(checkpoint_path, device=device)
        tokenizer = build_tokenizer(
            tokenizer_mode=args.tokenizer_mode,
            base_tokenizer_path=args.base_tokenizer,
            vocab_file=args.vocab_file,
            checkpoint_config=checkpoint["config"],
        )
        target_prefixes, off_target_prefixes, invalid = _build_prefix_sets(task_names, tokenizer, args.target_family)
        off_target_prefixes = [prefix for prefix in off_target_prefixes if prefix.task_name in set(off_target_families)]
        if not off_target_prefixes:
            raise ValueError("No valid off-target cases after family filter")

        baseline_cache = _baseline_cache(model, target_prefixes + off_target_prefixes, device)
        vocab_by_head = output_embedding_by_head(model)
        channel_rows, whole_rows = _candidate_rows(
            model.config.n_layer,
            model.config.n_head,
            scales,
            candidate_layers=candidate_layers,
            candidate_heads=candidate_heads,
            whole_layers=whole_layers,
        )
        rng = random.Random(args.random_seed)
        random_generator = make_generator(args.random_seed)

        global_best_channel = _select_best_row(
            model,
            target_prefixes,
            off_target_prefixes,
            baseline_cache,
            device,
            vocab_by_head,
            rows=channel_rows,
            stream=args.stream,
            stage=args.stage,
        )
        sanity_check = _monotonic_sanity_check(
            model,
            target_prefixes,
            baseline_cache,
            device,
            vocab_by_head,
            row=global_best_channel,
            stream=args.stream,
            stage=args.stage,
            scales=scales,
        )

        folds = []
        for heldout_idx in range(len(target_prefixes)):
            folds.append(
                _evaluate_fold(
                    model,
                    target_prefixes,
                    off_target_prefixes,
                    baseline_cache,
                    device,
                    vocab_by_head,
                    heldout_idx=heldout_idx,
                    channel_rows=channel_rows,
                    whole_rows=whole_rows,
                    stream=args.stream,
                    stage=args.stage,
                    rng=rng,
                    random_generator=random_generator,
                )
            )

        artifact = RunArtifact(
            run_id=output_dir.name,
            model_id=model_label,
            phase="steering",
            results=folds,
            metadata={
                "phase_name": "channelized_vs_whole_vector_selectivity",
                "target_family": args.target_family,
                "off_target_families": off_target_families,
                "target_case_ids": [prefix.case_id for prefix in target_prefixes],
                "off_target_case_ids": [prefix.case_id for prefix in off_target_prefixes],
                "n_off_target_cases": len(off_target_prefixes),
                "invalid_cases": invalid,
                "scales": scales,
                "stream": args.stream,
                "stage": args.stage,
                "candidate_layers": candidate_layers,
                "candidate_heads": candidate_heads,
                "whole_layers": whole_layers,
                "sanity_check": sanity_check,
                "mean_heldout_channelized_selectivity": sum(fold["channelized"]["selectivity"] for fold in folds) / len(folds),
                "mean_heldout_whole_vector_selectivity": sum(fold["whole_vector"]["selectivity"] for fold in folds) / len(folds),
                "mean_heldout_random_channel_selectivity": sum(fold["random_channel"]["selectivity"] for fold in folds) / len(folds),
                "mean_heldout_random_direction_selectivity": sum(fold["random_direction"]["selectivity"] for fold in folds) / len(folds),
                "channelized_wins": sum(1 for fold in folds if fold["channelized_beats_whole_vector"]),
                "n_folds": len(folds),
                "global_best_channelized_row": {
                    key: value
                    for key, value in global_best_channel.items()
                    if key.startswith("train_") or key in {"condition", "layer", "head", "scale"}
                },
            },
        )

        artifact_path = output_dir / "selectivity_comparison.json"
        report_path = output_dir / "report.md"
        save_json(asdict(artifact), artifact_path)
        report_path.write_text(_report_text(model_label, artifact), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
