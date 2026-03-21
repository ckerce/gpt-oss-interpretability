#!/usr/bin/env python3
"""Measure whether layerwise CASCADE quality tracks intervention sensitivity.

This implements the primary bridge experiment defined in lab-notebook Entry 20
on the currently audited clean pool. The script computes:

1. layerwise same-model CASCADE reconstruction metrics at the first divergent
   answer token
2. layerwise residual-delta block-ablation damage on the benchmark choice score
3. per-case and pooled correlations between CASCADE quality change and
   intervention damage
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

import torch

from gpt_oss_interp.benchmarks.tasks import all_tasks
from gpt_oss_interp.benchmarks.pools import PROVISIONAL_BRIDGE_POOL_CASE_IDS
from gpt_oss_interp.capture.activation_cache import ActivationCache
from gpt_oss_interp.config import (
    InterventionKind,
    InterventionSpec,
    InterventionTarget,
    TargetUnit,
)
from gpt_oss_interp.harmony.prompting import encode_prompt_with_completion


###############################################################################
#
# Case selection
#
###############################################################################


def _all_cases() -> dict[str, tuple[object, object]]:
    return {
        case.case_id: (task, case)
        for task in all_tasks()
        for case in task.cases
    }


def _load_selected_case_ids(
    stratification_json: Path,
    audit_json: Path,
    set_name: str,
    required_classification: str,
    excluded_case_ids: set[str],
) -> list[str]:
    strat_payload = json.loads(stratification_json.read_text())
    audit_payload = json.loads(audit_json.read_text())

    strat_ids = set(strat_payload["recommended_sets"][set_name])
    audit_ids = {
        row["case_id"]
        for row in audit_payload["rows"]
        if row["classification"] == required_classification
    }
    selected = sorted((strat_ids & audit_ids) - excluded_case_ids)
    if not selected:
        raise RuntimeError("No cases selected for the bridge experiment")
    return selected


###############################################################################
#
# Divergence and choice summaries
#
###############################################################################


def _first_divergence(tokenizer, prompt: str, choice_a: str, choice_b: str) -> dict[str, object]:
    full_a, start_a = encode_prompt_with_completion(tokenizer, prompt, choice_a)
    full_b, start_b = encode_prompt_with_completion(tokenizer, prompt, choice_b)
    if start_a != start_b:
        raise ValueError("Prompt prefix lengths differ")

    comp_a = full_a[start_a:]
    comp_b = full_b[start_b:]
    max_len = min(len(comp_a), len(comp_b))
    diff_positions = [i for i in range(max_len) if comp_a[i] != comp_b[i]]
    if not diff_positions:
        raise ValueError("Choices do not diverge")

    first = diff_positions[0]
    return {
        "decision_input_ids": full_a[: start_a + first],
        "decision_position_index": len(full_a[: start_a + first]) - 1,
        "shared_prefix_tokens": [tokenizer.decode([tid]) for tid in comp_a[:first]],
        "token_a": comp_a[first],
        "token_b": comp_b[first],
        "token_a_text": tokenizer.decode([comp_a[first]]),
        "token_b_text": tokenizer.decode([comp_b[first]]),
    }


def _summarize_choice_logprobs(choice_logprobs: dict[str, float], expected_label: str) -> dict[str, object]:
    predicted_label = max(choice_logprobs, key=choice_logprobs.get)
    best_logprob = choice_logprobs[predicted_label]
    expected_logprob = choice_logprobs[expected_label]
    sorted_values = sorted(choice_logprobs.values(), reverse=True)
    runner_up = sorted_values[1] if len(sorted_values) > 1 else best_logprob
    if predicted_label == expected_label:
        margin = best_logprob - runner_up
    else:
        margin = expected_logprob - best_logprob
    return {
        "predicted_label": predicted_label,
        "expected_label": expected_label,
        "correct": int(predicted_label == expected_label),
        "best_logprob": best_logprob,
        "expected_logprob": expected_logprob,
        "margin": margin,
        "choice_logprobs": choice_logprobs,
    }


###############################################################################
#
# CASCADE metrics
#
###############################################################################


def _capture_last_token_hiddens(backend, input_ids: list[int], layer_indices: list[int]) -> dict[int, torch.Tensor]:
    block_names = [backend.structure.block_names[i] for i in layer_indices]
    cache = ActivationCache(detach=True, to_cpu=True)
    handles = cache.register(backend.model, block_names)
    tensor_ids = torch.tensor([input_ids], device=backend.device)
    try:
        with torch.no_grad():
            backend.model(tensor_ids)
    finally:
        for handle in handles:
            handle.remove()

    records: dict[int, torch.Tensor] = {}
    for layer_idx in layer_indices:
        record = cache.last(backend.structure.block_names[layer_idx])
        if record is None:
            raise RuntimeError(f"No activation captured for layer {layer_idx}")
        records[layer_idx] = record.tensor[0, -1].float()
    return records


def _layer_log_probs(backend, hidden_vec: torch.Tensor) -> torch.Tensor:
    norm_module = backend.structure.final_norm
    lm_head = backend.structure.lm_head
    if norm_module is None or lm_head is None:
        raise RuntimeError("Model structure is missing final_norm or lm_head")

    norm_param = next(norm_module.parameters())
    lm_head_param = next(lm_head.parameters())
    normed = norm_module(hidden_vec.unsqueeze(0).to(device=norm_param.device, dtype=norm_param.dtype))
    logits = lm_head(normed.to(device=lm_head_param.device, dtype=lm_head_param.dtype)).float().cpu()[0]
    return torch.log_softmax(logits, dim=-1)


def _distribution_kl(teacher_log_probs: torch.Tensor, reconstructed_logits: torch.Tensor) -> float:
    teacher_probs = teacher_log_probs.exp()
    recon_log_probs = torch.log_softmax(reconstructed_logits, dim=-1)
    return float(torch.sum(teacher_probs * (teacher_log_probs - recon_log_probs)).item())


def _relative_residual(matrix: torch.Tensor, solution: torch.Tensor, rhs: torch.Tensor) -> float:
    residual = matrix @ solution - rhs
    denom = rhs.norm().item()
    if denom == 0.0:
        return float(residual.norm().item())
    return float(residual.norm().item() / denom)


def _layerwise_cascade_metrics(backend, decision_input_ids: list[int], layer_indices: list[int]) -> list[dict[str, object]]:
    hiddens = _capture_last_token_hiddens(backend, decision_input_ids, layer_indices)

    if backend.structure.embed is None or backend.structure.lm_head is None:
        raise RuntimeError("Model structure is missing embed or lm_head")

    input_tensor = torch.tensor([decision_input_ids], device=backend.device)
    with torch.no_grad():
        embed_out = backend.structure.embed(input_tensor)
        if isinstance(embed_out, tuple):
            embed_out = embed_out[0]
        x_t = embed_out[0, -1].float().cpu()

    unembedding = backend.structure.lm_head.weight.detach().float().cpu()
    centered_unembedding = unembedding - unembedding.mean(dim=0, keepdim=True)
    centered_xt = centered_unembedding @ x_t

    teacher_log_probs = [_layer_log_probs(backend, hiddens[layer_idx]) for layer_idx in layer_indices]
    teacher_matrix = torch.stack(teacher_log_probs, dim=1)
    centered_teacher = teacher_matrix - teacher_matrix.mean(dim=0, keepdim=True)
    rhs_matrix = centered_teacher - centered_xt.unsqueeze(1)

    solutions = torch.linalg.lstsq(centered_unembedding, rhs_matrix).solution
    reconstructed_logits = centered_unembedding @ (x_t.unsqueeze(1) + solutions)

    rows: list[dict[str, object]] = []
    prev_kl: float | None = None
    for column, layer_idx in enumerate(layer_indices):
        teacher_lp = teacher_matrix[:, column]
        rhs = rhs_matrix[:, column]
        solution = solutions[:, column]
        reconstructed = reconstructed_logits[:, column]
        reconstruction_kl = _distribution_kl(teacher_lp, reconstructed)
        relative_residual = _relative_residual(centered_unembedding, solution, rhs)
        delta_kl = None if prev_kl is None else reconstruction_kl - prev_kl
        rows.append(
            {
                "layer": layer_idx,
                "reconstruction_kl": reconstruction_kl,
                "relative_residual": relative_residual,
                "delta_kl": delta_kl,
            }
        )
        prev_kl = reconstruction_kl
    return rows


###############################################################################
#
# Intervention metrics
#
###############################################################################


def _layer_intervention_spec(layer_idx: int) -> InterventionSpec:
    return InterventionSpec(
        name=f"bridge_delta_L{layer_idx}",
        kind=InterventionKind.LAYER_SCALE,
        target=InterventionTarget(
            unit=TargetUnit.LAYER,
            layer_indices=(layer_idx,),
            note="Residual-delta block skip for CASCADE bridge measurement",
        ),
        scales=(0.0,),
        description=f"Residual-preserving block-delta suppression at layer {layer_idx}",
        params={"preserve_residual": True},
    )


def _case_intervention_rows(backend, case, baseline_summary: dict[str, object], layer_indices: list[int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for layer_idx in layer_indices:
        spec = _layer_intervention_spec(layer_idx)
        backend.clear_interventions()
        backend.apply_intervention(spec, 0.0)
        final_score = backend.score_case(case)
        backend.clear_interventions()

        final_summary = _summarize_choice_logprobs(final_score.choice_logprobs, case.expected_label)
        rows.append(
            {
                "layer": layer_idx,
                "intervened_margin": final_summary["margin"],
                "intervened_correct": final_summary["correct"],
                "margin_damage": baseline_summary["margin"] - final_summary["margin"],
                "correctness_flip": int(
                    bool(baseline_summary["correct"]) and not bool(final_summary["correct"])
                ),
                "intervened_predicted_label": final_summary["predicted_label"],
                "intervened_choice_logprobs": final_summary["choice_logprobs"],
            }
        )
    return rows


###############################################################################
#
# Statistics
#
###############################################################################


def _rankdata(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[indexed[k][0]] = rank
        i = j
    return ranks


def _pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(y)
    dx = [value - mean_x for value in x]
    dy = [value - mean_y for value in y]
    denom_x = math.sqrt(sum(value * value for value in dx))
    denom_y = math.sqrt(sum(value * value for value in dy))
    if math.isclose(denom_x, 0.0) or math.isclose(denom_y, 0.0):
        return None
    return sum(a * b for a, b in zip(dx, dy)) / (denom_x * denom_y)


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    return _pearson(_rankdata(x), _rankdata(y))


def _binom_cdf(k: int, n: int, p: float = 0.5) -> float:
    return sum(math.comb(n, i) * (p**i) * ((1.0 - p) ** (n - i)) for i in range(k + 1))


def _two_sided_sign_test_p(num_positive: int, num_nonzero: int) -> float | None:
    if num_nonzero == 0:
        return None
    tail = min(
        _binom_cdf(num_positive, num_nonzero),
        1.0 - _binom_cdf(num_positive - 1, num_nonzero) if num_positive > 0 else 1.0,
    )
    return min(1.0, 2.0 * tail)


def _bootstrap_case_resampled_pooled_rho(
    case_rows: dict[str, list[dict[str, object]]],
    num_samples: int,
    seed: int,
) -> dict[str, object]:
    rng = random.Random(seed)
    case_ids = list(case_rows)
    samples: list[float] = []
    for _ in range(num_samples):
        chosen = [rng.choice(case_ids) for _ in case_ids]
        delta_values: list[float] = []
        damage_values: list[float] = []
        for case_id in chosen:
            for row in case_rows[case_id]:
                delta_values.append(float(row["delta_kl"]))
                damage_values.append(float(row["margin_damage"]))
        rho = _spearman(delta_values, damage_values)
        if rho is not None:
            samples.append(rho)

    if not samples:
        return {"num_samples": 0, "ci_lower": None, "ci_upper": None}

    ordered = sorted(samples)
    lower_idx = max(0, int(0.025 * (len(ordered) - 1)))
    upper_idx = min(len(ordered) - 1, int(0.975 * (len(ordered) - 1)))
    return {
        "num_samples": len(samples),
        "ci_lower": ordered[lower_idx],
        "ci_upper": ordered[upper_idx],
    }


def _permutation_pooled_rho(
    case_rows: dict[str, list[dict[str, object]]],
    observed_rho: float,
    num_samples: int,
    seed: int,
) -> dict[str, object]:
    rng = random.Random(seed)
    hits = 0
    valid = 0
    for _ in range(num_samples):
        delta_values: list[float] = []
        permuted_damage_values: list[float] = []
        for rows in case_rows.values():
            case_delta = [float(row["delta_kl"]) for row in rows]
            case_damage = [float(row["margin_damage"]) for row in rows]
            rng.shuffle(case_damage)
            delta_values.extend(case_delta)
            permuted_damage_values.extend(case_damage)
        rho = _spearman(delta_values, permuted_damage_values)
        if rho is None:
            continue
        valid += 1
        if abs(rho) >= abs(observed_rho):
            hits += 1
    if valid == 0:
        return {"num_samples": 0, "p_value": None}
    return {"num_samples": valid, "p_value": (hits + 1) / (valid + 1)}


def _task_demeaned_spearman(rows: list[dict[str, object]]) -> float | None:
    task_to_delta: dict[str, list[float]] = {}
    task_to_damage: dict[str, list[float]] = {}
    for row in rows:
        task = str(row["task_name"])
        task_to_delta.setdefault(task, []).append(float(row["delta_kl"]))
        task_to_damage.setdefault(task, []).append(float(row["margin_damage"]))

    delta_means = {task: statistics.fmean(values) for task, values in task_to_delta.items()}
    damage_means = {task: statistics.fmean(values) for task, values in task_to_damage.items()}

    centered_delta = [float(row["delta_kl"]) - delta_means[str(row["task_name"])] for row in rows]
    centered_damage = [float(row["margin_damage"]) - damage_means[str(row["task_name"])] for row in rows]
    return _spearman(centered_delta, centered_damage)


def _partial_spearman_against_layer(rows: list[dict[str, object]]) -> float | None:
    x = _rankdata([float(row["delta_kl"]) for row in rows])
    y = _rankdata([float(row["margin_damage"]) for row in rows])
    z = _rankdata([float(row["layer"]) for row in rows])
    if len(x) < 3:
        return None

    mean_z = statistics.fmean(z)
    centered_z = [value - mean_z for value in z]
    denom = sum(value * value for value in centered_z)
    if math.isclose(denom, 0.0):
        return None

    def residualize(values: list[float]) -> list[float]:
        mean_v = statistics.fmean(values)
        centered_v = [value - mean_v for value in values]
        beta = sum(a * b for a, b in zip(centered_v, centered_z)) / denom
        return [v - beta * zc for v, zc in zip(centered_v, centered_z)]

    return _pearson(residualize(x), residualize(y))


###############################################################################
#
# Reporting
#
###############################################################################


def _write_report(path: Path, payload: dict[str, object]) -> None:
    summary = payload["summary"]
    selected_case_ids = payload["selected_case_ids"]

    lines = [
        "# CASCADE-Intervention Bridge",
        "",
        f"Model: `{payload['model']}`",
        f"Selected cases: `{len(selected_case_ids)}`",
        f"Layer range: `{payload['layer_start']}-{payload['layer_end']}`",
        "",
        "## Pool",
        "",
    ]
    for case_id in selected_case_ids:
        lines.append(f"- `{case_id}`")

    lines.extend(
        [
            "",
            "## Primary Summary",
            "",
            f"- Median per-case Spearman(delta_KL, margin_damage): `{summary['per_case']['median_rho']}`",
            f"- Positive per-case correlations: `{summary['per_case']['num_positive']}/{summary['per_case']['num_nonzero']}`",
            f"- Sign-test p-value: `{summary['per_case']['sign_test_p_value']}`",
            f"- Pooled Spearman(delta_KL, margin_damage): `{summary['pooled']['rho']}`",
            f"- Bootstrap 95% CI: `[{summary['pooled']['bootstrap_ci_lower']}, {summary['pooled']['bootstrap_ci_upper']}]`",
            f"- Permutation p-value: `{summary['pooled']['permutation_p_value']}`",
            f"- Family-demeaned pooled Spearman: `{summary['family_controlled_rho']}`",
            f"- Partial Spearman controlling for layer index: `{summary['layer_partial_rho']}`",
            "",
            "## Per-Case Correlations",
            "",
            "| Case | Task | Rho(delta_KL, damage) | Rho(residual, damage) | Positive |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )

    for row in payload["per_case_correlations"]:
        lines.append(
            f"| {row['case_id']} | {row['task_name']} | {row['delta_kl_vs_margin_damage_rho']} | "
            f"{row['relative_residual_vs_margin_damage_rho']} | {row['positive_primary']} |"
        )

    lines.extend(
        [
            "",
            "## Family Summary",
            "",
            "| Family | Cases | Median Rho | Positive / Nonzero |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for family, row in sorted(payload["family_summary"].items()):
        lines.append(
            f"| {family} | {row['num_cases']} | {row['median_rho']} | "
            f"{row['num_positive']}/{row['num_nonzero']} |"
        )

    lines.extend(
        [
            "",
            "## Layer Rows",
            "",
            "| Case | Layer | delta_KL | reconstruction_KL | residual | margin_damage | flip |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["pooled_rows"]:
        lines.append(
            f"| {row['case_id']} | {row['layer']} | {row['delta_kl']:.8f} | {row['reconstruction_kl']:.8f} | "
            f"{row['relative_residual']:.8f} | {row['margin_damage']:.8f} | {row['correctness_flip']} |"
        )

    path.write_text("\n".join(lines) + "\n")


###############################################################################
#
# Main
#
###############################################################################


def main() -> int:
    parser = argparse.ArgumentParser(description="Bridge CASCADE quality and intervention sensitivity")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument(
        "--stratification_json",
        default="runs/analysis_set_stratification/analysis_set_stratification.json",
    )
    parser.add_argument(
        "--audit_json",
        default="runs/retained_case_decision_audit/decision_audit.json",
    )
    parser.add_argument("--set_name", default="main_analysis_soft")
    parser.add_argument("--required_classification", default="local_support")
    parser.add_argument(
        "--pool_name",
        default=None,
        choices=["provisional_bridge"],
        help="Optional checked-in case pool to use instead of selecting from run artifacts",
    )
    parser.add_argument(
        "--case_ids",
        default=None,
        help="Optional comma-separated case IDs to run instead of auto-selecting from the audited pool",
    )
    parser.add_argument(
        "--exclude_case_ids",
        default="induction_002",
        help="Comma-separated case IDs to exclude from the bridge pool",
    )
    parser.add_argument("--layer_start", type=int, default=4)
    parser.add_argument("--layer_end", type=int, default=22)
    parser.add_argument("--bootstrap_samples", type=int, default=400)
    parser.add_argument("--permutation_samples", type=int, default=400)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output",
        default="runs/bridge_experiment",
        help="Output directory",
    )
    args = parser.parse_args()

    from gpt_oss_interp.backends.transformers_gpt_oss import GPTOSSTransformersBackend

    stratification_json = Path(args.stratification_json)
    if not stratification_json.is_absolute():
        stratification_json = Path.cwd() / stratification_json
    audit_json = Path(args.audit_json)
    if not audit_json.is_absolute():
        audit_json = Path.cwd() / audit_json

    excluded_case_ids = {
        case_id.strip() for case_id in args.exclude_case_ids.split(",") if case_id.strip()
    }
    if args.case_ids:
        selected_case_ids = [
            case_id.strip() for case_id in args.case_ids.split(",") if case_id.strip()
        ]
    elif args.pool_name == "provisional_bridge":
        selected_case_ids = list(PROVISIONAL_BRIDGE_POOL_CASE_IDS)
    else:
        selected_case_ids = _load_selected_case_ids(
            stratification_json=stratification_json,
            audit_json=audit_json,
            set_name=args.set_name,
            required_classification=args.required_classification,
            excluded_case_ids=excluded_case_ids,
        )

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    backend = GPTOSSTransformersBackend(model_name=args.model)
    case_lookup = _all_cases()
    layer_indices = list(range(args.layer_start, args.layer_end + 1))

    case_payloads: list[dict[str, object]] = []
    pooled_rows: list[dict[str, object]] = []
    case_rows_for_stats: dict[str, list[dict[str, object]]] = {}

    for idx, case_id in enumerate(selected_case_ids, start=1):
        print(f"[{idx}/{len(selected_case_ids)}] Measuring {case_id}")
        task, case = case_lookup[case_id]
        divergence = _first_divergence(backend.tokenizer, case.prompt, case.choices["A"], case.choices["B"])

        baseline_score = backend.score_case(case)
        baseline_summary = _summarize_choice_logprobs(baseline_score.choice_logprobs, case.expected_label)
        if not baseline_summary["correct"]:
            raise RuntimeError(f"Selected bridge case is not baseline-correct: {case_id}")

        cascade_rows = _layerwise_cascade_metrics(backend, divergence["decision_input_ids"], layer_indices)
        intervention_rows = _case_intervention_rows(backend, case, baseline_summary, layer_indices)
        merged_rows: list[dict[str, object]] = []
        for cascade_row, intervention_row in zip(cascade_rows, intervention_rows):
            merged = {
                "case_id": case_id,
                "task_name": task.name,
                "behavior": task.behavior,
                "layer": cascade_row["layer"],
                "reconstruction_kl": cascade_row["reconstruction_kl"],
                "relative_residual": cascade_row["relative_residual"],
                "delta_kl": cascade_row["delta_kl"],
                "margin_damage": intervention_row["margin_damage"],
                "correctness_flip": intervention_row["correctness_flip"],
                "intervened_margin": intervention_row["intervened_margin"],
                "intervened_correct": intervention_row["intervened_correct"],
                "intervened_predicted_label": intervention_row["intervened_predicted_label"],
            }
            merged_rows.append(merged)

        stat_rows = [row for row in merged_rows if row["delta_kl"] is not None]
        delta_values = [float(row["delta_kl"]) for row in stat_rows]
        damage_values = [float(row["margin_damage"]) for row in stat_rows]
        residual_values = [float(row["relative_residual"]) for row in stat_rows]
        primary_rho = _spearman(delta_values, damage_values)
        residual_rho = _spearman(residual_values, damage_values)

        case_payloads.append(
            {
                "case_id": case_id,
                "task_name": task.name,
                "behavior": task.behavior,
                "prompt": case.prompt,
                "choices": case.choices,
                "expected_label": case.expected_label,
                "baseline": baseline_summary,
                "decision": {
                    "position_index": divergence["decision_position_index"],
                    "shared_prefix_tokens": divergence["shared_prefix_tokens"],
                    "token_a": divergence["token_a_text"],
                    "token_b": divergence["token_b_text"],
                },
                "layers": merged_rows,
                "correlations": {
                    "delta_kl_vs_margin_damage_rho": primary_rho,
                    "relative_residual_vs_margin_damage_rho": residual_rho,
                },
            }
        )
        pooled_rows.extend(stat_rows)
        case_rows_for_stats[case_id] = stat_rows
        if primary_rho is None:
            print(f"  {case_id}: primary rho unavailable")
        else:
            print(f"  {case_id}: primary rho={primary_rho:.4f}")

    per_case_correlations = [
        {
            "case_id": payload["case_id"],
            "task_name": payload["task_name"],
            "delta_kl_vs_margin_damage_rho": payload["correlations"]["delta_kl_vs_margin_damage_rho"],
            "relative_residual_vs_margin_damage_rho": payload["correlations"]["relative_residual_vs_margin_damage_rho"],
            "positive_primary": int(
                payload["correlations"]["delta_kl_vs_margin_damage_rho"] is not None
                and payload["correlations"]["delta_kl_vs_margin_damage_rho"] > 0
            ),
        }
        for payload in case_payloads
    ]

    valid_rhos = [
        float(row["delta_kl_vs_margin_damage_rho"])
        for row in per_case_correlations
        if row["delta_kl_vs_margin_damage_rho"] is not None
    ]
    nonzero_rhos = [rho for rho in valid_rhos if not math.isclose(rho, 0.0, abs_tol=1e-12)]
    num_positive = sum(1 for rho in nonzero_rhos if rho > 0)

    pooled_delta = [float(row["delta_kl"]) for row in pooled_rows]
    pooled_damage = [float(row["margin_damage"]) for row in pooled_rows]
    pooled_rho = _spearman(pooled_delta, pooled_damage)
    bootstrap = _bootstrap_case_resampled_pooled_rho(
        case_rows=case_rows_for_stats,
        num_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    permutation = _permutation_pooled_rho(
        case_rows=case_rows_for_stats,
        observed_rho=pooled_rho if pooled_rho is not None else 0.0,
        num_samples=args.permutation_samples,
        seed=args.seed + 1,
    )

    family_summary: dict[str, dict[str, object]] = {}
    for task_name in sorted({payload["task_name"] for payload in case_payloads}):
        family_rows = [
            row["delta_kl_vs_margin_damage_rho"]
            for row in per_case_correlations
            if row["task_name"] == task_name and row["delta_kl_vs_margin_damage_rho"] is not None
        ]
        family_nonzero = [rho for rho in family_rows if not math.isclose(float(rho), 0.0, abs_tol=1e-12)]
        family_summary[task_name] = {
            "num_cases": len(family_rows),
            "median_rho": statistics.median(family_rows) if family_rows else None,
            "num_positive": sum(1 for rho in family_nonzero if float(rho) > 0),
            "num_nonzero": len(family_nonzero),
        }

    summary = {
        "per_case": {
            "median_rho": statistics.median(valid_rhos) if valid_rhos else None,
            "num_positive": num_positive,
            "num_nonzero": len(nonzero_rhos),
            "sign_test_p_value": _two_sided_sign_test_p(num_positive, len(nonzero_rhos)),
        },
        "pooled": {
            "rho": pooled_rho,
            "bootstrap_ci_lower": bootstrap["ci_lower"],
            "bootstrap_ci_upper": bootstrap["ci_upper"],
            "permutation_p_value": permutation["p_value"],
        },
        "family_controlled_rho": _task_demeaned_spearman(pooled_rows),
        "layer_partial_rho": _partial_spearman_against_layer(pooled_rows),
    }

    payload = {
        "model": args.model,
        "set_name": args.set_name,
        "required_classification": args.required_classification,
        "excluded_case_ids": sorted(excluded_case_ids),
        "selected_case_ids": selected_case_ids,
        "layer_start": args.layer_start,
        "layer_end": args.layer_end,
        "bootstrap_samples": args.bootstrap_samples,
        "permutation_samples": args.permutation_samples,
        "summary": summary,
        "per_case_correlations": per_case_correlations,
        "family_summary": family_summary,
        "pooled_rows": pooled_rows,
        "cases": case_payloads,
    }

    (output_dir / "bridge_cascade_intervention.json").write_text(json.dumps(payload, indent=2) + "\n")
    _write_report(output_dir / "bridge_cascade_intervention.md", payload)

    print(f"Bridge experiment complete on {len(selected_case_ids)} cases")
    if summary["pooled"]["rho"] is not None:
        print(
            "  pooled rho(delta_KL, margin_damage)="
            f"{summary['pooled']['rho']:.4f}"
        )
    if summary["per_case"]["median_rho"] is not None:
        print(
            "  median per-case rho(delta_KL, margin_damage)="
            f"{summary['per_case']['median_rho']:.4f}"
        )
    print(f"Reports written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
