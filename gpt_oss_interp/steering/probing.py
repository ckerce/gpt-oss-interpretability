"""Phase 1 per-channel differential probing for dual-stream symbolic models."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from gpt_oss_interp.common.artifacts import RunArtifact
from gpt_oss_interp.common.io import save_json
from gpt_oss_interp.steering.controls import (
    label_permuted_scores,
    make_generator,
    random_direction_slices_like,
    shuffled_head_scores,
)
from gpt_oss_interp.steering.interventions import (
    DEFAULT_BASE_TOKENIZER,
    DEFAULT_MODEL_PATHS,
    DEFAULT_VOCAB_FILE,
    ChoicePrefix,
    build_tokenizer,
    capture_stream_trajectory,
    direction_slices_for_case,
    load_symbolic_model,
    output_embedding_by_head,
    parse_model_specs,
    task_case_prefixes,
)


@dataclass
class ChannelProbeConfig:
    model_label: str
    checkpoint_path: str
    task_names: tuple[str, ...]
    tokenizer_mode: str = "gpt2"
    base_tokenizer: str = DEFAULT_BASE_TOKENIZER
    vocab_file: str = ""
    device: str = "cpu"
    null_samples: int = 128
    promotion_threshold: float = 0.70
    null_margin: float = 0.10
    random_seed: int = 0


def _case_tensor_records(
    model,
    prefixes: list[ChoicePrefix],
    device: torch.device,
) -> tuple[
    dict[str, list[ChoicePrefix]],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, list[float]],
]:
    vocab_by_head = output_embedding_by_head(model)
    by_family: dict[str, list[ChoicePrefix]] = {}
    real_scores: dict[str, list[torch.Tensor]] = {}
    sensitivity_scores: dict[str, list[torch.Tensor]] = {}
    direction_slices: dict[str, list[torch.Tensor]] = {}
    decision_vectors: dict[str, list[torch.Tensor]] = {}
    layer_deltas: dict[str, list[float]] = {}

    for prefix in prefixes:
        trajectory = capture_stream_trajectory(model, prefix.prefix_ids, device=device)
        layer_x_t = trajectory["layer_x_t"]
        n_layers, _, n_embd = layer_x_t.shape
        n_head = model.config.n_head
        head_dim = model.config.head_dim
        layer_xt_heads = layer_x_t.view(n_layers, layer_x_t.shape[1], n_head, head_dim)

        direction = direction_slices_for_case(prefix, vocab_by_head)
        decision_pos = prefix.decision_position_index
        decision_vecs = layer_xt_heads[:, decision_pos]
        real = (decision_vecs * direction.unsqueeze(0)).sum(dim=-1)
        delta = float((layer_x_t - layer_x_t[0:1]).abs().max().item())

        per_pos = (layer_xt_heads * direction.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        other_mask = torch.ones(layer_xt_heads.shape[1], dtype=torch.bool)
        other_mask[decision_pos] = False
        if bool(other_mask.any()):
            other_mean = per_pos[:, other_mask].abs().mean(dim=1)
        else:
            other_mean = torch.zeros_like(real)
        sensitivity = real.abs() - other_mean

        family = prefix.task_name
        by_family.setdefault(family, []).append(prefix)
        real_scores.setdefault(family, []).append(real)
        sensitivity_scores.setdefault(family, []).append(sensitivity)
        direction_slices.setdefault(family, []).append(direction)
        decision_vectors.setdefault(family, []).append(decision_vecs)
        layer_deltas.setdefault(family, []).append(delta)

    stacked_real = {family: torch.stack(rows, dim=0) for family, rows in real_scores.items()}
    stacked_sensitivity = {family: torch.stack(rows, dim=0) for family, rows in sensitivity_scores.items()}
    stacked_dirs = {family: torch.stack(rows, dim=0) for family, rows in direction_slices.items()}
    stacked_decision = {family: torch.stack(rows, dim=0) for family, rows in decision_vectors.items()}
    return by_family, stacked_real, stacked_sensitivity, stacked_dirs, stacked_decision, layer_deltas


def probe_channel_preferences(model, prefixes: list[ChoicePrefix], config: ChannelProbeConfig) -> dict[str, object]:
    device = torch.device(config.device)
    by_family, real_scores, sensitivity_scores, direction_slices, decision_vectors, layer_deltas = _case_tensor_records(
        model,
        prefixes,
        device,
    )
    vocab_by_head = output_embedding_by_head(model)

    family_results: dict[str, object] = {}
    for family, family_prefixes in by_family.items():
        scores = real_scores[family]
        sensitivities = sensitivity_scores[family]
        dirs = direction_slices[family]
        decision_vecs = decision_vectors[family]
        generator = make_generator(config.random_seed + hash(family) % 10_000)
        exclude_ids = {
            prefix.choice_a_token for prefix in family_prefixes
        } | {
            prefix.choice_b_token for prefix in family_prefixes
        }

        shuffled_accs: list[torch.Tensor] = []
        label_perm_accs: list[torch.Tensor] = []
        random_dir_accs: list[torch.Tensor] = []
        for _ in range(config.null_samples):
            shuffled = shuffled_head_scores(scores, generator)
            shuffled_accs.append((shuffled > 0).float().mean(dim=0))

            permuted = label_permuted_scores(scores, generator)
            label_perm_accs.append((permuted > 0).float().mean(dim=0))

            rand_dirs = random_direction_slices_like(vocab_by_head, dirs, generator, exclude_token_ids=exclude_ids)
            rand_scores = (decision_vecs * rand_dirs.unsqueeze(1)).sum(dim=-1)
            random_dir_accs.append((rand_scores > 0).float().mean(dim=0))

        family_results[family] = {
            "case_ids": [prefix.case_id for prefix in family_prefixes],
            "scores": scores,
            "sensitivity": sensitivities,
            "layer_max_deltas": layer_deltas[family],
            "nulls": {
                "shuffled_head_accuracy": torch.stack(shuffled_accs, dim=0),
                "label_permutation_accuracy": torch.stack(label_perm_accs, dim=0),
                "random_direction_accuracy": torch.stack(random_dir_accs, dim=0),
            },
        }
    return family_results


def rank_channels(result: dict[str, object], config: ChannelProbeConfig) -> dict[str, object]:
    ranked: dict[str, object] = {}
    for family, family_result in result.items():
        scores = family_result["scores"]
        sensitivities = family_result["sensitivity"]
        nulls = family_result["nulls"]

        mean_score = scores.mean(dim=0)
        sign_accuracy = (scores > 0).float().mean(dim=0)
        mean_sensitivity = sensitivities.mean(dim=0)

        shuffled_mean = nulls["shuffled_head_accuracy"].mean(dim=0)
        label_perm_mean = nulls["label_permutation_accuracy"].mean(dim=0)
        random_dir_mean = nulls["random_direction_accuracy"].mean(dim=0)
        null_ceiling = torch.stack((shuffled_mean, label_perm_mean, random_dir_mean), dim=0).max(dim=0).values

        rows: list[dict[str, object]] = []
        n_layers, n_heads = mean_score.shape
        for layer in range(n_layers):
            for head in range(n_heads):
                acc = float(sign_accuracy[layer, head].item())
                null_max = float(null_ceiling[layer, head].item())
                rows.append(
                    {
                        "family": family,
                        "layer": layer,
                        "head": head,
                        "mean_score": float(mean_score[layer, head].item()),
                        "heldout_sign_accuracy": acc,
                        "mean_position_sensitivity": float(mean_sensitivity[layer, head].item()),
                        "null_accuracy_mean": {
                            "shuffled_head": float(shuffled_mean[layer, head].item()),
                            "label_permutation": float(label_perm_mean[layer, head].item()),
                            "random_direction": float(random_dir_mean[layer, head].item()),
                        },
                        "null_accuracy_ceiling": null_max,
                        "passes_gate": bool(
                            acc >= config.promotion_threshold and acc >= null_max + config.null_margin
                        ),
                    }
                )

        rows.sort(
            key=lambda item: (
                item["passes_gate"],
                item["heldout_sign_accuracy"],
                abs(item["mean_score"]),
                item["mean_position_sensitivity"],
            ),
            reverse=True,
        )
        ranked[family] = {
            "case_ids": family_result["case_ids"],
            "layer_max_deltas": family_result["layer_max_deltas"],
            "median_layer_max_delta": float(torch.tensor(family_result["layer_max_deltas"]).median().item()),
            "rows": rows,
        }
    return ranked


def promote_channels(ranking: dict[str, object], threshold: float) -> list[dict[str, object]]:
    promoted: list[dict[str, object]] = []
    for family, family_ranking in ranking.items():
        for row in family_ranking["rows"]:
            if row["passes_gate"] and row["heldout_sign_accuracy"] >= threshold:
                promoted.append(
                    {
                        "family": family,
                        "layer": row["layer"],
                        "head": row["head"],
                        "heldout_sign_accuracy": row["heldout_sign_accuracy"],
                        "mean_score": row["mean_score"],
                        "mean_position_sensitivity": row["mean_position_sensitivity"],
                    }
                )
    promoted.sort(key=lambda item: (item["heldout_sign_accuracy"], abs(item["mean_score"])), reverse=True)
    return promoted


def _report_text(artifact: RunArtifact) -> str:
    lines = [
        "# Channel Probe Report",
        "",
        f"Run id: `{artifact.run_id}`",
        f"Model: `{artifact.model_id}`",
        "",
        "## Promotion Gate",
        "",
        f"- held-out sign accuracy threshold: `{artifact.metadata['promotion_threshold']:.2f}`",
        f"- null margin: `{artifact.metadata['null_margin']:.2f}`",
        f"- null samples: `{artifact.metadata['null_samples']}`",
        "",
        "## Tokenization Filter",
        "",
        f"- skipped invalid cases: `{len(artifact.metadata.get('invalid_cases', []))}`",
        "",
        "## Family Summaries",
        "",
    ]
    for family_summary in artifact.metadata["family_summaries"]:
        lines.append(f"### {family_summary['family']}")
        lines.append("")
        lines.append(f"- cases: {', '.join(family_summary['case_ids'])}")
        lines.append(f"- median x_t layer delta: {family_summary['median_layer_max_delta']:.6f}")
        lines.append(f"- promoted channels: {family_summary['n_promoted']}")
        lines.append(f"- top channel: L{family_summary['top_row']['layer']} H{family_summary['top_row']['head']}")
        lines.append(
            f"- top held-out sign accuracy: {family_summary['top_row']['heldout_sign_accuracy']:.3f}"
        )
        lines.append(
            f"- top null ceiling: {family_summary['top_row']['null_accuracy_ceiling']:.3f}"
        )
        lines.append("")
    if artifact.results:
        lines.extend(["## Promoted Channels", ""])
        for row in artifact.results:
            lines.append(
                f"- `{row['family']}` L{row['layer']} H{row['head']}: "
                f"held-out acc={row['heldout_sign_accuracy']:.3f}, "
                f"mean_score={row['mean_score']:+.3f}, "
                f"position_sensitivity={row['mean_position_sensitivity']:+.3f}"
            )
    else:
        lines.extend(["## Promoted Channels", "", "- none in this run"])
    return "\n".join(lines) + "\n"


def _payload_from_artifact(artifact: RunArtifact, ranking: dict[str, object]) -> dict[str, object]:
    return {
        "artifact": asdict(artifact),
        "ranking": ranking,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase 1 per-channel probing")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[f"{label}={path}" for label, path in DEFAULT_MODEL_PATHS.items() if label == "C-71"],
    )
    parser.add_argument(
        "--task-names",
        default="coreference,induction,recency_bias",
        help="Comma-separated task family names to probe.",
    )
    parser.add_argument("--tokenizer-mode", default="gpt2", choices=("auto", "gpt2", "reduced_gpt2"))
    parser.add_argument("--base-tokenizer", default=DEFAULT_BASE_TOKENIZER)
    parser.add_argument("--vocab-file", default=DEFAULT_VOCAB_FILE)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--null-samples", type=int, default=64)
    parser.add_argument("--promotion-threshold", type=float, default=0.70)
    parser.add_argument("--null-margin", type=float, default=0.10)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--output", default="runs/channel_probe_phase1")
    args = parser.parse_args(argv)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    task_names = tuple(name.strip() for name in args.task_names.split(",") if name.strip())

    for model_label, checkpoint_path in parse_model_specs(args.models):
        config = ChannelProbeConfig(
            model_label=model_label,
            checkpoint_path=checkpoint_path,
            task_names=task_names,
            tokenizer_mode=args.tokenizer_mode,
            base_tokenizer=args.base_tokenizer,
            vocab_file=args.vocab_file,
            device=args.device,
            null_samples=args.null_samples,
            promotion_threshold=args.promotion_threshold,
            null_margin=args.null_margin,
            random_seed=args.random_seed,
        )
        print(f"[channel-probe] loading {model_label} from {checkpoint_path}", flush=True)
        model, checkpoint = load_symbolic_model(checkpoint_path, device=device)
        tokenizer = build_tokenizer(
            tokenizer_mode=args.tokenizer_mode,
            base_tokenizer_path=args.base_tokenizer,
            vocab_file=args.vocab_file,
            checkpoint_config=checkpoint["config"],
        )
        prefixes, invalid_cases = task_case_prefixes(set(task_names), tokenizer, skip_invalid=True)
        print(
            f"[channel-probe] probing {model_label}: {len(prefixes)} valid cases "
            f"({len(invalid_cases)} skipped) across {len(task_names)} families",
            flush=True,
        )
        probe_result = probe_channel_preferences(model, prefixes, config)
        ranking = rank_channels(probe_result, config)
        promoted = promote_channels(ranking, config.promotion_threshold)

        family_summaries = []
        for family, family_ranking in ranking.items():
            rows = family_ranking["rows"]
            family_summaries.append(
                {
                    "family": family,
                    "case_ids": family_ranking["case_ids"],
                    "median_layer_max_delta": family_ranking["median_layer_max_delta"],
                    "n_promoted": sum(1 for row in rows if row["passes_gate"]),
                    "top_row": rows[0],
                }
            )

        artifact = RunArtifact(
            run_id=output_dir.name,
            model_id=model_label,
            phase="steering",
            results=promoted,
            metadata={
                "phase_name": "channel_probe_phase1",
                "checkpoint_path": checkpoint_path,
                "task_names": list(task_names),
                "promotion_threshold": args.promotion_threshold,
                "null_margin": args.null_margin,
                "null_samples": args.null_samples,
                "tokenizer_mode": args.tokenizer_mode,
                "invalid_cases": invalid_cases,
                "family_summaries": family_summaries,
            },
        )
        payload = _payload_from_artifact(artifact, ranking)
        save_json(payload, output_dir / f"{model_label.lower().replace('-', '_')}_channel_probe.json")
        (output_dir / f"{model_label.lower().replace('-', '_')}_report.md").write_text(
            _report_text(artifact),
            encoding="utf-8",
        )
        print(
            f"[channel-probe] {model_label}: {len(promoted)} promoted channels; "
            f"best families={[summary['family'] for summary in family_summaries]}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
