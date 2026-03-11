#!/usr/bin/env python3
"""Run direct vocabulary steering on small symbolic-transformer checkpoints.

This script is intentionally self-contained because the target checkpoints live
outside this repo and use a reduced GPT-2 vocabulary.  It loads one model at a
time, applies an exact vocabulary steering direction to the contextual stream
`x_e` after a chosen block, and measures both local first-divergent-token
effects and total choice-score effects on a small benchmark panel.
"""

from __future__ import annotations

import argparse
import contextlib
import enum
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from transformers import GPT2TokenizerFast

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SYMBOLIC_SRC = Path("companion-repo/neuro-symb-v2/src")
if str(SYMBOLIC_SRC) not in sys.path:
    sys.path.insert(0, str(SYMBOLIC_SRC))

from gpt_oss_interp.benchmarks.tasks import all_tasks
from symbolic_transformer.model.config import SymbolicTransformerConfig
from symbolic_transformer.model.transformer import SymbolicTransformer


DEFAULT_MODELS = (
    "E2_independent=/mnt/d/data/neuro-symb-v2/backup-data/experiments/"
    "cog-sys-paper-1/series_e_25K_test/E2_independent/checkpoint_epoch_2.pt",
    "E1_dense=/mnt/d/data/neuro-symb-v2/backup-data/experiments/"
    "cog-sys-paper-1/series_e_25K_test/E1_dense/checkpoint_epoch_2.pt",
)
DEFAULT_CASE_IDS = ("caps_005", "induction_009", "coref_010")
DEFAULT_VOCAB_FILE = "companion-repo/neuro-symb-v2/neurips-2026/data/grade_school_vocab.pkl"
DEFAULT_BASE_TOKENIZER = (
    "hf-cache/models--gpt2/snapshots/"
    "607a30d783dfa663caf39e06633721c8d4cfcd7e"
)


@dataclass
class ChoicePrefix:
    case_id: str
    prompt: str
    choice_a: str
    choice_b: str
    full_a: list[int]
    full_b: list[int]
    prompt_ids: list[int]
    prefix_ids: list[int]
    decision_position_index: int
    prompt_start_index: int
    choice_a_token: int
    choice_b_token: int
    suffix_a: list[int]
    suffix_b: list[int]


class ReducedGPT2Tokenizer:
    """Map GPT-2 tokenization into the reduced grade-school vocabulary."""

    def __init__(self, vocab_file: str, base_tokenizer_path: str):
        with open(vocab_file, "rb") as handle:
            import pickle

            vocab_data = pickle.load(handle)

        self.base_tokenizer = GPT2TokenizerFast.from_pretrained(
            base_tokenizer_path,
            local_files_only=True,
        )
        self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
        self.used_tokens = vocab_data["used_tokens"]
        self.token_to_id = vocab_data["token_to_id"]
        self.id_to_token = vocab_data["id_to_token"]
        self.vocab_size = vocab_data["vocab_size"]
        self.unk_id = vocab_data.get("unk_id", 0)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        base_tokens = self.base_tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return [self.token_to_id.get(token_id, self.unk_id) for token_id in base_tokens]

    def decode(self, token_ids: list[int]) -> str:
        base_ids = [self.id_to_token.get(token_id, self.base_tokenizer.unk_token_id or 0) for token_id in token_ids]
        return self.base_tokenizer.decode(base_ids)

    def __len__(self) -> int:
        return self.vocab_size


class RawGPT2Tokenizer:
    """Thin wrapper around GPT-2 tokenization with the same interface."""

    def __init__(self, base_tokenizer_path: str):
        self.base_tokenizer = GPT2TokenizerFast.from_pretrained(
            base_tokenizer_path,
            local_files_only=True,
        )
        self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
        self.vocab_size = len(self.base_tokenizer)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self.base_tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, token_ids: list[int]) -> str:
        return self.base_tokenizer.decode(token_ids)

    def __len__(self) -> int:
        return self.vocab_size


def _parse_csv(values: str, cast):
    return [cast(item.strip()) for item in values.split(",") if item.strip()]


def _parse_models(model_args: list[str]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for item in model_args:
        if "=" not in item:
            raise ValueError(f"Model entry must be label=path, got: {item}")
        label, path = item.split("=", 1)
        parsed.append((label.strip(), path.strip()))
    return parsed


def _find_case(case_id: str):
    for task in all_tasks():
        for case in task.cases:
            if case.case_id == case_id:
                return task, case
    raise KeyError(f"Unknown case_id: {case_id}")


def _choice_prefix(tokenizer: ReducedGPT2Tokenizer, case_id: str, prompt: str, choice_a: str, choice_b: str) -> ChoicePrefix:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_a = tokenizer.encode(prompt + choice_a, add_special_tokens=False)
    full_b = tokenizer.encode(prompt + choice_b, add_special_tokens=False)
    if full_a[: len(prompt_ids)] != prompt_ids or full_b[: len(prompt_ids)] != prompt_ids:
        raise ValueError(f"{case_id}: prompt tokens are not a strict prefix of full choice tokenization")

    comp_a = full_a[len(prompt_ids) :]
    comp_b = full_b[len(prompt_ids) :]
    diff_idx = None
    for idx, (tok_a, tok_b) in enumerate(zip(comp_a, comp_b)):
        if tok_a != tok_b:
            diff_idx = idx
            break
    if diff_idx is None:
        raise ValueError(f"{case_id}: choices do not diverge in tokenization")

    prefix_ids = full_a[: len(prompt_ids) + diff_idx]
    suffix_a = comp_a[diff_idx + 1 :]
    suffix_b = comp_b[diff_idx + 1 :]
    return ChoicePrefix(
        case_id=case_id,
        prompt=prompt,
        choice_a=choice_a,
        choice_b=choice_b,
        full_a=full_a,
        full_b=full_b,
        prompt_ids=prompt_ids,
        prefix_ids=prefix_ids,
        decision_position_index=len(prefix_ids) - 1,
        prompt_start_index=len(prompt_ids),
        choice_a_token=comp_a[diff_idx],
        choice_b_token=comp_b[diff_idx],
        suffix_a=suffix_a,
        suffix_b=suffix_b,
    )


def _load_model(checkpoint_path: str, device: torch.device) -> tuple[SymbolicTransformer, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = SymbolicTransformerConfig.from_dict(checkpoint["config"])
    model = SymbolicTransformer(config)
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    allowed_missing = {"ln_embed.weight", "ln_embed.bias"}
    disallowed_missing = set(missing) - allowed_missing
    if disallowed_missing or unexpected:
        raise RuntimeError(
            "Checkpoint compatibility error: "
            f"missing={sorted(disallowed_missing)} unexpected={sorted(unexpected)}"
        )
    model.to(device)
    model.eval()
    return model, checkpoint


def _build_tokenizer(
    tokenizer_mode: str,
    vocab_file: str,
    base_tokenizer_path: str,
    checkpoint_config: dict[str, object] | None = None,
):
    if tokenizer_mode == "gpt2":
        return RawGPT2Tokenizer(base_tokenizer_path)
    if tokenizer_mode == "reduced_gpt2":
        return ReducedGPT2Tokenizer(vocab_file, base_tokenizer_path)
    if tokenizer_mode != "auto":
        raise ValueError(f"Unsupported tokenizer mode: {tokenizer_mode}")

    vocab_size = None
    if checkpoint_config is not None:
        vocab_size = checkpoint_config.get("vocab_size")
    if vocab_size == 50257:
        return RawGPT2Tokenizer(base_tokenizer_path)
    return ReducedGPT2Tokenizer(vocab_file, base_tokenizer_path)


@contextlib.contextmanager
def _contextual_steering_hook(
    model: SymbolicTransformer,
    layer_idx: int | None,
    position_idx: int | None,
    vector: torch.Tensor | None,
) -> Iterator[None]:
    if layer_idx is None or position_idx is None or vector is None:
        yield
        return

    block = model.blocks[layer_idx]
    vector = vector.detach()

    def _hook(_module, _inputs, outputs):
        x_t, x_e, interpretations, present = outputs
        steered = x_e.clone()
        steered[:, position_idx, :] = steered[:, position_idx, :] + vector.to(
            device=steered.device,
            dtype=steered.dtype,
        )
        return x_t, steered, interpretations, present

    handle = block.register_forward_hook(_hook)
    try:
        yield
    finally:
        handle.remove()


def _forward_logits(
    model: SymbolicTransformer,
    input_ids: list[int],
    device: torch.device,
    layer_idx: int | None = None,
    position_idx: int | None = None,
    vector: torch.Tensor | None = None,
) -> torch.Tensor:
    tokens = torch.tensor([input_ids], device=device, dtype=torch.long)
    with torch.inference_mode():
        with _contextual_steering_hook(model, layer_idx, position_idx, vector):
            logits, _interpretations, _present = model(tokens)
    return logits[0].detach().cpu().float()


def _choice_score(
    model: SymbolicTransformer,
    full_ids: list[int],
    choice_start_idx: int,
    device: torch.device,
    layer_idx: int | None = None,
    position_idx: int | None = None,
    vector: torch.Tensor | None = None,
) -> float:
    logits = _forward_logits(
        model,
        full_ids,
        device=device,
        layer_idx=layer_idx,
        position_idx=position_idx,
        vector=vector,
    )
    log_probs = torch.log_softmax(logits, dim=-1)
    total = 0.0
    for token_pos in range(choice_start_idx, len(full_ids)):
        total += float(log_probs[token_pos - 1, full_ids[token_pos]].item())
    return total


def _local_metrics(
    model: SymbolicTransformer,
    prefix: ChoicePrefix,
    device: torch.device,
    layer_idx: int | None = None,
    vector: torch.Tensor | None = None,
) -> dict[str, float]:
    logits = _forward_logits(
        model,
        prefix.prefix_ids,
        device=device,
        layer_idx=layer_idx,
        position_idx=prefix.decision_position_index,
        vector=vector,
    )
    final_logits = logits[prefix.decision_position_index]
    final_log_probs = torch.log_softmax(final_logits, dim=-1)
    return {
        "logit_A": float(final_logits[prefix.choice_a_token].item()),
        "logit_B": float(final_logits[prefix.choice_b_token].item()),
        "local_logit_gap": float((final_logits[prefix.choice_a_token] - final_logits[prefix.choice_b_token]).item()),
        "local_logprob_gap": float((final_log_probs[prefix.choice_a_token] - final_log_probs[prefix.choice_b_token]).item()),
    }


def _baseline_case_summary(
    model: SymbolicTransformer,
    prefix: ChoicePrefix,
    expected_label: str,
    tokenizer: ReducedGPT2Tokenizer,
    device: torch.device,
) -> dict[str, object]:
    local = _local_metrics(model, prefix, device=device)
    choice_a_score = _choice_score(
        model,
        prefix.full_a,
        choice_start_idx=prefix.prompt_start_index,
        device=device,
    )
    choice_b_score = _choice_score(
        model,
        prefix.full_b,
        choice_start_idx=prefix.prompt_start_index,
        device=device,
    )
    predicted = "A" if choice_a_score >= choice_b_score else "B"
    return {
        "expected_label": expected_label,
        "predicted_label": predicted,
        "correct": int(predicted == expected_label),
        "choice_scores": {"A": choice_a_score, "B": choice_b_score},
        "total_gap": choice_a_score - choice_b_score,
        "local": local,
        "decision": {
            "choice_a_token_id": prefix.choice_a_token,
            "choice_b_token_id": prefix.choice_b_token,
            "choice_a_token_text": tokenizer.decode([prefix.choice_a_token]),
            "choice_b_token_text": tokenizer.decode([prefix.choice_b_token]),
            "decision_position_index": prefix.decision_position_index,
            "shared_suffix_a": [tokenizer.decode([tok]) for tok in prefix.suffix_a],
            "shared_suffix_b": [tokenizer.decode([tok]) for tok in prefix.suffix_b],
        },
    }


def _steering_rows_for_pair(
    model: SymbolicTransformer,
    tokenizer: ReducedGPT2Tokenizer,
    model_label: str,
    source_prefix: ChoicePrefix,
    target_prefix: ChoicePrefix,
    baseline_target: dict[str, object],
    layers: list[int],
    scales: list[float],
    device: torch.device,
) -> list[dict[str, object]]:
    weight = model.lm_head.weight.detach().cpu().float()
    direction = weight[source_prefix.choice_a_token] - weight[source_prefix.choice_b_token]
    rows: list[dict[str, object]] = []
    for layer_idx in layers:
        for scale in scales:
            vector = direction * scale
            local = _local_metrics(
                model,
                target_prefix,
                device=device,
                layer_idx=layer_idx,
                vector=vector,
            )
            score_a = _choice_score(
                model,
                target_prefix.full_a,
                choice_start_idx=target_prefix.prompt_start_index,
                device=device,
                layer_idx=layer_idx,
                position_idx=target_prefix.decision_position_index,
                vector=vector,
            )
            score_b = _choice_score(
                model,
                target_prefix.full_b,
                choice_start_idx=target_prefix.prompt_start_index,
                device=device,
                layer_idx=layer_idx,
                position_idx=target_prefix.decision_position_index,
                vector=vector,
            )
            total_gap = score_a - score_b
            local_logit_shift = local["local_logit_gap"] - baseline_target["local"]["local_logit_gap"]
            local_logprob_shift = local["local_logprob_gap"] - baseline_target["local"]["local_logprob_gap"]
            total_shift = total_gap - baseline_target["total_gap"]
            tail_shift = total_shift - local_logprob_shift
            rows.append(
                {
                    "model": model_label,
                    "source_case_id": source_prefix.case_id,
                    "target_case_id": target_prefix.case_id,
                    "on_target": int(source_prefix.case_id == target_prefix.case_id),
                    "layer": layer_idx,
                    "scale": scale,
                    "direction_norm": float(torch.norm(direction).item()),
                    "choice_a_token_id": target_prefix.choice_a_token,
                    "choice_b_token_id": target_prefix.choice_b_token,
                    "choice_a_token_text": tokenizer.decode([target_prefix.choice_a_token]),
                    "choice_b_token_text": tokenizer.decode([target_prefix.choice_b_token]),
                    "baseline_local_logit_gap": baseline_target["local"]["local_logit_gap"],
                    "intervened_local_logit_gap": local["local_logit_gap"],
                    "local_logit_shift": local_logit_shift,
                    "baseline_local_logprob_gap": baseline_target["local"]["local_logprob_gap"],
                    "intervened_local_logprob_gap": local["local_logprob_gap"],
                    "local_logprob_shift": local_logprob_shift,
                    "baseline_total_gap": baseline_target["total_gap"],
                    "intervened_total_gap": total_gap,
                    "total_shift": total_shift,
                    "tail_shift": tail_shift,
                    "tail_fraction": abs(tail_shift) / max(abs(total_shift), 1e-8),
                    "intervened_choice_scores": {"A": score_a, "B": score_b},
                }
            )
    return rows


def _summarize_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {"models": {}}
    by_model: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)

    for model, model_rows in by_model.items():
        on_target = [row for row in model_rows if row["on_target"]]
        by_case: dict[str, list[dict[str, object]]] = {}
        for row in on_target:
            by_case.setdefault(str(row["target_case_id"]), []).append(row)
        case_summaries: dict[str, object] = {}
        for case_id, case_rows in by_case.items():
            best = max(case_rows, key=lambda item: abs(float(item["local_logit_shift"])))
            best_total = max(case_rows, key=lambda item: abs(float(item["total_shift"])))
            sign_correct = [
                row
                for row in case_rows
                if (float(row["scale"]) > 0 and float(row["local_logit_shift"]) > 0)
                or (float(row["scale"]) < 0 and float(row["local_logit_shift"]) < 0)
            ]
            case_summaries[case_id] = {
                "n_runs": len(case_rows),
                "best_local_shift": {
                    "layer": best["layer"],
                    "scale": best["scale"],
                    "local_logit_shift": best["local_logit_shift"],
                    "tail_fraction": best["tail_fraction"],
                },
                "best_total_shift": {
                    "layer": best_total["layer"],
                    "scale": best_total["scale"],
                    "total_shift": best_total["total_shift"],
                    "tail_fraction": best_total["tail_fraction"],
                },
                "sign_correct_fraction": len(sign_correct) / len(case_rows) if case_rows else 0.0,
            }
        summary["models"][model] = {
            "n_rows": len(model_rows),
            "n_on_target_rows": len(on_target),
            "cases": case_summaries,
        }
    return summary


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, enum.Enum):
        return value.value
    if hasattr(value, "__dict__"):
        return _json_ready(vars(value))
    return value


def _write_markdown_report(
    output_path: Path,
    payload: dict[str, object],
) -> None:
    lines = [
        "# Direct Vocabulary Steering Report",
        "",
        f"Output root: `{output_path.parent}`",
        "",
        "## Models",
        "",
    ]
    for model_info in payload["models"]:
        lines.append(f"- `{model_info['label']}`: `{model_info['checkpoint']}`")

    lines.extend(["", "## Baselines", ""])
    for model_info in payload["models"]:
        lines.append(f"### {model_info['label']}")
        lines.append("")
        for case_id, info in model_info["baselines"].items():
            lines.append(
                f"- `{case_id}`: predicted `{info['predicted_label']}`, expected `{info['expected_label']}`, "
                f"correct={info['correct']}, total_gap={info['total_gap']:.4f}, "
                f"local_logit_gap={info['local']['local_logit_gap']:.4f}"
            )
        lines.append("")

    lines.extend(["## Strongest On-Target Effects", ""])
    for model_label, model_summary in payload["summary"]["models"].items():
        lines.append(f"### {model_label}")
        lines.append("")
        for case_id, case_summary in model_summary["cases"].items():
            best = case_summary["best_local_shift"]
            lines.append(
                f"- `{case_id}`: best local shift at layer {best['layer']} scale {best['scale']} "
                f"=> {best['local_logit_shift']:.4f} (tail_fraction={best['tail_fraction']:.3f}); "
                f"sign_correct_fraction={case_summary['sign_correct_fraction']:.3f}"
            )
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- `tail_fraction` is decomposed using local log-prob gap shift so it stays in the same units as total choice-score shift.")
    lines.append("- Off-target rows are present in the JSON artifact when source and target case differ.")
    output_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run direct vocabulary steering on symbolic checkpoints")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), help="Model specs in label=path form")
    parser.add_argument("--case-ids", default=",".join(DEFAULT_CASE_IDS))
    parser.add_argument("--layers", default="0,1,2,3,4,5")
    parser.add_argument("--scales", default="-2,-1,-0.5,0.5,1,2")
    parser.add_argument("--vocab-file", default=DEFAULT_VOCAB_FILE)
    parser.add_argument("--base-tokenizer", default=DEFAULT_BASE_TOKENIZER)
    parser.add_argument(
        "--tokenizer-mode",
        default="auto",
        choices=("auto", "gpt2", "reduced_gpt2"),
        help="Choose GPT-2 tokenization directly or reduced-vocab mapping.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-off-target", action="store_true")
    parser.add_argument(
        "--output",
        default="runs/direct_vocab_steering",
        help="Output directory for JSON and markdown artifacts",
    )
    args = parser.parse_args()

    if args.tokenizer_mode != "gpt2" and not Path(args.vocab_file).exists():
        raise FileNotFoundError(f"Vocab file not found: {args.vocab_file}")
    if not Path(args.base_tokenizer).exists():
        raise FileNotFoundError(f"Base tokenizer snapshot not found: {args.base_tokenizer}")

    device = torch.device(args.device)
    case_ids = _parse_csv(args.case_ids, str)
    layers = _parse_csv(args.layers, int)
    scales = _parse_csv(args.scales, float)
    model_specs = _parse_models(args.models)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, object] = {
        "config": {
            "models": args.models,
            "case_ids": case_ids,
            "layers": layers,
            "scales": scales,
            "vocab_file": args.vocab_file,
            "base_tokenizer": args.base_tokenizer,
            "tokenizer_mode": args.tokenizer_mode,
            "device": str(device),
            "skip_off_target": args.skip_off_target,
        },
        "cases": {},
        "models": [],
        "rows": [],
    }

    global_task_map: dict[str, dict[str, object]] = {}
    for model_label, checkpoint_path in model_specs:
        print(f"[direct-vocab] loading {model_label} from {checkpoint_path}", flush=True)
        model, checkpoint = _load_model(checkpoint_path, device=device)
        tokenizer = _build_tokenizer(
            tokenizer_mode=args.tokenizer_mode,
            vocab_file=args.vocab_file,
            base_tokenizer_path=args.base_tokenizer,
            checkpoint_config=checkpoint["config"],
        )
        print(
            f"[direct-vocab] tokenizer for {model_label}: "
            f"{tokenizer.__class__.__name__} (vocab={len(tokenizer)})",
            flush=True,
        )

        prefixes: dict[str, ChoicePrefix] = {}
        task_map: dict[str, dict[str, object]] = {}
        for case_id in case_ids:
            task, case = _find_case(case_id)
            prefix = _choice_prefix(
                tokenizer,
                case_id=case.case_id,
                prompt=case.prompt,
                choice_a=case.choices["A"],
                choice_b=case.choices["B"],
            )
            if prefix.choice_a_token == prefix.choice_b_token:
                raise ValueError(f"{model_label} {case_id}: tokenizer collapses the divergent token pair")
            prefixes[case_id] = prefix
            task_map[case_id] = {
                "task_name": task.name,
                "behavior": task.behavior,
                "prompt": case.prompt,
                "choices": case.choices,
                "expected_label": case.expected_label,
            }
            global_task_map.setdefault(case_id, task_map[case_id])

        baselines: dict[str, object] = {}
        for case_id in case_ids:
            baselines[case_id] = _baseline_case_summary(
                model=model,
                prefix=prefixes[case_id],
                expected_label=str(task_map[case_id]["expected_label"]),
                tokenizer=tokenizer,
                device=device,
            )
            print(
                f"[direct-vocab] baseline {model_label} {case_id}: "
                f"pred={baselines[case_id]['predicted_label']} correct={baselines[case_id]['correct']}",
                flush=True,
            )

        source_ids = case_ids
        target_ids = case_ids if not args.skip_off_target else case_ids
        total_runs = len(source_ids) * (len(target_ids) if not args.skip_off_target else 1) * len(layers) * len(scales)
        completed_runs = 0
        for source_case_id in source_ids:
            for target_case_id in target_ids:
                if args.skip_off_target and source_case_id != target_case_id:
                    continue
                rows = _steering_rows_for_pair(
                    model=model,
                    tokenizer=tokenizer,
                    model_label=model_label,
                    source_prefix=prefixes[source_case_id],
                    target_prefix=prefixes[target_case_id],
                    baseline_target=baselines[target_case_id],
                    layers=layers,
                    scales=scales,
                    device=device,
                )
                payload["rows"].extend(rows)
                completed_runs += len(rows)
                print(
                    f"[direct-vocab] {model_label}: {source_case_id} -> {target_case_id} "
                    f"completed ({completed_runs}/{total_runs} rows)",
                    flush=True,
                )

        payload["models"].append(
            {
                "label": model_label,
                "checkpoint": checkpoint_path,
                "epoch": checkpoint.get("epoch"),
                "loss": checkpoint.get("loss"),
                "config": checkpoint["config"],
                "tokenizer_class": tokenizer.__class__.__name__,
                "baselines": baselines,
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload["cases"] = {case_id: global_task_map[case_id] for case_id in case_ids}
    payload["summary"] = _summarize_rows(payload["rows"])
    (output_dir / "direct_vocab_steering.json").write_text(json.dumps(_json_ready(payload), indent=2))
    _write_markdown_report(output_dir / "report.md", payload)
    print(f"[direct-vocab] wrote {output_dir / 'direct_vocab_steering.json'}", flush=True)
    print(f"[direct-vocab] wrote {output_dir / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
