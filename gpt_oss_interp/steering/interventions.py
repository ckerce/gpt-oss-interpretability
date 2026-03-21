"""Shared symbolic-model helpers for steering and probing workflows."""

from __future__ import annotations

import contextlib
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from transformers import GPT2TokenizerFast

REPO_ROOT = Path(__file__).resolve().parents[2]
SYMBOLIC_SRC = Path("companion-repo/neuro-symb-v2/src")
if str(SYMBOLIC_SRC) not in sys.path:
    sys.path.insert(0, str(SYMBOLIC_SRC))

from gpt_oss_interp.benchmarks.tasks import all_tasks
from symbolic_transformer.model.config import StreamUpdateMode, SymbolicTransformerConfig
from symbolic_transformer.model.transformer import SymbolicTransformer

DEFAULT_BASE_TOKENIZER = (
    "hf-cache/models--gpt2/snapshots/"
    "607a30d783dfa663caf39e06633721c8d4cfcd7e"
)
DEFAULT_VOCAB_FILE = "companion-repo/neuro-symb-v2/neurips-2026/data/grade_school_vocab.pkl"
DEFAULT_MODEL_PATHS = {
    "SS-71": (
        "/mnt/d/mechanistic_interpretability/results/neurips-2026/training/"
        "gated_attention/dns-dns_S_G_gpt2/checkpoint_epoch_1.pt"
    ),
    "C-71": (
        "/mnt/d/mechanistic_interpretability/results/neurips-2026/training/"
        "gated_attention/dns-dns_S_G_cascade_gpt2/checkpoint_epoch_1.pt"
    ),
}


@dataclass
class ChoicePrefix:
    case_id: str
    task_name: str
    prompt: str
    choice_a: str
    choice_b: str
    expected_label: str
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
            vocab_data = pickle.load(handle)

        self.base_tokenizer = GPT2TokenizerFast.from_pretrained(
            base_tokenizer_path,
            local_files_only=True,
        )
        self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
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


def parse_model_specs(entries: list[str]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Model entry must be label=path, got: {entry}")
        label, path = entry.split("=", 1)
        parsed.append((label.strip(), path.strip()))
    return parsed


def parse_csv(values: str, cast):
    return [cast(item.strip()) for item in values.split(",") if item.strip()]


def load_symbolic_model(checkpoint_path: str, device: torch.device) -> tuple[SymbolicTransformer, dict[str, object]]:
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


def build_tokenizer(
    tokenizer_mode: str,
    base_tokenizer_path: str,
    vocab_file: str = DEFAULT_VOCAB_FILE,
    checkpoint_config: dict[str, object] | None = None,
):
    if tokenizer_mode == "gpt2":
        return RawGPT2Tokenizer(base_tokenizer_path)
    if tokenizer_mode == "reduced_gpt2":
        return ReducedGPT2Tokenizer(vocab_file, base_tokenizer_path)
    if tokenizer_mode != "auto":
        raise ValueError(f"Unsupported tokenizer mode: {tokenizer_mode}")

    vocab_size = checkpoint_config.get("vocab_size") if checkpoint_config is not None else None
    if vocab_size == 50257:
        return RawGPT2Tokenizer(base_tokenizer_path)
    return ReducedGPT2Tokenizer(vocab_file, base_tokenizer_path)


def find_case(case_id: str):
    for task in all_tasks():
        for case in task.cases:
            if case.case_id == case_id:
                return task, case
    raise KeyError(f"Unknown case_id: {case_id}")


def task_case_prefixes(
    task_names: set[str],
    tokenizer,
    *,
    skip_invalid: bool = False,
) -> tuple[list[ChoicePrefix], list[dict[str, str]]]:
    prefixes: list[ChoicePrefix] = []
    invalid: list[dict[str, str]] = []
    for task in all_tasks():
        if task.name not in task_names:
            continue
        for case in task.cases:
            try:
                prefixes.append(
                    build_choice_prefix(
                        tokenizer=tokenizer,
                        task_name=task.name,
                        case_id=case.case_id,
                        prompt=case.prompt,
                        choice_a=case.choices["A"],
                        choice_b=case.choices["B"],
                        expected_label=str(case.expected_label),
                    )
                )
            except ValueError as exc:
                if not skip_invalid:
                    raise
                invalid.append(
                    {
                        "task_name": task.name,
                        "case_id": case.case_id,
                        "reason": str(exc),
                    }
                )
    return prefixes, invalid


def build_choice_prefix(
    tokenizer,
    task_name: str,
    case_id: str,
    prompt: str,
    choice_a: str,
    choice_b: str,
    expected_label: str,
) -> ChoicePrefix:
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
        task_name=task_name,
        prompt=prompt,
        choice_a=choice_a,
        choice_b=choice_b,
        expected_label=expected_label,
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


def capture_stream_trajectory(
    model: SymbolicTransformer,
    input_ids: list[int],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    tokens = torch.tensor([input_ids], device=device, dtype=torch.long)
    with torch.inference_mode():
        x_t = model.embedding(tokens)
        x_e = torch.zeros_like(x_t)

        if model.config.stream_update_mode == StreamUpdateMode.CASCADE and model.config.cascade_cln_at_init:
            x_t = model.ln_embed(x_t)
        embedding_x_t = x_t[0].detach().cpu().float()

        vocab_embeddings = model.embedding.get_vocab_embeddings()
        output_vocab_embeddings = None
        if model.config.experimental.vocab_projection.vocab_source.value == "output_embedding":
            output_vocab_embeddings = model.get_output_vocab_embeddings()

        layer_x_t: list[torch.Tensor] = []
        layer_x_e: list[torch.Tensor] = []
        for block in model.blocks:
            x_t, x_e, _interp, _present = block(
                x_t,
                x_e,
                vocab_embeddings,
                output_vocab_embeddings,
                layer_past=None,
                use_cache=False,
                training_step=None,
                attention_mask=None,
            )
            layer_x_t.append(x_t[0].detach().cpu().float())
            layer_x_e.append(x_e[0].detach().cpu().float())

    return {
        "embedding_x_t": embedding_x_t,
        "layer_x_t": torch.stack(layer_x_t, dim=0),
        "layer_x_e": torch.stack(layer_x_e, dim=0),
    }


def output_embedding_by_head(model: SymbolicTransformer) -> torch.Tensor:
    weight = model.lm_head.weight.detach().cpu().float()
    return weight.view(weight.shape[0], model.config.n_head, model.config.head_dim)


def direction_slices_for_case(prefix: ChoicePrefix, vocab_by_head: torch.Tensor) -> torch.Tensor:
    token_pos = prefix.choice_a_token if prefix.expected_label == "A" else prefix.choice_b_token
    token_neg = prefix.choice_b_token if prefix.expected_label == "A" else prefix.choice_a_token
    return vocab_by_head[token_pos] - vocab_by_head[token_neg]


def direct_gap_direction_slices(prefix: ChoicePrefix, vocab_by_head: torch.Tensor) -> torch.Tensor:
    """Return the A-vs-B direction used for local gap evaluation."""
    return vocab_by_head[prefix.choice_a_token] - vocab_by_head[prefix.choice_b_token]


def _apply_slice_to_head(
    target: torch.Tensor,
    position_idx: int,
    head_idx: int,
    vector_slice: torch.Tensor,
) -> torch.Tensor:
    steered = target.clone()
    view = steered.view(steered.shape[0], steered.shape[1], -1, vector_slice.shape[-1])
    view[:, position_idx, head_idx, :] = (
        view[:, position_idx, head_idx, :]
        + vector_slice.to(device=steered.device, dtype=steered.dtype)
    )
    return steered


def _apply_full_vector(
    target: torch.Tensor,
    position_idx: int,
    vector: torch.Tensor,
) -> torch.Tensor:
    steered = target.clone()
    steered[:, position_idx, :] = steered[:, position_idx, :] + vector.to(
        device=steered.device,
        dtype=steered.dtype,
    )
    return steered


@contextlib.contextmanager
def slice_intervention_hook(
    model: SymbolicTransformer,
    layer_idx: int,
    position_idx: int,
    head_idx: int,
    vector_slice: torch.Tensor,
    vector: torch.Tensor | None = None,
    *,
    stream: str = "x_t",
    stage: str = "post_block",
) -> Iterator[None]:
    block = model.blocks[layer_idx]

    if stage == "post_block":
        def _hook(_module, _inputs, outputs):
            x_t, x_e, interpretations, present = outputs
            if vector is not None:
                if stream == "x_e":
                    return x_t, _apply_full_vector(x_e, position_idx, vector), interpretations, present
                return _apply_full_vector(x_t, position_idx, vector), x_e, interpretations, present
            if stream == "x_e":
                return x_t, _apply_slice_to_head(x_e, position_idx, head_idx, vector_slice), interpretations, present
            return _apply_slice_to_head(x_t, position_idx, head_idx, vector_slice), x_e, interpretations, present

        handle = block.register_forward_hook(_hook)
    elif stage == "pre_block":
        def _pre_hook(_module, inputs):
            input_list = list(inputs)
            stream_idx = 1 if stream == "x_e" else 0
            if vector is not None:
                input_list[stream_idx] = _apply_full_vector(input_list[stream_idx], position_idx, vector)
            else:
                input_list[stream_idx] = _apply_slice_to_head(
                    input_list[stream_idx],
                    position_idx,
                    head_idx,
                    vector_slice,
                )
            return tuple(input_list)

        handle = block.register_forward_pre_hook(_pre_hook)
    else:
        raise ValueError(f"Unsupported stage: {stage}")

    try:
        yield
    finally:
        handle.remove()


def forward_logits(
    model: SymbolicTransformer,
    input_ids: list[int],
    device: torch.device,
    *,
    layer_idx: int | None = None,
    position_idx: int | None = None,
    head_idx: int | None = None,
    vector_slice: torch.Tensor | None = None,
    vector: torch.Tensor | None = None,
    stream: str = "x_t",
    stage: str = "post_block",
    readout_source: str = "combined",
) -> torch.Tensor:
    tokens = torch.tensor([input_ids], device=device, dtype=torch.long)

    with torch.inference_mode():
        if layer_idx is None or position_idx is None or (
            vector is None and (head_idx is None or vector_slice is None)
        ):
            logits, _interpretations, _present, streams = model(tokens, return_streams=True)
        else:
            with slice_intervention_hook(
                model,
                layer_idx=layer_idx,
                position_idx=position_idx,
                head_idx=0 if head_idx is None else head_idx,
                vector_slice=vector_slice if vector_slice is not None else torch.zeros(model.config.head_dim),
                vector=vector,
                stream=stream,
                stage=stage,
            ):
                logits, _interpretations, _present, streams = model(tokens, return_streams=True)

        if readout_source == "combined":
            output = logits[0]
        elif readout_source == "x_t":
            output = model.lm_head(model.ln_f(streams["x_t"]))[0]
        elif readout_source == "x_e":
            output = model.lm_head(model.ln_f(streams["x_e"]))[0]
        else:
            raise ValueError(f"Unsupported readout_source: {readout_source}")
    return output.detach().cpu().float()


def local_gap(
    model: SymbolicTransformer,
    prefix: ChoicePrefix,
    device: torch.device,
    *,
    layer_idx: int | None = None,
    head_idx: int | None = None,
    vector_slice: torch.Tensor | None = None,
    vector: torch.Tensor | None = None,
    stream: str = "x_t",
    stage: str = "post_block",
    readout_source: str = "combined",
) -> float:
    logits = forward_logits(
        model,
        prefix.prefix_ids,
        device,
        layer_idx=layer_idx,
        position_idx=prefix.decision_position_index,
        head_idx=head_idx,
        vector_slice=vector_slice,
        vector=vector,
        stream=stream,
        stage=stage,
        readout_source=readout_source,
    )
    final_logits = logits[prefix.decision_position_index]
    return float((final_logits[prefix.choice_a_token] - final_logits[prefix.choice_b_token]).item())


def choice_score(
    model: SymbolicTransformer,
    full_ids: list[int],
    choice_start_idx: int,
    device: torch.device,
    *,
    layer_idx: int | None = None,
    position_idx: int | None = None,
    head_idx: int | None = None,
    vector_slice: torch.Tensor | None = None,
    vector: torch.Tensor | None = None,
    stream: str = "x_t",
    stage: str = "post_block",
    readout_source: str = "combined",
) -> float:
    logits = forward_logits(
        model,
        full_ids,
        device,
        layer_idx=layer_idx,
        position_idx=position_idx,
        head_idx=head_idx,
        vector_slice=vector_slice,
        vector=vector,
        stream=stream,
        stage=stage,
        readout_source=readout_source,
    )
    log_probs = torch.log_softmax(logits, dim=-1)
    total = 0.0
    for token_pos in range(choice_start_idx, len(full_ids)):
        total += float(log_probs[token_pos - 1, full_ids[token_pos]].item())
    return total


def choice_gap(
    model: SymbolicTransformer,
    prefix: ChoicePrefix,
    device: torch.device,
    *,
    layer_idx: int | None = None,
    head_idx: int | None = None,
    vector_slice: torch.Tensor | None = None,
    vector: torch.Tensor | None = None,
    stream: str = "x_t",
    stage: str = "post_block",
    readout_source: str = "combined",
) -> float:
    score_a = choice_score(
        model,
        prefix.full_a,
        prefix.prompt_start_index,
        device,
        layer_idx=layer_idx,
        position_idx=prefix.decision_position_index,
        head_idx=head_idx,
        vector_slice=vector_slice,
        vector=vector,
        stream=stream,
        stage=stage,
        readout_source=readout_source,
    )
    score_b = choice_score(
        model,
        prefix.full_b,
        prefix.prompt_start_index,
        device,
        layer_idx=layer_idx,
        position_idx=prefix.decision_position_index,
        head_idx=head_idx,
        vector_slice=vector_slice,
        vector=vector,
        stream=stream,
        stage=stage,
        readout_source=readout_source,
    )
    return score_a - score_b
