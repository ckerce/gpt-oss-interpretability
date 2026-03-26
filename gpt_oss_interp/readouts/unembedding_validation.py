from __future__ import annotations

import argparse
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from gpt_oss_interp.backends.transformers_gpt_oss import GPTOSSTransformersBackend
from gpt_oss_interp.capture.activation_cache import ActivationCache
from gpt_oss_interp.common.io import save_json
from gpt_oss_interp.harmony.prompting import encode_prompt

DEFAULT_WORD_POOL = (
    "bird", "cat", "dog", "fish", "horse", "plane", "river", "stone",
    "cloud", "apple", "chair", "train", "house", "field", "bread", "light",
    "green", "silver", "amber", "violet", "forest", "garden", "rocket", "pencil",
)

# Encoding modes
ENCODING_RAW = "raw"          # tokenizer.encode — bypasses chat template; tests logit-lens directly
ENCODING_CHAT = "chat"        # apply_chat_template — tests model as instruction follower
ENCODING_INSTRUCTION = "instruction"  # "Continue: X X X X X →" user turn — chat-formatted but pattern-explicit

_INSTRUCTION_TEMPLATE = "Continue the repeating pattern with exactly one word.\nPattern: {pattern}\nNext word:"


@dataclass
class PeriodicCase:
    case_id: str
    period: int
    cycle_words: tuple[str, ...]
    prompt: str
    expected_word: str
    valid_token_ids: tuple[int, ...]


@dataclass
class LayerValidationRecord:
    layer_idx: int
    expected_token_id: int
    expected_token: str
    expected_logprob: float
    expected_prob: float
    expected_rank: int
    valid_mass: float
    tail_mass: float
    top_token_id: int
    top_token: str
    top_logprob: float
    top_prob: float
    top_is_valid: bool
    top_tokens: list[str]
    top_token_ids: list[int]
    top_token_logprobs: list[float]


@dataclass
class CaseValidationResult:
    case_id: str
    period: int
    cycle_words: tuple[str, ...]
    prompt: str
    expected_word: str
    convergence_layer: int | None
    final_top_matches_expected: bool
    layers: list[LayerValidationRecord]


@dataclass
class LayerSummary:
    layer_idx: int
    n_cases: int
    mean_expected_prob: float
    mean_valid_mass: float
    mean_tail_mass: float
    mean_expected_rank: float
    top_is_valid_rate: float


@dataclass
class PeriodSummary:
    period: int
    n_cases: int
    final_top_match_rate: float
    mean_convergence_layer: float | None
    layers: list[LayerSummary]


def parse_periods(spec: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in spec.split(",") if part.strip())
    if not values:
        raise ValueError("At least one period must be provided")
    if any(v <= 0 for v in values):
        raise ValueError("Periods must be positive integers")
    return values


def continuation_token_id(tokenizer: Any, word: str) -> int:
    token_ids = tokenizer.encode(" " + word, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"Word {word!r} is not a single continuation token: {token_ids}")
    return int(token_ids[0])


def filter_single_token_words(tokenizer: Any, words: Sequence[str]) -> list[str]:
    kept: list[str] = []
    for word in words:
        try:
            continuation_token_id(tokenizer, word)
        except ValueError:
            continue
        kept.append(word)
    return kept


def build_periodic_prompt(cycle_words: Sequence[str], repeats: int) -> tuple[str, str]:
    if not cycle_words:
        raise ValueError("cycle_words must not be empty")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    cycle = list(cycle_words)
    prompt = " ".join(cycle * repeats)
    expected_word = cycle[0]
    return prompt, expected_word


def generate_periodic_cases(
    tokenizer: Any,
    *,
    periods: Sequence[int],
    examples_per_period: int,
    repeats: int,
    word_pool: Sequence[str] = DEFAULT_WORD_POOL,
    seed: int = 0,
) -> list[PeriodicCase]:
    rng = random.Random(seed)
    single_token_words = filter_single_token_words(tokenizer, word_pool)
    cases: list[PeriodicCase] = []

    for period in periods:
        if period > len(single_token_words):
            raise ValueError(
                f"Not enough single-token words for period {period}; only {len(single_token_words)} available"
            )
        for example_idx in range(examples_per_period):
            cycle_words = tuple(rng.sample(single_token_words, period))
            prompt, expected_word = build_periodic_prompt(cycle_words, repeats)
            token_id = continuation_token_id(tokenizer, expected_word)
            cases.append(
                PeriodicCase(
                    case_id=f"period_{period:02d}_{example_idx:03d}",
                    period=period,
                    cycle_words=cycle_words,
                    prompt=prompt,
                    expected_word=expected_word,
                    valid_token_ids=(token_id,),
                )
            )
    return cases


def _decode_token(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([token_id]).replace("\n", "\\n")


def _encode_for_mode(tokenizer: Any, prompt: str, encoding_mode: str) -> list[int]:
    """Return token ids for prompt under the given encoding mode.

    raw:         tokenizer.encode(prompt) — no chat template.  Tests whether
                 intermediate hidden states encode the induction signal.
    chat:        apply_chat_template wrapping prompt as a user turn.  The model
                 predicts the first assistant token.  Correct for benchmarks;
                 wrong for sequence-continuation readout tests because the
                 assistant turn changes the prediction target.
    instruction: same chat template but with an explicit continuation instruction
                 so the assistant's first token should be the expected word.
    """
    if encoding_mode == ENCODING_RAW:
        return tokenizer.encode(prompt, add_special_tokens=True)
    if encoding_mode == ENCODING_CHAT:
        return encode_prompt(tokenizer, prompt)
    if encoding_mode == ENCODING_INSTRUCTION:
        instruction = _INSTRUCTION_TEMPLATE.format(pattern=prompt)
        return encode_prompt(tokenizer, instruction)
    raise ValueError(f"Unknown encoding_mode {encoding_mode!r}")


def layer_log_probs_for_next_token(
    backend: GPTOSSTransformersBackend,
    prompt: str,
    encoding_mode: str = ENCODING_RAW,
) -> list[torch.Tensor]:
    prompt_ids = _encode_for_mode(backend.tokenizer, prompt, encoding_mode)
    input_ids = torch.tensor([prompt_ids], device=backend.device)

    cache = ActivationCache(detach=True, to_cpu=True)
    handles = cache.register(backend.model, backend.structure.block_names)
    try:
        with torch.no_grad():
            backend.model(input_ids)
    finally:
        for handle in handles:
            handle.remove()

    norm_device = next(backend.structure.final_norm.parameters()).device
    layer_log_probs: list[torch.Tensor] = []
    for block_name in backend.structure.block_names:
        record = cache.last(block_name)
        if record is None:
            continue
        hidden = record.tensor
        with torch.no_grad():
            normed = backend.structure.final_norm(hidden.to(norm_device))
            logits = backend.structure.lm_head(normed).cpu().float()[0, -1]
            log_probs = torch.log_softmax(logits, dim=-1)
        layer_log_probs.append(log_probs)
    return layer_log_probs


def distribution_metrics(
    log_probs: torch.Tensor,
    valid_token_ids: Sequence[int],
    tokenizer: Any,
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    if not valid_token_ids:
        raise ValueError("valid_token_ids must not be empty")

    probs = torch.exp(log_probs)
    expected_token_id = int(valid_token_ids[0])
    expected_logprob = float(log_probs[expected_token_id].item())
    expected_prob = float(probs[expected_token_id].item())
    expected_rank = int((log_probs > log_probs[expected_token_id]).sum().item())

    valid_mass = float(sum(probs[token_id].item() for token_id in valid_token_ids))
    tail_mass = max(0.0, 1.0 - valid_mass)

    k = min(top_k, log_probs.shape[0])
    top_vals, top_ids = torch.topk(log_probs, k=k)
    top_token_ids = [int(tid) for tid in top_ids.tolist()]
    top_tokens = [_decode_token(tokenizer, tid) for tid in top_token_ids]
    top_token_id = top_token_ids[0]

    return {
        "expected_token_id": expected_token_id,
        "expected_token": _decode_token(tokenizer, expected_token_id),
        "expected_logprob": expected_logprob,
        "expected_prob": expected_prob,
        "expected_rank": expected_rank,
        "valid_mass": valid_mass,
        "tail_mass": tail_mass,
        "top_token_id": top_token_id,
        "top_token": top_tokens[0],
        "top_logprob": float(top_vals[0].item()),
        "top_prob": float(math.exp(top_vals[0].item())),
        "top_is_valid": bool(top_token_id in set(int(t) for t in valid_token_ids)),
        "top_tokens": top_tokens,
        "top_token_ids": top_token_ids,
        "top_token_logprobs": [float(v) for v in top_vals.tolist()],
    }


def evaluate_case(
    backend: GPTOSSTransformersBackend,
    case: PeriodicCase,
    *,
    top_k: int = 5,
    encoding_mode: str = ENCODING_RAW,
) -> CaseValidationResult:
    layer_records: list[LayerValidationRecord] = []
    for layer_idx, log_probs in enumerate(
        layer_log_probs_for_next_token(backend, case.prompt, encoding_mode=encoding_mode)
    ):
        metrics = distribution_metrics(log_probs, case.valid_token_ids, backend.tokenizer, top_k=top_k)
        layer_records.append(LayerValidationRecord(layer_idx=layer_idx, **metrics))

    convergence_layer = next((record.layer_idx for record in layer_records if record.top_is_valid), None)
    final_top_matches_expected = bool(layer_records[-1].top_is_valid) if layer_records else False

    return CaseValidationResult(
        case_id=case.case_id,
        period=case.period,
        cycle_words=case.cycle_words,
        prompt=case.prompt,
        expected_word=case.expected_word,
        convergence_layer=convergence_layer,
        final_top_matches_expected=final_top_matches_expected,
        layers=layer_records,
    )


def summarize_period(case_results: Sequence[CaseValidationResult], period: int) -> PeriodSummary:
    period_cases = [result for result in case_results if result.period == period]
    if not period_cases:
        raise ValueError(f"No case results for period {period}")

    num_layers = len(period_cases[0].layers)
    layers: list[LayerSummary] = []
    for layer_idx in range(num_layers):
        layer_rows = [result.layers[layer_idx] for result in period_cases]
        layers.append(
            LayerSummary(
                layer_idx=layer_idx,
                n_cases=len(layer_rows),
                mean_expected_prob=sum(row.expected_prob for row in layer_rows) / len(layer_rows),
                mean_valid_mass=sum(row.valid_mass for row in layer_rows) / len(layer_rows),
                mean_tail_mass=sum(row.tail_mass for row in layer_rows) / len(layer_rows),
                mean_expected_rank=sum(row.expected_rank for row in layer_rows) / len(layer_rows),
                top_is_valid_rate=sum(1.0 for row in layer_rows if row.top_is_valid) / len(layer_rows),
            )
        )

    convergence_layers = [float(result.convergence_layer) for result in period_cases if result.convergence_layer is not None]
    return PeriodSummary(
        period=period,
        n_cases=len(period_cases),
        final_top_match_rate=sum(1.0 for result in period_cases if result.final_top_matches_expected) / len(period_cases),
        mean_convergence_layer=(sum(convergence_layers) / len(convergence_layers)) if convergence_layers else None,
        layers=layers,
    )


def summarize_results(
    case_results: Sequence[CaseValidationResult],
    periods: Sequence[int],
) -> list[PeriodSummary]:
    return [summarize_period(case_results, period) for period in periods]


def format_markdown_summary(
    summaries: Sequence[PeriodSummary],
    *,
    last_n_layers: int = 0,
    encoding_mode: str = ENCODING_RAW,
) -> str:
    lines = [
        "# Unembedding Validation Summary",
        "",
        f"**Encoding mode**: `{encoding_mode}`  ",
        "(`raw` = bypasses chat template, pure logit-lens test; "
        "`instruction` = chat-wrapped with explicit continuation prompt)",
        "",
    ]
    for summary in summaries:
        lines.append(f"## Period {summary.period}")
        lines.append("")
        lines.append(f"- Cases: {summary.n_cases}")
        lines.append(f"- Final top-1 matches expected: {summary.final_top_match_rate:.3f}")
        if summary.mean_convergence_layer is None:
            lines.append("- Mean convergence layer: n/a")
        else:
            lines.append(f"- Mean convergence layer: {summary.mean_convergence_layer:.2f}")
        lines.append("")
        lines.append("| Layer | Expected Prob | Valid Mass | Tail Mass | Expected Rank | Top-1 Valid |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
        display_layers = summary.layers if last_n_layers == 0 else summary.layers[-last_n_layers:]
        for layer in display_layers:
            lines.append(
                f"| {layer.layer_idx} | {layer.mean_expected_prob:.4f} | {layer.mean_valid_mass:.4f} | "
                f"{layer.mean_tail_mass:.4f} | {layer.mean_expected_rank:.2f} | {layer.top_is_valid_rate:.3f} |"
            )
        lines.append("")
    return "\n".join(lines)


def run_experiments(args: argparse.Namespace) -> dict[str, Any]:
    backend = GPTOSSTransformersBackend(
        model_name=args.model_name,
        dtype=args.dtype,
        local_files_only=not args.allow_downloads,
        trust_remote_code=args.trust_remote_code,
    )
    periods = parse_periods(args.periods)
    word_pool = tuple(word.strip() for word in args.word_pool.split(",") if word.strip())
    cases = generate_periodic_cases(
        backend.tokenizer,
        periods=periods,
        examples_per_period=args.examples_per_period,
        repeats=args.repeats,
        word_pool=word_pool,
        seed=args.seed,
    )
    if args.max_cases is not None:
        cases = cases[: args.max_cases]

    case_results = [
        evaluate_case(backend, case, top_k=args.top_k, encoding_mode=args.encoding_mode)
        for case in cases
    ]
    summaries = summarize_results(case_results, sorted({case.period for case in cases}))

    payload = {
        "config": {
            "model_name": args.model_name,
            "periods": list(periods),
            "examples_per_period": args.examples_per_period,
            "repeats": args.repeats,
            "seed": args.seed,
            "top_k": args.top_k,
            "encoding_mode": args.encoding_mode,
            "allow_downloads": bool(args.allow_downloads),
        },
        "case_results": [asdict(result) for result in case_results],
        "summaries": [asdict(summary) for summary in summaries],
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(payload, output_dir / "unembedding_validation.json")
    (output_dir / "unembedding_validation.md").write_text(
        format_markdown_summary(summaries, encoding_mode=args.encoding_mode)
    )
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run synthetic unembedding validation experiments")
    parser.add_argument("--model-name", default="openai/gpt-oss-20b")
    parser.add_argument("--periods", default="1,2,3")
    parser.add_argument("--examples-per-period", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--encoding-mode",
        default=ENCODING_RAW,
        choices=[ENCODING_RAW, ENCODING_CHAT, ENCODING_INSTRUCTION],
        help=(
            "raw: bypass chat template (tests logit-lens on sequence continuation); "
            "chat: full chat template (model sees bare pattern as user query); "
            "instruction: chat template + explicit continuation instruction"
        ),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--output-dir", default="./runs/unembedding_validation")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--word-pool", default=",".join(DEFAULT_WORD_POOL))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    payload = run_experiments(args)
    for summary in payload["summaries"]:
        period = summary["period"]
        final_match = summary["final_top_match_rate"]
        mean_conv = summary["mean_convergence_layer"]
        mean_conv_text = "n/a" if mean_conv is None else f"{mean_conv:.2f}"
        print(
            f"period={period} cases={summary['n_cases']} final_top1={final_match:.3f} "
            f"mean_convergence={mean_conv_text}"
        )
    print(f"Wrote outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
