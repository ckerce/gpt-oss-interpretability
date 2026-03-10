###############################################################################
#
# Harmony prompting helpers
#
###############################################################################

from __future__ import annotations

from typing import Any


def build_chat_messages(prompt: str) -> list[dict[str, str]]:
    """Wrap a benchmark prompt as a harmony-style user message."""
    return [{"role": "user", "content": prompt}]


def encode_prompt(tokenizer: Any, prompt: str) -> list[int]:
    """Tokenize a benchmark prompt with the harmony chat template.

    Uses the tokenizer's built-in chat template so that the model receives
    the format it was trained on.  Returns token ids up through the
    assistant-generation prefix.
    """
    messages = build_chat_messages(prompt)
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors=None,
    )


def encode_prompt_with_completion(
    tokenizer: Any,
    prompt: str,
    completion: str,
) -> tuple[list[int], int]:
    """Tokenize prompt + completion and return (ids, choice_start_idx).

    The choice_start_idx marks where the completion tokens begin so callers
    can isolate the logprobs that correspond to the completion.
    """
    prompt_ids = encode_prompt(tokenizer, prompt)
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion},
    ]
    full_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors=None,
    )
    return full_ids, len(prompt_ids)
