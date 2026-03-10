from __future__ import annotations

from gpt_oss_interp.config import PromptCase, PromptTask


###############################################################################
#
# Built-in tasks
#
###############################################################################

def recency_bias_task() -> PromptTask:
    cases = [
        PromptCase(
            case_id="recency_001",
            prompt=(
                "The trophy would not fit in the suitcase because the suitcase was too small. "
                "The word 'small' refers to the"
            ),
            choices={"A": " suitcase", "B": " trophy"},
            expected_label="A",
            metadata={"phenomenon": "recency_bias"},
        ),
        PromptCase(
            case_id="recency_002",
            prompt=(
                "The ball would not fit in the bag because the bag was too narrow. "
                "The word 'narrow' refers to the"
            ),
            choices={"A": " bag", "B": " ball"},
            expected_label="A",
            metadata={"phenomenon": "recency_bias"},
        ),
        PromptCase(
            case_id="recency_003",
            prompt=(
                "The cat could not catch the mouse because the mouse was too fast. "
                "The word 'fast' refers to the"
            ),
            choices={"A": " mouse", "B": " cat"},
            expected_label="A",
            metadata={"phenomenon": "recency_bias"},
        ),
        PromptCase(
            case_id="recency_004",
            prompt=(
                "The painting would not fit on the wall because the wall was too narrow. "
                "The word 'narrow' refers to the"
            ),
            choices={"A": " wall", "B": " painting"},
            expected_label="A",
            metadata={"phenomenon": "recency_bias"},
        ),
    ]
    return PromptTask(
        name="recency_bias",
        behavior="recency_bias",
        cases=cases,
        description="Recency-sensitive resolution of attributive adjectives.",
    )


def capitalization_task() -> PromptTask:
    cases = [
        PromptCase(
            case_id="caps_001",
            prompt="Complete the title in normal title case: the lord of the",
            choices={"A": " Rings", "B": " rings"},
            expected_label="A",
            metadata={"phenomenon": "capitalization"},
        ),
        PromptCase(
            case_id="caps_002",
            prompt="Write the US state in normal headline style: north",
            choices={"A": " Carolina", "B": " carolina"},
            expected_label="A",
            metadata={"phenomenon": "capitalization"},
        ),
        PromptCase(
            case_id="caps_003",
            prompt="Complete the name of the city: new",
            choices={"A": " York", "B": " york"},
            expected_label="A",
            metadata={"phenomenon": "capitalization"},
        ),
        PromptCase(
            case_id="caps_004",
            prompt="Complete the country name: united",
            choices={"A": " States", "B": " states"},
            expected_label="A",
            metadata={"phenomenon": "capitalization"},
        ),
    ]
    return PromptTask(
        name="capitalization",
        behavior="capitalization",
        cases=cases,
        description="Formatting behavior: title case and proper noun capitalization.",
    )


def induction_task() -> PromptTask:
    """Induction / copying behavior: the model should repeat a learned pattern."""
    cases = [
        PromptCase(
            case_id="induction_001",
            prompt="A7 B2 C9 D4 A7 B2 C9",
            choices={"A": " D4", "B": " E5"},
            expected_label="A",
            metadata={"phenomenon": "induction"},
        ),
        PromptCase(
            case_id="induction_002",
            prompt="red blue green red blue green red blue",
            choices={"A": " green", "B": " red"},
            expected_label="A",
            metadata={"phenomenon": "induction"},
        ),
        PromptCase(
            case_id="induction_003",
            prompt="1 2 3 4 1 2 3 4 1 2 3",
            choices={"A": " 4", "B": " 5"},
            expected_label="A",
            metadata={"phenomenon": "induction"},
        ),
        PromptCase(
            case_id="induction_004",
            prompt="alpha beta gamma alpha beta gamma alpha beta",
            choices={"A": " gamma", "B": " delta"},
            expected_label="A",
            metadata={"phenomenon": "induction"},
        ),
    ]
    return PromptTask(
        name="induction",
        behavior="induction",
        cases=cases,
        description="Induction-style copying: repeat a previously observed pattern.",
    )


def coreference_task() -> PromptTask:
    """Simple coreference resolution with unambiguous cases."""
    cases = [
        PromptCase(
            case_id="coref_001",
            prompt=(
                "Alice gave her old laptop to Bob. "
                "The word 'her' refers to"
            ),
            choices={"A": " Alice", "B": " Bob"},
            expected_label="A",
            metadata={"phenomenon": "coreference"},
        ),
        PromptCase(
            case_id="coref_002",
            prompt=(
                "The teacher praised the student because the student got every answer right. "
                "Later, the teacher said he would recommend the student for the award. "
                "The word 'he' refers to the"
            ),
            choices={"A": " teacher", "B": " student"},
            expected_label="A",
            metadata={"phenomenon": "coreference"},
        ),
        PromptCase(
            case_id="coref_003",
            prompt=(
                "John called Mary and told her about the meeting. "
                "The word 'her' refers to"
            ),
            choices={"A": " Mary", "B": " John"},
            expected_label="A",
            metadata={"phenomenon": "coreference"},
        ),
        PromptCase(
            case_id="coref_004",
            prompt=(
                "The mother picked up the child because she was strong enough to carry him. "
                "The word 'she' refers to the"
            ),
            choices={"A": " mother", "B": " child"},
            expected_label="A",
            metadata={"phenomenon": "coreference"},
        ),
    ]
    return PromptTask(
        name="coreference",
        behavior="coreference",
        cases=cases,
        description="Simple pronoun coreference resolution.",
    )


def syntax_agreement_task() -> PromptTask:
    """Subject-verb agreement across attractors."""
    cases = [
        PromptCase(
            case_id="syntax_001",
            prompt="Fill in the blank with the correct verb form: 'The keys to the cabinet ___.' Answer:",
            choices={"A": " are", "B": " is"},
            expected_label="A",
            metadata={"phenomenon": "syntax_agreement"},
        ),
        PromptCase(
            case_id="syntax_002",
            prompt="Fill in the blank with the correct verb form: 'The dog behind the fences ___.' Answer:",
            choices={"A": " barks", "B": " bark"},
            expected_label="A",
            metadata={"phenomenon": "syntax_agreement"},
        ),
        PromptCase(
            case_id="syntax_003",
            prompt="Fill in the blank with the correct verb form: 'The books on the shelf ___.' Answer:",
            choices={"A": " were", "B": " was"},
            expected_label="A",
            metadata={"phenomenon": "syntax_agreement"},
        ),
        PromptCase(
            case_id="syntax_004",
            prompt="Fill in the blank with the correct verb form: 'The child near the tall trees ___.' Answer:",
            choices={"A": " runs", "B": " run"},
            expected_label="A",
            metadata={"phenomenon": "syntax_agreement"},
        ),
    ]
    return PromptTask(
        name="syntax_agreement",
        behavior="syntax_agreement",
        cases=cases,
        description="Subject-verb agreement with prepositional phrase attractors.",
    )


###############################################################################
#
# Task registry
#
###############################################################################

ALL_TASKS = {
    "recency_bias": recency_bias_task,
    "capitalization": capitalization_task,
    "induction": induction_task,
    "coreference": coreference_task,
    "syntax_agreement": syntax_agreement_task,
}


def all_tasks() -> list[PromptTask]:
    return [fn() for fn in ALL_TASKS.values()]
