"""Tests for the analysis-set stratification logic."""

import pytest
import sys
from pathlib import Path

# The stratification logic is in a script, not a package module.
# Import the classify function by adding the scripts directory.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from stratify_analysis_set import classify_case


def _make_layer_details(winners_by_layer, expected_label="A"):
    """Build synthetic layer_details from a list of per-layer winners.

    Args:
        winners_by_layer: list of label strings, e.g. ["A", "A", "B", "A", "A"]
        expected_label: which label is the expected answer
    """
    details = []
    for i, winner in enumerate(winners_by_layer):
        choice_logprobs = {}
        if winner == expected_label:
            choice_logprobs[expected_label] = -1.0
            other = "B" if expected_label == "A" else "A"
            choice_logprobs[other] = -3.0
        else:
            choice_logprobs[expected_label] = -3.0
            other = "B" if expected_label == "A" else "A"
            choice_logprobs[other] = -1.0
        details.append({
            "layer": i,
            "winner": winner,
            "expected_logprob": choice_logprobs[expected_label],
            "choice_logprobs": choice_logprobs,
            "margin_vs_runner_up": abs(choice_logprobs["A"] - choice_logprobs["B"]),
        })
    return details


def _make_case(case_id, task_name, expected_label, final_winner, layer_details):
    return {
        "case_id": case_id,
        "task_name": task_name,
        "expected_label": expected_label,
        "expected_text": " yes",
        "final_winner": final_winner,
        "final_correct": 1 if final_winner == expected_label else 0,
        "expected_choice_convergence": 0,
        "final_choice_convergence": 0,
        "layer_details": layer_details,
    }


class TestStrictStratification:
    def test_correct_stable(self):
        # Expected wins from layer 2 onward, never loses
        winners = ["B", "B", "A", "A", "A", "A"]
        details = _make_layer_details(winners, "A")
        case = _make_case("test", "induction", "A", "A", details)
        result = classify_case(case, tail_length=4)
        assert result["strict_stratum"] == "correct_stable"

    def test_correct_unstable(self):
        # Expected wins, then loses, then wins again
        winners = ["A", "A", "B", "A", "A", "A"]
        details = _make_layer_details(winners, "A")
        case = _make_case("test", "induction", "A", "A", details)
        result = classify_case(case, tail_length=4)
        assert result["strict_stratum"] == "correct_unstable"

    def test_incorrect_early_expected(self):
        # Expected wins early but final answer is wrong
        winners = ["A", "A", "A", "B", "B", "B"]
        details = _make_layer_details(winners, "A")
        case = _make_case("test", "recency", "A", "B", details)
        result = classify_case(case, tail_length=4)
        assert result["strict_stratum"] == "incorrect_early_expected"

    def test_incorrect_never_expected(self):
        # Expected never wins
        winners = ["B", "B", "B", "B", "B", "B"]
        details = _make_layer_details(winners, "A")
        case = _make_case("test", "caps", "A", "B", details)
        result = classify_case(case, tail_length=4)
        assert result["strict_stratum"] == "incorrect_never_expected"


class TestSoftStratification:
    def test_late_stable(self):
        # Unstable early but stable for last 4 layers
        winners = ["A", "B", "A", "B", "A", "A", "A", "A"]
        details = _make_layer_details(winners, "A")
        case = _make_case("test", "induction", "A", "A", details)
        result = classify_case(case, tail_length=4)
        assert result["soft_stratum"] == "correct_late_stable"

    def test_late_unstable(self):
        # Correct final but not stable in last 4
        winners = ["A", "B", "A", "B", "A", "B", "A", "A"]
        details = _make_layer_details(winners, "A")
        case = _make_case("test", "syntax", "A", "A", details)
        result = classify_case(case, tail_length=4)
        # Should be correct_late_unstable because B appears at layer 5 (within last 4 = layers 4-7)
        assert result["soft_stratum"] == "correct_late_unstable"

    def test_always_correct_is_late_stable(self):
        # Expected wins every layer
        winners = ["A"] * 8
        details = _make_layer_details(winners, "A")
        case = _make_case("test", "induction", "A", "A", details)
        result = classify_case(case, tail_length=4)
        assert result["soft_stratum"] == "correct_late_stable"


class TestClassifyCaseFields:
    def test_output_has_required_fields(self):
        winners = ["A", "A", "A", "A"]
        details = _make_layer_details(winners, "A")
        case = _make_case("test", "induction", "A", "A", details)
        result = classify_case(case, tail_length=4)
        required = [
            "case_id", "task_name", "first_expected_layer", "last_expected_layer",
            "num_winner_flips", "stable_after_first_expected", "final_expected_streak",
            "strict_stratum", "soft_stratum",
        ]
        for field in required:
            assert field in result, f"Missing field: {field}"

    def test_first_expected_layer_correct(self):
        winners = ["B", "B", "A", "A"]
        details = _make_layer_details(winners, "A")
        case = _make_case("test", "induction", "A", "A", details)
        result = classify_case(case, tail_length=4)
        assert result["first_expected_layer"] == 2

    def test_num_winner_flips(self):
        winners = ["A", "B", "A", "B", "A", "A"]
        details = _make_layer_details(winners, "A")
        case = _make_case("test", "induction", "A", "A", details)
        result = classify_case(case, tail_length=4)
        # Flips: A→B, B→A, A→B, B→A = 4 flips
        assert result["num_winner_flips"] == 4
