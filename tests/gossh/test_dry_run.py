"""Tests for gossh.backends.dry_run."""

import pytest

from gossh.backends.dry_run import DryRunBackend
from gossh.config import (
    InterventionKind,
    InterventionSpec,
    InterventionTarget,
    PromptCase,
    TargetUnit,
)


def _make_case(case_id="test_001", expected_label="A"):
    return PromptCase(
        case_id=case_id,
        prompt="The trophy would not fit",
        choices={"A": " suitcase", "B": " trophy"},
        expected_label=expected_label,
        metadata={"phenomenon": "recency_bias"},
    )


def _make_intervention(kind=InterventionKind.HEAD_MASK, layer=20, scale=0.0):
    return InterventionSpec(
        name=f"test_{kind.value}_L{layer}",
        kind=kind,
        target=InterventionTarget(
            unit=TargetUnit.HEAD,
            layer_indices=(layer,),
            head_indices=(0,),
            expert_indices=(),
        ),
        scales=(scale,),
        description="test intervention",
    )


class TestDryRunScoring:
    def test_baseline_scoring_returns_both_choices(self):
        backend = DryRunBackend()
        score = backend.score_case(_make_case())
        assert "A" in score.choice_logprobs
        assert "B" in score.choice_logprobs

    def test_expected_label_gets_higher_score(self):
        backend = DryRunBackend()
        score = backend.score_case(_make_case(expected_label="A"))
        assert score.choice_logprobs["A"] > score.choice_logprobs["B"]

    def test_deterministic_scores(self):
        backend = DryRunBackend()
        s1 = backend.score_case(_make_case("c1"))
        s2 = backend.score_case(_make_case("c1"))
        assert s1.choice_logprobs == s2.choice_logprobs

    def test_metadata_backend_field(self):
        backend = DryRunBackend()
        score = backend.score_case(_make_case())
        assert score.metadata["backend"] == "dry_run"


class TestDryRunInterventions:
    def test_apply_and_clear_no_exception(self):
        backend = DryRunBackend()
        spec = _make_intervention()
        backend.apply_intervention(spec, 0.0)
        backend.score_case(_make_case())
        backend.clear_interventions()
        backend.score_case(_make_case())

    def test_head_mask_at_zero_reduces_margin(self):
        backend = DryRunBackend()
        score_base = backend.score_case(_make_case())
        base_margin = score_base.choice_logprobs["A"] - score_base.choice_logprobs["B"]

        spec = _make_intervention(InterventionKind.HEAD_MASK, scale=0.0)
        backend.apply_intervention(spec, 0.0)
        score_ablated = backend.score_case(_make_case())
        ablated_margin = score_ablated.choice_logprobs["A"] - score_ablated.choice_logprobs["B"]
        backend.clear_interventions()

        assert ablated_margin <= base_margin + 0.01

    def test_all_intervention_kinds_accepted(self):
        backend = DryRunBackend()
        for kind in InterventionKind:
            spec = _make_intervention(kind=kind, scale=0.5)
            backend.apply_intervention(spec, 0.5)
            score = backend.score_case(_make_case())
            assert "A" in score.choice_logprobs
            backend.clear_interventions()


class TestDryRunBehaviorBias:
    def test_custom_bias(self):
        backend = DryRunBackend(behavior_bias={"recency_bias": 2.0})
        score = backend.score_case(_make_case())
        assert score.choice_logprobs["A"] > score.choice_logprobs["B"]
