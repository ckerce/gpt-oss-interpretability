from __future__ import annotations

from gpt_oss_interp.backends.base import BaseBackend, BackendScore
from gpt_oss_interp.config import InterventionKind, InterventionSpec, PromptCase


###############################################################################
#
# Dry-run backend
#
###############################################################################

class DryRunBackend(BaseBackend):
    """Synthetic backend for exercising the repo end-to-end.

    The goal is not realism. The goal is to validate repo shape, config loading,
    intervention sweeps, and reporting.
    """

    def __init__(self, behavior_bias: dict[str, float] | None = None):
        self.behavior_bias = behavior_bias or {}
        self._active = None

    def score_case(self, case: PromptCase) -> BackendScore:
        phenomenon = case.metadata.get("phenomenon", "default")
        base = self.behavior_bias.get(phenomenon, 1.0)
        scores: dict[str, float] = {}

        for label in case.choices:
            if label == case.expected_label:
                score = base
            else:
                score = 0.0

            if self._active is not None:
                kind, scale = self._active
                if kind == InterventionKind.HEAD_MASK:
                    if label == case.expected_label:
                        score += (scale - 1.0) * 0.8
                    else:
                        score -= (scale - 1.0) * 0.2
                elif kind == InterventionKind.EXPERT_MASK:
                    if label == case.expected_label:
                        score += (scale - 1.0) * 0.6
                elif kind == InterventionKind.LAYER_SCALE:
                    if label == case.expected_label:
                        score += (scale - 1.0) * 0.4
                elif kind == InterventionKind.TEMPERATURE_SCALE:
                    if label == case.expected_label:
                        score += (scale - 1.0) * 0.3

            scores[label] = score

        return BackendScore(choice_logprobs=scores, metadata={"backend": "dry_run"})

    def apply_intervention(self, spec: InterventionSpec, scale: float) -> None:
        self._active = (spec.kind, scale)

    def clear_interventions(self) -> None:
        self._active = None
