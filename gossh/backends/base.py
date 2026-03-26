"""Backend contract for GOSSH."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from gossh.config import InterventionSpec, PromptCase


@dataclass
class BackendScore:
    choice_logprobs: dict[str, float]
    metadata: dict[str, Any]


class BaseBackend(ABC):
    @abstractmethod
    def score_case(self, case: PromptCase) -> BackendScore:
        raise NotImplementedError

    @abstractmethod
    def apply_intervention(self, spec: InterventionSpec, scale: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def clear_interventions(self) -> None:
        raise NotImplementedError
