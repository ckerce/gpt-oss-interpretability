"""Intervention run expansion for benchmark sweeps."""
from __future__ import annotations

from dataclasses import dataclass

from gossh.config import InterventionSpec


@dataclass
class InterventionRun:
    spec: InterventionSpec
    scale: float

    def run_name(self) -> str:
        return f"{self.spec.name}@{self.scale:g}"


def expand_runs(specs: list[InterventionSpec]) -> list[InterventionRun]:
    """Expand a list of InterventionSpecs into one run per (spec, scale) pair."""
    runs: list[InterventionRun] = []
    for spec in specs:
        for scale in spec.scales:
            runs.append(InterventionRun(spec=spec, scale=scale))
    return runs
