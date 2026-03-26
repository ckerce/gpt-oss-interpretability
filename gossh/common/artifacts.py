"""Canonical run artifact schemas.

OWNERSHIP RULE: single source of truth for JSON artifact structure.
Both steering and distillation must write and read artifacts through
these types. Schema drift between subpackages is a bug.

Do not put workflow logic here.
"""
from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = "1.0.0"


@dataclass
class RunArtifact:
    run_id: str
    model_id: str
    phase: str
    schema_version: str = SCHEMA_VERSION
    results: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
