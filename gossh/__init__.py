"""GOSSH — gpt-oss-inspectability-harness.

Public API surface:

  Tier 1 — config types (import without GPU):
    BenchmarkConfig, BenchmarkSweepConfig
    PromptTask, PromptCase
    InterventionSpec, InterventionTarget
    InterventionKind, TargetUnit, BackendKind, ModelKind

  Tier 2 — entry points (require torch + optional GPU):
    BenchmarkRunner
    ModelStructure
    get_arch_spec, list_supported_models
"""
from __future__ import annotations

# ── Tier 1: config types ───────────────────────────────────────────────────────
from .config import (
    BackendKind,
    BenchmarkConfig,
    BenchmarkSweepConfig,
    InterventionKind,
    InterventionSpec,
    InterventionTarget,
    ModelKind,
    PromptCase,
    PromptTask,
    TargetUnit,
)

# ── Model registry ─────────────────────────────────────────────────────────────
from .model_registry import ModelArchSpec, get_arch_spec, list_supported_models

# ── Tier 2: entry points (lazy to avoid mandatory torch import at install) ─────
def __getattr__(name: str):
    if name == "BenchmarkRunner":
        from .benchmarks.runner import BenchmarkRunner
        return BenchmarkRunner
    if name == "ModelStructure":
        from .backends.structure import ModelStructure
        return ModelStructure
    raise AttributeError(f"module 'gossh' has no attribute {name!r}")


__all__ = [
    # Tier 1
    "BackendKind",
    "BenchmarkConfig",
    "BenchmarkSweepConfig",
    "InterventionKind",
    "InterventionSpec",
    "InterventionTarget",
    "ModelKind",
    "PromptCase",
    "PromptTask",
    "TargetUnit",
    # Model registry
    "ModelArchSpec",
    "get_arch_spec",
    "list_supported_models",
    # Tier 2
    "BenchmarkRunner",
    "ModelStructure",
]
