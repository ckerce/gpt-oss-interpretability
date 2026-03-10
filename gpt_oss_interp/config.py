from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


###############################################################################
#
# Enums
#
###############################################################################

class BackendKind(Enum):
    DRY_RUN = "dry_run"
    GPT_OSS_TRANSFORMERS = "gpt_oss_transformers"


class InterventionKind(Enum):
    HEAD_MASK = "head_mask"
    EXPERT_MASK = "expert_mask"
    LAYER_SCALE = "layer_scale"
    TEMPERATURE_SCALE = "temperature_scale"


class TargetUnit(Enum):
    HEAD = "head"
    EXPERT = "expert"
    LAYER = "layer"
    MODEL = "model"


###############################################################################
#
# Task and intervention specs
#
###############################################################################

@dataclass
class PromptCase:
    case_id: str
    prompt: str
    choices: dict[str, str]
    expected_label: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptTask:
    name: str
    behavior: str
    cases: list[PromptCase]
    description: str = ""


@dataclass
class InterventionTarget:
    unit: TargetUnit
    layer_indices: tuple[int, ...] = ()
    head_indices: tuple[int, ...] = ()
    expert_indices: tuple[int, ...] = ()
    note: str = ""

    def signature(self) -> str:
        parts = [self.unit.value]
        if self.layer_indices:
            parts.append("L" + ",".join(str(x) for x in self.layer_indices))
        if self.head_indices:
            parts.append("H" + ",".join(str(x) for x in self.head_indices))
        if self.expert_indices:
            parts.append("E" + ",".join(str(x) for x in self.expert_indices))
        return ":".join(parts)


@dataclass
class InterventionSpec:
    name: str
    kind: InterventionKind
    target: InterventionTarget
    scales: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5)
    description: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    def signature(self) -> str:
        return f"{self.kind.value}:{self.target.signature()}:{self.name}"


###############################################################################
#
# Benchmark config
#
###############################################################################

@dataclass
class BenchmarkSweepConfig:
    repeats: int = 1
    max_examples: Optional[int] = None
    record_case_outputs: bool = True


@dataclass
class BenchmarkConfig:
    backend_kind: BackendKind
    backend_params: dict[str, Any]
    tasks: list[PromptTask]
    interventions: list[InterventionSpec]
    sweep: BenchmarkSweepConfig = field(default_factory=BenchmarkSweepConfig)
    seed: int = 7
    experiment_name: str = "GPT_OSS_Interp_Benchmark"
    output_dir: str = "./runs/gpt_oss_interp"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


###############################################################################
#
# Serialization helpers
#
###############################################################################

def _serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_serialize(v) for v in value]
    if isinstance(value, list):
        return [_serialize(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        return {
            field_name: _serialize(getattr(value, field_name))
            for field_name in value.__dataclass_fields__
        }
    return value
