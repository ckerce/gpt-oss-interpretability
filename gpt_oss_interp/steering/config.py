"""Typed config dataclasses for steering experiments.

Old configs/ files should gradually instantiate these rather than
carrying raw dicts. Do not import from distillation here.
"""
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class InterventionConfig:
    site: Literal["embedding_init", "pre_block", "post_block"] = "pre_block"
    stream: Literal["x_t", "x_e", "both"] = "x_t"
    layer: int = 0
    scale: float = 1.0


@dataclass
class SteeringRunConfig:
    run_id: str = ""
    model_id: str = ""
    case_pool: str = "default"
    intervention: InterventionConfig = field(default_factory=InterventionConfig)
