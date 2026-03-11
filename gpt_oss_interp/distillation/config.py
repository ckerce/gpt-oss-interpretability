"""Typed config dataclasses for distillation experiments.

Do not import from steering here.
"""
from dataclasses import dataclass, field


@dataclass
class StudentArchConfig:
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768
    architecture: str = "cascade"


@dataclass
class DistillationRunConfig:
    run_id: str = ""
    teacher_id: str = ""
    student: StudentArchConfig = field(default_factory=StudentArchConfig)
    output_only_first: bool = True
    kl_refinement: bool = False
