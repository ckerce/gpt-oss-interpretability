#!/usr/bin/env python3
"""
scaffold_refactor.py

Run from the repo root (gpt-oss-interp/).
Creates new package directories and stub modules only.
Does NOT modify any existing file.

This script is intentionally idempotent: existing files are skipped.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

STUBS: dict[str, str] = {
    # -- steering -------------------------------------------------------------
    "gpt_oss_interp/steering/__init__.py": "",
    "gpt_oss_interp/steering/specs.py": '''\
"""Intervention spec objects.

Temporarily re-exports from the old location so new code can import
from gpt_oss_interp.steering.specs immediately.

Migration step (later, not now):
    1. Move content of gpt_oss_interp/interventions/specs.py here.
    2. Replace gpt_oss_interp/interventions/specs.py with:
           from gpt_oss_interp.steering.specs import *  # noqa: F401,F403
"""
from gpt_oss_interp.interventions.specs import *  # noqa: F401,F403
''',
    "gpt_oss_interp/steering/config.py": '''\
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
''',
    "gpt_oss_interp/steering/interventions.py": '''\
"""Intervention runners: whole-vector, per-channel, composite.

Depends on: steering.specs, steering.config, common.artifacts
"""
''',
    "gpt_oss_interp/steering/probing.py": '''\
"""Per-channel differential probing.

Entry point for Phase 1 of PER_CHANNEL_XT_INTERVENTION_PLAN.

New code goes here directly. Do not accumulate it in scripts/.

Public API (to be implemented):
    probe_channel_preferences(model, cases, config) -> ChannelProbeResult
        Computes per-layer, per-head slice preference scores:
            pref(l, h) = <x_t[l, h], e_A,h - e_B,h>
        Includes null baselines:
            - shuffled head labels
            - matched same-norm random token-slice directions
            - within-family label permutation

    rank_channels(result) -> ChannelRanking
        Ranks channels by differential preference shift across
        minimal pairs. Reports stability, selectivity, and
        position sensitivity per channel.

    promote_channels(ranking, threshold) -> list[ChannelHypothesis]
        Promotes channels that exceed threshold on held-out minimal
        pairs. Threshold must be set before running, not post-hoc.
        Suggested first bar: sign prediction accuracy > 0.70.
"""
''',
    "gpt_oss_interp/steering/readouts.py": '''\
"""Readout decomposition: x_t-only, x_e-only, combined, per-head slice.

Required output of Phase 2 per PER_CHANNEL_XT_INTERVENTION_PLAN.

Public API (to be implemented):
    decompose_readout(model, intervention_result) -> ReadoutDecomposition

    stream_transfer_ratio(decomp) -> float
        Defined as: effect_xe / effect_xt at the decision token,
        where both effects are measured under the same intervention.

        Interpretation:
            ratio << 1  - effect is stronger in x_t than x_e
            ratio ~  1  - effect is similarly visible in both streams
            ratio >> 1  - effect is primarily cross-stream (x_e dominant)

        A "successful x_t intervention" whose ratio >> 1 requires a
        different mechanistic interpretation than one with ratio ~ 1.
"""
''',
    "gpt_oss_interp/steering/controls.py": '''\
"""Null and matched control interventions for causal validation.

Produces baselines:
    - shuffled head labels
    - same-norm random token-slice directions
    - within-family label permutation
    - low-ranked channel controls

All controls must accept the same interface as real interventions
so comparison code stays symmetric.
"""
''',
    "gpt_oss_interp/steering/reporting.py": '''\
"""Steering-specific report writers and figure generators.

Thin wrappers over common.io; steering-specific plotting goes here,
not in common.
"""
''',
    # -- distillation ---------------------------------------------------------
    "gpt_oss_interp/distillation/__init__.py": "",
    "gpt_oss_interp/distillation/config.py": '''\
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
''',
    "gpt_oss_interp/distillation/teacher_artifacts.py": '''\
"""Teacher artifact extraction: final distributions, layerwise readouts.

Generates the fixed-position metadata and distribution targets
consumed by cascade_targets.py.
"""
''',
    "gpt_oss_interp/distillation/cascade_targets.py": '''\
"""Closed-form CASCADE target construction (x_e* computation).

Depends on: distillation.teacher_artifacts, common.artifacts
"""
''',
    "gpt_oss_interp/distillation/train.py": '''\
"""Student training loop.

Stage order (per CASCADE_DISTILLATION.md):
    1. regression warmup on x_e*
    2. KL refinement against teacher distribution
"""
''',
    "gpt_oss_interp/distillation/evaluate.py": '''\
"""Teacher-vs-student comparison.

Reports:
    - teacher-relative benchmark accuracy
    - local steering preservation
    - tail fraction comparison
    - collateral damage comparison
"""
''',
    # -- common ---------------------------------------------------------------
    "gpt_oss_interp/common/__init__.py": "",
    "gpt_oss_interp/common/artifacts.py": '''\
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
''',
    "gpt_oss_interp/common/run_types.py": '''\
"""Shared run/config dataclasses used by both subpackages.

100-line rule: if a type here is only imported by one subpackage,
move it there.
"""
''',
    "gpt_oss_interp/common/io.py": '''\
"""JSON read/write helpers and report path conventions.

Mirrors existing reports/writers.py; consolidate gradually.
"""
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
''',
    "gpt_oss_interp/common/models.py": '''\
"""Checkpoint and model metadata interfaces.

Provides a registry-style lookup so backends and training code
reference models by id rather than raw paths.
"""
''',
}


def main() -> None:
    skipped: list[str] = []
    created: list[str] = []

    for rel_path, content in STUBS.items():
        path = ROOT / rel_path
        if path.exists():
            skipped.append(rel_path)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        created.append(rel_path)

    for rel_path in sorted(created):
        print(f"  created: {rel_path}")
    for rel_path in sorted(skipped):
        print(f"  skip (exists): {rel_path}")

    print(f"\nScaffold complete: {len(created)} created, {len(skipped)} skipped.")
    print("No existing files were modified.")
    print()
    print("Next steps:")
    print("  1. Commit this scaffold as a standalone commit.")
    print("  2. Copy content of interventions/specs.py into steering/specs.py.")
    print("  3. Write probing.py Phase 1 implementation natively in steering/probing.py.")
    print("  4. Only then replace interventions/specs.py with a shim.")


if __name__ == "__main__":
    main()
