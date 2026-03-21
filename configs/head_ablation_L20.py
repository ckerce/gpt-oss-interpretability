"""Per-head ablation sweep at layer 20 on the soft main-analysis set.

Ablates each of the 64 query heads individually at layer 20.
Tests the Hydra hypothesis: if ablation-effect variance is low,
the model has distributed redundancy (Hydra active).

Expected result: tight variance (Hydra active), since gpt-oss-20b
is standard-trained without per-layer supervision.

Reference: PLS paper Table 2 — PLS σ=0.47 vs control σ=0.08
"""

from gpt_oss_interp.benchmarks.pools import LEGACY_SOFT_MAIN_CASE_IDS, filter_tasks_by_case_ids
from gpt_oss_interp.config import (
    BackendKind,
    BenchmarkConfig,
    BenchmarkSweepConfig,
    InterventionKind,
    InterventionSpec,
    InterventionTarget,
    TargetUnit,
)


def _filtered_tasks():
    return filter_tasks_by_case_ids(LEGACY_SOFT_MAIN_CASE_IDS)


def _head_ablation_specs():
    """Create one HEAD_MASK intervention per query head at layer 20."""
    specs = []
    for head_idx in range(64):
        specs.append(
            InterventionSpec(
                name=f"head_L20_H{head_idx:02d}",
                kind=InterventionKind.HEAD_MASK,
                target=InterventionTarget(
                    unit=TargetUnit.HEAD,
                    layer_indices=(20,),
                    head_indices=(head_idx,),
                    note=f"Zero head {head_idx} at layer 20",
                ),
                scales=(0.0,),
                description=f"Ablate head {head_idx} at layer 20",
            )
        )
    return specs


config = BenchmarkConfig(
    backend_kind=BackendKind.GPT_OSS_TRANSFORMERS,
    backend_params={
        "model_name": "openai/gpt-oss-20b",
    },
    tasks=_filtered_tasks(),
    interventions=_head_ablation_specs(),
    sweep=BenchmarkSweepConfig(repeats=1, max_examples=None, record_case_outputs=True),
    experiment_name="HeadAblation_L20_SoftMain",
    output_dir="./runs/head_ablation_L20",
    notes=(
        "Per-head ablation sweep at layer 20 on the soft main-analysis set. "
        "Tests the Hydra hypothesis by measuring ablation-effect variance across 64 heads. "
        "Expected: tight variance (Hydra active) since gpt-oss-20b is standard-trained."
    ),
)
