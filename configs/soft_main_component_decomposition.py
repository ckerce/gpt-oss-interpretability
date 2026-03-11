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


###############################################################################
#
# Task filtering
#
###############################################################################

def _filtered_tasks():
    return filter_tasks_by_case_ids(LEGACY_SOFT_MAIN_CASE_IDS)


###############################################################################
#
# Late-layer component decomposition
#
###############################################################################

config = BenchmarkConfig(
    backend_kind=BackendKind.GPT_OSS_TRANSFORMERS,
    backend_params={
        "model_name": "openai/gpt-oss-20b",
    },
    tasks=_filtered_tasks(),
    interventions=[
        InterventionSpec(
            name="all_heads_L20",
            kind=InterventionKind.HEAD_MASK,
            target=InterventionTarget(
                unit=TargetUnit.HEAD,
                layer_indices=(20,),
                head_indices=tuple(range(64)),
                note="Zero all attention heads at layer 20",
            ),
            scales=(0.0,),
            description="Attention-path ablation at layer 20",
        ),
        InterventionSpec(
            name="all_experts_L20",
            kind=InterventionKind.EXPERT_MASK,
            target=InterventionTarget(
                unit=TargetUnit.EXPERT,
                layer_indices=(20,),
                expert_indices=tuple(range(32)),
                note="Zero all MoE experts at layer 20",
            ),
            scales=(0.0,),
            description="MoE-path ablation at layer 20",
        ),
        InterventionSpec(
            name="all_heads_L22",
            kind=InterventionKind.HEAD_MASK,
            target=InterventionTarget(
                unit=TargetUnit.HEAD,
                layer_indices=(22,),
                head_indices=tuple(range(64)),
                note="Zero all attention heads at layer 22",
            ),
            scales=(0.0,),
            description="Attention-path ablation at layer 22",
        ),
        InterventionSpec(
            name="all_experts_L22",
            kind=InterventionKind.EXPERT_MASK,
            target=InterventionTarget(
                unit=TargetUnit.EXPERT,
                layer_indices=(22,),
                expert_indices=tuple(range(32)),
                note="Zero all MoE experts at layer 22",
            ),
            scales=(0.0,),
            description="MoE-path ablation at layer 22",
        ),
    ],
    sweep=BenchmarkSweepConfig(repeats=1, max_examples=None, record_case_outputs=True),
    experiment_name="SoftMain_Component_Decomposition",
    output_dir="./runs/soft_main_component_decomposition",
    notes=(
        "Decompose the late-layer effect on the soft main-analysis set into "
        "attention-path versus MoE-path contributions."
    ),
)
