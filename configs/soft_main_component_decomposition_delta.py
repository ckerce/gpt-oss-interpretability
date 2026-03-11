from gpt_oss_interp.benchmarks.tasks import all_tasks
from gpt_oss_interp.config import (
    BackendKind,
    BenchmarkConfig,
    BenchmarkSweepConfig,
    InterventionKind,
    InterventionSpec,
    InterventionTarget,
    TargetUnit,
)


SOFT_MAIN_CASE_IDS = {
    "caps_002",
    "caps_003",
    "induction_001",
    "induction_002",
    "induction_003",
    "induction_004",
    "coref_002",
    "coref_003",
    "coref_004",
}


###############################################################################
#
# Task filtering
#
###############################################################################

def _filtered_tasks():
    tasks = []
    for task in all_tasks():
        filtered_cases = [case for case in task.cases if case.case_id in SOFT_MAIN_CASE_IDS]
        if filtered_cases:
            task.cases = filtered_cases
            tasks.append(task)
    return tasks


###############################################################################
#
# Residual-delta component decomposition
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
            name="late_delta_L20",
            kind=InterventionKind.LAYER_SCALE,
            target=InterventionTarget(
                unit=TargetUnit.LAYER,
                layer_indices=(20,),
                note="Residual-preserving full block-delta suppression at layer 20",
            ),
            scales=(0.0,),
            description="Residual-delta block ablation at layer 20",
            params={"preserve_residual": True},
        ),
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
            name="late_delta_L21",
            kind=InterventionKind.LAYER_SCALE,
            target=InterventionTarget(
                unit=TargetUnit.LAYER,
                layer_indices=(21,),
                note="Residual-preserving full block-delta suppression at layer 21",
            ),
            scales=(0.0,),
            description="Residual-delta block ablation at layer 21",
            params={"preserve_residual": True},
        ),
        InterventionSpec(
            name="all_heads_L21",
            kind=InterventionKind.HEAD_MASK,
            target=InterventionTarget(
                unit=TargetUnit.HEAD,
                layer_indices=(21,),
                head_indices=tuple(range(64)),
                note="Zero all attention heads at layer 21",
            ),
            scales=(0.0,),
            description="Attention-path ablation at layer 21",
        ),
        InterventionSpec(
            name="all_experts_L21",
            kind=InterventionKind.EXPERT_MASK,
            target=InterventionTarget(
                unit=TargetUnit.EXPERT,
                layer_indices=(21,),
                expert_indices=tuple(range(32)),
                note="Zero all MoE experts at layer 21",
            ),
            scales=(0.0,),
            description="MoE-path ablation at layer 21",
        ),
    ],
    sweep=BenchmarkSweepConfig(repeats=1, max_examples=None, record_case_outputs=True),
    experiment_name="SoftMain_Component_Decomposition_Delta",
    output_dir="./runs/soft_main_component_decomposition_delta",
    notes=(
        "Residual-delta component decomposition on the soft main-analysis set, "
        "comparing full block-delta suppression against attention and MoE hooks "
        "at the corrected late layers 20 and 21."
    ),
)
