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
# Residual-delta late-layer sweep
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
            name="late_delta_L18",
            kind=InterventionKind.LAYER_SCALE,
            target=InterventionTarget(
                unit=TargetUnit.LAYER,
                layer_indices=(18,),
                note="Identity-skip layer 18 by zeroing only the block delta",
            ),
            scales=(0.0,),
            description="Residual-preserving late-layer delta ablation at layer 18",
            params={"preserve_residual": True},
        ),
        InterventionSpec(
            name="late_delta_L19",
            kind=InterventionKind.LAYER_SCALE,
            target=InterventionTarget(
                unit=TargetUnit.LAYER,
                layer_indices=(19,),
                note="Identity-skip layer 19 by zeroing only the block delta",
            ),
            scales=(0.0,),
            description="Residual-preserving late-layer delta ablation at layer 19",
            params={"preserve_residual": True},
        ),
        InterventionSpec(
            name="late_delta_L20",
            kind=InterventionKind.LAYER_SCALE,
            target=InterventionTarget(
                unit=TargetUnit.LAYER,
                layer_indices=(20,),
                note="Identity-skip layer 20 by zeroing only the block delta",
            ),
            scales=(0.0,),
            description="Residual-preserving late-layer delta ablation at layer 20",
            params={"preserve_residual": True},
        ),
        InterventionSpec(
            name="late_delta_L21",
            kind=InterventionKind.LAYER_SCALE,
            target=InterventionTarget(
                unit=TargetUnit.LAYER,
                layer_indices=(21,),
                note="Identity-skip layer 21 by zeroing only the block delta",
            ),
            scales=(0.0,),
            description="Residual-preserving late-layer delta ablation at layer 21",
            params={"preserve_residual": True},
        ),
        InterventionSpec(
            name="late_delta_L22",
            kind=InterventionKind.LAYER_SCALE,
            target=InterventionTarget(
                unit=TargetUnit.LAYER,
                layer_indices=(22,),
                note="Identity-skip layer 22 by zeroing only the block delta",
            ),
            scales=(0.0,),
            description="Residual-preserving late-layer delta ablation at layer 22",
            params={"preserve_residual": True},
        ),
        InterventionSpec(
            name="late_delta_L23",
            kind=InterventionKind.LAYER_SCALE,
            target=InterventionTarget(
                unit=TargetUnit.LAYER,
                layer_indices=(23,),
                note="Identity-skip layer 23 by zeroing only the block delta",
            ),
            scales=(0.0,),
            description="Residual-preserving late-layer delta ablation at layer 23",
            params={"preserve_residual": True},
        ),
    ],
    sweep=BenchmarkSweepConfig(repeats=1, max_examples=None, record_case_outputs=True),
    experiment_name="SoftMain_LateLayer_Delta_Sweep",
    output_dir="./runs/soft_main_late_layer_delta_sweep",
    notes=(
        "Focused late-layer ablation on the soft main-analysis cases using "
        "residual-preserving block-delta scaling."
    ),
)
