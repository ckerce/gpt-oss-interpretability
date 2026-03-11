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


def _filtered_tasks():
    tasks = []
    for task in all_tasks():
        filtered_cases = [case for case in task.cases if case.case_id in SOFT_MAIN_CASE_IDS]
        if filtered_cases:
            task.cases = filtered_cases
            tasks.append(task)
    return tasks


config = BenchmarkConfig(
    backend_kind=BackendKind.GPT_OSS_TRANSFORMERS,
    backend_params={
        "model_name": "openai/gpt-oss-20b",
    },
    tasks=_filtered_tasks(),
    interventions=[
        InterventionSpec(
            name="late_layer_L18",
            kind=InterventionKind.LAYER_SCALE,
            target=InterventionTarget(
                unit=TargetUnit.LAYER,
                layer_indices=(18,),
                note="Zero layer 18 output on retained soft-main cases",
            ),
            scales=(0.0,),
            description="Late-layer ablation at layer 18",
        ),
        InterventionSpec(
            name="late_layer_L19",
            kind=InterventionKind.LAYER_SCALE,
            target=InterventionTarget(
                unit=TargetUnit.LAYER,
                layer_indices=(19,),
                note="Zero layer 19 output on retained soft-main cases",
            ),
            scales=(0.0,),
            description="Late-layer ablation at layer 19",
        ),
        InterventionSpec(
            name="late_layer_L20",
            kind=InterventionKind.LAYER_SCALE,
            target=InterventionTarget(
                unit=TargetUnit.LAYER,
                layer_indices=(20,),
                note="Zero layer 20 output on retained soft-main cases",
            ),
            scales=(0.0,),
            description="Late-layer ablation at layer 20",
        ),
        InterventionSpec(
            name="late_layer_L21",
            kind=InterventionKind.LAYER_SCALE,
            target=InterventionTarget(
                unit=TargetUnit.LAYER,
                layer_indices=(21,),
                note="Zero layer 21 output on retained soft-main cases",
            ),
            scales=(0.0,),
            description="Late-layer ablation at layer 21",
        ),
        InterventionSpec(
            name="late_layer_L22",
            kind=InterventionKind.LAYER_SCALE,
            target=InterventionTarget(
                unit=TargetUnit.LAYER,
                layer_indices=(22,),
                note="Zero layer 22 output on retained soft-main cases",
            ),
            scales=(0.0,),
            description="Late-layer ablation at layer 22",
        ),
        InterventionSpec(
            name="late_layer_L23",
            kind=InterventionKind.LAYER_SCALE,
            target=InterventionTarget(
                unit=TargetUnit.LAYER,
                layer_indices=(23,),
                note="Zero layer 23 output on retained soft-main cases",
            ),
            scales=(0.0,),
            description="Late-layer ablation at layer 23",
        ),
    ],
    sweep=BenchmarkSweepConfig(repeats=1, max_examples=None, record_case_outputs=True),
    experiment_name="SoftMain_LateLayer_Sweep",
    output_dir="./runs/soft_main_late_layer_sweep",
    notes="Focused late-layer ablation sweep on the soft main-analysis cases only.",
)
