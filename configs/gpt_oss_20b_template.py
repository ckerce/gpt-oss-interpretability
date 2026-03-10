from gpt_oss_interp.benchmarks.tasks import (
    capitalization_task,
    coreference_task,
    induction_task,
    recency_bias_task,
    syntax_agreement_task,
)
from gpt_oss_interp.config import (
    BackendKind,
    BenchmarkConfig,
    BenchmarkSweepConfig,
    InterventionKind,
    InterventionSpec,
    InterventionTarget,
    TargetUnit,
)

config = BenchmarkConfig(
    backend_kind=BackendKind.GPT_OSS_TRANSFORMERS,
    backend_params={
        "model_name": "openai/gpt-oss-20b",
    },
    tasks=[
        recency_bias_task(),
        capitalization_task(),
        induction_task(),
        coreference_task(),
        syntax_agreement_task(),
    ],
    interventions=[
        # Head-level sweep: test early-layer attention heads
        InterventionSpec(
            name="early_heads_L2",
            kind=InterventionKind.HEAD_MASK,
            target=InterventionTarget(
                unit=TargetUnit.HEAD,
                layer_indices=(2,),
                head_indices=(0, 1, 2, 3),
                note="Early-layer head sweep",
            ),
            scales=(0.0, 0.5, 1.0, 1.5),
            description="Suppress/amplify early attention heads",
        ),
        # Head-level sweep: test mid-layer attention heads
        InterventionSpec(
            name="mid_heads_L12",
            kind=InterventionKind.HEAD_MASK,
            target=InterventionTarget(
                unit=TargetUnit.HEAD,
                layer_indices=(12,),
                head_indices=(0, 1, 2, 3),
                note="Mid-layer head sweep",
            ),
            scales=(0.0, 0.5, 1.0, 1.5),
            description="Suppress/amplify mid-layer attention heads",
        ),
        # Expert-level sweep: suppress specific MoE experts
        InterventionSpec(
            name="experts_L8",
            kind=InterventionKind.EXPERT_MASK,
            target=InterventionTarget(
                unit=TargetUnit.EXPERT,
                layer_indices=(8,),
                expert_indices=(0, 1, 2, 3),
                note="Expert sweep at layer 8",
            ),
            scales=(0.0, 0.5, 1.0, 1.5),
            description="Suppress/amplify MoE experts to test routing sensitivity",
        ),
        # Layer-level scaling
        InterventionSpec(
            name="layer_scale_L20",
            kind=InterventionKind.LAYER_SCALE,
            target=InterventionTarget(
                unit=TargetUnit.LAYER,
                layer_indices=(20,),
                note="Late layer ablation",
            ),
            scales=(0.0, 0.5, 1.0, 2.0),
            description="Scale entire late layer output",
        ),
    ],
    sweep=BenchmarkSweepConfig(repeats=1, max_examples=None, record_case_outputs=True),
    experiment_name="GPT_OSS_20B_Intervention_Sweep",
    output_dir="./runs/gpt_oss_20b_sweep",
    notes="Full intervention sweep on gpt-oss-20b with 5 task families.",
)
