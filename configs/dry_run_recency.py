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
    backend_kind=BackendKind.DRY_RUN,
    backend_params={
        "behavior_bias": {
            "recency_bias": 1.4,
            "capitalization": 1.2,
            "induction": 1.6,
            "coreference": 1.3,
            "syntax_agreement": 1.1,
        }
    },
    tasks=[
        recency_bias_task(),
        capitalization_task(),
        induction_task(),
        coreference_task(),
        syntax_agreement_task(),
    ],
    interventions=[
        InterventionSpec(
            name="heads_L4H0H1",
            kind=InterventionKind.HEAD_MASK,
            target=InterventionTarget(
                unit=TargetUnit.HEAD,
                layer_indices=(4,),
                head_indices=(0, 1),
                note="Synthetic head sweep",
            ),
            scales=(0.0, 0.5, 1.0, 1.5),
            description="Head masking sweep",
        ),
        InterventionSpec(
            name="experts_L8E0E1",
            kind=InterventionKind.EXPERT_MASK,
            target=InterventionTarget(
                unit=TargetUnit.EXPERT,
                layer_indices=(8,),
                expert_indices=(0, 1),
                note="Synthetic expert sweep",
            ),
            scales=(0.0, 0.5, 1.0, 1.5),
            description="Expert masking sweep",
        ),
        InterventionSpec(
            name="temperature_sweep",
            kind=InterventionKind.TEMPERATURE_SCALE,
            target=InterventionTarget(
                unit=TargetUnit.MODEL,
                note="Model-wide temperature sweep",
            ),
            scales=(0.5, 1.0, 2.0, 4.0),
            description="Temperature scaling sweep",
        ),
    ],
    sweep=BenchmarkSweepConfig(repeats=1, max_examples=None, record_case_outputs=True),
    experiment_name="DryRun_Full_Suite",
    output_dir="./runs/dry_run_full",
    notes="Smoke-test config with all 5 task families and 3 intervention types.",
)
