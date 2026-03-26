"""Tests for gossh.config — types, serialization, and YAML round-trip."""

import pytest
import yaml

from gossh.config import (
    BackendKind,
    BenchmarkConfig,
    BenchmarkSweepConfig,
    InterventionKind,
    InterventionSpec,
    InterventionTarget,
    ModelKind,
    PromptCase,
    PromptTask,
    TargetUnit,
)


def _simple_task():
    return PromptTask(
        name="test_task",
        behavior="test",
        cases=[
            PromptCase("t1", "prompt A", {"A": " yes", "B": " no"}, "A", {}),
            PromptCase("t2", "prompt B", {"A": " yes", "B": " no"}, "B", {}),
        ],
        description="test task",
    )


def _simple_intervention():
    return InterventionSpec(
        name="head_ablate_L5",
        kind=InterventionKind.HEAD_MASK,
        target=InterventionTarget(
            unit=TargetUnit.HEAD,
            layer_indices=(5,),
            head_indices=(0, 1),
            expert_indices=(),
        ),
        scales=(0.0, 0.5, 1.0),
        description="test ablation",
    )


def _simple_config(**kwargs):
    defaults = dict(
        backend_kind=BackendKind.DRY_RUN,
        backend_params={},
        tasks=[_simple_task()],
        interventions=[_simple_intervention()],
        sweep=BenchmarkSweepConfig(repeats=1, max_examples=None, record_case_outputs=True),
        seed=42,
        experiment_name="test",
        output_dir="./test_output",
    )
    defaults.update(kwargs)
    return BenchmarkConfig(**defaults)


class TestModelKind:
    def test_enum_values(self):
        assert ModelKind.GPT_OSS_20B.value == "gpt_oss_20b"
        assert ModelKind.GPT_OSS_120B.value == "gpt_oss_120b"


class TestInterventionTarget:
    def test_signature_head(self):
        target = InterventionTarget(
            unit=TargetUnit.HEAD,
            layer_indices=(5,),
            head_indices=(0, 1),
        )
        sig = target.signature()
        assert "head" in sig
        assert "L5" in sig
        assert "H0,1" in sig

    def test_signature_model(self):
        target = InterventionTarget(unit=TargetUnit.MODEL)
        assert target.signature() == "model"


class TestInterventionSpec:
    def test_signature_format(self):
        spec = _simple_intervention()
        sig = spec.signature()
        assert "head_mask" in sig
        assert "head_ablate_L5" in sig

    def test_default_scales(self):
        spec = InterventionSpec(
            name="x",
            kind=InterventionKind.LAYER_SCALE,
            target=InterventionTarget(unit=TargetUnit.MODEL),
        )
        assert spec.scales == (0.0, 0.5, 1.0, 1.5)


class TestConfigSerialization:
    def test_to_dict_backend_kind(self):
        config = _simple_config()
        d = config.to_dict()
        assert d["backend_kind"] == "dry_run"

    def test_to_dict_tasks(self):
        config = _simple_config()
        d = config.to_dict()
        assert len(d["tasks"]) == 1
        assert d["tasks"][0]["name"] == "test_task"

    def test_to_dict_interventions(self):
        config = _simple_config()
        d = config.to_dict()
        assert len(d["interventions"]) == 1
        assert d["interventions"][0]["kind"] == "head_mask"

    def test_yaml_round_trip(self):
        config = _simple_config()
        yaml_str = config.to_yaml()
        d = yaml.safe_load(yaml_str)
        config2 = BenchmarkConfig.from_dict(d)

        assert config2.backend_kind == config.backend_kind
        assert config2.seed == config.seed
        assert config2.experiment_name == config.experiment_name
        assert len(config2.tasks) == len(config.tasks)
        assert len(config2.interventions) == len(config.interventions)

    def test_from_dict_restores_enums(self):
        config = _simple_config()
        d = config.to_dict()
        config2 = BenchmarkConfig.from_dict(d)
        assert config2.backend_kind == BackendKind.DRY_RUN
        assert config2.interventions[0].kind == InterventionKind.HEAD_MASK
        assert config2.interventions[0].target.unit == TargetUnit.HEAD

    def test_from_dict_restores_tuples(self):
        config = _simple_config()
        d = config.to_dict()
        config2 = BenchmarkConfig.from_dict(d)
        assert isinstance(config2.interventions[0].scales, tuple)
        assert isinstance(config2.interventions[0].target.layer_indices, tuple)
        assert isinstance(config2.interventions[0].target.head_indices, tuple)

    def test_sweep_defaults_preserved(self):
        config = _simple_config()
        d = config.to_dict()
        config2 = BenchmarkConfig.from_dict(d)
        assert config2.sweep.repeats == 1
        assert config2.sweep.max_examples is None
        assert config2.sweep.record_case_outputs is True
