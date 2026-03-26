"""Tests for gossh benchmark runner: config loading, intervention expansion, output."""

import json
import tempfile
from pathlib import Path

import pytest

from gossh.config import (
    BackendKind,
    BenchmarkConfig,
    BenchmarkSweepConfig,
    InterventionKind,
    InterventionSpec,
    InterventionTarget,
    PromptCase,
    PromptTask,
    TargetUnit,
)
from gossh.interventions.specs import InterventionRun, expand_runs
from gossh.reports.writers import summarize, write_case_csv, write_json
from gossh.benchmarks.runner import (
    BenchmarkRunner,
    load_config_from_dict,
    load_config_from_yaml,
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


class TestInterventionExpansion:
    def test_expand_runs_creates_one_run_per_scale(self):
        runs = expand_runs([_simple_intervention()])
        assert len(runs) == 3

    def test_run_names_include_scale(self):
        runs = expand_runs([_simple_intervention()])
        names = [r.run_name() for r in runs]
        assert "head_ablate_L5@0" in names
        assert "head_ablate_L5@0.5" in names
        assert "head_ablate_L5@1" in names

    def test_expand_multiple_specs(self):
        s2 = InterventionSpec(
            name="layer_L10",
            kind=InterventionKind.LAYER_SCALE,
            target=InterventionTarget(TargetUnit.LAYER, (10,), (), ()),
            scales=(0.0,),
        )
        runs = expand_runs([_simple_intervention(), s2])
        assert len(runs) == 4

    def test_empty_specs_returns_empty(self):
        assert expand_runs([]) == []


class TestReportWriters:
    def test_summarize_groups_by_run(self):
        rows = [
            {"run_name": "run_A", "correct": 1, "margin": 2.0, "task_name": "t1"},
            {"run_name": "run_A", "correct": 0, "margin": -1.0, "task_name": "t1"},
            {"run_name": "run_B", "correct": 1, "margin": 3.0, "task_name": "t2"},
        ]
        summary = summarize(rows)
        assert summary["run_A"]["accuracy"] == 0.5
        assert summary["run_A"]["mean_margin"] == 0.5
        assert "run_B" in summary

    def test_write_case_csv(self):
        rows = [
            {"run_name": "r1", "case_id": "c1", "correct": 1, "margin": 2.5},
            {"run_name": "r1", "case_id": "c2", "correct": 0, "margin": -0.5},
        ]
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            path = Path(f.name)
        write_case_csv(path, rows)
        content = path.read_text()
        assert "run_name" in content
        assert "c1" in content
        path.unlink()

    def test_write_json(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            path = Path(f.name)
        write_json(path, {"key": "value", "number": 42})
        loaded = json.loads(path.read_text())
        assert loaded["key"] == "value"
        path.unlink()


class TestDryRunEndToEnd:
    def test_full_dry_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _simple_config(
                experiment_name="pytest_dry_run",
                output_dir=tmpdir,
            )
            runner = BenchmarkRunner(config)
            results = runner.run()

            assert "rows" in results
            assert "summary" in results
            assert len(results["rows"]) > 0

            out = Path(tmpdir)
            assert (out / "case_results.csv").exists()
            assert (out / "summary.json").exists()
            assert (out / "report.md").exists()

    def test_max_examples_limits_cases(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _simple_config(
                sweep=BenchmarkSweepConfig(max_examples=1),
                output_dir=tmpdir,
            )
            runner = BenchmarkRunner(config)
            results = runner.run()
            # 3 scales × 1 case = 3 rows
            assert len(results["rows"]) == 3


class TestYamlConfigLoading:
    def test_load_config_from_dict(self):
        config = _simple_config()
        d = config.to_dict()
        config2 = load_config_from_dict(d)
        assert config2.backend_kind == BackendKind.DRY_RUN
        assert len(config2.tasks) == 1

    def test_load_config_from_yaml_file(self):
        config = _simple_config()
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False, encoding="utf-8") as f:
            f.write(config.to_yaml())
            yaml_path = f.name
        try:
            config2 = load_config_from_yaml(yaml_path)
            assert config2.backend_kind == BackendKind.DRY_RUN
            assert len(config2.tasks) == 1
            assert len(config2.interventions) == 1
        finally:
            Path(yaml_path).unlink()

    def test_yaml_loaded_config_runs(self):
        config = _simple_config()
        d = config.to_dict()
        config2 = load_config_from_dict(d)
        with tempfile.TemporaryDirectory() as tmpdir:
            config2.output_dir = tmpdir
            runner = BenchmarkRunner(config2)
            results = runner.run()
            assert len(results["rows"]) > 0
