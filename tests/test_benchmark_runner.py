"""Tests for benchmark harness: config, intervention expansion, and output."""

import json
import tempfile
from pathlib import Path

import pytest

from gpt_oss_interp.config import (
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
from gpt_oss_interp.interventions.specs import InterventionRun, expand_runs
from gpt_oss_interp.reports.writers import summarize, write_case_csv, write_json


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
            layer_indices=[5],
            head_indices=[0, 1],
            expert_indices=[],
        ),
        scales=(0.0, 0.5, 1.0),
        description="test ablation",
    )


class TestInterventionExpansion:
    def test_expand_runs_creates_one_run_per_scale(self):
        spec = _simple_intervention()
        runs = expand_runs([spec])
        assert len(runs) == 3

    def test_run_names_include_scale(self):
        spec = _simple_intervention()
        runs = expand_runs([spec])
        names = [r.run_name() for r in runs]
        assert "head_ablate_L5@0" in names
        assert "head_ablate_L5@0.5" in names
        assert "head_ablate_L5@1" in names

    def test_expand_multiple_specs(self):
        s1 = _simple_intervention()
        s2 = InterventionSpec(
            name="layer_L10",
            kind=InterventionKind.LAYER_SCALE,
            target=InterventionTarget(TargetUnit.LAYER, [10], [], []),
            scales=(0.0,),
            description="test",
        )
        runs = expand_runs([s1, s2])
        assert len(runs) == 4  # 3 + 1

    def test_empty_specs_returns_empty(self):
        assert expand_runs([]) == []


class TestConfigSerialization:
    def test_config_to_dict(self):
        config = BenchmarkConfig(
            backend_kind=BackendKind.DRY_RUN,
            backend_params={},
            tasks=[_simple_task()],
            interventions=[_simple_intervention()],
            sweep=BenchmarkSweepConfig(repeats=1, max_examples=None, record_case_outputs=True),
            seed=42,
            experiment_name="test",
            output_dir="./test_output",
        )
        d = config.to_dict()
        assert d["backend_kind"] == "dry_run"
        assert len(d["tasks"]) == 1
        assert len(d["interventions"]) == 1

    def test_intervention_signature(self):
        spec = _simple_intervention()
        sig = spec.signature()
        assert "HEAD_MASK" in sig or "head_mask" in sig.lower() or "L5" in sig


class TestReportWriters:
    def test_summarize_groups_by_run(self):
        rows = [
            {"run_name": "run_A", "correct": 1, "margin": 2.0, "task_name": "t1"},
            {"run_name": "run_A", "correct": 0, "margin": -1.0, "task_name": "t1"},
            {"run_name": "run_B", "correct": 1, "margin": 3.0, "task_name": "t2"},
        ]
        summary = summarize(rows)
        assert "run_A" in summary
        assert "run_B" in summary
        assert summary["run_A"]["accuracy"] == 0.5
        assert summary["run_A"]["mean_margin"] == 0.5

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
    """Run the full benchmark pipeline with the dry-run backend."""

    def test_full_dry_run(self):
        from gpt_oss_interp.benchmarks.runner import BenchmarkRunner

        with tempfile.TemporaryDirectory() as tmpdir:
            config = BenchmarkConfig(
                backend_kind=BackendKind.DRY_RUN,
                backend_params={},
                tasks=[_simple_task()],
                interventions=[_simple_intervention()],
                sweep=BenchmarkSweepConfig(
                    repeats=1, max_examples=None, record_case_outputs=True
                ),
                seed=42,
                experiment_name="pytest_dry_run",
                output_dir=tmpdir,
            )
            runner = BenchmarkRunner(config)
            results = runner.run()

            assert "rows" in results
            assert "summary" in results
            assert len(results["rows"]) > 0

            # Check output files were created
            out = Path(tmpdir)
            assert (out / "case_results.csv").exists()
            assert (out / "summary.json").exists()
            assert (out / "report.md").exists()
