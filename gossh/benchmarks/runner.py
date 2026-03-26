"""Benchmark runner for GOSSH intervention sweeps."""
from __future__ import annotations

import argparse
import importlib.util
import random
from pathlib import Path
from typing import Any

import yaml

from gossh.backends.base import BackendScore
from gossh.backends.dry_run import DryRunBackend
from gossh.backends.gpt_oss import GPTOSSTransformersBackend
from gossh.config import BackendKind, BenchmarkConfig
from gossh.interventions.specs import expand_runs
from gossh.reports.writers import summarize, write_case_csv, write_json, write_markdown


###############################################################################
# Scoring helpers
###############################################################################

def _score_choice_logprobs(choice_logprobs: dict[str, float], expected_label: str) -> dict[str, Any]:
    predicted_label = max(choice_logprobs, key=choice_logprobs.get)
    expected = choice_logprobs[expected_label]
    best = choice_logprobs[predicted_label]
    sorted_values = sorted(choice_logprobs.values(), reverse=True)
    runner_up = sorted_values[1] if len(sorted_values) > 1 else best
    return {
        "predicted_label": predicted_label,
        "expected_label": expected_label,
        "correct": int(predicted_label == expected_label),
        "expected_logprob": expected,
        "best_logprob": best,
        "margin": (best - runner_up) if predicted_label == expected_label else (expected - best),
    }


###############################################################################
# Config loading
###############################################################################

def load_config_from_file(config_path: str) -> BenchmarkConfig:
    """Load a BenchmarkConfig from a Python file that defines ``config``."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    spec = importlib.util.spec_from_file_location("gossh_config", config_file)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    if not hasattr(module, "config"):
        raise AttributeError(f"Config file {config_path} must define `config`.")
    config = module.config
    if not isinstance(config, BenchmarkConfig):
        raise TypeError("Config object must be a BenchmarkConfig instance.")
    config._config_file = str(config_file.resolve())
    config._repo_root = str(config_file.resolve().parent.parent)
    return config


def load_config_from_yaml(yaml_path: str) -> BenchmarkConfig:
    """Load a BenchmarkConfig from a YAML file."""
    yaml_file = Path(yaml_path)
    if not yaml_file.exists():
        raise FileNotFoundError(f"YAML config not found: {yaml_path}")
    with yaml_file.open("r", encoding="utf-8") as fh:
        d = yaml.safe_load(fh)
    config = BenchmarkConfig.from_dict(d)
    config._config_file = str(yaml_file.resolve())
    config._repo_root = str(yaml_file.resolve().parent.parent)
    return config


def load_config_from_dict(d: dict[str, Any]) -> BenchmarkConfig:
    """Load a BenchmarkConfig from a plain dict."""
    return BenchmarkConfig.from_dict(d)


###############################################################################
# Runner
###############################################################################

class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        random.seed(config.seed)
        self.backend = self._build_backend(config)

    def _build_backend(self, config: BenchmarkConfig):
        if config.backend_kind == BackendKind.DRY_RUN:
            return DryRunBackend(**config.backend_params)
        if config.backend_kind == BackendKind.GPT_OSS_TRANSFORMERS:
            return GPTOSSTransformersBackend(**config.backend_params)
        raise ValueError(f"Unsupported backend kind: {config.backend_kind}")

    def run(self) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        runs = expand_runs(self.config.interventions)

        for run in runs:
            self.backend.clear_interventions()
            self.backend.apply_intervention(run.spec, run.scale)
            run_name = run.run_name()
            for task in self.config.tasks:
                max_ex = self.config.sweep.max_examples
                cases = task.cases[:max_ex] if max_ex else task.cases
                for case in cases:
                    backend_score = self.backend.score_case(case)
                    scored = _score_choice_logprobs(backend_score.choice_logprobs, case.expected_label)
                    rows.append(
                        {
                            "run_name": run_name,
                            "task_name": task.name,
                            "behavior": task.behavior,
                            "case_id": case.case_id,
                            "intervention": run.spec.signature(),
                            "scale": run.scale,
                            **scored,
                        }
                    )
            self.backend.clear_interventions()

        summary = summarize(rows)
        output_dir = Path(self.config.output_dir)
        repo_root = Path(getattr(self.config, "_repo_root", Path.cwd()))
        if not output_dir.is_absolute():
            output_dir = repo_root / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        write_case_csv(output_dir / "case_results.csv", rows)
        write_json(output_dir / "summary.json", {"config": self.config.to_dict(), "summary": summary})
        write_markdown(output_dir / "report.md", self.config.experiment_name, summary)
        return {"rows": rows, "summary": summary}


###############################################################################
# CLI
###############################################################################

def main() -> int:
    parser = argparse.ArgumentParser(description="Run GOSSH intervention benchmark")
    parser.add_argument("--config", required=True, help="Path to config file (.py or .yaml)")
    args = parser.parse_args()

    config_path = args.config
    if config_path.endswith((".yaml", ".yml")):
        config = load_config_from_yaml(config_path)
    else:
        config = load_config_from_file(config_path)

    runner = BenchmarkRunner(config)
    payload = runner.run()
    print(f"Benchmark complete: {config.experiment_name}")
    for run_name, stats in sorted(payload["summary"].items()):
        print(f"  {run_name}: acc={stats['accuracy']:.3f} margin={stats['mean_margin']:.3f}")
    return 0
