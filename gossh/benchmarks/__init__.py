"""gossh.benchmarks — benchmark runner and task pools."""
from .runner import BenchmarkRunner, load_config_from_file, load_config_from_yaml

__all__ = ["BenchmarkRunner", "load_config_from_file", "load_config_from_yaml"]
