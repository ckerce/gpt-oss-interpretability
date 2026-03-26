"""gossh.common — shared artifact schemas and IO helpers."""
from .artifacts import RunArtifact, SCHEMA_VERSION
from .io import load_json, save_json

__all__ = ["RunArtifact", "SCHEMA_VERSION", "load_json", "save_json"]
