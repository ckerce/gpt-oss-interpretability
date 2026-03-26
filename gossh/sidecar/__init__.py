"""gossh.sidecar — MoE router sidecar subsystem.

The sidecar solves the MXFP4 router opacity problem: fused MXFP4 kernels
execute router + expert dispatch as a single native operation, making
token-to-expert routing invisible to Python-level hooks.  The sidecar holds
bf16 copies of the router weights (which are never quantized) in a separate
process and infers routing decisions from the residual stream captured via
pre-hooks on the MoE input.
"""
from .process import MoeSidecar
from .dequant import RouterWeightExtractor, RouterSidecarModel
from .validation import (
    SidecarValidationReport,
    LayerValidationStats,
    run_full_validation,
    validate_numerical,
    validate_sensitivity,
    format_validation_report,
)

__all__ = [
    "MoeSidecar",
    "RouterWeightExtractor",
    "RouterSidecarModel",
    "SidecarValidationReport",
    "LayerValidationStats",
    "run_full_validation",
    "validate_numerical",
    "validate_sensitivity",
    "format_validation_report",
]
