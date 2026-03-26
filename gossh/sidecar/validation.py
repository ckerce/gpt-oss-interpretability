"""Three-layer validation for the MoE sidecar.

Layer 1 — Numerical tap-off:
    Compare sidecar routing decisions against a reference (e.g., from a
    non-quantized bf16 run with output_router_logits=True).  Reports
    expert-selection exact-match rate, expert-weight MAE, and gate-logit
    KL divergence per layer.

Layer 2 — Routing sensitivity:
    Perturb input hidden states by Gaussian noise and measure how much the
    top-expert assignment changes.  High sensitivity at a layer indicates
    that the sidecar's bf16 clone may diverge from MXFP4 routing due to
    floating-point differences near decision boundaries.

Layer 3 — Cross-reference (numerical × semantic):
    Cross-reference layers where numerical mismatch exceeds threshold against
    sensitivity scores to classify each layer as:
    - ``robust``: mismatch < threshold → safe for analysis
    - ``noisy``: mismatch >= threshold but sensitivity is low → quantization
      tolerance; usable with caution
    - ``sensitive``: mismatch >= threshold and sensitivity is high → flag as
      unreliable; recommend fraction-scaling fallback with a warning
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from gossh.capture.router_capture import RouterDecision
from gossh.sidecar.process import MoeSidecar


###############################################################################
# Report types
###############################################################################

@dataclass
class LayerValidationStats:
    """Per-layer validation statistics."""
    layer_idx: int

    # Layer 1: Numerical
    top1_match_rate: float = float("nan")       # fraction of tokens with same top-1 expert
    topk_match_rate: float = float("nan")       # fraction of tokens with same top-k set
    weight_mae: float = float("nan")            # mean |sidecar_weight - ref_weight|
    logit_kl: float = float("nan")              # KL(ref_probs || sidecar_probs)

    # Layer 2: Sensitivity
    sensitivity: float = float("nan")           # fraction of tokens whose top-1 changes under noise

    # Layer 3: Classification
    status: str = "unknown"                     # "robust" | "noisy" | "sensitive" | "unknown"
    note: str = ""


@dataclass
class SidecarValidationReport:
    """Aggregate validation report for all layers."""
    per_layer: list[LayerValidationStats] = field(default_factory=list)
    mismatch_threshold: float = 0.05
    passed: bool = False
    flagged_layers: list[int] = field(default_factory=list)
    summary: str = ""


###############################################################################
# Layer 1: Numerical validation
###############################################################################

def validate_numerical(
    sidecar: MoeSidecar,
    reference_decisions: list[RouterDecision],
    hidden_states: dict[int, torch.Tensor],
    mismatch_threshold: float = 0.05,
) -> list[LayerValidationStats]:
    """Compare sidecar routing against reference decisions.

    Args:
        sidecar: Running MoeSidecar instance.
        reference_decisions: RouterDecision list from a bf16 run.
        hidden_states: ``{layer_idx: Tensor[seq, hidden]}`` from InputCapture.
        mismatch_threshold: Fraction of tokens that may be routed differently.

    Returns:
        List of LayerValidationStats, one per layer present in both inputs.
    """
    ref_by_layer = {d.layer_idx: d for d in reference_decisions}
    sidecar_decisions = sidecar.route(hidden_states)
    sidecar_by_layer = {d.layer_idx: d for d in sidecar_decisions}

    stats_list: list[LayerValidationStats] = []
    common_layers = sorted(set(ref_by_layer) & set(sidecar_by_layer))

    for layer_idx in common_layers:
        ref = ref_by_layer[layer_idx]
        sid = sidecar_by_layer[layer_idx]
        stats = LayerValidationStats(layer_idx=layer_idx)

        # Top-1 expert match
        ref_top1 = ref.selected_experts[..., 0]
        sid_top1 = sid.selected_experts[..., 0]
        n_tokens = ref_top1.numel()
        stats.top1_match_rate = float((ref_top1 == sid_top1).sum()) / max(n_tokens, 1)

        # Top-k set match (order-insensitive)
        if ref.selected_experts.shape == sid.selected_experts.shape:
            ref_sets = ref.selected_experts.sort(dim=-1).values
            sid_sets = sid.selected_experts.sort(dim=-1).values
            stats.topk_match_rate = float((ref_sets == sid_sets).all(dim=-1).sum()) / max(n_tokens, 1)

        # Expert weight MAE
        if ref.expert_weights.shape == sid.expert_weights.shape:
            stats.weight_mae = float((ref.expert_weights - sid.expert_weights).abs().mean())

        # Gate logit KL divergence (ref || sidecar)
        if ref.gate_logits is not None and sid.gate_logits is not None:
            ref_probs = F.softmax(ref.gate_logits.float(), dim=-1).clamp(min=1e-8)
            sid_probs = F.softmax(sid.gate_logits.float(), dim=-1).clamp(min=1e-8)
            kl = (ref_probs * (ref_probs / sid_probs).log()).sum(dim=-1).mean()
            stats.logit_kl = float(kl)

        # Threshold check
        mismatch = 1.0 - stats.top1_match_rate
        if mismatch < mismatch_threshold:
            stats.status = "robust"
        else:
            stats.status = "noisy"  # refined in Layer 3

        stats_list.append(stats)

    return stats_list


###############################################################################
# Layer 2: Sensitivity validation
###############################################################################

def validate_sensitivity(
    sidecar: MoeSidecar,
    hidden_states: dict[int, torch.Tensor],
    noise_scale: float = 0.01,
    n_trials: int = 3,
) -> dict[int, float]:
    """Estimate routing sensitivity to small perturbations.

    For each layer, applies Gaussian noise (std = noise_scale × hidden_norm)
    and measures the fraction of tokens whose top-1 expert assignment changes.

    Args:
        sidecar: Running MoeSidecar instance.
        hidden_states: ``{layer_idx: Tensor[seq, hidden]}`` from InputCapture.
        noise_scale: Noise std as a fraction of the per-layer hidden-state norm.
        n_trials: Number of noise trials to average.

    Returns:
        ``{layer_idx: sensitivity_score}`` where score is the mean fraction of
        tokens with a different top-1 expert after perturbation.
    """
    # Baseline routing
    baseline = {d.layer_idx: d for d in sidecar.route(hidden_states)}

    sensitivity: dict[int, float] = {}
    for layer_idx, hidden in sorted(hidden_states.items()):
        if layer_idx not in baseline:
            continue

        base_top1 = baseline[layer_idx].selected_experts[..., 0]
        n_tokens = base_top1.numel()
        total_flip_rate = 0.0

        scale = noise_scale * hidden.norm(dim=-1, keepdim=True).mean()

        for _ in range(n_trials):
            noise = torch.randn_like(hidden) * scale
            perturbed = {layer_idx: hidden + noise}
            perturbed_decisions = sidecar.route(perturbed)
            if not perturbed_decisions:
                continue
            perturbed_top1 = perturbed_decisions[0].selected_experts[..., 0]
            flip_rate = float((perturbed_top1 != base_top1).sum()) / max(n_tokens, 1)
            total_flip_rate += flip_rate

        sensitivity[layer_idx] = total_flip_rate / n_trials

    return sensitivity


###############################################################################
# Layer 3: Cross-reference and classification
###############################################################################

def _classify_layers(
    stats_list: list[LayerValidationStats],
    sensitivity: dict[int, float],
    mismatch_threshold: float,
    sensitivity_threshold: float = 0.10,
) -> None:
    """Update stats in-place with Layer 3 classification."""
    for stats in stats_list:
        mismatch = 1.0 - stats.top1_match_rate
        sens = sensitivity.get(stats.layer_idx, float("nan"))
        stats.sensitivity = sens

        if mismatch < mismatch_threshold:
            stats.status = "robust"
            stats.note = ""
        elif math.isnan(sens) or sens < sensitivity_threshold:
            stats.status = "noisy"
            stats.note = "Numerical mismatch present but routing is insensitive; usable with caution"
        else:
            stats.status = "sensitive"
            stats.note = (
                "Numerical mismatch correlates with routing sensitivity; "
                "recommend fraction-scaling fallback for this layer"
            )


###############################################################################
# Full validation pipeline
###############################################################################

def run_full_validation(
    sidecar: MoeSidecar,
    reference_decisions: list[RouterDecision],
    hidden_states: dict[int, torch.Tensor],
    mismatch_threshold: float = 0.05,
    noise_scale: float = 0.01,
) -> SidecarValidationReport:
    """Run all three validation layers and return a summary report."""
    # Layer 1
    stats_list = validate_numerical(
        sidecar, reference_decisions, hidden_states, mismatch_threshold
    )

    # Layer 2
    sensitivity = validate_sensitivity(sidecar, hidden_states, noise_scale)

    # Layer 3
    _classify_layers(stats_list, sensitivity, mismatch_threshold)

    flagged = [s.layer_idx for s in stats_list if s.status == "sensitive"]
    n_robust = sum(1 for s in stats_list if s.status == "robust")
    passed = len(flagged) == 0

    summary_lines = [
        f"Sidecar validation: {'PASSED' if passed else 'FAILED'}",
        f"  Layers checked: {len(stats_list)}",
        f"  Robust: {n_robust}  Noisy: {sum(1 for s in stats_list if s.status == 'noisy')}  Sensitive: {len(flagged)}",
    ]
    if flagged:
        summary_lines.append(f"  Flagged layers: {flagged}")

    return SidecarValidationReport(
        per_layer=stats_list,
        mismatch_threshold=mismatch_threshold,
        passed=passed,
        flagged_layers=flagged,
        summary="\n".join(summary_lines),
    )


###############################################################################
# Formatting
###############################################################################

def format_validation_report(report: SidecarValidationReport) -> str:
    """Format a validation report as a human-readable table."""
    lines = [report.summary, ""]
    header = f"{'Layer':>5}  {'Top1%':>6}  {'TopK%':>6}  {'WeightMAE':>10}  {'KL':>8}  {'Sens':>6}  Status"
    lines.append(header)
    lines.append("-" * len(header))

    for s in sorted(report.per_layer, key=lambda x: x.layer_idx):
        top1 = f"{s.top1_match_rate * 100:.1f}" if not math.isnan(s.top1_match_rate) else "  n/a"
        topk = f"{s.topk_match_rate * 100:.1f}" if not math.isnan(s.topk_match_rate) else "  n/a"
        mae = f"{s.weight_mae:.4f}" if not math.isnan(s.weight_mae) else "       n/a"
        kl = f"{s.logit_kl:.4f}" if not math.isnan(s.logit_kl) else "     n/a"
        sens = f"{s.sensitivity:.3f}" if not math.isnan(s.sensitivity) else "   n/a"
        lines.append(f"{s.layer_idx:>5}  {top1:>6}  {topk:>6}  {mae:>10}  {kl:>8}  {sens:>6}  {s.status}")

    return "\n".join(lines)
