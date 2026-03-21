"""Readout decomposition helpers for Phase 2 causal analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from gpt_oss_interp.steering.interventions import ChoicePrefix, local_gap


@dataclass
class ReadoutDecomposition:
    combined_effect: float
    xt_effect: float
    xe_effect: float


def decompose_readout(
    model,
    prefix: ChoicePrefix,
    device,
    *,
    layer_idx: int,
    head_idx: int,
    vector_slice,
    stream: str = "x_t",
    stage: str = "post_block",
) -> ReadoutDecomposition:
    baseline_combined = local_gap(model, prefix, device, readout_source="combined")
    baseline_xt = local_gap(model, prefix, device, readout_source="x_t")
    baseline_xe = local_gap(model, prefix, device, readout_source="x_e")

    intervened_combined = local_gap(
        model,
        prefix,
        device,
        layer_idx=layer_idx,
        head_idx=head_idx,
        vector_slice=vector_slice,
        stream=stream,
        stage=stage,
        readout_source="combined",
    )
    intervened_xt = local_gap(
        model,
        prefix,
        device,
        layer_idx=layer_idx,
        head_idx=head_idx,
        vector_slice=vector_slice,
        stream=stream,
        stage=stage,
        readout_source="x_t",
    )
    intervened_xe = local_gap(
        model,
        prefix,
        device,
        layer_idx=layer_idx,
        head_idx=head_idx,
        vector_slice=vector_slice,
        stream=stream,
        stage=stage,
        readout_source="x_e",
    )
    return ReadoutDecomposition(
        combined_effect=intervened_combined - baseline_combined,
        xt_effect=intervened_xt - baseline_xt,
        xe_effect=intervened_xe - baseline_xe,
    )


def stream_transfer_ratio(decomp: ReadoutDecomposition, eps: float = 1e-8) -> float:
    return abs(decomp.xe_effect) / max(abs(decomp.xt_effect), eps)


def json_ready(decomp: ReadoutDecomposition) -> dict[str, float]:
    payload = asdict(decomp)
    payload["transfer_ratio"] = stream_transfer_ratio(decomp)
    return payload
