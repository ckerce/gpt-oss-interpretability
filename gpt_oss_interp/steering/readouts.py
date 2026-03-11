"""Readout decomposition: x_t-only, x_e-only, combined, per-head slice.

Required output of Phase 2 per PER_CHANNEL_XT_INTERVENTION_PLAN.

Public API (to be implemented):
    decompose_readout(model, intervention_result) -> ReadoutDecomposition

    stream_transfer_ratio(decomp) -> float
        Defined as: effect_xe / effect_xt at the decision token,
        where both effects are measured under the same intervention.

        Interpretation:
            ratio << 1  - effect is stronger in x_t than x_e
            ratio ~  1  - effect is similarly visible in both streams
            ratio >> 1  - effect is primarily cross-stream (x_e dominant)

        A "successful x_t intervention" whose ratio >> 1 requires a
        different mechanistic interpretation than one with ratio ~ 1.
"""
