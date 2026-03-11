"""Per-channel differential probing.

Entry point for Phase 1 of PER_CHANNEL_XT_INTERVENTION_PLAN.

New code goes here directly. Do not accumulate it in scripts/.

Public API (to be implemented):
    probe_channel_preferences(model, cases, config) -> ChannelProbeResult
        Computes per-layer, per-head slice preference scores:
            pref(l, h) = <x_t[l, h], e_A,h - e_B,h>
        Includes null baselines:
            - shuffled head labels
            - matched same-norm random token-slice directions
            - within-family label permutation

    rank_channels(result) -> ChannelRanking
        Ranks channels by differential preference shift across
        minimal pairs. Reports stability, selectivity, and
        position sensitivity per channel.

    promote_channels(ranking, threshold) -> list[ChannelHypothesis]
        Promotes channels that exceed threshold on held-out minimal
        pairs. Threshold must be set before running, not post-hoc.
        Suggested first bar: sign prediction accuracy > 0.70.
"""
