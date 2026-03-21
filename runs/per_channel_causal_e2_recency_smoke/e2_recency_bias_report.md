# Per-Channel Causal Report

Model: `E2`
Family: `recency_bias`

## Summary

- probe vs causal rank Spearman: `0.108`
- held-out accuracy vs causal effect Spearman: `0.067`
- promoted channels: `18`
- promoted mean effect: `1.726`
- low-rank mean effect: `1.222`
- random-control mean effect: `1.517`

## Top Channels

- L5 H5: effect=2.768, best_scale=-2.0, probe_acc=1.000, promoted=True
- L4 H5: effect=2.329, best_scale=-2.0, probe_acc=1.000, promoted=True
- L2 H5: effect=2.254, best_scale=+2.0, probe_acc=1.000, promoted=True
- L3 H5: effect=2.250, best_scale=-2.0, probe_acc=1.000, promoted=True
- L1 H5: effect=2.124, best_scale=-2.0, probe_acc=1.000, promoted=True
- L0 H5: effect=2.027, best_scale=-2.0, probe_acc=0.750, promoted=True
- L5 H3: effect=1.982, best_scale=-2.0, probe_acc=0.500, promoted=False
- L0 H2: effect=1.968, best_scale=+2.0, probe_acc=0.000, promoted=False
- L5 H2: effect=1.964, best_scale=-2.0, probe_acc=0.750, promoted=True
- L1 H2: effect=1.948, best_scale=+2.0, probe_acc=0.000, promoted=False

## Readout Decomposition

- L5 H5 on `recency_002`: combined=-7.060, x_t=-8.358, x_e=+0.000, transfer=0.000
- L4 H5 on `recency_002`: combined=-5.912, x_t=-8.141, x_e=+2.948, transfer=0.362
- L2 H5 on `recency_002`: combined=+5.602, x_t=+6.698, x_e=-3.631, transfer=0.542
