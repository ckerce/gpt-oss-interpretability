# Per-Channel Causal Report

Model: `E2`
Family: `recency_bias`

## Summary

- probe vs causal rank Spearman: `0.221`
- held-out accuracy vs causal effect Spearman: `0.162`
- promoted channels: `18`
- promoted mean effect: `5.373`
- low-rank mean effect: `4.332`
- random-control mean effect: `4.942`

## Top Channels

- L5 H5: effect=6.660, best_scale=-8.0, probe_acc=1.000, promoted=True
- L0 H5: effect=6.462, best_scale=-8.0, probe_acc=0.750, promoted=True
- L4 H5: effect=6.329, best_scale=-8.0, probe_acc=1.000, promoted=True
- L1 H5: effect=6.312, best_scale=-8.0, probe_acc=1.000, promoted=True
- L3 H5: effect=6.283, best_scale=-8.0, probe_acc=1.000, promoted=True
- L2 H5: effect=6.247, best_scale=-8.0, probe_acc=1.000, promoted=True
- L5 H3: effect=5.809, best_scale=-8.0, probe_acc=0.500, promoted=False
- L5 H4: effect=5.728, best_scale=+8.0, probe_acc=0.000, promoted=False
- L5 H2: effect=5.635, best_scale=-8.0, probe_acc=0.750, promoted=True
- L2 H4: effect=5.623, best_scale=+8.0, probe_acc=0.750, promoted=False

## Readout Decomposition

- L5 H5 on `recency_002`: combined=-14.710, x_t=-16.110, x_e=+0.000, transfer=0.000
- L0 H5 on `recency_002`: combined=-15.022, x_t=-16.758, x_e=+9.173, transfer=0.547
- L4 H5 on `recency_002`: combined=-14.394, x_t=-16.073, x_e=+5.856, transfer=0.364
