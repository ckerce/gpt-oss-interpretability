# Per-Channel Causal Report

Model: `C-71`
Family: `recency_bias`

## Summary

- probe vs causal rank Spearman: `-0.060`
- held-out accuracy vs causal effect Spearman: `-0.096`
- promoted channels: `6`
- promoted mean effect: `0.753`
- low-rank mean effect: `0.652`
- random-control mean effect: `0.747`

## Top Channels

- L5 H5: effect=1.450, best_scale=+8.0, probe_acc=0.000, promoted=False
- L5 H2: effect=1.192, best_scale=-8.0, probe_acc=0.250, promoted=False
- L4 H5: effect=0.918, best_scale=+8.0, probe_acc=0.000, promoted=False
- L5 H3: effect=0.857, best_scale=-8.0, probe_acc=0.500, promoted=False
- L1 H3: effect=0.854, best_scale=-8.0, probe_acc=0.500, promoted=False
- L5 H4: effect=0.853, best_scale=-8.0, probe_acc=1.000, promoted=True
- L3 H5: effect=0.842, best_scale=+8.0, probe_acc=0.000, promoted=False
- L2 H3: effect=0.828, best_scale=-8.0, probe_acc=0.500, promoted=False
- L5 H1: effect=0.822, best_scale=-8.0, probe_acc=0.500, promoted=False
- L3 H3: effect=0.795, best_scale=-8.0, probe_acc=0.500, promoted=False

## Readout Decomposition

- L5 H4 on `recency_003`: combined=-1.112, x_t=-4.843, x_e=+0.000, transfer=0.000
- L4 H4 on `recency_003`: combined=-1.050, x_t=-4.843, x_e=+0.081, transfer=0.017
- L2 H4 on `recency_004`: combined=-1.162, x_t=-5.150, x_e=-0.276, transfer=0.054
