# Per-Channel Causal Report

Model: `C-71`
Family: `induction`

## Summary

- probe vs causal rank Spearman: `-0.363`
- held-out accuracy vs causal effect Spearman: `-0.288`
- promoted channels: `6`
- promoted mean effect: `0.909`
- low-rank mean effect: `0.787`
- random-control mean effect: `1.188`

## Top Channels

- L5 H5: effect=2.852, best_scale=-8.0, probe_acc=0.200, promoted=False
- L5 H2: effect=2.594, best_scale=-8.0, probe_acc=0.600, promoted=False
- L4 H2: effect=1.879, best_scale=-8.0, probe_acc=0.600, promoted=False
- L4 H5: effect=1.819, best_scale=-8.0, probe_acc=0.200, promoted=False
- L3 H2: effect=1.726, best_scale=-8.0, probe_acc=0.600, promoted=False
- L1 H2: effect=1.695, best_scale=-8.0, probe_acc=0.600, promoted=False
- L3 H5: effect=1.606, best_scale=-8.0, probe_acc=0.200, promoted=False
- L1 H5: effect=1.579, best_scale=-8.0, probe_acc=0.200, promoted=False
- L2 H5: effect=1.568, best_scale=-8.0, probe_acc=0.200, promoted=False
- L2 H2: effect=1.563, best_scale=-8.0, probe_acc=0.600, promoted=False

## Readout Decomposition

- L5 H4 on `induction_007`: combined=-1.994, x_t=-6.306, x_e=+0.000, transfer=0.000
- L4 H4 on `induction_002`: combined=-1.718, x_t=-4.156, x_e=-0.098, transfer=0.024
- L3 H4 on `induction_002`: combined=-1.606, x_t=-4.156, x_e=-0.001, transfer=0.000
