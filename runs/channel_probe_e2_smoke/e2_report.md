# Channel Probe Report

Run id: `channel_probe_e2_smoke`
Model: `E2`

## Promotion Gate

- held-out sign accuracy threshold: `0.70`
- null margin: `0.10`
- null samples: `8`

## Tokenization Filter

- skipped invalid cases: `1`

## Family Summaries

### recency_bias

- cases: recency_001, recency_002, recency_003, recency_004
- median x_t layer delta: 4.524618
- promoted channels: 16
- top channel: L1 H4
- top held-out sign accuracy: 1.000
- top null ceiling: 0.875

### induction

- cases: induction_001, induction_002, induction_003, induction_004, induction_005, induction_006, induction_007, induction_008, induction_009, induction_010
- median x_t layer delta: 4.197936
- promoted channels: 0
- top channel: L5 H1
- top held-out sign accuracy: 0.500
- top null ceiling: 0.600

### coreference

- cases: coref_001, coref_002, coref_003, coref_004, coref_005, coref_006, coref_008, coref_009, coref_010
- median x_t layer delta: 4.558893
- promoted channels: 0
- top channel: L3 H2
- top held-out sign accuracy: 0.667
- top null ceiling: 0.444

## Promoted Channels

- `recency_bias` L1 H4: held-out acc=1.000, mean_score=+0.519, position_sensitivity=+0.105
- `recency_bias` L5 H5: held-out acc=1.000, mean_score=+0.515, position_sensitivity=-0.136
- `recency_bias` L4 H5: held-out acc=1.000, mean_score=+0.455, position_sensitivity=+0.043
- `recency_bias` L2 H5: held-out acc=1.000, mean_score=+0.420, position_sensitivity=+0.086
- `recency_bias` L4 H0: held-out acc=1.000, mean_score=+0.397, position_sensitivity=+0.125
- `recency_bias` L3 H0: held-out acc=1.000, mean_score=+0.375, position_sensitivity=+0.107
- `recency_bias` L2 H3: held-out acc=1.000, mean_score=+0.355, position_sensitivity=+0.044
- `recency_bias` L3 H5: held-out acc=1.000, mean_score=+0.348, position_sensitivity=+0.000
- `recency_bias` L1 H5: held-out acc=1.000, mean_score=+0.326, position_sensitivity=-0.001
- `recency_bias` L0 H4: held-out acc=1.000, mean_score=+0.268, position_sensitivity=-0.092
- `recency_bias` L1 H3: held-out acc=1.000, mean_score=+0.264, position_sensitivity=-0.051
- `recency_bias` L0 H0: held-out acc=1.000, mean_score=+0.234, position_sensitivity=-0.036
- `recency_bias` L3 H3: held-out acc=0.750, mean_score=+0.451, position_sensitivity=+0.152
- `recency_bias` L0 H3: held-out acc=0.750, mean_score=+0.291, position_sensitivity=-0.039
- `recency_bias` L0 H5: held-out acc=0.750, mean_score=+0.255, position_sensitivity=-0.073
- `recency_bias` L5 H0: held-out acc=0.750, mean_score=+0.110, position_sensitivity=-0.183
