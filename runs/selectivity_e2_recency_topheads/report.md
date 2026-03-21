# Selectivity Comparison Report — E2

## Summary

- Target family: `recency_bias`
- Off-target families: `coreference, induction`
- Valid target cases: `recency_001, recency_002, recency_003, recency_004`
- Valid off-target cases (19): `induction_001, induction_002, induction_003, induction_004, induction_005, induction_006, induction_007, induction_008, induction_009, induction_010, coref_001, coref_002, coref_003, coref_004, coref_005, coref_006, coref_008, coref_009, coref_010`
- Mean held-out channelized selectivity: `2.6827`
- Mean held-out whole-vector selectivity: `2.7087`
- Channelized wins on held-out cases: `3 / 4`
- Sanity check passes: `True`

## Held-Out Folds

### recency_001

- Channelized row: `L0 H5 scale=-1.0`
- Whole-vector row: `L0 scale=-2.0`
- Channelized selectivity: `4.0306`
- Whole-vector selectivity: `5.8709`
- Random-channel selectivity: `2.6164`
- Random-direction selectivity: `0.1340`
- Channelized beats whole-vector: `False`

### recency_002

- Channelized row: `L0 H5 scale=-1.0`
- Whole-vector row: `L0 scale=2.0`
- Channelized selectivity: `3.2732`
- Whole-vector selectivity: `2.2829`
- Random-channel selectivity: `2.4887`
- Random-direction selectivity: `-0.2232`
- Channelized beats whole-vector: `True`

### recency_003

- Channelized row: `L0 H5 scale=-1.0`
- Whole-vector row: `L0 scale=-2.0`
- Channelized selectivity: `2.2135`
- Whole-vector selectivity: `1.8289`
- Random-channel selectivity: `1.5443`
- Random-direction selectivity: `-2.0201`
- Channelized beats whole-vector: `True`

### recency_004

- Channelized row: `L0 H5 scale=-1.0`
- Whole-vector row: `L1 scale=-2.0`
- Channelized selectivity: `1.2138`
- Whole-vector selectivity: `0.8519`
- Random-channel selectivity: `0.8204`
- Random-direction selectivity: `0.8854`
- Channelized beats whole-vector: `True`

