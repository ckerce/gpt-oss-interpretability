# Selectivity Comparison Report — C-71

## Summary

- Target family: `recency_bias`
- Off-target families: `coreference, induction`
- Valid target cases: `recency_001, recency_002, recency_003, recency_004`
- Valid off-target cases (19): `induction_001, induction_002, induction_003, induction_004, induction_005, induction_006, induction_007, induction_008, induction_009, induction_010, coref_001, coref_002, coref_003, coref_004, coref_005, coref_006, coref_008, coref_009, coref_010`
- Mean held-out channelized selectivity: `9.9442`
- Mean held-out whole-vector selectivity: `12.4456`
- Channelized wins on held-out cases: `1 / 4`
- Sanity check passes: `True`

## Held-Out Folds

### recency_001

- Channelized row: `L5 H3 scale=-8.0`
- Whole-vector row: `L4 scale=-8.0`
- Channelized selectivity: `5.2911`
- Whole-vector selectivity: `4.2917`
- Random-channel selectivity: `5.4008`
- Random-direction selectivity: `0.4742`
- Channelized beats whole-vector: `True`

### recency_002

- Channelized row: `L5 H3 scale=-8.0`
- Whole-vector row: `L4 scale=-8.0`
- Channelized selectivity: `11.4154`
- Whole-vector selectivity: `14.9503`
- Random-channel selectivity: `8.7745`
- Random-direction selectivity: `1.0325`
- Channelized beats whole-vector: `False`

### recency_003

- Channelized row: `L5 H3 scale=-8.0`
- Whole-vector row: `L4 scale=-8.0`
- Channelized selectivity: `7.8842`
- Whole-vector selectivity: `10.0068`
- Random-channel selectivity: `4.5578`
- Random-direction selectivity: `0.2098`
- Channelized beats whole-vector: `False`

### recency_004

- Channelized row: `L5 H3 scale=-8.0`
- Whole-vector row: `L4 scale=-8.0`
- Channelized selectivity: `15.1860`
- Whole-vector selectivity: `20.5335`
- Random-channel selectivity: `6.5298`
- Random-direction selectivity: `-0.3388`
- Channelized beats whole-vector: `False`

