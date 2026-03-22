# Selectivity Comparison Report — C-71

## Summary

- Target family: `induction`
- Off-target families: `coreference, recency_bias`
- Valid target cases: `induction_001, induction_002, induction_003, induction_004, induction_005, induction_006, induction_007, induction_008, induction_009, induction_010`
- Valid off-target cases (13): `recency_001, recency_002, recency_003, recency_004, coref_001, coref_002, coref_003, coref_004, coref_005, coref_006, coref_008, coref_009, coref_010`
- Mean held-out channelized selectivity: `11.8878`
- Mean held-out whole-vector selectivity: `19.9547`
- Channelized wins on held-out cases: `1 / 10`
- Sanity check passes: `True`

## Held-Out Folds

### induction_001

- Channelized row: `L5 H3 scale=-8.0`
- Whole-vector row: `L4 scale=-4.0`
- Channelized selectivity: `8.7538`
- Whole-vector selectivity: `17.4953`
- Random-channel selectivity: `7.3469`
- Random-direction selectivity: `2.1209`
- Channelized beats whole-vector: `False`

### induction_002

- Channelized row: `L5 H3 scale=-8.0`
- Whole-vector row: `L4 scale=-4.0`
- Channelized selectivity: `19.9622`
- Whole-vector selectivity: `41.0139`
- Random-channel selectivity: `15.2770`
- Random-direction selectivity: `1.2614`
- Channelized beats whole-vector: `False`

### induction_003

- Channelized row: `L5 H3 scale=-8.0`
- Whole-vector row: `L4 scale=-1.0`
- Channelized selectivity: `10.6303`
- Whole-vector selectivity: `17.4032`
- Random-channel selectivity: `10.3894`
- Random-direction selectivity: `1.2541`
- Channelized beats whole-vector: `False`

### induction_004

- Channelized row: `L5 H3 scale=-8.0`
- Whole-vector row: `L4 scale=-4.0`
- Channelized selectivity: `4.1175`
- Whole-vector selectivity: `5.3787`
- Random-channel selectivity: `5.2971`
- Random-direction selectivity: `-0.0101`
- Channelized beats whole-vector: `False`

### induction_005

- Channelized row: `L5 H3 scale=-4.0`
- Whole-vector row: `L4 scale=-4.0`
- Channelized selectivity: `16.8654`
- Whole-vector selectivity: `32.6892`
- Random-channel selectivity: `7.1078`
- Random-direction selectivity: `2.2169`
- Channelized beats whole-vector: `False`

### induction_006

- Channelized row: `L5 H3 scale=-8.0`
- Whole-vector row: `L4 scale=-4.0`
- Channelized selectivity: `17.8632`
- Whole-vector selectivity: `20.9956`
- Random-channel selectivity: `10.1335`
- Random-direction selectivity: `-0.0903`
- Channelized beats whole-vector: `False`

### induction_007

- Channelized row: `L5 H3 scale=-8.0`
- Whole-vector row: `L4 scale=-4.0`
- Channelized selectivity: `10.3195`
- Whole-vector selectivity: `14.1842`
- Random-channel selectivity: `8.1944`
- Random-direction selectivity: `1.6078`
- Channelized beats whole-vector: `False`

### induction_008

- Channelized row: `L5 H3 scale=-8.0`
- Whole-vector row: `L4 scale=-2.0`
- Channelized selectivity: `9.7174`
- Whole-vector selectivity: `21.1471`
- Random-channel selectivity: `4.5488`
- Random-direction selectivity: `0.1011`
- Channelized beats whole-vector: `False`

### induction_009

- Channelized row: `L5 H3 scale=-8.0`
- Whole-vector row: `L4 scale=-2.0`
- Channelized selectivity: `12.6510`
- Whole-vector selectivity: `11.9331`
- Random-channel selectivity: `8.9525`
- Random-direction selectivity: `0.7377`
- Channelized beats whole-vector: `True`

### induction_010

- Channelized row: `L5 H3 scale=-4.0`
- Whole-vector row: `L4 scale=-2.0`
- Channelized selectivity: `7.9979`
- Whole-vector selectivity: `17.3067`
- Random-channel selectivity: `6.0208`
- Random-direction selectivity: `0.8903`
- Channelized beats whole-vector: `False`

