# Attention Path Sensitivity Report

This analysis measures how much the attention pattern changes under embedding-level direct-vocabulary interventions in `C-71`.

Intervention stage: `embedding_init`
Scale: `-8.0`

We report mean absolute change in the attention distribution for the final query position at each layer.

## recency_001

Prompt: `The trophy would not fit in the suitcase because the suitcase was too small. The word 'small' refers to the`

Direction: `W[ suitcase] - W[ trophy]`

Layerwise decision-query attention change:

- layer 0: `x_t` delta=0.0024, `x_e` delta=0.0024
- layer 1: `x_t` delta=0.0025, `x_e` delta=0.0025
- layer 2: `x_t` delta=0.0021, `x_e` delta=0.0020
- layer 3: `x_t` delta=0.0042, `x_e` delta=0.0039
- layer 4: `x_t` delta=0.0025, `x_e` delta=0.0024
- layer 5: `x_t` delta=0.0027, `x_e` delta=0.0028

Summary: mean layer delta `x_t`=0.0027, `x_e`=0.0026

## induction_005

Prompt: `sun moon star sun moon star sun moon`

Direction: `W[ star] - W[ cloud]`

Layerwise decision-query attention change:

- layer 0: `x_t` delta=0.0034, `x_e` delta=0.0034
- layer 1: `x_t` delta=0.0046, `x_e` delta=0.0047
- layer 2: `x_t` delta=0.0062, `x_e` delta=0.0068
- layer 3: `x_t` delta=0.0065, `x_e` delta=0.0062
- layer 4: `x_t` delta=0.0073, `x_e` delta=0.0074
- layer 5: `x_t` delta=0.0063, `x_e` delta=0.0061

Summary: mean layer delta `x_t`=0.0057, `x_e`=0.0058

## coref_009

Prompt: `Natalie reminded Jacob that he needed to lock the door. The word 'he' refers to`

Direction: `W[ Jacob] - W[ Natalie]`

Layerwise decision-query attention change:

- layer 0: `x_t` delta=0.0059, `x_e` delta=0.0059
- layer 1: `x_t` delta=0.0042, `x_e` delta=0.0042
- layer 2: `x_t` delta=0.0058, `x_e` delta=0.0059
- layer 3: `x_t` delta=0.0093, `x_e` delta=0.0092
- layer 4: `x_t` delta=0.0062, `x_e` delta=0.0061
- layer 5: `x_t` delta=0.0069, `x_e` delta=0.0067

Summary: mean layer delta `x_t`=0.0064, `x_e`=0.0063

## Interpretation

- Larger values mean the intervention changed the attention routing more strongly at the answer query.
- This does not by itself prove stronger causal influence, but it does show whether the two streams propagate differently into attention.

