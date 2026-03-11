# Decision-Direction Projection: induction_002

Task: `induction`
Behavior: `induction`

## Prompt

`red blue green red blue green red blue`

## Decision Tokenization

- Shared completion prefix tokens: ['<|channel|>', 'final', '<|message|>']
- Divergent token A: ` green`
- Divergent token B: ` red`
- Decision position index: `77`

## Layerwise Decision Direction

| Condition | Layer | Local A-B | Total A-B | Suffix Contribution | Local Pred | Total Pred |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| baseline | 20 | -4.413 | 3.741 | 8.154 | `B` | `A` |
| baseline | 21 | -5.875 | 2.626 | 8.501 | `B` | `A` |
| baseline | 22 | -3.923 | 5.034 | 8.958 | `B` | `A` |
| baseline | 23 | -7.985 | 2.829 | 10.814 | `B` | `A` |
| late_delta_L20@0 | 20 | -4.894 | 2.382 | 7.275 | `B` | `A` |
| late_delta_L20@0 | 21 | -4.780 | 1.602 | 6.382 | `B` | `A` |
| late_delta_L20@0 | 22 | -3.407 | 3.433 | 6.841 | `B` | `A` |
| late_delta_L20@0 | 23 | -8.318 | -1.338 | 6.980 | `B` | `B` |
| all_experts_L20@0 | 20 | -6.033 | 3.018 | 9.051 | `B` | `A` |
| all_experts_L20@0 | 21 | -5.267 | 2.209 | 7.476 | `B` | `A` |
| all_experts_L20@0 | 22 | -3.805 | 4.050 | 7.855 | `B` | `A` |
| all_experts_L20@0 | 23 | -8.471 | -0.206 | 8.265 | `B` | `B` |
| late_delta_L21@0 | 20 | -4.413 | 3.741 | 8.154 | `B` | `A` |
| late_delta_L21@0 | 21 | -4.413 | 3.741 | 8.154 | `B` | `A` |
| late_delta_L21@0 | 22 | -3.054 | 3.170 | 6.225 | `B` | `A` |
| late_delta_L21@0 | 23 | -9.529 | -2.040 | 7.489 | `B` | `B` |
| all_heads_L21@0 | 20 | -4.413 | 3.741 | 8.154 | `B` | `A` |
| all_heads_L21@0 | 21 | -4.037 | 2.993 | 7.029 | `B` | `A` |
| all_heads_L21@0 | 22 | -2.775 | 4.377 | 7.152 | `B` | `A` |
| all_heads_L21@0 | 23 | -9.034 | -0.148 | 8.887 | `B` | `B` |
