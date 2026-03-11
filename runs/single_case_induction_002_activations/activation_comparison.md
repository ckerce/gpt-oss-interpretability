# Activation Comparison: induction_002

Task: `induction`
Behavior: `induction`
Layers analyzed: [20, 21, 22, 23]

## Prompt

`red blue green red blue green red blue`

## Baseline Final-Token Readout

- Layer 20: top-1 ` ` (-0.767)
- Layer 21: top-1 `<|channel|>` (-0.055)
- Layer 22: top-1 `<|channel|>` (-0.011)
- Layer 23: top-1 `<|channel|>` (0.000)

## Intervention Comparison

| Intervention | Layer | Cosine to Baseline | Delta Norm | Top-1 Token | Top-1 Logprob |
| --- | ---: | ---: | ---: | --- | ---: |
| late_delta_L20@0 | 20 | 0.959195 | 5344.430176 | `'s` | -1.051 |
| late_delta_L20@0 | 21 | 0.930707 | 6949.705078 | `<|channel|>` | -0.978 |
| late_delta_L20@0 | 22 | 0.946356 | 7901.897949 | `<|channel|>` | -0.110 |
| late_delta_L20@0 | 23 | 0.890538 | 10535.190430 | `<|channel|>` | 0.000 |
| all_experts_L20@0 | 20 | 0.965327 | 4822.587891 | `'s` | -0.986 |
| all_experts_L20@0 | 21 | 0.951619 | 5926.577148 | `<|channel|>` | -0.134 |
| all_experts_L20@0 | 22 | 0.964077 | 6527.701172 | `<|channel|>` | -0.013 |
| all_experts_L20@0 | 23 | 0.929836 | 9063.597656 | `<|channel|>` | 0.000 |
| late_delta_L21@0 | 20 | 1.000001 | 0.000000 | ` ` | -0.767 |
| late_delta_L21@0 | 21 | 0.900659 | 8159.420410 | ` ` | -0.767 |
| late_delta_L21@0 | 22 | 0.935868 | 8643.095703 | `<|channel|>` | -2.928 |
| late_delta_L21@0 | 23 | 0.870567 | 11446.830078 | `<|channel|>` | 0.000 |
| all_heads_L21@0 | 20 | 1.000001 | 0.000000 | ` ` | -0.767 |
| all_heads_L21@0 | 21 | 0.918705 | 7538.596191 | ` ` | -1.582 |
| all_heads_L21@0 | 22 | 0.942688 | 8174.871094 | `<|channel|>` | -2.685 |
| all_heads_L21@0 | 23 | 0.879712 | 11043.328125 | `<|channel|>` | 0.000 |

## Detailed Layer Summaries

### baseline

- Layer 20
  - top tokens: [' ', "'s", ' (', '?', ' #']
  - top logprobs: [-0.767, -2.392, -2.704, -3.454, -3.767]
- Layer 21
  - top tokens: ['<|channel|>', ' ', '...', ' (', ' ...']
  - top logprobs: [-0.055, -4.555, -5.68, -5.805, -5.993]
- Layer 22
  - top tokens: ['<|channel|>', ' ', ' (', ' "', ' intros']
  - top logprobs: [-0.011, -6.761, -7.324, -7.636, -8.136]
- Layer 23
  - top tokens: ['<|channel|>', '<|constrain|>', ' ', '<|message|>', ' (']
  - top logprobs: [0.0, -27.625, -27.812, -28.688, -30.312]

### late_delta_L20@0

- Layer 20
  - top tokens: ["'s", ' ', ' (', ' #', '?']
  - top logprobs: [-1.051, -2.051, -2.551, -3.051, -3.363]
  - metrics: cosine=0.959195, delta_norm=5344.430176, mean_abs_delta=53.688221
- Layer 21
  - top tokens: ['<|channel|>', '  ', ' ???', ' commentary', '??']
  - top logprobs: [-0.978, -2.853, -2.853, -3.041, -3.041]
  - metrics: cosine=0.930707, delta_norm=6949.705078, mean_abs_delta=96.790863
- Layer 22
  - top tokens: ['<|channel|>', ' ', '  ', ' "', ' commentary']
  - top logprobs: [-0.11, -4.423, -4.61, -5.423, -5.548]
  - metrics: cosine=0.946356, delta_norm=7901.897949, mean_abs_delta=108.879143
- Layer 23
  - top tokens: ['<|channel|>', ' ', '<|constrain|>', '<|message|>', 'comment']
  - top logprobs: [0.0, -22.375, -23.125, -24.125, -24.375]
  - metrics: cosine=0.890538, delta_norm=10535.190430, mean_abs_delta=150.706207

### all_experts_L20@0

- Layer 20
  - top tokens: ["'s", ' (', '?', ' ', ' #']
  - top logprobs: [-0.986, -2.048, -2.236, -2.548, -3.486]
  - metrics: cosine=0.965327, delta_norm=4822.587891, mean_abs_delta=46.464241
- Layer 21
  - top tokens: ['<|channel|>', '??', ' ???', '???', ' ']
  - top logprobs: [-0.134, -4.384, -4.384, -4.696, -5.009]
  - metrics: cosine=0.951619, delta_norm=5926.577148, mean_abs_delta=82.851250
- Layer 22
  - top tokens: ['<|channel|>', ' ', ' "', ' invalid', '  ']
  - top logprobs: [-0.013, -6.138, -7.138, -7.263, -7.45]
  - metrics: cosine=0.964077, delta_norm=6527.701172, mean_abs_delta=90.293343
- Layer 23
  - top tokens: ['<|channel|>', ' ', '<|message|>', '<|constrain|>', ' invalid']
  - top logprobs: [0.0, -23.25, -24.812, -25.0, -25.812]
  - metrics: cosine=0.929836, delta_norm=9063.597656, mean_abs_delta=129.703888

### late_delta_L21@0

- Layer 20
  - top tokens: [' ', "'s", ' (', '?', ' #']
  - top logprobs: [-0.767, -2.392, -2.704, -3.454, -3.767]
  - metrics: cosine=1.000001, delta_norm=0.000000, mean_abs_delta=0.000000
- Layer 21
  - top tokens: [' ', "'s", ' (', '?', ' #']
  - top logprobs: [-0.767, -2.392, -2.704, -3.454, -3.767]
  - metrics: cosine=0.900659, delta_norm=8159.420410, mean_abs_delta=101.550865
- Layer 22
  - top tokens: ['<|channel|>', ' ', ' (', "'s", ' "']
  - top logprobs: [-2.928, -3.366, -3.553, -4.522, -4.991]
  - metrics: cosine=0.935868, delta_norm=8643.095703, mean_abs_delta=115.643692
- Layer 23
  - top tokens: ['<|channel|>', '<|constrain|>', ' ', ' (', '<|message|>']
  - top logprobs: [0.0, -21.125, -22.062, -23.688, -24.125]
  - metrics: cosine=0.870567, delta_norm=11446.830078, mean_abs_delta=163.137527

### all_heads_L21@0

- Layer 20
  - top tokens: [' ', "'s", ' (', '?', ' #']
  - top logprobs: [-0.767, -2.392, -2.704, -3.454, -3.767]
  - metrics: cosine=1.000001, delta_norm=0.000000, mean_abs_delta=0.000000
- Layer 21
  - top tokens: [' ', ' (', "'s", ' O', ' #']
  - top logprobs: [-1.582, -2.832, -3.02, -3.395, -3.645]
  - metrics: cosine=0.918705, delta_norm=7538.596191, mean_abs_delta=99.823524
- Layer 22
  - top tokens: ['<|channel|>', ' (', ' ', ' "', "'s"]
  - top logprobs: [-2.685, -3.31, -3.372, -4.185, -4.81]
  - metrics: cosine=0.942688, delta_norm=8174.871094, mean_abs_delta=113.164711
- Layer 23
  - top tokens: ['<|channel|>', '<|constrain|>', ' ', ' (', '<|message|>']
  - top logprobs: [0.0, -21.625, -23.625, -24.75, -25.062]
  - metrics: cosine=0.879712, delta_norm=11043.328125, mean_abs_delta=159.670807
