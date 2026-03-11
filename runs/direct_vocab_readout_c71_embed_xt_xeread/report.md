# Direct Vocabulary Steering Report

Output root: `runs/direct_vocab_readout_c71_embed_xt_xeread`

## Models

- `C-71`: `/mnt/d/mechanistic_interpretability/results/neurips-2026/training/gated_attention/dns-dns_S_G_cascade_gpt2/checkpoint_epoch_1.pt`

## Baselines

### C-71

- `recency_001`: predicted `A`, expected `A`, correct=1, total_gap=4.7840, local_logit_gap=4.7840
- `induction_005`: predicted `A`, expected `A`, correct=1, total_gap=1.0956, local_logit_gap=1.0956
- `coref_009`: predicted `A`, expected `A`, correct=1, total_gap=3.3359, local_logit_gap=3.3359

## Strongest On-Target Effects

### C-71

- `recency_001`: best local shift at layer 0 scale -8.0 => -0.2059 (tail_fraction=0.000); sign_correct_fraction=0.600
- `induction_005`: best local shift at layer 0 scale -8.0 => 3.3559 (tail_fraction=0.000); sign_correct_fraction=0.000
- `coref_009`: best local shift at layer 0 scale 8.0 => -4.0194 (tail_fraction=0.000); sign_correct_fraction=0.000

## Notes

- `tail_fraction` is decomposed using local log-prob gap shift so it stays in the same units as total choice-score shift.
- Off-target rows are present in the JSON artifact when source and target case differ.
