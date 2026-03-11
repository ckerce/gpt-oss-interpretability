# Direct Vocabulary Steering Report

Output root: `runs/direct_vocab_stream_ablation_xt`

## Models

- `SS-71`: `/mnt/d/mechanistic_interpretability/results/neurips-2026/training/gated_attention/dns-dns_S_G_gpt2/checkpoint_epoch_1.pt`
- `C-71`: `/mnt/d/mechanistic_interpretability/results/neurips-2026/training/gated_attention/dns-dns_S_G_cascade_gpt2/checkpoint_epoch_1.pt`

## Baselines

### SS-71

- `coref_009`: predicted `A`, expected `A`, correct=1, total_gap=5.2419, local_logit_gap=5.2419
- `induction_005`: predicted `A`, expected `A`, correct=1, total_gap=4.6401, local_logit_gap=4.6401
- `recency_001`: predicted `A`, expected `A`, correct=1, total_gap=2.4002, local_logit_gap=2.4002
- `coref_006`: predicted `A`, expected `A`, correct=1, total_gap=8.5198, local_logit_gap=8.5198

### C-71

- `coref_009`: predicted `A`, expected `A`, correct=1, total_gap=5.9668, local_logit_gap=5.9668
- `induction_005`: predicted `A`, expected `A`, correct=1, total_gap=1.8161, local_logit_gap=1.8161
- `recency_001`: predicted `A`, expected `A`, correct=1, total_gap=4.1972, local_logit_gap=4.1972
- `coref_006`: predicted `A`, expected `A`, correct=1, total_gap=11.4641, local_logit_gap=11.4641

## Strongest On-Target Effects

### SS-71

- `coref_009`: best local shift at layer 5 scale -8.0 => -9.0688 (tail_fraction=0.000); sign_correct_fraction=1.000
- `induction_005`: best local shift at layer 5 scale -8.0 => -5.0894 (tail_fraction=0.000); sign_correct_fraction=1.000
- `recency_001`: best local shift at layer 3 scale -8.0 => -3.0615 (tail_fraction=0.000); sign_correct_fraction=1.000
- `coref_006`: best local shift at layer 5 scale -8.0 => -7.7135 (tail_fraction=0.000); sign_correct_fraction=1.000

### C-71

- `coref_009`: best local shift at layer 5 scale -8.0 => -6.5231 (tail_fraction=0.000); sign_correct_fraction=1.000
- `induction_005`: best local shift at layer 5 scale -8.0 => -5.3749 (tail_fraction=0.000); sign_correct_fraction=1.000
- `recency_001`: best local shift at layer 2 scale -8.0 => -4.9451 (tail_fraction=0.000); sign_correct_fraction=1.000
- `coref_006`: best local shift at layer 5 scale -8.0 => -5.4793 (tail_fraction=0.000); sign_correct_fraction=1.000

## Notes

- `tail_fraction` is decomposed using local log-prob gap shift so it stays in the same units as total choice-score shift.
- Off-target rows are present in the JSON artifact when source and target case differ.
