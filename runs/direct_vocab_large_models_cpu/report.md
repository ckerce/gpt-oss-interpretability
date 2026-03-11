# Direct Vocabulary Steering Report

Output root: `runs/direct_vocab_large_models_cpu`

## Models

- `dns-dns_S_G_gpt2`: `/mnt/d/mechanistic_interpretability/results/neurips-2026/training/gated_attention/dns-dns_S_G_gpt2/checkpoint_epoch_1.pt`
- `dns-dns_S_G_cascade_gpt2`: `/mnt/d/mechanistic_interpretability/results/neurips-2026/training/gated_attention/dns-dns_S_G_cascade_gpt2/checkpoint_epoch_1.pt`
- `dns-dns_S_G_cascade_pls_xe_gpt2`: `/mnt/d/mechanistic_interpretability/results/neurips-2026/training/gated_attention/dns-dns_S_G_cascade_pls_xe_gpt2/checkpoint_epoch_1.pt`
- `dns-dns_S_G_cascade_pls_xe_codelion_80M`: `/mnt/d/mechanistic_interpretability/results/neurips-2026/training/gated_attention/dns-dns_S_G_cascade_pls_xe_codelion_80M/checkpoint_epoch_1.pt`
- `dns-dns_S_G_cascade_pls_xe_gpt2_base`: `/mnt/d/mechanistic_interpretability/results/neurips-2026/training/gated_attention/dns-dns_S_G_cascade_pls_xe_gpt2_base/checkpoint_epoch_1.pt`

## Baselines

### dns-dns_S_G_gpt2

- `caps_005`: predicted `A`, expected `A`, correct=1, total_gap=19.1060, local_logit_gap=3.3511
- `induction_009`: predicted `A`, expected `A`, correct=1, total_gap=2.8306, local_logit_gap=2.8306
- `coref_010`: predicted `A`, expected `A`, correct=1, total_gap=7.8795, local_logit_gap=7.8795

### dns-dns_S_G_cascade_gpt2

- `caps_005`: predicted `A`, expected `A`, correct=1, total_gap=15.7432, local_logit_gap=2.8890
- `induction_009`: predicted `A`, expected `A`, correct=1, total_gap=2.0833, local_logit_gap=2.0833
- `coref_010`: predicted `A`, expected `A`, correct=1, total_gap=3.2024, local_logit_gap=3.2024

### dns-dns_S_G_cascade_pls_xe_gpt2

- `caps_005`: predicted `A`, expected `A`, correct=1, total_gap=18.5528, local_logit_gap=4.2449
- `induction_009`: predicted `A`, expected `A`, correct=1, total_gap=2.3028, local_logit_gap=2.3029
- `coref_010`: predicted `A`, expected `A`, correct=1, total_gap=6.5898, local_logit_gap=6.5898

### dns-dns_S_G_cascade_pls_xe_codelion_80M

- `caps_005`: predicted `A`, expected `A`, correct=1, total_gap=19.7855, local_logit_gap=6.4063
- `induction_009`: predicted `A`, expected `A`, correct=1, total_gap=1.9866, local_logit_gap=1.9866
- `coref_010`: predicted `A`, expected `A`, correct=1, total_gap=6.2339, local_logit_gap=6.2339

### dns-dns_S_G_cascade_pls_xe_gpt2_base

- `caps_005`: predicted `A`, expected `A`, correct=1, total_gap=24.1790, local_logit_gap=2.7739
- `induction_009`: predicted `A`, expected `A`, correct=1, total_gap=1.2768, local_logit_gap=1.2768
- `coref_010`: predicted `A`, expected `A`, correct=1, total_gap=8.4137, local_logit_gap=8.4137

## Strongest On-Target Effects

### dns-dns_S_G_gpt2

- `caps_005`: best local shift at layer 5 scale -2.0 => -2.4814 (tail_fraction=0.000); sign_correct_fraction=1.000
- `induction_009`: best local shift at layer 4 scale -2.0 => -1.1547 (tail_fraction=0.000); sign_correct_fraction=1.000
- `coref_010`: best local shift at layer 5 scale -2.0 => -0.9414 (tail_fraction=0.000); sign_correct_fraction=1.000

### dns-dns_S_G_cascade_gpt2

- `caps_005`: best local shift at layer 5 scale -2.0 => -3.4387 (tail_fraction=0.000); sign_correct_fraction=1.000
- `induction_009`: best local shift at layer 4 scale -2.0 => -1.1213 (tail_fraction=0.000); sign_correct_fraction=1.000
- `coref_010`: best local shift at layer 5 scale -2.0 => -1.2711 (tail_fraction=0.000); sign_correct_fraction=1.000

### dns-dns_S_G_cascade_pls_xe_gpt2

- `caps_005`: best local shift at layer 5 scale -2.0 => -2.1599 (tail_fraction=0.000); sign_correct_fraction=1.000
- `induction_009`: best local shift at layer 4 scale -2.0 => -1.1616 (tail_fraction=0.000); sign_correct_fraction=1.000
- `coref_010`: best local shift at layer 5 scale -2.0 => -1.1505 (tail_fraction=0.000); sign_correct_fraction=1.000

### dns-dns_S_G_cascade_pls_xe_codelion_80M

- `caps_005`: best local shift at layer 0 scale -2.0 => -0.7423 (tail_fraction=0.033); sign_correct_fraction=1.000
- `induction_009`: best local shift at layer 0 scale 2.0 => 0.4191 (tail_fraction=0.000); sign_correct_fraction=1.000
- `coref_010`: best local shift at layer 5 scale -2.0 => -0.1667 (tail_fraction=0.000); sign_correct_fraction=1.000

### dns-dns_S_G_cascade_pls_xe_gpt2_base

- `caps_005`: best local shift at layer 2 scale -2.0 => -0.8102 (tail_fraction=0.003); sign_correct_fraction=1.000
- `induction_009`: best local shift at layer 5 scale 2.0 => 0.5575 (tail_fraction=0.000); sign_correct_fraction=1.000
- `coref_010`: best local shift at layer 5 scale -2.0 => -0.7417 (tail_fraction=0.000); sign_correct_fraction=1.000

## Notes

- `tail_fraction` is decomposed using local log-prob gap shift so it stays in the same units as total choice-score shift.
- Off-target rows are present in the JSON artifact when source and target case differ.
