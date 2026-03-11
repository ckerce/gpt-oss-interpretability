# Analysis Set Stratification

## Overall Counts

| Rule | Stratum | Count |
| --- | --- | ---: |
| strict | correct_stable | 4 |
| strict | correct_unstable | 9 |
| strict | incorrect_early_expected | 6 |
| strict | incorrect_never_expected | 1 |
| soft_tail_4 | correct_late_stable | 9 |
| soft_tail_4 | correct_late_unstable | 4 |
| soft_tail_4 | incorrect_early_expected | 6 |
| soft_tail_4 | incorrect_never_expected | 1 |

## By Task: Strict Rule

| Task | correct_stable | correct_unstable | incorrect_early_expected | incorrect_never_expected |
| --- | ---: | ---: | ---: | ---: |
| capitalization | 2 | 0 | 1 | 1 |
| coreference | 1 | 3 | 0 | 0 |
| induction | 1 | 3 | 0 | 0 |
| recency_bias | 0 | 1 | 3 | 0 |
| syntax_agreement | 0 | 2 | 2 | 0 |

## By Task: Soft Rule (final streak >= 4)

| Task | correct_late_stable | correct_late_unstable | incorrect_early_expected | incorrect_never_expected |
| --- | ---: | ---: | ---: | ---: |
| capitalization | 2 | 0 | 1 | 1 |
| coreference | 3 | 1 | 0 | 0 |
| induction | 4 | 0 | 0 | 0 |
| recency_bias | 0 | 1 | 3 | 0 |
| syntax_agreement | 0 | 2 | 2 | 0 |

## Case Details

| Task | Case | Strict | Soft | Final Winner | First Expected | Last Expected | Final Streak | Flips |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| recency_bias | recency_001 | incorrect_early_expected | incorrect_early_expected | B | 0 | 2 | 0 | 1 |
| recency_bias | recency_002 | incorrect_early_expected | incorrect_early_expected | B | 0 | 15 | 0 | 1 |
| recency_bias | recency_003 | incorrect_early_expected | incorrect_early_expected | B | 0 | 10 | 0 | 3 |
| recency_bias | recency_004 | correct_unstable | correct_late_unstable | A | 0 | 23 | 1 | 8 |
| capitalization | caps_001 | incorrect_early_expected | incorrect_early_expected | B | 19 | 22 | 0 | 2 |
| capitalization | caps_002 | correct_stable | correct_late_stable | A | 0 | 23 | 24 | 0 |
| capitalization | caps_003 | correct_stable | correct_late_stable | A | 4 | 23 | 20 | 1 |
| capitalization | caps_004 | incorrect_never_expected | incorrect_never_expected | B | None | None | 0 | 0 |
| induction | induction_001 | correct_unstable | correct_late_stable | A | 0 | 23 | 5 | 2 |
| induction | induction_002 | correct_stable | correct_late_stable | A | 17 | 23 | 7 | 1 |
| induction | induction_003 | correct_unstable | correct_late_stable | A | 0 | 23 | 7 | 2 |
| induction | induction_004 | correct_unstable | correct_late_stable | A | 0 | 23 | 7 | 4 |
| coreference | coref_001 | correct_unstable | correct_late_unstable | A | 0 | 23 | 1 | 4 |
| coreference | coref_002 | correct_unstable | correct_late_stable | A | 1 | 23 | 9 | 5 |
| coreference | coref_003 | correct_stable | correct_late_stable | A | 18 | 23 | 6 | 1 |
| coreference | coref_004 | correct_unstable | correct_late_stable | A | 1 | 23 | 7 | 5 |
| syntax_agreement | syntax_001 | correct_unstable | correct_late_unstable | A | 0 | 23 | 1 | 8 |
| syntax_agreement | syntax_002 | correct_unstable | correct_late_unstable | A | 19 | 23 | 3 | 3 |
| syntax_agreement | syntax_003 | incorrect_early_expected | incorrect_early_expected | B | 15 | 19 | 0 | 2 |
| syntax_agreement | syntax_004 | incorrect_early_expected | incorrect_early_expected | B | 4 | 19 | 0 | 6 |
