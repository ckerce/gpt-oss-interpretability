# Analysis Set Stratification

## Overall Counts

| Rule | Stratum | Count |
| --- | --- | ---: |
| strict | correct_stable | 3 |
| strict | correct_unstable | 8 |
| strict | incorrect_early_expected | 5 |
| soft_tail_4 | correct_late_stable | 10 |
| soft_tail_4 | correct_late_unstable | 1 |
| soft_tail_4 | incorrect_early_expected | 5 |

## By Task: Strict Rule

| Task | correct_stable | correct_unstable | incorrect_early_expected | incorrect_never_expected |
| --- | ---: | ---: | ---: | ---: |
| capitalization | 2 | 1 | 1 | 0 |
| coreference | 0 | 4 | 2 | 0 |
| induction | 1 | 3 | 2 | 0 |

## By Task: Soft Rule (final streak >= 4)

| Task | correct_late_stable | correct_late_unstable | incorrect_early_expected | incorrect_never_expected |
| --- | ---: | ---: | ---: | ---: |
| capitalization | 3 | 0 | 1 | 0 |
| coreference | 3 | 1 | 2 | 0 |
| induction | 4 | 0 | 2 | 0 |

## Case Details

| Task | Case | Strict | Soft | Final Winner | First Expected | Last Expected | Final Streak | Flips |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| capitalization | caps_005 | correct_stable | correct_late_stable | A | 13 | 25 | 13 | 1 |
| capitalization | caps_006 | correct_stable | correct_late_stable | A | 0 | 25 | 26 | 0 |
| capitalization | caps_007 | correct_unstable | correct_late_stable | A | 8 | 25 | 5 | 7 |
| capitalization | caps_008 | incorrect_early_expected | incorrect_early_expected | B | 0 | 19 | 0 | 5 |
| induction | induction_005 | incorrect_early_expected | incorrect_early_expected | B | 7 | 23 | 0 | 4 |
| induction | induction_006 | correct_unstable | correct_late_stable | A | 8 | 25 | 11 | 3 |
| induction | induction_007 | correct_unstable | correct_late_stable | A | 5 | 25 | 6 | 5 |
| induction | induction_008 | correct_stable | correct_late_stable | A | 0 | 25 | 26 | 0 |
| induction | induction_009 | correct_unstable | correct_late_stable | A | 2 | 25 | 5 | 9 |
| induction | induction_010 | incorrect_early_expected | incorrect_early_expected | B | 0 | 24 | 0 | 5 |
| coreference | coref_005 | correct_unstable | correct_late_stable | A | 3 | 25 | 5 | 9 |
| coreference | coref_006 | incorrect_early_expected | incorrect_early_expected | B | 0 | 24 | 0 | 1 |
| coreference | coref_007 | correct_unstable | correct_late_stable | A | 1 | 25 | 8 | 5 |
| coreference | coref_008 | incorrect_early_expected | incorrect_early_expected | B | 4 | 24 | 0 | 6 |
| coreference | coref_009 | correct_unstable | correct_late_unstable | A | 3 | 25 | 3 | 9 |
| coreference | coref_010 | correct_unstable | correct_late_stable | A | 6 | 25 | 6 | 5 |
