# Choice-Relative Convergence Calibration

| Task | Cases | Final Correct Rate | Expected Mean | Expected Std | Expected Range | Final Mean | Final Std | Final Range |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| capitalization | 4 | 0.50 | 7.67 | 8.18 | 0 - 19 | 1.00 | 1.73 | 0 - 4 |
| coreference | 4 | 1.00 | 5.00 | 7.52 | 0 - 18 | 5.00 | 7.52 | 0 - 18 |
| induction | 4 | 1.00 | 4.25 | 7.36 | 0 - 17 | 4.25 | 7.36 | 0 - 17 |
| recency_bias | 4 | 0.25 | 0.00 | 0.00 | 0 - 0 | 7.00 | 6.12 | 0 - 16 |
| syntax_agreement | 4 | 0.50 | 9.50 | 7.76 | 0 - 19 | 4.75 | 8.23 | 0 - 19 |

## Case Details

| Task | Case | Expected | Final Winner | Expected Conv | Final Conv |
| --- | --- | --- | --- | ---: | ---: |
| recency_bias | recency_001 | ` suitcase` | B | 0 | 3 |
| recency_bias | recency_002 | ` bag` | B | 0 | 16 |
| recency_bias | recency_003 | ` mouse` | B | 0 | 9 |
| recency_bias | recency_004 | ` wall` | A | 0 | 0 |
| capitalization | caps_001 | ` Rings` | B | 19 | 0 |
| capitalization | caps_002 | ` Carolina` | A | 0 | 0 |
| capitalization | caps_003 | ` York` | A | 4 | 4 |
| capitalization | caps_004 | ` States` | B | None | 0 |
| induction | induction_001 | ` D4` | A | 0 | 0 |
| induction | induction_002 | ` green` | A | 17 | 17 |
| induction | induction_003 | ` 4` | A | 0 | 0 |
| induction | induction_004 | ` gamma` | A | 0 | 0 |
| coreference | coref_001 | ` Alice` | A | 0 | 0 |
| coreference | coref_002 | ` teacher` | A | 1 | 1 |
| coreference | coref_003 | ` Mary` | A | 18 | 18 |
| coreference | coref_004 | ` mother` | A | 1 | 1 |
| syntax_agreement | syntax_001 | ` are` | A | 0 | 0 |
| syntax_agreement | syntax_002 | ` barks` | A | 19 | 19 |
| syntax_agreement | syntax_003 | ` were` | B | 15 | 0 |
| syntax_agreement | syntax_004 | ` runs` | B | 4 | 0 |
