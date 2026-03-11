# Direct Vocabulary Steering Demo

Output root: `runs/direct_vocab_demo_matched_71m`

This demo uses exact vocabulary directions `W[token_A] - W[token_B]` on matched 71M symbolic models.

## Recommended Examples

- `coref_009` on `SS-71`:  Jacob vs  Natalie; baseline `A` -> negative-steered `B`
- `induction_005` on `C-71`:  star vs  cloud; baseline `A` -> negative-steered `B`
- `recency_001` on `SS-71`:  suitcase vs  trophy; baseline `A` -> negative-steered `B`
- `recency_001` on `C-71`:  suitcase vs  trophy; baseline `A` -> negative-steered `B`
- `coref_009` on `C-71`:  Jacob vs  Natalie; baseline `A` -> negative-steered `B`
- `induction_005` on `SS-71`:  star vs  cloud; baseline `A` -> negative-steered `B`

## SS-71

### coref_009

Prompt: `Natalie reminded Jacob that he needed to lock the door. The word 'he' refers to`

Choices: `A= Jacob` vs `B= Natalie`; divergent tokens ` Jacob` vs ` Natalie`; clean=True

Baseline: pred=A (expected A), local_gap=+5.242, total_gap=+5.242

- Best positive: layer 5, scale +8.0, local shift +8.664, total gap +13.906 (baseline +5.242), pred=A, tail_fraction=0.000
- Best negative: layer 5, scale -8.0, local shift -9.069, total gap -3.827 (baseline +5.242), pred=B, tail_fraction=0.000
- Flip to B: layer 5, scale -8.0, local shift -9.069, total gap -3.827 (baseline +5.242), pred=B, tail_fraction=0.000

### induction_005

Prompt: `sun moon star sun moon star sun moon`

Choices: `A= star` vs `B= cloud`; divergent tokens ` star` vs ` cloud`; clean=True

Baseline: pred=A (expected A), local_gap=+4.640, total_gap=+4.640

- Best positive: layer 4, scale +8.0, local shift +5.039, total gap +9.679 (baseline +4.640), pred=A, tail_fraction=0.000
- Best negative: layer 5, scale -8.0, local shift -5.089, total gap -0.449 (baseline +4.640), pred=B, tail_fraction=0.000
- Flip to B: layer 5, scale -8.0, local shift -5.089, total gap -0.449 (baseline +4.640), pred=B, tail_fraction=0.000

### recency_001

Prompt: `The trophy would not fit in the suitcase because the suitcase was too small. The word 'small' refers to the`

Choices: `A= suitcase` vs `B= trophy`; divergent tokens ` suitcase` vs ` trophy`; clean=True

Baseline: pred=A (expected A), local_gap=+2.400, total_gap=+2.400

- Best positive: layer 3, scale +8.0, local shift +2.886, total gap +5.286 (baseline +2.400), pred=A, tail_fraction=0.000
- Best negative: layer 3, scale -8.0, local shift -3.062, total gap -0.661 (baseline +2.400), pred=B, tail_fraction=0.000
- Flip to B: layer 3, scale -8.0, local shift -3.062, total gap -0.661 (baseline +2.400), pred=B, tail_fraction=0.000

### coref_006

Prompt: `Megan called Thomas after he sent the address. The word 'he' refers to`

Choices: `A= Thomas` vs `B= Megan`; divergent tokens ` Thomas` vs ` Megan`; clean=True

Baseline: pred=A (expected A), local_gap=+8.520, total_gap=+8.520

- Best positive: layer 5, scale +8.0, local shift +7.352, total gap +15.872 (baseline +8.520), pred=A, tail_fraction=0.000
- Best negative: layer 5, scale -8.0, local shift -7.713, total gap +0.806 (baseline +8.520), pred=A, tail_fraction=0.000
- Flip to B: none in tested layer/scale sweep

## C-71

### coref_009

Prompt: `Natalie reminded Jacob that he needed to lock the door. The word 'he' refers to`

Choices: `A= Jacob` vs `B= Natalie`; divergent tokens ` Jacob` vs ` Natalie`; clean=True

Baseline: pred=A (expected A), local_gap=+5.967, total_gap=+5.967

- Best positive: layer 5, scale +8.0, local shift +6.364, total gap +12.330 (baseline +5.967), pred=A, tail_fraction=0.000
- Best negative: layer 5, scale -8.0, local shift -6.523, total gap -0.556 (baseline +5.967), pred=B, tail_fraction=0.000
- Flip to B: layer 5, scale -8.0, local shift -6.523, total gap -0.556 (baseline +5.967), pred=B, tail_fraction=0.000

### induction_005

Prompt: `sun moon star sun moon star sun moon`

Choices: `A= star` vs `B= cloud`; divergent tokens ` star` vs ` cloud`; clean=True

Baseline: pred=A (expected A), local_gap=+1.816, total_gap=+1.816

- Best positive: layer 4, scale +8.0, local shift +5.338, total gap +7.154 (baseline +1.816), pred=A, tail_fraction=0.000
- Best negative: layer 4, scale -8.0, local shift -5.400, total gap -3.584 (baseline +1.816), pred=B, tail_fraction=0.000
- Flip to B: layer 4, scale -8.0, local shift -5.400, total gap -3.584 (baseline +1.816), pred=B, tail_fraction=0.000

### recency_001

Prompt: `The trophy would not fit in the suitcase because the suitcase was too small. The word 'small' refers to the`

Choices: `A= suitcase` vs `B= trophy`; divergent tokens ` suitcase` vs ` trophy`; clean=True

Baseline: pred=A (expected A), local_gap=+4.197, total_gap=+4.197

- Best positive: layer 3, scale +8.0, local shift +4.589, total gap +8.786 (baseline +4.197), pred=A, tail_fraction=0.000
- Best negative: layer 2, scale -8.0, local shift -4.853, total gap -0.656 (baseline +4.197), pred=B, tail_fraction=0.000
- Flip to B: layer 2, scale -8.0, local shift -4.853, total gap -0.656 (baseline +4.197), pred=B, tail_fraction=0.000

### coref_006

Prompt: `Megan called Thomas after he sent the address. The word 'he' refers to`

Choices: `A= Thomas` vs `B= Megan`; divergent tokens ` Thomas` vs ` Megan`; clean=True

Baseline: pred=A (expected A), local_gap=+11.464, total_gap=+11.464

- Best positive: layer 5, scale +8.0, local shift +5.140, total gap +16.604 (baseline +11.464), pred=A, tail_fraction=0.000
- Best negative: layer 5, scale -8.0, local shift -5.479, total gap +5.985 (baseline +11.464), pred=A, tail_fraction=0.000
- Flip to B: none in tested layer/scale sweep

## Notes

- The best live demo cases are the ones that are tokenizer-clean and baseline-correct on both matched models.
- This demo uses a wider steering sweep than the memo figures because visible choice flips require stronger intervention than simple margin shifts.
- Capitalization was not used here because most proper-noun examples split into dirty sub-token contrasts under GPT-2 tokenization.
- Syntax agreement and most recency cases are weak in these training runs, so they are less reliable as demos.

