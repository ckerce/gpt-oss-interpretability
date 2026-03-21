# Thread 13: Attention Path Sensitivity

**Status**: Theoretical (proof-of-concept only) — **Objective**: Measure attention path sensitivity

## Question
How sensitive are model outputs to specific attention paths through the network?

## Scripts
- `run_attention_path_sensitivity.py` — attention circuit sensitivity analysis

## Runs (in `runs/`)
- `attention_path_sensitivity/`

## Current state
- Single script, single case, single JSON output (12 KB)
- Proof-of-concept only — demonstrates the analysis is possible but draws no conclusions

## Gaps
- Needs systematic coverage across tasks and cases
- No figures or written analysis
- Unclear how this connects to the main narrative arc

## Package dependencies
`capture.activation_cache`, `backends.transformers_gpt_oss`

## Related threads
- [5-hydra-head-redundancy](../../solid/5-hydra-head-redundancy/) — complementary view of attention head importance
- [2-late-layer-ablation](../../solid/2-late-layer-ablation/) — layer-level ablation that this could decompose further
