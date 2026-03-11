# Memo Figure Plan

Source artifact:
- `runs/direct_vocab_large_models_cpu/direct_vocab_steering.json`

Primary audience:
- technically fluent reader
- likely skimming
- needs one fast visual for "does this work?", one for "is it selective?", and one for "does CASCADE help?"

Figure set:

1. `fig_baseline_local_gaps`
- Type: grouped bar chart
- Shows: baseline local logit gap for each case across all evaluated models
- Purpose: establish that all large models are baseline-correct and comparable before steering

2. `fig_matched_pair_strength_scans`
- Type: perturbation-strength line plots
- Shows: local logit-gap shift versus intervention scale for the matched 71M pair
- Layout: one panel per case
- Purpose: literature-style steering curve showing bidirectionality and relative effect size

3. `fig_matched_pair_layer_profiles`
- Type: line chart over layers
- Shows: best absolute on-target local shift by layer for the matched 71M pair
- Layout: one panel per case
- Purpose: direct CASCADE vs single-stream comparison in depth

4. `fig_matched_pair_heatmaps`
- Type: heatmap panel
- Shows: local logit-gap shift over `(layer, scale)` for the matched 71M pair
- Layout: 2 models x 3 cases
- Purpose: compact visual summary of sign symmetry, layer preference, and effect magnitude

5. `fig_selectivity_scatter`
- Type: scatter plot
- Shows: strongest on-target effect versus mean off-target drift for all models and source cases
- Purpose: demonstrate collateral discipline and support the claim that the intervention is not just a blunt perturbation

Why these five:
- together they cover baseline competence, steering response curves, architecture comparison, depth behavior, and selectivity
- they reuse the current artifact directly
- they match the visual language common in steering papers better than notebook-style diagnostic plots

What is intentionally omitted:
- tokenization diagrams for `caps_005`
- all-5-model heatmap grid
- dense tables converted into figures without an interpretive gain
