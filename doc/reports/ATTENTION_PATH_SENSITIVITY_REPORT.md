# Attention Path Sensitivity Report

## Why We Checked This

One remaining possibility was:

- maybe `x_t` and `x_e` interventions do not look very different at the output
- but they might still propagate differently through attention

So this analysis measured a very specific quantity:

- how much the attention distribution changes at the answer query
- under embedding-level `x_t` vs `x_e` interventions
- in `C-71`, the model where stream differences had started to appear elsewhere

## Setup

Model:

- `C-71`

Intervention stage:

- embedding-level

Scale:

- `-8.0`

Cases:

- `recency_001`
- `induction_005`
- `coref_009`

Metric:

- mean absolute change in the attention distribution for the final query position
- measured layer by layer
- compared against the unperturbed baseline

Artifacts:

- [attention_path_sensitivity.json](runs/attention_path_sensitivity/attention_path_sensitivity.json)
- [report.md](runs/attention_path_sensitivity/report.md)

## The Result

This variant did **not** reveal a meaningful separation.

For all three cases, the attention deltas from `x_t` and `x_e` were almost identical.

Examples:

- `recency_001`
  - mean layer delta `x_t = 0.0027`
  - mean layer delta `x_e = 0.0026`
- `induction_005`
  - mean layer delta `x_t = 0.0057`
  - mean layer delta `x_e = 0.0058`
- `coref_009`
  - mean layer delta `x_t = 0.0064`
  - mean layer delta `x_e = 0.0063`

That is too close to support a strong claim that the two intervention streams are propagating differently into attention routing under this measurement.

## A Concrete Example: `coref_009`

Prompt:

`Natalie reminded Jacob that he needed to lock the door. The word 'he' refers to`

We measured how much the answer-query attention distribution changed at each layer after the intervention.

Layerwise deltas:

- layer 0
  - `x_t = 0.0059`
  - `x_e = 0.0059`
- layer 1
  - `x_t = 0.0042`
  - `x_e = 0.0042`
- layer 2
  - `x_t = 0.0058`
  - `x_e = 0.0059`
- layer 3
  - `x_t = 0.0093`
  - `x_e = 0.0092`
- layer 4
  - `x_t = 0.0062`
  - `x_e = 0.0061`
- layer 5
  - `x_t = 0.0069`
  - `x_e = 0.0067`

This is not a pattern where one stream clearly perturbs attention more than the other.

## What This Means

This is a useful negative result.

It says:

- if the streams differ, this simple attention-change metric is not where the difference shows up clearly

That matters because it tells us not to overclaim.

The earlier reports did uncover some stream-specific effects:

- embedding-level `x_t` vs `x_e`
- readout decomposition
- wrong-position control

But this attention-path metric does not add a new strong separation on top of those.

## How To Read This In Context

The right reading is not:

- "attention does not matter"

The right reading is:

- this particular attention-path summary is too coarse to expose the difference

Possible reasons:

- the real difference may live in values rather than attention weights
- the difference may be localized to specific heads rather than the mean over all heads
- the difference may appear in stream-specific readouts even if routing changes are similar

So this analysis helps narrow the search:

- the next strong evidence is more likely to come from stream readout and intervention placement than from coarse average attention drift

## Bottom Line

This variant is best treated as a negative control on interpretation.

It does **not** support a claim that `x_t` and `x_e` differ sharply in attention-routing effect under this metric.

That is still useful:

- it keeps the story honest
- it prevents us from pretending every internal measure shows the same thing
- it points attention back to the more informative analyses: initialization, readout decomposition, and positional control
