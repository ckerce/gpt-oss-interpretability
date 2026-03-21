# Phase 1 Channel Probe Initial Report

## Why This Report Exists

This is the first real output of the new per-channel probing workflow in
`gpt_oss_interp/steering/probing.py`.

The goal of the first pass was not to prove the full channelized symbolic
intervention story. It was to answer a simpler question first:

> on which currently available models is Phase 1 probing actually meaningful?

That turned out to matter immediately.

## Models Probed

1. `C-71`
- artifact:
  [c_71_channel_probe.json](runs/channel_probe_c71_phase1/c_71_channel_probe.json)
- report:
  [c_71_report.md](runs/channel_probe_c71_phase1/c_71_report.md)

2. `SS-71`
- artifact:
  [ss_71_channel_probe.json](runs/channel_probe_ss71_phase1/ss_71_channel_probe.json)
- report:
  [ss_71_report.md](runs/channel_probe_ss71_phase1/ss_71_report.md)

3. `E2_independent`
- artifact:
  [e2_channel_probe.json](runs/channel_probe_e2_phase1/e2_channel_probe.json)
- report:
  [e2_report.md](runs/channel_probe_e2_phase1/e2_report.md)

## The Main Result

The matched `71M` pair is **not** the right place to do layerwise `x_t`
probing.

Why:

- `C-71` median `x_t` layer delta is `0.000000` for recency, induction, and
  coreference
- `SS-71` median `x_t` layer delta is also `0.000000` for all three families

That means the symbolic stream we are probing is effectively frozen across
layers in both models. The probe still finds head preferences, but the layer
axis is degenerate. So a table like "L0.H4 vs L5.H4" is not discovering a
layerwise symbolic mechanism there. It is rediscovering the same frozen
symbolic signal six times.

This is not a failure of the probe. It is a correct diagnosis of the matched
`71M` architecture choice.

## Why This Matters

Before this run, the natural temptation was:

- use the matched `71M` pair for everything, because those are the best
  larger-model steering results

After this run, that is no longer the right plan for Phase 1 channel probing.

For the specific question:

> which symbolic channels change across depth and predict later causal potency?

the matched `71M` pair is structurally the wrong substrate.

It can still be useful later for:

- embedding-level or frozen-stream symbolic write tests
- causal head-slice intervention at different insertion depths
- readout decomposition after a symbolic write

But it is not the right model family for discovering layerwise `x_t` channel
roles.

## The First Model Where The Plan Becomes Meaningful

`E2_independent` is the first model in this workflow where the layerwise part
of the plan actually becomes real.

What changed:

- median `x_t` layer delta is about `4.2` to `4.6`, not zero
- the probe therefore sees genuine depth-dependent symbolic variation

That is exactly what we needed from a Phase 1 target.

## What `E2_independent` Says

The first stable signal is **recency**, not induction or coreference.

With `64` null samples:

- `recency_bias` has `18` promoted channels
- `induction` has `0`
- `coreference` has `0`

And one coreference case was filtered out up front because the reduced
tokenizer collapsed the choice pair:

- `coref_007`

So the first honest conclusion is:

- the mutable symbolic stream does show family-specific channel structure
- but in this first small model it is strongest for recency
- induction and coreference do not yet clear the full null-controlled
  promotion gate

That is useful. It means the gate is doing real work instead of simply
promoting whatever looks visually differentiated.

## What This Changes In The Experimental Plan

The next step should split into two tracks.

### Track A: frozen-stream larger models (`SS-71`, `C-71`)

Treat these as:

- symbolic-write and readout-decomposition targets
- not as the main substrate for layerwise `x_t` role discovery

### Track B: mutable-`x_t` symbolic models (`E2_independent` and related)

Treat these as the real Phase 1 probing substrate:

- channel ranking
- null-controlled promotion
- probing-rank vs causal-rank comparison

This is the first place where the full plan can actually be executed as
written.

## Concrete Next Move

Run the next causal follow-up on the promoted `E2_independent` recency
channels:

1. take the top promoted recency channels from
   [e2_channel_probe.json](runs/channel_probe_e2_phase1/e2_channel_probe.json)
2. do one-channel-at-a-time symbolic writes in `x_t`
3. compare against:
- whole-vector baseline
- random-channel control
- low-ranked-channel control
4. measure:
- local token shift
- total shift
- tail fraction
- stream-transfer ratio

That is the cleanest continuation, because it uses the first model/family where
Phase 1 actually produced a nondegenerate promoted set.

## Bottom Line

The first probing wave already answered an important meta-question:

- yes, the per-channel probing pipeline works
- but no, the matched larger `71M` pair is not the right substrate for
  layerwise `x_t` discovery because `x_t` is frozen there
- the first genuinely informative Phase 1 target is `E2_independent`
- and the first family worth pushing into Phase 2 is `recency_bias`
