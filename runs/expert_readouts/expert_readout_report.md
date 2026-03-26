# Thread 15: MoE Expert Readout Analysis

Prompts analyzed: 135 across 9 task families

## Measurement 1: Routing Entropy by Layer

| Layer | Entropy (nats) | % of max |
| ---: | ---: | ---: |
| L0 | 3.309 | 95.5% |
| L1 | 3.264 | 94.2% |
| L2 | 3.191 | 92.1% |
| L3 | 3.061 | 88.3% |
| L4 | 3.105 | 89.6% |
| L5 | 2.997 | 86.5% |
| L6 | 3.199 | 92.3% |
| L7 | 3.238 | 93.4% |
| L8 | 3.214 | 92.7% |
| L9 | 3.061 | 88.3% |
| L10 | 3.113 | 89.8% |
| L11 | 3.111 | 89.8% |
| L12 | 3.194 | 92.2% |
| L13 | 3.154 | 91.0% |
| L14 | 3.158 | 91.1% |
| L15 | 3.270 | 94.4% |
| L16 | 3.124 | 90.1% |
| L17 | 3.206 | 92.5% |
| L18 | 3.230 | 93.2% |
| L19 | 3.241 | 93.5% |
| L20 | 3.249 | 93.7% |
| L21 | 3.171 | 91.5% |
| L22 | 3.060 | 88.3% |
| L23 | 2.978 | 85.9% |

## Measurement 2: Layer Logit-Delta (Top Promoted Tokens at Last Position)

| Layer | Top promoted | Top suppressed |
| ---: | --- | --- |
| L1 | `cue`(+19.68) ` XL`(+19.29) `olg`(+17.14) | ` Ocean`(+2.64) ` Kiss`(+2.68) ` Ruff`(+2.70) |
| L2 | ` betre`(+11.77) ` robin`(+11.55) ` Pur`(+11.41) | `_fin`(+1.17) `bie`(+1.20) `息`(+1.31) |
| L3 | `/forums`(+11.84) ` Junge`(+11.65) ` fore`(+11.48) | ` land`(+1.61) ` outlets`(+1.62) `occup`(+1.64) |
| L4 | ` Boyd`(+9.93) ` mine`(+9.93) `Af`(+9.84) | ` unders`(+3.01) ` Underground`(+3.04) `alsa`(+3.08) |
| L5 | ` Rek`(+9.97) ` Sinclair`(+9.66) `enk`(+9.57) | `Rede`(+1.62) ` vat`(+1.64) `olk`(+1.66) |
| L6 | `NOR`(+10.82) ` Weiss`(+10.77) ` iceberg`(+10.63) | ` compet`(+2.34) `cr`(+2.35) ` fl`(+2.36) |
| L7 | `chron`(+6.19) ` TY`(+5.73) ` Mead`(+5.67) | `622`(+0.65) `.as`(+0.65) ` asymmetric`(+0.67) |
| L8 | ` Nad`(+8.96) ` matric`(+8.96) `metic`(+8.90) | ` PU`(+1.02) ` Motiv`(+1.04) ` verke`(+1.04) |
| L9 | `erton`(+8.29) ` NTN`(+7.95) ` Revenge`(+7.64) | `ainte`(-0.88) `tings`(-0.85) ` transformation`(-0.75) |
| L10 | `room`(+8.88) `Asp`(+8.37) `macher`(+8.26) | ` Through`(+2.01) `LV`(+2.02) ` Breda`(+2.05) |
| L11 | `-enh`(+6.20) `Kt`(+6.17) `.decorate`(+6.04) | ` today`(-1.51) ` mixing`(-1.49) `-con`(-1.48) |
| L12 | `ostat`(+10.70) `huk`(+10.58) ` cá`(+10.25) | `Cos`(+3.27) `ily`(+3.27) `nr`(+3.29) |
| L13 | ` Pee`(+6.06) `roj`(+5.72) ` $?`(+5.63) | `_topics`(-1.96) `Topics`(-1.96) `pre`(-1.95) |
| L14 | ` Gug`(+10.07) ` (!)`(+9.70) ` Shay`(+9.45) | `hrer`(+1.71) `iver`(+1.71) `usz`(+1.72) |
| L15 | `Kev`(+6.99) `BRO`(+5.97) `éf`(+5.95) | ` however`(-1.56) ` along`(-1.55) ` within`(-1.48) |
| L16 | ` cult`(+8.31) ` tint`(+8.09) ` BUT`(+7.89) | ` bowed`(-0.21) ` MUST`(-0.19) ` VERY`(-0.16) |
| L17 | ` Done`(+6.61) ` specials`(+5.99) ` oms`(+5.77) | ` Thirty`(-4.83) ` Fif`(-4.78) ` Dw`(-4.70) |
| L18 | ` top`(+9.93) ` now`(+9.76) ` strong`(+9.39) | ` having`(-5.27) ` your`(-5.27) ` blinking`(-5.16) |
| L19 | ` sculpt`(+15.22) ` cat`(+14.94) ` sculpture`(+14.34) | ` true`(-6.12) ` falso`(-6.09) `:false`(-5.96) |
| L20 | ` night`(+10.53) `hein`(+10.14) ` night's`(+9.78) | ` ..."
`(+0.31) ` …..`(+0.34) ` ................`(+0.37) |
| L21 | ` fast`(+20.27) `_fast`(+19.44) ` slow`(+19.33) | `大阪`(-0.97) ` Tokio`(-0.97) `東京都`(-0.82) |
| L22 | `"><?=$`(+12.31) `/'.$`(+12.25) `/')`(+12.21) | ` Dy`(+1.10) `.




`(+1.11) `halter`(+1.16) |
| L23 | `ajat`(+16.63) `argos`(+16.57) `ocat`(+16.44) | `holder`(+0.75) ` shall`(+0.75) ` `(+0.76) |

## Measurement 3: Expert Vocabulary Profiles

_Skipped: no hookable expert modules found (MXFP4 quantization)._
_Use a non-quantized checkpoint to enable per-expert readouts._

## Measurement 4: Information-Theoretic Routing Analysis

### 4a — Specialization gain D_KL(routing ‖ uniform)

_Nats saved by knowing the routing policy vs. guessing uniform._
_Higher = more structured routing at this layer._

| Layer | Spec. gain (nats) | % of max entropy |
| ---: | ---: | ---: |
| L0 | 0.1572 | 4.5% |
| L1 | 0.2020 | 5.8% |
| L2 | 0.2752 | 7.9% |
| L3 | 0.4052 | 11.7% |
| L4 | 0.3609 | 10.4% |
| L5 | 0.4685 | 13.5% |
| L6 | 0.2671 | 7.7% |
| L7 | 0.2282 | 6.6% |
| L8 | 0.2520 | 7.3% |
| L9 | 0.4046 | 11.7% |
| L10 | 0.3530 | 10.2% |
| L11 | 0.3544 | 10.2% |
| L12 | 0.2715 | 7.8% |
| L13 | 0.3117 | 9.0% |
| L14 | 0.3080 | 8.9% |
| L15 | 0.1953 | 5.6% |
| L16 | 0.3417 | 9.9% |
| L17 | 0.2593 | 7.5% |
| L18 | 0.2362 | 6.8% |
| L19 | 0.2249 | 6.5% |
| L20 | 0.2167 | 6.3% |
| L21 | 0.2952 | 8.5% |
| L22 | 0.4057 | 11.7% |
| L23 | 0.4881 | 14.1% |

### 4b — Routing velocity D_KL(routing_l ‖ routing_{l-1})

_Extra nats burned encoding layer l's routing using layer l-1's codebook._
_A spike marks a routing phase transition — where policy changes fastest._

| Layer pair | Velocity (nats) |
| ---: | ---: |
| L0→L1 | 0.48326 |
| L1→L2 | 0.47523 |
| L2→L3 | 0.81667 ← PHASE TRANSITION |
| L3→L4 | 0.64423 |
| L4→L5 | 0.70311 |
| L5→L6 | 0.61527 |
| L6→L7 | 0.47475 |
| L7→L8 | 0.31585 |
| L8→L9 | 0.75550 |
| L9→L10 | 0.49941 |
| L10→L11 | 0.76437 |
| L11→L12 | 0.61432 |
| L12→L13 | 0.53710 |
| L13→L14 | 0.75471 |
| L14→L15 | 0.47354 |
| L15→L16 | 0.76764 |
| L16→L17 | 0.66980 |
| L17→L18 | 0.71355 |
| L18→L19 | 0.41310 |
| L19→L20 | 0.38543 |
| L20→L21 | 0.74076 |
| L21→L22 | 0.79228 |
| L22→L23 | 0.74158 |

### 4c — Task routing mutual information I(expert; task_family)

_Nats of task identity encoded in the routing decision at each layer._
_Near-zero = routing is task-agnostic. Peak = routing encodes task structure._

| Layer | I(expert; task) (nats) |
| ---: | ---: |
| L0 | 0.22938 |
| L1 | 0.25575 |
| L2 | 0.33075 |
| L3 | 0.29027 |
| L4 | 0.35005 |
| L5 | 0.29377 |
| L6 | 0.29722 |
| L7 | 0.30937 |
| L8 | 0.37753 |
| L9 | 0.39684 |
| L10 | 0.44421 |
| L11 | 0.42935 |
| L12 | 0.45955 ← PEAK |
| L13 | 0.42203 |
| L14 | 0.39362 |
| L15 | 0.38940 |
| L16 | 0.43482 |
| L17 | 0.38641 |
| L18 | 0.39403 |
| L19 | 0.40920 |
| L20 | 0.41495 |
| L21 | 0.37391 |
| L22 | 0.34035 |
| L23 | 0.38795 |

### 4d — Routing capacity budget

- Theoretical routing capacity: log₂(C(32,4)) = 15.13 bits = 10.490 nats
- Peak I(expert; task): 0.45955 nats at L12 (4.38% of capacity)

**Interpretation**: At the peak task-routing alignment layer, only 4.4% of the routing mechanism's combinatorial capacity is used for task-discriminating computation. The remainder encodes token surface form, positional context, and other signals not captured by the 5-family task taxonomy. This does not mean routing is inefficient — it means routing is a multi-purpose mechanism, most of whose capacity serves non-task purposes.

