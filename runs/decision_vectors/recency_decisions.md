# Decision Trajectory: recency

**Prompt:** `The trophy would not fit in the suitcase because the suitcase was too`
**Model:** openai/gpt-oss-20b
**Layers:** 24

## Key Decision Positions (13 positions with 2+ transitions)

### Position 8: target = ` the`
- Convergence layer: **17**
- Decision transitions: **12**

| Transition | From → To | Semantic | Top gainers | Top losers |
|---|---|---|---|---|
| L0→L1 | ` inh` → `�` | content→'�' | `Sab` (+15.77), ` generous` (+15.27), ` sober` (+15.02) | ` Sm` (0.1), ` inh` (-1.9), ` Ton` (-13.54) |
| L1→L2 | `�` → ` ce` | content→letter/short | `esc` (+16.35), ` Sab` (+15.35), ` esc` (+14.35) | ` eff` (-1.54), `.sm` (-2.92), ` Sm` (-14.52) |
| L2→L3 | ` ce` → ` sab` | letter/short→'sab' | ` allow` (+15.59), ` rational` (+15.4), `CI` (+15.09) | `�` (-3.33), ` esc` (-14.35), ` generous` (-14.35) |
| L5→L6 | ` sab` → ` rational` | content→'rational' | ` bin` (+16.61), ` exp` (+16.43), ` evaluate` (+16.3) | ` sab` (-2.08), `ace` (-15.75), ` predetermined` (-15.94) |
| L6→L7 | ` rational` → ` abstract` | content→'abstract' | `ace` (+17.17), ` dec` (+16.86), ` cannot` (+16.11) | ` ex` (-1.19), ` rational` (-2.13), `hand` (-15.99) |
| L8→L9 | ` abstract` → ` dec` | content→'dec' | `<|endoftext|>` (+16.87), ` cannot` (+16.68), ` dec` (+0.78) | ` exp` (-0.03), ` abstract` (-0.59), ` bin` (-0.91) |
| L9→L10 | ` dec` → `<|endoftext|>` | content→padding | ` mar` (+15.8), ` allow` (+15.49), `f` (+15.42) | ` evaluate` (-1.19), ` abstract` (-1.69), ` rational` (-16.18) |
| L14→L15 | `<|endoftext|>` → ` ` | padding→padding | ` ` (+17.63), ` and` (+16.82), ` the` (+16.25) | `_` (-14.67), ` not` (-14.86), ` ex` (-15.11) |
| L15→L16 | ` ` → ` (` | padding→punctuation | ` "` (+16.44), `:` (+16.32), ` that` (+16.19) | ` with` (-15.75), `—` (-16.63), ` and` (-16.82) |
| L16→L17 | ` (` → ` the` | punctuation→'the' | ` it` (+14.81), ` '` (+13.74), ` there` (+13.18) | ` (` (-4.57), ``` (-15.26), `,` (-15.69) |
| L19→L20 | ` the` → ` it` | content→letter/short | ` there's` (+10.9), ` its` (+4.13), ` it's` (+3.01) | ` ` (0.82), ` they` (0.76), ` of` (0.13) |
| L22→L23 | ` it` → ` the` | letter/short→'the' | ` a` (+15.27), ` its` (+0.69), ` you` (+0.69) | ` of` (-0.12), ` there` (-0.12), ` it's` (-0.31) |

**Decision arc:** L0: `inh` → L1: `�` → L2: `ce` → L5: `sab` → L6: `rational` → L8: `abstract` → L9: `dec` → L14: `<|endoftext|>` → L15: `` → L16: `(` → L19: `the` → L22: `it` → L23: `the`

### Position 2: target = ` be`
- Convergence layer: **18**
- Decision transitions: **6**

| Transition | From → To | Semantic | Top gainers | Top losers |
|---|---|---|---|---|
| L1→L2 | ` dep` → ` alle` | content→'alle' | ` plaus` (+14.89), ` Fact` (+14.52), `ng` (+14.39) | `Alle` (0.48), ` dep` (-1.27), `ROB` (-12.67) |
| L6→L7 | ` alle` → ` ex` | content→letter/short | ` rough` (+16.26), ` NP` (+16.26), `�` (+16.19) | ` alle` (-0.97), ` plaus` (-15.47), ` Alle` (-15.6) |
| L7→L8 | ` ex` → `<|endoftext|>` | letter/short→padding | ` item` (+16.39), ` rational` (+15.64), ` intangible` (+0.64) | `�` (-0.43), ` verst` (-0.86), ` ex` (-0.99) |
| L15→L16 | `<|endoftext|>` → ` (` | padding→punctuation | ` ` (+17.81), `'` (+17.25), `"` (+16.75) | `_` (-15.84), `A` (-16.09), ` item` (-16.09) |
| L16→L17 | ` (` → ` not` | punctuation→'not' | ` be` (+16.5), ` only` (+16.32), `:` (+16.32) | ` (` (-1.12), `,` (-16.06), ` and` (-16.31) |
| L17→L18 | ` not` → ` be` | content→letter/short | ` have` (+16.42), ` become` (+16.3), ` probably` (+15.61) | ` only` (-16.32), `:` (-16.32), ` for` (-16.94) |

**Decision arc:** L1: `dep` → L6: `alle` → L7: `ex` → L15: `<|endoftext|>` → L16: `(` → L17: `not` → L18: `be`

### Position 3: target = ` be`
- Convergence layer: **18**
- Decision transitions: **9**

| Transition | From → To | Semantic | Top gainers | Top losers |
|---|---|---|---|---|
| L3→L4 | ` succ` → `�` | content→'�' | ` apparent` (+16.87), ` yet` (+16.62), ` verst` (+16.62) | ` Ct` (-15.54), ` há` (-15.66), ` Nations` (-15.91) |
| L5→L6 | `�` → ` ex` | content→letter/short | ` MC` (+15.71), ` quite` (+15.59), ` há` (+15.46) | ` rec` (-0.07), `�` (-1.38), ` succ` (-15.22) |
| L6→L7 | ` ex` → `�` | letter/short→'�' | `<|reserved_200016|>` (+15.28), ` synt` (+14.97), ` fin` (+14.91) | ` rec` (-1.31), ` ordinary` (-15.4), ` há` (-15.46) |
| L8→L9 | `�` → `<|endoftext|>` | content→padding | ` synergy` (+15.19), ` fin` (+15.0), ` verst` (+14.88) | ` division` (-0.59), `�` (-1.09), ` word` (-14.71) |
| L15→L16 | `<|endoftext|>` → ` without` | padding→'without' | ` unless` (+16.5), ` necessarily` (+16.13), ` greatly` (+16.0) | `-st` (-15.68), `‑` (-16.12), `<|reserved_200016|>` (-16.18) |
| L16→L17 | ` without` → ` be` | content→letter/short | ` exist` (+17.17), ` become` (+16.04), ` get` (+15.79) | ` (` (-15.38), ` regret` (-15.5), ` greatly` (-16.0) |
| L17→L18 | ` be` → ` have` | letter/short→'have' | ` have` (+19.51), ` appear` (+14.07), ` change` (+13.7) | ` necessarily` (-3.66), ` being` (-15.54), ` even` (-15.54) |
| L19→L20 | ` have` → ` be` | content→letter/short | ` show` (+14.65), ` give` (+13.9), ` hold` (+13.77) | ` have` (-0.21), ` exist` (-1.33), ` change` (-13.79) |
| L20→L21 | ` be` → ` have` | letter/short→'have' | ` win` (+14.48), ` fit` (+13.67), ` give` (+0.15) | ` help` (-0.22), ` exist` (-0.47), ` become` (-0.79) |

**Decision arc:** L3: `succ` → L5: `�` → L6: `ex` → L8: `�` → L15: `<|endoftext|>` → L16: `without` → L17: `be` → L19: `have` → L20: `be` → L21: `have`

### Position 5: target = ` the`
- Convergence layer: **18**
- Decision transitions: **9**

| Transition | From → To | Semantic | Top gainers | Top losers |
|---|---|---|---|---|
| L1→L2 | ` Vegas` → ` unre` | content→'unre' | ` overhe` (+15.26), ` unre` (+2.77), `Verd` (+1.46) | ` sp` (-0.1), ` Hadd` (-0.42), `ues` (-0.67) |
| L3→L4 | ` unre` → ` sp` | content→letter/short | ` sext` (+16.87), ` denomin` (+16.06), ` breath` (+15.93) | ` unre` (-1.62), ` Bott` (-14.36), `Verd` (-15.92) |
| L4→L5 | ` sp` → ` eng` | letter/short→'eng' | `BBB` (+16.79), ` pack` (+16.73), ` bott` (+1.42) | ` sp` (-0.64), ` unre` (-1.33), ` herm` (-1.7) |
| L6→L7 | ` eng` → ` bott` | content→'bott' | ` breath` (+16.43), ` ex` (+16.18), ` herm` (+15.93) | ` eng` (-1.2), ` leg` (-1.45), ` windows` (-16.2) |
| L7→L8 | ` bott` → ` pack` | content→'pack' | `BBB` (+16.56), `<|reserved_200016|>` (+15.75), ` windows` (+15.63) | ` eng` (-0.93), ` bott` (-2.37), ` herm` (-15.93) |
| L9→L10 | ` pack` → `<|endoftext|>` | content→padding | ` drop` (+15.68), ` exp` (+15.55), `(run` (+0.45) | ` ex` (-0.3), ` breath` (-0.68), ` pack` (-0.99) |
| L15→L16 | `<|endoftext|>` → `.` | padding→punctuation | `.` (+16.33), ` this` (+16.33), ` windows` (+15.14) | `…` (-14.97), ` box` (-15.16), ` shipping` (-15.53) |
| L16→L17 | `.` → ` this` | punctuation→'this' | ` the` (+18.02), ` that` (+17.58), ` above` (+15.15) | ` cannot` (-14.95), ` feas` (-15.02), ` windows` (-15.14) |
| L17→L18 | ` this` → ` the` | content→'the' | ` ` (+14.23), ` any` (+13.8), ` a` (+12.61) | `."` (-14.96), ` with` (-15.02), ` exactly` (-15.08) |

**Decision arc:** L1: `Vegas` → L3: `unre` → L4: `sp` → L6: `eng` → L7: `bott` → L9: `pack` → L15: `<|endoftext|>` → L16: `.` → L17: `this` → L18: `the`

### Position 12: target = ` small`
- Convergence layer: **18**
- Decision transitions: **12**

| Transition | From → To | Semantic | Top gainers | Top losers |
|---|---|---|---|---|
| L0→L1 | ` medd` → ` don` | content→'don' | ` quint` (+17.85), ` ballistic` (+15.85), ` don` (+1.65) | `cant` (-1.48), ` fil` (-1.6), ` medd` (-2.6) |
| L1→L2 | ` don` → ` quint` | content→'quint' | `
                    
` (+16.34), ` herm` (+15.84), ` cushion` (+15.71) | ` don` (-2.13), `cant` (-15.22), ` medd` (-15.72) |
| L4→L5 | ` quint` → ` don` | content→'don' | ` fest` (+17.21), ` sext` (+15.96), ` Dop` (+15.52) | ` wild` (-0.35), ` quint` (-1.1), `ball` (-15.31) |
| L5→L6 | ` don` → ` sink` | content→'sink' | ` fin` (+17.41), ` windows` (+16.78), `pot` (+16.16) | ` wild` (-0.93), ` Dop` (-15.52), `(er` (-16.33) |
| L6→L7 | ` sink` → ` firm` | content→'firm' | `‑` (+16.46), ` drop` (+16.15), ` don` (+16.09) | ` quint` (-0.38), ` fin` (-0.75), `pot` (-16.16) |
| L7→L8 | ` firm` → `‑` | content→'‑' | ` too` (+16.99), `<|endoftext|>` (+16.74), ` pack` (+16.68) | ` quint` (-0.79), ` firm` (-0.97), ` don` (-16.09) |
| L12→L13 | `‑` → ` pack` | content→'pack' | ` inadequate` (+16.6), `<|endoftext|>` (+16.04), ` insuff` (+15.6) | ` drop` (-0.76), `‑` (-1.58), ` firm` (-15.24) |
| L13→L14 | ` pack` → `<|endoftext|>` | content→padding | `ache` (+16.64), ` with` (+15.96), `_h` (+15.96) | ` drop` (-15.54), ` insuff` (-15.6), ` size` (-15.98) |
| L14→L15 | `<|endoftext|>` → `‑` | padding→'‑' | ` not` (+16.21), ` and` (+16.15), `—` (+16.03) | ` mar` (-15.64), ` with` (-15.96), `_h` (-15.96) |
| L15→L16 | `‑` → ` (` | content→punctuation | ` "` (+17.17), ` ` (+17.17), `

` (+16.42) | ` of` (-0.11), ` too` (-1.3), ` dec` (-15.78) |
| L16→L17 | ` (` → ` too` | punctuation→'too' | ` small` (+18.05), ` smaller` (+17.61), ` narrow` (+16.99) | ` of` (-15.85), `

` (-16.42), ` and` (-16.85) |
| L17→L18 | ` too` → ` small` | content→'small' | ` low` (+17.27), ` limited` (+15.39), ` much` (+15.14) | ` close` (-2.72), ` too` (-4.84), ` (` (-15.86) |

**Decision arc:** L0: `medd` → L1: `don` → L4: `quint` → L5: `don` → L6: `sink` → L7: `firm` → L12: `‑` → L13: `pack` → L14: `<|endoftext|>` → L15: `‑` → L16: `(` → L17: `too` → L18: `small`

### Position 4: target = ` in`
- Convergence layer: **19**
- Decision transitions: **9**

| Transition | From → To | Semantic | Top gainers | Top losers |
|---|---|---|---|---|
| L0→L1 | ` gird` → ` don` | content→'don' | ` tightly` (+16.63), ` unre` (+15.51), ` performing` (+15.51) | `Anything` (-15.28), ` alleg` (-15.28), `tes` (-15.65) |
| L2→L3 | ` don` → `extr` | content→'extr' | ` firm` (+15.97), ` breath` (+15.6), `extr` (+1.73) | ` don` (-0.77), ` RE` (-1.02), ` tightly` (-1.52) |
| L6→L7 | `extr` → ` pack` | content→'pack' | ` breath` (+16.05), ` perfectly` (+15.87), ` placement` (+15.55) | ` cr` (-16.25), ` performing` (-16.38), ` firm` (-16.38) |
| L7→L8 | ` pack` → `<|endoftext|>` | content→padding | `<|endoftext|>` (+17.88), `<|reserved_200016|>` (+16.07), `LL` (+15.88) | ` cond` (-2.23), ` placement` (-15.55), ` pocket` (-15.55) |
| L15→L16 | `<|endoftext|>` → ` (` | padding→punctuation | ` (` (+17.65), ` with` (+16.77), `**` (+16.33) | `,` (-15.39), ` cannot` (-15.58), ` and` (-15.89) |
| L17→L18 | ` (` → `;` | punctuation→punctuation | ` in` (+17.37), ` if` (+16.37), ` into` (+16.37) | ` on` (-0.38), `:` (-0.51), ` ` (-16.01) |
| L18→L19 | `;` → ` in` | punctuation→letter/short | ` the` (+16.97), ` in` (+1.66), ` on` (+0.22) | ` into` (-0.15), `;` (-0.34), `.` (-0.9) |
| L20→L21 | ` in` → ` into` | letter/short→'into' | ` into` (+0.41), ` if` (+0.22), ` inside` (+0.16) | ` because` (0.09), ` the` (-0.22), ` any` (-0.22) |
| L21→L22 | ` into` → ` in` | content→letter/short | `;` (+16.14), `.` (+1.03), ` if` (+0.97) | ` the` (0.09), ` in` (-0.28), ` inside` (-0.91) |

**Decision arc:** L0: `gird` → L2: `don` → L6: `extr` → L7: `pack` → L15: `<|endoftext|>` → L17: `(` → L18: `;` → L20: `in` → L21: `into` → L22: `in`

### Position 6: target = ` box`
- Convergence layer: **20**
- Decision transitions: **7**

| Transition | From → To | Semantic | Top gainers | Top losers |
|---|---|---|---|---|
| L0→L1 | ` beh` → ` eng` | content→'eng' | ` eng` (+19.24), ` circ` (+16.74), ` rhetorical` (+16.62) | ` Griffith` (-16.07), ` NC` (-16.19), ` Tens` (-16.32) |
| L1→L2 | ` eng` → ` dist` | content→'dist' | `Verd` (+16.73), ` fin` (+16.6), ` nou` (+16.48) | ` eng` (-1.14), ` fort` (-16.37), ` Ba` (-16.49) |
| L2→L3 | ` dist` → `Verd` | content→'Verd' | ` Verd` (+16.66), ` Ul` (+15.6), `uci` (+15.35) | ` beh` (-1.31), ` dist` (-2.75), ` Easter` (-15.98) |
| L4→L5 | `Verd` → `manship` | content→'manship' | `utum` (+16.25), ` firm` (+15.69), ` Balk` (+15.57) | `Verd` (-1.06), ` con` (-15.69), ` barr` (-15.75) |
| L7→L8 | `manship` → `<|endoftext|>` | content→padding | ` water` (+15.98), `123` (+15.41), `â` (+15.35) | ` Hil` (-15.45), ` mai` (-15.45), ` Lauf` (-16.01) |
| L15→L16 | `<|endoftext|>` → ` ` | padding→padding | ` first` (+16.36), ` while` (+15.68), ` in` (+15.55) | ` ` (-16.03), `_` (-16.35), ` ` (-17.22) |
| L19→L20 | ` ` → ` box` | padding→'box' | ` room` (+15.45), ` storage` (+15.32), ` container` (+2.61) | ` empty` (-0.46), ` ` (-0.83), ` first` (-1.39) |

**Decision arc:** L0: `beh` → L1: `eng` → L2: `dist` → L4: `Verd` → L7: `manship` → L15: `<|endoftext|>` → L19: `` → L20: `box`

### Position 10: target = ` is`
- Convergence layer: **20**
- Decision transitions: **10**

| Transition | From → To | Semantic | Top gainers | Top losers |
|---|---|---|---|---|
| L0→L1 | ` Pon` → `intel` | content→'intel' | ` Pines` (+16.96), ` Polaris` (+16.34), ` primit` (+16.21) | ` accomod` (-16.33), ` Ache` (-16.58), `_tem` (-16.7) |
| L1→L2 | `intel` → ` Sid` | content→'Sid' | ` Sid` (+17.99), `-mus` (+16.74), ` Joy` (+16.74) | `master` (-16.21), ` ec` (-16.46), ` Pines` (-16.96) |
| L2→L3 | ` Sid` → `用品` | content→letter/short | `用品` (+17.58), `izz` (+16.89), ` Heck` (+16.46) | ` clen` (-16.11), `&p` (-16.11), ` vibr` (-16.24) |
| L4→L5 | `用品` → ` commod` | letter/short→'commod' | ` mens` (+15.88), `_tem` (+15.57), ` Joy` (+15.51) | ` heav` (-0.39), `用品` (-0.45), `izz` (-16.14) |
| L7→L8 | ` commod` → `<|endoftext|>` | content→padding | `<|endoftext|>` (+17.55), ` coff` (+16.37), ` compartments` (+15.99) | `itia` (-16.0), ` cond` (-16.0), ` Sid` (-16.06) |
| L12→L13 | `<|endoftext|>` → ` contents` | padding→'contents' | `ier` (+16.8), `容量` (+15.8), ` capacity` (+15.8) | ` contents` (0.1), `<|endoftext|>` (-1.4), ` pack` (-15.2) |
| L13→L14 | ` contents` → `<|endoftext|>` | content→padding | ` set` (+16.49), `<|endoftext|>` (+0.75), ` size` (+0.62) | ` contents` (-0.69), ` constraints` (-0.88), `ier` (-1.25) |
| L15→L16 | `<|endoftext|>` → ` and` | padding→'and' | ` is` (+15.75), ` cannot` (+14.43), ` and` (+1.85) | ` size` (-1.15), ` with` (-1.27), `<|endoftext|>` (-2.52) |
| L16→L17 | ` and` → `'s` | content→''s' | ` was` (+15.46), ` limit` (+14.84), ` only` (+1.78) | `,` (-1.1), ` and` (-2.53), ` (` (-2.66) |
| L19→L20 | `'s` → ` is` | content→letter/short | ` would` (+15.65), ` had` (+14.65), ` could` (+12.53) | ` cannot` (-0.78), `'s` (-1.65), ` (` (-12.55) |

**Decision arc:** L0: `Pon` → L1: `intel` → L2: `Sid` → L4: `用品` → L7: `commod` → L12: `<|endoftext|>` → L13: `contents` → L15: `<|endoftext|>` → L16: `and` → L19: `'s` → L20: `is`

### Position 11: target = ` too`
- Convergence layer: **20**
- Decision transitions: **8**

| Transition | From → To | Semantic | Top gainers | Top losers |
|---|---|---|---|---|
| L3→L4 | ` cur` → ` presumably` | content→'presumably' | ` бы` (+15.36), ` apparent` (+14.73), ` glimps` (+14.67) | ` ras` (-13.2), ` lin` (-13.38), ` isot` (-13.45) |
| L6→L7 | ` presumably` → ` fin` | content→'fin' | ` metaphor` (+15.35), `‑` (+15.28), ` ` (+15.16) | ` presumably` (-1.68), ` бы` (-2.37), ` glimps` (-15.47) |
| L9→L10 | ` fin` → `<|endoftext|>` | content→padding | `<|reserved_200016|>` (+15.12), ` dec` (+15.12), ` pack` (+14.93) | ` metaphor` (-1.5), ` fin` (-2.37), ` raised` (-15.05) |
| L11→L12 | `<|endoftext|>` → ` pack` | padding→'pack' | ` capacity` (+15.94), ` size` (+15.56), `装` (+1.65) | ` ex` (-0.1), ` pack` (-0.92), `<|endoftext|>` (-1.98) |
| L14→L15 | ` pack` → ` at` | content→letter/short | ` (` (+16.8), ` the` (+16.18), ` R` (+16.05) | ` pack` (-1.47), ` ex` (-16.08), ` size` (-16.21) |
| L15→L16 | ` at` → ` not` | letter/short→'not' | ` "` (+18.01), ` a` (+17.57), ` ` (+16.88) | ` at` (-0.92), ` set` (-15.93), ` R` (-16.05) |
| L17→L18 | ` not` → ` only` | content→'only' | ` insufficient` (+13.81), ` only` (+3.11), ` "` (+-0.2) | ` not` (-0.64), ` ` (-2.14), ` the` (-2.14) |
| L19→L20 | ` only` → ` too` | content→'too' | ` limited` (+11.2), ` too` (+1.56), ` already` (+0.43) | ` not` (-1.94), ` insufficient` (-2.32), ` a` (-2.82) |

**Decision arc:** L3: `cur` → L6: `presumably` → L9: `fin` → L11: `<|endoftext|>` → L14: `pack` → L15: `at` → L17: `not` → L19: `only` → L20: `too`

### Position 0: target = ` code`
- Convergence layer: **21**
- Decision transitions: **9**

| Transition | From → To | Semantic | Top gainers | Top losers |
|---|---|---|---|---|
| L0→L1 | `ira` → ` entr` | content→'entr' | `utum` (+16.2), `spe` (+15.83), ` epit` (+15.08) | ` doma` (-15.09), ` bride` (-15.09), ` cre` (-15.71) |
| L2→L3 | ` entr` → `orem` | content→'orem' | `321` (+16.27), `Emb` (+16.02), `orem` (+3.52) | ` epit` (-0.23), `spe` (-1.35), ` entr` (-2.73) |
| L3→L4 | `orem` → ` epit` | content→'epit' | ` gro` (+16.84), `DE` (+16.84), ` pit` (+16.59) | `orem` (-1.37), `spe` (-16.27), `oretical` (-16.4) |
| L4→L5 | ` epit` → `DE` | content→letter/short | `assist` (+16.04), ` developed` (+15.73), `DD` (+15.73) | ` pit` (-0.8), `leg` (-16.34), ` gro` (-16.84) |
| L5→L6 | `DE` → ` BOTH` | letter/short→'BOTH' | ` BOTH` (+18.75), ` både` (+18.63), `both` (+17.63) | `orem` (-16.91), `321` (-17.16), ` epit` (-17.91) |
| L16→L17 | ` BOTH` → `By` | content→letter/short | `By` (+17.1), `();
` (+16.73), ` and` (+16.54) | ` ]
` (-16.07), ` Both` (-16.94), `both` (-17.82) |
| L17→L18 | `By` → ` ` | letter/short→padding | ` ` (+17.96), ` main` (+17.09), ` `` (+16.77) | `);
` (-16.48), `Field` (-16.54), `_` (-16.54) |
| L18→L19 | ` ` → ` main` | padding→'main' | ` next` (+16.8), ` "` (+16.05), ` overall` (+15.73) | ` `` (-0.54), ``` (-15.96), ` =` (-16.02) |
| L20→L21 | ` main` → ` code` | content→'code' | ` function` (+16.04), ` following` (+15.79), ` user` (+15.73) | ` first` (-0.56), ` main` (-1.25), ` overall` (-15.67) |

**Decision arc:** L0: `ira` → L2: `entr` → L3: `orem` → L4: `epit` → L5: `DE` → L16: `BOTH` → L17: `By` → L18: `` → L20: `main` → L21: `code`

### Position 1: target = ` was`
- Convergence layer: **21**
- Decision transitions: **7**

| Transition | From → To | Semantic | Top gainers | Top losers |
|---|---|---|---|---|
| L0→L1 | ` Dum` → ` Sav` | content→'Sav' | ` sle` (+17.31), ` Nus` (+17.19), `odor` (+16.69) | ` reco` (-16.31), `&p` (-16.43), ` cho` (-16.43) |
| L1→L2 | ` Sav` → `odor` | content→'odor' | ` graz` (+15.2), ` sur` (+15.08), ` Dra` (+14.95) | ` dum` (-1.05), ` Sav` (-1.93), ` incom` (-16.19) |
| L3→L4 | `odor` → `manship` | content→'manship' | ` tipped` (+15.46), ` surf` (+15.03), ` ` (+14.71) | `odor` (-0.54), ` Sta` (-1.1), `olem` (-14.88) |
| L10→L11 | `manship` → `<|endoftext|>` | content→padding | ` sports` (+15.25), `BC` (+15.06), `‑` (+1.37) | ` sur` (0.12), `man` (-0.26), `manship` (-1.2) |
| L14→L15 | `<|endoftext|>` → `,` | padding→punctuation | ` for` (+16.79), ` (` (+16.41), ` in` (+16.35) | `_` (-0.97), `<|endoftext|>` (-1.29), ` or` (-15.64) |
| L18→L19 | `,` → ` is` | punctuation→letter/short | `'s` (+16.15), ` has` (+15.9), ` that` (+2.19) | ` for` (0.87), ` and` (-0.88), `,` (-1.38) |
| L20→L21 | ` is` → ` was` | letter/short→'was' | ` will` (+16.45), ` of` (+0.42), ` was` (+0.36) | `'s` (-0.14), `,` (-0.27), ` has` (-0.58) |

**Decision arc:** L0: `Dum` → L1: `Sav` → L3: `odor` → L10: `manship` → L14: `<|endoftext|>` → L18: `,` → L20: `is` → L21: `was`

### Position 7: target = `,`
- Convergence layer: **23**
- Decision transitions: **6**

| Transition | From → To | Semantic | Top gainers | Top losers |
|---|---|---|---|---|
| L5→L6 | ` Sid` → `bay` | content→'bay' | ` coff` (+15.28), ` viv` (+15.03), `rein` (+14.91) | ` vert` (-14.18), ` heav` (-14.49), ` hil` (-14.56) |
| L6→L7 | `bay` → ` Sid` | content→'Sid' | `<|endoftext|>` (+15.22), `‑` (+15.03), ` phot` (+15.03) | ` commod` (-14.78), `holder` (-14.85), `rein` (-14.91) |
| L7→L8 | ` Sid` → `<|endoftext|>` | content→padding | ` contents` (+15.53), ` coff` (+15.41), `re` (+15.34) | `bay` (-1.12), ` Sid` (-2.06), ` viv` (-14.72) |
| L15→L16 | `<|endoftext|>` → `,` | padding→punctuation | `.` (+16.12), `;` (+15.74), `?` (+15.49) | `‑` (-1.18), `i` (-14.48), `

` (-15.04) |
| L19→L20 | `,` → ` because` | punctuation→'because' | ` if` (+15.49), ` due` (+14.99), `."` (+14.93) | `?` (-14.72), ` (` (-14.78), ` that` (-14.91) |
| L22→L23 | ` because` → `."` | content→'."' | `."

` (+16.12), `?` (+0.72), `.

` (+0.59) | `;` (-0.1), `,` (-0.22), ` if` (-0.35) |

**Decision arc:** L5: `Sid` → L6: `bay` → L7: `Sid` → L15: `<|endoftext|>` → L19: `,` → L22: `because` → L23: `."`

### Position 9: target = ` suitcase`
- Convergence layer: **23**
- Decision transitions: **12**

| Transition | From → To | Semantic | Top gainers | Top losers |
|---|---|---|---|---|
| L1→L2 | ` cant` → ` eng` | content→'eng' | ` entries` (+17.43), ` ex` (+17.3), ` Patrick` (+16.3) | ` cant` (-1.88), ` entr` (-15.68), ` plat` (-16.18) |
| L4→L5 | ` eng` → ` Dra` | content→'Dra' | ` ci` (+17.53), ` fin` (+17.03), ` firm` (+16.78) | ` circ` (-16.2), ` epit` (-16.26), ` verd` (-16.32) |
| L5→L6 | ` Dra` → ` ex` | content→letter/short | ` con` (+16.75), ` pal` (+16.25), ` glimps` (+15.56) | ` alt` (-0.72), ` magn` (-0.97), ` Easter` (-16.28) |
| L6→L7 | ` ex` → ` firm` | letter/short→'firm' | `123` (+16.48), `<|endoftext|>` (+16.29), ` paradox` (+15.91) | ` ci` (-0.65), ` entries` (-0.84), ` alt` (-15.56) |
| L7→L8 | ` firm` → `123` | content→number | ` compound` (+15.8), ` rational` (+15.74), ` presumably` (+15.49) | ` glimps` (-15.85), ` paradox` (-15.91), ` pal` (-16.16) |
| L8→L9 | `123` → ` metaphor` | number→'metaphor' | ` metaphor` (+17.56), ` paradox` (+16.25), ` intangible` (+16.06) | `108` (-15.37), `-f` (-15.43), ` presumably` (-15.49) |
| L9→L10 | ` metaphor` → `<|endoftext|>` | content→padding | ` intended` (+15.65), ` ex` (+15.59), ` pack` (+15.59) | ` rational` (-0.91), ` harmon` (-16.0), ` paradox` (-16.25) |
| L13→L14 | `<|endoftext|>` → `‑` | padding→'‑' | `D` (+16.31), `-` (+15.63), ` volume` (+15.13) | ` constraints` (-15.01), `Hal` (-15.07), ` abstract` (-15.14) |
| L14→L15 | `‑` → ` size` | content→'size' | ` (` (+16.49), ` "` (+16.18), ` shape` (+16.18) | `<|endoftext|>` (-0.32), `‑` (-1.44), `-d` (-15.06) |
| L17→L18 | ` size` → ` "` | content→punctuation | ` first` (+15.11), ` **` (+14.86), ` entire` (+14.11) | ` dimensions` (-0.94), ` size` (-1.31), ` larger` (-14.61) |
| L19→L20 | ` "` → ` size` | punctuation→'size' | ` latter` (+16.52), ` last` (+15.77), ` height` (+1.58) | ` '` (-0.35), ` ` (-0.67), ` "` (-1.42) |
| L22→L23 | ` size` → ` suitcase` | content→'suitcase' | ` space` (+15.14), ` trophy` (+1.6), ` suitcase` (+1.6) | ` box` (-0.78), ` dimensions` (-1.22), ` shape` (-1.28) |

**Decision arc:** L1: `cant` → L4: `eng` → L5: `Dra` → L6: `ex` → L7: `firm` → L8: `123` → L9: `metaphor` → L13: `<|endoftext|>` → L14: `‑` → L17: `size` → L19: `"` → L22: `size` → L23: `suitcase`

## Summary
- Mean convergence layer: 19.7
- Range: L17 – L23
- Total decision transitions across all positions: 116

## CASCADE Insight

Each decision transition (top-1 change) identifies a layer where the model's
prediction shifts. In CASCADE mode, the logit-space difference `Δz = z^(l+1) − z^(l)`
at a decision layer is a **self-supervised steering direction**: it captures the
semantic decision the model makes, without requiring curated contrast pairs.

The gauge-safe projection `v_steer = (CW)⁺ · C·Δz` maps this direction into the
student's embedding space, yielding a closed-form steering vector. This is
fundamentally different from contrastive activation addition (CAA), which requires
100+ positive/negative example pairs per concept.
