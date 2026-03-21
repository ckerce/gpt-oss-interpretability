# Decision Trajectory: induction

**Prompt:** `A7 B2 C9 D4 A7 B2 C9`
**Model:** openai/gpt-oss-20b
**Layers:** 24

## Key Decision Positions (14 positions with 2+ transitions)

### Position 2: target = `8`
- Convergence layer: **13**
- Decision transitions: **14**

| Transition | From → To | Semantic | Top gainers | Top losers |
|---|---|---|---|---|
| L0→L1 | `OL` → `pred` | letter/short→'pred' | ` fore` (+16.8), `osl` (+15.05), `Vert` (+14.92) | `mol` (-0.91), `OL` (-1.16), `sg` (-14.95) |
| L2→L3 | `pred` → ` astr` | content→'astr' | `Vert` (+17.34), ` ro` (+16.84), `ucr` (+16.09) | `pred` (-1.82), `vb` (-16.1), `mol` (-16.16) |
| L3→L4 | ` astr` → `<|endoftext|>` | content→padding | `<|endoftext|>` (+18.41), `WB` (+16.34), ` adjust` (+15.66) | `pred` (-1.5), `osl` (-15.53), `OL` (-16.84) |
| L8→L9 | `<|endoftext|>` → ` rede` | padding→'rede' | `7` (+16.25), `minor` (+16.19), `bon` (+15.69) | `<|endoftext|>` (-1.47), ` cou` (-15.85), `ą` (-15.85) |
| L9→L10 | ` rede` → `6` | content→number | `_i` (+16.66), `12` (+16.29), `11` (+15.98) | ` rede` (-1.9), `bon` (-15.69), `02` (-15.69) |
| L10→L11 | `6` → `<|endoftext|>` | number→padding | `9` (+16.12), `<|reserved_200016|>` (+16.06), `5` (+15.75) | `6` (-0.54), `_i` (-0.66), `MD` (-15.66) |
| L11→L12 | `<|endoftext|>` → `6` | padding→number | `_` (+16.72), `5` (+0.91), `8` (+0.72) | `6` (0.41), `7` (0.28), `_i` (0.28) |
| L12→L13 | `6` → `8` | number→number | `2` (+16.23), `4` (+16.23), `1` (+0.89) | `9` (-0.05), `11` (-0.11), `<|endoftext|>` (-1.17) |
| L14→L15 | `8` → `1` | number→number | ` ` (+16.99), `3` (+16.62), `,` (+16.12) | `8` (0.03), `7` (-0.41), `m` (-16.21) |
| L15→L16 | `1` → ` ` | number→padding | `.` (+16.58), ` ` (+1.33), `,` (+0.96) | `1` (-0.29), `8` (-0.54), `5` (-0.67) |
| L16→L17 | ` ` → `4` | padding→number | `0` (+15.98), `9` (+15.98), `4` (+1.28) | `5` (-0.1), `1` (-1.1), ` ` (-1.1) |
| L17→L18 | `4` → `2` | number→number | `1` (+1.08), ` ` (+1.08), `0` (+0.83) | `6` (-0.42), `9` (-0.42), `5` (-0.54) |
| L19→L20 | `2` → `4` | number→number | `7` (+0.8), `8` (+0.68), `5` (+0.55) | `0` (0.05), `6` (-0.2), `1` (-0.32) |
| L20→L21 | `4` → `8` | number→number | `8` (+1.68), `7` (+1.05), `9` (+0.43) | `5` (-1.07), `3` (-1.45), `1` (-1.7) |

**Decision arc:** L0: `OL` → L2: `pred` → L3: `astr` → L8: `<|endoftext|>` → L9: `rede` → L10: `6` → L11: `<|endoftext|>` → L12: `6` → L14: `8` → L15: `1` → L16: `` → L17: `4` → L19: `2` → L20: `4` → L21: `8`

### Position 1: target = `:`
- Convergence layer: **17**
- Decision transitions: **8**

| Transition | From → To | Semantic | Top gainers | Top losers |
|---|---|---|---|---|
| L0→L1 | `corn` → `BB` | content→letter/short | `BB` (+19.02), `CQ` (+16.77), ` �` (+16.77) | ` lad` (-17.04), ` grooves` (-17.42), ` Mari` (-17.54) |
| L5→L6 | `BB` → `JB` | letter/short→letter/short | ` reb` (+16.07), `Q` (+15.7), `JB` (+1.77) | `823` (-0.48), `mol` (-0.73), `BB` (-1.23) |
| L6→L7 | `JB` → `BB` | letter/short→letter/short | ` mir` (+16.71), ` rede` (+16.4), ` synthes` (+15.96) | `MD` (0.08), `JB` (-1.49), `Q` (-15.7) |
| L7→L8 | `BB` → `th` | letter/short→letter/short | `<|endoftext|>` (+15.98), ` generate` (+15.92), `th` (+1.45) | ` rede` (-0.05), `823` (-0.17), `BB` (-2.05) |
| L11→L12 | `th` → `_` | letter/short→'_' | `B` (+16.49), `_` (+1.21), `v` (+0.46) | `7` (-0.36), `MD` (-0.61), `<|endoftext|>` (-0.61) |
| L13→L14 | `_` → `A` | content→letter/short | `.` (+16.89), `-` (+16.64), `V` (+15.89) | `8` (-1.29), `7` (-1.48), `6` (-16.37) |
| L15→L16 | `A` → ` (` | letter/short→punctuation | ` (` (+18.24), `:` (+17.99), ` ` (+16.8) | `A` (-1.91), `V` (-15.96), `D` (-16.03) |
| L16→L17 | ` (` → `:` | punctuation→punctuation | ` =` (+16.63), ` ` (+1.33), `-` (+0.64) | `A` (-0.05), `.` (-0.86), `,` (-0.98) |

**Decision arc:** L0: `corn` → L5: `BB` → L6: `JB` → L7: `BB` → L11: `th` → L13: `_` → L15: `A` → L16: `(` → L17: `:`

### Position 3: target = ` `
- Convergence layer: **17**
- Decision transitions: **7**

| Transition | From → To | Semantic | Top gainers | Top losers |
|---|---|---|---|---|
| L0→L1 | `️⃣` → `mol` | content→'mol' | `CCC` (+16.06), `pla` (+15.68), ` �` (+15.56) | `bull` (-15.72), `assem` (-15.97), ` ment` (-16.34) |
| L1→L2 | `mol` → ` mol` | content→'mol' | ` Zi` (+15.69), `gate` (+15.5), ` Ruff` (+14.75) | `Ws` (-15.31), ` �` (-15.43), ` �` (-15.56) |
| L5→L6 | ` mol` → `<|endoftext|>` | content→padding | `MD` (+15.98), ` har` (+14.86), `<|endoftext|>` (+1.78) | ` mas` (-1.41), `02` (-1.53), ` mol` (-2.53) |
| L13→L14 | `<|endoftext|>` → ` and` | padding→'and' | ` and` (+18.22), `-` (+17.03), `_` (+16.28) | `_i` (-15.8), `<|reserved_200016|>` (-16.05), `1` (-16.17) |
| L14→L15 | ` and` → `-` | content→punctuation | `,` (+17.97), `1` (+16.41), ` ` (+16.28) | `O` (-15.78), `R` (-15.97), `ad` (-16.78) |
| L15→L16 | `-` → ` and` | punctuation→'and' | ` -` (+16.73), `:` (+16.35), ` &` (+16.1) | `

` (-15.66), `V` (-15.91), `8` (-16.1) |
| L16→L17 | ` and` → ` ` | content→padding | ` +` (+15.58), `/` (+15.21), ` ` (+1.36) | `,` (-0.14), `-` (-0.64), ` and` (-0.64) |

**Decision arc:** L0: `️⃣` → L1: `mol` → L5: `mol` → L13: `<|endoftext|>` → L14: `and` → L15: `-` → L16: `and` → L17: ``

### Position 5: target = ` `
- Convergence layer: **17**
- Decision transitions: **9**

| Transition | From → To | Semantic | Top gainers | Top losers |
|---|---|---|---|---|
| L0→L1 | ` Mons` → `�` | content→'�' | `mol` (+16.19), ` tou` (+14.57), `ksi` (+13.69) | ` Bang` (-15.39), `inja` (-15.76), ` Jud` (-16.39) |
| L2→L3 | `�` → `mol` | content→'mol' | `/th` (+12.44), ` Kle` (+12.44), `mol` (+0.75) | `\,` (-1.0), ` Mons` (-1.25), `�` (-2.75) |
| L9→L10 | `mol` → `<|endoftext|>` | content→padding | ` com` (+15.82), ` VB` (+15.75), `7` (+15.63) | ` rec` (-16.49), ` Luna` (-16.68), ` mas` (-16.8) |
| L13→L14 | `<|endoftext|>` → `8` | padding→number | ` and` (+17.18), `B` (+15.93), `O` (+15.87) | `33` (-15.5), `<|reserved_200016|>` (-15.56), ` O` (-16.18) |
| L14→L15 | `8` → `-` | number→punctuation | `-` (+17.88), `1` (+17.2), ` (` (+17.01) | ` -` (-15.68), `O` (-15.87), `B` (-15.93) |
| L15→L16 | `-` → ` and` | punctuation→'and' | ` ` (+17.62), ` -` (+17.37), `?` (+17.25) | `8` (-0.88), `M` (-16.2), `_` (-16.45) |
| L16→L17 | ` and` → ` ` | content→padding | `.` (+16.56), `+` (+15.69), ` ` (+1.31) | ` (` (-0.06), `,` (-0.19), ` and` (-2.19) |
| L20→L21 | ` ` → ` D` | padding→letter/short | ` E` (+14.83), ` F` (+13.33), ` A` (+13.2) | `?` (-14.12), ` etc` (-14.37), `:` (-14.81) |
| L22→L23 | ` D` → ` ` | letter/short→padding | `

` (+16.23), `"` (+15.92), ` B` (+5.07) | ` C` (3.0), ` E` (2.13), ` D` (-1.75) |

**Decision arc:** L0: `Mons` → L2: `�` → L9: `mol` → L13: `<|endoftext|>` → L14: `8` → L15: `-` → L16: `and` → L20: `` → L22: `D` → L23: ``

### Position 0: target = `,`
- Convergence layer: **18**
- Decision transitions: **10**

| Transition | From → To | Semantic | Top gainers | Top losers |
|---|---|---|---|---|
| L2→L3 | ` lest` → ` vert` | content→'vert' | `xi` (+16.7), ` hydro` (+15.51), `Vert` (+15.32) | `tempt` (-14.91), `xp` (-15.41), `XP` (-16.28) |
| L5→L6 | ` vert` → ` BOTH` | content→'BOTH' | ` BOTH` (+18.75), ` både` (+18.63), `both` (+17.63) | `jour` (-16.44), `XP` (-16.69), `ID` (-16.69) |
| L13→L14 | ` BOTH` → ` både` | content→'både' | `	AND` (+15.71), `both` (+0.17), `	component` (+0.04) | ` ]
` (-0.08), ` BOTH` (-0.08), ` ],
` (-0.08) |
| L14→L15 | ` både` → ` BOTH` | content→'BOTH' | `"struct` (+15.67), ` zarówno` (+0.08), ` Both` (+0.08) | ` ]
` (-0.04), `both` (-0.04), ` ],
` (-0.17) |
| L16→L17 | ` BOTH` → `_` | content→'_' | `_` (+18.4), ` =` (+17.9), ` and` (+17.33) | ` ]
` (-16.17), ` Both` (-16.92), `both` (-17.92) |
| L17→L18 | `_` → `,` | content→punctuation | `.` (+17.21), `2` (+17.02), `
` (+16.46) | `_` (-1.19), `}` (-16.15), `,
` (-16.27) |
| L18→L19 | `,` → `2` | punctuation→number | `4` (+16.52), `2` (+1.43), `)` (+1.3) | `_` (-0.45), `,` (-0.57), ` =` (-0.76) |
| L19→L20 | `2` → ` =` | number→'=' | `\` (+16.42), ` =` (+0.47), `_` (+0.22) | `)` (-0.16), `2` (-0.53), `3` (-0.72) |
| L21→L22 | ` =` → `)` | content→punctuation | `$` (+15.65), `'` (+15.58), `,` (+-0.48) | ` =` (-1.3), ` +` (-1.3), `.` (-1.48) |
| L22→L23 | `)` → `,` | punctuation→punctuation | `}` (+16.68), `-Z` (+15.93), `\` (+0.66) | ` =` (0.29), `1` (0.1), `)` (0.04) |

**Decision arc:** L2: `lest` → L5: `vert` → L13: `BOTH` → L14: `både` → L16: `BOTH` → L17: `_` → L18: `,` → L19: `2` → L21: `=` → L22: `)` → L23: `,`

### Position 10: target = `2`
- Convergence layer: **19**
- Decision transitions: **9**

| Transition | From → To | Semantic | Top gainers | Top losers |
|---|---|---|---|---|
| L0→L1 | `mol` → ` bull` | content→'bull' | `gate` (+16.92), `uh` (+15.67), `Gate` (+15.05) | `jp` (-2.26), `ment` (-15.3), `OL` (-15.43) |
| L1→L2 | ` bull` → `gate` | content→'gate' | `391` (+15.23), `-n` (+15.23), ` gate` (+14.48) | ` chore` (-14.8), `vb` (-14.92), `rol` (-15.05) |
| L3→L4 | `gate` → `Vert` | content→'Vert' | ` bra` (+16.72), `???` (+16.6), `orra` (+16.41) | `Fn` (-15.53), ` bull` (-15.53), `xi` (-15.53) |
| L4→L5 | `Vert` → `???` | content→'???' | ` fore` (+17.74), ` equals` (+16.18), `fore` (+16.06) | `-n` (-0.79), `mol` (-16.16), `gate` (-16.6) |
| L6→L7 | `???` → `orra` | content→'orra' | ` grand` (+16.15), ` rede` (+15.77), `739` (+15.71) | `???` (-0.76), ` presumably` (-15.47), ` shows` (-15.59) |
| L9→L10 | `orra` → `<|endoftext|>` | content→padding | ` rec` (+16.49), ` recom` (+16.24), `662` (+15.8) | `orra` (-1.61), ` rese` (-15.79), `453` (-16.1) |
| L11→L12 | `<|endoftext|>` → `8` | padding→number | `7` (+16.08), ` again` (+15.83), `_S` (+15.77) | ` rede` (-15.55), ` recom` (-15.74), ` rec` (-15.8) |
| L14→L15 | `8` → `1` | number→number | `i` (+16.0), `3` (+15.82), `1` (+2.59) | `7` (-0.29), `8` (-0.41), `5` (-0.54) |
| L18→L19 | `1` → `2` | number→number | `8` (+14.56), `2` (+1.0), `3` (+-0.12) | `5` (-1.0), `1` (-1.5), `0` (-2.25) |

**Decision arc:** L0: `mol` → L1: `bull` → L3: `gate` → L4: `Vert` → L6: `???` → L9: `orra` → L11: `<|endoftext|>` → L14: `8` → L18: `1` → L19: `2`

### Position 12: target = `9`
- Convergence layer: **19**
- Decision transitions: **13**

| Transition | From → To | Semantic | Top gainers | Top losers |
|---|---|---|---|---|
| L0→L1 | ` ba` → `mol` | letter/short→'mol' | ` hel` (+17.07), `stip` (+16.32), `Balancer` (+15.95) | `Gy` (-14.98), `Bang` (-15.11), ` Bang` (-15.86) |
| L2→L3 | `mol` → `stip` | content→'stip' | ` bra` (+17.32), ` entr` (+17.19), ` neut` (+16.26) | ` �` (-16.32), ` Fitch` (-16.32), `Gate` (-16.45) |
| L3→L4 | `stip` → ` bra` | content→'bra' | ` epit` (+17.45), ` grade` (+16.45), ` Cass` (+16.2) | ` ba` (-16.19), `rd` (-16.19), `vert` (-16.19) |
| L4→L5 | ` bra` → ` neut` | content→'neut' | `???` (+16.55), ` Bennett` (+16.36), ` unders` (+16.18) | `�` (-15.51), `stip` (-15.51), ` share` (-15.76) |
| L6→L7 | ` neut` → `480` | content→number | `454` (+16.59), `123` (+16.53), `442` (+16.46) | ` neut` (-1.26), ` cut` (-15.97), ` share` (-16.1) |
| L7→L8 | `480` → `454` | number→number | `8` (+16.26), ` sn` (+16.2), `454` (+1.17) | ` rec` (-0.58), ` unders` (-0.7), `480` (-1.26) |
| L8→L9 | `454` → ` rec` | number→'rec' | `7` (+16.64), `456` (+16.14), ` rede` (+15.83) | `orra` (-0.56), `454` (-2.06), ` unders` (-15.7) |
| L10→L11 | ` rec` → `<|endoftext|>` | content→padding | `VI` (+14.87), ` followed` (+14.87), `<|endoftext|>` (+2.74) | `456` (-0.32), `5` (-0.38), `7` (-0.38) |
| L11→L12 | `<|endoftext|>` → `8` | padding→number | `v` (+15.73), `vi` (+15.41), `9` (+15.41) | `VI` (-14.87), ` followed` (-14.87), `5` (-14.93) |
| L15→L16 | `8` → ` and` | number→'and' | `-` (+17.16), `,` (+16.79), ` ` (+16.66) | `7` (-1.48), `<|reserved_200016|>` (-15.83), `4` (-16.01) |
| L16→L17 | ` and` → `1` | content→number | `9` (+17.07), `5` (+16.69), ` ` (+1.03) | `8` (0.28), `2` (0.16), ` and` (-2.47) |
| L17→L18 | `1` → ` ` | number→padding | `6` (+16.69), `4` (+16.44), `2` (+1.25) | `1` (-0.25), `3` (-0.38), `8` (-1.25) |
| L18→L19 | ` ` → `9` | padding→number | `9` (+19.3), `0` (+13.92), `8` (+1.23) | `3` (-1.02), `4` (-1.77), `2` (-1.9) |

**Decision arc:** L0: `ba` → L2: `mol` → L3: `stip` → L4: `bra` → L6: `neut` → L7: `480` → L8: `454` → L10: `rec` → L11: `<|endoftext|>` → L15: `8` → L16: `and` → L17: `1` → L18: `` → L19: `9`

### Position 4: target = `3`
- Convergence layer: **20**
- Decision transitions: **11**

| Transition | From → To | Semantic | Top gainers | Top losers |
|---|---|---|---|---|
| L1→L2 | ` astr` → ` dont` | content→'dont' | ` Kath` (+17.16), `wn` (+16.66), `Gate` (+15.6) | ` astr` (-0.82), `PEG` (-15.86), `vert` (-15.98) |
| L3→L4 | ` dont` → `mine` | content→'mine' | ` ro` (+17.87), ` grade` (+16.12), ` neut` (+16.06) | ` Bennett` (-15.94), `Vert` (-16.0), `aceous` (-16.12) |
| L4→L5 | `mine` → ` ro` | content→letter/short | ` Bennett` (+16.6), ` fer` (+16.16), ` spl` (+16.1) | ` ц` (-15.62), `MD` (-15.62), ` Cass` (-15.81) |
| L5→L6 | ` ro` → `<|endoftext|>` | letter/short→padding | ` rede` (+17.19), `MD` (+15.87), ` arbitr` (+15.87) | ` Bennett` (-0.85), `JB` (-15.97), ` spl` (-16.1) |
| L6→L7 | `<|endoftext|>` → ` rede` | padding→'rede' | `8` (+16.13), ` ro` (+15.95), `Md` (+15.82) | ` neut` (-1.37), ` modify` (-15.56), ` Bennett` (-15.75) |
| L7→L8 | ` rede` → `MD` | content→letter/short | `138` (+16.01), `1` (+15.95), `md` (+15.89) | `331` (-15.7), ` arbitr` (-15.82), ` ro` (-15.95) |
| L8→L9 | `MD` → ` rede` | letter/short→'rede' | `22` (+16.65), ` wise` (+16.65), ` ro` (+16.09) | `Md` (-15.45), `6` (-15.7), `md` (-15.89) |
| L9→L10 | ` rede` → `<|endoftext|>` | content→padding | ` adjust` (+15.91), `6` (+15.85), `<|endoftext|>` (+1.51) | `MD` (-0.81), `22` (-1.18), ` rede` (-1.68) |
| L12→L13 | `<|endoftext|>` → `8` | padding→number | `2` (+15.2), `1` (+1.28), `8` (+1.16) | `83` (0.22), `6` (0.16), `<|endoftext|>` (-0.53) |
| L17→L18 | `8` → `2` | number→number | `0` (+1.67), `1` (+1.67), `2` (+1.67) | `7` (-0.08), `4` (-0.21), `6` (-0.46) |
| L19→L20 | `2` → `3` | number→number | `4` (+0.79), `7` (+0.67), `3` (+0.54) | `9` (-0.21), `2` (-0.46), `6` (-0.71) |

**Decision arc:** L1: `astr` → L3: `dont` → L4: `mine` → L5: `ro` → L6: `<|endoftext|>` → L7: `rede` → L8: `MD` → L9: `rede` → L12: `<|endoftext|>` → L17: `8` → L19: `2` → L20: `3`

### Position 6: target = `4`
- Convergence layer: **20**
- Decision transitions: **14**

| Transition | From → To | Semantic | Top gainers | Top losers |
|---|---|---|---|---|
| L0→L1 | ` Hass` → `SS` | content→letter/short | `�` (+16.72), `gate` (+16.59), ` �` (+15.84) | ` Bang` (-15.93), `mol` (-16.68), `mus` (-16.8) |
| L1→L2 | `SS` → `gate` | letter/short→'gate' | `icl` (+16.87), ` nat` (+16.74), `stip` (+15.87) | `�` (-15.28), `�` (-15.78), ` �` (-15.84) |
| L4→L5 | `gate` → ` neut` | content→'neut' | `ws` (+16.82), `WS` (+16.38), `J` (+1.44) | `his` (0.13), ` Cass` (-0.81), `gate` (-2.25) |
| L6→L7 | ` neut` → `ws` | content→letter/short | `k` (+16.33), ` ro` (+15.83), ` cass` (+15.7) | `ra` (-0.5), ` neut` (-1.19), `_cs` (-15.71) |
| L7→L8 | `ws` → ` neut` | letter/short→'neut' | `lar` (+16.6), `442` (+16.29), `123` (+15.92) | `ra` (-1.04), ` cass` (-15.7), ` ro` (-15.83) |
| L9→L10 | ` neut` → ` ro` | content→letter/short | `<|endoftext|>` (+16.32), `MD` (+15.95), `_m` (+15.76) | `150` (-15.65), ` dif` (-15.9), `296` (-15.9) |
| L10→L11 | ` ro` → `<|endoftext|>` | letter/short→padding | `<|reserved_200016|>` (+16.25), `VI` (+15.63), `150` (+15.06) | `823` (-15.32), `662` (-15.39), `_s` (-15.57) |
| L12→L13 | `<|endoftext|>` → `8` | padding→number | `7` (+16.92), `1` (+16.42), `5` (+16.23) | `<|reserved_200016|>` (-0.58), `<|endoftext|>` (-0.71), `‑` (-15.0) |
| L17→L18 | `8` → `1` | number→number | `0` (+16.68), ` ` (+1.61), `1` (+0.48) | `5` (-0.27), `7` (-0.39), `6` (-0.77) |
| L18→L19 | `1` → `6` | number→number | `9` (+16.62), `6` (+1.07), `8` (+1.07) | `1` (0.07), `4` (-0.31), `2` (-0.56) |
| L19→L20 | `6` → `4` | number→number | `4` (+1.59), `5` (+0.47), `3` (+0.22) | `8` (-0.28), `6` (-0.41), `0` (-0.53) |
| L20→L21 | `4` → `8` | number→number | `8` (+1.56), `6` (+0.19), `0` (+-0.19) | `4` (-1.31), `1` (-1.44), `9` (-1.44) |
| L21→L22 | `8` → `6` | number→number | `2` (+2.17), `5` (+1.67), `0` (+1.29) | `9` (1.04), `4` (0.54), `7` (0.42) |
| L22→L23 | `6` → `4` | number→number | `7` (+0.64), `9` (+0.64), `2` (+0.52) | `0` (0.33), `1` (0.27), `4` (-0.11) |

**Decision arc:** L0: `Hass` → L1: `SS` → L4: `gate` → L6: `neut` → L7: `ws` → L9: `neut` → L10: `ro` → L12: `<|endoftext|>` → L17: `8` → L18: `1` → L19: `6` → L20: `4` → L21: `8` → L22: `6` → L23: `4`

### Position 7: target = ` E`
- Convergence layer: **21**
- Decision transitions: **10**

| Transition | From → To | Semantic | Top gainers | Top losers |
|---|---|---|---|---|
| L0→L1 | ` ment` → `�` | content→'�' | `bev` (+16.63), `BE` (+16.5), ` Greg` (+3.84) | `want` (-0.78), `ment` (-1.28), ` ment` (-2.78) |
| L1→L2 | `�` → ` Mach` | content→'Mach' | `mol` (+17.59), ` mas` (+17.34), `CB` (+15.71) | ` ment` (-16.5), `BE` (-16.5), `want` (-16.63) |
| L2→L3 | ` Mach` → `mol` | content→'mol' | ` Mons` (+15.55), ` tons` (+15.42), `CB` (+1.71) | `�` (-0.54), ` Mach` (-0.92), ` Greg` (-1.04) |
| L5→L6 | `mol` → `<|endoftext|>` | content→padding | ` lap` (+15.85), ` mol` (+15.66), `aff` (+15.54) | `CB` (-2.04), `�` (-15.39), ` tons` (-15.83) |
| L6→L7 | `<|endoftext|>` → `mol` | padding→'mol' | ` ro` (+16.08), ` priv` (+16.08), ` Christ` (+16.02) | `<|endoftext|>` (-0.58), `BB` (-15.47), `aff` (-15.54) |
| L7→L8 | `mol` → `<|endoftext|>` | content→padding | `opers` (+15.69), `BB` (+15.44), ` entry` (+15.31) | ` mas` (-1.08), `mol` (-2.46), `odor` (-15.71) |
| L14→L15 | `<|endoftext|>` → ` and` | padding→'and' | `,` (+16.35), `k` (+16.22), `9` (+16.16) | `_h` (-15.45), `re` (-15.58), `',
` (-15.89) |
| L16→L17 | ` and` → `?` | content→punctuation | `
` (+16.33), ` ` (+2.5), `.` (+2.25) | `(` (0.69), `-` (0.25), `,` (-0.06) |
| L17→L18 | `?` → ` ` | punctuation→padding | ` -` (+15.39), `;` (+14.02), ` ` (+1.07) | `:` (-1.18), `?` (-2.18), ` and` (-4.06) |
| L20→L21 | ` ` → ` E` | padding→letter/short | ` A` (+13.18), `E` (+13.18), ` e` (+12.68) | `
` (-4.39), `,` (-15.2), `.` (-15.38) |

**Decision arc:** L0: `ment` → L1: `�` → L2: `Mach` → L5: `mol` → L6: `<|endoftext|>` → L7: `mol` → L14: `<|endoftext|>` → L16: `and` → L17: `?` → L20: `` → L21: `E`

### Position 8: target = `5`
- Convergence layer: **21**
- Decision transitions: **10**

| Transition | From → To | Semantic | Top gainers | Top losers |
|---|---|---|---|---|
| L0→L1 | ` lest` → ` dig` | content→'dig' | ` bull` (+17.82), ` Abr` (+16.07), ` ic` (+15.94) | `quier` (-15.45), `YP` (-15.7), `UC` (-15.7) |
| L5→L6 | ` dig` → ` Cass` | content→'Cass' | `cs` (+16.54), `orra` (+16.47), `<|endoftext|>` (+16.41) | ` dig` (-2.9), `xi` (-15.44), ` viv` (-15.5) |
| L6→L7 | ` Cass` → `orra` | content→'orra' | `LL` (+16.97), `8` (+16.79), ` letters` (+15.85) | `cker` (-15.54), ` dig` (-15.6), ` hol` (-15.66) |
| L7→L8 | `orra` → ` letters` | content→'letters' | ` hol` (+16.57), `md` (+16.19), ` crisp` (+15.94) | `J` (-0.78), ` re` (-15.41), `42` (-15.41) |
| L8→L9 | ` letters` → `md` | content→letter/short | `‑` (+16.26), `LL` (+16.26), ` dig` (+15.89) | ` letters` (-1.43), `J` (-15.88), ` neut` (-15.88) |
| L9→L10 | `md` → `orra` | letter/short→'orra' | `7` (+16.08), ` cer` (+15.65), ` cass` (+15.52) | ` crisp` (-0.93), `123` (-15.76), ` dig` (-15.89) |
| L10→L11 | `orra` → `8` | content→number | `-d` (+16.68), `<|endoftext|>` (+16.31), `9` (+16.12) | ` massive` (-15.46), ` crisp` (-15.46), ` cass` (-15.52) |
| L15→L16 | `8` → `1` | number→number | `-` (+16.66), `6` (+15.91), `1` (+1.19) | `5` (-0.37), `8` (-0.62), `7` (-1.19) |
| L18→L19 | `1` → `7` | number→number | `6` (+18.52), `9` (+17.14), `7` (+2.91) | `1` (-1.21), `0` (-1.21), `2` (-1.21) |
| L20→L21 | `7` → `5` | number→number | `5` (+0.97), `8` (+0.97), `0` (+0.72) | `3` (-0.53), `2` (-0.66), `4` (-1.16) |

**Decision arc:** L0: `lest` → L5: `dig` → L6: `Cass` → L7: `orra` → L8: `letters` → L9: `md` → L10: `orra` → L15: `8` → L18: `1` → L20: `7` → L21: `5`

### Position 9: target = ` B`
- Convergence layer: **21**
- Decision transitions: **12**

| Transition | From → To | Semantic | Top gainers | Top losers |
|---|---|---|---|---|
| L0→L1 | ` Bert` → `Specifier` | content→'Specifier' | ` lad` (+18.08), `mol` (+16.45), `ment` (+16.08) | `ee` (-15.59), ` Mari` (-16.22), `izon` (-16.59) |
| L2→L3 | `Specifier` → ` lev` | content→'lev' | ` lev` (+19.09), ` Teng` (+17.09), `折` (+16.46) | ` carbide` (-15.81), `Dor` (-15.94), `BB` (-16.81) |
| L3→L4 | ` lev` → ` unfolding` | content→'unfolding' | ` capturing` (+16.06), `nu` (+16.06), ` unfold` (+15.63) | `登` (-15.96), `odor` (-16.09), ` Teng` (-17.09) |
| L4→L5 | ` unfolding` → ` lev` | content→'lev' | ` neut` (+16.29), ` ul` (+16.17), ` tail` (+15.86) | ` capturing` (-0.33), ` unfolding` (-1.08), ` unfold` (-15.63) |
| L5→L6 | ` lev` → `nu` | content→letter/short | ` dot` (+16.7), ` cup` (+16.51), `nb` (+16.07) | ` capturing` (-15.73), ` ul` (-16.17), ` equals` (-16.23) |
| L9→L10 | `nu` → ` wing` | letter/short→'wing' | ` digits` (+16.19), ` reb` (+15.75), ` OK` (+15.63) | ` ` (-15.46), ` capturing` (-15.53), ` port` (-15.65) |
| L10→L11 | ` wing` → ` repeating` | content→'repeating' | ` repeating` (+17.38), `<|endoftext|>` (+16.88), ` repeated` (+15.69) | ` loops` (-15.63), ` reb` (-15.75), ` generate` (-15.75) |
| L11→L12 | ` repeating` → ` bound` | content→'bound' | ` bound` (+16.25), ` assigned` (+15.63), ` policy` (+15.32) | `MD` (-15.63), ` dig` (-15.63), `-d` (-15.63) |
| L12→L13 | ` bound` → ` again` | content→'again' | `-d` (+15.41), `??` (+15.35), ` dec` (+15.29) | ` assigned` (-0.28), ` digits` (-0.66), ` numbers` (-15.0) |
| L16→L17 | ` again` → ` (` | content→punctuation | `2` (+15.69), `+` (+15.5), ` (` (+1.3) | `?` (-0.39), ` -` (-0.45), ` and` (-1.33) |
| L17→L18 | ` (` → ` ` | punctuation→padding | `
` (+15.89), `

` (+14.89), `,` (+1.45) | ` -` (-0.8), `(` (-1.92), `-` (-2.55) |
| L20→L21 | ` ` → ` B` | padding→letter/short | ` E` (+16.4), ` D` (+14.65), ` F` (+14.4) | `:` (-14.79), ` -` (-15.1), ` ...` (-15.1) |

**Decision arc:** L0: `Bert` → L2: `Specifier` → L3: `lev` → L4: `unfolding` → L5: `lev` → L9: `nu` → L10: `wing` → L11: `repeating` → L12: `bound` → L16: `again` → L17: `(` → L20: `` → L21: `B`

### Position 11: target = ` C`
- Convergence layer: **21**
- Decision transitions: **7**

| Transition | From → To | Semantic | Top gainers | Top losers |
|---|---|---|---|---|
| L1→L2 | `️⃣` → ` Ruff` | content→'Ruff' | ` Ruff` (+18.67), ` mol` (+17.17), `gate` (+16.17) | ` �` (-15.89), ` Ost` (-15.89), `mentor` (-16.01) |
| L2→L3 | ` Ruff` → `mol` | content→'mol' | ` tra` (+15.0), ` Mons` (+14.57), ` Berd` (+14.07) | `ment` (-15.55), ` trunk` (-16.11), `pla` (-16.11) |
| L4→L5 | `mol` → ` equals` | content→'equals' | `<|endoftext|>` (+17.23), `nu` (+15.66), ` equals` (+2.41) | ` shows` (0.35), ` mol` (-1.03), `mol` (-1.65) |
| L5→L6 | ` equals` → `<|endoftext|>` | content→padding | `\.` (+15.33), ` insert` (+15.14), ` blast` (+15.14) | `mol` (-2.46), `nu` (-15.66), ` tra` (-16.04) |
| L14→L15 | `<|endoftext|>` → ` and` | padding→'and' | `y` (+16.9), `z` (+15.34), `E` (+15.34) | `??` (-0.21), `<|endoftext|>` (-1.02), `重` (-15.8) |
| L16→L17 | ` and` → ` ` | content→padding | ` -` (+15.83), ` +` (+15.58), `.` (+15.39) | `?` (0.3), ` and` (-1.32), `C` (-14.59) |
| L20→L21 | ` ` → ` C` | padding→letter/short | `C` (+12.37), ` E` (+12.0), ` D` (+11.75) | ` etc` (-12.83), `?` (-13.08), `

` (-13.58) |

**Decision arc:** L1: `️⃣` → L2: `Ruff` → L4: `mol` → L5: `equals` → L14: `<|endoftext|>` → L16: `and` → L20: `` → L21: `C`

### Position 13: target = ` D`
- Convergence layer: **21**
- Decision transitions: **9**

| Transition | From → To | Semantic | Top gainers | Top losers |
|---|---|---|---|---|
| L0→L1 | ` Mons` → `�` | content→'�' | `mol` (+16.75), `�` (+15.87), `lea` (+15.75) | `�` (-15.86), `lau` (-16.11), ` Bang` (-16.24) |
| L2→L3 | `�` → `mol` | content→'mol' | ` Levi` (+13.99), ` lodging` (+12.49), ` Roi` (+12.11) | `�` (-5.06), ` Bang` (-14.54), ` Hass` (-14.79) |
| L7→L8 | `mol` → ` inc` | content→'inc' | ` rec` (+16.01), ` plus` (+15.58), `123` (+15.45) | `mol` (-1.35), `_CN` (-15.74), ` gran` (-16.36) |
| L8→L9 | ` inc` → ` rec` | content→'rec' | ` digits` (+16.56), `重` (+16.19), ` again` (+16.0) | `123` (-15.45), ` Num` (-15.45), ` alpha` (-15.45) |
| L9→L10 | ` rec` → `<|endoftext|>` | content→padding | ` alt` (+15.82), ` repeating` (+15.14), `<|endoftext|>` (+1.7) | ` capturing` (0.26), ` rec` (-0.62), ` inc` (-0.99) |
| L13→L14 | `<|endoftext|>` → ` again` | padding→'again' | ` and` (+17.04), ` inc` (+16.66), ` positions` (+16.04) | `<|endoftext|>` (-1.55), ` repeat` (-16.22), ` repeated` (-16.34) |
| L14→L15 | ` again` → ` and` | content→'and' | `?` (+16.85), `M` (+16.16), `...` (+15.97) | ` positions` (-16.04), ` repeating` (-16.1), ` mir` (-16.6) |
| L16→L17 | ` and` → ` ` | content→padding | ` (` (+17.21), `1` (+16.02), `.` (+15.84) | `,` (0.24), ` and` (-1.51), `(` (-15.1) |
| L20→L21 | ` ` → ` D` | padding→letter/short | ` d` (+13.62), ` E` (+12.49), `D` (+12.24) | ` (` (-13.05), ` etc` (-13.42), ` -` (-13.55) |

**Decision arc:** L0: `Mons` → L2: `�` → L7: `mol` → L8: `inc` → L9: `rec` → L13: `<|endoftext|>` → L14: `again` → L16: `and` → L20: `` → L21: `D`

## Summary
- Mean convergence layer: 18.9
- Range: L13 – L21
- Total decision transitions across all positions: 143

## CASCADE Insight

Each decision transition (top-1 change) identifies a layer where the model's
prediction shifts. In CASCADE mode, the logit-space difference `Δz = z^(l+1) − z^(l)`
at a decision layer is a **self-supervised steering direction**: it captures the
semantic decision the model makes, without requiring curated contrast pairs.

The gauge-safe projection `v_steer = (CW)⁺ · C·Δz` maps this direction into the
student's embedding space, yielding a closed-form steering vector. This is
fundamentally different from contrastive activation addition (CAA), which requires
100+ positive/negative example pairs per concept.
