# Decision Trajectory: syntax

**Prompt:** `The keys to the cabinet are on the table, so they`
**Model:** openai/gpt-oss-20b
**Layers:** 24

## Key Decision Positions (12 positions with 2+ transitions)

### Position 9: target = ` and`
- Convergence layer: **16**
- Decision transitions: **6**

| Transition | From → To | Semantic | Top gainers | Top losers |
|---|---|---|---|---|
| L1→L2 | ` Ba` → `bet` | letter/short→'bet' | ` interfer` (+16.61), ` ent` (+16.48), ` bet` (+1.93) | `�` (-0.07), `Tra` (-0.45), ` Ba` (-2.95) |
| L2→L3 | `bet` → ` ple` | content→'ple' | ` trained` (+16.47), ` recon` (+15.6), ` master` (+15.6) | `�` (-1.19), ` fins` (-15.98), `Tra` (-16.23) |
| L3→L4 | ` ple` → `bet` | content→'bet' | ` ro` (+16.81), ` bet` (+16.5), ` abstract` (+15.5) | ` trained` (-0.85), ` ple` (-1.6), ` con` (-15.54) |
| L10→L11 | `bet` → ` harmon` | content→'harmon' | ` correspond` (+14.65), `‑` (+14.52), ` harmon` (+2.32) | ` hidden` (-1.18), `bet` (-1.93), ` door` (-1.93) |
| L12→L13 | ` harmon` → `<|endoftext|>` | content→padding | ` sentence` (+15.06), `."` (+14.87), `—` (+14.75) | ` bet` (-0.55), ` harmon` (-1.17), `bet` (-14.73) |
| L15→L16 | `<|endoftext|>` → ` and` | padding→'and' | ` ` (+15.71), ` 
` (+15.27), ` which` (+15.21) | ` harmon` (-14.71), ` keys` (-14.77), `"` (-15.4) |

**Decision arc:** L1: `Ba` → L2: `bet` → L3: `ple` → L10: `bet` → L12: `harmon` → L15: `<|endoftext|>` → L16: `and`

### Position 2: target = ` the`
- Convergence layer: **17**
- Decision transitions: **11**

| Transition | From → To | Semantic | Top gainers | Top losers |
|---|---|---|---|---|
| L1→L2 | ` Ba` → ` ble` | letter/short→'ble' | ` ble` (+18.35), `onka` (+17.48), ` Bar` (+16.98) | ` CIN` (-15.69), ` vu` (-15.82), `�` (-16.57) |
| L2→L3 | ` ble` → ` absol` | content→'absol' | ` absol` (+17.5), ` Kun` (+17.25), `とも` (+17.12) | `hole` (-16.35), ` misc` (-16.48), ` BS` (-16.6) |
| L3→L4 | ` absol` → ` ort` | content→'ort' | ` misc` (+17.58), ` CIM` (+16.14), ` surf` (+15.83) | `とも` (-1.23), ` Far` (-16.25), ` ble` (-16.43) |
| L5→L6 | ` ort` → ` takeaway` | content→'takeaway' | ` tight` (+17.41), ` CIM` (+16.72), ` Nu` (+15.72) | ` Vir` (-0.73), ` ort` (-1.92), ` Ci` (-15.83) |
| L6→L7 | ` takeaway` → ` tight` | content→'tight' | ` blind` (+15.84), ` harmon` (+15.34), ` honest` (+15.34) | ` orth` (-1.19), ` Nu` (-15.72), ` misc` (-15.91) |
| L10→L11 | ` tight` → `<|endoftext|>` | content→padding | ` answer` (+15.25), ` coaching` (+15.13), ` solution` (+14.81) | ` important` (-1.03), ` tight` (-1.4), ` Key` (-15.28) |
| L13→L14 | `<|endoftext|>` → ` success` | padding→'success' | `o` (+15.42), ` Re` (+15.11), ` success` (+1.83) | `Successful` (-0.11), ` successful` (-0.17), `<|endoftext|>` (-2.3) |
| L16→L17 | ` success` → ` the` | content→'the' | ` this` (+15.26), ` to` (+13.2), ` the` (+2.99) | ` for` (-2.26), ` success` (-2.39), ` re` (-3.26) |
| L18→L19 | ` the` → ` successful` | content→'successful' | ` successfully` (+12.35), ` Successful` (+11.42), ` successful` (+4.03) | ` this` (-0.35), ` unlocking` (-1.47), ` the` (-2.72) |
| L21→L22 | ` successful` → ` success` | content→'success' | ` creating` (+15.84), ` building` (+2.65), ` this` (+2.34) | ` a` (1.71), ` successfully` (0.59), ` success` (0.21) |
| L22→L23 | ` success` → ` the` | content→'the' | ` effective` (+15.55), ` understanding` (+15.49), ` the` (+0.77) | ` achieving` (-0.41), ` successful` (-1.16), ` success` (-1.73) |

**Decision arc:** L1: `Ba` → L2: `ble` → L3: `absol` → L5: `ort` → L6: `takeaway` → L10: `tight` → L13: `<|endoftext|>` → L16: `success` → L18: `the` → L21: `successful` → L22: `success` → L23: `the`

### Position 6: target = ` the`
- Convergence layer: **17**
- Decision transitions: **5**

| Transition | From → To | Semantic | Top gainers | Top losers |
|---|---|---|---|---|
| L4→L5 | ` eng` → ` door` | content→'door' | `<|endoftext|>` (+17.02), ` pill` (+15.14), ` Jack` (+15.08) | `ao` (-0.39), ` eng` (-2.2), ` equilibrium` (-15.66) |
| L11→L12 | ` door` → `<|endoftext|>` | content→padding | ` Monday` (+15.45), ` hier` (+15.07), ` positions` (+14.95) | ` imaginary` (-1.36), ` door` (-1.99), ` literally` (-15.06) |
| L14→L15 | `<|endoftext|>` → `‑` | padding→'‑' | ` (` (+16.73), ` somewhere` (+16.48), ` inside` (+15.48) | `<|endoftext|>` (-3.16), `-m` (-13.65), ` placed` (-13.71) |
| L15→L16 | `‑` → ` (` | content→punctuation | ` the` (+17.99), `,` (+16.93), ` that` (+15.93) | `-f` (-15.36), ` inside` (-15.48), ` somewhere` (-16.48) |
| L16→L17 | ` (` → ` the` | punctuation→'the' | ` a` (+15.73), ` this` (+13.23), ` [` (+11.35) | `—` (-15.12), `"` (-15.37), `?` (-15.37) |

**Decision arc:** L4: `eng` → L11: `door` → L14: `<|endoftext|>` → L15: `‑` → L16: `(` → L17: `the`

### Position 11: target = ` can`
- Convergence layer: **18**
- Decision transitions: **12**

| Transition | From → To | Semantic | Top gainers | Top losers |
|---|---|---|---|---|
| L0→L1 | `hin` → ` miss` | content→'miss' | ` miss` (+17.26), ` noc` (+17.26), `itas` (+16.38) | ` ste` (-1.62), `@test` (-15.88), `iali` (-16.0) |
| L1→L2 | ` miss` → `Miss` | content→'Miss' | ` imb` (+17.2), ` Ss` (+16.7), ` Shields` (+16.45) | ` зав` (-16.01), ` Salamanca` (-16.51), `.vol` (-16.63) |
| L3→L4 | `Miss` → ` hi` | content→letter/short | ` hi` (+18.55), ` tra` (+16.49), ` esc` (+16.36) | ` stra` (-0.12), ` ste` (-1.06), ` �` (-16.3) |
| L6→L7 | ` hi` → ` spl` | letter/short→'spl' | ` bill` (+17.76), ` tun` (+16.2), ` thes` (+16.08) | ` tra` (-15.79), ` esc` (-15.79), ` quer` (-15.97) |
| L7→L8 | ` spl` → ` properly` | content→'properly' | ` recon` (+17.59), ` abstract` (+16.4), ` grand` (+16.21) | ` divides` (-15.33), ` metaphor` (-15.39), ` stra` (-16.01) |
| L10→L11 | ` properly` → ` locks` | content→'locks' | ` unlocking` (+17.11), ` downstairs` (+16.67), ` unlocked` (+16.23) | ` uniform` (-15.68), ` access` (-15.74), ` information` (-15.8) |
| L11→L12 | ` locks` → `—` | content→punctuation | ` access` (+16.61), ` correspond` (+16.11), `-f` (+16.04) | ` door` (-15.73), ` blind` (-15.8), ` lock` (-15.86) |
| L13→L14 | `—` → ` ` | punctuation→padding | ` ` (+16.47), `‑` (+15.91), ` and` (+15.72) | ` unlocking` (-15.42), ` locked` (-15.74), ` access` (-15.99) |
| L14→L15 | ` ` → ` won't` | padding→'won't' | ` won't` (+17.49), ` only` (+17.24), ` is` (+17.05) | ` to` (-15.6), `Keys` (-15.66), `/key` (-15.91) |
| L15→L16 | ` won't` → ` only` | content→'only' | ` [` (+16.16), ` (` (+15.78), ` are` (+15.78) | ` at` (-16.11), ` ` (-16.24), `‑` (-16.55) |
| L16→L17 | ` only` → ` can't` | content→'can't' | ` will` (+16.5), ` just` (+16.31), ` should` (+16.25) | ` is` (-0.59), ` never` (-15.66), ` (` (-15.78) |
| L17→L18 | ` can't` → ` can` | content→'can' | ` may` (+17.13), ` never` (+16.63), ` also` (+16.57) | ` can't` (-0.81), ` just` (-16.31), ` will` (-16.5) |

**Decision arc:** L0: `hin` → L1: `miss` → L3: `Miss` → L6: `hi` → L7: `spl` → L10: `properly` → L11: `locks` → L13: `—` → L14: `` → L15: `won't` → L16: `only` → L17: `can't` → L18: `can`

### Position 7: target = ` table`
- Convergence layer: **19**
- Decision transitions: **16**

| Transition | From → To | Semantic | Top gainers | Top losers |
|---|---|---|---|---|
| L0→L1 | ` beh` → ` eng` | content→'eng' | ` eng` (+19.81), ` Easter` (+17.31), ` Balk` (+15.44) | ` NC` (-16.27), ` Kit` (-16.27), `ekl` (-16.39) |
| L2→L3 | ` eng` → ` hed` | content→'hed' | `企` (+16.82), `-thumb` (+16.01), `458` (+16.01) | `�` (-11.12), ` circ` (-11.99), ` dist` (-12.24) |
| L3→L4 | ` hed` → ` eng` | content→'eng' | ` swing` (+16.08), `付` (+15.96), ` cass` (+15.65) | ` barg` (-15.38), ` filing` (-15.38), `458` (-16.01) |
| L4→L5 | ` eng` → ` spl` | content→'spl' | ` spl` (+17.41), `458` (+17.16), ` drives` (+16.79) | ` cass` (-15.65), `付` (-15.96), `企` (-16.08) |
| L5→L6 | ` spl` → `/fire` | content→'/fire' | `/fire` (+16.84), ` slips` (+16.47), ` heel` (+16.41) | `-own` (-16.16), ` medi` (-16.16), ` bent` (-16.16) |
| L7→L8 | `/fire` → ` thr` | content→'thr' | ` clasp` (+16.87), ` pocket` (+16.12), ` cupboard` (+16.05) | ` negoti` (-0.48), `匙` (-0.6), ` �` (-15.97) |
| L8→L9 | ` thr` → ` door` | content→'door' | ` doors` (+16.47), ` cabinet` (+16.09), ` slips` (+15.78) | ` clasp` (0.1), ` thr` (-1.52), ` drives` (-16.05) |
| L10→L11 | ` door` → ` downstairs` | content→'downstairs' | ` upstairs` (+16.69), `钥` (+15.25), `/key` (+14.13) | ` cupboard` (-1.89), ` thr` (-15.58), `<|endoftext|>` (-15.77) |
| L12→L13 | ` downstairs` → ` upstairs` | content→'upstairs' | `/key` (+16.17), `—` (+16.17), `<|endoftext|>` (+15.36) | ` door` (-0.39), ` downstairs` (-0.58), ` clasp` (-14.81) |
| L13→L14 | ` upstairs` → `匙` | content→letter/short | ` keys` (+15.87), `‑` (+15.62), ` stored` (+1.2) | `匙` (0.14), ` upstairs` (-1.36), ` downstairs` (-2.55) |
| L14→L15 | `匙` → ` for` | letter/short→'for' | ` for` (+17.22), ` long` (+16.53), `"` (+16.47) | `钥` (-16.49), `<|endoftext|>` (-16.49), `/key` (-16.93) |
| L15→L16 | ` for` → ` ` | content→padding | ` ` (+18.05), `
` (+17.36), ` on` (+17.05) | ` separate` (-16.03), ` attached` (-16.35), `"` (-16.47) |
| L16→L17 | ` ` → ` first` | padding→'first' | ` first` (+18.53), ` right` (+16.66), ` top` (+16.59) | `_` (-15.99), ` L` (-16.11), ` **` (-16.61) |
| L18→L19 | ` first` → ` table` | content→'table' | ` bottom` (+15.74), ` floor` (+15.37), ` left` (+1.9) | ` first` (-0.66), ` second` (-1.41), ` same` (-1.54) |
| L19→L20 | ` table` → ` second` | content→'second' | ` third` (+16.33), ` bottom` (+1.96), ` floor` (+1.71) | ` first` (-0.42), ` left` (-0.79), ` right` (-0.79) |
| L20→L21 | ` second` → ` table` | content→'table' | ` shelf` (+16.13), ` desk` (+15.38), ` wall` (+15.0) | ` second` (-2.95), ` third` (-16.33), ` ` (-16.7) |

**Decision arc:** L0: `beh` → L2: `eng` → L3: `hed` → L4: `eng` → L5: `spl` → L7: `/fire` → L8: `thr` → L10: `door` → L12: `downstairs` → L13: `upstairs` → L14: `匙` → L15: `for` → L16: `` → L18: `first` → L19: `table` → L20: `second` → L21: `table`

### Position 1: target = ` are`
- Convergence layer: **20**
- Decision transitions: **9**

| Transition | From → To | Semantic | Top gainers | Top losers |
|---|---|---|---|---|
| L0→L1 | `Hv` → ` Aren` | letter/short→'Aren' | ` Aren` (+18.26), `icl` (+17.14), ` �` (+16.39) | `OH` (-1.32), `heg` (-16.21), `Mol` (-16.21) |
| L2→L3 | ` Aren` → ` Sel` | content→'Sel' | `Virgin` (+16.74), `epi` (+16.24), ` MC` (+15.8) | ` Mel` (-0.44), ` Aren` (-0.94), ` Miss` (-15.81) |
| L3→L4 | ` Sel` → ` takeaway` | content→'takeaway' | `bert` (+16.09), ` Vir` (+15.97), `Kun` (+15.66) | ` Sel` (-0.96), `oku` (-1.08), `.generic` (-15.99) |
| L5→L6 | ` takeaway` → ` important` | content→'important' | ` MC` (+15.84), `善` (+15.59), ` converse` (+15.46) | ` takeaway` (-1.16), `oku` (-15.81), `�` (-15.93) |
| L9→L10 | ` important` → `<|endoftext|>` | content→padding | `ver` (+14.94), `<|endoftext|>` (+0.72), ` summary` (+-0.09) | `Key` (-0.34), ` important` (-0.84), `ra` (-1.28) |
| L13→L14 | `<|endoftext|>` → `Key` | padding→'Key' | `Keys` (+16.22), ` in` (+15.91), ` for` (+15.85) | `Key` (-0.08), ` important` (-15.86), `.key` (-16.24) |
| L14→L15 | `Key` → ` and` | content→'and' | ` and` (+18.15), ` value` (+15.71), ` for` (+2.05) | `A` (-0.83), `key` (-1.01), `Key` (-1.2) |
| L15→L16 | ` and` → ` for` | content→'for' | ` ` (+16.94), ` to` (+15.94), ` with` (+15.26) | ` key` (-1.58), ` value` (-15.71), `key` (-15.96) |
| L19→L20 | ` for` → ` are` | content→'are' | ` used` (+14.9), ` are` (+3.74), ` that` (+0.74) | ` '` (-0.07), ` and` (-0.38), ` of` (-0.45) |

**Decision arc:** L0: `Hv` → L2: `Aren` → L3: `Sel` → L5: `takeaway` → L9: `important` → L13: `<|endoftext|>` → L14: `Key` → L15: `and` → L19: `for` → L20: `are`

### Position 4: target = ` are`
- Convergence layer: **20**
- Decision transitions: **7**

| Transition | From → To | Semantic | Top gainers | Top losers |
|---|---|---|---|---|
| L0→L1 | ` Scha` → `iliary` | content→'iliary' | ` mens` (+17.77), `.ck` (+17.02), ` Bers` (+17.02) | ` Ac` (-0.43), ` Scha` (-1.05), `495` (-15.95) |
| L1→L2 | `iliary` → ` mens` | content→'mens' | `.tf` (+16.77), ` furnishing` (+16.52), `室` (+16.33) | `/ac` (-16.15), ` precip` (-16.52), ` Lif` (-16.52) |
| L2→L3 | ` mens` → `.ck` | content→'.ck' | ` thes` (+16.98), `onis` (+16.79), ` conven` (+16.6) | `526` (-16.27), ` Ac` (-16.39), ` furnishing` (-16.52) |
| L4→L5 | `.ck` → `.bl` | content→'.bl' | ` harmon` (+16.21), `室` (+16.09), `bag` (+16.09) | ` mens` (-15.95), ` asi` (-16.01), ` conven` (-16.08) |
| L5→L6 | `.bl` → `<|endoftext|>` | content→padding | `<|endoftext|>` (+18.69), ` doors` (+16.07), `ace` (+15.76) | `.ck` (-16.09), `bag` (-16.09), `mates` (-16.34) |
| L14→L15 | `<|endoftext|>` → `,` | padding→punctuation | `'s` (+17.56), ` and` (+17.38), `/` (+17.38) | ` locks` (-15.29), `

` (-15.47), `‑` (-15.54) |
| L19→L20 | `,` → ` are` | punctuation→'are' | ` were` (+17.26), ` should` (+14.01), ` will` (+13.57) | ` is` (-4.08), ` in` (-15.16), ` was` (-15.28) |

**Decision arc:** L0: `Scha` → L1: `iliary` → L2: `mens` → L4: `.ck` → L5: `.bl` → L14: `<|endoftext|>` → L19: `,` → L20: `are`

### Position 10: target = ` you`
- Convergence layer: **20**
- Decision transitions: **6**

| Transition | From → To | Semantic | Top gainers | Top losers |
|---|---|---|---|---|
| L7→L8 | ` far` → ` presumably` | content→'presumably' | ` visibility` (+15.92), `Den` (+14.99), ` integrated` (+14.8) | ` far` (-1.18), `Far` (-14.98), ` mechanical` (-15.1) |
| L11→L12 | ` presumably` → ` visibility` | content→'visibility' | ` allow` (+16.16), ` accessible` (+16.1), ` safe` (+15.91) | ` presumably` (-1.6), ` elevator` (-15.26), ` door` (-15.51) |
| L12→L13 | ` visibility` → ` L` | content→letter/short | ` L` (+16.71), `‑` (+16.28), ` accessibility` (+15.78) | ` rest` (-15.47), ` conveniently` (-15.85), ` arrangement` (-15.85) |
| L14→L15 | ` L` → ` (` | letter/short→punctuation | ` that` (+16.65), ` and` (+15.9), ` starting` (+15.77) | `L` (-15.2), ` ex` (-15.39), ` access` (-15.39) |
| L16→L17 | ` (` → ` the` | punctuation→'the' | ` we` (+16.18), ` no` (+15.12), ` there` (+14.68) | ` in` (-15.26), ` ` (-15.38), ` or` (-15.63) |
| L19→L20 | ` the` → ` you` | content→'you' | ` you'll` (+14.56), ` I` (+1.47), ` there` (+1.35) | ` you` (0.6), ` it` (0.47), ` that` (0.1) |

**Decision arc:** L7: `far` → L11: `presumably` → L12: `visibility` → L14: `L` → L16: `(` → L19: `the` → L20: `you`

### Position 0: target = ` code`
- Convergence layer: **21**
- Decision transitions: **9**

| Transition | From → To | Semantic | Top gainers | Top losers |
|---|---|---|---|---|
| L0→L1 | `ira` → ` entr` | content→'entr' | `utum` (+16.21), `spe` (+15.83), ` epit` (+15.08) | ` doma` (-15.09), ` bride` (-15.09), ` cre` (-15.71) |
| L2→L3 | ` entr` → `orem` | content→'orem' | `321` (+16.31), `Emb` (+16.06), `orem` (+3.44) | ` epit` (-0.19), `spe` (-1.44), ` entr` (-2.69) |
| L3→L4 | `orem` → ` epit` | content→'epit' | ` gro` (+16.87), `DE` (+16.87), ` pit` (+16.62) | `orem` (-1.32), `spe` (-16.19), `oretical` (-16.44) |
| L4→L5 | ` epit` → `Emb` | content→'Emb' | `assist` (+16.01), ` developed` (+15.7), `DD` (+15.63) | ` pit` (-0.86), `leg` (-16.37), ` gro` (-16.87) |
| L5→L6 | `Emb` → ` BOTH` | content→'BOTH' | ` BOTH` (+18.75), ` både` (+18.63), `both` (+17.63) | `orem` (-16.88), `321` (-17.13), ` epit` (-17.88) |
| L16→L17 | ` BOTH` → `By` | content→letter/short | `By` (+17.09), `();
` (+16.71), ` and` (+16.59) | ` ]
` (-16.07), ` Both` (-16.94), `both` (-17.82) |
| L17→L18 | `By` → ` ` | letter/short→padding | ` ` (+17.95), ` main` (+17.08), ` `` (+16.7) | `);
` (-16.46), `_` (-16.52), `Field` (-16.59) |
| L18→L19 | ` ` → ` main` | padding→'main' | ` next` (+16.8), ` "` (+16.05), ` overall` (+15.73) | `
` (-0.53), ``` (-15.95), ` =` (-16.01) |
| L20→L21 | ` main` → ` code` | content→'code' | ` function` (+16.05), ` following` (+15.8), ` user` (+15.73) | ` first` (-0.58), ` main` (-1.27), ` overall` (-15.62) |

**Decision arc:** L0: `ira` → L2: `entr` → L3: `orem` → L4: `epit` → L5: `Emb` → L16: `BOTH` → L17: `By` → L18: `` → L20: `main` → L21: `code`

### Position 3: target = ` success`
- Convergence layer: **21**
- Decision transitions: **14**

| Transition | From → To | Semantic | Top gainers | Top losers |
|---|---|---|---|---|
| L0→L1 | `-cond` → ` cant` | content→'cant' | ` eng` (+18.06), `DV` (+16.68), `Cm` (+16.68) | ` doma` (-15.98), ` Griffith` (-16.11), ` ca` (-16.73) |
| L1→L2 | ` cant` → ` eng` | content→'eng' | ` mil` (+16.06), ` Gibbs` (+15.44), ` Howard` (+15.19) | ` HO` (-15.18), `ekl` (-15.37), ` sust` (-15.93) |
| L4→L5 | ` eng` → `ore` | content→'ore' | ` maj` (+15.76), ` tim` (+15.7), ` sur` (+15.64) | ` inse` (-15.69), `dele` (-15.94), ` NC` (-16.0) |
| L5→L6 | `ore` → `ort` | content→'ort' | ` fut` (+16.5), ` address` (+16.12), ` apparently` (+15.62) | ` Ax` (-14.57), ` interfer` (-14.89), ` eng` (-15.45) |
| L7→L8 | `ort` → ` tight` | content→'tight' | ` vir` (+16.34), ` hom` (+16.15), ` host` (+16.15) | ` modified` (-15.75), `zou` (-16.0), ` joint` (-16.31) |
| L9→L10 | ` tight` → `<|endoftext|>` | content→padding | `<|endoftext|>` (+16.82), ` door` (+16.01), ` analogy` (+15.94) | ` tight` (-1.51), ` routes` (-15.52), ` fut` (-15.71) |
| L11→L12 | `<|endoftext|>` → ` keys` | padding→'keys' | ` unlocking` (+16.69), ` codes` (+15.51), ` unlock` (+15.26) | ` vir` (-15.48), ` tight` (-15.6), ` sur` (-15.6) |
| L12→L13 | ` keys` → `<|endoftext|>` | content→padding | ` lost` (+15.78), `o` (+15.16), `<|endoftext|>` (+2.21) | ` door` (0.09), `ort` (-0.54), ` keys` (-1.23) |
| L13→L14 | `<|endoftext|>` → `o` | padding→letter/short | `i` (+16.31), `in` (+15.81), ` ` (+15.81) | ` door` (-15.22), `ort` (-15.72), ` lost` (-15.78) |
| L14→L15 | `o` → ` keys` | letter/short→'keys' | ` for` (+16.28), ` L` (+15.97), ` in` (+15.97) | `in` (-15.81), ` issue` (-15.81), `i` (-16.31) |
| L15→L16 | ` keys` → ` ` | content→padding | `
` (+18.07), ` state` (+16.0), ` `` (+15.94) | ` access` (-15.72), `or` (-15.84), ` to` (-15.91) |
| L17→L18 | ` ` → ` `` | padding→'`' | ` success` (+16.12), ` map` (+15.37), ` second` (+15.06) | ` data` (-1.07), ` ` (-1.95), ` list` (-15.07) |
| L19→L20 | ` `` → ` map` | content→'map' | ` dictionary` (+16.6), ` successful` (+16.22), ` success` (+1.56) | ` '` (-0.94), ` ` (-1.12), ` `` (-1.75) |
| L20→L21 | ` map` → ` success` | content→'success' | ` key` (+16.15), ` table` (+15.96), ` puzzle` (+15.96) | ` first` (-0.63), ` map` (-0.7), ` '` (-16.22) |

**Decision arc:** L0: `-cond` → L1: `cant` → L4: `eng` → L5: `ore` → L7: `ort` → L9: `tight` → L11: `<|endoftext|>` → L12: `keys` → L13: `<|endoftext|>` → L14: `o` → L15: `keys` → L17: `` → L19: ``` → L20: `map` → L21: `success`

### Position 8: target = `."`
- Convergence layer: **22**
- Decision transitions: **8**

| Transition | From → To | Semantic | Top gainers | Top losers |
|---|---|---|---|---|
| L3→L4 | ` conven` → ` hil` | content→'hil' | ` hil` (+18.49), ` firmly` (+17.99), `holders` (+16.8) | ` pin` (-15.14), ` precip` (-15.2), `/ms` (-15.2) |
| L4→L5 | ` hil` → ` firmly` | content→'firmly' | ` pocket` (+17.27), `用品` (+16.52), ` firm` (+16.08) | `盘` (-15.42), `/legal` (-15.67), `�` (-15.67) |
| L5→L6 | ` firmly` → ` hil` | content→'hil' | `<|endoftext|>` (+17.49), ` chest` (+16.42), ` cr` (+16.24) | `�` (-15.58), `Pla` (-15.71), `/head` (-15.77) |
| L6→L7 | ` hil` → `<|endoftext|>` | content→padding | `�` (+15.93), `\S` (+15.36), `‑` (+15.3) | ` chest` (-0.75), ` pocket` (-15.92), ` cur` (-15.92) |
| L14→L15 | `<|endoftext|>` → `‑` | padding→'‑' | ` or` (+16.29), `/s` (+16.04), `—the` (+15.67) | `<|endoftext|>` (-1.96), `/table` (-14.57), `上的` (-14.82) |
| L15→L16 | `‑` → `,` | content→punctuation | `,` (+18.75), `.` (+18.62), ` (` (+17.75) | `/s` (-16.04), ` located` (-16.17), `/h` (-16.23) |
| L16→L17 | `,` → `.` | punctuation→punctuation | `.
` (+13.75), `.

` (+13.68), `."` (+13.37) | ` (` (-3.5), `—` (-15.62), ` or` (-15.87) |
| L21→L22 | `.` → `."` | punctuation→'."' | `."
` (+16.55), ` next` (+15.93), `"` (+15.86) | `,` (-0.56), `.` (-1.19), `;` (-14.86) |

**Decision arc:** L3: `conven` → L4: `hil` → L5: `firmly` → L6: `hil` → L14: `<|endoftext|>` → L15: `‑` → L16: `,` → L21: `.` → L22: `."`

### Position 5: target = ` in`
- Convergence layer: **23**
- Decision transitions: **17**

| Transition | From → To | Semantic | Top gainers | Top losers |
|---|---|---|---|---|
| L1→L2 | ` �` → ` buzz` | content→'buzz' | `blind` (+17.58), `lot` (+17.08), ` Mother` (+17.08) | `SOS` (-16.11), ` desp` (-16.11), `ears` (-16.61) |
| L2→L3 | ` buzz` → ` ce` | content→letter/short | ` don` (+17.43), ` mus` (+16.49), ` generous` (+16.37) | ` Nin` (-15.96), ` Jack` (-16.21), ` GO` (-16.46) |
| L3→L4 | ` ce` → ` don` | letter/short→'don' | ` pull` (+17.45), `hoes` (+16.64), ` kin` (+16.45) | ` Mother` (-16.12), `rit` (-16.18), ` generous` (-16.37) |
| L4→L5 | ` don` → ` whatever` | content→'whatever' | ` limp` (+16.93), ` kne` (+16.93), `hole` (+16.81) | ` kin` (-16.45), `hoes` (-16.64), ` buzz` (-16.64) |
| L5→L6 | ` whatever` → `terior` | content→'terior' | ` cho` (+17.56), `-guid` (+16.75), ` routes` (+16.5) | ` Vir` (-16.56), ` entail` (-16.68), ` don` (-16.68) |
| L6→L7 | `terior` → ` blind` | content→'blind' | ` vers` (+16.14), `…` (+15.95), ` metaphor` (+15.83) | `458` (-16.19), ` literally` (-16.38), ` routes` (-16.5) |
| L8→L9 | ` blind` → `…` | content→'…' | `—you` (+16.41), ` literally` (+15.72), `…` (+1.27) | ` vers` (-0.29), ` pull` (-0.79), ` cho` (-1.17) |
| L10→L11 | `…` → `—` | content→punctuation | `匙` (+16.54), ` locks` (+16.35), ` locked` (+16.16) | `—or` (-15.54), ` Ms` (-15.73), ` pull` (-15.86) |
| L12→L13 | `—` → ` locked` | punctuation→'locked' | `_unlock` (+16.33), `_lock` (+15.95), `Lock` (+15.83) | `or` (-0.89), ` door` (-15.46), ` ` (-15.59) |
| L13→L14 | ` locked` → `—` | content→punctuation | `:` (+16.78), ` and` (+16.16), `钥` (+16.03) | `Lock` (-15.83), `_lock` (-15.95), `_unlock` (-16.33) |
| L14→L15 | `—` → ` (` | punctuation→punctuation | ` (` (+17.43), ` located` (+17.37), ` only` (+17.06) | `,` (-15.91), `…` (-15.91), `钥` (-16.03) |
| L15→L16 | ` (` → `"` | punctuation→punctuation | `"` (+18.71), ` ` (+16.96), `'` (+16.65) | ` accessible` (-16.18), `—` (-16.24), `?` (-16.56) |
| L16→L17 | `"` → ` not` | punctuation→'not' | ` a` (+16.24), ` on` (+15.81), ` the` (+15.74) | `:
` (-15.53), `...` (-16.46), `'` (-16.65) |
| L19→L20 | ` not` → `:` | content→punctuation | ` missing` (+16.23), ` lost` (+0.79), ` hidden` (+0.73) | ` the` (0.42), `:` (-0.02), ` ` (-0.71) |
| L20→L21 | `:` → ` hidden` | punctuation→'hidden' | ` locked` (+16.3), ` inside` (+15.3), ` missing` (+1.01) | ` not` (-0.49), ` in` (-0.92), `:` (-1.42) |
| L21→L22 | ` hidden` → ` on` | content→letter/short | ` located` (+16.26), ` on` (+1.7), ` in` (+0.83) | ` the` (-0.36), ` not` (-0.67), ` hidden` (-0.8) |
| L22→L23 | ` on` → ` in` | letter/short→letter/short | ` a` (+16.28), ` ` (+16.03), ` the` (+0.84) | ` on` (-0.66), ` hidden` (-0.85), ` missing` (-1.16) |

**Decision arc:** L1: `�` → L2: `buzz` → L3: `ce` → L4: `don` → L5: `whatever` → L6: `terior` → L8: `blind` → L10: `…` → L12: `—` → L13: `locked` → L14: `—` → L15: `(` → L16: `"` → L19: `not` → L20: `:` → L21: `hidden` → L22: `on` → L23: `in`

## Summary
- Mean convergence layer: 19.5
- Range: L16 – L23
- Total decision transitions across all positions: 120

## CASCADE Insight

Each decision transition (top-1 change) identifies a layer where the model's
prediction shifts. In CASCADE mode, the logit-space difference `Δz = z^(l+1) − z^(l)`
at a decision layer is a **self-supervised steering direction**: it captures the
semantic decision the model makes, without requiring curated contrast pairs.

The gauge-safe projection `v_steer = (CW)⁺ · C·Δz` maps this direction into the
student's embedding space, yielding a closed-form steering vector. This is
fundamentally different from contrastive activation addition (CAA), which requires
100+ positive/negative example pairs per concept.
