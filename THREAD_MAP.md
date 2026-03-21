# Research Thread Map

Mechanistic interpretability of gpt-oss-20b, organized around a five-beat narrative arc:

**Measure** (where computation happens) **->** **Structure** (how it's organized) **->** **Steer** (prove understanding via intervention) **->** **Automate** (extract directions automatically) **->** **Generalize** (compare across models)

## Solid — publication-ready results

| # | Thread | Beat | Key Figure | Scripts | Runs | Directory |
|---|--------|------|------------|---------|------|-----------|
| 1 | [Convergence / Logit Lens](threads/solid/1-convergence-logit-lens/) | Measure | fig1 | 2 | 4 | `threads/solid/1-convergence-logit-lens/` |
| 2 | [Late-Layer Ablation](threads/solid/2-late-layer-ablation/) | Measure | fig2 | 1 + shared | 6 | `threads/solid/2-late-layer-ablation/` |
| 3 | [Analysis Set Filtering](threads/solid/3-analysis-set-filtering/) | Measure | fig3 | 2 + shared | 3 | `threads/solid/3-analysis-set-filtering/` |
| 4 | [Decision Trajectories](threads/solid/4-decision-trajectories/) | Structure | fig4 | 2 | 1 | `threads/solid/4-decision-trajectories/` |
| 5 | [Hydra / Head Redundancy](threads/solid/5-hydra-head-redundancy/) | Structure | fig5 | 1 + shared | 1 | `threads/solid/5-hydra-head-redundancy/` |
| 6 | [Direct Vocab Steering](threads/solid/6-direct-vocab-steering/) | Steer | memo figs (4) | 3 | 13 | `threads/solid/6-direct-vocab-steering/` |

## In progress — code exists, experiments need expansion

| # | Thread | Beat | Scripts | Runs | Gap | Directory |
|---|--------|------|---------|------|-----|-----------|
| 7 | [Channel Probing](threads/in-progress/7-channel-probing/) | Steer | 2 | 7 | Needs figures | `threads/in-progress/7-channel-probing/` |
| 8 | [Selectivity](threads/in-progress/8-selectivity/) | Steer | 1 | 4 | Needs broader task coverage | `threads/in-progress/8-selectivity/` |
| 9 | [Feature Extraction](threads/in-progress/9-feature-extraction/) | Measure | 1 | 3 | Thin experiments (12 tokens) | `threads/in-progress/9-feature-extraction/` |
| 10 | [Bridge / Cross-Model](threads/in-progress/10-bridge-cross-model/) | Generalize | 3 | 5 | Only 1 non-gpt-oss model tested | `threads/in-progress/10-bridge-cross-model/` |

## Theoretical — framework documented, implementation incomplete

| # | Thread | Beat | Gap | Directory |
|---|--------|------|-----|-----------|
| 11 | [CASCADE Distillation](threads/theoretical/11-cascade-distillation/) | Automate | Stubs only — no working training code | `threads/theoretical/11-cascade-distillation/` |
| 12 | [Geometric Framework](threads/theoretical/12-geometric-framework/) | Generalize | Zero empirical validation | `threads/theoretical/12-geometric-framework/` |
| 13 | [Attention Path Sensitivity](threads/theoretical/13-attention-path-sensitivity/) | Structure | Single proof-of-concept case | `threads/theoretical/13-attention-path-sensitivity/` |

## Shared infrastructure

Scripts that serve multiple threads (remain in `scripts/`):

| Script | Used by threads |
|--------|----------------|
| `scripts/run_benchmark.py` | 2, 3, 5 (config-driven) |
| `scripts/generate_phase1_figures.py` | 1, 2, 3, 4, 5 (generates fig1–fig5) |
| `scripts/capture_routing.py` | General MoE utility |
| `scripts/inspect_model.py` | General model inspection |
| `scripts/one_off/` | 6 exploratory / one-off analysis scripts |

## Flat directories (cross-referenced in thread READMEs)

- `configs/` — 7 benchmark config files
- `runs/` — 52 experiment output directories
- `figures/` — 16 publication-quality figures (PDF + PNG pairs)
- `doc/references/` — 3 literature reviews + 5 academic papers
