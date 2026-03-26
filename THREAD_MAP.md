# Research Thread Map

Mechanistic interpretability of gpt-oss-20b, organized as 14 research threads at three maturity levels.

## Solid — publication-ready results

| # | Thread | Objective | Key Figure | Scripts | Runs | Directory |
|---|--------|-----------|------------|---------|------|-----------|
| 1 | [Convergence / Logit Lens](threads/solid/1-convergence-logit-lens/) | Measure convergence depth | fig1 | 2 | 4 | `threads/solid/1-convergence-logit-lens/` |
| 2 | [Late-Layer Ablation](threads/solid/2-late-layer-ablation/) | Measure layer criticality | fig2 | 1 + shared | 6 | `threads/solid/2-late-layer-ablation/` |
| 3 | [Analysis Set Filtering](threads/solid/3-analysis-set-filtering/) | Establish analysis methodology | fig3 | 2 + shared | 3 | `threads/solid/3-analysis-set-filtering/` |
| 4 | [Decision Trajectories](threads/solid/4-decision-trajectories/) | Characterize decision structure | fig4 | 2 | 1 | `threads/solid/4-decision-trajectories/` |
| 5 | [Hydra / Head Redundancy](threads/solid/5-hydra-head-redundancy/) | Measure head redundancy | fig5 | 1 + shared | 1 | `threads/solid/5-hydra-head-redundancy/` |
| 6 | [Direct Vocab Steering](threads/solid/6-direct-vocab-steering/) | Validate steering precision | memo figs (4) | 3 | 13 | `threads/solid/6-direct-vocab-steering/` |
| 7 | [Channel Probing](threads/solid/7-channel-probing/) | Resolve steering to channel level | fig6 | 3 | 9 | `threads/solid/7-channel-probing/` |
| 8 | [Selectivity](threads/solid/8-selectivity/) | Measure steering specificity | fig7 | 2 | 8 | `threads/solid/8-selectivity/` |
| 14 | [Bregman Geometry](threads/solid/14-bregman-geometry/) | Measure geometric prerequisites for linear intervention | fig9, fig10, fig11 | 1 | — | `threads/solid/14-bregman-geometry/` |

## In progress — code exists, experiments need expansion

| # | Thread | Objective | Key Figure | Scripts | Runs | Gap | Directory |
|---|--------|-----------|------------|---------|------|-----|-----------|
| 9 | [Feature Extraction](threads/in-progress/9-feature-extraction/) | Measure computational modes | fig8 | 2 | 3 | Connection to downstream tasks not demonstrated | `threads/in-progress/9-feature-extraction/` |
| 10 | [Bridge / Cross-Model](threads/in-progress/10-bridge-cross-model/) | Validate cross-model generalization | — | 3 | 5 | Only 1 non-gpt-oss model tested | `threads/in-progress/10-bridge-cross-model/` |
| 15 | [MoE Expert Readouts](threads/in-progress/15-expert-readouts/) | Characterise expert specialisation across 768 modules | — | 2 | — | Requires model run; figures not yet generated | `threads/in-progress/15-expert-readouts/` |

## Theoretical — framework documented, implementation incomplete

| # | Thread | Objective | Gap | Directory |
|---|--------|-----------|-----|-----------|
| 11 | [CASCADE Distillation](threads/theoretical/11-cascade-distillation/) | Automate steering direction extraction | Stubs only — no working training code | `threads/theoretical/11-cascade-distillation/` |
| 12 | [Geometric Framework](threads/theoretical/12-geometric-framework/) | Enable principled cross-model comparison | Zero empirical validation | `threads/theoretical/12-geometric-framework/` |
| 13 | [Attention Path Sensitivity](threads/theoretical/13-attention-path-sensitivity/) | Measure attention path sensitivity | Single proof-of-concept case | `threads/theoretical/13-attention-path-sensitivity/` |

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
- `figures/` — 19 publication-quality figures (PDF + PNG pairs)
- `doc/references/` — 3 literature reviews + 5 academic papers
