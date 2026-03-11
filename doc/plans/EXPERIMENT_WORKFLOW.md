# Experiment Workflow

This repo uses a simple experiment-history workflow:

## Branch Types

- `bootstrap/...`
  - history-reconstruction or repository-organization branches
- `exp/...`
  - one branch per major experiment
- `fix/...`
  - targeted bugfixes or cleanup that are not experiment-specific

## Branch Naming

Use short, specific names:

- `exp/entry20-case-generation-spec`
- `exp/entry20-bridge-measurement`
- `exp/cross-vocab-cascade-feasibility`
- `fix/logit-lens-schema`

## Commit Pattern For Experiment Branches

Prefer `2-3` commits per experiment:

1. code/config
   - scripts
   - library changes
   - config files
2. docs/notebook
   - notebook entry
   - design memo
   - interpretation notes
3. outputs
   - `runs/...`
   - generated reports
   - regenerated analysis artifacts

If an experiment is tiny, `2` commits is fine:
- implementation
- outputs + notebook

## Output Layout

Each experiment should write to a dedicated directory under `runs/`.

Examples:

- `runs/bridge_preflight/`
- `runs/bridge_experiment/`
- `runs/cross_vocab_feasibility/`

Avoid mixing outputs from different experiments into one shared directory unless the experiment is explicitly a rerun of the same pipeline.

## Notebook Rule

When an experiment becomes real work rather than a loose idea:

- create the branch first
- add or update the relevant notebook/design entry
- then implement

This keeps the branch name, notebook entry, code, and outputs aligned.

## Merge Rule

Merge an experiment branch only when it has:

- a coherent code/config change set
- a matching notebook or design note
- preserved outputs if those outputs matter to future analysis

If the result is inconclusive but still useful, merge it anyway with an explicit notebook note saying it was negative or ambiguous.

## Default Starting Point

Unless there is a reason not to, start new experiment branches from:

- `bootstrap/research-history`

That branch is the organized baseline for the current repo state.
