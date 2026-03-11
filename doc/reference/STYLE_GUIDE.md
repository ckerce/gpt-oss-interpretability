# Style Guide

## Goal
Preserve the parts of the `symbolic-transformer` coding style that improve readability and retoolability, while reducing the repo drift and path confusion that accumulated over time.

## High-level rules
- prefer explicit configuration over hard-coded experiment logic
- keep backend details out of benchmark logic
- make data flow and intervention flow obvious
- comment for structural understanding, not narration
- optimize only after instrumentation and correctness are stable

## Banner comment style
Yes: keep the old banner-comment style.

Use it for:
- major module sections
- architecture-critical blocks
- execution phases in long scripts
- adapter boundaries

Recommended format:
```python
###############################################################################
#
# Activation Capture
#
###############################################################################
```

Do not use it for:
- every short helper
- tiny control-flow branches
- repetitive low-value notes

## Comment layering
Use three levels only:

1. Banner comments
- for major sections

2. Docstrings
- for public classes and functions
- explain contract, not implementation trivia

3. Sparse inline comments
- only where tensor logic, hooks, or side effects are not obvious

## Module layout
Prefer modules with a visible structure:
- imports
- constants / enums / dataclasses
- public classes
- small helper functions
- CLI entrypoint if needed

Long files should be broken into banner-comment sections.

## Config style
Keep the Python config module pattern.

Preferred shape:
```python
config = BenchmarkConfig(
    ...
)

config.experiment_name = "..."
config.output_dir = "..."
```

This matches the existing `symbolic-transformer` feel and keeps experimentation quick.

## Backend abstraction
Never let benchmark code know backend internals.

Backend modules may use:
- `einsum`
- PyTorch primitives
- custom kernels
- layout-sensitive fused paths

But benchmark modules should only see:
- score prompt / choice
- apply intervention
- clear intervention
- capture requested artifacts

## On `einsum`
Using `einsum` broadly is acceptable in research code if:
- tensor semantics remain clear
- hot paths are isolated
- you can later swap implementations without rewriting the stack

Do not bake the benchmark design around `einsum` assumptions.

## Naming conventions
- keep names literal and descriptive
- prefer `layer_indices`, `head_indices`, `router_logits`, `choice_logprobs`
- avoid overly compressed names in benchmark or reporting code

## Reporting outputs
Every runnable config should emit:
- `summary.json`
- `case_results.csv`
- `report.md`

That should be treated as standard repo behavior.

## What to avoid
- implicit path hacks spread across modules
- benchmark logic embedded in backend code
- stale comments that contradict the code
- one-off paper scripts copied without a shared interface
