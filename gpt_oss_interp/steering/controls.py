"""Null and matched control interventions for causal validation.

Produces baselines:
    - shuffled head labels
    - same-norm random token-slice directions
    - within-family label permutation
    - low-ranked channel controls

All controls must accept the same interface as real interventions
so comparison code stays symmetric.
"""
