"""Intervention spec objects.

Temporarily re-exports from the old location so new code can import
from gpt_oss_interp.steering.specs immediately.

Migration step (later, not now):
    1. Move content of gpt_oss_interp/interventions/specs.py here.
    2. Replace gpt_oss_interp/interventions/specs.py with:
           from gpt_oss_interp.steering.specs import *  # noqa: F401,F403
"""
from gpt_oss_interp.interventions.specs import *  # noqa: F401,F403
