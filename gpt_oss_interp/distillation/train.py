"""Student training loop.

Stage order (per CASCADE_DISTILLATION.md):
    1. regression warmup on x_e*
    2. KL refinement against teacher distribution
"""
