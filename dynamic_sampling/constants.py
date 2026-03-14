"""
TLDR: Fixed constants for dynamic_sampling (MiniF2F size, default budget).

Used by config and runner to initialize dataset size and budget when not overridden.
"""

# MiniF2F test split size (cat-searcher/minif2f-lean4, split=test).
MINIF2F_TEST_SIZE: int = 244

# Default max total attempts across all problems when budget not specified.
DEFAULT_BUDGET: int = 256
