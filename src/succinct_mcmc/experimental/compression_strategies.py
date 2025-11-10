"""
Experimental: alternative compression / block size strategies.

Purpose:
- Explore choices beyond b = ceil(sqrt(T)):
    - memory-budget aware strategies,
    - target random-access latency vs memory usage,
    - adaptive based on observed kernel cost.

These functions are not wired into core yet; they're for experimentation and
paper-style evaluations.
"""

from math import ceil


def sqrt_strategy(num_steps: int) -> int:
    """
    Classic √T strategy:
    - Minimizes b + T/b, giving balanced time-space trade-off.
    """
    return ceil(num_steps ** 0.5)


def memory_bound_strategy(num_steps: int, memory_budget_units: int) -> int:
    """
    Choose a block size to fit within a given "anchor memory" budget.

    Args:
        num_steps: T
        memory_budget_units: number of anchors (or anchor-sized units) we can store.

    Roughly:
        K = T / b <= memory_budget_units  =>  b >= T / memory_budget_units
    We pick:
        b = max(1, ceil(T / memory_budget_units))
    """
    if memory_budget_units <= 0:
        raise ValueError("memory_budget_units must be positive")
    return max(1, ceil(num_steps / memory_budget_units))
