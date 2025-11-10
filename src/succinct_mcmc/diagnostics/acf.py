"""
Autocorrelation utilities.

Purpose:
- Provide a simple ACF implementation that higher-level diagnostics (ESS, etc.)
  can build on.
- Initial implementation uses dense arrays; later we can:
    - implement block-wise algorithms that work directly on succinct chains.

For now this is a straightforward, well-documented helper for small-to-medium N.
"""

from typing import List


def autocorrelation(x: List[float]) -> List[float]:
    """
    Compute autocorrelation rho_k for k = 0..N-1 for a sequence x.

    Naive O(N^2) implementation for clarity.
    For production, replace with FFT-based method when N is large.

    Returns:
        List of length N where:
            rho[0] = 1.0
            rho[k] = Cov(x_t, x_{t+k}) / Var(x)
    """
    N = len(x)
    if N == 0:
        return []
    if N == 1:
        return [1.0]

    mean = sum(x) / N
    var_num = sum((xi - mean) ** 2 for xi in x)
    if var_num == 0:
        return [1.0] + [0.0] * (N - 1)
    var = var_num / N

    rhos = []
    for k in range(N):
        cov = 0.0
        count = N - k
        for t in range(count):
            cov += (x[t] - mean) * (x[t + k] - mean)
        cov /= N
        rhos.append(cov / var)
    return rhos
