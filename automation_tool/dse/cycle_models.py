import math


def cycle_terms(
    *,
    m: int,
    k: int,
    n: int,
    rho: float,
    k0: int,
    m0: int,
    b_ch: int,
    c_ch: int,
    a_ch: int,
    pes_per_ch: int,
    n0: int,
    delta_load: float,
    t2_used: float | None = None,
) -> dict:
    """
    Compute term1/term2/term3 variants.

    Term2 is computed outside this function (run-length or nnz+delta). If not provided, fall back.
    """
    # Term1 (B movement) with per-tile setup overhead.
    SB = 60
    t1 = (
        ((k0 * n) / (16 * b_ch) + SB * math.ceil(n / n0))
        * math.ceil(k / k0)
        * math.ceil(m / m0)
    )

    if t2_used is None:
        t2_used = (m * k * rho) / (a_ch * pes_per_ch) * (n / n0) * (1.0 + delta_load)

    # Term3 (C movement): packed model (move only the real M rows).
    t3 = (m * n) / (16.0 * c_ch) if (m > 0 and n > 0 and c_ch != 0) else 0.0

    return {
        "t1": t1,
        "t2": float(t2_used),
        "t3": t3,
    }


def total_sum(*, t1: float, t2_used: float, t3: float) -> float:
    """Actual sum model: TOTAL = t1 + t2 + t3."""
    return float(t1 + t2_used + t3)

import math


def cycle_terms(
    *,
    m: int,
    k: int,
    n: int,
    rho: float,
    k0: int,
    m0: int,
    b_ch: int,
    c_ch: int,
    a_ch: int,
    pes_per_ch: int,
    n0: int,
    delta_load: float,
    t2_used: float | None = None,
) -> dict:
    """
    Compute term1/term2_global/term3 variants.

    NOTE: This intentionally preserves the existing scaling behavior from the prior script
    for term3: (rows*n)/(16*C_CH) / n * N0.
    """
    # Term1 (B movement) with per-tile setup overhead:
    #   ((K0*N)/(16*B_CH) + SB*ceil(N/N0)) * ceil(K/K0) * ceil(M/M0)
    # SB is a fixed per-tile setup cost (empirical).
    SB = 60
    t1 = (
        ((k0 * n) / (16 * b_ch) + SB * math.ceil(n / n0))
        * math.ceil(k / k0)
        * math.ceil(m / m0)
    )

    # Term2 is computed outside this function (run-length or nnz+delta). If not provided, fall back.
    if t2_used is None:
        t2_used = (m * k * rho) / (a_ch * pes_per_ch) * (n / n0) * (1.0 + delta_load)

    # Term3 (C movement): packed model (move only the real M rows).
    #   t3 = (M * N) / (16 * C_CH)
    t3 = (m * n) / (16.0 * c_ch) if (m > 0 and n > 0 and c_ch != 0) else 0.0

    return {
        "t1": t1,
        "t2": float(t2_used),
        "t3": t3,
    }


def total_sum(*, t1: float, t2_used: float, t3: float) -> float:
    """Actual sum model: TOTAL = t1 + t2 + t3."""
    return float(t1 + t2_used + t3)


