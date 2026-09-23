"""
stats_utils.py — Pure NumPy/stdlib statistical functions
=========================================================
Drop-in replacements for scipy.stats used in benchmark.py.
No scipy dependency — works on any Python 3.8+ with NumPy.
"""

import math
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
#  Low-level special functions  (math stdlib only)
# ══════════════════════════════════════════════════════════════════════════════

def _norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erfc."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _betainc(a: float, b: float, x: float) -> float:
    """
    Regularized incomplete beta function  I_x(a, b).
    Uses Lentz continued-fraction method (Numerical Recipes §6.4).
    """
    if x <= 0.0: return 0.0
    if x >= 1.0: return 1.0
    # Symmetry: use the side that converges faster
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _betainc(b, a, 1.0 - x)
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - lbeta) / a
    TINY  = 1e-30
    f, C, D = TINY, TINY, 0.0
    for m in range(200):
        for s in (0, 1):
            if m == 0 and s == 0:
                d = 1.0
            elif s == 0:
                d = m * (b - m) * x / ((a + 2*m - 1) * (a + 2*m))
            else:
                d = -(a + m) * (a + b + m) * x / ((a + 2*m) * (a + 2*m + 1))
            D = 1.0 / max(1.0 + d * D, TINY)
            C = max(1.0 + d / C, TINY)
            delta = C * D
            f *= delta
            if abs(delta - 1.0) < 1e-12:
                return front * f
    return front * f


def _t_cdf(t: float, df: float) -> float:
    """CDF of Student's t-distribution."""
    x = df / (df + t * t)
    p = _betainc(df / 2.0, 0.5, x) / 2.0
    return p if t < 0 else 1.0 - p


def _gammainc_series(a: float, x: float) -> float:
    """Regularized lower incomplete gamma (series, for x < a+1)."""
    ap, d, total = a, 1.0 / a, 1.0 / a
    for _ in range(500):
        ap   += 1.0
        d    *= x / ap
        total += d
        if abs(d) < abs(total) * 1e-12:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gammainc_cf(a: float, x: float) -> float:
    """Regularized upper incomplete gamma (continued fraction, for x >= a+1)."""
    TINY = 1e-30
    f, C, D = TINY, TINY, 0.0
    b = x + 1.0 - a
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        D  = 1.0 / max(b + an * D, TINY)
        C  = max(b + an / C, TINY)
        f *= C * D
        if abs(C * D - 1.0) < 1e-12:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * f


def _gammainc(a: float, x: float) -> float:
    """Regularized lower incomplete gamma  P(a, x)."""
    if x <= 0.0:
        return 0.0
    return _gammainc_series(a, x) if x < a + 1.0 else 1.0 - _gammainc_cf(a, x)


def _chi2_sf(x: float, df: int) -> float:
    """Survival function of chi-squared distribution  (1 - CDF)."""
    return 1.0 - _gammainc(df / 2.0, x / 2.0)


# ══════════════════════════════════════════════════════════════════════════════
#  Public API — mirrors scipy.stats signatures
# ══════════════════════════════════════════════════════════════════════════════

def t_ppf(p: float, df: float) -> float:
    """
    Percent-point function (quantile) of Student's t-distribution.
    Equivalent to scipy.stats.t.ppf(p, df).
    Uses bisection on the CDF.
    """
    if df >= 1e6:
        # Normal approximation
        return math.sqrt(2.0) * math.erfinv(2.0 * p - 1.0)
    lo, hi = 0.0, 100.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if _t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def shapiro(x: np.ndarray) -> tuple[float, float]:
    """
    D'Agostino–Pearson K² omnibus normality test.
    Returns (statistic, p_value).  p > 0.05 → cannot reject normality.
    Equivalent to scipy.stats.shapiro for the purpose of reporting.
    """
    a  = np.asarray(x, dtype=np.float64)
    n  = len(a)
    if n < 8:
        return float("nan"), float("nan")
    m  = a.mean()
    s  = a.std(ddof=1)
    if s == 0.0:
        return 0.0, 1.0
    g1 = float(np.mean(((a - m) / s) ** 3))   # skewness
    g2 = float(np.mean(((a - m) / s) ** 4)) - 3.0  # excess kurtosis

    # ── Skewness Z-score ──────────────────────────────────────────
    Y      = g1 * math.sqrt((n + 1) * (n + 3) / (6.0 * (n - 2)))
    beta2  = (3.0 * (n**2 + 27*n - 70) * (n + 1) * (n + 3)
              / ((n - 2) * (n + 5) * (n + 7) * (n + 9)))
    W2     = -1.0 + math.sqrt(2.0 * (beta2 - 1.0))
    delta  = 1.0 / math.sqrt(math.log(math.sqrt(W2)))
    alpha  = math.sqrt(2.0 / (W2 - 1.0))
    arg    = Y / alpha + math.sqrt((Y / alpha) ** 2 + 1.0)
    Z1     = delta * math.log(arg) if alpha > 0 else 0.0

    # ── Kurtosis Z-score ──────────────────────────────────────────
    E_b2   = 3.0 * (n - 1.0) / (n + 1.0)
    var_b2 = (24.0 * n * (n - 2) * (n - 3)
              / ((n + 1) ** 2 * (n + 3) * (n + 5)))
    x_k    = (g2 - E_b2) / math.sqrt(var_b2) if var_b2 > 0 else 0.0
    beta1  = (6.0 * (n**2 - 5*n + 2) / ((n + 7) * (n + 9))
              * math.sqrt(6.0 * (n + 3) * (n + 5) / (n * (n - 2) * (n - 3))))
    A      = 6.0 + (8.0 / beta1) * (2.0 / beta1 + math.sqrt(1.0 + 4.0 / beta1**2)) \
             if beta1 > 0 else 6.0
    inner  = ((1.0 - 2.0 / A) / (1.0 + x_k * math.sqrt(2.0 / (A - 4.0)))) \
             if A > 4.0 else 1.0
    Z2     = ((1.0 - 2.0 / (9.0 * A) - math.copysign(abs(inner) ** (1.0/3), inner))
              / math.sqrt(2.0 / (9.0 * A))) if A > 0 else 0.0

    K2 = Z1 ** 2 + Z2 ** 2          # chi-squared, 2 df
    p  = _chi2_sf(K2, df=2)
    return float(K2), float(p)


def skew(a: np.ndarray) -> float:
    """
    Sample skewness (biased, matching scipy.stats.skew default).
    """
    a = np.asarray(a, dtype=np.float64)
    m = a.mean()
    s = a.std()           # ddof=0 (biased)
    return float(np.mean(((a - m) / s) ** 3)) if s > 0 else 0.0


def kurtosis(a: np.ndarray, fisher: bool = True) -> float:
    """
    Sample kurtosis.
    fisher=True → excess kurtosis (normal = 0), matching scipy default.
    """
    a  = np.asarray(a, dtype=np.float64)
    m  = a.mean()
    s  = a.std()           # ddof=0
    k4 = float(np.mean(((a - m) / s) ** 4)) if s > 0 else 0.0
    return k4 - 3.0 if fisher else k4


def ttest_ind(a: np.ndarray, b: np.ndarray,
              equal_var: bool = False) -> tuple[float, float]:
    """
    Two-sample t-test.  equal_var=False → Welch's t-test.
    Returns (t_statistic, p_value, two-tailed).
    Equivalent to scipy.stats.ttest_ind(a, b, equal_var=equal_var).
    """
    a, b   = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = len(a), len(b)
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    if equal_var:
        sp  = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
        se  = sp * math.sqrt(1.0/na + 1.0/nb)
        df  = float(na + nb - 2)
    else:
        se_sq = va / na + vb / nb
        se    = math.sqrt(se_sq) if se_sq > 0 else 0.0
        df    = se_sq ** 2 / ((va/na)**2 / (na - 1) + (vb/nb)**2 / (nb - 1)) \
                if se_sq > 0 else 1.0
    t_stat = (ma - mb) / se if se > 0 else 0.0
    p      = 2.0 * (1.0 - _t_cdf(abs(t_stat), df))
    return float(t_stat), float(p)


def mannwhitneyu(a: np.ndarray, b: np.ndarray,
                 alternative: str = "two-sided") -> tuple[float, float]:
    """
    Mann–Whitney U test with normal approximation (valid for n > ~20).
    Returns (U_statistic, p_value).
    Equivalent to scipy.stats.mannwhitneyu(a, b, alternative=alternative).
    """
    a, b   = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = len(a), len(b)
    combined = np.concatenate([a, b])

    # Ranks with tie averaging
    order  = np.argsort(combined, kind="stable")
    ranks  = np.empty(len(combined))
    ranks[order] = np.arange(1, len(combined) + 1, dtype=float)
    for val in np.unique(combined):
        idx = np.where(combined == val)[0]
        if len(idx) > 1:
            ranks[idx] = ranks[idx].mean()

    R1 = ranks[:na].sum()
    U1 = R1 - na * (na + 1) / 2.0
    U2 = float(na * nb) - U1
    U  = min(U1, U2)

    # Tie-corrected normal approximation
    mu    = na * nb / 2.0
    # Tie correction factor
    vals, counts = np.unique(combined, return_counts=True)
    tie_corr = np.sum(counts ** 3 - counts) / 12.0
    sigma_sq = (na * nb / (len(combined) * (len(combined) - 1))) * \
               (len(combined) ** 3 - len(combined) - 12.0 * tie_corr) / 12.0
    sigma = math.sqrt(max(sigma_sq, 1e-30))

    z = (U - mu) / sigma
    if alternative == "two-sided":
        p = 2.0 * _norm_cdf(-abs(z))
    elif alternative == "less":
        p = _norm_cdf(z)
    else:
        p = 1.0 - _norm_cdf(z)

    return float(U), float(p)

