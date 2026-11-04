"""Negative-binomial pricing and provider-price handling.

Corner counts are overdispersed, so both the model and the provider's
over/under ladder are represented as negative-binomial (NB) distributions
parameterised by a mean ``mu`` and dispersion ``alpha``
(variance = mu + alpha * mu^2).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.optimize import minimize
from scipy.stats import nbinom

from corners.config import MARKET_ALPHA_PRIOR, MARKET_MAX_ALPHA, PMF_MAX_K

# scipy.stats validates and broadcasts on every call, which dominated runtime
# when fitting thousands of ladders. The compiled kernel underneath gives
# identical values for valid arguments; fall back to the public API if a
# future scipy moves it.
try:
    from scipy.special._ufuncs import _nbinom_pmf as _nbinom_pmf_kernel
except ImportError:  # pragma: no cover
    _nbinom_pmf_kernel = nbinom.pmf

PriceLadder = Mapping[float, tuple[float, float]]
"""``{line: (over_price, under_price)}``; a missing price is 0.0."""


def nb_pmf(mu: float, alpha: float, k_max: int = PMF_MAX_K) -> np.ndarray:
    """NB probabilities for counts ``0 .. k_max - 1``, renormalised to sum to 1."""
    mu = max(float(mu), 1e-6)
    alpha = max(float(alpha), 1e-6)
    var = mu + alpha * mu**2
    if var <= mu:
        var = mu * 1.0001
    r = mu**2 / (var - mu)
    p = mu / var
    pmf = _nbinom_pmf_kernel(np.arange(k_max), r, p)
    total = pmf.sum()
    if total <= 0 or not np.isfinite(total):
        raise ValueError(f"Invalid NB PMF for mu={mu}, alpha={alpha}")
    return pmf / total


def p_over(mu: float, alpha: float, line: float, k_max: int = PMF_MAX_K) -> float:
    """P(count > line)."""
    pmf = nb_pmf(mu, alpha, k_max)
    return float(pmf[np.arange(k_max) > line].sum())


def devig(over_price: float, under_price: float) -> tuple[float, float]:
    """Proportional de-vig: returns ``(fair P(over), overround)``."""
    inv_over, inv_under = 1.0 / over_price, 1.0 / under_price
    overround = inv_over + inv_under
    return inv_over / overround, overround


def fair_over_prob(over_price: float, under_price: float) -> float:
    """Fair P(over) from a two-way price, or NaN if either price is missing."""
    if not (over_price > 1 and under_price > 1):
        return float("nan")
    return devig(over_price, under_price)[0]


def is_valid_price_pair(over_price: float, under_price: float) -> bool:
    """Both prices present and the overround plausible (0.95 to 1.25)."""
    if not (np.isfinite(over_price) and np.isfinite(under_price)):
        return False
    if over_price <= 1.0 or under_price <= 1.0:
        return False
    return 0.95 <= 1.0 / over_price + 1.0 / under_price <= 1.25


def ladder_points(ladder: PriceLadder) -> list[tuple[float, float]]:
    """Every usable ``(line, fair P(over))`` pair in a ladder."""
    return [
        (line, devig(over, under)[0])
        for line, (over, under) in sorted(ladder.items())
        if is_valid_price_pair(over, under)
    ]


def _mu_for_single_line(line: float, p_target: float, alpha: float) -> float:
    """Bisect for the NB mean that gives P(over line) = p_target."""
    lo, hi = 3.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if p_over(mid, alpha, line) < p_target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def fit_market_ladder(
    ladder: PriceLadder, alpha_prior: float = MARKET_ALPHA_PRIOR
) -> tuple[float | None, float | None]:
    """Fit an NB distribution to a provider's whole over/under ladder.

    Returns the market-implied ``(mu, alpha)``, or ``(None, None)`` if the
    ladder has no usable prices. With fewer than three prices the dispersion
    is poorly identified, so it is held at ``alpha_prior``.
    """
    points = ladder_points(ladder)
    if not points:
        return None, None
    if len(points) < 3:
        mu = float(np.mean([_mu_for_single_line(line, p, alpha_prior) for line, p in points]))
        return mu, alpha_prior

    lines = np.array([line for line, _ in points])
    targets = np.array([p for _, p in points])

    def squared_error(theta: np.ndarray) -> float:
        mu, alpha = theta
        if mu <= 0 or alpha <= 0:
            return 1e6
        predicted = np.array([p_over(mu, alpha, line) for line in lines])
        return float(((predicted - targets) ** 2).sum())

    best = min(
        (
            minimize(
                squared_error,
                [mu0, alpha_prior],
                method="Nelder-Mead",
                options={"xatol": 1e-4, "fatol": 1e-9, "maxiter": 600},
            )
            for mu0 in (8.0, 10.0, 12.0)
        ),
        key=lambda result: result.fun,
    )
    mu = float(max(best.x[0], 1e-6))
    alpha = float(np.clip(best.x[1], 1e-4, MARKET_MAX_ALPHA))
    return mu, alpha
