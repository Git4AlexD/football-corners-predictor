import numpy as np
import pytest

from corners.pricing import (
    devig,
    fair_over_prob,
    fit_market_ladder,
    is_valid_price_pair,
    nb_pmf,
    p_over,
)


def test_nb_pmf_is_a_distribution_with_the_right_mean():
    pmf = nb_pmf(10.0, 0.05, k_max=80)
    assert pmf.sum() == pytest.approx(1.0)
    assert (pmf * np.arange(80)).sum() == pytest.approx(10.0, rel=1e-3)


def test_p_over_falls_with_line_and_rises_with_mean():
    assert p_over(10, 0.05, 8.5) > p_over(10, 0.05, 10.5)
    assert p_over(11, 0.05, 9.5) > p_over(9, 0.05, 9.5)


def test_devig_removes_the_margin_proportionally():
    p, overround = devig(1.90, 1.90)
    assert p == pytest.approx(0.5)
    assert overround == pytest.approx(2 / 1.9)


def test_fair_over_prob_is_nan_when_a_price_is_missing():
    assert np.isnan(fair_over_prob(1.9, 0.0))


@pytest.mark.parametrize(
    ("over", "under", "valid"),
    [(1.9, 1.9, True), (0.0, 1.9, False), (1.2, 1.2, False), (np.nan, 1.9, False)],
)
def test_is_valid_price_pair(over, under, valid):
    assert is_valid_price_pair(over, under) is valid


def _fair_ladder(mu, alpha, lines):
    ladder = {}
    for line in lines:
        p = p_over(mu, alpha, line)
        ladder[line] = (1 / p, 1 / (1 - p))
    return ladder


def test_fit_market_ladder_recovers_known_parameters():
    mu, alpha = fit_market_ladder(_fair_ladder(10.2, 0.04, [7.5, 8.5, 9.5, 10.5, 11.5]))
    assert mu == pytest.approx(10.2, abs=0.05)
    assert alpha == pytest.approx(0.04, abs=0.01)


def test_fit_market_ladder_with_two_lines_keeps_the_alpha_prior():
    mu, alpha = fit_market_ladder(_fair_ladder(9.0, 0.1, [8.5, 9.5]), alpha_prior=0.1)
    assert mu == pytest.approx(9.0, abs=0.05)
    assert alpha == 0.1


def test_fit_market_ladder_without_prices():
    assert fit_market_ladder({8.5: (0.0, 0.0)}) == (None, None)


def test_an_impossible_distribution_is_an_error_not_a_silent_nan():
    with pytest.raises(ValueError, match="Invalid NB PMF"):
        nb_pmf(float("nan"), 0.1)
