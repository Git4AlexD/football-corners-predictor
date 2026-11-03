import numpy as np
import pytest

from corners.config import StakingConfig
from corners.pricing import p_over
from corners.staking import decide_stake, kelly_stake

CONFIG = StakingConfig(
    bankroll=1000,
    ev_threshold=0.05,
    kelly_fraction=0.25,
    max_kelly_ev=0.04,
    max_stake_pct=0.10,
    min_odds=1.0,
    max_odds=3.0,
)


def test_no_stake_below_the_threshold():
    assert kelly_stake(0.04, 2.0, 1000, CONFIG) == (0.0, 0.0)


def test_ev_is_capped_before_sizing():
    # Quarter Kelly with EV capped at 4% at even money: 0.25 * 0.04 / 1.0 = 1%.
    assert kelly_stake(0.30, 2.0, 1000, CONFIG) == (10.0, 1.0)


def test_stake_is_capped_as_a_share_of_bankroll():
    config = StakingConfig(kelly_fraction=1.0, max_kelly_ev=1.0, max_stake_pct=0.10)
    assert kelly_stake(0.5, 1.5, 1000, config) == (100.0, 10.0)


def test_negative_ev_never_produces_a_negative_stake():
    config = StakingConfig(ev_threshold=-np.inf)
    assert kelly_stake(-0.2, 2.0, 1000, config) == (0.0, 0.0)


def _ladder(mu, alpha, lines, margin=1.0):
    return {
        line: (1 / (p_over(mu, alpha, line, 31) * margin), 1 / ((1 - p_over(mu, alpha, line, 31)) * margin))
        for line in lines
    }


def test_decide_stake_bets_the_side_the_model_prefers():
    # Market prices a mean of 9; the model believes 11, so overs are value.
    ladder = _ladder(9.0, 0.05, [8.5, 9.5, 10.5])
    decision = decide_stake(ladder, 11.0, 0.05, [8.5, 9.5, 10.5], 30, 1000, CONFIG)
    assert decision.is_bet
    assert decision.side == "Over"
    assert decision.ev >= CONFIG.ev_threshold
    assert 0 < decision.stake <= 100


def test_decide_stake_reports_the_best_ev_when_not_betting():
    ladder = _ladder(10.0, 0.05, [9.5], margin=1.05)
    decision = decide_stake(ladder, 10.0, 0.05, [9.5], 30, 1000, CONFIG)
    assert not decision.is_bet
    assert decision.ev == pytest.approx(-0.05 / 1.05, abs=0.02)
    assert decision.stake == 0


def test_decide_stake_respects_the_odds_range():
    ladder = {9.5: (3.5, 1.4)}  # over outside the 1.0-3.0 range
    decision = decide_stake(ladder, 14.0, 0.05, [9.5], 30, 1000, CONFIG)
    assert decision.side == "Under"


def test_decide_stake_without_valid_prices():
    decision = decide_stake({9.5: (0.0, 0.0)}, 10.0, 0.05, [9.5], 30, 1000, CONFIG)
    assert not decision.is_bet
    assert decision.ev is None
