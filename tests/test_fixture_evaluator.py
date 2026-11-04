"""How the evaluator handles fixtures it cannot, or should not, price."""

import numpy as np
import pandas as pd
import pytest

from corners.data.price_history import price_column, provider_column
from corners.decisions import NO_DATA, NOT_SELECTED, SELECTED, TEAM_TOO_NEW, FixtureEvaluator


@pytest.fixture
def match(trained):
    """A late-season Premier League match (league code E0), so both teams have recent form."""
    return trained.test_season().query("League == 'E0'").iloc[-1].copy()


def test_a_known_team_without_history_is_too_new_not_unmatched(workspace, trained, match, make_price_row):
    everyone = {league: dict(names) for league, names in trained.evaluator.lookup.items()}
    everyone["E0"]["brand new"] = "Brand New"
    evaluator = FixtureEvaluator(
        model=trained.evaluator.model,
        form=trained.evaluator.form,
        lookup=trained.evaluator.lookup,
        broad_lookup=everyone,
    )
    match["HomeTeam"] = "Brand New FC"

    assert evaluator.evaluate(make_price_row(match), "total").status == TEAM_TOO_NEW
    assert not (workspace / "data" / "unmatched_teams.csv").exists()  # nothing for a human to fix


def test_stale_form_means_no_prediction(trained, match, make_price_row):
    match["Date"] = match["Date"] + pd.Timedelta(days=400)
    assert trained.evaluator.evaluate(make_price_row(match), "total").status == NO_DATA


def test_the_total_model_needs_goals_odds_but_the_side_models_do_not(trained, match, make_price_row):
    row = make_price_row(match)
    row[price_column("early", "over", 2.5, "goals")] = np.nan

    assert trained.evaluator.evaluate(row, "total").status == NO_DATA
    assert trained.evaluator.evaluate(row, "home").status in (SELECTED, NOT_SELECTED)


def test_a_market_with_no_prices_is_priced_off_the_model_alone(trained, match, make_price_row):
    row = make_price_row(match)
    row[[column for column in row.index if "home_corners" in column]] = np.nan

    evaluation = trained.evaluator.evaluate(row, "home")
    assert evaluation.status == NOT_SELECTED
    assert evaluation.decision.ev is None
    assert evaluation.blended_mu == evaluation.model_mu


def test_the_size_scales_with_the_capital(trained, match, make_price_row):
    row = make_price_row(match, mu=7.5)
    small = trained.evaluator.evaluate(row, "total", capital=100)
    large = trained.evaluator.evaluate(row, "total", capital=1000)

    assert small.status == large.status == SELECTED
    assert large.decision.size == pytest.approx(10 * small.decision.size, rel=0.03)


def test_the_provider_behind_the_chosen_price_is_reported(trained, match, make_price_row):
    row = make_price_row(match, mu=7.5)
    chosen = trained.evaluator.evaluate(row, "total").decision
    row[provider_column("early", chosen.side.lower(), chosen.line, "corners")] = "BookA"

    assert trained.evaluator.evaluate(row, "total").provider == "BookA"
