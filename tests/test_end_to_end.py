"""Results -> features -> trained model -> decision decision -> graded decision."""

import numpy as np

from corners.decisions import MARKETS, NO_TEAM_MATCH, NOT_SELECTED, SELECTED
from corners.model import MODEL_SPECS
from corners.tracking import grade


def test_models_are_trained_for_every_market(trained):
    assert set(trained.params["models"]) == set(MODEL_SPECS)
    for model in trained.params["models"].values():
        assert model["alpha"] > 0
        assert "Intercept" in model["coefficients"]
    assert trained.params["test_seasons"] == ["2425"]
    assert trained.params["train_seasons"] == ["2122", "2223", "2324"]


def test_a_test_season_fixture_is_priced_and_graded(trained, make_price_row):
    match = trained.test_season().iloc[-1]
    for market in MARKETS:
        evaluation = trained.evaluator.evaluate(make_price_row(match), market, ev_threshold=-np.inf)
        assert evaluation.status in (SELECTED, NOT_SELECTED)
        assert evaluation.blended_mu > 0
        record = grade(evaluation, trained.results)
        actual = match[MARKETS[market].actual_column]
        assert record["actual"] == actual
        won = actual > record["line"] if record["side"] == "Over" else actual < record["line"]
        assert record["status"] == ("CORRECT" if won else "INCORRECT")


def test_an_unknown_team_is_reported_not_priced(trained, workspace, make_price_row):
    match = trained.features.iloc[-1].copy()
    match["HomeTeam"] = "Nonexistent Wanderers"
    evaluation = trained.evaluator.evaluate(make_price_row(match), "total")
    assert evaluation.status == NO_TEAM_MATCH
    assert (workspace / "data" / "unmatched_teams.csv").exists()
