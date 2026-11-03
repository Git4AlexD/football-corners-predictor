"""Results -> features -> trained model -> staking decision -> graded bet."""

import numpy as np
import pandas as pd
import pytest

from corners.data.odds_history import ODDS_COLUMNS, odds_column
from corners.features import FormBook, _team_rows, build_match_features, build_team_form, clean_results
from corners.model import MODEL_SPECS, TrainedModel, complete_rows, fit_models
from corners.pricing import p_over
from corners.staking import BET, MARKETS, NO_BET, NO_TEAM_MATCH, FixtureEvaluator
from corners.teams import build_team_lookup
from corners.tracking import ResultBook, grade


@pytest.fixture(scope="module")
def pipeline(raw_results):
    matches = clean_results(raw_results)
    rows = _team_rows(matches)
    features = build_match_features(matches, rows)
    params = fit_models(complete_rows(features))
    evaluator = FixtureEvaluator(
        model=TrainedModel.from_params(params),
        form=FormBook(build_team_form(rows)),
        lookup=build_team_lookup(features),
    )
    return matches, features, params, evaluator


def _odds_row(match, fixture_id=1, mu=9.5, alpha=0.05):
    """An odds-history row priced from an NB distribution with a 5% margin."""
    row = {column: np.nan for column in ODDS_COLUMNS} | {
        "fixture_id": fixture_id,
        "league": match["League"],
        "start_datetime": pd.Timestamp(match["Date"]).tz_localize("UTC") + pd.Timedelta(hours=15),
        "home_team": match["HomeTeam"],
        "away_team": match["AwayTeam"],
        "early_fetched_at": pd.Timestamp("2000-01-01", tz="UTC"),
    }
    for spec, mean in ((MARKETS["total"], mu), (MARKETS["home"], mu * 0.55), (MARKETS["away"], mu * 0.45)):
        for line in spec.lines:
            p = p_over(mean, alpha, line)
            row[odds_column("early", "over", line, spec.odds_key)] = 1 / (p * 1.05)
            row[odds_column("early", "under", line, spec.odds_key)] = 1 / ((1 - p) * 1.05)
    row[odds_column("early", "over", 2.5, "goals")] = 1.9
    row[odds_column("early", "under", 2.5, "goals")] = 1.9
    return pd.Series(row)


def test_models_are_trained_for_every_market(pipeline):
    _, _, params, _ = pipeline
    assert set(params["models"]) == set(MODEL_SPECS)
    for model in params["models"].values():
        assert model["alpha"] > 0
        assert "Intercept" in model["coefficients"]
    assert params["test_seasons"] == ["2425"]
    assert params["train_seasons"] == ["2122", "2223", "2324"]


def test_a_test_season_fixture_is_priced_and_graded(pipeline):
    matches, features, _, evaluator = pipeline
    match = features[features["Season"] == "2425"].iloc[-1]
    results = ResultBook(matches)
    for market in MARKETS:
        evaluation = evaluator.evaluate(_odds_row(match), market, ev_threshold=-np.inf)
        assert evaluation.status in (BET, NO_BET)
        assert evaluation.blended_mu > 0
        record = grade(evaluation, results)
        actual = match[MARKETS[market].actual_column]
        assert record["actual"] == actual
        won = actual > record["line"] if record["side"] == "Over" else actual < record["line"]
        assert record["status"] == ("WIN" if won else "LOSS")


def test_an_unknown_team_is_reported_not_priced(pipeline, tmp_path, monkeypatch):
    from corners import teams

    monkeypatch.setattr(teams, "UNMATCHED_TEAMS_FILE", tmp_path / "unmatched.csv")
    _, features, _, evaluator = pipeline
    match = features.iloc[-1].copy()
    match["HomeTeam"] = "Nonexistent Wanderers"
    evaluation = evaluator.evaluate(_odds_row(match), "total")
    assert evaluation.status == NO_TEAM_MATCH
    assert (tmp_path / "unmatched.csv").exists()
