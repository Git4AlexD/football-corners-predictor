"""The append-only log of decisions as they were made."""

import pandas as pd
import pytest

from corners.decision_log import grade_decision_log, load_decision_log, log_upcoming_decisions
from corners.staking import MARKETS


@pytest.fixture
def history(trained, make_odds_row, make_history):
    matches = trained.test_season().iloc[::25].head(5)
    # An early price far below the model's mean, so the Over looks like value.
    return make_history(
        [make_odds_row(m, fixture_id=100 + i, mu=7.5) for i, (_, m) in enumerate(matches.iterrows())]
    )


def before_all(history):
    return history["start_datetime"].min() - pd.Timedelta(days=1)


def test_upcoming_fixtures_are_logged_once(workspace, trained, history):
    first = log_upcoming_decisions(trained.evaluator, history, now=before_all(history))
    logged = load_decision_log()

    assert first == len(logged) > 0
    assert set(logged["market"]) <= set(MARKETS)
    assert set(logged["status"]) <= {"BET", "NO BET"}
    assert (logged["model_fingerprint"] == trained.evaluator.model.fingerprint).all()
    assert "BET" in set(logged["status"])

    assert log_upcoming_decisions(trained.evaluator, history, now=before_all(history)) == 0
    assert len(load_decision_log()) == len(logged)


def test_fixtures_already_under_way_are_not_logged(workspace, trained, history):
    after_all = history["start_datetime"].max() + pd.Timedelta(days=1)
    assert log_upcoming_decisions(trained.evaluator, history, now=after_all) == 0
    assert load_decision_log().empty
    assert not (workspace / "data" / "decision_log.csv").exists()


def test_fixtures_without_an_early_snapshot_are_not_logged(workspace, trained, history):
    history["early_fetched_at"] = pd.NaT
    assert log_upcoming_decisions(trained.evaluator, history, now=before_all(history)) == 0


def test_logged_bets_are_graded_against_results(workspace, trained, history):
    log_upcoming_decisions(trained.evaluator, history, now=before_all(history))
    graded = grade_decision_log(trained.results)

    assert not graded.empty
    assert set(graded["status"]) <= {"WIN", "LOSS"}  # the synthetic matches have been played
    wins, losses = graded[graded["status"] == "WIN"], graded[graded["status"] == "LOSS"]
    assert (wins["profit"] == wins["stake"] * (wins["odds"] - 1)).all()
    assert (losses["profit"] == -losses["stake"]).all()
    assert (workspace / "results" / "logged_bets.csv").exists()


def test_grading_an_empty_log_is_a_no_op(workspace, trained):
    assert grade_decision_log(trained.results).empty
    assert not (workspace / "results" / "logged_bets.csv").exists()
