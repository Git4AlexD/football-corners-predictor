"""The append-only log of decisions as they were made."""

import pandas as pd
import pytest

from corners.decision_log import COLUMNS, grade_decision_log, load_decision_log, log_upcoming_decisions
from corners.decisions import MARKETS


@pytest.fixture
def history(trained, make_price_row, make_history):
    matches = trained.test_season().iloc[::25].head(5)
    # An early price far below the model's mean, so the Over looks like value.
    return make_history(
        [make_price_row(m, fixture_id=100 + i, mu=7.5) for i, (_, m) in enumerate(matches.iterrows())]
    )


def before_all(history):
    return history["start_datetime"].min() - pd.Timedelta(days=1)


def test_upcoming_fixtures_are_logged_once(workspace, trained, history):
    first = log_upcoming_decisions(trained.evaluator, history, now=before_all(history))
    logged = load_decision_log()

    assert first == len(logged) > 0
    assert set(logged["market"]) <= set(MARKETS)
    assert set(logged["status"]) <= {"SELECTED", "NOT SELECTED"}
    assert (logged["model_fingerprint"] == trained.evaluator.model.fingerprint).all()
    assert "SELECTED" in set(logged["status"])

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
    assert set(graded["status"]) <= {"CORRECT", "INCORRECT"}  # the synthetic matches have been played
    wins, losses = graded[graded["status"] == "CORRECT"], graded[graded["status"] == "INCORRECT"]
    assert (wins["gain"] == wins["size"] * (wins["price"] - 1)).all()
    assert (losses["gain"] == -losses["size"]).all()
    assert (workspace / "results" / "logged_decisions.csv").exists()


def test_grading_an_empty_log_is_a_no_op(workspace, trained):
    assert grade_decision_log(trained.results).empty
    assert not (workspace / "results" / "logged_decisions.csv").exists()


def test_a_log_written_with_the_old_column_names_is_converted(workspace):
    legacy = pd.DataFrame(
        {
            "logged_at": ["2026-11-02T10:36:01+00:00"] * 2,
            "fixture_id": [1, 2],
            "league": ["E0", "E0"],
            "kickoff": ["2026-11-02 16:00:00+00:00"] * 2,
            "home": ["A", "C"],
            "away": ["B", "D"],
            "home_team": ["A", "C"],
            "away_team": ["B", "D"],
            "market": ["total", "total"],
            "status": ["BET", "NO BET"],
            "side": ["Over", "Under"],
            "line": [9.5, 9.5],
            "odds": [2.1, 1.9],
            "bookmaker": ["BookA", None],
            "ev": [0.07, 0.01],
            "stake": [6.5, 0.0],
            "model_fingerprint": ["abc", "abc"],
        }
    )
    legacy.to_csv(workspace / "data" / "decision_log.csv", index=False)

    converted = load_decision_log()
    assert list(converted.columns) == COLUMNS
    assert converted["status"].tolist() == ["SELECTED", "NOT SELECTED"]
    assert converted["price"].tolist() == [2.1, 1.9]
    assert converted["size"].tolist() == [6.5, 0.0]
    assert converted["provider"].iloc[0] == "BookA"

    # Reading it again, now that nothing needs converting, changes nothing.
    assert load_decision_log().equals(converted)
