"""Back-testing and grading the model over the price history."""

import pandas as pd
import pytest

from corners.config import LEAGUE_NAME_TO_CODE
from corners.decisions import MARKETS, NO_DATA, SELECTED, Evaluation
from corners.tracking import (
    ResultBook,
    backtest,
    export_decisions,
    format_summary,
    grade,
    league_summary,
    track,
)

DECISION_STATUSES = ["CORRECT", "INCORRECT", "PENDING"]


@pytest.fixture
def history(trained, make_price_row, make_history):
    matches = trained.test_season().iloc[::12].head(12)
    # An early price well below the model's mean, so the Over looks like value.
    rows = [make_price_row(m, fixture_id=200 + i, mu=7.5) for i, (_, m) in enumerate(matches.iterrows())]
    return make_history(rows)


def test_every_fixture_is_evaluated_in_every_market(trained, history):
    records = backtest(trained.evaluator, history, trained.results)
    assert len(records) == len(history) * len(MARKETS)

    decisions = records[records["status"].isin(DECISION_STATUSES)]
    assert not decisions.empty
    assert (decisions["size"] > 0).all()
    assert (records.drop(decisions.index)["size"] == 0).all()


def test_flat_and_compounding_sizes_differ_only_by_capital(trained, history):
    flat = backtest(trained.evaluator, history, trained.results)
    compounding = backtest(trained.evaluator, history, trained.results, compounding=True)

    assert "capital_at_decision" not in flat
    assert (compounding.groupby("market")["capital_at_decision"].first() == 1000.0).all()
    assert compounding.groupby("market")["capital_at_decision"].nunique().max() > 1  # it moves with results
    # Fixtures kicking off together are all sized off the same capital.
    assert (compounding.groupby(["market", "date"])["capital_at_decision"].nunique() == 1).all()


def evaluation(status):
    return Evaluation(
        fixture_id=1,
        league="E0",
        kickoff=pd.Timestamp("2026-11-03 20:00", tz="UTC"),
        home="A",
        away="B",
        market="total",
        status=status,
    )


def test_an_unpriced_fixture_is_recorded_without_a_size(trained):
    record = grade(evaluation(NO_DATA), trained.results)
    assert (record["status"], record["size"], record["gain"]) == (NO_DATA, 0.0, 0.0)


def test_a_bet_with_no_result_yet_stays_pending(trained, make_price_row):
    match = trained.features.iloc[-1].copy()
    match["Date"] = match["Date"] + pd.Timedelta(days=10)  # a fixture that hasn't been played
    priced = trained.evaluator.evaluate(make_price_row(match, mu=7.5), "total")
    assert priced.status == SELECTED

    record = grade(priced, trained.results)
    assert (record["status"], record["gain"], record["actual"]) == ("PENDING", 0.0, None)


def test_summary_tables_add_up(trained, history):
    records = backtest(trained.evaluator, history, trained.results)
    summary = league_summary(records)
    assert summary["decisions"].sum() == records["status"].isin(DECISION_STATUSES).sum()
    assert (summary["correct"] + summary["incorrect"] + summary["pending"] == summary["decisions"]).all()

    text = format_summary(records, "Back-test")
    assert text.startswith("Back-test")
    assert "Total corners" in text
    assert "All leagues:" in text


def test_a_summary_without_bets_says_so():
    empty = pd.DataFrame({"market": [], "league": [], "status": [], "size": [], "gain": []})
    assert format_summary(empty, "Back-test").endswith("No decisions yet.")


def test_exported_bets_exclude_fixtures_that_were_not_bet(trained, history, tmp_path):
    records = backtest(trained.evaluator, history, trained.results)
    export_decisions(records, tmp_path / "out" / "decisions.csv")
    exported = pd.read_csv(tmp_path / "out" / "decisions.csv")
    assert len(exported) == records["status"].isin(DECISION_STATUSES).sum()
    assert set(exported["status"]) <= set(DECISION_STATUSES)
    assert "fixture_id" not in exported


def test_tracking_writes_tables_and_charts_for_both_capital_modes(workspace, trained, history):
    outputs = track(trained.evaluator, history, trained.results)
    assert set(outputs) == {"flat", "compounding"}
    for mode in outputs:
        folder = workspace / "results" / mode
        assert (folder / "decisions.csv").stat().st_size > 0
        assert (folder / "equity_combined.png").stat().st_size > 0
        assert (folder / "cash_flow.png").stat().st_size > 0
        assert (folder / "returns_by_target.png").stat().st_size > 0


def test_the_result_book_loads_stored_results(workspace, raw_results):
    raw_results.to_csv(workspace / "data" / "raw_results.csv", index=False)
    book = ResultBook.load()
    match = raw_results.iloc[100]
    league = LEAGUE_NAME_TO_CODE[match["League"]]
    assert book.actual(league, match["HomeTeam"], match["AwayTeam"], match["Date"], "HC") == match["HC"]
