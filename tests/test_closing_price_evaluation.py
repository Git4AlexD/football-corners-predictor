"""Valuing decisions at the closing price: the evaluation against closing prices."""

import numpy as np
import pytest

from corners.decisions import MARKETS
from corners.evaluation import _verdict, by_ev_band, evaluate_closing_prices, pick_records, summarise


@pytest.fixture
def priced(trained, make_price_row, make_history):
    """Fixtures with an early and a closing price, the closing price a little lower."""
    matches = trained.test_season().iloc[::10].head(12)
    rows = [
        make_price_row(m, fixture_id=300 + i, mu=7.5, closing_mu=7.0)
        for i, (_, m) in enumerate(matches.iterrows())
    ]
    return make_history(rows)


@pytest.fixture
def picks(trained, priced):
    return pick_records(trained.evaluator, priced, trained.results)


def test_every_priced_fixture_gets_a_pick_and_only_some_are_decisions(trained, picks):
    threshold = trained.evaluator.config.ev_threshold
    assert set(picks["market"]) == set(MARKETS)
    assert (picks["selected"] == (picks["calc_ev"] >= threshold)).all()
    assert picks["selected"].any()
    assert (picks.loc[~picks["selected"], "size"] == 0).all()
    assert np.allclose(picks["expected_gain"], picks["size"] * picks["ev_at_close"])
    assert picks.loc[picks["selected"], "status"].isin(["CORRECT", "INCORRECT"]).all()


def test_each_market_gets_a_consistent_verdict(picks):
    summary = summarise(picks)
    assert "Total corners" in set(summary["market"])
    assert set(summary["verdict"]) <= {"ahead of market", "behind market", "not clear yet"}
    assert (summary["expected_roi_low"] <= summary["expected_roi"]).all()
    assert (summary["expected_roi"] <= summary["expected_roi_high"]).all()
    assert (
        summary["decisions"]
        == [picks[(picks["market"] == m) & picks["selected"]].shape[0] for m in _keys(summary)]
    ).all()


def _keys(summary):
    return [key for key, spec in MARKETS.items() if spec.label in set(summary["market"])]


def test_picks_are_grouped_into_ev_bands_without_losing_any(picks):
    table = by_ev_band(picks)
    assert table["picks"].sum() == len(picks)
    assert table["beat_closing_price"].between(0, 1).all()


@pytest.mark.parametrize(
    ("low", "high", "verdict"),
    [(0.1, 0.3, "ahead of market"), (-0.3, -0.1, "behind market"), (-0.1, 0.1, "not clear yet")],
)
def test_the_verdict_depends_on_the_whole_interval(low, high, verdict):
    assert _verdict(low, high) == verdict


def test_the_evaluation_writes_its_tables_and_charts(workspace, trained, priced):
    summary = evaluate_closing_prices(trained.evaluator, priced, trained.results)
    assert not summary.empty
    folder = workspace / "results" / "evaluation"
    for name in (
        "summary.csv",
        "selections.csv",
        "by_ev_band.csv",
        "expected_vs_realised_return.png",
        "cumulative_return.png",
    ):
        assert (folder / name).stat().st_size > 0, name


def test_nothing_is_written_without_closing_prices(workspace, trained, make_price_row, make_history):
    matches = trained.test_season().head(3)
    history = make_history(
        [make_price_row(m, fixture_id=i, mu=7.5) for i, (_, m) in enumerate(matches.iterrows())]
    )
    assert evaluate_closing_prices(trained.evaluator, history, trained.results).empty
    assert not (workspace / "results" / "evaluation").exists()
