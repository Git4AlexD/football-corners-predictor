"""Valuing bets at the closing price: the profitability evaluation."""

import numpy as np
import pytest

from corners.evaluation import _verdict, by_ev_band, evaluate_clv, pick_records, summarise
from corners.staking import MARKETS


@pytest.fixture
def priced(trained, make_odds_row, make_history):
    """Fixtures with an early and a closing price, the closing price a little lower."""
    matches = trained.test_season().iloc[::10].head(12)
    rows = [
        make_odds_row(m, fixture_id=300 + i, mu=7.5, closing_mu=7.0)
        for i, (_, m) in enumerate(matches.iterrows())
    ]
    return make_history(rows)


@pytest.fixture
def picks(trained, priced):
    return pick_records(trained.evaluator, priced, trained.results)


def test_every_priced_fixture_gets_a_pick_and_only_some_are_bets(trained, picks):
    threshold = trained.evaluator.config.ev_threshold
    assert set(picks["market"]) == set(MARKETS)
    assert (picks["bet"] == (picks["calc_ev"] >= threshold)).all()
    assert picks["bet"].any()
    assert (picks.loc[~picks["bet"], "stake"] == 0).all()
    assert np.allclose(picks["expected_profit"], picks["stake"] * picks["ev_at_close"])
    assert picks.loc[picks["bet"], "status"].isin(["WIN", "LOSS"]).all()


def test_each_market_gets_a_consistent_verdict(picks):
    summary = summarise(picks)
    assert "Total corners" in set(summary["market"])
    assert set(summary["verdict"]) <= {"profitable", "losing", "not clear yet"}
    assert (summary["expected_roi_low"] <= summary["expected_roi"]).all()
    assert (summary["expected_roi"] <= summary["expected_roi_high"]).all()
    assert (
        summary["bets"] == [picks[(picks["market"] == m) & picks["bet"]].shape[0] for m in _keys(summary)]
    ).all()


def _keys(summary):
    return [key for key, spec in MARKETS.items() if spec.label in set(summary["market"])]


def test_picks_are_grouped_into_ev_bands_without_losing_any(picks):
    table = by_ev_band(picks)
    assert table["picks"].sum() == len(picks)
    assert table["beat_closing_price"].between(0, 1).all()


@pytest.mark.parametrize(
    ("low", "high", "verdict"),
    [(0.1, 0.3, "profitable"), (-0.3, -0.1, "losing"), (-0.1, 0.1, "not clear yet")],
)
def test_the_verdict_depends_on_the_whole_interval(low, high, verdict):
    assert _verdict(low, high) == verdict


def test_the_evaluation_writes_its_tables_and_charts(workspace, trained, priced):
    summary = evaluate_clv(trained.evaluator, priced, trained.results)
    assert not summary.empty
    folder = workspace / "results" / "clv"
    for name in (
        "summary.csv",
        "picks.csv",
        "by_ev_band.csv",
        "expected_vs_actual_roi.png",
        "profit_over_time.png",
    ):
        assert (folder / name).stat().st_size > 0, name


def test_nothing_is_written_without_closing_prices(workspace, trained, make_odds_row, make_history):
    matches = trained.test_season().head(3)
    history = make_history(
        [make_odds_row(m, fixture_id=i, mu=7.5) for i, (_, m) in enumerate(matches.iterrows())]
    )
    assert evaluate_clv(trained.evaluator, history, trained.results).empty
    assert not (workspace / "results" / "clv").exists()
