"""Capturing early and closing price snapshots into the price history."""

from datetime import date

import pandas as pd

from corners.data.api_football import PlanRestrictedError
from corners.data.price_history import (
    PRICE_COLUMNS,
    PRICE_MARKETS,
    _discover,
    capture_prices,
    load_price_history,
    price_column,
    stage_ladder,
)


class FakePriceClient:
    """Shows a list of fixtures on the first day asked for, then hits the plan's limit."""

    def __init__(self, fixtures, odds_error=None):
        self.fixtures = fixtures
        self.odds_error = odds_error
        self.days_asked = 0
        self.odds_calls = []

    def market_ids(self):
        return {"corners": 1, "home_corners": 2, "away_corners": 3, "goals": 4}

    def fixtures_on(self, day):
        self.days_asked += 1
        if self.days_asked == 1:
            return self.fixtures
        raise PlanRestrictedError("outside the plan's window")

    def best_prices(self, fixture_id, markets):
        self.odds_calls.append(fixture_id)
        if self.odds_error:
            raise self.odds_error
        return {
            market: (
                {line: {"Over": 2.0, "Under": 1.8} for line in lines},
                {line: {"Over": "BookA", "Under": "BookB"} for line in lines},
            )
            for market, (_, lines) in markets.items()
        }


def fixture(status="NS", fixture_id=101):
    return {
        "fixture_id": fixture_id,
        "league": "E0",
        "start_datetime": pd.Timestamp("2026-11-03 20:00", tz="UTC"),
        "home_team": "Leeds",
        "away_team": "Fulham",
        "status": status,
    }


def test_an_upcoming_fixture_gets_an_early_snapshot_only(workspace):
    history = capture_prices(FakePriceClient([fixture("NS")]))

    assert len(history) == 1
    row = history.iloc[0]
    assert pd.notna(row["early_fetched_at"])
    assert pd.isna(row["closing_fetched_at"])
    assert row[price_column("early", "over", 9.5, "corners")] == 2.0
    assert row["early_over_9_5_corners_ft_best_bookmaker"] == "BookA"
    assert pd.isna(row[price_column("closing", "over", 9.5, "corners")])


def test_snapshots_are_locked_once_taken(workspace):
    capture_prices(FakePriceClient([fixture("NS")]))
    again = FakePriceClient([fixture("NS")])
    capture_prices(again)
    assert again.odds_calls == []  # already has an early snapshot, so no API call


def test_a_finished_fixture_gets_a_closing_snapshot_without_losing_the_early_one(workspace):
    capture_prices(FakePriceClient([fixture("NS")]))
    finished = FakePriceClient([fixture("FT")])
    history = capture_prices(finished)

    assert finished.odds_calls == [101]
    assert len(history) == 1
    row = history.iloc[0]
    assert pd.notna(row["early_fetched_at"])
    assert pd.notna(row["closing_fetched_at"])
    assert row[price_column("early", "over", 9.5, "corners")] == 2.0
    assert row[price_column("closing", "under", 9.5, "corners")] == 1.8


def test_a_fixture_first_seen_after_full_time_has_no_early_snapshot(workspace):
    history = capture_prices(FakePriceClient([fixture("FT")]))
    assert pd.isna(history.iloc[0]["early_fetched_at"])
    assert pd.notna(history.iloc[0]["closing_fetched_at"])


def test_an_api_error_stops_fetching_without_losing_the_run(workspace):
    client = FakePriceClient([fixture("NS", 101), fixture("NS", 102)], odds_error=RuntimeError("quota gone"))
    history = capture_prices(client)
    assert client.odds_calls == [101]  # stopped at the first failure
    assert history.empty
    assert not (workspace / "data" / "odds_history.csv").exists()


def test_discovery_stops_on_an_api_error_but_keeps_earlier_days():
    class Flaky:
        calls = 0

        def fixtures_on(self, day):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("daily quota exhausted")
            return [fixture()]

    found = _discover(Flaky(), range(0, 5), date(2026, 11, 3))
    assert len(found) == 1


def test_history_loads_with_the_full_schema_even_when_empty(workspace):
    history = load_price_history()
    assert history.empty
    assert list(history.columns) == PRICE_COLUMNS


def test_a_stage_ladder_reads_one_snapshot(workspace):
    history = capture_prices(FakePriceClient([fixture("NS")]))
    ladder = stage_ladder(history.iloc[0], "corners", "early")
    assert ladder[9.5] == (2.0, 1.8)
    assert set(ladder) == set(PRICE_MARKETS["corners"])
    assert stage_ladder(history.iloc[0], "corners", "closing")[9.5] == (0.0, 0.0)
