import pandas as pd
import pytest

from corners.data import price_history
from corners.data.price_history import (
    PRICE_COLUMNS,
    decision_ladder,
    decision_providers,
    market_over25_prob,
    merge_snapshots,
    price_column,
)


@pytest.fixture
def history_file(tmp_path, monkeypatch):
    path = tmp_path / "odds_history.csv"
    monkeypatch.setattr(price_history, "PRICE_HISTORY_FILE", path)
    return path


def _meta(fixture_id=1):
    return {
        "fixture_id": fixture_id,
        "league": "E0",
        "home_team": "Leeds",
        "away_team": "Fulham",
        "start_datetime": pd.Timestamp("2026-10-10 14:00", tz="UTC"),
    }


def test_closing_snapshot_does_not_erase_early(history_file):
    early = _meta() | {
        "early_fetched_at": pd.Timestamp("2026-10-09", tz="UTC"),
        price_column("early", "over", 9.5, "corners"): 1.95,
    }
    closing = _meta() | {
        "closing_fetched_at": pd.Timestamp("2026-10-11", tz="UTC"),
        price_column("closing", "over", 9.5, "corners"): 1.80,
    }
    merge_snapshots([early])
    history = merge_snapshots([closing])
    assert len(history) == 1
    row = history.iloc[0]
    assert row[price_column("early", "over", 9.5, "corners")] == 1.95
    assert row[price_column("closing", "over", 9.5, "corners")] == 1.80
    assert list(history.columns[: len(PRICE_COLUMNS)]) == PRICE_COLUMNS


def test_decision_ladder_prefers_early_and_falls_back_per_price():
    row = pd.Series(
        {
            price_column("early", "over", 9.5, "corners"): 1.95,
            price_column("closing", "over", 9.5, "corners"): 1.80,
            price_column("closing", "under", 9.5, "corners"): 1.90,
            "early_over_9_5_corners_ft_best_bookmaker": "BookA",
            "closing_under_9_5_corners_ft_best_bookmaker": "BookB",
        }
    )
    assert decision_ladder(row, "corners")[9.5] == (1.95, 1.90)
    assert decision_ladder(row, "corners")[8.5] == (0.0, 0.0)
    assert decision_providers(row, "corners")[9.5] == ("BookA", "BookB")


def test_market_over25_prob():
    row = pd.Series(
        {
            price_column("early", "over", 2.5, "goals"): 2.0,
            price_column("early", "under", 2.5, "goals"): 2.0,
        }
    )
    assert market_over25_prob(row) == pytest.approx(0.5)
    assert market_over25_prob(pd.Series(dtype=float)) is None
