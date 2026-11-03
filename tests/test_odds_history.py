import pandas as pd
import pytest

from corners.data import odds_history
from corners.data.odds_history import (
    ODDS_COLUMNS,
    betting_bookmakers,
    betting_ladder,
    market_over25_prob,
    merge_snapshots,
    odds_column,
)


@pytest.fixture
def history_file(tmp_path, monkeypatch):
    path = tmp_path / "odds_history.csv"
    monkeypatch.setattr(odds_history, "ODDS_HISTORY_FILE", path)
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
        odds_column("early", "over", 9.5, "corners"): 1.95,
    }
    closing = _meta() | {
        "closing_fetched_at": pd.Timestamp("2026-10-11", tz="UTC"),
        odds_column("closing", "over", 9.5, "corners"): 1.80,
    }
    merge_snapshots([early])
    history = merge_snapshots([closing])
    assert len(history) == 1
    row = history.iloc[0]
    assert row[odds_column("early", "over", 9.5, "corners")] == 1.95
    assert row[odds_column("closing", "over", 9.5, "corners")] == 1.80
    assert list(history.columns[: len(ODDS_COLUMNS)]) == ODDS_COLUMNS


def test_betting_ladder_prefers_early_and_falls_back_per_price():
    row = pd.Series(
        {
            odds_column("early", "over", 9.5, "corners"): 1.95,
            odds_column("closing", "over", 9.5, "corners"): 1.80,
            odds_column("closing", "under", 9.5, "corners"): 1.90,
            "early_over_9_5_corners_ft_best_bookmaker": "BookA",
            "closing_under_9_5_corners_ft_best_bookmaker": "BookB",
        }
    )
    assert betting_ladder(row, "corners")[9.5] == (1.95, 1.90)
    assert betting_ladder(row, "corners")[8.5] == (0.0, 0.0)
    assert betting_bookmakers(row, "corners")[9.5] == ("BookA", "BookB")


def test_market_over25_prob():
    row = pd.Series(
        {
            odds_column("early", "over", 2.5, "goals"): 2.0,
            odds_column("early", "under", 2.5, "goals"): 2.0,
        }
    )
    assert market_over25_prob(row) == pytest.approx(0.5)
    assert market_over25_prob(pd.Series(dtype=float)) is None
