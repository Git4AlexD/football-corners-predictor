"""Charts: each is drawn to a file; empty inputs draw nothing."""

import numpy as np
import pandas as pd
import pytest

from corners.plots import plot_cash_flow, plot_equity_curves, plot_profit_over_time, plot_roi_comparison


@pytest.fixture
def records():
    dates = pd.date_range("2026-09-14", periods=12, freq="3D")
    return pd.DataFrame(
        {
            "market": ["total", "home", "away"] * 4,
            "status": ["WIN", "LOSS"] * 6,
            "date": dates,
            "stake": 10.0,
            "profit": [9.0, -10.0] * 6,
            "expected_profit": 0.3,
        }
    )


@pytest.fixture
def with_pending(records):
    pending = pd.DataFrame(
        {
            "market": ["total"],
            "status": ["PENDING"],
            "date": [pd.Timestamp.now() + pd.Timedelta(days=2)],
            "stake": [12.0],
            "profit": [0.0],
            "expected_profit": [0.0],
        }
    )
    return pd.concat([records, pending], ignore_index=True)


def drawn(path):
    return path.exists() and path.stat().st_size > 0


def test_equity_curves_are_drawn_for_each_market_and_combined(records, tmp_path):
    plot_equity_curves(records, tmp_path)
    for name in ("total", "home", "away", "combined"):
        assert drawn(tmp_path / f"equity_{name}.png")


def test_equity_curves_need_settled_bets(records, tmp_path):
    plot_equity_curves(records.assign(status="PENDING"), tmp_path)
    assert list(tmp_path.glob("*.png")) == []


def test_cash_flow_includes_pending_stakes(with_pending, tmp_path):
    plot_cash_flow(with_pending, tmp_path / "cash.png")
    assert drawn(tmp_path / "cash.png")


def test_cash_flow_needs_bets(records, tmp_path):
    plot_cash_flow(records.iloc[0:0], tmp_path / "cash.png")
    assert not (tmp_path / "cash.png").exists()


def test_return_on_stake_chart_handles_a_market_with_no_settled_bets(tmp_path):
    summary = pd.DataFrame(
        {
            "market": ["Total corners", "Home corners"],
            "bets": [40, 3],
            "verdict": ["profitable", "not clear yet"],
            "expected_roi": [0.025, -0.01],
            "expected_roi_low": [0.006, -0.05],
            "expected_roi_high": [0.045, 0.03],
            "actual_roi": [0.02, np.nan],
            "actual_roi_low": [-0.3, np.nan],
            "actual_roi_high": [0.39, np.nan],
        }
    )
    plot_roi_comparison(summary, tmp_path / "roi.png")
    assert drawn(tmp_path / "roi.png")


def test_profit_over_time_has_a_panel_per_market(records, tmp_path):
    plot_profit_over_time(records, tmp_path / "profit.png")
    assert drawn(tmp_path / "profit.png")


def test_profit_over_time_needs_settled_bets(records, tmp_path):
    plot_profit_over_time(records.assign(status="PENDING"), tmp_path / "profit.png")
    assert not (tmp_path / "profit.png").exists()
