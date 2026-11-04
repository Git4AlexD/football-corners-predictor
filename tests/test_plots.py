"""Charts: each is drawn to a file; empty inputs draw nothing."""

import numpy as np
import pandas as pd
import pytest

from corners.plots import (
    plot_cash_flow,
    plot_cumulative_return,
    plot_equity_curves,
    plot_returns_by_target,
    plot_roi_comparison,
)


@pytest.fixture
def records():
    dates = pd.date_range("2026-09-14", periods=12, freq="3D")
    return pd.DataFrame(
        {
            "market": ["total", "home", "away"] * 4,
            "status": ["CORRECT", "INCORRECT"] * 6,
            "date": dates,
            "size": 10.0,
            "gain": [9.0, -10.0] * 6,
            "expected_gain": 0.3,
        }
    )


@pytest.fixture
def with_pending(records):
    pending = pd.DataFrame(
        {
            "market": ["total"],
            "status": ["PENDING"],
            "date": [pd.Timestamp.now() + pd.Timedelta(days=2)],
            "size": [12.0],
            "gain": [0.0],
            "expected_gain": [0.0],
        }
    )
    return pd.concat([records, pending], ignore_index=True)


def drawn(path):
    return path.exists() and path.stat().st_size > 0


def test_equity_curves_are_drawn_for_each_market_and_combined(records, tmp_path):
    plot_equity_curves(records, tmp_path)
    for name in ("total", "home", "away", "combined"):
        assert drawn(tmp_path / f"equity_{name}.png")


def test_equity_curves_need_settled_decisions(records, tmp_path):
    plot_equity_curves(records.assign(status="PENDING"), tmp_path)
    assert list(tmp_path.glob("*.png")) == []


def test_cash_flow_includes_pending_sizes(with_pending, tmp_path):
    plot_cash_flow(with_pending, tmp_path / "cash.png")
    assert drawn(tmp_path / "cash.png")


def test_cash_flow_needs_decisions(records, tmp_path):
    plot_cash_flow(records.iloc[0:0], tmp_path / "cash.png")
    assert not (tmp_path / "cash.png").exists()


def test_return_chart_handles_a_market_with_no_settled_decisions(tmp_path):
    summary = pd.DataFrame(
        {
            "market": ["Total corners", "Home corners"],
            "decisions": [40, 3],
            "verdict": ["ahead of market", "not clear yet"],
            "expected_roi": [0.025, -0.01],
            "expected_roi_low": [0.006, -0.05],
            "expected_roi_high": [0.045, 0.03],
            "realised_roi": [0.02, np.nan],
            "realised_roi_low": [-0.3, np.nan],
            "realised_roi_high": [0.39, np.nan],
        }
    )
    plot_roi_comparison(summary, tmp_path / "roi.png")
    assert drawn(tmp_path / "roi.png")


def test_cumulative_return_has_a_panel_per_market(records, tmp_path):
    plot_cumulative_return(records, tmp_path / "gain.png")
    assert drawn(tmp_path / "gain.png")


def test_cumulative_return_needs_settled_decisions(records, tmp_path):
    plot_cumulative_return(records.assign(status="PENDING"), tmp_path / "gain.png")
    assert not (tmp_path / "gain.png").exists()


def test_returns_by_target_draws_each_target_and_the_combined_line(records, tmp_path):
    plot_returns_by_target(records, tmp_path / "returns.png")
    assert drawn(tmp_path / "returns.png")


def test_returns_by_target_copes_with_one_target_and_little_committed(records, tmp_path):
    only_total = records[records["market"] == "total"].head(2)  # £20 committed: no percentage line yet
    plot_returns_by_target(only_total, tmp_path / "returns.png")
    assert drawn(tmp_path / "returns.png")


def test_returns_by_target_needs_settled_decisions(records, tmp_path):
    plot_returns_by_target(records.assign(status="PENDING"), tmp_path / "returns.png")
    assert not (tmp_path / "returns.png").exists()
