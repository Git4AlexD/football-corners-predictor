import numpy as np
import pandas as pd
import pytest

from corners.evaluation import _bootstrap_ratio, summarise


def _decisions(market, values, sizes, gains):
    return pd.DataFrame(
        {
            "market": market,
            "selected": True,
            "size": sizes,
            "ev_at_close": values,
            "expected_gain": np.array(sizes) * np.array(values),
            "status": ["CORRECT" if p > 0 else "INCORRECT" for p in gains],
            "gain": gains,
        }
    )


def test_bootstrap_interval_contains_the_estimate():
    num, den = np.array([1.0, 2.0, -1.0, 3.0]), np.array([10.0, 10.0, 10.0, 10.0])
    low, high = _bootstrap_ratio(num, den)
    assert low <= num.sum() / den.sum() <= high


def test_summary_measures_expected_and_actual_return():
    records = pd.concat(
        [
            _decisions("total", [0.03, 0.05, 0.04, 0.02] * 5, [10.0] * 20, [9.0, -10.0] * 10),
            _decisions("home", [-0.04, -0.03, -0.05, -0.02] * 5, [10.0] * 20, [-10.0, 9.0] * 10),
        ]
    )
    summary = summarise(records).set_index("market")

    total = summary.loc["Total corners"]
    assert total["expected_roi"] == pytest.approx(0.035)
    assert total["verdict"] == "ahead of market"
    assert total["realised_roi"] == pytest.approx(-0.05)
    assert total["beat_closing_price"] == 1.0
    assert summary.loc["Home corners", "verdict"] == "behind market"
