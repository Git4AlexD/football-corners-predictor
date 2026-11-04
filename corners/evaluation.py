"""Do the forecasts beat the market? Evaluated against closing prices.

The realised outcome of a few hundred decisions is mostly luck: whether each one
lands is close to a coin flip, so the realised return has a margin of error of
roughly +/-15%. A far more precise measure is what each decision was *worth*.
The closing price, just before kickoff, is the market's most accurate estimate of
the true probability, so each decision taken at the early price is valued at it:

    expected gain = size x (early price x fair closing probability - 1)

Summed over decisions, this is the expected gain, and divided by the amount
committed, the expected return on that amount (ROI). Forecasts that consistently
find better prices than the fair closing price carry real information, whatever
their realised results so far.

For every fixture with both an early and a closing price snapshot, the decision
engine's choice is replicated on the early price, valued at the close, and settled
against the actual result. Outputs go to results/evaluation/.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from corners.config import RESULTS_DIR
from corners.data.price_history import stage_ladder
from corners.decisions import MARKETS, FixtureEvaluator
from corners.plots import plot_cumulative_return, plot_roi_comparison
from corners.pricing import fair_over_prob
from corners.tracking import ResultBook, out_of_sample, settle

log = logging.getLogger(__name__)

EV_BANDS = [-np.inf, 0.0, 0.02, 0.05, 0.10, np.inf]
EV_BAND_LABELS = ["below 0%", "0 to 2%", "2 to 5%", "5 to 10%", "10% or more"]
BOOTSTRAP_SAMPLES = 2000


def _has_prices(ladder: dict) -> bool:
    return any(over > 1 and under > 1 for over, under in ladder.values())


def pick_records(evaluator: FixtureEvaluator, history: pd.DataFrame, results: ResultBook) -> pd.DataFrame:
    """The engine's pick for every fixture and market with early and closing price.

    Every fixture is included, decision or not (``selected`` marks the ones that clear the
    EV threshold), so the selections can be compared with the picks that were passed up.
    """
    rows = []
    threshold = evaluator.config.ev_threshold
    for _, row in out_of_sample(history, evaluator.model.train_end_date).iterrows():
        for market, spec in MARKETS.items():
            early = stage_ladder(row, spec.price_key, "early")
            closing = stage_ladder(row, spec.price_key, "closing")
            if not (_has_prices(early) and _has_prices(closing)):
                continue
            evaluation = evaluator.evaluate(row, market, ladder=early, ev_threshold=-np.inf)
            d = evaluation.decision
            if d is None or d.side is None:
                continue
            p_close_over = fair_over_prob(*closing[d.line])
            if np.isnan(p_close_over):
                continue

            over = d.side == "Over"
            p_close = p_close_over if over else 1 - p_close_over
            selected = d.ev >= threshold
            size = d.size if selected else 0.0
            actual = results.actual(
                evaluation.league,
                evaluation.home_team,
                evaluation.away_team,
                evaluation.fixture_date,
                spec.actual_column,
            )
            status, gain = settle(d.side, d.line, d.price, size, actual)
            rows.append(
                {
                    "date": evaluation.fixture_date.normalize(),
                    "market": market,
                    "league": evaluation.league,
                    "home": evaluation.home,
                    "away": evaluation.away,
                    "selection": f"{d.side} {d.line}",
                    "early_price": d.price,
                    "closing_price": closing[d.line][0 if over else 1],
                    "calc_ev": d.ev,
                    "ev_at_close": d.price * p_close - 1,
                    "selected": selected,
                    "size": size,
                    "expected_gain": size * (d.price * p_close - 1),
                    "status": status,
                    "actual": actual,
                    "gain": gain,
                }
            )
    return pd.DataFrame(rows)


def _bootstrap_ratio(numerator: np.ndarray, denominator: np.ndarray) -> tuple[float, float]:
    """95% bootstrap interval for sum(numerator) / sum(denominator) over decisions."""
    if len(numerator) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(numerator), size=(BOOTSTRAP_SAMPLES, len(numerator)))
    ratios = numerator[idx].sum(axis=1) / denominator[idx].sum(axis=1)
    low, high = np.percentile(ratios, [2.5, 97.5])
    return float(low), float(high)


def _verdict(low: float, high: float) -> str:
    if low > 0:
        return "ahead of market"
    if high < 0:
        return "behind market"
    return "not clear yet"


def summarise(records: pd.DataFrame) -> pd.DataFrame:
    """Expected and actual return on size for each market's decisions."""
    rows = []
    for market, spec in MARKETS.items():
        decisions = records[(records["market"] == market) & records["selected"]]
        if decisions.empty:
            continue
        size = decisions["size"].to_numpy()
        exp_low, exp_high = _bootstrap_ratio(decisions["expected_gain"].to_numpy(), size)
        settled = decisions[decisions["status"].isin(["CORRECT", "INCORRECT"])]
        act_low, act_high = _bootstrap_ratio(settled["gain"].to_numpy(), settled["size"].to_numpy())
        rows.append(
            {
                "market": spec.label,
                "decisions": len(decisions),
                "committed": size.sum(),
                "expected_gain": decisions["expected_gain"].sum(),
                "expected_roi": decisions["expected_gain"].sum() / size.sum(),
                "expected_roi_low": exp_low,
                "expected_roi_high": exp_high,
                "beat_closing_price": (decisions["ev_at_close"] > 0).mean(),
                "verdict": _verdict(exp_low, exp_high),
                "settled": len(settled),
                "realised_gain": settled["gain"].sum(),
                "realised_roi": settled["gain"].sum() / settled["size"].sum() if len(settled) else np.nan,
                "realised_roi_low": act_low,
                "realised_roi_high": act_high,
            }
        )
    return pd.DataFrame(rows)


def by_ev_band(records: pd.DataFrame) -> pd.DataFrame:
    """Does a higher calculated EV mean a better price? Every pick, decision or not."""
    df = records.assign(band=pd.cut(records["calc_ev"], EV_BANDS, labels=EV_BAND_LABELS, right=False))
    table = df.groupby(["market", "band"], observed=True).agg(
        picks=("calc_ev", "size"),
        mean_calc_ev=("calc_ev", "mean"),
        mean_ev_at_close=("ev_at_close", "mean"),
        beat_closing_price=("ev_at_close", lambda s: (s > 0).mean()),
    )
    return table.loc[[m for m in MARKETS if m in table.index.get_level_values(0)]]


def _format_summary(summary: pd.DataFrame) -> str:
    def pct(value: float) -> str:
        return "n/a" if pd.isna(value) else f"{value:+.1%}"

    lines = [
        "Do the forecasts beat the market? Decisions at the early price, valued at the fair closing price."
    ]
    for row in summary.itertuples():
        lines += [
            f"  {row.market}: {row.decisions} decisions, £{row.committed:,.2f} committed",
            f"    expected ROI at closing prices {pct(row.expected_roi)} "
            f"(95% CI {pct(row.expected_roi_low)} to {pct(row.expected_roi_high)}): {row.verdict.upper()}",
            f"    beat the fair closing price on {row.beat_closing_price:.0%} of decisions",
            f"    actual ROI from {row.settled} settled decisions {pct(row.realised_roi)} "
            f"(95% CI {pct(row.realised_roi_low)} to {pct(row.realised_roi_high)})",
        ]
    return "\n".join(lines)


def evaluate_closing_prices(
    evaluator: FixtureEvaluator, history: pd.DataFrame, results: ResultBook
) -> pd.DataFrame:
    """Run the evaluation, save its tables and charts, and log the summary."""
    records = pick_records(evaluator, history, results)
    if records.empty or not records["selected"].any():
        log.info("No decisions with both early and closing prices yet")
        return pd.DataFrame()

    summary = summarise(records)
    out_dir = RESULTS_DIR / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "summary.csv", index=False)
    records.to_csv(out_dir / "selections.csv", index=False)
    by_ev_band(records).to_csv(out_dir / "by_ev_band.csv")
    plot_roi_comparison(summary, out_dir / "expected_vs_realised_return.png")
    plot_cumulative_return(records[records["selected"]], out_dir / "cumulative_return.png")

    log.info("\n%s", _format_summary(summary))
    return summary
