"""Is the strategy profitable? Measured with closing-line value (CLV).

Profit from a few hundred bets is mostly luck: whether a bet wins is close
to a coin flip, so actual ROI has a margin of error of roughly +/-15%. A far
more precise measure is what each bet was *worth*. The closing price, just
before kickoff, is the market's most accurate estimate of the true
probability, so each bet placed at early odds is valued at it:

    expected profit = stake x (early odds x fair closing probability - 1)

Summed over bets, this is the strategy's expected profit, and divided by
the amount staked, its expected return on stake (ROI). A strategy that
consistently gets better prices than the fair closing price makes money in
the long run, whatever its results so far.

For every fixture with both an early and a closing odds snapshot, the
staking engine's decision is replicated on the early odds, valued at the
close, and settled against the actual result. Outputs go to results/clv/.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from corners.config import RESULTS_DIR
from corners.data.odds_history import stage_ladder
from corners.plots import plot_profit_over_time, plot_roi_comparison
from corners.pricing import fair_over_prob
from corners.staking import MARKETS, FixtureEvaluator
from corners.tracking import ResultBook, out_of_sample, settle

log = logging.getLogger(__name__)

EV_BANDS = [-np.inf, 0.0, 0.02, 0.05, 0.10, np.inf]
EV_BAND_LABELS = ["below 0%", "0 to 2%", "2 to 5%", "5 to 10%", "10% or more"]
BOOTSTRAP_SAMPLES = 2000


def _has_prices(ladder: dict) -> bool:
    return any(over > 1 and under > 1 for over, under in ladder.values())


def pick_records(evaluator: FixtureEvaluator, history: pd.DataFrame, results: ResultBook) -> pd.DataFrame:
    """The engine's pick for every fixture and market with early and closing odds.

    Every fixture is included, bet or not (``bet`` marks the ones that clear the
    EV threshold), so the bets can be compared with the picks that were passed up.
    """
    rows = []
    threshold = evaluator.config.ev_threshold
    for _, row in out_of_sample(history, evaluator.model.train_end_date).iterrows():
        for market, spec in MARKETS.items():
            early = stage_ladder(row, spec.odds_key, "early")
            closing = stage_ladder(row, spec.odds_key, "closing")
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
            is_bet = d.ev >= threshold
            stake = d.stake if is_bet else 0.0
            actual = results.actual(
                evaluation.league,
                evaluation.home_team,
                evaluation.away_team,
                evaluation.fixture_date,
                spec.actual_column,
            )
            status, profit = settle(d.side, d.line, d.odds, stake, actual)
            rows.append(
                {
                    "date": evaluation.fixture_date.normalize(),
                    "market": market,
                    "league": evaluation.league,
                    "home": evaluation.home,
                    "away": evaluation.away,
                    "selection": f"{d.side} {d.line}",
                    "early_odds": d.odds,
                    "closing_odds": closing[d.line][0 if over else 1],
                    "calc_ev": d.ev,
                    "ev_at_close": d.odds * p_close - 1,
                    "bet": is_bet,
                    "stake": stake,
                    "expected_profit": stake * (d.odds * p_close - 1),
                    "status": status,
                    "actual": actual,
                    "profit": profit,
                }
            )
    return pd.DataFrame(rows)


def _bootstrap_ratio(numerator: np.ndarray, denominator: np.ndarray) -> tuple[float, float]:
    """95% bootstrap interval for sum(numerator) / sum(denominator) over bets."""
    if len(numerator) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(numerator), size=(BOOTSTRAP_SAMPLES, len(numerator)))
    ratios = numerator[idx].sum(axis=1) / denominator[idx].sum(axis=1)
    low, high = np.percentile(ratios, [2.5, 97.5])
    return float(low), float(high)


def _verdict(low: float, high: float) -> str:
    if low > 0:
        return "profitable"
    if high < 0:
        return "losing"
    return "not clear yet"


def summarise(records: pd.DataFrame) -> pd.DataFrame:
    """Expected and actual return on stake for each market's bets."""
    rows = []
    for market, spec in MARKETS.items():
        bets = records[(records["market"] == market) & records["bet"]]
        if bets.empty:
            continue
        stake = bets["stake"].to_numpy()
        exp_low, exp_high = _bootstrap_ratio(bets["expected_profit"].to_numpy(), stake)
        settled = bets[bets["status"].isin(["WIN", "LOSS"])]
        act_low, act_high = _bootstrap_ratio(settled["profit"].to_numpy(), settled["stake"].to_numpy())
        rows.append(
            {
                "market": spec.label,
                "bets": len(bets),
                "staked": stake.sum(),
                "expected_profit": bets["expected_profit"].sum(),
                "expected_roi": bets["expected_profit"].sum() / stake.sum(),
                "expected_roi_low": exp_low,
                "expected_roi_high": exp_high,
                "beat_closing_price": (bets["ev_at_close"] > 0).mean(),
                "verdict": _verdict(exp_low, exp_high),
                "settled": len(settled),
                "actual_profit": settled["profit"].sum(),
                "actual_roi": settled["profit"].sum() / settled["stake"].sum() if len(settled) else np.nan,
                "actual_roi_low": act_low,
                "actual_roi_high": act_high,
            }
        )
    return pd.DataFrame(rows)


def by_ev_band(records: pd.DataFrame) -> pd.DataFrame:
    """Does a higher calculated EV mean a better price? Every pick, bet or not."""
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

    lines = ["Is each market profitable? Bets placed at early odds, valued at the fair closing price."]
    for row in summary.itertuples():
        lines += [
            f"  {row.market}: {row.bets} bets, £{row.staked:,.2f} staked",
            f"    expected ROI at closing prices {pct(row.expected_roi)} "
            f"(95% CI {pct(row.expected_roi_low)} to {pct(row.expected_roi_high)}): {row.verdict.upper()}",
            f"    beat the fair closing price on {row.beat_closing_price:.0%} of bets",
            f"    actual ROI from {row.settled} settled bets {pct(row.actual_roi)} "
            f"(95% CI {pct(row.actual_roi_low)} to {pct(row.actual_roi_high)})",
        ]
    return "\n".join(lines)


def evaluate_clv(evaluator: FixtureEvaluator, history: pd.DataFrame, results: ResultBook) -> pd.DataFrame:
    """Run the evaluation, save its tables and charts, and log the summary."""
    records = pick_records(evaluator, history, results)
    if records.empty or not records["bet"].any():
        log.info("No bets with both early and closing odds yet")
        return pd.DataFrame()

    summary = summarise(records)
    out_dir = RESULTS_DIR / "clv"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "summary.csv", index=False)
    records.to_csv(out_dir / "picks.csv", index=False)
    by_ev_band(records).to_csv(out_dir / "by_ev_band.csv")
    plot_roi_comparison(summary, out_dir / "expected_vs_actual_roi.png")
    plot_profit_over_time(records[records["bet"]], out_dir / "profit_over_time.png")

    log.info("\n%s", _format_summary(summary))
    return summary
