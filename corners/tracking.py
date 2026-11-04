"""Back-test the current model over the whole price history and grade it.

Every fixture in the price history is re-evaluated with the current model and
graded against the actual result where one exists. Two capital modes are
reported: *flat* (every size sized off the same capital) and
*compounding* (each market's capital grows and shrinks with its own settled
results). Fixtures on or before the model's last training date are left out,
so a back-test never grades a match the model was trained on.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from corners.config import DECISION_CONFIG, RAW_RESULTS_FILE, RESULT_DATE_TOLERANCE_DAYS, RESULTS_DIR
from corners.decisions import MARKETS, SELECTED, Evaluation, FixtureEvaluator
from corners.features import clean_results
from corners.plots import plot_cash_flow, plot_equity_curves, plot_returns_by_target

log = logging.getLogger(__name__)

SETTLED = ("CORRECT", "INCORRECT")
DECISION_STATUSES = ("CORRECT", "INCORRECT", "PENDING")


class ResultBook:
    """Actual corner counts per match, looked up by teams and approximate date."""

    def __init__(self, results: pd.DataFrame) -> None:
        df = results.assign(total_corners=results["HC"] + results["AC"])
        self._matches: dict[tuple[str, str, str], pd.DataFrame] = {
            key: group.sort_values("Date")
            for key, group in df.groupby(["League", "HomeTeam", "AwayTeam"], sort=False)
        }

    @classmethod
    def load(cls) -> ResultBook:
        raw = pd.read_csv(RAW_RESULTS_FILE, parse_dates=["Date"], low_memory=False)
        return cls(clean_results(raw))

    def actual(self, league: str, home: str, away: str, when: pd.Timestamp, column: str) -> float | None:
        """The result of the meeting closest to ``when``, if within the tolerance.

        The same teams meet every season, so the date check stops a fixture
        being graded against a different season's meeting.
        """
        matches = self._matches.get((league, home, away))
        if matches is None:
            return None
        gaps = (matches["Date"] - when).abs()
        nearest = gaps.idxmin()
        if gaps[nearest] > pd.Timedelta(days=RESULT_DATE_TOLERANCE_DAYS):
            return None
        return float(matches.loc[nearest, column])


def settle(side: str, line: float, price: float, size: float, actual: float | None) -> tuple[str, float]:
    """``(status, gain)`` for a decision given the actual count (None = not played yet)."""
    if actual is None:
        return "PENDING", 0.0
    correct = actual > line if side == "Over" else actual < line
    return ("CORRECT", size * (price - 1.0)) if correct else ("INCORRECT", -size)


def grade(evaluation: Evaluation, results: ResultBook) -> dict[str, Any]:
    """Flatten an evaluation into a record, settling it if it was a decision."""
    d = evaluation.decision
    record = {
        "market": evaluation.market,
        "fixture_id": evaluation.fixture_id,
        "league": evaluation.league,
        "date": evaluation.fixture_date,
        "home": evaluation.home,
        "away": evaluation.away,
        "status": evaluation.status,
        "side": d.side if d else None,
        "line": d.line if d else None,
        "price": d.price if d else None,
        "provider": evaluation.provider,
        "ev": d.ev if d else None,
        "size": d.size if d else 0.0,
        "actual": None,
        "gain": 0.0,
    }
    if evaluation.status != SELECTED:
        return record

    actual = results.actual(
        evaluation.league,
        evaluation.home_team,
        evaluation.away_team,
        evaluation.fixture_date,
        MARKETS[evaluation.market].actual_column,
    )
    status, gain = settle(d.side, d.line, d.price, d.size, actual)
    record.update(status=status, actual=actual, gain=gain)
    return record


def out_of_sample(history: pd.DataFrame, train_end_date: pd.Timestamp) -> pd.DataFrame:
    """Fixtures after the model's training data ends."""
    keep = history["start_datetime"].dt.tz_convert(None).dt.normalize() > train_end_date
    if (~keep).any():
        log.info("Excluding %d fixture(s) that fall inside the training period", (~keep).sum())
    return history[keep]


def backtest(
    evaluator: FixtureEvaluator,
    history: pd.DataFrame,
    results: ResultBook,
    compounding: bool = False,
) -> pd.DataFrame:
    """Evaluate and grade every out-of-sample fixture in every market.

    In compounding mode, fixtures sharing a kickoff time are all sized off
    the same capital, which only moves once their results are in.
    """
    history = out_of_sample(history, evaluator.model.train_end_date)
    records = []
    for market in MARKETS:
        capital = evaluator.config.capital
        for _, kickoff_group in history.groupby("start_datetime", sort=True):
            sizing_capital = max(capital, 0.0)
            for _, row in kickoff_group.iterrows():
                evaluation = evaluator.evaluate(row, market, capital=sizing_capital if compounding else None)
                record = grade(evaluation, results)
                if compounding:
                    record["capital_at_decision"] = round(sizing_capital, 2)
                    if record["status"] in SETTLED:
                        capital += record["gain"]
                records.append(record)
        log.info("Back-tested %s over %d fixtures", MARKETS[market].label.lower(), len(history))
    return pd.DataFrame(records)


def league_summary(records: pd.DataFrame) -> pd.DataFrame:
    """Decisions, record, size and return per market and league."""
    decisions = records[records["status"].isin(DECISION_STATUSES)]
    rows = []
    for (market, league), group in decisions.groupby(["market", "league"]):
        settled = group[group["status"].isin(SETTLED)]
        committed = settled["size"].sum()
        rows.append(
            {
                "market": market,
                "league": league,
                "decisions": len(group),
                "correct": int((settled["status"] == "CORRECT").sum()),
                "incorrect": int((settled["status"] == "INCORRECT").sum()),
                "pending": int((group["status"] == "PENDING").sum()),
                "settled_committed": round(committed, 2),
                "gain": round(settled["gain"].sum(), 2),
                "roi": settled["gain"].sum() / committed if committed > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def format_summary(records: pd.DataFrame, title: str) -> str:
    """A per-market, per-league results table as text."""
    summary = league_summary(records)
    lines = [title]
    if summary.empty:
        return title + "\n    No decisions yet."
    for market, spec in MARKETS.items():
        group = summary[summary["market"] == market]
        if group.empty:
            continue
        lines.append(f"  {spec.label}")
        table = group.drop(columns="market").copy()
        table["roi"] = table["roi"].map(lambda v: "n/a" if pd.isna(v) else f"{v:+.1%}")
        lines += ["    " + line for line in table.to_string(index=False).splitlines()]
        committed, gain = group["settled_committed"].sum(), group["gain"].sum()
        roi = f"{gain / committed:+.1%}" if committed else "n/a"
        lines.append(
            f"    All leagues: {group['decisions'].sum()} decisions, "
            f"{group['correct'].sum()} correct, {group['incorrect'].sum()} incorrect, "
            f"{group['pending'].sum()} pending, settled committed {committed:.2f}, "
            f"gain {gain:+.2f} ({roi})"
        )
    return "\n".join(lines)


def export_decisions(records: pd.DataFrame, path: Path) -> None:
    """Write placed decisions (settled or pending) to CSV."""
    decisions = records[records["status"].isin(DECISION_STATUSES)].copy()
    decisions["date"] = pd.to_datetime(decisions["date"]).dt.strftime("%Y-%m-%d")
    decisions = decisions.sort_values(["market", "date", "home"])
    path.parent.mkdir(parents=True, exist_ok=True)
    decisions.drop(columns=[c for c in ("fixture_id",) if c in decisions]).to_csv(path, index=False)


def track(evaluator: FixtureEvaluator, history: pd.DataFrame, results: ResultBook) -> dict[str, pd.DataFrame]:
    """Run both capital modes, write their tables and charts, and log summaries."""
    outputs = {}
    for mode, compounding in (("flat", False), ("compounding", True)):
        records = backtest(evaluator, history, results, compounding=compounding)
        out_dir = RESULTS_DIR / mode
        export_decisions(records, out_dir / "decisions.csv")
        plot_equity_curves(records, out_dir)
        plot_cash_flow(records, out_dir / "cash_flow.png")
        plot_returns_by_target(records, out_dir / "returns_by_target.png")
        description = (
            f"compounding: each market starts at {DECISION_CONFIG.capital:,.0f} and keeps its returns"
            if compounding
            else f"flat: every size sized off {DECISION_CONFIG.capital:,.0f}"
        )
        log.info("\n%s", format_summary(records, f"Back-test results ({description})"))
        outputs[mode] = records
    return outputs
