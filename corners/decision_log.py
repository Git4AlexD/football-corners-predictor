"""An append-only record of decisions as they were made.

The back-test in ``tracking`` re-evaluates history with today's model, which
answers "how would the current model have done". This log answers "what did
the pipeline actually recommend at the time": each upcoming fixture's
decision is written once, before kickoff, and never changed.
"""

from __future__ import annotations

import logging

import pandas as pd

from corners.config import DECISION_LOG_FILE, RESULTS_DIR
from corners.staking import BET, MARKETS, NO_BET, FixtureEvaluator
from corners.tracking import ResultBook, format_summary, settle

log = logging.getLogger(__name__)

COLUMNS = [
    "logged_at",
    "fixture_id",
    "league",
    "kickoff",
    "home",
    "away",
    "home_team",
    "away_team",
    "market",
    "status",
    "side",
    "line",
    "odds",
    "bookmaker",
    "ev",
    "stake",
    "model_fingerprint",
]


def load_decision_log() -> pd.DataFrame:
    if not DECISION_LOG_FILE.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(DECISION_LOG_FILE)
    df["kickoff"] = pd.to_datetime(df["kickoff"], utc=True)
    return df


def log_upcoming_decisions(
    evaluator: FixtureEvaluator, history: pd.DataFrame, now: pd.Timestamp | None = None
) -> int:
    """Record a decision for each upcoming fixture and market not yet logged.

    Only fixtures with an early odds snapshot are logged, and only once they
    can be priced, so a fixture missing form or goals odds is retried later.
    Returns the number of new entries.
    """
    now = now or pd.Timestamp.now(tz="UTC")
    existing = load_decision_log()
    logged = set(zip(existing["fixture_id"], existing["market"], strict=True))
    upcoming = history[(history["start_datetime"] > now) & history["early_fetched_at"].notna()]

    new = []
    for _, row in upcoming.iterrows():
        for market in MARKETS:
            if (row["fixture_id"], market) in logged:
                continue
            evaluation = evaluator.evaluate(row, market)
            if evaluation.status not in (BET, NO_BET):
                continue
            d = evaluation.decision
            new.append(
                {
                    "logged_at": now.isoformat(timespec="seconds"),
                    "fixture_id": evaluation.fixture_id,
                    "league": evaluation.league,
                    "kickoff": evaluation.kickoff,
                    "home": evaluation.home,
                    "away": evaluation.away,
                    "home_team": evaluation.home_team,
                    "away_team": evaluation.away_team,
                    "market": market,
                    "status": evaluation.status,
                    "side": d.side,
                    "line": d.line,
                    "odds": d.odds,
                    "bookmaker": evaluation.bookmaker,
                    "ev": d.ev,
                    "stake": d.stake,
                    "model_fingerprint": evaluator.model.fingerprint,
                }
            )
    if new:
        combined = (
            pd.concat([existing, pd.DataFrame(new)], ignore_index=True)
            if len(existing)
            else pd.DataFrame(new)
        )
        DECISION_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        combined[COLUMNS].to_csv(DECISION_LOG_FILE, index=False)
    bets = sum(entry["status"] == BET for entry in new)
    log.info("Decision log: %d new entr%s (%d bets)", len(new), "y" if len(new) == 1 else "ies", bets)
    return len(new)


def grade_decision_log(results: ResultBook) -> pd.DataFrame:
    """Settle logged bets against actual results and save the graded log."""
    df = load_decision_log()
    bets = df[df["status"] == BET].copy()
    if bets.empty:
        log.info("Decision log: no logged bets yet")
        return bets

    statuses, actuals, profits = [], [], []
    for _, bet in bets.iterrows():
        actual = results.actual(
            bet["league"],
            bet["home_team"],
            bet["away_team"],
            bet["kickoff"].tz_convert(None),
            MARKETS[bet["market"]].actual_column,
        )
        status, profit = settle(bet["side"], bet["line"], bet["odds"], bet["stake"], actual)
        statuses.append(status)
        actuals.append(actual)
        profits.append(profit)
    bets = bets.assign(
        status=statuses, actual=actuals, profit=profits, date=bets["kickoff"].dt.tz_convert(None)
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    bets.drop(columns=["date"]).to_csv(RESULTS_DIR / "logged_bets.csv", index=False)
    log.info("\n%s", format_summary(bets, "Logged decisions (as recommended at the time)"))
    return bets
