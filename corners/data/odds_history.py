"""The permanent odds history: two locked snapshots per fixture.

For every fixture the pipeline sees, it stores the best price per line for
total, home and away corners and for goals over/under 2.5, at two moments:

* **early**: the first time the fixture is seen before kickoff;
* **closing**: the first time it is seen after full time (API-Football keeps
  serving the frozen pre-match price, which works as a closing line).

Each snapshot is written once and never overwritten, so comparing them
measures how the market moved. Rows are never deleted. A fixture first seen
after it finished has a closing snapshot only.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd
import requests

from corners.config import (
    AWAY_CORNER_LINES,
    DISCOVERY_DAYS_AHEAD,
    DISCOVERY_DAYS_BACK,
    GOALS_LINES,
    HOME_CORNER_LINES,
    ODDS_HISTORY_FILE,
    TOTAL_CORNER_LINES,
)
from corners.data.api_football import ApiFootballClient, BestPrices, PlanRestrictedError
from corners.pricing import OddsLadder, fair_over_prob

log = logging.getLogger(__name__)

STAGES = ("early", "closing")
SIDES = ("over", "under")

# Column key used in the history file -> lines stored for it.
ODDS_MARKETS: dict[str, list[float]] = {
    "corners": TOTAL_CORNER_LINES,
    "home_corners": HOME_CORNER_LINES,
    "away_corners": AWAY_CORNER_LINES,
    "goals": GOALS_LINES,
}
META_COLUMNS = ["fixture_id", "league", "start_datetime", "home_team", "away_team"]


def _suffix(line: float) -> str:
    return str(line).replace(".5", "_5")


def odds_column(stage: str, side: str, line: float, market: str) -> str:
    return f"{stage}_{side}_{_suffix(line)}_{market}_ft_best_odds"


def bookmaker_column(stage: str, side: str, line: float, market: str) -> str:
    return f"{stage}_{side}_{_suffix(line)}_{market}_ft_best_bookmaker"


def _schema() -> list[str]:
    columns = META_COLUMNS + [f"{stage}_fetched_at" for stage in STAGES]
    for market, lines in ODDS_MARKETS.items():
        for column_name in (odds_column, bookmaker_column):
            columns += [
                column_name(stage, side, line, market) for stage in STAGES for line in lines for side in SIDES
            ]
    return columns


ODDS_COLUMNS = _schema()


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def load_odds_history() -> pd.DataFrame:
    """The full history, padded to the current schema."""
    if not ODDS_HISTORY_FILE.exists():
        return pd.DataFrame(columns=ODDS_COLUMNS)
    df = pd.read_csv(ODDS_HISTORY_FILE)
    df = df.reindex(columns=list(dict.fromkeys([*ODDS_COLUMNS, *df.columns])))
    df["start_datetime"] = pd.to_datetime(df["start_datetime"], utc=True)
    for stage in STAGES:
        df[f"{stage}_fetched_at"] = pd.to_datetime(df[f"{stage}_fetched_at"], utc=True)
    return df


def _price(row: pd.Series, column: str) -> float:
    value = pd.to_numeric(row.get(column), errors="coerce")
    return float(value) if pd.notna(value) and value > 0 else 0.0


def stage_ladder(row: pd.Series, market: str, stage: str) -> OddsLadder:
    """``{line: (over, under)}`` from a single snapshot (0.0 where missing)."""
    return {
        line: tuple(_price(row, odds_column(stage, side, line, market)) for side in SIDES)
        for line in ODDS_MARKETS[market]
    }


def betting_ladder(row: pd.Series, market: str) -> OddsLadder:
    """The prices the strategy bets at: early, falling back to closing per price.

    Early prices are preferred because any beatable inefficiency is more
    likely to survive there than in the sharper closing line. Closing only
    fills in prices for fixtures first seen after they had started.
    """
    early, closing = stage_ladder(row, market, "early"), stage_ladder(row, market, "closing")
    return {
        line: tuple(e or c for e, c in zip(early[line], closing[line], strict=True))
        for line in ODDS_MARKETS[market]
    }


def betting_bookmakers(row: pd.Series, market: str) -> dict[float, tuple[str | None, str | None]]:
    """Bookmaker behind each price in ``betting_ladder`` (same stage priority)."""
    result = {}
    for line in ODDS_MARKETS[market]:
        pair = []
        for side in SIDES:
            name = None
            for stage in STAGES:
                if _price(row, odds_column(stage, side, line, market)):
                    value = row.get(bookmaker_column(stage, side, line, market))
                    name = value if pd.notna(value) else None
                    break
            pair.append(name)
        result[line] = (pair[0], pair[1])
    return result


def market_over25_prob(row: pd.Series) -> float | None:
    """Fair P(over 2.5 goals) from the betting ladder, or None if unpriced.

    The model is trained on football-data.co.uk's pre-closing goals odds, so
    the early snapshot is the like-for-like input at prediction time.
    """
    over, under = betting_ladder(row, "goals")[2.5]
    prob = fair_over_prob(over, under)
    return None if prob != prob else prob  # NaN check


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def merge_snapshots(updates: list[dict[str, Any]]) -> pd.DataFrame:
    """Merge snapshot fields into the history file and return the result.

    Fields are merged into an existing fixture's row rather than replacing it,
    so recording the closing snapshot never erases the early one.
    """
    history = load_odds_history()
    if not updates:
        return history

    history = history.set_index("fixture_id", drop=False)
    new_rows = []
    for fields in updates:
        fixture_id = fields["fixture_id"]
        if fixture_id in history.index:
            for column, value in fields.items():
                history.loc[fixture_id, column] = value
        else:
            new_rows.append({column: pd.NA for column in ODDS_COLUMNS} | fields)
    if new_rows:
        history = pd.concat([history, pd.DataFrame(new_rows)], ignore_index=True)

    history = history.reset_index(drop=True).sort_values("start_datetime").reset_index(drop=True)
    ODDS_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(ODDS_HISTORY_FILE, index=False)
    return history


def snapshot_fields(
    fixture: dict[str, Any],
    stage: str,
    odds: dict[str, tuple[BestPrices, BestPrices]],
    fetched_at: pd.Timestamp,
) -> dict[str, Any]:
    """One fixture's row fields for a single snapshot stage."""
    fields: dict[str, Any] = {column: fixture[column] for column in META_COLUMNS}
    fields[f"{stage}_fetched_at"] = fetched_at
    for market, lines in ODDS_MARKETS.items():
        prices, bookmakers = odds[market]
        for line in lines:
            for side in SIDES:
                label = side.capitalize()
                fields[odds_column(stage, side, line, market)] = prices[line][label]
                fields[bookmaker_column(stage, side, line, market)] = bookmakers[line][label]
    return fields


def _discover(client: ApiFootballClient, offsets: range, today: date) -> list[dict[str, Any]]:
    """Fixtures on each day offset from today, stopping at the plan's limit.

    An API error also ends the pass rather than the whole run; later days are
    picked up on the next run.
    """
    fixtures: list[dict[str, Any]] = []
    for offset in offsets:
        day = (today + timedelta(days=offset)).isoformat()
        try:
            found = client.fixtures_on(day)
        except PlanRestrictedError:
            log.info("%s is outside the plan's date window; discovery stops here", day)
            break
        except (requests.RequestException, RuntimeError) as exc:
            log.warning("Fixture discovery stopped at %s: %s", day, exc)
            break
        log.info("%s: %d tracked fixture(s)", day, len(found))
        fixtures.extend(found)
    return fixtures


def capture_odds(client: ApiFootballClient) -> pd.DataFrame:
    """Record any early or closing snapshot each visible fixture still lacks."""
    bet_ids = client.bet_ids()
    markets = {market: (bet_ids[market], lines) for market, lines in ODDS_MARKETS.items()}
    today = datetime.now(UTC).date()

    # Backward first, to close out fixtures that finished since the last run.
    discovered = _discover(client, range(-1, -DISCOVERY_DAYS_BACK - 1, -1), today)
    discovered += _discover(client, range(0, DISCOVERY_DAYS_AHEAD + 1), today)

    history = load_odds_history()
    captured = {
        stage: set(history.loc[history[f"{stage}_fetched_at"].notna(), "fixture_id"]) for stage in STAGES
    }
    todo = [
        (fixture, stage)
        for fixture in discovered
        for status, stage in (("NS", "early"), ("FT", "closing"))
        if fixture["status"] == status and fixture["fixture_id"] not in captured[stage]
    ]
    log.info("%d fixture(s) need an odds snapshot", len(todo))

    updates = []
    for fixture, stage in todo:
        log.info(
            "  %s odds: %s %s v %s (%s)",
            stage,
            fixture["league"],
            fixture["home_team"],
            fixture["away_team"],
            fixture["start_datetime"].date(),
        )
        try:
            odds = client.best_odds(fixture["fixture_id"], markets)
        except (requests.RequestException, RuntimeError) as exc:
            log.warning("Odds fetching stopped (%s); saving progress so far", exc)
            break
        updates.append(snapshot_fields(fixture, stage, odds, pd.Timestamp.now(tz="UTC")))

    history = merge_snapshots(updates)
    if not history.empty:
        log.info(
            "Odds history: %d fixtures (%s to %s), %d with early and %d with closing odds",
            len(history),
            history["start_datetime"].min().date(),
            history["start_datetime"].max().date(),
            history["early_fetched_at"].notna().sum(),
            history["closing_fetched_at"].notna().sum(),
        )
    return history
