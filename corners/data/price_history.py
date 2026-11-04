"""The permanent price history: two locked snapshots per fixture.

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
    PRICE_HISTORY_FILE,
    TOTAL_CORNER_LINES,
)
from corners.data.api_football import ApiFootballClient, BestPrices, PlanRestrictedError
from corners.pricing import PriceLadder, fair_over_prob

log = logging.getLogger(__name__)

STAGES = ("early", "closing")
SIDES = ("over", "under")

# Column key used in the history file -> lines stored for it.
PRICE_MARKETS: dict[str, list[float]] = {
    "corners": TOTAL_CORNER_LINES,
    "home_corners": HOME_CORNER_LINES,
    "away_corners": AWAY_CORNER_LINES,
    "goals": GOALS_LINES,
}
META_COLUMNS = ["fixture_id", "league", "start_datetime", "home_team", "away_team"]


def _suffix(line: float) -> str:
    return str(line).replace(".5", "_5")


def price_column(stage: str, side: str, line: float, market: str) -> str:
    return f"{stage}_{side}_{_suffix(line)}_{market}_ft_best_odds"


def provider_column(stage: str, side: str, line: float, market: str) -> str:
    return f"{stage}_{side}_{_suffix(line)}_{market}_ft_best_bookmaker"


def _schema() -> list[str]:
    columns = META_COLUMNS + [f"{stage}_fetched_at" for stage in STAGES]
    for market, lines in PRICE_MARKETS.items():
        for column_name in (price_column, provider_column):
            columns += [
                column_name(stage, side, line, market) for stage in STAGES for line in lines for side in SIDES
            ]
    return columns


PRICE_COLUMNS = _schema()


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def load_price_history() -> pd.DataFrame:
    """The full history, padded to the current schema."""
    if not PRICE_HISTORY_FILE.exists():
        return pd.DataFrame(columns=PRICE_COLUMNS)
    df = pd.read_csv(PRICE_HISTORY_FILE)
    df = df.reindex(columns=list(dict.fromkeys([*PRICE_COLUMNS, *df.columns])))
    # Fix the types of columns that can still be empty. Left alone, pandas would store
    # an empty timestamp column at a coarser resolution and an empty provider column as
    # numbers, and then refuse the first real value written into either.
    for column in ("start_datetime", *(f"{stage}_fetched_at" for stage in STAGES)):
        df[column] = pd.to_datetime(df[column], utc=True).astype("datetime64[ns, UTC]")
    for column in df.columns[df.columns.str.endswith("_best_bookmaker")]:
        df[column] = df[column].astype(object)
    return df


def _price(row: pd.Series, column: str) -> float:
    value = pd.to_numeric(row.get(column), errors="coerce")
    return float(value) if pd.notna(value) and value > 0 else 0.0


def stage_ladder(row: pd.Series, market: str, stage: str) -> PriceLadder:
    """``{line: (over, under)}`` from a single snapshot (0.0 where missing)."""
    return {
        line: tuple(_price(row, price_column(stage, side, line, market)) for side in SIDES)
        for line in PRICE_MARKETS[market]
    }


def decision_ladder(row: pd.Series, market: str) -> PriceLadder:
    """The prices the strategy decisions at: early, falling back to closing per price.

    Early prices are preferred because any beatable inefficiency is more
    likely to survive there than in the sharper closing line. Closing only
    fills in prices for fixtures first seen after they had started.
    """
    early, closing = stage_ladder(row, market, "early"), stage_ladder(row, market, "closing")
    return {
        line: tuple(e or c for e, c in zip(early[line], closing[line], strict=True))
        for line in PRICE_MARKETS[market]
    }


def decision_providers(row: pd.Series, market: str) -> dict[float, tuple[str | None, str | None]]:
    """Provider behind each price in ``decision_ladder`` (same stage priority)."""
    result = {}
    for line in PRICE_MARKETS[market]:
        pair = []
        for side in SIDES:
            name = None
            for stage in STAGES:
                if _price(row, price_column(stage, side, line, market)):
                    value = row.get(provider_column(stage, side, line, market))
                    name = value if pd.notna(value) else None
                    break
            pair.append(name)
        result[line] = (pair[0], pair[1])
    return result


def market_over25_prob(row: pd.Series) -> float | None:
    """Fair P(over 2.5 goals) from the decision ladder, or None if unpriced.

    The model is trained on football-data.co.uk's pre-closing goals price, so
    the early snapshot is the like-for-like input at prediction time.
    """
    over, under = decision_ladder(row, "goals")[2.5]
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
    history = load_price_history()
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
            new_rows.append({column: pd.NA for column in PRICE_COLUMNS} | fields)
    if new_rows:
        history = pd.concat([history, pd.DataFrame(new_rows)], ignore_index=True)

    history = history.reset_index(drop=True).sort_values("start_datetime").reset_index(drop=True)
    PRICE_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(PRICE_HISTORY_FILE, index=False)
    return history


def snapshot_fields(
    fixture: dict[str, Any],
    stage: str,
    price: dict[str, tuple[BestPrices, BestPrices]],
    fetched_at: pd.Timestamp,
) -> dict[str, Any]:
    """One fixture's row fields for a single snapshot stage."""
    fields: dict[str, Any] = {column: fixture[column] for column in META_COLUMNS}
    fields[f"{stage}_fetched_at"] = fetched_at
    for market, lines in PRICE_MARKETS.items():
        prices, providers = price[market]
        for line in lines:
            for side in SIDES:
                label = side.capitalize()
                fields[price_column(stage, side, line, market)] = prices[line][label]
                fields[provider_column(stage, side, line, market)] = providers[line][label]
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


def capture_prices(client: ApiFootballClient) -> pd.DataFrame:
    """Record any early or closing snapshot each visible fixture still lacks."""
    market_ids = client.market_ids()
    markets = {market: (market_ids[market], lines) for market, lines in PRICE_MARKETS.items()}
    today = datetime.now(UTC).date()

    # Backward first, to close out fixtures that finished since the last run.
    discovered = _discover(client, range(-1, -DISCOVERY_DAYS_BACK - 1, -1), today)
    discovered += _discover(client, range(0, DISCOVERY_DAYS_AHEAD + 1), today)

    history = load_price_history()
    captured = {
        stage: set(history.loc[history[f"{stage}_fetched_at"].notna(), "fixture_id"]) for stage in STAGES
    }
    todo = [
        (fixture, stage)
        for fixture in discovered
        for status, stage in (("NS", "early"), ("FT", "closing"))
        if fixture["status"] == status and fixture["fixture_id"] not in captured[stage]
    ]
    log.info("%d fixture(s) need an price snapshot", len(todo))

    updates = []
    for fixture, stage in todo:
        log.info(
            "  %s price: %s %s v %s (%s)",
            stage,
            fixture["league"],
            fixture["home_team"],
            fixture["away_team"],
            fixture["start_datetime"].date(),
        )
        try:
            price = client.best_prices(fixture["fixture_id"], markets)
        except (requests.RequestException, RuntimeError) as exc:
            log.warning("Price fetching stopped (%s); saving progress so far", exc)
            break
        updates.append(snapshot_fields(fixture, stage, price, pd.Timestamp.now(tz="UTC")))

    history = merge_snapshots(updates)
    if not history.empty:
        log.info(
            "Price history: %d fixtures (%s to %s), %d with early and %d with closing prices",
            len(history),
            history["start_datetime"].min().date(),
            history["start_datetime"].max().date(),
            history["early_fetched_at"].notna().sum(),
            history["closing_fetched_at"].notna().sum(),
        )
    return history
