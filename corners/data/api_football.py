"""Thin client for the API-Football v3 REST API (free plan).

The free plan only exposes fixtures within a rolling window of roughly
yesterday to tomorrow and has no historical prices, which is why the pipeline
captures prices itself as fixtures come into view.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import pandas as pd
import requests

from corners.config import (
    API_BASE_URL,
    API_LEAGUE_ID_TO_CODE,
    API_MAX_RETRIES,
    API_REQUEST_PAUSE_SECONDS,
    MARKET_ID_CACHE_FILE,
    api_key,
)

log = logging.getLogger(__name__)

# Statuses worth fetching prices for: not started (early price) and full time
# (closing price). Postponed, abandoned and in-play fixtures are skipped.
WANTED_STATUSES = {"NS", "FT"}

# Market name as API-Football lists it -> substring fallback.
_MARKET_NAMES = {
    "corners": ("corners over under", "corner"),
    "home_corners": ("home corners over/under", "home corner"),
    "away_corners": ("away corners over/under", "away corner"),
    "goals": ("goals over/under", "goals over"),
}

BestPrices = dict[float, dict[str, Any]]
"""``{line: {'Over': value, 'Under': value}}``"""


class PlanRestrictedError(RuntimeError):
    """The request is outside what the current subscription plan allows."""


class ApiFootballClient:
    """API-Football client with rate-limit retries and quota tracking."""

    def __init__(self, key: str | None = None, session: requests.Session | None = None) -> None:
        self._key = key or api_key()
        self._session = session or requests.Session()
        self.requests_remaining: int | None = None  # daily quota, from response headers

    def get(self, endpoint: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """GET an endpoint and return its ``response`` list.

        Per-minute rate limits are retried with a growing back-off. API-Football
        signals them either as HTTP 429 or as HTTP 200 with a ``rateLimit``
        error in the body. Raises PlanRestrictedError for plan limits and
        RuntimeError for any other API error, including an exhausted daily quota.
        """
        for attempt in range(1, API_MAX_RETRIES + 1):
            resp = self._session.get(
                f"{API_BASE_URL}/{endpoint}",
                headers={"x-apisports-key": self._key},
                params=params or {},
                timeout=20,
            )
            if resp.status_code == 429 and attempt < API_MAX_RETRIES:
                self._back_off(endpoint, attempt)
                continue
            resp.raise_for_status()
            payload = resp.json()
            errors = payload.get("errors")
            if isinstance(errors, dict) and "rateLimit" in errors and attempt < API_MAX_RETRIES:
                self._back_off(endpoint, attempt)
                continue
            break

        if errors:
            if isinstance(errors, dict) and "plan" in errors:
                raise PlanRestrictedError(errors["plan"])
            raise RuntimeError(f"API-Football error on '{endpoint}': {errors}")

        remaining = resp.headers.get("x-ratelimit-requests-remaining")
        if remaining is not None:
            self.requests_remaining = int(remaining)
            log.debug("%s: %s requests left today", endpoint, remaining)

        # Only page 1 is ever requested. Price responses have never been
        # paginated, but a silently truncated response would understate the
        # best available price, so make it loud if it ever happens.
        paging = payload.get("paging") or {}
        if paging.get("total", 1) > 1:
            log.warning("%s response has %s pages; only page 1 was read", endpoint, paging.get("total"))

        time.sleep(API_REQUEST_PAUSE_SECONDS)
        return payload.get("response", [])

    @staticmethod
    def _back_off(endpoint: str, attempt: int) -> None:
        wait = 5 * attempt
        log.warning("%s rate limited, retrying in %ss", endpoint, wait)
        time.sleep(wait)

    def market_ids(self) -> dict[str, int]:
        """API-Football's market ids for each tracked market (cached on disk)."""
        cached: dict[str, int] = {}
        if MARKET_ID_CACHE_FILE.exists():
            cached = json.loads(MARKET_ID_CACHE_FILE.read_text())
        missing = [market for market in _MARKET_NAMES if market not in cached]
        if missing:
            listed = self.get("odds/bets")  # the provider's name for its market list
            for market in missing:
                exact, contains = _MARKET_NAMES[market]
                names = [(b, (b.get("name") or "").strip().lower()) for b in listed]
                match = next((b for b, n in names if n == exact), None) or next(
                    (b for b, n in names if contains in n), None
                )
                if match is None:
                    raise RuntimeError(f"No '{market}' market found in /odds/bets")
                cached[market] = match["id"]
            MARKET_ID_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            MARKET_ID_CACHE_FILE.write_text(json.dumps(cached))
        return {market: cached[market] for market in _MARKET_NAMES}

    def fixtures_on(self, date_str: str) -> list[dict[str, Any]]:
        """Tracked-league fixtures on a date with a status worth fetching prices for."""
        fixtures = []
        for f in self.get("fixtures", {"date": date_str}):
            league = API_LEAGUE_ID_TO_CODE.get(f["league"]["id"])
            status = f["fixture"]["status"]["short"]
            if league is None or status not in WANTED_STATUSES:
                continue
            fixtures.append(
                {
                    "fixture_id": f["fixture"]["id"],
                    "league": league,
                    "start_datetime": pd.to_datetime(f["fixture"]["date"], utc=True),
                    "home_team": f["teams"]["home"]["name"],
                    "away_team": f["teams"]["away"]["name"],
                    "status": status,
                }
            )
        return fixtures

    def best_prices(
        self, fixture_id: int, markets: dict[str, tuple[int, list[float]]]
    ) -> dict[str, tuple[BestPrices, BestPrices]]:
        """Best over/under price per line, and the provider offering it.

        One unfiltered ``/odds`` call returns every market for the fixture,
        which is four times cheaper than one call per market.

        Args:
            fixture_id: API-Football fixture id.
            markets: ``{market: (market_id, lines)}``.

        Returns:
            ``{market: (prices, providers)}`` where a missing price is 0.0
            and a missing provider is None.
        """
        response = self.get("odds", {"fixture": fixture_id})
        market_by_id = {market_id: market for market, (market_id, _) in markets.items()}
        prices = {
            m: {line: {"Over": 0.0, "Under": 0.0} for line in lines} for m, (_, lines) in markets.items()
        }
        books = {
            m: {line: {"Over": None, "Under": None} for line in lines} for m, (_, lines) in markets.items()
        }

        for entry in response:
            for provider in entry.get("bookmakers", []):  # the provider's field names
                for listing in provider.get("bets", []):
                    market = market_by_id.get(listing.get("id"))
                    if market is None:
                        continue
                    for value in listing.get("values", []):
                        parts = str(value.get("value", "")).split()
                        if len(parts) != 2 or parts[0] not in ("Over", "Under"):
                            continue
                        try:
                            line, odd = float(parts[1]), float(value.get("odd"))
                        except (TypeError, ValueError):
                            continue
                        side = parts[0]
                        if line in prices[market] and odd > prices[market][line][side]:
                            prices[market][line][side] = odd
                            books[market][line][side] = provider.get("name")

        return {market: (prices[market], books[market]) for market in markets}
