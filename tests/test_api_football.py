"""The API-Football client, tested against scripted fake responses (no network)."""

import json
import logging

import pandas as pd
import pytest
import requests

from corners.data import api_football
from corners.data.api_football import ApiFootballClient, PlanRestrictedError


class FakeResponse:
    def __init__(self, payload=None, status=200, headers=None):
        self._payload = payload if payload is not None else {"response": [], "errors": [], "paging": {}}
        self.status_code = status
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "params": params})
        return self.responses.pop(0)


def ok(response, remaining="90", paging=None):
    payload = {"response": response, "errors": [], "paging": paging or {"total": 1}}
    return FakeResponse(payload, headers={"x-ratelimit-requests-remaining": remaining})


def error_body(errors):
    return FakeResponse({"response": [], "errors": errors, "paging": {}})


def client(*responses):
    return ApiFootballClient(key="test-key", session=FakeSession(*responses))


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(api_football.time, "sleep", lambda seconds: None)


def test_get_returns_the_response_and_tracks_the_daily_quota():
    api = client(ok([{"id": 1}], remaining="42"))
    assert api.get("fixtures", {"date": "2026-11-03"}) == [{"id": 1}]
    assert api.requests_remaining == 42
    call = api._session.calls[0]
    assert call["url"].endswith("/fixtures")
    assert call["headers"] == {"x-apisports-key": "test-key"}
    assert call["params"] == {"date": "2026-11-03"}


def test_rate_limits_are_retried_whether_signalled_by_status_or_body():
    api = client(FakeResponse(status=429), error_body({"rateLimit": "Too many requests"}), ok(["done"]))
    assert api.get("fixtures") == ["done"]
    assert len(api._session.calls) == 3


def test_it_gives_up_after_the_retry_limit():
    limited = [error_body({"rateLimit": "Too many requests"}) for _ in range(api_football.API_MAX_RETRIES)]
    with pytest.raises(RuntimeError, match="rateLimit"):
        client(*limited).get("fixtures")


def test_plan_errors_are_distinguished_from_other_api_errors():
    with pytest.raises(PlanRestrictedError):
        client(error_body({"plan": "Free plans cannot access this date"})).get("fixtures")

    with pytest.raises(RuntimeError, match="daily limit") as quota:
        client(error_body({"requests": "daily limit reached"})).get("fixtures")
    assert not isinstance(quota.value, PlanRestrictedError)


def test_http_errors_are_raised():
    with pytest.raises(requests.HTTPError):
        client(FakeResponse(status=500)).get("fixtures")


def test_a_paginated_response_is_flagged(caplog):
    caplog.set_level(logging.WARNING, logger="corners.data.api_football")
    client(ok([], paging={"total": 3})).get("odds")
    assert "only page 1 was read" in caplog.text


BETS = [
    {"id": 45, "name": "Corners Over Under"},
    {"id": 57, "name": "Home Corners Over/Under"},
    {"id": 58, "name": "Away Corners Over/Under"},
    {"id": 5, "name": "Goals Over/Under"},
    {"id": 99, "name": "Match Winner"},
]


def test_bet_ids_are_looked_up_once_and_then_cached(workspace):
    expected = {"corners": 45, "home_corners": 57, "away_corners": 58, "goals": 5}
    assert client(ok(BETS)).bet_ids() == expected
    assert json.loads((workspace / "data" / "api_bet_ids.json").read_text()) == expected
    # No responses are scripted, so a second client would fail if it called the API.
    assert client().bet_ids() == expected


def test_a_missing_market_is_an_error(workspace):
    with pytest.raises(RuntimeError, match="corners"):
        client(ok([{"id": 1, "name": "Match Winner"}])).bet_ids()


def fixture(fixture_id, league_id, status):
    return {
        "league": {"id": league_id},
        "fixture": {"id": fixture_id, "date": "2026-11-03T20:00:00+00:00", "status": {"short": status}},
        "teams": {"home": {"name": "Leeds"}, "away": {"name": "Fulham"}},
    }


def test_fixtures_are_limited_to_tracked_leagues_and_useful_statuses():
    found = client(
        ok([fixture(1, 39, "NS"), fixture(2, 39, "FT"), fixture(3, 39, "PST"), fixture(4, 9999, "NS")])
    ).fixtures_on("2026-11-03")
    assert [f["fixture_id"] for f in found] == [1, 2]
    assert found[0]["league"] == "E0"
    assert found[0]["status"] == "NS"
    assert found[0]["start_datetime"] == pd.Timestamp("2026-11-03 20:00", tz="UTC")


def bookmaker(name, bet_id, values):
    bet = {"id": bet_id, "values": [{"value": value, "odd": odd} for value, odd in values]}
    return {"name": name, "bets": [bet]}


def test_best_odds_keeps_the_highest_price_and_who_offered_it():
    response = [
        {
            "bookmakers": [
                bookmaker("BookA", 45, [("Over 9.5", "1.90"), ("Under 9.5", "1.95"), ("Over 11.5", "3.0")]),
                bookmaker("BookB", 45, [("Over 9.5", "2.05"), ("Under 9.5", "1.85"), ("Exactly 9", "5.0")]),
                bookmaker("BookC", 45, [("Over 8.5", "not a number")]),
                bookmaker("BookD", 999, [("Over 9.5", "9.0")]),  # a market we don't track
            ]
        }
    ]
    prices, books = client(ok(response)).best_odds(1, {"corners": (45, [8.5, 9.5])})["corners"]

    assert prices[9.5] == {"Over": 2.05, "Under": 1.95}
    assert books[9.5] == {"Over": "BookB", "Under": "BookA"}
    assert prices[8.5] == {"Over": 0.0, "Under": 0.0}
    assert books[8.5] == {"Over": None, "Under": None}
