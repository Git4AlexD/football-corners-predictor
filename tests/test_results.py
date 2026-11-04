"""Downloading and merging match results, tested without network access."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import requests

from corners.config import current_season_code
from corners.data import results as results_module
from corners.data.api_football import PlanRestrictedError
from corners.data.results import (
    RESULT_COLUMNS,
    _merge_canonical,
    _result_label,
    add_derived_columns,
    download_football_data,
    fetch_recent_via_api,
    update_results,
)

CSV = (
    "Date,HomeTeam,AwayTeam,FTHG,FTAG,HTHG,HTAG,HC,AC,HS,AS,HST,AST,HF,AF,HY,AY,HR,AR,Avg>2.5,Avg<2.5,B365H\n"
    "02/08/2026,Leeds,Fulham,2,1,1,0,6,3,12,8,5,2,10,11,1,2,0,0,1.9,1.95,2.1\n"
    "03/08/2026,Spurs,Arsenal,0,0,0,0,,4,9,9,3,3,8,9,0,1,0,0,1.8,2.0,3.0\n"
)


class Reply:
    def __init__(self, text="", status=200):
        self.text, self.status_code = text, status


def serve(monkeypatch, replies):
    """Make requests.get answer from ``{url fragment: Reply}``, 404 otherwise."""

    def fake_get(url, headers=None, timeout=None):
        for fragment, reply in replies.items():
            if fragment in url:
                return reply
        return Reply(status=404)

    monkeypatch.setattr(results_module.requests, "get", fake_get)


# ---------------------------------------------------------------------------
# football-data.co.uk
# ---------------------------------------------------------------------------
def test_a_season_file_is_parsed_and_incomplete_rows_dropped(monkeypatch):
    serve(monkeypatch, {"/2627/E0.csv": Reply(CSV)})
    df = results_module._download_season("2627", "E0")

    assert len(df) == 1  # the second row has no corner count
    row = df.iloc[0]
    assert (row["HomeTeam"], row["AwayTeam"]) == ("Leeds", "Fulham")
    assert row["League"] == "Premier League"
    assert row["Season"] == "2026/2027"
    assert row["Date"] == pd.Timestamp("2026-08-02")  # day-first dates
    assert "B365H" not in df.columns  # unused provider columns are dropped


@pytest.mark.parametrize(
    "reply",
    [Reply(status=404), Reply("a,b\n1,2\n"), Reply("<html>blocked</html>"), Reply("")],
    ids=["not found", "wrong columns", "html page", "empty body"],
)
def test_unusable_downloads_are_skipped(monkeypatch, reply):
    serve(monkeypatch, {"E0.csv": reply})
    assert results_module._download_season("2627", "E0") is None


def test_a_connection_error_is_skipped(monkeypatch):
    def refuse(*args, **kwargs):
        raise requests.ConnectionError("blocked")

    monkeypatch.setattr(results_module.requests, "get", refuse)
    assert results_module._download_season("2627", "E0") is None


def test_download_collects_every_available_league(monkeypatch):
    serve(monkeypatch, {"/2627/E0.csv": Reply(CSV), "/2627/D1.csv": Reply(CSV)})
    df = download_football_data(["2627"])

    assert list(df.columns) == RESULT_COLUMNS
    assert set(df["League"]) == {"Premier League", "Bundesliga"}
    assert df["HC"].dtype.kind in "if"
    assert df["home_shot_conversion"].iloc[0] == pytest.approx(5 / 12)


def test_download_with_nothing_available_is_empty(monkeypatch):
    serve(monkeypatch, {})
    df = download_football_data(["2627"])
    assert df.empty
    assert list(df.columns) == RESULT_COLUMNS


def test_derived_columns():
    df = pd.DataFrame(
        {
            "HS": [10, 0], "HST": [4, 0], "AS": [6, 3], "AST": [3, 1],
            "HC": [6, 2], "AC": [3, 5], "HTHG": [0, 1], "HTAG": [1, 0],
            "HR": [0, 1], "AR": [0, 0], "HF": [9, 7], "AF": [11, 8],
        }
    )  # fmt: skip
    out = add_derived_columns(df)
    assert out["home_shot_conversion"].tolist() == [0.4, 0.0]  # no division by zero
    assert out["total_match_corners"].tolist() == [9, 7]
    assert out["home_trailing_ht"].tolist() == [1, 0]
    assert out["away_trailing_ht"].tolist() == [0, 1]
    assert out["has_red_card"].tolist() == [0, 1]
    assert out["total_fouls"].tolist() == [20, 15]


@pytest.mark.parametrize(
    ("home", "away", "label"), [(2, 1, "H"), (0, 3, "A"), (1, 1, "D"), (None, None, "D")]
)
def test_result_label(home, away, label):
    assert _result_label(home, away) == label


def test_fresh_rows_replace_stored_ones_even_if_a_team_was_spelled_differently():
    existing = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-11-01", "2026-11-01", "2026-10-25"]),
            "League": ["Premier League"] * 3,
            "HomeTeam": ["Leeds United", "Spurs", "Brighton"],  # first row came from the API fallback
            "AwayTeam": ["Fulham", "Arsenal", "Everton"],
            "HC": [1, 2, 3],
        }
    )
    fresh = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-11-01"]),
            "League": ["Premier League"],
            "HomeTeam": ["Leeds"],
            "AwayTeam": ["Fulham"],
            "HC": [9],
        }
    )
    merged = _merge_canonical(existing, fresh)
    assert len(merged) == 3
    assert "Leeds United" not in set(merged["HomeTeam"])  # replaced via the matching away team
    assert merged.loc[merged["HomeTeam"] == "Leeds", "HC"].item() == 9


# ---------------------------------------------------------------------------
# API-Football fallback
# ---------------------------------------------------------------------------
class FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime(2026, 11, 3, 12, 0, tzinfo=tz)


class FakeApi:
    def __init__(self, fixtures=None, stats=None, remaining=None, blocked_days=(), stats_error=None):
        self.fixtures = fixtures or {}
        self.stats = stats or {}
        self.requests_remaining = remaining
        self.blocked_days = set(blocked_days)
        self.stats_error = stats_error
        self.stats_calls = []

    def get(self, endpoint, params=None):
        if endpoint == "fixtures":
            if params["date"] in self.blocked_days:
                raise PlanRestrictedError("outside the plan's window")
            return self.fixtures.get(params["date"], [])
        self.stats_calls.append(params["fixture"])
        if self.stats_error:
            raise self.stats_error
        return self.stats.get(params["fixture"], [])


def api_fixture(fixture_id, home, away, status="FT", league_id=39, goals=(2, 1), half=(1, 0)):
    return {
        "league": {"id": league_id},
        "fixture": {"id": fixture_id, "status": {"short": status}},
        "teams": {"home": {"name": home}, "away": {"name": away}},
        "score": {
            "fulltime": {"home": goals[0], "away": goals[1]},
            "halftime": {"home": half[0], "away": half[1]},
        },
    }


def team_stats(team, corners, shots=12, on_target=5):
    values = {
        "Corner Kicks": corners, "Total Shots": shots, "Shots on Goal": on_target,
        "Fouls": 9, "Yellow Cards": 1, "Red Cards": 0,
    }  # fmt: skip
    return {"team": {"name": team}, "statistics": [{"type": k, "value": v} for k, v in values.items()]}


@pytest.fixture
def stored():
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-10-20", "2026-11-01"]),
            "League": ["Premier League"] * 2,
            "HomeTeam": ["Brighton", "Leeds"],
            "AwayTeam": ["Everton", "Fulham"],
        }
    )


@pytest.fixture
def frozen_today(monkeypatch):
    monkeypatch.setattr(results_module, "datetime", FrozenDatetime)


def test_fallback_adds_new_results_under_the_stored_team_names(stored, frozen_today):
    api = FakeApi(
        fixtures={
            "2026-11-01": [api_fixture(2, "Leeds", "Fulham")],  # already stored
            "2026-11-02": [
                api_fixture(1, "Brighton & Hove Albion", "Everton"),
                api_fixture(3, "Madrid Giants", "Madrid Rivals", league_id=1),  # not a tracked league
                api_fixture(4, "Leeds", "Brighton", status="NS"),  # not played yet
                api_fixture(5, "Fulham", "Leeds"),  # statistics not published yet
            ],
        },
        stats={1: [team_stats("Brighton & Hove Albion", 7, 14, 6), team_stats("Everton", 3, 8, 2)]},
    )
    new = fetch_recent_via_api(stored, api)

    assert list(new.columns) == RESULT_COLUMNS
    assert len(new) == 1
    row = new.iloc[0]
    assert (row["HomeTeam"], row["AwayTeam"]) == ("Brighton", "Everton")  # matched to stored spelling
    assert (row["HC"], row["AC"]) == (7, 3)
    assert (row["FTR"], row["HTR"]) == ("H", "H")
    assert row["Date"] == pd.Timestamp("2026-11-02")
    assert row["Season"] == f"20{current_season_code()[:2]}/20{current_season_code()[2:]}"
    assert api.stats_calls == [1, 5]  # the stored match was not fetched again


def test_fallback_flags_a_name_it_cannot_match_and_keeps_the_api_spelling(stored, frozen_today, workspace):
    api = FakeApi(
        fixtures={"2026-11-03": [api_fixture(6, "Zzyzx Rovers", "Everton")]},
        stats={6: [team_stats("Zzyzx Rovers", 4), team_stats("Everton", 5)]},
    )
    new = fetch_recent_via_api(stored, api)

    assert new.iloc[0]["HomeTeam"] == "Zzyzx Rovers"
    assert "Zzyzx Rovers" in (workspace / "data" / "unmatched_teams.csv").read_text()


def test_fallback_leaves_a_reserve_of_requests_for_odds(stored, frozen_today):
    api = FakeApi(fixtures={"2026-11-02": [api_fixture(1, "Brighton", "Everton")]}, remaining=10)
    new = fetch_recent_via_api(stored, api)
    assert new.empty
    assert list(new.columns) == RESULT_COLUMNS
    assert api.stats_calls == []


def test_fallback_keeps_what_it_gathered_if_the_quota_runs_out(stored, frozen_today):
    api = FakeApi(
        fixtures={"2026-11-02": [api_fixture(1, "Brighton", "Everton")]},
        stats_error=RuntimeError("daily quota exhausted"),
    )
    assert fetch_recent_via_api(stored, api).empty  # stops cleanly rather than raising


def test_fallback_skips_a_day_outside_the_plans_window(stored, frozen_today):
    api = FakeApi(
        fixtures={"2026-11-02": [api_fixture(1, "Brighton", "Everton")]},
        stats={1: [team_stats("Brighton", 7), team_stats("Everton", 3)]},
        blocked_days={"2026-11-01"},
    )
    assert len(fetch_recent_via_api(stored, api)) == 1


# ---------------------------------------------------------------------------
# Updating the stored file
# ---------------------------------------------------------------------------
def test_a_first_run_saves_the_downloaded_results(monkeypatch, workspace, raw_results):
    monkeypatch.setattr(results_module, "download_football_data", lambda seasons: raw_results.head(40))
    out = update_results()
    assert len(out) == 40
    assert len(pd.read_csv(workspace / "data" / "raw_results.csv")) == 40


def test_an_update_replaces_stored_matches_with_corrected_ones(monkeypatch, workspace, raw_results):
    raw_results.head(40).to_csv(workspace / "data" / "raw_results.csv", index=False)
    corrected = raw_results.head(10).assign(HC=99)
    monkeypatch.setattr(results_module, "download_football_data", lambda seasons: corrected)

    out = update_results(client=FakeApi(remaining=1))  # no quota left for the fallback

    assert len(out) == 40
    assert (out["HC"] == 99).sum() == 10
    assert not out.duplicated(["Date", "HomeTeam", "AwayTeam", "League"]).any()


def test_stored_results_are_kept_when_the_download_returns_nothing(monkeypatch, workspace, raw_results):
    raw_results.head(40).to_csv(workspace / "data" / "raw_results.csv", index=False)
    monkeypatch.setattr(results_module, "download_football_data", lambda seasons: pd.DataFrame())
    assert len(update_results()) == 40


def test_having_no_results_anywhere_is_an_error(monkeypatch, workspace):
    monkeypatch.setattr(results_module, "download_football_data", lambda seasons: pd.DataFrame())
    with pytest.raises(RuntimeError, match="No match results"):
        update_results()


def test_nan_odds_are_allowed_in_fallback_rows(stored, frozen_today):
    api = FakeApi(
        fixtures={"2026-11-02": [api_fixture(1, "Brighton", "Everton")]},
        stats={1: [team_stats("Brighton", 7), team_stats("Everton", 3)]},
    )
    new = fetch_recent_via_api(stored, api)
    assert np.isnan(new.iloc[0]["Avg>2.5"])  # the API has no pre-closing goals price
