"""Completed match results and statistics.

football-data.co.uk is the canonical source (one CSV per league and season,
including pre-closing goals odds). When it is unreachable, recent results
are filled in from API-Football's fixture statistics so the pipeline stays
current. football-data.co.uk rows replace fallback rows once published.
"""

from __future__ import annotations

import io
import logging
from datetime import UTC, date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import requests

from corners.config import (
    API_LEAGUE_ID_TO_CODE,
    LEAGUE_NAME_TO_CODE,
    LEAGUES,
    RAW_RESULTS_FILE,
    RESULTS_FALLBACK_QUOTA_RESERVE,
    RESULTS_FALLBACK_RESCAN_DAYS,
    current_season_code,
    season_codes,
)
from corners.data.api_football import ApiFootballClient, PlanRestrictedError
from corners.teams import build_team_lookup, log_unmatched_team, match_team

log = logging.getLogger(__name__)

FOOTBALL_DATA_URL = "https://www.football-data.co.uk/mmz4231/{season}/{league}.csv"

STAT_COLUMNS = [
    "FTHG",
    "FTAG",
    "HTHG",
    "HTAG",
    "HC",
    "AC",
    "HS",
    "AS",
    "HST",
    "AST",
    "HF",
    "AF",
    "HY",
    "AY",
    "HR",
    "AR",
]
# Pre-closing average over/under 2.5 goals odds (from 2019/20 onwards).
# Pre-closing rather than closing because that is what exists before kickoff.
GOALS_ODDS_COLUMNS = ["Avg>2.5", "Avg<2.5"]
_SOURCE_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTR", "HTR", *STAT_COLUMNS, *GOALS_ODDS_COLUMNS]
RESULT_COLUMNS = [
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
    "HTHG",
    "HTAG",
    "HTR",
    "HC",
    "AC",
    "HS",
    "AS",
    "HST",
    "AST",
    "HF",
    "AF",
    "HY",
    "AY",
    "HR",
    "AR",
    "League",
    "Season",
    "home_shots_off_target_or_blocked",
    "home_shot_conversion",
    "away_shots_off_target_or_blocked",
    "away_shot_conversion",
    "total_match_corners",
    "total_match_shots",
    "ht_goal_diff",
    "home_trailing_ht",
    "away_trailing_ht",
    "has_red_card",
    "total_fouls",
    *GOALS_ODDS_COLUMNS,
]
# API-Football statistic name -> our column suffix (H*/A*).
_API_STATS = {
    "Corner Kicks": "C",
    "Total Shots": "S",
    "Shots on Goal": "ST",
    "Fouls": "F",
    "Yellow Cards": "Y",
    "Red Cards": "R",
}


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Per-match shot, half-time and discipline columns."""
    df["home_shots_off_target_or_blocked"] = df["HS"] - df["HST"]
    df["home_shot_conversion"] = np.where(df["HS"] > 0, df["HST"] / df["HS"], 0.0)
    df["away_shots_off_target_or_blocked"] = df["AS"] - df["AST"]
    df["away_shot_conversion"] = np.where(df["AS"] > 0, df["AST"] / df["AS"], 0.0)
    df["total_match_corners"] = df["HC"] + df["AC"]
    df["total_match_shots"] = df["HS"] + df["AS"]
    df["ht_goal_diff"] = df["HTHG"] - df["HTAG"]
    df["home_trailing_ht"] = (df["HTHG"] < df["HTAG"]).astype(int)
    df["away_trailing_ht"] = (df["HTAG"] < df["HTHG"]).astype(int)
    df["has_red_card"] = ((df["HR"] > 0) | (df["AR"] > 0)).astype(int)
    df["total_fouls"] = df["HF"] + df["AF"]
    return df


def _download_season(season: str, league: str) -> pd.DataFrame | None:
    """One league-season CSV from football-data.co.uk, or None if unavailable."""
    url = FOOTBALL_DATA_URL.format(season=season, league=league)
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    except requests.RequestException as exc:
        log.debug("Download failed for %s: %s", url, exc)
        return None
    if resp.status_code != 200:
        return None
    try:
        df = pd.read_csv(io.StringIO(resp.text), encoding="unicode_escape")
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        log.warning("Could not parse %s: %s", url, exc)
        return None
    if not {"HC", "AC", "HomeTeam", "AwayTeam", "Date"}.issubset(df.columns):
        return None

    # The files carry 100+ bookmaker columns; keep only what is used.
    df = df.reindex(columns=_SOURCE_COLUMNS)
    df["League"] = LEAGUES[league][0]
    df["Season"] = f"20{season[:2]}/20{season[2:]}"
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    return df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "HC", "AC"]).copy()


def download_football_data(seasons: list[str]) -> pd.DataFrame:
    """All tracked leagues for the given seasons, in ``RESULT_COLUMNS`` shape."""
    frames = [
        frame
        for season in seasons
        for league in LEAGUES
        if (frame := _download_season(season, league)) is not None and not frame.empty
    ]
    if not frames:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    for column in STAT_COLUMNS + GOALS_ODDS_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return add_derived_columns(df)[RESULT_COLUMNS]


def _merge_canonical(existing: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    """Merge football-data.co.uk rows over existing ones, which they replace.

    A team plays at most once a day, so a match is identified by date, league
    and either team. That still catches a fallback row that stored one side
    under a different spelling.
    """
    slots = set(zip(fresh["Date"], fresh["League"], fresh["HomeTeam"], strict=True)) | set(
        zip(fresh["Date"], fresh["League"], fresh["AwayTeam"], strict=True)
    )
    superseded = np.array(
        [
            (d, lg, h) in slots or (d, lg, a) in slots
            for d, lg, h, a in zip(
                pd.to_datetime(existing["Date"]),
                existing["League"],
                existing["HomeTeam"],
                existing["AwayTeam"],
                strict=True,
            )
        ],
        dtype=bool,
    )
    log.info(
        "football-data.co.uk: %d current-season match(es), %d new, %d refreshed",
        len(fresh),
        len(fresh) - superseded.sum(),
        superseded.sum(),
    )
    return pd.concat([existing[~superseded], fresh], ignore_index=True)


def _stat(team_stats: dict[str, Any], name: str) -> Any:
    return next((s["value"] for s in team_stats.get("statistics", []) if s["type"] == name), None)


def _result_label(home: int | None, away: int | None) -> str:
    home, away = home or 0, away or 0
    return "H" if home > away else "A" if away > home else "D"


class _QuotaExhausted(Exception):
    """Raised to end the fallback early and keep what was gathered."""


def fetch_recent_via_api(results: pd.DataFrame, client: ApiFootballClient) -> pd.DataFrame:
    """Fill recent results missing from ``results`` using API-Football.

    Covers the gap from the latest stored date to today, always re-scanning
    the last few days because an interrupted run can leave a day half-filled.
    Stops early to keep a reserve of the daily request quota for odds, which
    unlike results cannot be fetched later. Team names are mapped to the
    spellings already in ``results`` so each club keeps one identity.
    """
    today = datetime.now(UTC).date()
    latest = pd.to_datetime(results["Date"]).max().date()
    start = min(latest + timedelta(days=1), today - timedelta(days=RESULTS_FALLBACK_RESCAN_DAYS))

    lookup = build_team_lookup(results.assign(League=results["League"].replace(LEAGUE_NAME_TO_CODE)))
    already_have = set(
        zip(
            pd.to_datetime(results["Date"]).dt.date,
            results["HomeTeam"],
            results["AwayTeam"],
            results["League"],
            strict=True,
        )
    )
    season = current_season_code()
    season_label = f"20{season[:2]}/20{season[2:]}"

    rows: list[dict[str, Any]] = []
    day = start
    try:
        while day <= today:
            _check_quota(client)
            try:
                fixtures = client.get("fixtures", {"date": day.isoformat()})
            except PlanRestrictedError:
                fixtures = []
            except RuntimeError as exc:
                raise _QuotaExhausted(str(exc)) from exc
            for fx in fixtures:
                league = API_LEAGUE_ID_TO_CODE.get(fx["league"]["id"])
                if league is not None and fx["fixture"]["status"]["short"] == "FT":
                    row = _fallback_row(fx, league, day, lookup, already_have, season_label, client)
                    if row is not None:
                        rows.append(row)
            day += timedelta(days=1)
    except _QuotaExhausted as exc:
        log.info("Results fallback stopped early on %s: %s", day, exc)

    log.info("Results fallback checked %s to %s: %d new match(es)", start, today, len(rows))
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.dropna(subset=["HC", "AC", "HS", "AS", "HST", "AST"])
    if df.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    return add_derived_columns(df)[RESULT_COLUMNS]


def _check_quota(client: ApiFootballClient) -> None:
    remaining = client.requests_remaining
    if remaining is not None and remaining <= RESULTS_FALLBACK_QUOTA_RESERVE:
        raise _QuotaExhausted(f"{remaining} requests left, kept in reserve for odds")


def _fallback_row(
    fx: dict[str, Any],
    league: str,
    day: date,
    lookup: dict[str, dict[str, str]],
    already_have: set[tuple],
    season_label: str,
    client: ApiFootballClient,
) -> dict[str, Any] | None:
    """One result row from API-Football, or None if it is stored or not ready yet."""
    home_api, away_api = fx["teams"]["home"]["name"], fx["teams"]["away"]["name"]
    home = match_team(home_api, league, lookup)
    away = match_team(away_api, league, lookup)
    if home is None:
        log_unmatched_team(home_api, league, opponent=away_api, fixture_date=str(day))
    if away is None:
        log_unmatched_team(away_api, league, opponent=home_api, fixture_date=str(day))
    home, away = home or home_api, away or away_api
    league_name = LEAGUES[league][0]
    if (day, home, away, league_name) in already_have:
        return None

    _check_quota(client)
    try:
        stats = client.get("fixtures/statistics", {"fixture": fx["fixture"]["id"]})
    except (requests.RequestException, RuntimeError) as exc:
        raise _QuotaExhausted(str(exc)) from exc

    home_stats = next((t for t in stats if t["team"]["name"] == home_api), None)
    away_stats = next((t for t in stats if t["team"]["name"] == away_api), None)
    fthg, ftag = fx["score"]["fulltime"]["home"], fx["score"]["fulltime"]["away"]
    if home_stats is None or away_stats is None or fthg is None or ftag is None:
        return None  # statistics not published yet; retried on a later run

    hthg, htag = fx["score"]["halftime"]["home"], fx["score"]["halftime"]["away"]
    row: dict[str, Any] = {
        "Date": pd.Timestamp(day),
        "HomeTeam": home,
        "AwayTeam": away,
        "FTHG": fthg,
        "FTAG": ftag,
        "FTR": _result_label(fthg, ftag),
        "HTHG": hthg,
        "HTAG": htag,
        "HTR": _result_label(hthg, htag),
        "League": league_name,
        "Season": season_label,
        "Avg>2.5": np.nan,
        "Avg<2.5": np.nan,
    }
    for api_name, suffix in _API_STATS.items():
        row[f"H{suffix}"] = _stat(home_stats, api_name)
        row[f"A{suffix}"] = _stat(away_stats, api_name)
    return row


def update_results(client: ApiFootballClient | None = None, full_rebuild: bool = False) -> pd.DataFrame:
    """Refresh the raw results file and return it.

    Incremental runs re-download the whole current season (one CSV per
    league) rather than only rows after the latest stored date, because the
    fallback can advance that date with a partial day.
    """
    existing = None
    if RAW_RESULTS_FILE.exists() and not full_rebuild:
        existing = pd.read_csv(RAW_RESULTS_FILE, parse_dates=["Date"], low_memory=False)
        log.info("Stored results: %d matches up to %s", len(existing), existing["Date"].max().date())

    seasons = season_codes() if existing is None else [current_season_code()]
    fresh = download_football_data(seasons)

    if fresh.empty:
        log.info("No data returned by football-data.co.uk")
        results = existing
    elif existing is None:
        log.info("Full rebuild: %d matches", len(fresh))
        results = fresh
    else:
        results = _merge_canonical(existing, fresh)

    if existing is not None and results is not None and client is not None:
        fallback = fetch_recent_via_api(results, client)
        if not fallback.empty:
            log.info("Added %d match(es) from API-Football", len(fallback))
            results = pd.concat([results, fallback], ignore_index=True)

    if results is None or results.empty:
        raise RuntimeError("No match results available from any source")

    results = (
        results.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam", "League"])
        .sort_values(["Date", "League", "HomeTeam"])
        .reset_index(drop=True)
    )
    RAW_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RAW_RESULTS_FILE, index=False)
    log.info("Saved %d matches to %s", len(results), RAW_RESULTS_FILE.name)
    return results
