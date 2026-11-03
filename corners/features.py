"""Team-form features from match results.

Each team's form is an exponentially weighted mean (EWM) of its per-match
statistics, carried continuously across seasons, leagues and promotion.
Two tables are produced:

* ``features.csv``: one row per match with both teams' form *going into* it,
  the training set for the model;
* ``team_form.csv``: one row per team per match with that team's form *after*
  it, which is what predicting the team's next fixture needs.

The two agree by construction: a team's form after match n is exactly its
form going into match n + 1. Live predictions read the second table so they
include each team's most recent result.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

import numpy as np
import pandas as pd

from corners.config import (
    EWM_HALFLIFE,
    FEATURES_FILE,
    LEAGUE_NAME_TO_CODE,
    MAX_FORM_AGE_DAYS,
    MIN_PRIOR_MATCHES,
    RAW_RESULTS_FILE,
    TEAM_FORM_FILE,
)

log = logging.getLogger(__name__)

TEAM_STATS = [
    "corners_for",
    "corners_against",
    "shots_for",
    "shots_against",
    "sot_for",
    "sot_against",
    "offtarget_for",
    "offtarget_against",
    "goals_for",
    "goals_against",
    "fouls_for",
    "fouls_against",
    "trailing_ht",
]
DERIVED_STATS = ["corners_per_shot", "shot_accuracy", "shot_diff"]
FORM_COLUMNS = [f"roll_{stat}" for stat in TEAM_STATS + DERIVED_STATS]

_MATCH_COLUMNS = [
    "Date",
    "League",
    "Season",
    "HomeTeam",
    "AwayTeam",
    "HC",
    "AC",
    "HS",
    "AS",
    "HST",
    "AST",
    "FTHG",
    "FTAG",
    "has_red_card",
    "Avg>2.5",
    "Avg<2.5",
]
_SEASON_LONG = re.compile(r"^(\d{4})/(\d{4})$")


def normalise_season(season: object) -> object:
    """``'2026/2027'`` -> ``'2627'``; short codes pass through unchanged."""
    match = _SEASON_LONG.match(str(season))
    return match.group(1)[2:] + match.group(2)[2:] if match else season


def clean_results(raw: pd.DataFrame) -> pd.DataFrame:
    """Results with league codes, short season codes and a stable match id."""
    df = raw.copy()
    df["League"] = df["League"].replace(LEAGUE_NAME_TO_CODE)
    df["Season"] = df["Season"].apply(normalise_season)
    df = df.dropna(subset=["HC", "AC", "HS", "AS", "HST", "AST"])
    df = df.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    df["match_id"] = np.arange(len(df))
    return df


def _team_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Two rows per match, one from each team's point of view."""
    sides = []
    for is_home, us, them in ((True, "H", "A"), (False, "A", "H")):
        team, opponent = ("HomeTeam", "AwayTeam") if is_home else ("AwayTeam", "HomeTeam")
        goals_us, goals_them = ("FTHG", "FTAG") if is_home else ("FTAG", "FTHG")
        sides.append(
            pd.DataFrame(
                {
                    "Date": df["Date"],
                    "League": df["League"],
                    "Season": df["Season"],
                    "match_id": df["match_id"],
                    "club": df[team],
                    "opponent": df[opponent],
                    "is_home": is_home,
                    "corners_for": df[f"{us}C"],
                    "corners_against": df[f"{them}C"],
                    "shots_for": df[f"{us}S"],
                    "shots_against": df[f"{them}S"],
                    "sot_for": df[f"{us}ST"],
                    "sot_against": df[f"{them}ST"],
                    "offtarget_for": df[f"{us}S"] - df[f"{us}ST"],
                    "offtarget_against": df[f"{them}S"] - df[f"{them}ST"],
                    "goals_for": df[goals_us],
                    "goals_against": df[goals_them],
                    "fouls_for": df[f"{us}F"],
                    "fouls_against": df[f"{them}F"],
                    "trailing_ht": df["home_trailing_ht" if is_home else "away_trailing_ht"],
                }
            )
        )
    rows = pd.concat(sides, ignore_index=True)
    return rows.sort_values(["club", "Date", "match_id"]).reset_index(drop=True)


def _add_form(rows: pd.DataFrame, include_current: bool) -> pd.DataFrame:
    """Add EWM form columns and the number of matches they cover.

    With ``include_current=False`` each row's form excludes its own match
    (pre-match form); with True it includes it (post-match form).
    """
    shift: Callable[[pd.Series], pd.Series] = (lambda s: s) if include_current else (lambda s: s.shift(1))
    grouped = rows.groupby("club", sort=False)
    rolled = grouped[TEAM_STATS].transform(
        lambda s: shift(s).ewm(halflife=EWM_HALFLIFE, min_periods=MIN_PRIOR_MATCHES).mean()
    )
    out = rows.copy()
    for stat in TEAM_STATS:
        out[f"roll_{stat}"] = rolled[stat]
    out["played"] = (
        grouped["corners_for"].transform(lambda s: shift(s).expanding(min_periods=1).count()).fillna(0)
    )
    shots = out["roll_shots_for"].replace(0, np.nan)
    out["roll_corners_per_shot"] = out["roll_corners_for"] / shots
    out["roll_shot_accuracy"] = out["roll_sot_for"] / shots
    out["roll_shot_diff"] = out["roll_shots_for"] - out["roll_shots_against"]
    return out[out["played"] >= MIN_PRIOR_MATCHES].dropna(subset=FORM_COLUMNS)


def build_match_features(matches: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """One row per match where both teams have enough history."""
    pre = _add_form(rows, include_current=False)
    home = pre[pre["is_home"]].set_index("match_id")
    away = pre[~pre["is_home"]].set_index("match_id")
    both = home.index.intersection(away.index)

    out = matches.set_index("match_id").loc[both, _MATCH_COLUMNS].copy()
    for column in FORM_COLUMNS:
        out[f"h_{column}"] = home.loc[both, column]
        out[f"a_{column}"] = away.loc[both, column]
    out["total_corners"] = out["HC"] + out["AC"]

    inv_over, inv_under = 1.0 / out["Avg>2.5"], 1.0 / out["Avg<2.5"]
    out["mkt_over25_prob"] = inv_over / (inv_over + inv_under)  # NaN before 2019/20
    out = out.drop(columns=["Avg>2.5", "Avg<2.5"])
    return out.reset_index().sort_values(["Date", "match_id"]).reset_index(drop=True)


def build_team_form(rows: pd.DataFrame) -> pd.DataFrame:
    """Each team's form after each of its matches."""
    post = _add_form(rows, include_current=True)
    return post[["Date", "League", "Season", "club", "played", *FORM_COLUMNS]].reset_index(drop=True)


def build_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild and save both feature tables from the raw results file."""
    if not RAW_RESULTS_FILE.exists():
        raise FileNotFoundError(f"{RAW_RESULTS_FILE} is missing; fetch results first")
    matches = clean_results(pd.read_csv(RAW_RESULTS_FILE, parse_dates=["Date"], low_memory=False))
    rows = _team_rows(matches)
    features = build_match_features(matches, rows)
    team_form = build_team_form(rows)

    features.to_csv(FEATURES_FILE, index=False)
    team_form.to_csv(TEAM_FORM_FILE, index=False)
    log.info(
        "Features: %d matches (%s to %s); %d of them have form for both teams",
        len(matches),
        matches["Date"].min().date(),
        matches["Date"].max().date(),
        len(features),
    )
    log.info(
        "Corners per match: home mean %.2f, away mean %.2f, total variance/mean %.2f "
        "(above 1 means overdispersed, hence the negative-binomial model)",
        features["HC"].mean(),
        features["AC"].mean(),
        features["total_corners"].var() / features["total_corners"].mean(),
    )
    return features, team_form


def load_features() -> pd.DataFrame:
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(f"{FEATURES_FILE} is missing; build features first")
    df = pd.read_csv(FEATURES_FILE, parse_dates=["Date"], dtype={"Season": "string"}, low_memory=False)
    return df.sort_values(["Date", "match_id"]).reset_index(drop=True)


def load_team_form() -> pd.DataFrame:
    if not TEAM_FORM_FILE.exists():
        raise FileNotFoundError(f"{TEAM_FORM_FILE} is missing; build features first")
    return pd.read_csv(TEAM_FORM_FILE, parse_dates=["Date"], low_memory=False)


class FormBook:
    """Fast lookup of a team's latest form before a given date."""

    def __init__(self, team_form: pd.DataFrame) -> None:
        self._by_team: dict[tuple[str, str], tuple[np.ndarray, pd.DataFrame]] = {}
        for (league, club), group in team_form.groupby(["League", "club"], sort=False):
            group = group.sort_values("Date")
            self._by_team[(league, club)] = (group["Date"].to_numpy(), group.reset_index(drop=True))

    def before(self, league: str, team: str, when: pd.Timestamp) -> pd.Series | None:
        """Form after the team's last match on an earlier calendar day than ``when``.

        Returns None if there is none, or if it is older than
        ``MAX_FORM_AGE_DAYS`` (stale form is treated as missing).
        """
        entry = self._by_team.get((league, team))
        if entry is None:
            return None
        dates, rows = entry
        day = pd.Timestamp(when).normalize().to_datetime64()
        idx = int(np.searchsorted(dates, day, side="left")) - 1
        if idx < 0:
            return None
        row = rows.iloc[idx]
        if (pd.Timestamp(when) - row["Date"]).days > MAX_FORM_AGE_DAYS:
            return None
        return row
