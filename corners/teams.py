"""Matching team names between data providers.

football-data.co.uk and API-Football spell clubs differently ("Man United"
vs "Manchester United", "Ath Bilbao" vs "Athletic Club"). Names are
normalised, then matched by exact key, explicit alias, token overlap (with a
string-similarity tie-break) and finally strict fuzzy matching.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher, get_close_matches

import pandas as pd

from corners.config import UNMATCHED_TEAMS_FILE

TeamLookup = dict[str, dict[str, str]]
"""``{league_code: {normalised_name: dataset_name}}``"""

_STRIP_WORDS = {"fc", "cf", "afc", "ac", "sc", "ssd", "us", "usl", "calcio", "club"}

# Odds-feed names that no similarity heuristic can bridge, usually a
# different abbreviation convention. Keys and values are both normalised
# (see normalise_name), so check a new key with normalise_name() first.
TEAM_ALIASES = {
    "athletic": "ath bilbao",  # "Athletic Club"
    "borussia monchengladbach": "m gladbach",
    "sporting cp": "sp lisbon",  # not "Sp Braga", a different club
    "stade brestois 29": "brest",
    "vitoria": "guimaraes",  # "Vitória SC" (Guimarães), not Setúbal
    "hamburger sv": "hamburg",
    "amed": "amedspor",
    "basaksehir": "buyuksehyr",
}

# Token overlap ignores words this short: articles such as "le" otherwise
# matched "Le Mans" to "Le Havre".
_MIN_TOKEN_LENGTH = 3
_MIN_TOKEN_SCORE = 0.3
# High enough to reject coincidences such as "Elversberg" ~ "Leverkusen" (0.60)
# and "Amedspor" ~ "Samsunspor" (0.67); no genuine match in the data needs less.
_FUZZY_CUTOFF = 0.75

_UNMATCHED_COLUMNS = [
    "unmatched_name",
    "league",
    "times_seen",
    "first_seen",
    "last_seen",
    "example_opponent",
    "example_date",
]


def normalise_name(name: str) -> str:
    """Lower-case, accent-free, punctuation-free, without generic club words."""
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return " ".join(word for word in text.split() if word not in _STRIP_WORDS)


def build_team_lookup(df: pd.DataFrame) -> TeamLookup:
    """Index every team in ``df`` (columns League, HomeTeam, AwayTeam) by league."""
    lookup: TeamLookup = {}
    for league, group in df.groupby("League"):
        teams = set(group["HomeTeam"]) | set(group["AwayTeam"])
        lookup[str(league)] = {normalise_name(team): team for team in teams}
    return lookup


def _tokens(name: str) -> set[str]:
    return {token for token in name.split() if len(token) >= _MIN_TOKEN_LENGTH}


def match_team(name: str, league: str, lookup: TeamLookup) -> str | None:
    """The dataset's spelling of ``name`` in ``league``, or None if no match.

    Token overlap is tried before plain string similarity because whole-string
    similarity alone ranks "Leeds United" closer to "Man United" than to
    "Leeds". Similarity still breaks token ties, e.g. "Manchester United"
    shares one token with both "Man United" and "Sheffield United".
    """
    candidates = lookup.get(league, {})
    norm = normalise_name(name)

    if norm in candidates:
        return candidates[norm]
    if TEAM_ALIASES.get(norm) in candidates:
        return candidates[TEAM_ALIASES[norm]]

    query_tokens = _tokens(norm)
    if query_tokens:
        scored = []
        for key in candidates:
            key_tokens = _tokens(key)
            overlap = query_tokens & key_tokens
            if overlap:
                token_score = len(overlap) / len(query_tokens | key_tokens)
                scored.append((token_score, SequenceMatcher(None, norm, key).ratio(), key))
        if scored:
            best_score, _, best_key = max(scored)
            if best_score >= _MIN_TOKEN_SCORE:
                return candidates[best_key]

    close = get_close_matches(norm, candidates.keys(), n=1, cutoff=_FUZZY_CUTOFF)
    return candidates[close[0]] if close else None


def log_unmatched_team(name: str, league: str, opponent: str = "", fixture_date: str = "") -> None:
    """Record a name that matched nothing, for a human to add an alias.

    One row per (name, league); repeat sightings update the counters rather
    than appending, so the file stays a short review list.
    """
    now = pd.Timestamp.now(tz="UTC").isoformat()
    if UNMATCHED_TEAMS_FILE.exists():
        df = pd.read_csv(UNMATCHED_TEAMS_FILE)
    else:
        df = pd.DataFrame(columns=_UNMATCHED_COLUMNS)

    seen = (df["unmatched_name"] == name) & (df["league"] == league)
    if seen.any():
        idx = df.index[seen][0]
        df.loc[idx, "times_seen"] += 1
        df.loc[idx, "last_seen"] = now
    else:
        row = {
            "unmatched_name": name,
            "league": league,
            "times_seen": 1,
            "first_seen": now,
            "last_seen": now,
            "example_opponent": opponent,
            "example_date": fixture_date,
        }
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True) if len(df) else pd.DataFrame([row])

    UNMATCHED_TEAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(UNMATCHED_TEAMS_FILE, index=False)


def unmatched_teams() -> pd.DataFrame:
    """Currently logged unmatched names (empty if none)."""
    if not UNMATCHED_TEAMS_FILE.exists():
        return pd.DataFrame(columns=_UNMATCHED_COLUMNS)
    return pd.read_csv(UNMATCHED_TEAMS_FILE)


def prune_resolved(lookup: TeamLookup) -> pd.DataFrame:
    """Drop logged names that now match (e.g. after adding an alias); return the rest."""
    teams = unmatched_teams()
    if teams.empty:
        return teams
    still_unmatched = teams[
        [match_team(row.unmatched_name, row.league, lookup) is None for row in teams.itertuples()]
    ]
    if len(still_unmatched) < len(teams):
        still_unmatched.to_csv(UNMATCHED_TEAMS_FILE, index=False)
    return still_unmatched
