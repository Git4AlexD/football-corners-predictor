import pandas as pd
import pytest

from corners import teams
from corners.teams import (
    build_team_lookup,
    log_unmatched_team,
    match_team,
    normalise_name,
    prune_resolved,
)


@pytest.fixture
def lookup():
    names = {
        "E0": ["Man United", "Leeds", "Sheffield United", "Nott'm Forest"],
        "SP1": ["Ath Bilbao", "Ath Madrid", "Real Madrid"],
        "F1": ["Le Havre", "Brest"],
        "D1": ["Leverkusen"],
    }
    rows = [{"League": lg, "HomeTeam": t, "AwayTeam": t} for lg, clubs in names.items() for t in clubs]
    return build_team_lookup(pd.DataFrame(rows))


def test_normalise_strips_accents_punctuation_and_club_words():
    assert normalise_name("Atlético Madrid CF") == "atletico madrid"
    assert normalise_name("Nott'm Forest") == "nott m forest"
    assert normalise_name("Borussia Mönchengladbach") == "borussia monchengladbach"


@pytest.mark.parametrize(
    ("league", "feed_name", "expected"),
    [
        ("E0", "Manchester United", "Man United"),  # token tie broken by similarity
        ("E0", "Leeds United", "Leeds"),  # not "Man United"
        ("E0", "Nottingham Forest", "Nott'm Forest"),
        ("SP1", "Athletic Club", "Ath Bilbao"),  # explicit alias
        ("SP1", "Atletico Madrid", "Ath Madrid"),
        ("F1", "Stade Brestois 29", "Brest"),
    ],
)
def test_match_team(lookup, league, feed_name, expected):
    assert match_team(feed_name, league, lookup) == expected


@pytest.mark.parametrize(
    ("league", "feed_name"),
    [("F1", "Le Mans"), ("D1", "Elversberg"), ("E0", "Arsenal")],
)
def test_unrelated_names_do_not_match(lookup, league, feed_name):
    assert match_team(feed_name, league, lookup) is None


def test_resolved_names_drop_off_the_unmatched_list(lookup, tmp_path, monkeypatch):
    monkeypatch.setattr(teams, "UNMATCHED_TEAMS_FILE", tmp_path / "unmatched.csv")
    log_unmatched_team("Arsenal", "E0")
    log_unmatched_team("Athletic Club", "SP1")  # now covered by an alias
    log_unmatched_team("Arsenal", "E0")
    remaining = prune_resolved(lookup)
    assert remaining["unmatched_name"].tolist() == ["Arsenal"]
    assert remaining["times_seen"].tolist() == [2]
