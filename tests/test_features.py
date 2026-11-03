import numpy as np
import pandas as pd
import pytest

from corners.config import MIN_PRIOR_MATCHES
from corners.features import (
    FORM_COLUMNS,
    FormBook,
    _team_rows,
    build_match_features,
    build_team_form,
    clean_results,
    normalise_season,
)


@pytest.fixture(scope="module")
def tables(raw_results):
    matches = clean_results(raw_results)
    rows = _team_rows(matches)
    return matches, build_match_features(matches, rows), build_team_form(rows)


def test_normalise_season():
    assert normalise_season("2026/2027") == "2627"
    assert normalise_season("1415") == "1415"


def test_pre_match_form_excludes_the_match_itself(tables):
    matches, features, _ = tables
    row = features.iloc[-1]
    team = row["HomeTeam"]
    history = matches[
        ((matches["HomeTeam"] == team) | (matches["AwayTeam"] == team)) & (matches["Date"] < row["Date"])
    ]
    corners = np.where(history["HomeTeam"] == team, history["HC"], history["AC"])
    expected = pd.Series(corners, dtype=float).ewm(halflife=25, min_periods=MIN_PRIOR_MATCHES).mean().iloc[-1]
    assert row["h_roll_corners_for"] == pytest.approx(expected)


def test_post_match_form_equals_next_pre_match_form(tables):
    _, features, team_form = tables
    book = FormBook(team_form)
    for _, row in features.tail(20).iterrows():
        form = book.before(row["League"], row["HomeTeam"], row["Date"] + pd.Timedelta(hours=15))
        for column in FORM_COLUMNS:
            assert form[column] == pytest.approx(row[f"h_{column}"])


def test_form_book_ignores_same_day_and_stale_form(tables):
    _, _, team_form = tables
    book = FormBook(team_form)
    last = team_form.sort_values("Date").iloc[-1]
    same_day = book.before(last["League"], last["club"], last["Date"])
    assert same_day is None or same_day["Date"] < last["Date"]
    assert book.before(last["League"], last["club"], last["Date"] + pd.Timedelta(days=400)) is None
    assert book.before(last["League"], "Unknown FC", last["Date"]) is None


def test_every_feature_row_has_complete_form_for_both_teams(tables):
    _, features, _ = tables
    columns = [f"{side}_{c}" for side in "ha" for c in FORM_COLUMNS]
    assert not features[columns].isna().any().any()
    assert (features["total_corners"] == features["HC"] + features["AC"]).all()
