"""Shared synthetic data for the tests: no network or real data files needed."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from corners.config import LEAGUES
from corners.data.odds_history import ODDS_COLUMNS, odds_column
from corners.data.results import RESULT_COLUMNS, add_derived_columns
from corners.features import FormBook, _team_rows, build_match_features, build_team_form, clean_results
from corners.model import TrainedModel, complete_rows, fit_models
from corners.pricing import p_over
from corners.staking import MARKETS, FixtureEvaluator
from corners.teams import build_team_lookup
from corners.tracking import ResultBook


def synthetic_results(
    seasons: tuple[str, ...] = ("2122", "2223", "2324", "2425"),
    leagues: tuple[str, ...] = ("E0", "SP1"),
    teams_per_league: int = 8,
    seed: int = 0,
) -> pd.DataFrame:
    """Raw results in football-data.co.uk shape: double round robins with
    team-specific corner rates, so the model has real signal to find."""
    rng = np.random.default_rng(seed)
    rows = []
    for league in leagues:
        teams = [f"{league} Club {chr(65 + i)}" for i in range(teams_per_league)]
        corner_rate = dict(zip(teams, rng.uniform(3.5, 6.5, len(teams)), strict=True))
        pairs = list(permutations(teams, 2))
        for season in seasons:
            start = pd.Timestamp(f"20{season[:2]}-08-01")
            # Spread fixtures so each team plays at most once per date.
            schedule: dict[int, set[str]] = {}
            for home, away in pairs:
                day = 0
                while home in schedule.get(day, set()) or away in schedule.get(day, set()):
                    day += 1
                schedule.setdefault(day, set()).update({home, away})
                hs, as_ = rng.poisson(13), rng.poisson(11)
                fthg, ftag = rng.poisson(1.5), rng.poisson(1.1)
                over = rng.uniform(1.6, 2.4)
                rows.append(
                    {
                        "Date": start + pd.Timedelta(days=7 * day),
                        "HomeTeam": home,
                        "AwayTeam": away,
                        "FTHG": fthg,
                        "FTAG": ftag,
                        "FTR": "H" if fthg > ftag else "A" if ftag > fthg else "D",
                        "HTHG": min(fthg, rng.poisson(0.7)),
                        "HTAG": min(ftag, rng.poisson(0.5)),
                        "HTR": "D",
                        "HC": rng.poisson(corner_rate[home] * 1.1),
                        "AC": rng.poisson(corner_rate[away] * 0.9),
                        "HS": hs,
                        "AS": as_,
                        "HST": rng.binomial(hs, 0.35),
                        "AST": rng.binomial(as_, 0.35),
                        "HF": rng.poisson(11),
                        "AF": rng.poisson(11),
                        "HY": rng.poisson(2),
                        "AY": rng.poisson(2),
                        "HR": int(rng.random() < 0.05),
                        "AR": int(rng.random() < 0.05),
                        "League": LEAGUES[league][0],
                        "Season": f"20{season[:2]}/20{season[2:]}",
                        "Avg>2.5": over,
                        "Avg<2.5": 1 / (1.05 - 1 / over),
                    }
                )
    df = add_derived_columns(pd.DataFrame(rows))
    return df[RESULT_COLUMNS].sort_values(["Date", "League", "HomeTeam"]).reset_index(drop=True)


@pytest.fixture(scope="session")
def raw_results() -> pd.DataFrame:
    return synthetic_results()


# ---------------------------------------------------------------------------
# Keeping tests away from real data
# ---------------------------------------------------------------------------
# Modules import these constants by name, so each module's own copy has to be
# redirected, not just the one in corners.config.
PATH_CONSTANTS = {
    "PROJECT_ROOT": "",
    "DATA_DIR": "data",
    "RESULTS_DIR": "results",
    "LOG_DIR": "logs",
    "RAW_RESULTS_FILE": "data/raw_results.csv",
    "FEATURES_FILE": "data/features.csv",
    "TEAM_FORM_FILE": "data/team_form.csv",
    "ODDS_HISTORY_FILE": "data/odds_history.csv",
    "MODEL_FILE": "data/model_params.json",
    "DECISION_LOG_FILE": "data/decision_log.csv",
    "UNMATCHED_TEAMS_FILE": "data/unmatched_teams.csv",
    "BET_ID_CACHE_FILE": "data/api_bet_ids.json",
    "LOCK_FILE": "data/.pipeline.lock",
}


@pytest.fixture(scope="session")
def redirect_paths():
    """A function that points every file path in the package at a folder."""

    def apply(monkeypatch, root) -> Path:
        root = Path(root)
        (root / "data").mkdir(parents=True, exist_ok=True)
        for module in list(sys.modules.values()):
            name = getattr(module, "__name__", "")
            if isinstance(name, str) and name.startswith("corners"):
                for constant, relative in PATH_CONSTANTS.items():
                    if hasattr(module, constant):
                        monkeypatch.setattr(module, constant, root / relative if relative else root)
        return root

    return apply


@pytest.fixture
def workspace(tmp_path, monkeypatch, redirect_paths) -> Path:
    """An empty project folder; the package reads and writes only here."""
    return redirect_paths(monkeypatch, tmp_path)


# ---------------------------------------------------------------------------
# A trained pipeline on synthetic data, built once per test run
# ---------------------------------------------------------------------------
@dataclass
class Trained:
    matches: pd.DataFrame
    features: pd.DataFrame
    team_form: pd.DataFrame
    params: dict
    evaluator: FixtureEvaluator
    results: ResultBook

    def test_season(self) -> pd.DataFrame:
        """Matches from the held-out season (after the model's training cut-off)."""
        return self.features[self.features["Season"] == "2425"]


@pytest.fixture(scope="session")
def trained(raw_results) -> Trained:
    matches = clean_results(raw_results)
    rows = _team_rows(matches)
    features = build_match_features(matches, rows)
    team_form = build_team_form(rows)
    params = fit_models(complete_rows(features))
    evaluator = FixtureEvaluator(
        model=TrainedModel.from_params(params),
        form=FormBook(team_form),
        lookup=build_team_lookup(features),
    )
    return Trained(matches, features, team_form, params, evaluator, ResultBook(matches))


@pytest.fixture(scope="session")
def make_odds_row():
    """Build an odds-history row for a match, priced from a negative binomial.

    ``mu`` is the mean the early prices imply (the model's own mean is higher,
    so a low ``mu`` makes the Over look like good value). Pass ``closing_mu``
    to add a closing snapshot too.
    """

    def build(match, fixture_id=1, mu=9.5, closing_mu=None, alpha=0.05, hours=15) -> pd.Series:
        row = {column: np.nan for column in ODDS_COLUMNS} | {
            "fixture_id": fixture_id,
            "league": match["League"],
            "start_datetime": pd.Timestamp(match["Date"]).tz_localize("UTC") + pd.Timedelta(hours=hours),
            "home_team": match["HomeTeam"],
            "away_team": match["AwayTeam"],
            "early_fetched_at": pd.Timestamp("2000-01-01", tz="UTC"),
        }
        stages = [("early", mu)]
        if closing_mu is not None:
            stages.append(("closing", closing_mu))
            row["closing_fetched_at"] = pd.Timestamp("2000-01-02", tz="UTC")
        for stage, stage_mu in stages:
            for spec, mean in (
                (MARKETS["total"], stage_mu),
                (MARKETS["home"], stage_mu * 0.55),
                (MARKETS["away"], stage_mu * 0.45),
            ):
                for line in spec.lines:
                    p = p_over(mean, alpha, line)
                    row[odds_column(stage, "over", line, spec.odds_key)] = 1 / (p * 1.05)
                    row[odds_column(stage, "under", line, spec.odds_key)] = 1 / ((1 - p) * 1.05)
            row[odds_column(stage, "over", 2.5, "goals")] = 1.9
            row[odds_column(stage, "under", 2.5, "goals")] = 1.9
        return pd.Series(row)

    return build


@pytest.fixture(scope="session")
def make_history():
    """Turn odds-history rows into a DataFrame with proper timestamp columns."""

    def build(rows) -> pd.DataFrame:
        history = pd.DataFrame(rows).reset_index(drop=True)
        for column in ("start_datetime", "early_fetched_at", "closing_fetched_at"):
            history[column] = pd.to_datetime(history[column], utc=True)
        return history

    return build
