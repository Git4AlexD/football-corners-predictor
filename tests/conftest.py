"""Shared synthetic data for the tests: no network or real data files needed."""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pandas as pd
import pytest

from corners.config import LEAGUES
from corners.data.results import RESULT_COLUMNS, add_derived_columns


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
