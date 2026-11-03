"""Project-wide paths and tunable settings.

Everything a user might want to change lives here; modules import from this
file rather than hard-coding values. The API key is read from the
environment (or a git-ignored ``.env`` file in the project root).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
LOG_DIR = PROJECT_ROOT / "logs"

RAW_RESULTS_FILE = DATA_DIR / "raw_results.csv"
FEATURES_FILE = DATA_DIR / "features.csv"
TEAM_FORM_FILE = DATA_DIR / "team_form.csv"
ODDS_HISTORY_FILE = DATA_DIR / "odds_history.csv"
MODEL_FILE = DATA_DIR / "model_params.json"
DECISION_LOG_FILE = DATA_DIR / "decision_log.csv"
UNMATCHED_TEAMS_FILE = DATA_DIR / "unmatched_teams.csv"
BET_ID_CACHE_FILE = DATA_DIR / "api_bet_ids.json"

# ---------------------------------------------------------------------------
# Leagues: our code -> (football-data.co.uk name, API-Football league id)
# ---------------------------------------------------------------------------
LEAGUES: dict[str, tuple[str, int]] = {
    "E0": ("Premier League", 39),
    "SP1": ("La Liga", 140),
    "I1": ("Serie A", 135),
    "D1": ("Bundesliga", 78),
    "F1": ("Ligue 1", 61),
    "N1": ("Eredivisie", 88),
    "B1": ("Belgian Pro League", 144),
    "P1": ("Primeira Liga", 94),
    "T1": ("Super Lig", 203),
    "G1": ("Super League Greece", 197),  # not 494, which is the second division
}
LEAGUE_NAME_TO_CODE = {name: code for code, (name, _) in LEAGUES.items()}
API_LEAGUE_ID_TO_CODE = {api_id: code for code, (_, api_id) in LEAGUES.items()}

FIRST_SEASON = "1415"  # earliest season downloaded on a full rebuild

# ---------------------------------------------------------------------------
# Betting lines tracked per market
# ---------------------------------------------------------------------------
TOTAL_CORNER_LINES = [7.5, 8.5, 9.5, 10.5]
HOME_CORNER_LINES = [4.5, 5.5, 6.5]  # centred on the historical home mean (~5.4)
AWAY_CORNER_LINES = [2.5, 3.5, 4.5]  # centred on the historical away mean (~4.4)
GOALS_LINES = [2.5]  # only used for the mkt_over25_prob model feature

# ---------------------------------------------------------------------------
# API-Football
# ---------------------------------------------------------------------------
API_BASE_URL = "https://v3.football.api-sports.io"
API_REQUEST_PAUSE_SECONDS = 1.5  # stays under the free plan's per-minute limit
API_MAX_RETRIES = 5
DISCOVERY_DAYS_AHEAD = 30  # discovery stops earlier at the plan's date window
DISCOVERY_DAYS_BACK = 14
RESULTS_FALLBACK_QUOTA_RESERVE = 30  # requests kept back for odds fetching
RESULTS_FALLBACK_RESCAN_DAYS = 2

# ---------------------------------------------------------------------------
# Features and model
# ---------------------------------------------------------------------------
EWM_HALFLIFE = 25  # matches; a sweep of 10-50 peaked here
MIN_PRIOR_MATCHES = 8  # history a team needs before it gets a form row
MAX_FORM_AGE_DAYS = 120  # older form than this is treated as missing
MODEL_MAX_COUNT = 20  # corner counts above this share the top bucket
REPORT_LINES = [7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5]

# ---------------------------------------------------------------------------
# Pricing and staking
# ---------------------------------------------------------------------------
PMF_MAX_K = 30  # count support used when fitting the bookmaker's ladder
MARKET_ALPHA_PRIOR = 0.1  # dispersion used when fewer than 3 prices exist
MARKET_MAX_ALPHA = 0.5
RESULT_DATE_TOLERANCE_DAYS = 2


@dataclass(frozen=True)
class StakingConfig:
    """Rules for turning a price and a probability into a stake."""

    bankroll: float = 1000.0
    ev_threshold: float = 0.05
    model_weight: float = 0.5  # weight on the model mean vs the market mean
    kelly_fraction: float = 0.25
    max_kelly_ev: float = 0.04  # EV is capped here before Kelly sizing
    max_stake_pct: float = 0.10
    min_odds: float = 1.0
    max_odds: float = 3.0


STAKING = StakingConfig()


def current_season_code(today: date | None = None) -> str:
    """Season code in football-data.co.uk style, e.g. ``'2627'`` for 2026/27.

    Seasons are treated as starting in July.
    """
    today = today or date.today()
    start = today.year if today.month >= 7 else today.year - 1
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


def season_codes(first: str = FIRST_SEASON, last: str | None = None) -> list[str]:
    """All season codes from ``first`` to ``last`` inclusive."""
    last = last or current_season_code()
    start, end = int(first[:2]), int(last[:2])
    return [f"{y:02d}{(y + 1) % 100:02d}" for y in range(start, end + 1)]


def _load_dotenv(path: Path) -> None:
    """Minimal ``.env`` loader: KEY=VALUE lines, existing env vars win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def api_key() -> str:
    """The API-Football key from the environment or ``.env``."""
    _load_dotenv(PROJECT_ROOT / ".env")
    key = os.environ.get("API_FOOTBALL_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "API_FOOTBALL_KEY is not set. Add it to a .env file in the project "
            "root or set it as an environment variable."
        )
    return key
