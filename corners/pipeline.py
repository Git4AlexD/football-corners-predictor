"""The daily pipeline and its building blocks.

Run order matters: odds are fetched first because a missed odds snapshot
can never be recovered (the free API plan has no historical odds), while a
missed result can be picked up on any later run. Each step that fails falls
back to the last good output on disk when one exists.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from corners.config import (
    DATA_DIR,
    FEATURES_FILE,
    LEAGUE_NAME_TO_CODE,
    MODEL_FILE,
    ODDS_HISTORY_FILE,
    PROJECT_ROOT,
    RAW_RESULTS_FILE,
    UNMATCHED_TEAMS_FILE,
)
from corners.data.api_football import ApiFootballClient
from corners.data.odds_history import capture_odds, load_odds_history
from corners.data.results import update_results
from corners.decision_log import grade_decision_log, log_upcoming_decisions
from corners.evaluation import evaluate_clv
from corners.features import FormBook, build_features, load_features, load_team_form
from corners.model import TrainedModel, needs_retraining, train_models
from corners.staking import FixtureEvaluator
from corners.teams import build_team_lookup, prune_resolved, unmatched_teams
from corners.tracking import ResultBook, track

log = logging.getLogger(__name__)

LOCK_FILE = DATA_DIR / ".pipeline.lock"
STALE_LOCK_SECONDS = 3 * 60 * 60  # a lock this old was left by a crashed run


@dataclass(frozen=True)
class Step:
    name: str
    run: Callable[[], object]
    output: Path | None = None  # existing output that makes a failure survivable
    fatal: bool = True


def build_evaluator() -> FixtureEvaluator:
    """An evaluator over the current model, form table and team names."""
    raw_teams = pd.read_csv(RAW_RESULTS_FILE, usecols=["League", "HomeTeam", "AwayTeam"], low_memory=False)
    raw_teams["League"] = raw_teams["League"].replace(LEAGUE_NAME_TO_CODE)
    return FixtureEvaluator(
        model=TrainedModel.load(),
        form=FormBook(load_team_form()),
        lookup=build_team_lookup(load_features()),
        broad_lookup=build_team_lookup(raw_teams),
    )


def train_if_needed(force: bool = False) -> None:
    if force or needs_retraining():
        train_models()
    else:
        model = TrainedModel.load()
        log.info(
            "Training data unchanged since the model was trained on %s; not retraining",
            model.params["trained_at"][:10],
        )


def log_decisions() -> None:
    log_upcoming_decisions(build_evaluator(), load_odds_history())


def track_results() -> None:
    history = load_odds_history()
    if history.empty:
        raise RuntimeError(f"{ODDS_HISTORY_FILE} is empty; fetch odds first")
    results = ResultBook.load()
    track(build_evaluator(), history, results)
    grade_decision_log(results)


def evaluate_closing_line_value() -> None:
    evaluate_clv(build_evaluator(), load_odds_history(), ResultBook.load())


def offline_steps(force_retrain: bool = False) -> list[Step]:
    """Steps that only use data already on disk."""
    return [
        Step("Build features", build_features, FEATURES_FILE),
        Step("Train model if needed", lambda: train_if_needed(force_retrain), MODEL_FILE),
        Step("Log upcoming decisions", log_decisions, fatal=False),
        Step("Back-test and grade", track_results, fatal=False),
        Step("Evaluate profitability at closing prices", evaluate_closing_line_value, fatal=False),
    ]


def report_steps() -> list[Step]:
    """Refresh the results from stored data without changing it.

    For a copy of the data kept up to date elsewhere (e.g. by the scheduled
    GitHub workflow): no API calls, no training and no decision logging, so
    nothing in the stored data is modified.
    """
    return [
        Step("Build features", build_features, FEATURES_FILE),
        Step("Back-test and grade", track_results, fatal=False),
        Step("Evaluate profitability at closing prices", evaluate_closing_line_value, fatal=False),
    ]


def daily_steps(force_retrain: bool = False) -> list[Step]:
    """The full daily run, starting with the API calls."""
    client: ApiFootballClient | None = None

    def api() -> ApiFootballClient:
        nonlocal client
        client = client or ApiFootballClient()
        return client

    return [
        Step("Capture odds", lambda: capture_odds(api()), ODDS_HISTORY_FILE, fatal=False),
        Step("Update results", lambda: update_results(api()), RAW_RESULTS_FILE),
        *offline_steps(force_retrain),
    ]


@contextmanager
def _exclusive_run() -> Iterator[bool]:
    """Hold a lock file for the duration of a run; yields False if another run holds it.

    Two runs at once would write the same files, including the odds history,
    which cannot be rebuilt if it is corrupted.
    """
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    if LOCK_FILE.exists() and time.time() - LOCK_FILE.stat().st_mtime > STALE_LOCK_SECONDS:
        LOCK_FILE.unlink(missing_ok=True)
    try:
        handle = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        yield False
        return
    os.write(handle, str(os.getpid()).encode())
    os.close(handle)
    try:
        yield True
    finally:
        LOCK_FILE.unlink(missing_ok=True)


def run_steps(steps: list[Step]) -> int:
    """Run steps in order and return a process exit code."""
    with _exclusive_run() as acquired:
        if not acquired:
            log.error(
                "Another pipeline run is already in progress (%s exists); not starting.", LOCK_FILE.name
            )
            return 1
        return _run_steps(steps)


def _run_steps(steps: list[Step]) -> int:
    started = time.monotonic()
    outcomes: list[tuple[str, str]] = []
    exit_code = 0
    for number, step in enumerate(steps, 1):
        log.info("Step %d/%d: %s", number, len(steps), step.name)
        try:
            step.run()
            outcomes.append((step.name, "ok"))
            continue
        except Exception:
            log.exception("%s failed", step.name)
        if step.output is not None and step.output.exists():
            outcomes.append((step.name, "failed, using existing data"))
        elif step.fatal:
            outcomes.append((step.name, "failed, stopping"))
            exit_code = 1
            break
        else:
            outcomes.append((step.name, "failed"))

    unmatched = _current_unmatched_teams()
    lines = [f"    {status:<28} {name}" for name, status in outcomes]
    lines.append(f"    Unmatched team names: {len(unmatched) or 'none'}")
    log.info("Pipeline summary (%.0fs)\n%s", time.monotonic() - started, "\n".join(lines))
    if not unmatched.empty:
        _flag_unmatched_teams(unmatched)
    return exit_code


def _current_unmatched_teams() -> pd.DataFrame:
    """Logged unmatched names, minus any that an alias added since now resolves."""
    try:
        return prune_resolved(build_team_lookup(load_features()))
    except FileNotFoundError:
        return unmatched_teams()


def _flag_unmatched_teams(teams: pd.DataFrame) -> None:
    """Make unmatched names impossible to miss: a banner, plus a GitHub annotation."""
    names = "\n".join(
        f"    {row.unmatched_name!r} ({row.league}), seen {row.times_seen}x, "
        f"e.g. v {row.example_opponent!r} on {row.example_date}"
        for row in teams.itertuples()
    )
    rule = "!" * 78
    log.warning(
        "\n%s\nACTION NEEDED: %d team name(s) could not be matched, so their fixtures are skipped.\n%s\n"
        "Fix: add each to TEAM_ALIASES in corners/teams.py, then run `python -m corners offline`.\n"
        "List: %s\n%s",
        rule, len(teams), names, UNMATCHED_TEAMS_FILE.relative_to(PROJECT_ROOT), rule,
    )  # fmt: skip
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(
            f"::warning title=Unmatched team names::{len(teams)} team name(s) need an alias in TEAM_ALIASES"
        )
