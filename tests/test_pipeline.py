"""The pipeline end to end, run in a throwaway project folder on synthetic data."""

import logging
import os
import time

import pytest

from corners import pipeline
from corners.decisions import FixtureEvaluator
from corners.features import build_features
from corners.model import needs_retraining, train_models
from corners.pipeline import Step, run_steps
from corners.teams import log_unmatched_team


@pytest.fixture(scope="module")
def project(raw_results, redirect_paths, tmp_path_factory):
    """A project folder with results, features and a trained model already in place."""
    root = tmp_path_factory.mktemp("project")
    with pytest.MonkeyPatch.context() as patch:
        redirect_paths(patch, root)
        raw_results.to_csv(root / "data" / "raw_results.csv", index=False)
        build_features()
        train_models()
        yield root


@pytest.fixture
def price_history(project, trained, make_price_row, make_history):
    """An price history for test-season fixtures, saved where the pipeline looks for it."""
    matches = trained.test_season().iloc[::12].head(10)
    rows = [
        make_price_row(m, fixture_id=400 + i, mu=7.5, closing_mu=7.0)
        for i, (_, m) in enumerate(matches.iterrows())
    ]
    path = project / "data" / "odds_history.csv"
    make_history(rows).to_csv(path, index=False)
    yield path
    path.unlink()


def boom():
    raise RuntimeError("step failed")


# ---------------------------------------------------------------------------
# Training only when needed
# ---------------------------------------------------------------------------
def test_the_model_is_retrained_only_when_the_training_data_changes(project, caplog):
    caplog.set_level(logging.INFO)
    assert not needs_retraining()

    pipeline.train_if_needed()
    assert "not retraining" in caplog.text

    caplog.clear()
    pipeline.train_if_needed(force=True)
    assert "Training on" in caplog.text


def test_the_evaluator_is_built_from_the_stored_files(project):
    evaluator = pipeline.build_evaluator()
    assert isinstance(evaluator, FixtureEvaluator)
    assert evaluator.broad_lookup is not None


# ---------------------------------------------------------------------------
# Reporting steps
# ---------------------------------------------------------------------------
def test_tracking_needs_a_price_history(project):
    with pytest.raises(RuntimeError, match="fetch prices first"):
        pipeline.track_results()


def test_tracking_writes_the_back_test(project, price_history):
    pipeline.track_results()
    for mode in ("flat", "compounding"):
        assert (project / "results" / mode / "decisions.csv").exists()


def test_the_closing_price_evaluation_writes_its_summary(project, price_history):
    pipeline.evaluate_against_closing_prices()
    assert (project / "results" / "evaluation" / "summary.csv").exists()


def test_logging_decisions_skips_fixtures_that_have_already_started(project, price_history):
    pipeline.log_decisions()  # every fixture in the history is in the past
    assert not (project / "data" / "decision_log.csv").exists()


def test_the_offline_and_report_runs_complete(project, price_history):
    assert run_steps(pipeline.offline_steps()) == 0
    assert run_steps(pipeline.report_steps()) == 0


def test_the_steps_run_in_the_documented_order():
    offline = [step.name for step in pipeline.offline_steps()]
    assert offline == [
        "Build features",
        "Train model if needed",
        "Log upcoming decisions",
        "Back-test and grade",
        "Evaluate against closing prices",
    ]
    daily = [step.name for step in pipeline.daily_steps()]
    assert daily[:2] == ["Capture prices", "Update results"]  # prices first: they can't be recovered later
    assert daily[2:] == offline
    assert [step.name for step in pipeline.report_steps()] == [
        "Build features",
        "Back-test and grade",
        "Evaluate against closing prices",
    ]


# ---------------------------------------------------------------------------
# Running steps
# ---------------------------------------------------------------------------
def test_a_failed_step_with_earlier_output_does_not_stop_the_run(project, caplog):
    caplog.set_level(logging.INFO)
    ran = []
    steps = [
        Step("Flaky", boom, output=project / "data" / "raw_results.csv"),
        Step("Next", lambda: ran.append(True)),
    ]
    assert run_steps(steps) == 0
    assert ran == [True]
    assert "failed, using existing data" in caplog.text


def test_a_fatal_failure_with_nothing_to_fall_back_on_stops_the_run(project, caplog):
    caplog.set_level(logging.INFO)
    ran = []
    steps = [
        Step("Needs output", boom, output=project / "data" / "missing.csv"),
        Step("Next", lambda: ran.append(1)),
    ]
    assert run_steps(steps) == 1
    assert ran == []
    assert "failed, stopping" in caplog.text


def test_a_non_fatal_failure_is_reported_and_the_run_continues(project, caplog):
    caplog.set_level(logging.INFO)
    ran = []
    assert run_steps([Step("Optional", boom, fatal=False), Step("Next", lambda: ran.append(1))]) == 0
    assert ran == [1]
    assert "failed" in caplog.text


# ---------------------------------------------------------------------------
# Preventing overlapping runs
# ---------------------------------------------------------------------------
def test_a_fresh_lock_blocks_a_second_run(project, caplog):
    lock = pipeline.LOCK_FILE
    lock.write_text("12345")
    try:
        assert run_steps([Step("Never runs", lambda: pytest.fail("should not run"))]) == 1
        assert "already in progress" in caplog.text
        assert lock.exists()  # the refused run must not release the other run's lock
    finally:
        lock.unlink()


def test_a_lock_left_by_a_crashed_run_is_ignored(project):
    lock = pipeline.LOCK_FILE
    lock.write_text("12345")
    old = time.time() - pipeline.STALE_LOCK_SECONDS - 60
    os.utime(lock, (old, old))

    ran = []
    assert run_steps([Step("Runs anyway", lambda: ran.append(True))]) == 0
    assert ran == [True]
    assert not lock.exists()


# ---------------------------------------------------------------------------
# Unmatched team names
# ---------------------------------------------------------------------------
def test_a_clean_run_says_there_are_no_unmatched_names(project, caplog):
    caplog.set_level(logging.INFO)
    run_steps([Step("Nothing", lambda: None)])
    assert "Unmatched team names: none" in caplog.text
    assert "ACTION NEEDED" not in caplog.text


def test_unmatched_names_are_flagged_loudly(project, caplog, capsys, monkeypatch):
    caplog.set_level(logging.INFO)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    log_unmatched_team("Ottawa Wanderers", "E0", opponent="Leeds", fixture_date="2026-11-03")
    try:
        run_steps([Step("Nothing", lambda: None)])
    finally:
        (project / "data" / "unmatched_teams.csv").unlink(missing_ok=True)

    assert "Unmatched team names: 1" in caplog.text
    assert "ACTION NEEDED" in caplog.text
    assert "Ottawa Wanderers" in caplog.text
    assert "::warning" in capsys.readouterr().out  # a GitHub annotation
