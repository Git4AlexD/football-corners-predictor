import pandas as pd

from corners import cli, pipeline


def _capture_steps(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "daily_steps", lambda retrain=False: ["daily", retrain])
    monkeypatch.setattr(pipeline, "offline_steps", lambda retrain=False: ["offline", retrain])
    monkeypatch.setattr(pipeline, "report_steps", lambda: ["report"])
    monkeypatch.setattr(pipeline, "run_steps", lambda steps: calls.append(steps) or 0)
    return calls


def test_no_command_runs_the_daily_pipeline(monkeypatch):
    calls = _capture_steps(monkeypatch)
    assert cli.main([]) == 0
    assert calls == [["daily", False]]


def test_commands_and_flags_are_passed_through(monkeypatch):
    calls = _capture_steps(monkeypatch)
    cli.main(["offline", "--retrain"])
    cli.main(["report"])
    assert calls == [["offline", True], ["report"]]


def test_a_second_run_is_refused_while_one_is_in_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "LOCK_FILE", tmp_path / ".pipeline.lock")
    monkeypatch.setattr(pipeline, "_current_unmatched_teams", lambda: pd.DataFrame())
    inner = []
    step = pipeline.Step("Try a nested run", lambda: inner.append(pipeline.run_steps([])))
    assert pipeline.run_steps([step]) == 0
    assert inner == [1]  # the nested run saw the lock and refused
    assert not (tmp_path / ".pipeline.lock").exists()  # released afterwards
