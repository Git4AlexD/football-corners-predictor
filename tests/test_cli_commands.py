"""Command-line commands and the file entry point."""

import runpy
import sys
from pathlib import Path

import pytest

import corners
from corners import cli, pipeline


@pytest.fixture(autouse=True)
def quiet_logging(monkeypatch):
    monkeypatch.setattr(cli, "configure_logging", lambda *args, **kwargs: None)


@pytest.fixture
def three_fixtures(trained, make_odds_row, make_history):
    matches = trained.test_season().iloc[-30::10]
    rows = [make_odds_row(m, fixture_id=500 + i, mu=7.5) for i, (_, m) in enumerate(matches.iterrows())]
    return make_history(rows)


def test_stakes_lists_each_market_for_the_requested_dates(monkeypatch, capsys, trained, three_fixtures):
    monkeypatch.setattr(cli, "load_odds_history", lambda: three_fixtures)
    monkeypatch.setattr(pipeline, "build_evaluator", lambda: trained.evaluator)
    kickoffs = three_fixtures["start_datetime"]

    cli.main(["stakes", kickoffs.min().strftime("%Y-%m-%d"), kickoffs.max().strftime("%Y-%m-%d")])

    output = capsys.readouterr().out
    for label in ("Total corners", "Home corners", "Away corners"):
        assert label in output
    assert "bet(s)" in output
    assert " v " in output  # fixtures are listed


def test_stakes_for_a_date_with_no_fixtures(monkeypatch, capsys, three_fixtures):
    monkeypatch.setattr(cli, "load_odds_history", lambda: three_fixtures)
    cli.main(["stakes", "2030-01-01"])
    assert "No fixtures in the odds history" in capsys.readouterr().out


def test_each_command_calls_the_right_function(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "ApiFootballClient", lambda: "client")
    monkeypatch.setattr(cli, "capture_odds", lambda client: calls.append(("odds", client)))
    monkeypatch.setattr(
        cli, "update_results", lambda client, full_rebuild: calls.append(("results", client, full_rebuild))
    )
    monkeypatch.setattr(cli, "build_features", lambda: calls.append(("features",)))
    monkeypatch.setattr(pipeline, "train_if_needed", lambda force=False: calls.append(("train", force)))
    monkeypatch.setattr(pipeline, "track_results", lambda: calls.append(("track",)))
    monkeypatch.setattr(pipeline, "evaluate_closing_line_value", lambda: calls.append(("clv",)))

    for command in (
        ["odds"],
        ["results"],
        ["results", "--full"],
        ["features"],
        ["train"],
        ["train", "--force"],
    ):
        assert cli.main(command) == 0
    for command in (["track"], ["-v", "clv"]):
        assert cli.main(command) == 0

    assert calls == [
        ("odds", "client"),
        ("results", "client", False),
        ("results", "client", True),
        ("features",),
        ("train", False),
        ("train", True),
        ("track",),
        ("clv",),
    ]


def test_running_the_file_directly_shows_the_help(monkeypatch, capsys):
    script = Path(corners.__file__).parent / "__main__.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--help"])
    monkeypatch.setattr(sys, "path", list(sys.path))  # the script adds the project root to it
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")  # the script sets this; keep it from leaking

    with pytest.raises(SystemExit) as stop:
        runpy.run_path(str(script), run_name="__main__")

    assert stop.value.code == 0
    assert "usage: corners" in capsys.readouterr().out


def test_the_charts_command_reports_whether_anything_was_published(monkeypatch, tmp_path):
    copied = []
    monkeypatch.setattr(
        cli, "copy_publishable_charts", lambda destination: copied.append(destination) or ["a.png"]
    )
    assert cli.main(["charts", str(tmp_path / "out")]) == 0
    assert copied == [tmp_path / "out"]

    monkeypatch.setattr(cli, "copy_publishable_charts", lambda destination: [])
    assert cli.main(["charts", str(tmp_path / "out")]) == 1  # nothing to publish is a failure
