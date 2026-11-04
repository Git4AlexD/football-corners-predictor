"""Only whitelisted charts can be published."""

from corners.publish import PUBLISHABLE_CHARTS, copy_publishable_charts


def make_results(workspace, names):
    for name in names:
        path = workspace / "results" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")


def test_only_the_whitelisted_charts_are_copied(workspace, tmp_path):
    make_results(
        workspace,
        [*PUBLISHABLE_CHARTS, "flat/decisions.csv", "evaluation/selections.csv", "logged_decisions.csv"],
    )
    copied = copy_publishable_charts(tmp_path / "out")

    assert sorted(copied) == sorted(PUBLISHABLE_CHARTS.values())
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == sorted(PUBLISHABLE_CHARTS.values())


def test_charts_are_published_under_their_public_names(workspace, tmp_path):
    make_results(workspace, ["evaluation/expected_vs_realised_return.png"])
    assert copy_publishable_charts(tmp_path / "out") == ["expected_vs_realised_return.png"]
    assert (tmp_path / "out" / "expected_vs_realised_return.png").read_bytes() == b"png"


def test_a_chart_that_does_not_exist_yet_is_skipped(workspace, tmp_path, caplog):
    make_results(workspace, ["flat/cash_flow.png"])
    assert copy_publishable_charts(tmp_path / "out") == ["cash_flow.png"]
    assert "not produced yet" in caplog.text


def test_nothing_is_copied_when_no_results_exist(workspace, tmp_path):
    assert copy_publishable_charts(tmp_path / "out") == []
