import dataclasses
import logging
from datetime import date

import pytest

from corners.config import DecisionConfig, api_key, current_season_code, season_codes
from corners.log import configure_logging


@pytest.mark.parametrize(
    ("today", "code"),
    [
        (date(2026, 6, 30), "2526"),  # seasons roll over on 1 July
        (date(2026, 7, 1), "2627"),
        (date(2026, 11, 3), "2627"),
        (date(2027, 1, 15), "2627"),
        (date(2027, 7, 1), "2728"),
    ],
)
def test_season_code(today, code):
    assert current_season_code(today) == code


def test_season_codes_form_a_continuous_range():
    assert season_codes("2425", "2627") == ["2425", "2526", "2627"]
    assert season_codes()[0] == "1415"
    assert season_codes()[-1] == current_season_code()


def test_the_staking_rules_cannot_be_changed_after_creation():
    with pytest.raises(dataclasses.FrozenInstanceError):
        DecisionConfig().capital = 5


@pytest.fixture
def clean_env(workspace, monkeypatch):
    """No API key in the environment, and a project folder to put a .env file in."""
    for name in ("API_FOOTBALL_KEY", "CORNERS_TEST_OTHER"):
        monkeypatch.setenv(name, "placeholder")  # registers the variable for clean-up
        monkeypatch.delenv(name)
    return workspace


def test_the_key_comes_from_the_environment(clean_env, monkeypatch):
    monkeypatch.setenv("API_FOOTBALL_KEY", "from-environment")
    assert api_key() == "from-environment"


def test_the_key_can_come_from_a_dotenv_file(clean_env):
    (clean_env / ".env").write_text(
        "# a comment\n\nCORNERS_TEST_OTHER=1\nnot a setting\nAPI_FOOTBALL_KEY='from-file'\n", encoding="utf-8"
    )
    assert api_key() == "from-file"


def test_the_environment_wins_over_the_dotenv_file(clean_env, monkeypatch):
    (clean_env / ".env").write_text("API_FOOTBALL_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("API_FOOTBALL_KEY", "from-environment")
    assert api_key() == "from-environment"


def test_a_missing_key_is_a_clear_error(clean_env):
    with pytest.raises(RuntimeError, match="API_FOOTBALL_KEY"):
        api_key()


def test_logging_goes_to_the_console_and_a_rotating_file(workspace):
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    try:
        configure_logging(logging.DEBUG, log_file="test.log")
        logging.getLogger("corners.test").info("hello from the log")
        for handler in root.handlers:
            handler.flush()
        assert "hello from the log" in (workspace / "logs" / "test.log").read_text(encoding="utf-8")

        configure_logging(logging.INFO, log_file=None)
        assert len(root.handlers) == 1  # console only
    finally:
        for handler in root.handlers:
            if handler not in saved_handlers:
                handler.close()
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)
