import pandas as pd
import pytest

from corners.tracking import ResultBook, out_of_sample, settle


@pytest.mark.parametrize(
    ("side", "actual", "expected"),
    [
        ("Over", 11, ("WIN", 10.0)),
        ("Over", 9, ("LOSS", -10.0)),
        ("Under", 9, ("WIN", 10.0)),
        ("Over", None, ("PENDING", 0.0)),
    ],
)
def test_settle(side, actual, expected):
    assert settle(side, 9.5, 2.0, 10.0, actual) == expected


def test_result_book_only_matches_a_nearby_date():
    results = pd.DataFrame(
        {
            "League": ["E0", "E0"],
            "HomeTeam": ["Leeds", "Leeds"],
            "AwayTeam": ["Fulham", "Fulham"],
            "Date": pd.to_datetime(["2025-10-10", "2026-10-10"]),
            "HC": [4, 7],
            "AC": [3, 2],
        }
    )
    book = ResultBook(results)
    assert book.actual("E0", "Leeds", "Fulham", pd.Timestamp("2026-10-10 19:00"), "total_corners") == 9
    assert book.actual("E0", "Leeds", "Fulham", pd.Timestamp("2026-10-20"), "HC") is None
    assert book.actual("E0", "Fulham", "Leeds", pd.Timestamp("2026-10-10"), "HC") is None


def test_out_of_sample_drops_fixtures_inside_the_training_period():
    history = pd.DataFrame(
        {
            "start_datetime": pd.to_datetime(
                ["2025-05-31 18:00", "2025-06-01 18:00", "2025-06-02 18:00"], utc=True
            )
        }
    )
    kept = out_of_sample(history, pd.Timestamp("2025-06-01"))
    assert len(kept) == 1
