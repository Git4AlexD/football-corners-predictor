"""Choosing which results are safe to share publicly.

Only a fixed list of charts is ever published. They show markets, dates and
pounds, with no fixtures, odds, bookmakers or other data from the providers,
so a new file in ``results/`` can never be published by accident.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from corners.config import RESULTS_DIR

log = logging.getLogger(__name__)

# File under results/ -> name it is published under.
PUBLISHABLE_CHARTS = {
    "clv/expected_vs_actual_roi.png": "return_on_stake.png",
    "clv/profit_over_time.png": "profit_over_time.png",
    "flat/equity_combined.png": "equity_combined.png",
    "flat/equity_total.png": "equity_total.png",
    "flat/equity_home.png": "equity_home.png",
    "flat/equity_away.png": "equity_away.png",
    "flat/cash_flow.png": "cash_flow.png",
    "flat/returns_by_target.png": "returns_by_target.png",
}


def copy_publishable_charts(destination: Path) -> list[str]:
    """Copy the shareable charts into ``destination``; returns the names copied.

    A chart that hasn't been produced yet (for example, no bets have settled)
    is skipped rather than treated as an error.
    """
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    for source, name in PUBLISHABLE_CHARTS.items():
        path = RESULTS_DIR / source
        if path.exists():
            shutil.copyfile(path, destination / name)
            copied.append(name)
        else:
            log.warning("Chart not produced yet, skipped: %s", source)
    log.info("Copied %d of %d charts to %s", len(copied), len(PUBLISHABLE_CHARTS), destination)
    return copied
