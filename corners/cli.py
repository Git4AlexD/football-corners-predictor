"""Command-line entry point: ``python -m corners <command>`` (default: ``run``)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from corners import pipeline
from corners.config import DECISION_CONFIG
from corners.data.api_football import ApiFootballClient
from corners.data.price_history import capture_prices, load_price_history
from corners.data.results import update_results
from corners.decisions import MARKETS, SELECTED
from corners.features import build_features
from corners.log import configure_logging
from corners.publish import copy_publishable_charts


def _decisions(start: str, end: str | None) -> None:
    """Print the model's decisions for fixtures kicking off between two dates."""
    first = pd.Timestamp(start, tz="UTC")
    last = pd.Timestamp(end or start, tz="UTC")
    history = load_price_history()
    kickoff = history["start_datetime"]
    in_range = history[(kickoff >= first) & (kickoff < last + pd.Timedelta(days=1))]
    if in_range.empty:
        print(f"No fixtures in the price history between {first.date()} and {last.date()}.")
        return

    evaluator = pipeline.build_evaluator()
    for market, spec in MARKETS.items():
        rows = []
        for _, row in in_range.sort_values("start_datetime").iterrows():
            e = evaluator.evaluate(row, market)
            d = e.decision
            rows.append(
                {
                    "kickoff": e.kickoff.strftime("%Y-%m-%d %H:%M"),
                    "league": e.league,
                    "fixture": f"{e.home} v {e.away}",
                    "selection": f"{d.side} {d.line} @ {d.price:.2f}" if d and d.selected else "-",
                    "provider": e.provider or "-",
                    "ev": f"{d.ev:+.1%}" if d and d.ev is not None else "-",
                    "size": f"£{d.size:.2f}" if d and d.selected else "-",
                    "status": e.status,
                }
            )
        table = pd.DataFrame(rows)
        selected = table[table["status"] == SELECTED]
        total = sum(float(s.lstrip("£")) for s in selected["size"])
        print(f"\n{spec.label} (capital £{DECISION_CONFIG.capital:,.0f})")
        print(table.to_string(index=False))
        print(f"{len(selected)} selected, £{total:.2f} committed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="corners", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="full daily pipeline (uses the API)")
    run.add_argument("--retrain", action="store_true", help="retrain even if the training data is unchanged")
    offline = commands.add_parser("offline", help="rebuild features, model and results without API calls")
    offline.add_argument(
        "--retrain", action="store_true", help="retrain even if the training data is unchanged"
    )
    commands.add_parser("report", help="refresh results from stored data without changing it")
    charts = commands.add_parser("charts", help="copy the charts that are safe to share into a folder")
    charts.add_argument("destination", help="folder to copy them into")
    commands.add_parser("prices", help="capture price snapshots")
    results = commands.add_parser("results", help="update match results")
    results.add_argument("--full", action="store_true", help="re-download every season")
    commands.add_parser("features", help="rebuild feature tables")
    train = commands.add_parser("train", help="train the models if the training data changed")
    train.add_argument("--force", action="store_true", help="retrain regardless")
    commands.add_parser("track", help="back-test and grade results")
    commands.add_parser("evaluate", help="evaluate the forecasts against closing prices")
    decisions = commands.add_parser("decisions", help="the model's decisions for a date range")
    decisions.add_argument("start", help="first kickoff date, YYYY-MM-DD")
    decisions.add_argument("end", nargs="?", help="last kickoff date (default: same as start)")

    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv or ["run"])  # no command: the daily pipeline
    configure_logging(logging.DEBUG if args.verbose else logging.INFO)

    match args.command:
        case "run":
            return pipeline.run_steps(pipeline.daily_steps(args.retrain))
        case "offline":
            return pipeline.run_steps(pipeline.offline_steps(args.retrain))
        case "report":
            return pipeline.run_steps(pipeline.report_steps())
        case "charts":
            return 0 if copy_publishable_charts(Path(args.destination)) else 1  # 1: nothing to publish
        case "prices":
            capture_prices(ApiFootballClient())
        case "results":
            update_results(ApiFootballClient(), full_rebuild=args.full)
        case "features":
            build_features()
        case "train":
            pipeline.train_if_needed(force=args.force)
        case "track":
            pipeline.track_results()
        case "evaluate":
            pipeline.evaluate_against_closing_prices()
        case "decisions":
            _decisions(args.start, args.end)
    return 0
