"""Charts for back-test and evaluation results.

Uses matplotlib's object-oriented API (``Figure``) so no display or global
pyplot state is needed when running unattended.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from corners.decisions import MARKETS

log = logging.getLogger(__name__)

# Light-mode palette: categorical blue, orange and aqua; status green and red.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GOOD, BAD = "#0ca30c", "#d03b3b"
MIN_COMMITTED_FOR_RETURN_PCT = 100.0  # £ committed before a percentage return is drawn
GRAY, GRID, INK, MUTED, SURFACE = "#c3c2b7", "#e1e0d9", "#0b0b0b", "#52514e", "#fcfcfb"


def _end_label(ax, x, y, text: str, color: str, dy: int = 0, bold: bool = True) -> None:
    ax.plot(x, y, marker="o", markersize=8, color=color, markeredgecolor="white", markeredgewidth=1.5)
    ax.annotate(
        text,
        (x, y),
        xytext=(8, dy),
        textcoords="offset points",
        va="center",
        color=MUTED,
        fontsize=9,
        fontweight="bold" if bold else "normal",
    )


def _save(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())


def plot_equity_curves(records: pd.DataFrame, out_dir: Path) -> None:
    """One chart per market plus all markets combined."""
    groups = {key: records[records["market"] == key] for key in MARKETS}
    groups["combined"] = records
    for key, group in groups.items():
        label = MARKETS[key].label if key in MARKETS else "All targets"
        _plot_equity_curve(group, label, out_dir / f"equity_{key}.png")


def _plot_equity_curve(records: pd.DataFrame, label: str, path: Path) -> None:
    """Cumulative committed vs committed + gain (top) and cumulative ROI (bottom).

    Two panels sharing a date axis rather than one chart with two y-axes,
    because pounds and percentages are different scales.
    """
    settled = records[records["status"].isin(["CORRECT", "INCORRECT"])].sort_values("date", kind="stable")
    if settled.empty:
        return
    dates = [settled["date"].iloc[0], *settled["date"]]
    committed = [0.0, *settled["size"].cumsum()]
    gain = [0.0, *settled["gain"].cumsum()]
    value = [s + p for s, p in zip(committed, gain, strict=True)]
    roi = [100 * p / s for s, p in zip(committed[1:], gain[1:], strict=True)]

    fig = Figure(figsize=(10, 7))
    ax_gbp, ax_roi = fig.subplots(2, 1, sharex=True, height_ratios=[2, 1])

    ax_gbp.step(
        dates, committed, where="post", color=GRAY, linestyle="--", linewidth=2, label="Cumulative committed"
    )
    ax_gbp.step(dates, value, where="post", color=BLUE, linewidth=2, label="Committed + gain")
    _end_label(ax_gbp, dates[-1], committed[-1], f"£{committed[-1]:,.0f}", GRAY, bold=False)
    _end_label(ax_gbp, dates[-1], value[-1], f"£{value[-1]:,.0f}", BLUE)
    ax_gbp.set_ylabel("£")
    ax_gbp.set_title(f"{label}: {len(settled)} settled decisions")
    ax_gbp.legend(loc="upper left")

    # Colour each step by its own sign, not the final one.
    step_x = np.repeat(dates[1:], 2)[1:]
    step_y = np.repeat(roi, 2)[:-1]
    positive = np.where(step_y >= 0, step_y, np.nan)
    negative = np.where(step_y < 0, step_y, np.nan)
    ax_roi.axhline(0, color=GRAY, linewidth=1)
    for series, color in ((positive, GOOD), (negative, BAD)):
        ax_roi.plot(step_x, series, color=color, linewidth=2)
        ax_roi.fill_between(step_x, series, 0, color=color, alpha=0.10)
    _end_label(ax_roi, dates[-1], roi[-1], f"{roi[-1]:+.1f}%", GOOD if roi[-1] >= 0 else BAD)
    ax_roi.set_ylabel("ROI %")
    ax_roi.set_xlabel("Date")

    for ax in (ax_gbp, ax_roi):
        ax.grid(True, color=GRID, linewidth=1)
        ax.set_axisbelow(True)
    fig.autofmt_xdate()
    fig.tight_layout()
    _save(fig, path)


def _dodged_end_labels(fig: Figure, ax, x, items: list[tuple[float, str, str]]) -> None:
    """Mark each line's end point and label it, nudging labels apart so none overlap.

    ``items`` holds ``(y, colour, text)``. Labels are spaced in screen units, so the
    spacing holds whatever the data range. The markers stay on the true values.
    """
    if not items:
        return
    min_gap_px = 15
    pixels = [ax.transData.transform((mdates.date2num(x), y))[1] for y, _, _ in items]
    order = sorted(range(len(items)), key=lambda i: pixels[i])
    placed: dict[int, float] = {}
    previous = -np.inf
    for i in order:
        previous = max(pixels[i], previous + min_gap_px)
        placed[i] = previous
    shift = np.mean([pixels[i] for i in placed]) - np.mean(list(placed.values()))  # re-centre the group
    for i, (y, colour, text) in enumerate(items):
        dy = (placed[i] + shift - pixels[i]) * 72 / fig.dpi
        _end_label(ax, x, y, text, colour, dy=round(dy))


def plot_returns_by_target(records: pd.DataFrame, path: Path) -> None:
    """Cumulative return per target with a combined line (top) and return on amount committed (bottom).

    Built from every settled decision in the back-test. Two panels sharing a date axis
    rather than one chart with two y-axes, because pounds and percentages are different scales.
    """
    settled = records[records["status"].isin(["CORRECT", "INCORRECT"])].copy()
    if settled.empty:
        return
    settled["day"] = pd.to_datetime(settled["date"]).dt.normalize()
    days = pd.DatetimeIndex(sorted(settled["day"].unique()))
    start = days[0] - pd.Timedelta(days=1)

    lines = [
        (MARKETS[key].label, colour, 2.0, settled[settled["market"] == key])
        for key, colour in (("total", BLUE), ("home", ORANGE), ("away", AQUA))
    ] + [("All targets combined", INK, 3.0, settled)]
    lines = [line for line in lines if not line[3].empty]

    fig = Figure(figsize=(10, 7))
    fig.patch.set_facecolor(SURFACE)
    ax_gbp, ax_roi = fig.subplots(2, 1, sharex=True, height_ratios=[2, 1])
    ends_gbp, ends_roi = [], []
    for label, colour, width, group in lines:
        daily = group.groupby("day")[["size", "gain"]].sum().reindex(days, fill_value=0.0).cumsum()
        ret = [0.0, *daily["gain"]]
        # The first few small decisions would swing the percentage between extremes.
        committed = daily["size"].where(daily["size"] >= MIN_COMMITTED_FOR_RETURN_PCT)
        roi = (100 * daily["gain"] / committed).to_numpy()
        ax_gbp.step([start, *days], ret, where="post", color=colour, linewidth=width, label=label)
        ax_roi.step(days, roi, where="post", color=colour, linewidth=width)
        ends_gbp.append((ret[-1], colour, f"{'+' if ret[-1] >= 0 else '-'}£{abs(ret[-1]):,.0f}"))
        if not np.isnan(roi[-1]):
            ends_roi.append((roi[-1], colour, f"{roi[-1]:+.1f}%"))

    for ax in (ax_gbp, ax_roi):
        ax.set_facecolor(SURFACE)
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.grid(True, color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.set_xlim(start, days[-1] + (days[-1] - start) * 0.16)  # room for the end labels
    ax_gbp.set_ylabel("Cumulative return (£)", color=MUTED, fontsize=9)
    ax_roi.set_ylabel("Return on amount committed (%)", color=MUTED, fontsize=9)
    ax_roi.text(
        0.01,
        0.95,
        f"Shown once £{MIN_COMMITTED_FOR_RETURN_PCT:,.0f} has been committed",
        transform=ax_roi.transAxes,
        va="top",
        fontsize=8,
        color=MUTED,
    )
    ax_gbp.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK)
    ax_roi.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    ax_roi.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    _titled_layout(
        fig,
        "Cumulative return by target, and combined",
        f"Back-test of every settled decision ({len(settled)}), including those priced at the closing price.",
    )
    _dodged_end_labels(fig, ax_gbp, days[-1], ends_gbp)
    _dodged_end_labels(fig, ax_roi, days[-1], ends_roi)
    _save(fig, path)


def plot_cash_flow(records: pd.DataFrame, path: Path) -> None:
    """Own capital in play over time vs net return, across all targets.

    "Own capital in play" is the amount committed minus the returns received,
    floored at zero once returns cover everything committed. Each match day's
    positions are taken before its returns come back, so the line peaks within the
    day; the highest peak is the most of your own capital ever in play at once.
    Positions still pending count from today.
    """
    decisions = records[records["status"].isin(["CORRECT", "INCORRECT", "PENDING"])].copy()
    if decisions.empty:
        return
    correct = decisions["status"] == "CORRECT"
    settled = decisions["status"].isin(["CORRECT", "INCORRECT"])
    decisions["returned"] = np.where(correct, decisions["size"] + decisions["gain"], 0.0)
    decisions["made"] = np.where(settled, decisions["gain"], 0.0)
    today = pd.Timestamp.now().normalize()
    decisions["day"] = pd.to_datetime(decisions["date"]).dt.normalize().clip(upper=today)
    daily = decisions.groupby("day")[["size", "returned", "made"]].sum().sort_index()
    pending = decisions.loc[decisions["status"] == "PENDING", "size"].sum()

    oop_x, oop_y = [daily.index[0]], [0.0]
    peak, peak_day, net = 0.0, daily.index[0], 0.0
    for day, row in daily.iterrows():
        before_returns = max(net + row["size"], 0.0)
        if before_returns > peak:
            peak, peak_day = before_returns, day
        net += row["size"] - row["returned"]
        oop_x += [day, day, day]
        oop_y += [oop_y[-1], before_returns, max(net, 0.0)]
    dates = [daily.index[0], *daily.index]
    made = [0.0, *daily["made"].cumsum()]

    fig = Figure(figsize=(10, 5.5))
    ax = fig.subplots()
    ax.axhline(0, color=GRAY, linewidth=1)
    ax.plot(
        oop_x,
        oop_y,
        color=BLUE,
        linewidth=2,
        label="Own capital in play (losses + pending - returns, min £0)",
    )
    ax.step(dates, made, where="post", color=AQUA, linewidth=2, label="Net return (settled)")
    ax.plot(
        peak_day, peak, marker="o", markersize=8, color=BLUE, markeredgecolor="white", markeredgewidth=1.5
    )
    ax.annotate(
        f"Most of your own capital in play at once: £{peak:,.0f} ({peak_day:%d %b})",
        (peak_day, peak),
        xytext=(-10, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        color=MUTED,
        fontsize=9,
    )

    # Separate the two end labels if they would overlap.
    span = (max(oop_y + made) - min(oop_y + made)) or 1.0
    dy_oop = dy_made = 0
    if abs(oop_y[-1] - made[-1]) < 0.08 * span:
        dy_oop, dy_made = (7, -7) if oop_y[-1] >= made[-1] else (-7, 7)
    sign = "+" if made[-1] >= 0 else "-"
    _end_label(ax, oop_x[-1], oop_y[-1], f"£{oop_y[-1]:,.0f} of own capital in play", BLUE, dy=dy_oop)
    _end_label(ax, dates[-1], made[-1], f"{sign}£{abs(made[-1]):,.0f} net return", AQUA, dy=dy_made)

    ax.set_title(f"Cash flow, all targets: {len(decisions)} decisions, £{pending:,.0f} pending")
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_xlabel("Date")
    ax.set_ylabel("£")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    ax.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    fig.autofmt_xdate()
    fig.tight_layout()
    _save(fig, path)


SCOPE_NOTE = "Only decisions with both an early and a closing price recorded, so fewer than the back-test."


def _titled_layout(fig: Figure, title: str, note: str) -> None:
    """Left-aligned title with a muted note underneath, then tight layout below them."""
    height = fig.get_figheight()
    fig.text(0.01, 1 - 0.2 / height, title, ha="left", va="top", fontsize=11, color=INK)
    fig.text(0.01, 1 - 0.45 / height, note, ha="left", va="top", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.7 / height))


def plot_roi_comparison(summary: pd.DataFrame, path: Path) -> None:
    """Expected vs actual return on size per market, with 95% intervals.

    A target shows an advantage when its whole expected-ROI interval sits right of
    zero. The actual-ROI interval is much wider because results are noisy.
    """
    fig = Figure(figsize=(9, 1.6 + 1.2 * len(summary)))
    fig.patch.set_facecolor(SURFACE)
    ax = fig.subplots()
    ax.set_facecolor(SURFACE)
    ax.axvline(0, color=MUTED, linewidth=1)

    series = (
        ("expected_roi", BLUE, -0.15, "Expected, valued at closing prices"),
        ("realised_roi", ORANGE, 0.15, "Realised, from outcomes so far"),
    )
    positions = np.arange(len(summary))
    for column, colour, offset, label in series:
        y = positions + offset
        value = summary[column].to_numpy() * 100
        low = summary[f"{column}_low"].to_numpy() * 100
        high = summary[f"{column}_high"].to_numpy() * 100
        ax.errorbar(
            value,
            y,
            xerr=[value - low, high - value],
            fmt="o",
            color=colour,
            markersize=8,
            elinewidth=2,
            capsize=4,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            label=label,
        )
        for xi, yi, hi in zip(value, y, high, strict=True):
            if not np.isnan(xi):
                ax.annotate(
                    f"{xi:+.1f}%",
                    (hi, yi),
                    xytext=(6, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=9,
                    color=INK,
                )

    labels = [f"{row.market}\n{row.decisions} decisions: {row.verdict}" for row in summary.itertuples()]
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:+.0f}%")
    ax.set_xlabel(
        "Return on amount committed (dot) with 95% confidence interval (line)", color=MUTED, fontsize=9
    )
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.legend(
        frameon=False,
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25 / len(summary) - 0.12),
        ncol=2,
        labelcolor=INK,
    )
    _titled_layout(fig, "Return by target: expected at closing prices vs realised", SCOPE_NOTE)
    _save(fig, path)


def plot_cumulative_return(decisions: pd.DataFrame, path: Path) -> None:
    """Cumulative expected and actual gain on settled decisions, one panel per market."""
    settled = decisions[decisions["status"].isin(["CORRECT", "INCORRECT"])]
    markets = [m for m in MARKETS if (settled["market"] == m).any()]
    if not markets:
        return
    fig = Figure(figsize=(4.2 * len(markets), 4.2))
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_1d(fig.subplots(1, len(markets), sharey=True))
    for ax, market in zip(axes, markets, strict=True):
        group = settled[settled["market"] == market]
        # One step per match day, so same-day decisions don't draw vertical spikes.
        daily = group.groupby("date")[["expected_gain", "gain"]].sum().cumsum()
        dates = [daily.index[0] - pd.Timedelta(days=1), *daily.index]
        expected = [0.0, *daily["expected_gain"]]
        actual = [0.0, *daily["gain"]]
        ax.step(dates, expected, where="post", color=BLUE, linewidth=2, label="Expected at closing prices")
        ax.step(dates, actual, where="post", color=ORANGE, linewidth=2, label="Realised")

        # Nudge the end labels apart if they would overlap.
        span = (max(expected + actual) - min(expected + actual)) or 1.0
        dy_exp = dy_act = 0
        if abs(expected[-1] - actual[-1]) < 0.08 * span:
            dy_exp, dy_act = (7, -7) if expected[-1] >= actual[-1] else (-7, 7)
        for value, colour, dy in ((expected[-1], BLUE, dy_exp), (actual[-1], ORANGE, dy_act)):
            sign = "+" if value >= 0 else "-"
            _end_label(ax, dates[-1], value, f"{sign}£{abs(value):,.0f}", colour, dy=dy)

        ax.set_facecolor(SURFACE)
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.set_title(
            f"{MARKETS[market].label} ({len(group)} settled decisions)", fontsize=10, color=INK, loc="left"
        )
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.tick_params(colors=MUTED, labelsize=9)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("Cumulative return (£)", color=MUTED, fontsize=9)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK)
    _titled_layout(fig, "Cumulative return: expected at closing prices vs realised", SCOPE_NOTE)
    _save(fig, path)
