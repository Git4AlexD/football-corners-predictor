"""From a fixture's price to a decision decision.

For each market the model's expected corners is blended with the mean
implied by the provider's whole over/under ladder. Every line and side is
then priced from the blended negative-binomial distribution, and the single
highest-EV selection is decision if its EV clears the threshold, sized by capped
fractional Kelly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from corners.config import (
    AWAY_CORNER_LINES,
    DECISION_CONFIG,
    HOME_CORNER_LINES,
    TOTAL_CORNER_LINES,
    DecisionConfig,
)
from corners.data.price_history import decision_ladder, decision_providers, market_over25_prob
from corners.features import FormBook
from corners.model import TrainedModel
from corners.pricing import PriceLadder, fit_market_ladder, is_valid_price_pair, nb_pmf
from corners.teams import TeamLookup, log_unmatched_team, match_team

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Market:
    key: str
    label: str
    price_key: str  # column key in the price history
    lines: list[float]
    actual_column: str  # result column the market settles on


MARKETS = {
    "total": Market("total", "Total corners", "corners", TOTAL_CORNER_LINES, "total_corners"),
    "home": Market("home", "Home corners", "home_corners", HOME_CORNER_LINES, "HC"),
    "away": Market("away", "Away corners", "away_corners", AWAY_CORNER_LINES, "AC"),
}

# Statuses for fixtures that could not be priced.
NO_TEAM_MATCH = "NO TEAM MATCH"  # a name matched nothing; may need an alias
TEAM_TOO_NEW = "TEAM TOO NEW"  # a known team without enough history yet
NO_DATA = "NO DATA"  # stale form or missing goals price
SELECTED, NOT_SELECTED = "SELECTED", "NOT SELECTED"


@dataclass(frozen=True)
class Decision:
    """The best selection for one market, whether or not it is decision."""

    selected: bool
    side: str | None = None
    line: float | None = None
    price: float = 0.0
    ev: float | None = None  # None when the ladder had no usable price
    size: float = 0.0
    size_pct: float = 0.0


def kelly_size(
    ev: float, price: float, capital: float, config: DecisionConfig = DECISION_CONFIG
) -> tuple[float, float]:
    """Capped fractional-Kelly size as ``(amount, percent of capital)``.

    EV is capped before sizing, and the size is capped as a share of the
    capital, because at short price Kelly can still ask for a large size.
    """
    if ev < config.ev_threshold or price <= 1.0:
        return 0.0, 0.0
    kelly = config.kelly_fraction * (min(ev, config.max_kelly_ev) / (price - 1.0))
    fraction = max(0.0, min(kelly, config.max_size_pct))
    return round(capital * fraction, 2), round(fraction * 100, 2)


def decide_selection(
    ladder: PriceLadder,
    mu: float,
    alpha: float,
    lines: list[float],
    max_count: int,
    capital: float,
    config: DecisionConfig = DECISION_CONFIG,
) -> Decision:
    """Price every line and side in the ladder and pick the highest EV."""
    pmf = nb_pmf(mu, alpha, max_count + 1)
    counts = np.arange(max_count + 1)
    best: tuple[float, str, float, float] | None = None  # (ev, side, line, price)

    for line in lines:
        over, under = ladder[line]
        if not is_valid_price_pair(over, under):
            continue
        for side, price, prob in (
            ("Over", over, pmf[counts > line].sum()),
            ("Under", under, pmf[counts < line].sum()),
        ):
            if config.min_price <= price <= config.max_price:
                ev = float(prob * price - 1.0)
                if best is None or ev > best[0]:
                    best = (ev, side, line, price)

    if best is None:
        return Decision(selected=False)
    ev, side, line, price = best
    if ev < config.ev_threshold:
        return Decision(selected=False, side=side, line=line, price=price, ev=ev)
    size, pct = kelly_size(ev, price, capital, config)
    return Decision(selected=True, side=side, line=line, price=price, ev=ev, size=size, size_pct=pct)


@dataclass(frozen=True)
class Evaluation:
    """Everything known about one fixture-market decision."""

    fixture_id: int
    league: str
    kickoff: pd.Timestamp
    home: str  # price-feed names
    away: str
    market: str
    status: str
    home_team: str | None = None  # dataset names
    away_team: str | None = None
    decision: Decision | None = None
    provider: str | None = None
    model_mu: float | None = None
    blended_mu: float | None = None
    alpha: float | None = None

    @property
    def fixture_date(self) -> pd.Timestamp:
        return self.kickoff.tz_convert(None)


class FixtureEvaluator:
    """Applies the model and decision rules to rows of the price history."""

    def __init__(
        self,
        model: TrainedModel,
        form: FormBook,
        lookup: TeamLookup,
        broad_lookup: TeamLookup | None = None,
        config: DecisionConfig = DECISION_CONFIG,
    ) -> None:
        self.model = model
        self.form = form
        self.lookup = lookup
        self.broad_lookup = broad_lookup  # all known teams, to recognise TEAM TOO NEW
        self.config = config
        self._ladder_fits: dict[tuple, tuple[float | None, float | None]] = {}

    def evaluate(
        self,
        row: pd.Series,
        market: str,
        capital: float | None = None,
        ladder: PriceLadder | None = None,
        ev_threshold: float | None = None,
    ) -> Evaluation:
        """Decide what to decision on one fixture in one market.

        Args:
            row: A row of the price history.
            market: A key of ``MARKETS``.
            capital: Capital to size the size against (default: config).
            ladder: Prices to use instead of the usual early-then-closing ladder.
            ev_threshold: Override the EV threshold, e.g. ``-inf`` to always pick.
        """
        spec = MARKETS[market]
        base = Evaluation(
            fixture_id=int(row["fixture_id"]),
            league=row["league"],
            kickoff=row["start_datetime"],
            home=row["home_team"],
            away=row["away_team"],
            market=market,
            status=NO_DATA,
        )
        home_team = match_team(base.home, base.league, self.lookup)
        away_team = match_team(base.away, base.league, self.lookup)
        if home_team is None or away_team is None:
            return replace(base, status=self._unmatched_status(base, home_team, away_team))
        base = replace(base, home_team=home_team, away_team=away_team)

        home_form = self.form.before(base.league, home_team, base.fixture_date)
        away_form = self.form.before(base.league, away_team, base.fixture_date)
        if home_form is None or away_form is None:
            return base

        model = self.model.markets[market]
        match_features = {}
        if "mkt_over25_prob" in model.features:
            prob = market_over25_prob(row)
            if prob is None:
                return base
            match_features["mkt_over25_prob"] = prob
        model_mu = model.predict(home_form, away_form, base.league, match_features)

        ladder = ladder if ladder is not None else decision_ladder(row, spec.price_key)
        market_mu, market_alpha = self._fit_ladder(ladder)
        if market_mu is not None:
            w = self.config.model_weight
            mu = round(w * model_mu + (1 - w) * market_mu, 2)
            alpha = market_alpha or model.alpha
        else:
            mu, alpha = model_mu, model.alpha

        config = self.config if ev_threshold is None else replace(self.config, ev_threshold=ev_threshold)
        decision = decide_selection(
            ladder,
            mu,
            alpha,
            spec.lines,
            self.model.max_count,
            capital if capital is not None else config.capital,
            config,
        )
        provider = None
        if decision.selected:
            over_book, under_book = decision_providers(row, spec.price_key).get(decision.line, (None, None))
            provider = over_book if decision.side == "Over" else under_book
        return replace(
            base,
            status=SELECTED if decision.selected else NOT_SELECTED,
            decision=decision,
            provider=provider,
            model_mu=model_mu,
            blended_mu=mu,
            alpha=alpha,
        )

    def _fit_ladder(self, ladder: PriceLadder) -> tuple[float | None, float | None]:
        key = tuple(sorted((line, over, under) for line, (over, under) in ladder.items()))
        if key not in self._ladder_fits:
            self._ladder_fits[key] = fit_market_ladder(ladder)
        return self._ladder_fits[key]

    def _unmatched_status(self, base: Evaluation, home_team: str | None, away_team: str | None) -> str:
        unresolved = [n for n, t in ((base.home, home_team), (base.away, away_team)) if t is None]
        if self.broad_lookup is not None and all(
            match_team(name, base.league, self.broad_lookup) is not None for name in unresolved
        ):
            return TEAM_TOO_NEW
        for name in unresolved:
            opponent = base.away if name == base.home else base.home
            log_unmatched_team(name, base.league, opponent=opponent, fixture_date=str(base.fixture_date))
        return NO_TEAM_MATCH
