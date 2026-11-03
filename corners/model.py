"""Negative-binomial GLMs for total, home and away corners.

Each market has its own model: a negative-binomial GLM with a log link,
team-form features and league fixed effects. The models are trained on
every completed season and tested on the season in progress, which is never
used for training. The training set therefore only changes when a new season
starts (or historical data is corrected), so the pipeline retrains only when
a fingerprint of the training data changes.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from corners.config import MODEL_FILE, MODEL_MAX_COUNT, REPORT_LINES
from corners.features import load_features
from corners.pricing import nb_pmf

log = logging.getLogger(__name__)

# Market -> (target column, features). Feature choices:
# * total: corners for/against for both sides, away shots against and the
#   market's P(over 2.5 goals). Other shot and on-target terms were not
#   significant and BIC preferred the smaller model.
# * home/away: chosen separately by forward stepwise selection (AIC, p < 0.05,
#   training data only). Modelling each side separately removes the noise
#   from the other side's corners, so these fit far better than the total.
MODEL_SPECS: dict[str, tuple[str, list[str]]] = {
    "total": (
        "total_corners",
        [
            "h_roll_corners_for",
            "h_roll_corners_against",
            "a_roll_corners_for",
            "a_roll_corners_against",
            "a_roll_shots_against",
            "mkt_over25_prob",
        ],
    ),
    "home": (
        "HC",
        [
            "h_roll_corners_for",
            "a_roll_shot_diff",
            "h_roll_goals_against",
            "a_roll_corners_against",
            "h_roll_shots_for",
        ],
    ),
    "away": (
        "AC",
        [
            "h_roll_shot_diff",
            "a_roll_goals_for",
            "h_roll_corners_against",
            "a_roll_corners_for",
            "a_roll_trailing_ht",
            "h_roll_goals_against",
        ],
    ),
}
SCHEMA_VERSION = 2


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MarketModel:
    """A fitted model for one market."""

    target: str
    features: list[str]
    coefficients: dict[str, float]
    alpha: float

    def predict(
        self,
        home_form: pd.Series,
        away_form: pd.Series,
        league: str,
        match_features: dict[str, float] | None = None,
    ) -> float:
        """Expected corners for one fixture.

        ``h_roll_x``/``a_roll_x`` features are read from each team's form row
        (column ``roll_x``); anything else, such as ``mkt_over25_prob``, comes
        from ``match_features``.
        """
        match_features = match_features or {}
        linear = self.coefficients.get("Intercept", 0.0)
        for feature in self.features:
            if "_roll_" in feature:
                side, stat = feature.split("_roll_", 1)
                value = (home_form if side == "h" else away_form)[f"roll_{stat}"]
            else:
                value = match_features[feature]
            linear += self.coefficients.get(feature, 0.0) * value
        linear += self.coefficients.get(f"C(League)[T.{league}]", 0.0)
        return round(max(float(np.exp(linear)), 1e-6), 2)


@dataclass(frozen=True)
class TrainedModel:
    """The three market models plus metadata about how they were trained."""

    markets: dict[str, MarketModel]
    max_count: int
    train_end_date: pd.Timestamp
    fingerprint: str
    params: dict[str, Any]

    @classmethod
    def load(cls) -> TrainedModel:
        if not MODEL_FILE.exists():
            raise FileNotFoundError(f"{MODEL_FILE} is missing; train the model first")
        return cls.from_params(json.loads(MODEL_FILE.read_text(encoding="utf-8")))

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> TrainedModel:
        if params.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"{MODEL_FILE} uses an old format; retrain with --force")
        markets = {
            name: MarketModel(m["target"], m["features"], m["coefficients"], m["alpha"])
            for name, m in params["models"].items()
        }
        return cls(
            markets=markets,
            max_count=params["max_count"],
            train_end_date=pd.Timestamp(params["train_end_date"]),
            fingerprint=params["fingerprint"],
            params=params,
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def season_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Train on every completed season; hold out the season in progress as test data."""
    seasons = df.groupby("Season")["Date"].max().sort_values().index.tolist()
    if len(seasons) < 2:
        raise ValueError("Need at least two seasons of data to train and test")
    test_seasons = seasons[-1:]
    is_test = df["Season"].isin(test_seasons)
    return df[~is_test].copy(), df[is_test].copy(), [str(s) for s in test_seasons]


_MODEL_COLUMNS = sorted(
    {
        *(target for target, _ in MODEL_SPECS.values()),
        *(feature for _, features in MODEL_SPECS.values() for feature in features),
    }
)


def complete_rows(features: pd.DataFrame) -> pd.DataFrame:
    """Matches with every model input present (``mkt_over25_prob`` starts in 2019/20)."""
    return features.dropna(subset=["Date", *_MODEL_COLUMNS]).reset_index(drop=True)


def training_data() -> pd.DataFrame:
    return complete_rows(load_features())


def fingerprint(train: pd.DataFrame) -> str:
    """Hash of the training rows (in any order) and the model specification."""
    columns = ["Date", "League", "Season", *_MODEL_COLUMNS]
    row_hashes = np.sort(pd.util.hash_pandas_object(train[columns], index=False).to_numpy())
    digest = hashlib.sha256(row_hashes.tobytes())
    digest.update(json.dumps(MODEL_SPECS, sort_keys=True).encode())
    return digest.hexdigest()[:16]


def needs_retraining() -> bool:
    """True if there is no model, or the training data or specification changed."""
    try:
        current = TrainedModel.load()
    except (FileNotFoundError, ValueError, KeyError):
        return True
    train, _, _ = season_split(training_data())
    return fingerprint(train) != current.fingerprint


def estimate_alpha(y: np.ndarray, mu: np.ndarray) -> float:
    """Cameron-Trivedi moment estimate of NB dispersion, Var(Y) = mu + alpha * mu^2."""
    mu = np.clip(np.asarray(mu, dtype=float), 1e-6, None)
    y = np.asarray(y, dtype=float)
    z = ((y - mu) ** 2 - y) / mu
    return max(float((z * mu).sum() / max((mu * mu).sum(), 1e-12)), 1e-4)


def _predict(fit: Any, df: pd.DataFrame, train_leagues: list[str]) -> np.ndarray:
    """Model means, mapping leagues unseen in training to the reference league."""
    data = df.copy()
    data.loc[~data["League"].isin(train_leagues), "League"] = sorted(train_leagues)[0]
    return np.clip(np.asarray(fit.predict(data), dtype=float), 1e-6, None)


def fit_market(train: pd.DataFrame, target: str, features: list[str]) -> tuple[Any, float, str]:
    """Fit one NB GLM: a Poisson pass gives a starting dispersion for the NB fit,
    and the reported dispersion is re-estimated from the NB fitted means."""
    formula = f"{target} ~ {' + '.join(features)} + C(League)"
    poisson = smf.glm(formula, data=train, family=sm.families.Poisson()).fit()
    alpha = estimate_alpha(train[target], poisson.fittedvalues)
    fit = smf.glm(formula, data=train, family=sm.families.NegativeBinomial(alpha=alpha)).fit()
    return fit, estimate_alpha(train[target], fit.fittedvalues), formula


def _r2(pred: np.ndarray, actual: pd.Series) -> float | None:
    return float(np.corrcoef(pred, actual)[0, 1] ** 2) if len(actual) > 1 else None


def train_models() -> dict[str, Any]:
    """Fit all three market models on the feature table and save them."""
    params = fit_models(training_data())
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    MODEL_FILE.write_text(json.dumps(params, indent=2), encoding="utf-8")
    log.info("Saved model to %s", MODEL_FILE.name)
    return params


def fit_models(df: pd.DataFrame) -> dict[str, Any]:
    """Fit all three market models and return their parameters."""
    train, test, test_seasons = season_split(df)
    train_leagues = sorted(train["League"].unique())
    log.info(
        "Training on %d matches up to %s; testing on %d matches from seasons %s",
        len(train),
        train["Date"].max().date(),
        len(test),
        ", ".join(test_seasons),
    )

    models = {}
    for market, (target, features) in MODEL_SPECS.items():
        fit, alpha, formula = fit_market(train, target, features)
        train_pred = _predict(fit, train, train_leagues)
        test_pred = _predict(fit, test, train_leagues)
        test_r2 = _r2(test_pred, test[target])
        models[market] = {
            "target": target,
            "features": features,
            "formula": formula,
            "coefficients": {k: round(float(v), 6) for k, v in fit.params.items()},
            "p_values": {k: round(float(v), 6) for k, v in fit.pvalues.items()},
            "alpha": round(alpha, 6),
            "train_r2": round(_r2(train_pred, train[target]), 6),
            "test_r2": round(test_r2, 6) if test_r2 is not None else None,
        }
        _log_fit(market, models[market])
        if market == "total":
            _log_calibration(train, test, test_pred, alpha)

    params = {
        "schema_version": SCHEMA_VERSION,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "fingerprint": fingerprint(train),
        "train_seasons": sorted(str(s) for s in train["Season"].unique()),
        "test_seasons": test_seasons,
        "train_end_date": str(train["Date"].max().date()),
        "n_train": len(train),
        "n_test": len(test),
        "max_count": MODEL_MAX_COUNT,
        "models": models,
    }
    return params


def _log_fit(market: str, model: dict[str, Any]) -> None:
    lines = [f"{market} model: {model['formula']}"]
    for name, coef in model["coefficients"].items():
        p = model["p_values"][name]
        lines.append(f"    {name[:38]:<40}{coef:>+9.4f}  p={p:.4f}{' *' if p < 0.05 else ''}")
    test_r2 = "n/a" if model["test_r2"] is None else f"{model['test_r2']:.2%}"
    lines.append(f"    alpha={model['alpha']:.4f}  train R2={model['train_r2']:.2%}  test R2={test_r2}")
    log.info("\n".join(lines))


def _line_probs(means: np.ndarray, alpha: float, lines: list[float]) -> np.ndarray:
    k = np.arange(MODEL_MAX_COUNT + 1)
    pmfs = np.array([nb_pmf(m, alpha, MODEL_MAX_COUNT + 1) for m in means])
    return np.column_stack([pmfs[:, k > line].sum(axis=1) for line in lines])


def _log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _log_calibration(train: pd.DataFrame, test: pd.DataFrame, test_pred: np.ndarray, alpha: float) -> None:
    """Held-out over/under calibration of the total model against a constant baseline."""
    if test.empty:
        return
    actual = test["total_corners"].to_numpy()
    model_p = _line_probs(test_pred, alpha, REPORT_LINES)
    rows = [
        "Test-set over/under calibration (log loss: lower is better)",
        f"    {'line':<10}{'observed':>9}{'model':>9}{'base LL':>10}{'model LL':>10}",
    ]
    for i, line in enumerate(REPORT_LINES):
        y = (actual > line).astype(float)
        base_p = np.full(len(y), (train["total_corners"] > line).mean())
        rows.append(
            f"    over {line:<5}{y.mean():>9.1%}{model_p[:, i].mean():>9.1%}"
            f"{_log_loss(base_p, y):>10.4f}{_log_loss(model_p[:, i], y):>10.4f}"
        )
    log.info("\n".join(rows))
