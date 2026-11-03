"""Training data, the season split, and when the models need retraining."""

import json

import numpy as np
import pytest

from corners.model import (
    TrainedModel,
    complete_rows,
    estimate_alpha,
    fingerprint,
    needs_retraining,
    season_split,
)


def test_models_train_on_completed_seasons_and_are_tested_on_the_current_one(trained):
    train, test, test_seasons = season_split(complete_rows(trained.features))
    assert test_seasons == ["2425"]
    assert set(test["Season"]) == {"2425"}
    assert "2425" not in set(train["Season"])
    assert len(train) > len(test) > 0


def test_a_single_season_cannot_be_split(trained):
    with pytest.raises(ValueError, match="two seasons"):
        season_split(trained.test_season())


def test_the_fingerprint_ignores_row_order_but_not_content(trained):
    train, _, _ = season_split(complete_rows(trained.features))
    assert fingerprint(train.sample(frac=1, random_state=1)) == fingerprint(train)
    assert fingerprint(train.iloc[1:]) != fingerprint(train)

    corrected = train.copy()
    corrected.loc[corrected.index[0], "HC"] += 1
    assert fingerprint(corrected) != fingerprint(train)


def test_the_fingerprint_ignores_numerical_noise_but_not_real_changes(trained):
    train, _, _ = season_split(complete_rows(trained.features))
    noisy = train.copy()
    noisy["h_roll_corners_for"] *= 1 + 1e-12  # the kind of difference between machines
    assert fingerprint(noisy) == fingerprint(train)

    shifted = train.copy()
    shifted["h_roll_corners_for"] += 0.01  # a real change to the data
    assert fingerprint(shifted) != fingerprint(train)


def test_old_model_files_are_rejected(trained):
    with pytest.raises(ValueError, match="old format"):
        TrainedModel.from_params({**trained.params, "schema_version": 1})


def test_a_model_is_needed_when_none_exists_or_its_format_is_old(workspace):
    assert needs_retraining()  # no file yet

    (workspace / "data" / "model_params.json").write_text(json.dumps({"schema_version": 1}))
    assert needs_retraining()


def test_loading_a_missing_model_is_an_error(workspace):
    with pytest.raises(FileNotFoundError, match="train the model first"):
        TrainedModel.load()


def test_dispersion_is_recovered_from_overdispersed_counts():
    rng = np.random.default_rng(0)
    counts = rng.negative_binomial(5, 1 / 3, 200_000)  # mean 10, variance 30, so alpha = 0.2
    assert estimate_alpha(counts, np.full(len(counts), counts.mean())) == pytest.approx(0.2, abs=0.02)

    poisson = rng.poisson(10, 200_000)  # no overdispersion
    assert estimate_alpha(poisson, np.full(len(poisson), poisson.mean())) < 0.01
