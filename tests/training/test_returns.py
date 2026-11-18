"""Tests de l'entraînement du modèle `rendement`."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from features import FEATURE_COLUMNS
from training.dataset import ReturnTrainingDataset
from training.returns import (
    BASELINES,
    TARGET_COLUMN,
    _candidates,
    cross_validate,
    fit_return_model,
    score,
)


@pytest.fixture
def synthetic() -> ReturnTrainingDataset:
    """Panel multi-symbole de 18 mois, avec un signal transversal déterministe."""
    rng = np.random.default_rng(42)
    times = pd.date_range('2022-01-01', '2023-06-01', freq='MS', tz='UTC')
    ranks = np.arange(-2, 3, dtype='float64')
    symbols = ('BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT')
    rows = []
    for month_index, timestamp in enumerate(times):
        for symbol_index, (symbol, rank) in enumerate(zip(symbols, ranks, strict=True)):
            row = {column: float(rng.normal()) for column in FEATURE_COLUMNS}
            row.update(
                symbol=symbol,
                open_time=timestamp,
                return_1h=-rank * 0.001,
                return_4h=rank * 0.0001,
                target_return=rank * 0.001 + (month_index + symbol_index) * 1e-7,
            )
            rows.append(row)
    frame = pd.DataFrame(rows)
    return ReturnTrainingDataset(
        frame=frame,
        identity=None,
        quality_report=None,
        as_of=pd.Timestamp('2023-06-02', tz='UTC'),
    )


def test_cross_validation_scores_candidates_and_baselines_on_each_fold(synthetic):
    evaluation = cross_validate(synthetic.frame)

    assert len(evaluation.folds) == 3
    assert set(evaluation.predictors()) == {*BASELINES, *_candidates()}
    for fold in evaluation.folds:
        assert all(set(metrics) == {'ic_mean', 'icir'} for metrics in fold.values())
        assert all(np.isfinite(fold[name]['ic_mean']) for name in ('ridge', 'hgb'))
        assert np.isnan(fold['zero']['ic_mean'])


def test_fit_selects_on_validation_and_scores_holdout_once(synthetic):
    result = fit_return_model(synthetic)

    assert result.candidate in {'ridge', 'hgb'}
    assert len(result.evaluation.folds) == 3
    assert set(result.holdout) == {*BASELINES, result.candidate}
    assert np.isnan(result.holdout['zero']['ic_mean'])
    assert set(result.forecasts.columns) == {
        'symbol',
        'open_time',
        TARGET_COLUMN,
        *BASELINES,
        result.candidate,
    }
    holdout_start = pd.Timestamp('2023-04-01', tz='UTC')
    assert result.forecasts['open_time'].min() >= holdout_start
    assert result.forecasts['open_time'].max() == pd.Timestamp('2023-06-01', tz='UTC')
    assert isinstance(result.beats_baselines(), bool)


def test_return_model_predictions_are_finite_and_match_holdout_forecasts(synthetic):
    result = fit_return_model(synthetic)
    rows = synthetic.frame.loc[:, list(FEATURE_COLUMNS)].tail(10)

    predictions = result.model.predict(rows)

    assert predictions.shape == (len(rows),)
    assert np.isfinite(predictions).all()


def test_zero_prediction_baseline_has_undefined_ic():
    rows = pd.DataFrame(
        {
            'symbol': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
            'open_time': [pd.Timestamp('2023-01-01', tz='UTC')] * 3,
            TARGET_COLUMN: [0.1, 0.2, 0.3],
        }
    )
    assert np.isnan(score(np.zeros(len(rows)), rows)['ic_mean'])


def test_fit_rejects_an_undefined_selection_score(synthetic, monkeypatch):
    from training import returns

    evaluation = SimpleNamespace(
        folds=[{'ridge': {'ic_mean': float('nan')}, 'hgb': {'ic_mean': float('nan')}}],
        predictors=lambda: ['ridge', 'hgb'],
        mean=lambda predictor, metric: float('nan'),
    )

    monkeypatch.setattr(returns, 'cross_validate', lambda frame: evaluation)
    with pytest.raises(ValueError, match='aucun candidat'):
        fit_return_model(synthetic)
