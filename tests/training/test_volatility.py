"""Tests du jeu, des métriques et de l'entraînement du modèle de volatilité."""

from __future__ import annotations

import dataclasses
import io
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from features import FEATURE_COLUMNS
from marketdata.sources import normalize
from tests.conftest import raw_rows
from training import figures, log_volatility_mae, prepare_volatility_dataset, qlike
from training.dataset import VolatilityTrainingDataset
from training.volatility import (
    BASELINES,
    VolatilityPyfunc,
    cross_validate,
    fit_volatility_model,
    strategy_scores,
)

# -- métriques -----------------------------------------------------------------------------------


def test_qlike_is_zero_for_a_perfect_forecast():
    realized = np.array([0.01, 0.02, 0.05])
    assert qlike(realized, realized) == pytest.approx(0.0)
    assert log_volatility_mae(realized, realized) == pytest.approx(0.0)


def test_qlike_penalizes_underestimation_more_than_overestimation():
    realized = np.full(4, 0.02)
    under = qlike(realized / 2, realized)
    over = qlike(realized * 2, realized)
    assert under > over > 0
    assert log_volatility_mae(realized / 2, realized) == pytest.approx(
        log_volatility_mae(realized * 2, realized)
    )


def test_qlike_matches_the_variance_formula():
    predicted, realized = np.array([0.02]), np.array([0.03])
    ratio = (0.03 / 0.02) ** 2
    assert qlike(predicted, realized) == pytest.approx(ratio - np.log(ratio) - 1)


@pytest.mark.parametrize(
    ('predicted', 'realized'),
    [
        ([], []),
        ([0.01, 0.02], [0.01]),
        ([0.0], [0.01]),
        ([0.01], [-0.01]),
        ([np.nan], [0.01]),
        ([np.inf], [0.01]),
    ],
)
def test_volatility_losses_reject_invalid_inputs(predicted, realized):
    with pytest.raises(ValueError):
        qlike(predicted, realized)
    with pytest.raises(ValueError):
        log_volatility_mae(predicted, realized)


# -- jeu de données ------------------------------------------------------------------------------


def _candles(rows: int = 220) -> pd.DataFrame:
    candles = normalize(raw_rows(rows), 'BTCUSDT')
    wiggle = 1 + 0.01 * np.sin(np.arange(rows))
    for column in ('open', 'high', 'low', 'close'):
        candles[column] = candles[column] * wiggle
    candles['high'] = candles[['open', 'high', 'low', 'close']].max(axis=1) * 1.001
    candles['low'] = candles[['open', 'high', 'low', 'close']].min(axis=1) * 0.999
    return candles


def test_prepare_volatility_dataset_keeps_complete_rows_with_positive_target():
    candles = _candles()
    as_of = candles.loc[len(candles) - 1, 'close_time'] + pd.Timedelta(microseconds=1)

    result = prepare_volatility_dataset(candles, hours=24, as_of=as_of)

    assert result.frame.columns.tolist() == ['symbol', 'open_time', *FEATURE_COLUMNS, 'target_volatility']
    assert not result.frame.isna().any().any()
    assert (result.frame['target_volatility'] > 0).all()
    assert result.hours == 24
    assert result.frame['open_time'].max() == candles.loc[len(candles) - 1 - 24, 'open_time']


def test_prepare_volatility_dataset_rejects_naive_cutoff_and_bad_candles():
    candles = _candles()
    with pytest.raises(ValueError, match='fuseau'):
        prepare_volatility_dataset(candles, as_of=pd.Timestamp('2024-01-01'))
    candles.loc[0, 'open'] = -1
    with pytest.raises(ValueError, match='invalide'):
        prepare_volatility_dataset(candles)


# -- entraînement --------------------------------------------------------------------------------


@pytest.fixture
def synthetic() -> VolatilityTrainingDataset:
    """Panel horaire de 17 mois où la volatilité future dépend de la volatilité passée."""
    rng = np.random.default_rng(0)
    times = pd.date_range('2022-01-01', '2023-05-31 23:00', freq='h', tz='UTC')
    frames = []
    for symbol, level in (('BTCUSDT', 0.02), ('ETHUSDT', 0.03)):
        n = len(times)
        features = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=list(FEATURE_COLUMNS))
        past = level * np.exp(0.5 * rng.normal(size=n))
        features['realized_vol_24h'] = past
        features['realized_vol_168h'] = level * np.sqrt(7) * np.exp(0.1 * rng.normal(size=n))
        target = past * np.exp(0.1 * rng.normal(size=n))
        frames.append(
            pd.concat(
                [
                    pd.DataFrame({'symbol': symbol, 'open_time': times}),
                    features,
                    pd.Series(target, name='target_volatility'),
                ],
                axis=1,
            )
        )
    frame = pd.concat(frames, ignore_index=True)
    return VolatilityTrainingDataset(
        frame=frame, identity=None, quality_report=None, as_of=pd.Timestamp('2023-06-02', tz='UTC'), hours=24
    )


def test_cross_validation_scores_every_predictor_on_every_fold(synthetic):
    evaluation = cross_validate(synthetic.frame)
    assert len(evaluation.folds) == 2
    assert set(evaluation.predictors()) == {*BASELINES, 'ridge', 'hgb'}
    for fold in evaluation.folds:
        for metrics in fold.values():
            assert set(metrics) == {'qlike', 'mae_log_vol'}
            assert all(np.isfinite(value) for value in metrics.values())


def test_fit_selects_a_learned_model_scored_once_on_the_holdout(synthetic):
    result = fit_volatility_model(synthetic)
    assert result.candidate in {'ridge', 'hgb'}
    assert set(result.holdout) == {*BASELINES, result.candidate}
    # la volatilité future ne dépend que de la volatilité passée : la persistance est forte,
    # le modèle doit au moins faire aussi bien que la moyenne par symbole
    assert result.holdout[result.candidate]['qlike'] < result.holdout['moyenne']['qlike']


def test_served_model_returns_positive_volatility(synthetic):
    result = fit_volatility_model(synthetic)
    rows = synthetic.frame.loc[:, list(FEATURE_COLUMNS)].head(10)
    served = VolatilityPyfunc(result.model.estimator).predict(None, rows)
    np.testing.assert_allclose(served, result.model.predict(rows))
    assert (served > 0).all()


def test_fit_scores_the_volatility_targeting_strategy(synthetic):
    result = fit_volatility_model(synthetic)
    assert set(result.strategy) == {'brut', 'persistance', result.candidate}
    for metrics in result.strategy.values():
        assert {'sharpe', 'max_drawdown', 'total_return', 'fees'} <= set(metrics)
        assert all(np.isfinite(value) for value in metrics.values())
    assert isinstance(result.beats_persistence_strategy(), bool)


def test_strategy_scores_rewards_the_better_volatility_forecast():
    """Rendements à volatilité alternée : la prévision exacte réduit la perte maximale."""
    rng = np.random.default_rng(1)
    times = pd.date_range('2025-01-01', periods=2000, freq='h', tz='UTC')
    sigma = np.where(np.arange(len(times)) % 400 < 200, 0.002, 0.02)
    holdout = pd.DataFrame({'symbol': 'BTCUSDT', 'open_time': times, 'return_1h': rng.normal(0, sigma)})
    oracle = np.roll(sigma, -1)
    flat = np.full(len(times), sigma.mean())
    scores = strategy_scores(holdout, {'oracle': oracle, 'constante': flat}, target=0.002, fee=0.0)
    assert scores['oracle']['max_drawdown'] > scores['constante']['max_drawdown']
    assert scores['oracle']['max_drawdown'] > scores['brut']['max_drawdown']


def test_promote_registers_first_champion_then_compares(synthetic, tmp_path, monkeypatch):
    import mlflow

    from connectors.config import MlflowConfig
    from connectors.mlflow import MlflowConnector
    from lineage import provenance
    from registry import CHAMPION, status
    from tests.connectors.test_mlflow import RELEASED
    from training.volatility import MODEL_NAME, log_training, promote

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(provenance, 'current', lambda: RELEASED)
    uri = f'sqlite:///{(tmp_path / "mlflow.db").as_posix()}'
    connector = MlflowConnector(MlflowConfig(tracking_uri=uri, registry_uri=uri, experiment='e2e'))
    identity = SimpleNamespace(tags=lambda: {'data.symbols': 'BTCUSDT,ETHUSDT'})
    report = SimpleNamespace(summary=lambda: 'aucun problème')
    try:
        result = fit_volatility_model(
            dataclasses.replace(synthetic, identity=identity, quality_report=report)
        )
        _, info = log_training(result, connector)
        decision, version = promote(result, connector, info)
        if result.beats_baselines() and result.beats_persistence_strategy():
            assert decision.alias == CHAMPION
            assert status(connector, MODEL_NAME)[CHAMPION] == str(version.version)
            # même modèle relancé : ne bat pas strictement le champion -> challenger
            _, info = log_training(result, connector)
            again, _ = promote(result, connector, info)
            assert again.alias != CHAMPION
        else:
            assert decision.alias != CHAMPION
    finally:
        mlflow.set_tracking_uri(None)
        mlflow.set_registry_uri(None)


# -- graphiques -------------------------------------------------------------------------------------


def test_fit_keeps_the_data_behind_the_run_figures(synthetic):
    result = fit_volatility_model(synthetic)
    forecasts = result.forecasts
    assert {'symbol', 'open_time', 'realise', *BASELINES, result.candidate} <= set(forecasts.columns)
    assert (forecasts.drop(columns=['symbol', 'open_time']) > 0).all().all()
    assert set(result.equity) == set(result.strategy)
    curve = result.learning_curve
    assert list(curve.columns) == ['iteration', 'train', 'validation', 'qlike_validation']
    assert curve['train'].iloc[-1] < curve['train'].iloc[0]  # le boosting apprend
    assert set(result.importance.index) == set(FEATURE_COLUMNS)


def test_figures_render_and_skip_missing_data(synthetic):
    result = fit_volatility_model(synthetic)
    built = figures.build(result)
    assert set(built) == {
        'courbe_apprentissage',
        'qlike_par_fold',
        'test_final_scores',
        'test_final_qlike_hebdo',
        'test_final_prevision_vs_realise',
        'test_final_dispersion',
        'strategies_valeur_portefeuille',
        'importance_features',
    }
    for figure in built.values():
        figure.savefig(io.BytesIO(), format='png')
    bare = dataclasses.replace(result, forecasts=None, equity={}, learning_curve=None, importance=None)
    assert set(figures.build(bare)) == {'qlike_par_fold', 'test_final_scores'}
    figures.close_all()
