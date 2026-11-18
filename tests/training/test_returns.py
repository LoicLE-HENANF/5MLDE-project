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
    RendementPyfunc,
    _candidates,
    cross_validate,
    fit_return_model,
    holdout_frame,
    log_training,
    score,
    strategy_backtests,
)


@pytest.fixture
def synthetic() -> ReturnTrainingDataset:
    """Panel multi-symbole de 18 mois, avec un signal transversal déterministe."""
    rng = np.random.default_rng(42)
    month_starts = pd.date_range('2022-01-01', '2023-06-01', freq='MS', tz='UTC')
    times = month_starts.repeat(6) + pd.to_timedelta(
        np.tile(np.arange(6), len(month_starts)),
        unit='h',
    )
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
    assert result.forecasts['open_time'].max() == pd.Timestamp('2023-06-01 05:00', tz='UTC')
    assert isinstance(result.beats_baselines(), bool)
    assert set(result.equity) == set(result.strategy)
    assert result.strategy


def test_return_model_predictions_are_finite_and_match_holdout_forecasts(synthetic):
    result = fit_return_model(synthetic)
    rows = synthetic.frame.loc[:, list(FEATURE_COLUMNS)].tail(10)

    predictions = result.model.predict(rows)

    assert predictions.shape == (len(rows),)
    assert np.isfinite(predictions).all()


def test_pyfunc_uses_only_model_features():
    class Predictor:
        def predict(self, frame):
            return frame.iloc[:, 0].to_numpy()

    sample = pd.DataFrame({column: [float(i)] for i, column in enumerate(FEATURE_COLUMNS)})
    sample.loc[0, FEATURE_COLUMNS[0]] = 42.0
    predictions = RendementPyfunc(Predictor()).predict(None, sample)

    assert predictions.shape == (1,)
    assert predictions[0] == 42.0


def test_strategy_backtests_score_long_short_and_long_cash(synthetic):
    holdout = holdout_frame(synthetic)
    predictions = holdout[TARGET_COLUMN].to_numpy()

    results = strategy_backtests(holdout, {'modele': predictions}, candidate='modele', fee=0.0)

    assert {'brut', 'modele_long_short', 'modele_long_cash'} <= set(results)
    assert all(np.isfinite(result.metrics['sharpe']) for result in results.values())
    assert all(result.metrics['hours'] > 0 for result in results.values())


def test_promote_compares_negative_ic_and_long_short_sharpe(synthetic, monkeypatch):
    import registry
    from training import returns

    calls = {}
    decision = object()
    result = SimpleNamespace(
        candidate='ridge',
        dataset=synthetic,
        holdout={'ridge': {'ic_mean': 0.4}},
        strategy={'ridge_long_short': {'sharpe': 7.0}},
        beats_baselines=lambda: True,
        beats_long_short_strategy=lambda: True,
    )
    monkeypatch.setattr(returns, 'champion_scores', lambda connector, holdout: (-0.3, 5.0))
    monkeypatch.setattr(registry, 'decide', lambda **kwargs: calls.update(kwargs) or decision)
    monkeypatch.setattr(registry, 'publish', lambda *args, **kwargs: 'version-1')

    actual_decision, version = returns.promote(result, object(), object())

    assert actual_decision is decision
    assert version == 'version-1'
    assert calls == {
        'level1': True,
        'level2': True,
        'score': -0.4,
        'champion_score': -0.3,
    }


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


def test_log_training_records_provenance_metrics_and_a_loadable_model(synthetic, tmp_path, monkeypatch):
    import mlflow

    from connectors.config import MlflowConfig
    from connectors.mlflow import MlflowConnector
    from lineage import provenance
    from tests.connectors.test_mlflow import RELEASED

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(provenance, 'current', lambda: RELEASED)
    identity = SimpleNamespace(tags=lambda: {'data.symbols': 'BTCUSDT,ETHUSDT'})
    report = SimpleNamespace(summary=lambda: 'aucun problème')
    dataset = synthetic.__class__(
        frame=synthetic.frame,
        identity=identity,
        quality_report=report,
        as_of=synthetic.as_of,
    )
    uri = f'sqlite:///{(tmp_path / "mlflow.db").as_posix()}'
    connector = MlflowConnector(MlflowConfig(tracking_uri=uri, registry_uri=uri, experiment='rendement'))

    try:
        result = fit_return_model(dataset)
        run_id, info = log_training(result, connector)

        run = connector.client.get_run(run_id)
        assert run.data.tags['data.symbols'] == 'BTCUSDT,ETHUSDT'
        assert run.data.tags['modele'] == 'rendement'
        assert run.data.params['horizon_heures'] == '4'
        assert 'test_bat_references' in run.data.metrics
        model = mlflow.pyfunc.load_model(info.model_uri)
        predictions = model.predict(dataset.frame.loc[:, list(FEATURE_COLUMNS)].head(3))
        assert predictions.shape == (3,)
        assert np.isfinite(predictions).all()
    finally:
        mlflow.set_tracking_uri(None)
        mlflow.set_registry_uri(None)
