"""Tests de l'inférence et des retours (niveau 3) sur bougies synthétiques."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from features import FEATURE_COLUMNS, target_volatility
from marketdata.sources import normalize
from serving import health_rows, latest_features, pending_feedback, predict_latest, realized_feedback
from tests.conftest import raw_rows
from training import qlike

SYMBOLS = ('BTCUSDT', 'ETHUSDT')


def _candles(rows: int = 260) -> pd.DataFrame:
    frames = []
    for offset, symbol in enumerate(SYMBOLS):
        candles = normalize(raw_rows(rows), symbol)
        wiggle = 1 + 0.01 * np.sin(np.arange(rows) * (1 + offset))
        for column in ('open', 'high', 'low', 'close'):
            candles[column] = candles[column] * wiggle
        candles['high'] = candles[['open', 'high', 'low', 'close']].max(axis=1) * 1.001
        candles['low'] = candles[['open', 'high', 'low', 'close']].min(axis=1) * 0.999
        frames.append(candles)
    return pd.concat(frames, ignore_index=True)


def _as_of(candles: pd.DataFrame) -> pd.Timestamp:
    return candles['close_time'].max() + pd.Timedelta(microseconds=1)


def _constant(value: float):
    return lambda rows: np.full(len(rows), value)


# -- inférence -----------------------------------------------------------------------------------


def test_latest_features_keeps_the_last_complete_row_of_each_symbol():
    candles = _candles()
    latest = latest_features(candles, as_of=_as_of(candles))

    assert latest['symbol'].tolist() == list(SYMBOLS)
    assert (latest['open_time'] == candles['open_time'].max()).all()
    assert not latest.loc[:, list(FEATURE_COLUMNS)].isna().any().any()


def test_latest_features_ignores_candles_not_closed_at_as_of():
    candles = _candles()
    as_of = _as_of(candles) - pd.Timedelta(hours=1)

    latest = latest_features(candles, as_of=as_of)

    assert (latest['open_time'] == candles['open_time'].max() - pd.Timedelta(hours=1)).all()


def test_latest_features_rejects_a_too_short_history():
    candles = _candles(rows=30)
    with pytest.raises(ValueError, match='historique trop court'):
        latest_features(candles, as_of=_as_of(candles))


def test_predict_latest_returns_prediction_baseline_and_positions():
    candles = _candles()
    seen = []

    def predict(rows):
        seen.append(rows.columns.tolist())
        return np.full(len(rows), 0.01)

    forecast = predict_latest(predict, candles, as_of=_as_of(candles))

    assert seen == [list(FEATURE_COLUMNS)]
    frame = forecast.frame
    assert frame.columns.tolist() == ['symbol', 'open_time', 'prediction', 'baseline', 'position']
    assert frame['symbol'].tolist() == list(SYMBOLS)
    assert forecast.open_time == candles['open_time'].max()
    assert (frame['prediction'] == 0.01).all()
    assert (frame['baseline'] > 0).all()
    assert (frame['position'] > 0).all()
    assert frame['position'].sum() <= 1.0 + 1e-9


def test_predict_latest_lowers_exposure_when_the_forecast_is_riskier():
    candles = _candles()
    calm = predict_latest(_constant(0.001), candles, as_of=_as_of(candles)).frame
    risky = predict_latest(_constant(1.0), candles, as_of=_as_of(candles)).frame

    assert (risky['position'] < calm['position']).all()


def test_predict_latest_skips_positions_for_the_challenger():
    candles = _candles()
    frame = predict_latest(_constant(0.01), candles, as_of=_as_of(candles), with_positions=False).frame

    assert frame['position'].isna().all()


def test_predict_latest_serves_only_symbols_up_to_date():
    candles = _candles()
    late = (candles['symbol'] == 'ETHUSDT') & (candles['open_time'] == candles['open_time'].max())
    frame = predict_latest(_constant(0.01), candles.loc[~late], as_of=_as_of(candles)).frame

    assert frame['symbol'].tolist() == ['BTCUSDT']


@pytest.mark.parametrize('output', [[0.01], [0.01, -0.01], [0.01, np.nan], [0.0, 0.01]])
def test_predict_latest_rejects_invalid_model_output(output):
    candles = _candles()
    with pytest.raises(ValueError, match='volatilité finie et positive'):
        predict_latest(lambda rows: np.array(output), candles, as_of=_as_of(candles))


def test_predict_latest_requires_an_aware_as_of():
    candles = _candles()
    with pytest.raises(ValueError, match='fuseau'):
        predict_latest(_constant(0.01), candles, as_of=pd.Timestamp('2025-01-10'))


# -- retours -------------------------------------------------------------------------------------


def _predictions(open_times) -> list[dict]:
    rows = []
    for i, (symbol, open_time) in enumerate((s, t) for t in open_times for s in SYMBOLS):
        rows.append(
            {
                'prediction_id': i + 1,
                'symbol': symbol,
                'open_time': open_time.to_pydatetime(),
                'horizon_hours': 24,
                'prediction': 0.01,
                'baseline': 0.02,
            }
        )
    return rows


def test_pending_feedback_waits_for_the_horizon_to_close():
    candles = _candles()
    t0 = candles['open_time'].min() + pd.Timedelta(hours=100)
    rows = _predictions([t0])

    assert pending_feedback(rows, t0 + pd.Timedelta(hours=24)).empty
    due = pending_feedback(rows, t0 + pd.Timedelta(hours=25))
    assert due['prediction_id'].tolist() == [1, 2]
    assert pending_feedback([], t0).empty


def test_realized_feedback_matches_the_training_target():
    candles = _candles()
    t0 = candles['open_time'].min() + pd.Timedelta(hours=100)
    as_of = _as_of(candles)
    pending = pending_feedback(_predictions([t0]), as_of)

    scored = realized_feedback(pending, candles, as_of=as_of)

    target = target_volatility(candles, hours=24, as_of=as_of)
    expected = target.loc[target['open_time'] == t0].set_index('symbol')['target_volatility']
    assert scored['prediction_id'].tolist() == [1, 2]
    assert scored['realized'].tolist() == pytest.approx(expected.loc[list(SYMBOLS)].tolist())
    assert scored.loc[0, 'loss'] == pytest.approx(qlike([0.01], [scored.loc[0, 'realized']]))
    assert scored.loc[0, 'baseline_loss'] == pytest.approx(qlike([0.02], [scored.loc[0, 'realized']]))


def test_realized_feedback_skips_predictions_with_a_gap_in_the_window():
    candles = _candles()
    t0 = candles['open_time'].min() + pd.Timedelta(hours=100)
    hole = (candles['symbol'] == 'ETHUSDT') & (candles['open_time'] == t0 + pd.Timedelta(hours=5))
    as_of = _as_of(candles)
    pending = pending_feedback(_predictions([t0]), as_of)

    scored = realized_feedback(pending, candles.loc[~hole], as_of=as_of)

    assert scored['prediction_id'].tolist() == [1]
    assert realized_feedback(pending.iloc[0:0], candles, as_of=as_of).empty


def _scored(candles: pd.DataFrame, loss: float, baseline_loss: float, **extra) -> pd.DataFrame:
    start = candles['open_time'].max() - pd.Timedelta(hours=48)
    rows = []
    for hour in range(24):
        for symbol in SYMBOLS:
            rows.append(
                {
                    'model_name': 'volatilite',
                    'version': 3,
                    'alias': 'champion',
                    'mode': 'live',
                    'symbol': symbol,
                    'open_time': start + pd.Timedelta(hours=hour),
                    'position': 0.5,
                    'loss': loss,
                    'baseline_loss': baseline_loss,
                    **extra,
                }
            )
    return pd.DataFrame(rows)


def _day(candles: pd.DataFrame) -> date:
    return (candles['open_time'].max() - pd.Timedelta(hours=25)).date()


def test_health_rows_flags_a_version_worse_than_persistence():
    candles = _candles()
    day = _day(candles)

    good = health_rows(_scored(candles, 0.1, 0.2), candles, day=day, window_hours=72)
    bad = health_rows(_scored(candles, 0.3, 0.2), candles, day=day, window_hours=72)

    assert len(good) == 1
    row = good[0]
    assert (row['model_name'], row['version'], row['mode'], row['day']) == ('volatilite', 3, 'live', day)
    assert row['metric'] == 'qlike'
    assert row['samples'] > 0
    assert row['value'] == pytest.approx(0.1)
    assert row['baseline_value'] == pytest.approx(0.2)
    assert row['degraded'] is False
    assert bad[0]['degraded'] is True
    assert isinstance(row['pnl'], float)
    assert isinstance(row['baseline_pnl'], float)
    assert row['details']['aliases'] == ['champion']


def test_health_rows_without_positions_has_no_pnl():
    candles = _candles()
    scored = _scored(candles, 0.1, 0.2, alias='challenger').assign(position=np.nan)

    row = health_rows(scored, candles, day=_day(candles), window_hours=72)[0]

    assert row['pnl'] is None
    assert row['baseline_pnl'] is None


def test_health_rows_ignores_predictions_outside_the_window():
    candles = _candles()
    scored = _scored(candles, 0.1, 0.2)
    assert health_rows(scored, candles, day=date(2020, 1, 1)) == []
    assert health_rows(scored.iloc[0:0], candles, day=date(2020, 1, 1)) == []
    with pytest.raises(ValueError):
        health_rows(scored, candles, day=date(2020, 1, 1), window_hours=0)
