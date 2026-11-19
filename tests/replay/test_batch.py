"""Le calcul en lot du rejeu doit redonner, heure par heure, ce que le live aurait servi."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from replay import batch
from serving import predict_latest, predict_latest_returns
from tests.serving.test_serving import _candles

HOUR = pd.Timedelta(hours=1)


def _signal(rows: pd.DataFrame) -> np.ndarray:
    """Prévision dépendant des features : un écart de features entre lot et live se verrait."""
    return 0.01 + rows.abs().sum(axis=1).to_numpy() * 1e-3


def _signed(rows: pd.DataFrame) -> np.ndarray:
    return rows['return_4h'].to_numpy() - rows['return_1h'].to_numpy() * 0.5


def _window(candles: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    times = sorted(candles['open_time'].unique())
    return pd.Timestamp(times[-24]), pd.Timestamp(times[-1]) + HOUR


def test_month_starts_are_consecutive_utc_months():
    starts = batch.month_starts('2025-11', 3)

    assert [s.strftime('%Y-%m') for s in starts] == ['2025-11', '2025-12', '2026-01']
    assert all(str(s.tz) == 'UTC' for s in starts)
    with pytest.raises(ValueError):
        batch.month_starts('2025-11', 0)


def test_closed_before_drops_the_candle_still_open():
    candles = _candles()
    last = candles['open_time'].max()

    assert candles.loc[candles['open_time'] == last].shape[0] > 0
    assert (batch.closed_before(candles, last)['open_time'] < last).all()
    assert batch.closed_before(candles, last + HOUR)['open_time'].max() == last


def test_month_features_ignore_candles_after_the_end():
    candles = _candles()
    start, end = _window(candles)
    cut = start + 6 * HOUR

    full = batch.hours_of(batch.month_features(candles, start, end), start, cut)
    truncated = batch.hours_of(batch.month_features(candles, start, cut), start, cut)

    pd.testing.assert_frame_equal(full.reset_index(drop=True), truncated.reset_index(drop=True))


def test_volatility_rows_match_live_inference_hour_by_hour():
    candles = _candles()
    start, end = _window(candles)
    features = batch.month_features(candles, start, end)
    rows = batch.volatility_rows(
        features, batch.predict_rows(_signal, features), start, end, with_positions=True
    )

    assert rows['open_time'].nunique() == 24
    for stamp in sorted(rows['open_time'].unique())[::5]:
        live = predict_latest(_signal, candles, as_of=pd.Timestamp(stamp) + HOUR).frame
        lot = rows.loc[rows['open_time'] == stamp].reset_index(drop=True)
        pd.testing.assert_frame_equal(lot, live.reset_index(drop=True), check_dtype=False)


def test_return_rows_match_live_inference_including_staggered_positions():
    candles = _candles()
    start, end = _window(candles)
    features = batch.month_features(candles, start, end)
    rows = batch.return_rows(
        features,
        batch.predict_rows(_signed, features),
        start,
        end,
        horizon_hours=4,
        with_positions=True,
    )

    for stamp in sorted(rows['open_time'].unique())[::5]:
        live = predict_latest_returns(_signed, candles, as_of=pd.Timestamp(stamp) + HOUR, horizon_hours=4)
        lot = rows.loc[rows['open_time'] == stamp].reset_index(drop=True)
        pd.testing.assert_frame_equal(lot, live.frame.reset_index(drop=True), check_dtype=False)


def test_rows_without_positions_for_the_challenger():
    candles = _candles()
    start, end = _window(candles)
    features = batch.month_features(candles, start, end)
    predicted = batch.predict_rows(_signal, features)

    assert (
        batch.volatility_rows(features, predicted, start, end, with_positions=False)['position'].isna().all()
    )
    assert (
        batch.return_rows(features, predicted, start, end, horizon_hours=4, with_positions=False)['position']
        .isna()
        .all()
    )


def test_predict_rows_rejects_a_wrong_shape():
    features = batch.month_features(_candles(), *_window(_candles()))

    with pytest.raises(ValueError):
        batch.predict_rows(lambda rows: np.zeros(3), features)
