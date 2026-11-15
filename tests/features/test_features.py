"""Tests du calcul commun de features."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from features import FEATURE_COLUMNS, build_features
from marketdata.sources import normalize
from tests.conftest import START_MS, raw_rows


def _candles(rows: int = 180, *, symbol: str = 'BTCUSDT') -> pd.DataFrame:
    return normalize(raw_rows(rows), symbol)


def _as_of_after(frame: pd.DataFrame) -> datetime:
    return frame['close_time'].max().to_pydatetime() + timedelta(microseconds=1)


def test_build_features_returns_ordered_numeric_features_and_identifiers():
    candles = _candles()

    result = build_features(candles, as_of=_as_of_after(candles))

    assert result.columns.tolist() == ['symbol', 'open_time', *FEATURE_COLUMNS]
    assert result[['symbol', 'open_time']].equals(candles[['symbol', 'open_time']])
    assert all(pd.api.types.is_numeric_dtype(result[column]) for column in FEATURE_COLUMNS)
    assert result['symbol_BTCUSDT'].eq(1).all()
    assert result['symbol_ETHUSDT'].eq(0).all()
    assert result.loc[1, 'return_1h'] == pytest.approx(np.log(102 / 101))
    assert result.loc[0, 'return_168h'] != result.loc[0, 'return_168h']


def test_features_match_formulas_and_volume_baselines_exclude_current_candle():
    candles = _candles()
    candles.loc[100, 'quote_volume'] = 1_000_000
    candles.loc[100, 'trades'] = 1_000_000

    result = build_features(candles, as_of=_as_of_after(candles))
    row = result.iloc[100]
    returns = np.log(candles['close'] / candles['close'].shift(1))

    assert row['return_4h'] == pytest.approx(np.log(candles.loc[100, 'close'] / candles.loc[96, 'close']))
    assert row['realized_vol_6h'] == pytest.approx(np.sqrt(np.square(returns.iloc[95:101]).sum()))
    assert row['candle_range'] == pytest.approx(np.log(candles.loc[100, 'high'] / candles.loc[100, 'low']))
    assert row['candle_body'] == pytest.approx(np.log(candles.loc[100, 'close'] / candles.loc[100, 'open']))
    expected_baseline = candles.loc[76:99, 'quote_volume'].median()
    assert row['quote_volume_ratio_24h'] == pytest.approx(
        candles.loc[100, 'quote_volume'] / expected_baseline
    )
    assert row['trade_count_ratio_24h'] == pytest.approx(
        candles.loc[100, 'trades'] / candles.loc[76:99, 'trades'].median()
    )
    assert row['taker_buy_ratio'] == pytest.approx(
        candles.loc[100, 'taker_buy_quote_volume'] / candles.loc[100, 'quote_volume']
    )


def test_market_and_relative_returns_are_transversal():
    btc = _candles(5, symbol='BTCUSDT')
    eth = _candles(5, symbol='ETHUSDT')
    eth['open'] = 100
    eth['high'] = 102
    eth['low'] = 98
    eth['close'] = 100
    all_candles = pd.concat([btc, eth], ignore_index=True).sort_values(
        ['symbol', 'open_time'], ignore_index=True
    )

    result = build_features(all_candles, as_of=_as_of_after(all_candles))
    at_last_hour = result.loc[result['open_time'].eq(btc['open_time'].iloc[-1])]
    expected_median = np.median([np.log(btc['close'].iloc[-1] / btc['close'].iloc[-2]), 0])

    assert at_last_hour['market_return_1h'].eq(expected_median).all()
    btc_row = at_last_hour.loc[at_last_hour['symbol'].eq('BTCUSDT')].iloc[0]
    assert btc_row['relative_return_1h'] == pytest.approx(btc_row['return_1h'] - expected_median)


def test_open_candles_are_excluded_and_do_not_affect_previous_features():
    candles = _candles(3)
    as_of = candles.loc[2, 'open_time'] + pd.Timedelta(minutes=30)

    result = build_features(candles, as_of=as_of)

    assert result['open_time'].tolist() == candles.loc[:1, 'open_time'].tolist()


def test_future_candle_changes_do_not_change_features_at_earlier_hours():
    candles = _candles(180)
    changed = candles.copy()
    future = changed['open_time'] >= changed.loc[170, 'open_time']
    changed.loc[future, ['open', 'high', 'low', 'close']] *= 10
    changed.loc[future, ['volume', 'quote_volume', 'trades']] *= 100
    changed.loc[future, ['taker_buy_base_volume', 'taker_buy_quote_volume']] *= 100

    original_features = build_features(candles, as_of=_as_of_after(candles))
    changed_features = build_features(changed, as_of=_as_of_after(changed))

    earlier = original_features['open_time'] < candles.loc[170, 'open_time']
    pd.testing.assert_frame_equal(
        original_features.loc[earlier].reset_index(drop=True),
        changed_features.loc[earlier].reset_index(drop=True),
    )


def test_missing_hour_is_not_treated_as_an_adjacent_observation():
    candles = _candles(10).drop(index=5).reset_index(drop=True)

    result = build_features(candles, as_of=_as_of_after(candles))
    hour_after_gap = result.loc[result['open_time'].eq(candles.loc[5, 'open_time'])].iloc[0]

    assert np.isnan(hour_after_gap['return_1h'])
    assert np.isnan(hour_after_gap['realized_vol_6h'])


def test_zero_volume_uses_zero_ratios_and_short_history_stays_missing():
    candles = _candles(26)
    candles.loc[:24, ['quote_volume', 'taker_buy_quote_volume', 'trades']] = 0

    result = build_features(candles, as_of=_as_of_after(candles))

    assert np.isnan(result.loc[23, 'quote_volume_ratio_24h'])
    assert result.loc[24, 'quote_volume_ratio_24h'] == 0
    assert result.loc[24, 'trade_count_ratio_24h'] == 0
    assert result.loc[24, 'taker_buy_ratio'] == 0


def test_duplicate_symbol_and_hour_is_rejected():
    candles = _candles(2)
    duplicate = pd.concat([candles, candles.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match='doublons'):
        build_features(duplicate, as_of=_as_of_after(candles))


def test_naive_as_of_is_rejected():
    candles = _candles(2)

    with pytest.raises(ValueError, match='fuseau horaire'):
        build_features(candles, as_of=datetime(2025, 1, 1))


def test_empty_input_returns_stable_schema():
    candles = _candles(1).iloc[:0]

    result = build_features(candles, as_of=pd.Timestamp(START_MS, unit='ms', tz='UTC'))

    assert result.empty
    assert result.columns.tolist() == ['symbol', 'open_time', *FEATURE_COLUMNS]
