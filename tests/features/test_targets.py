"""Tests de génération des cibles futures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features import target_return
from marketdata.sources import normalize
from tests.conftest import raw_rows


def _candles(rows: int = 8, *, symbol: str = 'BTCUSDT') -> pd.DataFrame:
    return normalize(raw_rows(rows), symbol)


def _as_of_after(frame: pd.DataFrame) -> pd.Timestamp:
    return frame['close_time'].max() + pd.Timedelta(microseconds=1)


def test_target_return_uses_exact_horizon_and_keeps_symbols_separate():
    btc = _candles()
    eth = _candles(symbol='ETHUSDT')
    eth.loc[:, ['open', 'high', 'low', 'close']] = [500, 501, 499, 500]
    candles = pd.concat([btc, eth], ignore_index=True)

    result = target_return(candles, hours=4, as_of=_as_of_after(candles))

    assert result.columns.tolist() == ['symbol', 'open_time', 'target_return']
    btc_target = result.loc[result['symbol'].eq('BTCUSDT')].reset_index(drop=True)
    eth_target = result.loc[result['symbol'].eq('ETHUSDT')].reset_index(drop=True)
    assert btc_target.loc[0, 'target_return'] == pytest.approx(
        np.log(btc.loc[4, 'close'] / btc.loc[0, 'close'])
    )
    assert eth_target.loc[0, 'target_return'] == pytest.approx(0)
    assert btc_target['target_return'].iloc[-4:].isna().all()


def test_missing_exact_target_is_missing_instead_of_shifting_to_next_candle():
    candles = _candles(8).drop(index=4).reset_index(drop=True)

    result = target_return(candles, as_of=_as_of_after(candles))

    assert pd.isna(result.loc[0, 'target_return'])
    assert result.loc[1, 'target_return'] == pytest.approx(
        np.log(candles.loc[4, 'close'] / candles.loc[1, 'close'])
    )


def test_target_is_unavailable_until_future_candle_is_closed():
    candles = _candles(6)
    target_close_time = candles.loc[4, 'close_time']

    at_close = target_return(candles, as_of=target_close_time)
    after_close = target_return(candles, as_of=target_close_time + pd.Timedelta(microseconds=1))

    assert pd.isna(at_close.loc[0, 'target_return'])
    assert after_close.loc[0, 'target_return'] == pytest.approx(
        np.log(candles.loc[4, 'close'] / candles.loc[0, 'close'])
    )


@pytest.mark.parametrize('hours', [0, -1, True, 1.5])
def test_non_positive_or_non_integer_horizon_is_rejected(hours):
    with pytest.raises(ValueError, match='entier positif'):
        target_return(_candles(2), hours=hours)


def test_duplicate_symbol_and_hour_is_rejected():
    candles = _candles(5)
    duplicate = pd.concat([candles, candles.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match='doublons'):
        target_return(duplicate, as_of=_as_of_after(candles))
