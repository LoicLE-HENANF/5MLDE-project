"""Calcul des features communes à l'entraînement et à l'inférence."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from features._common import closed_as_of
from features.constants import (
    CONTEXT_HORIZONS,
    FEATURE_COLUMNS,
    IDENTIFIER_COLUMNS,
    RETURN_HORIZONS,
    SYMBOL_FEATURE_COLUMNS,
    VOLATILITY_HORIZONS,
)
from marketdata.schema import validate
from marketdata.universe import INTERVAL_DELTA, SYMBOLS


def _empty_result() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'symbol': pd.Series(dtype='str'),
            'open_time': pd.Series(dtype='datetime64[us, UTC]'),
            **{name: pd.Series(dtype='float64') for name in FEATURE_COLUMNS},
        },
        columns=[*IDENTIFIER_COLUMNS, *FEATURE_COLUMNS],
    )


def _features_for_symbol(frame: pd.DataFrame, timeline: pd.DatetimeIndex) -> pd.DataFrame:
    candles = frame.set_index('open_time').reindex(timeline)
    close = candles['close']
    returns = pd.Series(np.log(close / close.shift(1)), index=timeline)
    features = pd.DataFrame(index=timeline)

    for hours in RETURN_HORIZONS:
        features[f'return_{hours}h'] = np.log(close / close.shift(hours))

    for hours in VOLATILITY_HORIZONS:
        features[f'realized_vol_{hours}h'] = np.sqrt(returns.pow(2).rolling(hours, min_periods=hours).sum())

    features['candle_range'] = np.log(candles['high'] / candles['low'])
    features['candle_body'] = np.log(candles['close'] / candles['open'])
    spread = candles['high'] - candles['low']
    features['close_location'] = (
        (2 * candles['close'] - candles['high'] - candles['low']).div(spread.where(spread.ne(0))).fillna(0)
    )

    for column, feature in (
        ('quote_volume', 'quote_volume_ratio_24h'),
        ('trades', 'trade_count_ratio_24h'),
    ):
        baseline = candles[column].shift(1).rolling(24, min_periods=24).median()
        ratio = candles[column].div(baseline.where(baseline.ne(0)))
        features[feature] = ratio.mask(baseline.eq(0), 0)

    features['taker_buy_ratio'] = (
        candles['taker_buy_quote_volume']
        .div(candles['quote_volume'].where(candles['quote_volume'].ne(0)))
        .fillna(0)
    )

    hour = timeline.hour
    weekday = timeline.day_of_week
    features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    features['weekday_sin'] = np.sin(2 * np.pi * weekday / 7)
    features['weekday_cos'] = np.cos(2 * np.pi * weekday / 7)
    return features


def build_features(candles: pd.DataFrame, *, as_of: datetime | pd.Timestamp | None = None) -> pd.DataFrame:
    """Construit les features des bougies clôturées, sans utiliser d'information future.

    L'entrée respecte le schéma `marketdata.schema`. Les heures absentes sont réintroduites
    pendant les calculs pour qu'elles ne soient pas confondues avec des observations consécutives.
    Les lignes absentes de l'entrée ne sont pas créées dans le résultat. Les features nécessitant
    un historique incomplet restent NaN.

    `as_of` permet de fixer l'instant d'observation ; par défaut, l'heure UTC courante est utilisée.
    """
    validate(candles)
    if candles.duplicated(['symbol', 'open_time']).any():
        raise ValueError('doublons sur la clé (symbol, open_time)')

    closed = candles.loc[candles['close_time'] < closed_as_of(as_of)].copy()
    if closed.empty:
        return _empty_result()

    closed = closed.sort_values(['symbol', 'open_time'], ignore_index=True)
    start = closed['open_time'].min()
    end = closed['open_time'].max()
    timeline = pd.date_range(start, end, freq=INTERVAL_DELTA)

    symbol_features: list[pd.DataFrame] = []
    for _symbol, group in closed.groupby('symbol', sort=True):
        symbol_features.append(_features_for_symbol(group, timeline))
    symbols = sorted(closed['symbol'].unique())
    calculated = pd.concat(symbol_features, keys=symbols, names=['symbol', 'open_time'])

    calculated['market_return_1h'] = calculated.groupby(level='open_time')['return_1h'].transform('median')
    calculated['market_return_4h'] = calculated.groupby(level='open_time')['return_4h'].transform('median')
    calculated['market_return_24h'] = calculated.groupby(level='open_time')['return_24h'].transform('median')
    for hours in CONTEXT_HORIZONS:
        calculated[f'relative_return_{hours}h'] = (
            calculated[f'return_{hours}h'] - calculated[f'market_return_{hours}h']
        )

    keys = pd.MultiIndex.from_frame(closed[['symbol', 'open_time']])
    output = calculated.reindex(keys).copy()
    output.index = keys
    output = output.reset_index()
    for symbol, column in zip(SYMBOLS, SYMBOL_FEATURE_COLUMNS, strict=True):
        output[column] = (output['symbol'] == symbol).astype('float64')
    return output.loc[:, [*IDENTIFIER_COLUMNS, *FEATURE_COLUMNS]]
