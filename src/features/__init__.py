"""Calcul des features communes à l'entraînement et à l'inférence."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from marketdata.schema import validate
from marketdata.universe import INTERVAL_DELTA, SYMBOLS

SYMBOL_FEATURE_COLUMNS = tuple(f'symbol_{symbol}' for symbol in SYMBOLS)
BASE_FEATURE_COLUMNS = (
    'return_1h',
    'return_4h',
    'return_12h',
    'return_24h',
    'return_72h',
    'return_168h',
    'realized_vol_6h',
    'realized_vol_24h',
    'realized_vol_72h',
    'realized_vol_168h',
    'candle_range',
    'candle_body',
    'close_location',
    'quote_volume_ratio_24h',
    'taker_buy_ratio',
    'trade_count_ratio_24h',
    'market_return_1h',
    'market_return_4h',
    'market_return_24h',
    'relative_return_1h',
    'relative_return_4h',
    'relative_return_24h',
    'hour_sin',
    'hour_cos',
    'weekday_sin',
    'weekday_cos',
)
FEATURE_COLUMNS = (*BASE_FEATURE_COLUMNS, *SYMBOL_FEATURE_COLUMNS)
IDENTIFIER_COLUMNS = ('symbol', 'open_time')
_RETURN_HORIZONS = (1, 4, 12, 24, 72, 168)
_VOLATILITY_HORIZONS = (6, 24, 72, 168)
_CONTEXT_HORIZONS = (1, 4, 24)
_HOUR = pd.Timedelta(hours=1)


def _closed_as_of(as_of: datetime | pd.Timestamp | None) -> pd.Timestamp:
    if as_of is None:
        return pd.Timestamp.now(tz='UTC')
    cutoff = pd.Timestamp(as_of)
    if cutoff.tzinfo is None:
        raise ValueError('as_of doit inclure un fuseau horaire')
    return cutoff.tz_convert('UTC')


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

    for hours in _RETURN_HORIZONS:
        features[f'return_{hours}h'] = np.log(close / close.shift(hours))

    for hours in _VOLATILITY_HORIZONS:
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

    closed = candles.loc[candles['close_time'] < _closed_as_of(as_of)].copy()
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
    for hours in _CONTEXT_HORIZONS:
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


def target_return(
    candles: pd.DataFrame, hours: int = 4, *, as_of: datetime | pd.Timestamp | None = None
) -> pd.DataFrame:
    """Calcule le rendement futur exact par symbole, sans décaler en cas de bougie manquante.

    Chaque ligne conserve ses identifiants `(symbol, open_time)` et reçoit la colonne
    `target_return = log(close[t + hours] / close[t])`. La cible vaut NaN si la bougie de
    départ ou la bougie cible n'est pas clôturée à `as_of`, ou si cette dernière est absente.
    """
    if isinstance(hours, bool) or not isinstance(hours, int) or hours < 1:
        raise ValueError(f'hours doit être un entier positif, reçu : {hours!r}')
    validate(candles)
    if candles.duplicated(['symbol', 'open_time']).any():
        raise ValueError('doublons sur la clé (symbol, open_time)')

    cutoff = _closed_as_of(as_of)
    ordered = candles.sort_values(['symbol', 'open_time'], ignore_index=True)
    closed = ordered.loc[ordered['close_time'] < cutoff]
    closes = closed.set_index(['symbol', 'open_time'])['close']
    target_times = ordered['open_time'] + pd.Timedelta(hours=hours)
    target_keys = pd.MultiIndex.from_arrays([ordered['symbol'], target_times], names=['symbol', 'open_time'])
    future_closes = closes.reindex(target_keys).to_numpy(dtype='float64')

    valid = ordered['close_time'].lt(cutoff).to_numpy() & np.isfinite(future_closes)
    values = np.full(len(ordered), np.nan, dtype='float64')
    values[valid] = np.log(future_closes[valid] / ordered.loc[valid, 'close'].to_numpy())

    result = ordered.loc[:, list(IDENTIFIER_COLUMNS)].copy()
    result['target_return'] = values
    return result


__all__ = [
    'BASE_FEATURE_COLUMNS',
    'FEATURE_COLUMNS',
    'IDENTIFIER_COLUMNS',
    'build_features',
    'target_return',
]
