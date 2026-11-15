"""Calcul des cibles futures."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from features._common import closed_as_of
from features.constants import IDENTIFIER_COLUMNS
from marketdata.schema import validate


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

    cutoff = closed_as_of(as_of)
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
