"""Schéma pandera d'une bougie normalisée (sortie de `sources.normalize`).

Le schéma porte les types, les unités et les bornes colonne par colonne. Les règles qui
relient plusieurs lignes ou colonnes (doublons, trous, cohérence OHLC) sont dans `quality`.

| Colonne | Type | Unité |
|---|---|---|
| symbol | str | paire spot, ex. `BTCUSDT` |
| open_time, close_time | datetime UTC (µs) | début / fin incluse de la bougie |
| open, high, low, close | float64 | prix en devise de cotation (USDT) |
| volume, taker_buy_base_volume | float64 | quantité de l'actif de base |
| quote_volume, taker_buy_quote_volume | float64 | montant en USDT |
| trades | int64 | nombre de transactions |
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

from marketdata.sources import PRICE_FIELDS
from marketdata.universe import SYMBOLS

TIMESTAMP = pd.DatetimeTZDtype(unit='us', tz='UTC')

CANDLE_SCHEMA = pa.DataFrameSchema(
    {
        'symbol': pa.Column(str, pa.Check.isin(SYMBOLS)),
        'open_time': pa.Column(TIMESTAMP),
        **{name: pa.Column('float64', pa.Check.gt(0)) for name in PRICE_FIELDS},
        'volume': pa.Column('float64', pa.Check.ge(0)),
        'close_time': pa.Column(TIMESTAMP),
        'quote_volume': pa.Column('float64', pa.Check.ge(0)),
        'trades': pa.Column('int64', pa.Check.ge(0)),
        'taker_buy_base_volume': pa.Column('float64', pa.Check.ge(0)),
        'taker_buy_quote_volume': pa.Column('float64', pa.Check.ge(0)),
    },
    strict=True,
    ordered=True,
    name='candle',
)

COLUMNS = tuple(CANDLE_SCHEMA.columns)


def validate(frame: pd.DataFrame) -> pd.DataFrame:
    """Valide le schéma en remontant toutes les violations (pandera SchemaErrors)."""
    return CANDLE_SCHEMA.validate(frame, lazy=True)
