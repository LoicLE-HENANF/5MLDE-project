"""Noms et horizons des features."""

from __future__ import annotations

from marketdata.universe import SYMBOLS

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
RETURN_HORIZONS = (1, 4, 12, 24, 72, 168)
VOLATILITY_HORIZONS = (6, 24, 72, 168)
CONTEXT_HORIZONS = (1, 4, 24)
