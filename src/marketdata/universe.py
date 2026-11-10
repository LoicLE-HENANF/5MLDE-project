"""Univers étudié : 5 paires spot cotées en USDT, bougies 1 h, historique depuis 2022-01."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

SYMBOLS = ('BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT')
INTERVAL = '1h'
INTERVAL_DELTA = timedelta(hours=1)
HISTORY_START = datetime(2022, 1, 1, tzinfo=UTC)


def check_symbol(symbol: str) -> str:
    """Renvoie le symbole s'il appartient à l'univers, sinon ValueError."""
    if symbol not in SYMBOLS:
        raise ValueError(f'symbole hors univers : {symbol!r} (attendu : {", ".join(SYMBOLS)})')
    return symbol
