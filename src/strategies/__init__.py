"""Stratégies simulées : positions dérivées des prédictions et backtest horaire frais inclus."""

from strategies.backtest import BacktestResult, backtest, next_returns, summarize
from strategies.positions import (
    DEFAULT_FEE,
    equal_weight,
    long_cash,
    long_short,
    scale_by_volatility,
    staggered,
    to_wide,
)

__all__ = [
    'DEFAULT_FEE',
    'BacktestResult',
    'backtest',
    'equal_weight',
    'long_cash',
    'long_short',
    'next_returns',
    'scale_by_volatility',
    'staggered',
    'summarize',
    'to_wide',
]
