"""Backtest horaire de positions simulées (paper trading), frais inclus.

Convention temporelle : le poids de l'heure `t` est décidé à la clôture de la bougie `t` (là où
la prédiction est disponible) et porte sur le rendement de la bougie suivante, de close(t) à
close(t + 1 h). Aucune information postérieure à la décision n'est utilisée. Les frais sont
payés sur chaque variation de poids (rotation), à partir d'un portefeuille initial en cash.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from marketdata.universe import INTERVAL_DELTA
from strategies.positions import DEFAULT_FEE

HOURS_PER_YEAR = 24 * 365


@dataclass(frozen=True)
class BacktestResult:
    """Série horaire (`gross`, `fees`, `net`, `equity`, `turnover`) et métriques agrégées."""

    frame: pd.DataFrame
    metrics: dict[str, float]


def next_returns(candles: pd.DataFrame) -> pd.DataFrame:
    """Rendement simple de close(t) à close(t + 1 h), heure x symbole (NaN si une bougie manque)."""
    missing = [name for name in ('symbol', 'open_time', 'close') if name not in candles.columns]
    if missing:
        raise ValueError(f'colonnes absentes : {missing}')
    if candles.duplicated(['symbol', 'open_time']).any():
        raise ValueError('doublons sur (symbol, open_time)')
    closes = candles.pivot(index='open_time', columns='symbol', values='close').astype('float64')
    if (closes <= 0).any().any():
        raise ValueError('les prix de clôture doivent être strictement positifs')
    hours = pd.date_range(closes.index.min(), closes.index.max(), freq=INTERVAL_DELTA)
    closes = closes.reindex(hours)
    returns = closes.shift(-1) / closes - 1.0
    returns.index.name = 'open_time'
    returns.columns.name = 'symbol'
    return returns


def backtest(positions: pd.DataFrame, candles: pd.DataFrame, fee: float = DEFAULT_FEE) -> BacktestResult:
    """Rejoue les poids heure par heure sur les rendements réalisés des bougies.

    Un rendement inconnu (bougie suivante absente ou hors des données) ne produit ni gain ni
    perte ; les heures sans bougie suivante en fin de période sont exclues du résultat.
    """
    if not (np.isfinite(fee) and 0 <= fee < 1):
        raise ValueError('fee doit être dans [0, 1)')
    if positions.empty:
        raise ValueError('aucune position')
    if positions.isna().any().any() or np.isinf(positions.to_numpy(dtype='float64')).any():
        raise ValueError('les positions doivent être finies')
    returns = next_returns(candles)
    unknown = sorted(set(positions.columns) - set(returns.columns))
    if unknown:
        raise ValueError(f'symboles sans bougies : {unknown}')

    weights = positions.sort_index()
    realized = returns.reindex(index=weights.index, columns=weights.columns)
    known = realized.notna().any(axis=1)
    last_known = known[known].index.max() if known.any() else None
    if last_known is None:
        raise ValueError('aucun rendement réalisé disponible pour ces positions')
    weights = weights.loc[:last_known]
    realized = realized.loc[:last_known]

    gross = (weights * realized.fillna(0.0)).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = weights.iloc[0].abs().sum()
    fees = turnover * fee
    net = gross - fees
    frame = pd.DataFrame(
        {'gross': gross, 'fees': fees, 'net': net, 'equity': (1.0 + net).cumprod(), 'turnover': turnover}
    )
    return BacktestResult(frame=frame, metrics=summarize(frame))


def summarize(frame: pd.DataFrame) -> dict[str, float]:
    """Rendement total, Sharpe annualisé, perte maximale, rotation moyenne et frais cumulés."""
    net = frame['net']
    equity = frame['equity']
    deviation = float(net.std(ddof=1)) if len(net) > 1 else 0.0
    sharpe = float(net.mean() / deviation * np.sqrt(HOURS_PER_YEAR)) if deviation > 0 else 0.0
    drawdown = equity / equity.cummax().clip(lower=1.0) - 1.0
    return {
        'total_return': float(equity.iloc[-1] - 1.0),
        'sharpe': sharpe,
        'max_drawdown': float(drawdown.min()),
        'turnover': float(frame['turnover'].mean()),
        'fees': float(frame['fees'].sum()),
        'hours': float(len(frame)),
    }
