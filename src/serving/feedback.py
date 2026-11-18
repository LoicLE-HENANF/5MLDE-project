"""Retours sur les prédictions servies et santé glissante des modèles (niveau 3).

Une prédiction émise sur la bougie `t` (horizon `h`) devient vérifiable quand la
bougie `t + h` est clôturée : sa volatilité réalisée est alors exactement la cible
d'entraînement (`features.target_volatility`). La perte est la QLIKE, celle qui a
servi à choisir le modèle ; la persistance est évaluée de la même façon.

Le niveau 3 compare, sur une fenêtre glissante, la QLIKE de chaque version servie
à celle de la persistance, et mesure le PnL simulé des positions du champion.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from features import target_volatility
from strategies import DEFAULT_FEE, backtest, to_wide
from training.metrics import qlike

DEFAULT_WINDOW_HOURS = 7 * 24
INTERVAL = pd.Timedelta(hours=1)
PENDING_COLUMNS = ('prediction_id', 'symbol', 'open_time', 'horizon_hours', 'prediction', 'baseline')


def _frame(rows: Iterable[Mapping[str, Any]], columns: Iterable[str]) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows), columns=list(columns))
    if not frame.empty:
        frame['open_time'] = pd.to_datetime(frame['open_time'], utc=True)
    return frame


def pending_feedback(rows: Iterable[Mapping[str, Any]], now: pd.Timestamp) -> pd.DataFrame:
    """Prédictions sans retour dont l'horizon est clôturé à `now`."""
    frame = _frame(rows, PENDING_COLUMNS)
    if frame.empty:
        return frame
    horizon = pd.to_timedelta(frame['horizon_hours'].astype('int64'), unit='h')
    due = frame['open_time'] + INTERVAL + horizon <= now
    return frame.loc[due].reset_index(drop=True)


def _loss(predicted: float, realized: float) -> float | None:
    if not (np.isfinite(predicted) and predicted > 0 and np.isfinite(realized) and realized > 0):
        return None
    return qlike([predicted], [realized])


def realized_feedback(pending: pd.DataFrame, candles: pd.DataFrame, *, as_of: pd.Timestamp) -> pd.DataFrame:
    """Volatilité réalisée et pertes QLIKE (modèle et persistance) des prédictions dont l'horizon est clôturé.

    Les prédictions dont la volatilité réalisée reste indéfinie (bougie manquante
    dans la fenêtre) sont écartées : elles seront retentées au passage suivant.
    """
    columns = ['prediction_id', 'realized', 'loss', 'baseline_loss']
    if pending.empty:
        return pd.DataFrame(columns=columns)
    scored = []
    for hours, group in pending.groupby('horizon_hours'):
        realized = target_volatility(candles, hours=int(hours), as_of=as_of)
        merged = group.merge(realized, on=['symbol', 'open_time'], how='inner')
        merged = merged.loc[np.isfinite(merged['target_volatility'].to_numpy(dtype='float64'))]
        scored.append(merged)
    merged = pd.concat(scored, ignore_index=True)
    if merged.empty:
        return pd.DataFrame(columns=columns)
    merged['realized'] = merged['target_volatility'].astype('float64')
    merged['loss'] = [_loss(p, r) for p, r in zip(merged['prediction'], merged['realized'], strict=True)]
    merged['baseline_loss'] = [
        None if pd.isna(b) else _loss(float(b), r)
        for b, r in zip(merged['baseline'], merged['realized'], strict=True)
    ]
    merged = merged.loc[merged['loss'].notna()]
    return merged.loc[:, columns].reset_index(drop=True)


def _pnl(positions: pd.DataFrame, candles: pd.DataFrame, fee: float) -> float | None:
    """Rendement simulé des positions (frais inclus) ; None sans rendement réalisé."""
    known = positions.dropna(subset=['position'])
    if known.empty:
        return None
    wide = to_wide(known.rename(columns={'position': 'prediction'})).fillna(0.0)
    try:
        return float(backtest(wide, candles, fee=fee).metrics['total_return'])
    except ValueError:
        return None


def health_rows(
    scored: pd.DataFrame,
    candles: pd.DataFrame,
    *,
    day: date,
    window_hours: int = DEFAULT_WINDOW_HOURS,
    fee: float = DEFAULT_FEE,
) -> list[dict[str, Any]]:
    """Une ligne de santé par (version, mode) pour la fenêtre qui se termine à la fin de `day`.

    `scored` contient les prédictions avec retour : model_name, version, alias, mode,
    symbol, open_time, position, loss, baseline_loss. Une version est dégradée si sa
    QLIKE moyenne dépasse celle de la persistance sur la même fenêtre.
    """
    if window_hours <= 0:
        raise ValueError('window_hours doit être strictement positif')
    if scored.empty:
        return []
    end = pd.Timestamp(day, tz='UTC') + pd.Timedelta(days=1)
    start = end - pd.Timedelta(hours=window_hours)
    frame = scored.assign(open_time=pd.to_datetime(scored['open_time'], utc=True))
    frame = frame.loc[(frame['open_time'] >= start) & (frame['open_time'] < end)]
    rows = []
    for (name, version, mode), group in frame.groupby(['model_name', 'version', 'mode'], sort=True):
        losses = group['loss'].astype('float64')
        baseline = group['baseline_loss'].astype('float64').dropna()
        value = float(losses.mean())
        baseline_value = float(baseline.mean()) if not baseline.empty else None
        positions = group.loc[:, ['symbol', 'open_time', 'position']]
        equal = group.loc[:, ['symbol', 'open_time']].assign(position=1.0)
        equal['position'] = equal['position'] / equal.groupby('open_time')['symbol'].transform('count')
        has_positions = group['position'].notna().any()
        rows.append(
            {
                'model_name': name,
                'version': int(version),
                'mode': mode,
                'day': day,
                'window_hours': window_hours,
                'samples': len(group),
                'metric': 'qlike',
                'value': value,
                'baseline_value': baseline_value,
                'pnl': _pnl(positions, candles, fee) if has_positions else None,
                'baseline_pnl': _pnl(equal, candles, fee) if has_positions else None,
                'degraded': baseline_value is not None and value > baseline_value,
                'details': {
                    'aliases': sorted(group['alias'].unique().tolist()),
                    'symboles': sorted(group['symbol'].unique().tolist()),
                    'debut': start.isoformat(),
                    'fin': end.isoformat(),
                },
            }
        )
    return rows
