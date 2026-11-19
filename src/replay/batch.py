"""Calculs en lot du rejeu : mois simulés, features et prévisions horaires sans fuite.

En live, chaque heure, le flow de prévision recalcule les features des bougies clôturées et sert
la dernière ligne complète par symbole. Le rejeu calcule les features d'un mois en une fois :
chaque ligne ne dépend que des bougies clôturées à son heure (`features.build_features` ne regarde
jamais vers le futur), le résultat est donc celui qu'aurait servi le live heure par heure.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from features import FEATURE_COLUMNS, build_features
from serving import target_volatility_level
from serving.inference import TARGET_WINDOW_HOURS
from strategies import long_short, scale_by_volatility, staggered, to_wide

# Historique lu avant le mois pour les fenêtres glissantes des features et la cible de volatilité.
FEATURE_MARGIN = pd.Timedelta(days=60)
HOUR = pd.Timedelta(hours=1)

Predictor = Callable[[pd.DataFrame], np.ndarray]


def month_starts(start: str, months: int) -> list[pd.Timestamp]:
    """Débuts de mois UTC : `start` au format AAAA-MM, `months` mois consécutifs."""
    if months < 1:
        raise ValueError(f'months doit être positif : {months!r}')
    first = pd.Timestamp(f'{start}-01', tz='UTC')
    return [first + pd.DateOffset(months=i) for i in range(months)]


def closed_before(candles: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Bougies clôturées avant `as_of` (la bougie ouverte à t clôture à t + 1 h)."""
    return candles.loc[candles['open_time'] + HOUR <= as_of]


def month_features(candles: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Features complètes de [start - marge, end), calculées avec les bougies clôturées avant `end`."""
    window = candles.loc[candles['open_time'] >= start - FEATURE_MARGIN]
    features = build_features(closed_before(window, end), as_of=end)
    return features.dropna(subset=list(FEATURE_COLUMNS)).reset_index(drop=True)


def predict_rows(predict: Predictor, features: pd.DataFrame) -> pd.Series:
    """Prévisions d'une version pour toutes les lignes (index aligné sur `features`)."""
    if features.empty:
        return pd.Series(dtype='float64')
    predicted = np.asarray(predict(features.loc[:, list(FEATURE_COLUMNS)]), dtype='float64')
    if predicted.shape != (len(features),):
        raise ValueError('le modèle doit renvoyer une prévision par ligne')
    return pd.Series(predicted, index=features.index)


def hours_of(features: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return features.loc[(features['open_time'] >= start) & (features['open_time'] < end)]


def volatility_rows(
    features: pd.DataFrame,
    predicted: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    with_positions: bool,
) -> pd.DataFrame:
    """Lignes de prévision volatilité de [start, end), positions du ciblage comme en live.

    Chaque heure : équipondéré entre les symboles servis, multiplié par cible / volatilité prédite,
    la cible étant la médiane de la volatilité réalisée 24 h des 30 derniers jours.
    """
    rows = hours_of(features, start, end)
    frame = rows.loc[:, ['symbol', 'open_time']].assign(
        prediction=predicted.loc[rows.index].to_numpy(),
        baseline=rows['realized_vol_24h'].to_numpy(dtype='float64'),
        position=np.nan,
    )
    if not with_positions or frame.empty:
        return frame.reset_index(drop=True)
    positions = []
    for stamp, group in frame.groupby('open_time', sort=True):
        history = features.loc[
            (features['open_time'] <= stamp)
            & (features['open_time'] > stamp - pd.Timedelta(hours=TARGET_WINDOW_HOURS))
        ]
        target = target_volatility_level(history, stamp)
        keys = group.loc[:, ['symbol', 'open_time']]
        raw = to_wide(keys.assign(prediction=1.0 / len(group))).fillna(0.0)
        volatility = to_wide(keys.assign(prediction=group['prediction'].to_numpy()))
        row = scale_by_volatility(raw, volatility, target=target).loc[stamp]
        positions.append(group['symbol'].map(row).astype('float64'))
    frame['position'] = pd.concat(positions).loc[frame.index]
    return frame.reset_index(drop=True)


def return_rows(
    features: pd.DataFrame,
    predicted: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    horizon_hours: int,
    with_positions: bool,
) -> pd.DataFrame:
    """Lignes de prévision rendement de [start, end), portefeuille long/short en tranches.

    Comme en live, la position d'une heure est la moyenne des poids long/short des `horizon_hours`
    dernières heures, recalculés avec la version servie.
    """
    rows = hours_of(features, start, end)
    frame = rows.loc[:, ['symbol', 'open_time']].assign(
        prediction=predicted.loc[rows.index].to_numpy(),
        baseline=rows['return_4h'].to_numpy(dtype='float64'),
        position=np.nan,
    )
    if not with_positions or frame.empty:
        return frame.reset_index(drop=True)
    recent = hours_of(features, start - pd.Timedelta(hours=horizon_hours - 1), end)
    keys = recent.loc[:, ['symbol', 'open_time']].assign(prediction=predicted.loc[recent.index].to_numpy())
    weights = staggered(long_short(keys), horizon_hours)
    stacked = weights.stack()
    index = pd.MultiIndex.from_frame(frame.loc[:, ['open_time', 'symbol']])
    frame['position'] = stacked.reindex(index).to_numpy(dtype='float64')
    return frame.reset_index(drop=True)
