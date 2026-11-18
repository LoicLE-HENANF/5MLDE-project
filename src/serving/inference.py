"""Inférence du modèle `volatilite` sur la dernière bougie clôturée de chaque symbole.

Les features sont calculées par `features.build_features`, la même fonction qu'à
l'entraînement : aucune transformation propre au service. Une prédiction porte sur
la dernière bougie dont toutes les features sont définies ; la référence naïve
(persistance) et la position de la stratégie de ciblage sont calculées avec elle.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from features import FEATURE_COLUMNS, build_features
from strategies import scale_by_volatility, to_wide

# Fenêtre (en heures) sur laquelle est mesurée la volatilité cible de la stratégie.
TARGET_WINDOW_HOURS = 30 * 24

Predictor = Callable[[pd.DataFrame], np.ndarray]


@dataclass(frozen=True)
class Forecast:
    """Prédictions d'un modèle pour l'heure courante (une ligne par symbole)."""

    frame: pd.DataFrame  # symbol, open_time, prediction, baseline, position
    as_of: pd.Timestamp

    @property
    def open_time(self) -> pd.Timestamp:
        return pd.Timestamp(self.frame['open_time'].max())


def latest_features(candles: pd.DataFrame, *, as_of: datetime | pd.Timestamp | None = None) -> pd.DataFrame:
    """Dernière ligne de features complète par symbole (bougies clôturées à `as_of`)."""
    features = build_features(candles, as_of=as_of)
    complete = features.dropna(subset=list(FEATURE_COLUMNS))
    if complete.empty:
        raise ValueError('aucune ligne de features complète : historique trop court')
    latest = complete.sort_values(['symbol', 'open_time']).groupby('symbol', sort=True).tail(1)
    return latest.reset_index(drop=True)


def target_volatility_level(features: pd.DataFrame, latest: pd.Timestamp) -> float:
    """Volatilité cible : médiane de la volatilité réalisée 24 h sur les 30 derniers jours."""
    window = features['open_time'] > latest - pd.Timedelta(hours=TARGET_WINDOW_HOURS)
    values = features.loc[window, 'realized_vol_24h'].to_numpy(dtype='float64')
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        raise ValueError('volatilité cible indéfinie : aucune volatilité réalisée positive')
    return float(np.median(values))


def predict_latest(
    predict: Predictor,
    candles: pd.DataFrame,
    *,
    as_of: datetime | pd.Timestamp | None = None,
    with_positions: bool = True,
) -> Forecast:
    """Prédit la volatilité 24 h de chaque symbole à partir de sa dernière bougie clôturée.

    `predict` reçoit les colonnes `FEATURE_COLUMNS` et renvoie des volatilités (le
    `predict` du modèle pyfunc). Seuls les symboles dont la dernière bougie complète
    est la plus récente de l'univers sont servis : un symbole en retard n'est pas
    prédit sur des données périmées.
    """
    stamp = pd.Timestamp.now(tz='UTC') if as_of is None else pd.Timestamp(as_of)
    if stamp.tzinfo is None:
        raise ValueError('as_of doit inclure un fuseau horaire')
    features = build_features(candles, as_of=stamp)
    latest = latest_features(candles, as_of=stamp)
    newest = latest['open_time'].max()
    latest = latest.loc[latest['open_time'] == newest].reset_index(drop=True)

    predicted = np.asarray(predict(latest.loc[:, list(FEATURE_COLUMNS)]), dtype='float64')
    if predicted.shape != (len(latest),) or not np.isfinite(predicted).all() or (predicted <= 0).any():
        raise ValueError('le modèle doit renvoyer une volatilité finie et positive par symbole')

    frame = latest.loc[:, ['symbol', 'open_time']].copy()
    frame['prediction'] = predicted
    frame['baseline'] = latest['realized_vol_24h'].to_numpy(dtype='float64')
    frame['position'] = np.nan
    if with_positions:
        target = target_volatility_level(features, newest)
        keys = frame.loc[:, ['symbol', 'open_time']]
        raw = to_wide(keys.assign(prediction=1.0 / len(frame))).fillna(0.0)
        volatility = to_wide(keys.assign(prediction=predicted))
        positions = scale_by_volatility(raw, volatility, target=target)
        row = positions.loc[newest]
        frame['position'] = frame['symbol'].map(row).astype('float64')
    return Forecast(frame=frame, as_of=stamp)
