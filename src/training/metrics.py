"""Métriques statistiques pour l'évaluation des modèles."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def information_coefficient(
    frame: pd.DataFrame,
    *,
    prediction_column: str = 'prediction',
    target_column: str = 'target_return',
    time_column: str = 'open_time',
    symbol_column: str = 'symbol',
) -> pd.Series:
    """Renvoie l'IC de Spearman transversal pour chaque horodatage.

    Les lignes sans prédiction ou cible sont ignorées. Un IC vaut NaN si moins
    de trois symboles sont exploitables ou si les prédictions ou les cibles sont
    constantes à cet horodatage.
    """
    columns = (symbol_column, time_column, prediction_column, target_column)
    if len(set(columns)) != len(columns):
        raise ValueError('les colonnes symbole, temps, prédiction et cible doivent être distinctes')
    if frame.empty:
        raise ValueError('impossible de calculer IC sur un jeu de données vide')
    missing_columns = [column for column in columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f'colonnes absentes : {missing_columns}')

    raw_times = frame[time_column]
    if not isinstance(raw_times.dtype, pd.DatetimeTZDtype):
        raise ValueError(f'{time_column} doit contenir des horodatages avec fuseau')
    if raw_times.isna().any():
        raise ValueError(f'{time_column} ne doit pas contenir de valeurs manquantes')
    if frame[symbol_column].isna().any():
        raise ValueError(f'{symbol_column} ne doit pas contenir de valeurs manquantes')
    if frame.duplicated([symbol_column, time_column]).any():
        raise ValueError(f'doublons sur la clé ({symbol_column}, {time_column})')

    data = frame.loc[:, list(columns)].copy()
    data[time_column] = raw_times.dt.tz_convert('UTC')
    for column in (prediction_column, target_column):
        if not pd.api.types.is_numeric_dtype(data[column].dtype) or pd.api.types.is_bool_dtype(
            data[column].dtype
        ):
            raise ValueError(f'{column} doit contenir des valeurs numériques')
        values = data[column].to_numpy(dtype='float64', na_value=np.nan)
        if np.isinf(values).any():
            raise ValueError(f'{column} ne doit pas contenir de valeurs infinies')
        data[column] = values

    timestamps: list[pd.Timestamp] = []
    coefficients: list[float] = []
    for timestamp, group in data.groupby(time_column, sort=True):
        paired = group.dropna(subset=[prediction_column, target_column])
        coefficient = np.nan
        if (
            len(paired) >= 3
            and paired[prediction_column].nunique() > 1
            and paired[target_column].nunique() > 1
        ):
            ranks = paired.loc[:, [prediction_column, target_column]].rank(method='average')
            coefficient = float(ranks[prediction_column].corr(ranks[target_column]))
        timestamps.append(timestamp)
        coefficients.append(coefficient)

    index = pd.DatetimeIndex(timestamps, name=time_column)
    return pd.Series(coefficients, index=index, name='ic', dtype='float64')


def icir(information_coefficients: pd.Series) -> float:
    """Renvoie la moyenne des IC divisée par leur écart-type d'échantillon."""
    if not isinstance(information_coefficients, pd.Series):
        raise ValueError('information_coefficients doit être une série pandas')
    if not pd.api.types.is_numeric_dtype(information_coefficients.dtype) or pd.api.types.is_bool_dtype(
        information_coefficients.dtype
    ):
        raise ValueError('les coefficients IC doivent être numériques')

    values = information_coefficients.to_numpy(dtype='float64', na_value=np.nan)
    if np.isinf(values).any():
        raise ValueError('les coefficients IC ne peuvent pas être infinis')
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float('nan')
    standard_deviation = np.std(values, ddof=1)
    if standard_deviation == 0:
        return float('nan')
    return float(np.mean(values) / standard_deviation)


def qlike(predicted_volatility: ArrayLike, realized_volatility: ArrayLike) -> float:
    """Perte QLIKE moyenne entre volatilités prédites et réalisées (même horizon).

    Calculée sur les variances : `mean(r² / p² - log(r² / p²) - 1)`. Nulle pour une
    prévision parfaite, elle pénalise davantage la sous-estimation du risque que sa
    surestimation et reste robuste au bruit de la volatilité réalisée.
    """
    predicted, realized = _positive_pair(predicted_volatility, realized_volatility)
    ratio = (realized / predicted) ** 2
    return float(np.mean(ratio - np.log(ratio) - 1))


def log_volatility_mae(predicted_volatility: ArrayLike, realized_volatility: ArrayLike) -> float:
    """Erreur absolue moyenne entre les logarithmes des volatilités prédites et réalisées."""
    predicted, realized = _positive_pair(predicted_volatility, realized_volatility)
    return float(np.mean(np.abs(np.log(predicted) - np.log(realized))))


def _positive_pair(predicted: ArrayLike, realized: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    predicted_values = np.asarray(predicted, dtype='float64')
    realized_values = np.asarray(realized, dtype='float64')
    if predicted_values.ndim != 1 or predicted_values.shape != realized_values.shape:
        raise ValueError('prédictions et réalisations doivent être deux vecteurs de même longueur')
    if predicted_values.size == 0:
        raise ValueError('impossible de calculer une perte sur un jeu de données vide')
    for name, values in (('prédites', predicted_values), ('réalisées', realized_values)):
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f'les volatilités {name} doivent être finies et strictement positives')
    return predicted_values, realized_values
