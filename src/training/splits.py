"""Découpage temporel, sans fuite entre entraînement, validation et test final."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

MIN_TRAIN_MONTHS = 12
VALIDATION_MONTHS = 1
FINAL_TEST_MONTHS = 3


def _duration(value: timedelta | pd.Timedelta, name: str) -> pd.Timedelta:
    if not isinstance(value, (timedelta, pd.Timedelta)):
        raise ValueError(f'{name} doit être une durée positive')
    result = pd.Timedelta(value)
    if result <= pd.Timedelta(0):
        raise ValueError(f'{name} doit être une durée positive')
    return result


def _month_start(moment: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=moment.year, month=moment.month, day=1, tz='UTC')


def _prepare_split_data(
    frame: pd.DataFrame,
    time_column: str,
    target_column: str | None,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    if frame.empty:
        raise ValueError('impossible de découper un jeu de données vide')
    if time_column not in frame.columns:
        raise ValueError(f'colonne temporelle absente : {time_column!r}')
    if target_column is not None and target_column not in frame.columns:
        raise ValueError(f'colonne cible absente : {target_column!r}')

    raw_times = frame[time_column]
    if not isinstance(raw_times.dtype, pd.DatetimeTZDtype):
        raise ValueError(f'{time_column} doit contenir des horodatages avec fuseau')
    if raw_times.isna().any():
        raise ValueError(f'{time_column} ne doit pas contenir de valeurs manquantes')
    times = pd.DatetimeIndex(raw_times.dt.tz_convert('UTC'))
    if 'symbol' in frame.columns and frame.duplicated(['symbol', time_column]).any():
        raise ValueError(f'doublons sur la clé (symbol, {time_column})')

    eligible = (
        frame[target_column].notna().to_numpy()
        if target_column is not None
        else np.ones(len(frame), dtype=bool)
    )
    return times, eligible


def walk_forward_splits(
    frame: pd.DataFrame,
    *,
    horizon: timedelta | pd.Timedelta = timedelta(hours=4),
    embargo: timedelta | pd.Timedelta | None = None,
    time_column: str = 'open_time',
    target_column: str | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Renvoie les indices positionnels `(train, validation)` des folds walk-forward V1.

    L'entraînement est expansif depuis le début des données, avec 12 mois initiaux. Chaque
    validation couvre un mois calendaire complet et avance d'un mois. Les trois derniers mois
    calendaires disponibles sont réservés au test final. L'embargo vaut par défaut l'horizon
    et ne peut pas lui être inférieur.

    Les lignes d'entraînement dont la cible atteint le début de validation sont purgées. Les
    cibles de validation qui atteindraient le début du test final sont exclues pour préserver
    l'étanchéité du holdout. Toutes les lignes d'un même horodatage restent dans le même fold.

    `target_column`, lorsqu'il est fourni, permet d'exclure les lignes dont la cible est NaN.
    Les indices renvoyés sont des positions à utiliser avec `iloc`.
    """
    times, eligible = _prepare_split_data(frame, time_column, target_column)
    horizon_delta = _duration(horizon, 'horizon')
    embargo_delta = horizon_delta if embargo is None else _duration(embargo, 'embargo')
    if embargo_delta < horizon_delta:
        raise ValueError('embargo doit être supérieur ou égal à horizon')

    first_month = _month_start(times.min())
    last_month = _month_start(times.max())
    holdout_start = last_month - pd.DateOffset(months=FINAL_TEST_MONTHS - 1)
    validation_start = first_month + pd.DateOffset(months=MIN_TRAIN_MONTHS)
    target_ends = times + horizon_delta
    folds: list[tuple[np.ndarray, np.ndarray]] = []

    while validation_start + pd.DateOffset(months=VALIDATION_MONTHS) <= holdout_start:
        validation_end = validation_start + pd.DateOffset(months=VALIDATION_MONTHS)
        train_mask = (
            (times < validation_start)
            & (times + horizon_delta < validation_start)
            & (times < validation_start - embargo_delta)
            & eligible
        )
        validation_mask = (
            (times >= validation_start) & (times < validation_end) & (target_ends < holdout_start) & eligible
        )
        train_indices = np.flatnonzero(train_mask)
        validation_indices = np.flatnonzero(validation_mask)
        if not len(train_indices) or not len(validation_indices):
            raise ValueError(
                f'fold vide pour la validation [{validation_start}, {validation_end}) : '
                'vérifier la couverture des données et les cibles disponibles'
            )
        folds.append((train_indices, validation_indices))
        validation_start = validation_end

    if not folds:
        raise ValueError(
            'pas assez de données pour produire un fold : 12 mois initiaux, '
            'un mois de validation et trois mois réservés au test final sont requis'
        )
    return folds


def holdout_split(
    frame: pd.DataFrame,
    *,
    horizon: timedelta | pd.Timedelta = timedelta(hours=4),
    embargo: timedelta | pd.Timedelta | None = None,
    time_column: str = 'open_time',
    target_column: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Renvoie les indices positionnels `(train, holdout)` du test final V1.

    Le holdout couvre les trois derniers mois calendaires disponibles. Les lignes
    d'entraînement dont la cible atteint son début sont purgées, et l'embargo
    d'entraînement vaut au moins l'horizon. `target_column`, lorsqu'il est fourni,
    exclut les lignes dont la cible est NaN des deux ensembles.
    """
    times, eligible = _prepare_split_data(frame, time_column, target_column)
    horizon_delta = _duration(horizon, 'horizon')
    embargo_delta = horizon_delta if embargo is None else _duration(embargo, 'embargo')
    if embargo_delta < horizon_delta:
        raise ValueError('embargo doit être supérieur ou égal à horizon')

    holdout_start = _month_start(times.max()) - pd.DateOffset(months=FINAL_TEST_MONTHS - 1)
    target_ends = times + horizon_delta
    train_mask = (
        (times < holdout_start)
        & (target_ends < holdout_start)
        & (times < holdout_start - embargo_delta)
        & eligible
    )
    holdout_mask = (times >= holdout_start) & eligible
    train_indices = np.flatnonzero(train_mask)
    holdout_indices = np.flatnonzero(holdout_mask)

    if not len(train_indices) or not len(holdout_indices):
        empty_sets = ', '.join(
            name
            for name, indices in (('train', train_indices), ('holdout', holdout_indices))
            if not len(indices)
        )
        raise ValueError(
            f'ensemble(s) {empty_sets} vide(s) après découpage : vérifier les données et les cibles'
        )

    return train_indices, holdout_indices
