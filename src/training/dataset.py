"""Préparation des entrées brutes du modèle de rendement."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from features import FEATURE_COLUMNS, IDENTIFIER_COLUMNS, build_features, target_return, target_volatility
from marketdata.identity import DatasetIdentity, identify
from marketdata.quality import QualityReport, check


@dataclass(frozen=True)
class ReturnDatasetInputs:
    features: pd.DataFrame
    targets: pd.DataFrame
    identity: DatasetIdentity
    quality_report: QualityReport
    as_of: pd.Timestamp


@dataclass(frozen=True)
class ReturnTrainingDataset:
    frame: pd.DataFrame
    identity: DatasetIdentity
    quality_report: QualityReport
    as_of: pd.Timestamp


def prepare_return_inputs(
    candles: pd.DataFrame,
    *,
    as_of: datetime | pd.Timestamp | None = None,
) -> ReturnDatasetInputs:
    """Valide et identifie les bougies, puis calcule features et cible au même instant.

    Les features et les cibles restent dans deux DataFrames distincts. Leur jointure et
    le filtrage des lignes exploitables sont la prochaine étape de préparation.
    """
    quality_report = check(candles)
    if not quality_report.ok:
        raise ValueError(f'jeu de bougies invalide : {quality_report.summary()}')
    identity = identify(candles)

    cutoff = pd.Timestamp.now(tz='UTC') if as_of is None else pd.Timestamp(as_of)
    if cutoff.tzinfo is None:
        raise ValueError('as_of doit inclure un fuseau horaire')
    cutoff = cutoff.tz_convert('UTC')

    features = build_features(candles, as_of=cutoff)
    targets = target_return(candles, hours=4, as_of=cutoff)
    return ReturnDatasetInputs(
        features=features,
        targets=targets,
        identity=identity,
        quality_report=quality_report,
        as_of=cutoff,
    )


def prepare_return_dataset(
    candles: pd.DataFrame,
    *,
    as_of: datetime | pd.Timestamp | None = None,
) -> ReturnTrainingDataset:
    """Assemble le jeu exploitable avec identifiants, features et cible rendement."""
    inputs = prepare_return_inputs(candles, as_of=as_of)
    dataset = inputs.features.merge(
        inputs.targets,
        on=list(IDENTIFIER_COLUMNS),
        how='inner',
        validate='one_to_one',
    )
    complete = dataset.dropna(subset=[*FEATURE_COLUMNS, 'target_return'])
    frame = complete.loc[:, [*IDENTIFIER_COLUMNS, *FEATURE_COLUMNS, 'target_return']].reset_index(drop=True)
    if frame.empty:
        raise ValueError(
            'aucune ligne exploitable après jointure : vérifier la couverture historique '
            'et la disponibilité des features et cibles'
        )
    return ReturnTrainingDataset(
        frame=frame,
        identity=inputs.identity,
        quality_report=inputs.quality_report,
        as_of=inputs.as_of,
    )


@dataclass(frozen=True)
class VolatilityTrainingDataset:
    frame: pd.DataFrame
    identity: DatasetIdentity
    quality_report: QualityReport
    as_of: pd.Timestamp
    hours: int


def prepare_volatility_dataset(
    candles: pd.DataFrame,
    *,
    hours: int = 24,
    as_of: datetime | pd.Timestamp | None = None,
) -> VolatilityTrainingDataset:
    """Assemble le jeu exploitable avec identifiants, features et volatilité réalisée future.

    Mêmes contrôles et même instant de référence que le jeu rendement : bougies validées et
    identifiées, features et cible calculées à `as_of`, lignes incomplètes écartées. La cible
    `target_volatility` est strictement positive (une volatilité nulle n'a pas de log).
    """
    quality_report = check(candles)
    if not quality_report.ok:
        raise ValueError(f'jeu de bougies invalide : {quality_report.summary()}')
    identity = identify(candles)
    cutoff = _cutoff(as_of)

    features = build_features(candles, as_of=cutoff)
    targets = target_volatility(candles, hours=hours, as_of=cutoff)
    dataset = features.merge(targets, on=list(IDENTIFIER_COLUMNS), how='inner', validate='one_to_one')
    complete = dataset.dropna(subset=[*FEATURE_COLUMNS, 'target_volatility'])
    complete = complete[complete['target_volatility'] > 0]
    frame = complete.loc[:, [*IDENTIFIER_COLUMNS, *FEATURE_COLUMNS, 'target_volatility']].reset_index(
        drop=True
    )
    if frame.empty:
        raise ValueError(
            'aucune ligne exploitable après jointure : vérifier la couverture historique '
            'et la disponibilité des features et cibles'
        )
    return VolatilityTrainingDataset(
        frame=frame,
        identity=identity,
        quality_report=quality_report,
        as_of=cutoff,
        hours=hours,
    )


def _cutoff(as_of: datetime | pd.Timestamp | None) -> pd.Timestamp:
    cutoff = pd.Timestamp.now(tz='UTC') if as_of is None else pd.Timestamp(as_of)
    if cutoff.tzinfo is None:
        raise ValueError('as_of doit inclure un fuseau horaire')
    return cutoff.tz_convert('UTC')
