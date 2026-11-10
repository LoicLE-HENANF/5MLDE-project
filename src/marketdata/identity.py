"""Identité d'un jeu de bougies : symboles, période, nombre de lignes et empreinte SHA-256.

L'empreinte porte sur le contenu canonique (lignes triées par symbole puis open_time,
colonnes dans l'ordre du schéma, flottants en représentation exacte) : elle ne dépend ni
de l'ordre de chargement ni du format de stockage (Parquet, PostgreSQL). Posée en tags
sur les runs et modèles MLflow, à côté de la provenance du code.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from marketdata.schema import COLUMNS, validate
from marketdata.universe import INTERVAL

TAG_PREFIX = 'data.'


@dataclass(frozen=True)
class DatasetIdentity:
    symbols: tuple[str, ...]
    interval: str
    start: datetime
    end: datetime
    rows: int
    sha256: str

    def tags(self) -> dict[str, str]:
        return {
            f'{TAG_PREFIX}symbols': ','.join(self.symbols),
            f'{TAG_PREFIX}interval': self.interval,
            f'{TAG_PREFIX}start': self.start.isoformat(),
            f'{TAG_PREFIX}end': self.end.isoformat(),
            f'{TAG_PREFIX}rows': str(self.rows),
            f'{TAG_PREFIX}sha256': self.sha256,
        }


def identify(frame: pd.DataFrame) -> DatasetIdentity:
    """Identité d'un jeu de bougies non vide et conforme au schéma."""
    if frame.empty:
        raise ValueError('jeu de bougies vide : aucune identité possible')
    validate(frame)
    return DatasetIdentity(
        symbols=tuple(sorted(frame['symbol'].unique())),
        interval=INTERVAL,
        start=frame['open_time'].min().to_pydatetime(),
        end=frame['open_time'].max().to_pydatetime(),
        rows=len(frame),
        sha256=fingerprint(frame),
    )


def fingerprint(frame: pd.DataFrame) -> str:
    canonical = frame.loc[:, list(COLUMNS)].sort_values(['symbol', 'open_time'])
    text = canonical.to_csv(index=False, date_format='%Y-%m-%dT%H:%M:%S.%fZ', lineterminator='\n')
    return hashlib.sha256(text.encode('utf-8')).hexdigest()
