"""Mise en forme des réponses de l'API pour l'affichage (tableaux pandas, libellés en français)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

EVENT_LABELS = {'registered': 'enregistré', 'promoted': 'promu', 'retired': 'retiré'}


def _times(frame: pd.DataFrame, *columns: str) -> pd.DataFrame:
    for column in columns:
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame


def predictions_frame(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Une ligne par symbole : bougie, prévision, référence naïve, écart et position."""
    columns = ['Symbole', 'Bougie', 'Horizon (h)', 'Prévision', 'Référence', 'Écart', 'Position', 'Version']
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame = _times(frame, 'open_time')
    out = pd.DataFrame(
        {
            'Symbole': frame['symbol'],
            'Bougie': frame['open_time'],
            'Horizon (h)': frame['horizon_hours'],
            'Prévision': frame['prediction'],
            'Référence': frame['baseline'],
            'Écart': frame['prediction'] / frame['baseline'] - 1.0,
            'Position': frame['position'],
            'Version': frame['version'],
        }
    )
    return out.sort_values('Symbole', ignore_index=True)


def health_frame(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Santé quotidienne (niveau 3), du plus ancien au plus récent."""
    columns = [
        'Jour',
        'Version',
        'Mode',
        'Échantillons',
        'Métrique',
        'Valeur',
        'Référence',
        'PnL',
        'PnL équipondéré',
        'Dégradé',
    ]
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame(
        {
            'Jour': pd.to_datetime(frame['day']).dt.date,
            'Version': frame['version'],
            'Mode': frame['mode'],
            'Échantillons': frame['samples'],
            'Métrique': frame['metric'],
            'Valeur': frame['value'],
            'Référence': frame['baseline_value'],
            'PnL': frame['pnl'],
            'PnL équipondéré': frame['baseline_pnl'],
            'Dégradé': frame['degraded'],
        }
    )
    return out.sort_values(['Jour', 'Version'], ignore_index=True)


def _details(details: Any) -> str:
    if not details:
        return ''
    if isinstance(details, Mapping):
        parts = []
        for key, value in details.items():
            if isinstance(value, list):
                value = ' ; '.join(map(str, value))
            parts.append(key if value is True else f'{key} : {value}')
        return ' | '.join(parts)
    return str(details)


def events_frame(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Événements du registre, du plus récent au plus ancien, détails à plat."""
    columns = ['Date', 'Version', 'Événement', 'Alias', 'Détails']
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame = _times(frame, 'recorded_at')
    return pd.DataFrame(
        {
            'Date': frame['recorded_at'],
            'Version': frame['version'],
            'Événement': frame['event'].map(lambda e: EVENT_LABELS.get(e, e)),
            'Alias': frame['alias'].fillna(''),
            'Détails': frame['details'].map(_details),
        }
    )


def system_frame(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Historique marche / arrêt, du plus récent au plus ancien."""
    columns = ['Date', 'État', 'Par', 'Raison']
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame = _times(frame, 'recorded_at')
    return pd.DataFrame(
        {
            'Date': frame['recorded_at'],
            'État': frame['active'].map({True: 'marche', False: 'arrêt'}),
            'Par': frame['changed_by'],
            'Raison': frame['reason'],
        }
    )
