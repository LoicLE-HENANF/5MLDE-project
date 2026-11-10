"""Règles qualité d'un lot de bougies normalisées.

Bloquantes (le lot est rejeté) : schéma, doublons, horodatages non alignés sur l'heure,
incohérence OHLC, volume acheteur supérieur au volume total, variation horaire extrême.
Signalées (le lot passe, l'anomalie est enregistrée) : bougies manquantes, variation
horaire forte. Les trous deviennent bloquants au-delà de `MAX_GAP_RATIO` : Binance a eu
quelques interruptions courtes, un historique troué au-delà indique un téléchargement raté.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pandera.errors

from marketdata import schema
from marketdata.universe import INTERVAL_DELTA

MAX_GAP_RATIO = 0.01
# Valeur absolue du log-rendement entre deux bougies consécutives.
RETURN_WARN = 0.20
RETURN_BLOCK = 0.50
MAX_EXAMPLES = 5


@dataclass(frozen=True)
class Issue:
    rule: str
    blocking: bool
    count: int
    examples: tuple[str, ...] = ()

    def __str__(self) -> str:
        level = 'BLOQUANT' if self.blocking else 'signalé'
        examples = f' (ex. {", ".join(self.examples)})' if self.examples else ''
        return f'[{level}] {self.rule} : {self.count}{examples}'


@dataclass(frozen=True)
class QualityReport:
    rows: int
    issues: tuple[Issue, ...]

    @property
    def ok(self) -> bool:
        return not any(issue.blocking for issue in self.issues)

    def summary(self) -> str:
        if not self.issues:
            return f'{self.rows} bougies, aucune anomalie'
        return '\n'.join([f'{self.rows} bougies', *(str(issue) for issue in self.issues)])


def check(frame: pd.DataFrame) -> QualityReport:
    """Applique toutes les règles. Si le schéma échoue, les autres règles ne sont pas évaluées."""
    try:
        schema.validate(frame)
    except pandera.errors.SchemaErrors as exc:
        cases = exc.failure_cases
        examples = tuple(
            f'{row.column}: {row.check}' for row in cases.drop_duplicates(['column', 'check']).itertuples()
        )
        issue = Issue('schéma', True, len(cases), examples[:MAX_EXAMPLES])
        return QualityReport(len(frame), (issue,))

    issues = [
        _rows_issue('doublons (symbol, open_time)', True, frame, _duplicates(frame)),
        _rows_issue("open_time non aligné sur l'heure", True, frame, _misaligned(frame)),
        _rows_issue('incohérence OHLC', True, frame, _ohlc_incoherent(frame)),
        _rows_issue('volume acheteur > volume total', True, frame, _taker_above_total(frame)),
        *_gap_issues(frame),
        *_return_issues(frame),
    ]
    return QualityReport(len(frame), tuple(issue for issue in issues if issue is not None))


def _duplicates(frame: pd.DataFrame) -> pd.Series:
    return frame.duplicated(['symbol', 'open_time'], keep=False)


def _misaligned(frame: pd.DataFrame) -> pd.Series:
    return frame['open_time'] != frame['open_time'].dt.floor(INTERVAL_DELTA)


def _ohlc_incoherent(frame: pd.DataFrame) -> pd.Series:
    body_high = frame[['open', 'close']].max(axis=1)
    body_low = frame[['open', 'close']].min(axis=1)
    return (frame['high'] < body_high) | (frame['low'] > body_low)


def _taker_above_total(frame: pd.DataFrame) -> pd.Series:
    return (frame['taker_buy_base_volume'] > frame['volume']) | (
        frame['taker_buy_quote_volume'] > frame['quote_volume']
    )


def _gap_issues(frame: pd.DataFrame) -> list[Issue]:
    """Bougies absentes entre la première et la dernière de chaque symbole."""
    missing: list[str] = []
    expected = 0
    for symbol, times in frame.groupby('symbol')['open_time']:
        hours = pd.date_range(times.min(), times.max(), freq=INTERVAL_DELTA)
        expected += len(hours)
        missing += [f'{symbol} {t:%Y-%m-%d %H:%M}' for t in hours.difference(times)]
    if not missing:
        return []
    blocking = len(missing) > MAX_GAP_RATIO * expected
    return [Issue('bougies manquantes', blocking, len(missing), tuple(missing[:MAX_EXAMPLES]))]


def _return_issues(frame: pd.DataFrame) -> list[Issue]:
    """Log-rendements entre bougies consécutives (une heure d'écart exactement)."""
    ordered = frame.sort_values(['symbol', 'open_time'])
    grouped = ordered.groupby('symbol')
    consecutive = grouped['open_time'].diff() == INTERVAL_DELTA
    log_return = np.log(ordered['close'] / grouped['close'].shift()).abs().where(consecutive)
    issues = []
    for rule, blocking, mask in (
        (f'variation horaire > {RETURN_BLOCK:.0%}', True, log_return > RETURN_BLOCK),
        (
            f'variation horaire > {RETURN_WARN:.0%}',
            False,
            (log_return > RETURN_WARN) & (log_return <= RETURN_BLOCK),
        ),
    ):
        issue = _rows_issue(rule, blocking, ordered, mask)
        if issue is not None:
            issues.append(issue)
    return issues


def _rows_issue(rule: str, blocking: bool, frame: pd.DataFrame, mask: pd.Series) -> Issue | None:
    count = int(mask.sum())
    if not count:
        return None
    rows = frame.loc[mask].head(MAX_EXAMPLES)
    examples = tuple(f'{r.symbol} {r.open_time:%Y-%m-%d %H:%M}' for r in rows.itertuples())
    return Issue(rule, blocking, count, examples)
