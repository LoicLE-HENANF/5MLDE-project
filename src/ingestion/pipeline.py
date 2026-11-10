"""Traitement d'un lot : téléchargement, normalisation, qualité, écriture, trace.

Un lot = un symbole et une période (un mois d'archive, ou les heures récentes via l'API REST).
Chaque tentative laisse une ligne dans tracing.ingestion_batches, acceptée ou rejetée : un lot
rejeté n'écrit rien et garde la raison (anomalie bloquante, format, checksum, téléchargement).
Seules les bougies clôturées sont gardées : la bougie en cours change jusqu'à sa clôture.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from ingestion.download import BinanceClient, DownloadError
from ingestion.store import CandleTable, ParquetStore, month_start
from marketdata import identity, quality, sources
from marketdata.quality import Issue
from marketdata.universe import HISTORY_START, INTERVAL_DELTA

Recorder = Callable[..., int]


@dataclass(frozen=True)
class BatchResult:
    source: str
    symbol: str
    period_start: datetime
    period_end: datetime
    status: str  # 'accepted', 'rejected' ou 'unavailable' (archive pas encore publiée, non tracé)
    rows: int = 0
    data_sha256: str | None = None
    issues: tuple[Issue, ...] = field(default_factory=tuple)

    @property
    def accepted(self) -> bool:
        return self.status == 'accepted'

    def __str__(self) -> str:
        period = f'{self.period_start:%Y-%m-%d %H:%M} -> {self.period_end:%Y-%m-%d %H:%M}'
        lines = [f'{self.symbol} {self.source} {period} : {self.status}, {self.rows} bougies']
        return '\n  '.join([*lines, *(str(i) for i in self.issues)])


def _issue_rows(issues: tuple[Issue, ...]) -> list[dict[str, Any]]:
    return [
        {'rule': i.rule, 'blocking': i.blocking, 'count': i.count, 'examples': list(i.examples)}
        for i in issues
    ]


def closed_only(frame: pd.DataFrame, now: datetime) -> pd.DataFrame:
    """Bougies clôturées à `now` (close_time strictement passé)."""
    return frame[frame['close_time'] < now].reset_index(drop=True)


def months_between(start: datetime, end: datetime) -> list[datetime]:
    """Débuts de mois m tels que [m, m + 1 mois) est inclus dans [start, end)."""
    month, months = month_start(start), []
    if month < start:
        month = month_start(month + pd.DateOffset(months=1))
    while (nxt := month_start(month + pd.DateOffset(months=1))) <= end:
        months.append(month)
        month = nxt
    return months


class BatchProcessor:
    def __init__(
        self,
        client: BinanceClient,
        *,
        record: Recorder,
        parquet: ParquetStore | None = None,
        table: CandleTable | None = None,
        flow_run_id: str | None = None,
    ) -> None:
        self.client = client
        self.record = record
        self.parquet = parquet
        self.table = table
        self.flow_run_id = flow_run_id

    def _finish(
        self,
        source: str,
        symbol: str,
        start: datetime,
        end: datetime,
        *,
        frame: pd.DataFrame | None = None,
        issues: tuple[Issue, ...] = (),
        source_sha256: str | None = None,
        write: Callable[[pd.DataFrame], object] | None = None,
    ) -> BatchResult:
        accepted = frame is not None and not any(i.blocking for i in issues)
        rows = 0 if frame is None else len(frame)
        digest = identity.fingerprint(frame) if accepted and rows else None
        if accepted and rows and write is not None:
            write(frame)
        status = 'accepted' if accepted else 'rejected'
        self.record(
            source=source,
            symbol=symbol,
            period_start=start,
            period_end=end,
            status=status,
            rows=rows,
            source_sha256=source_sha256,
            data_sha256=digest,
            issues=_issue_rows(issues),
            prefect_flow_run_id=self.flow_run_id,
        )
        return BatchResult(source, symbol, start, end, status, rows, digest, issues)

    def _fetch_failure(self, exc: Exception) -> tuple[Issue, ...]:
        rule = 'téléchargement' if isinstance(exc, DownloadError) else 'format source'
        return (Issue(rule, True, 1, (str(exc)[:300],)),)

    def archive_month(self, symbol: str, month: datetime) -> BatchResult:
        """Mois complet depuis l'archive vérifiée, écrit en Parquet s'il est accepté."""
        if self.parquet is None:
            raise RuntimeError('ParquetStore requis pour les archives')
        start = month_start(month)
        end = month_start(start + pd.DateOffset(months=1))
        try:
            fetched = self.client.archive_rows(symbol, start)
            if fetched is None:
                return BatchResult('archive', symbol, start, end, 'unavailable')
            raw, checksum = fetched
            frame = sources.normalize(raw, symbol)
        except (DownloadError, sources.SourceFormatError) as exc:
            return self._finish('archive', symbol, start, end, issues=self._fetch_failure(exc))
        issues = quality.check(frame).issues
        outside = int(((frame['open_time'] < start) | (frame['open_time'] >= end)).sum())
        if outside:
            issues += (Issue('bougies hors du mois', True, outside),)
        if frame.empty:
            issues += (Issue('archive vide', True, 1),)
        return self._finish(
            'archive',
            symbol,
            start,
            end,
            frame=frame,
            issues=issues,
            source_sha256=checksum,
            write=lambda f: self.parquet.write_month(f, symbol, start),
        )

    def recent(self, symbol: str, start: datetime, now: datetime) -> BatchResult:
        """Bougies clôturées ouvertes dans [start, now), upsert dans market.candles."""
        if self.table is None:
            raise RuntimeError('CandleTable requise pour les bougies récentes')
        try:
            raw = self.client.recent_rows(symbol, start, now)
            frame = closed_only(sources.normalize(raw, symbol), now)
        except (DownloadError, sources.SourceFormatError) as exc:
            return self._finish('rest', symbol, start, now, issues=self._fetch_failure(exc))
        payload = '\n'.join(','.join(map(str, row)) for row in raw).encode()
        return self._finish(
            'rest',
            symbol,
            start,
            now,
            frame=frame,
            issues=quality.check(frame).issues if len(frame) else (),
            source_sha256=hashlib.sha256(payload).hexdigest(),
            write=self.table.upsert,
        )


def missing_months(parquet: ParquetStore, symbol: str, now: datetime) -> list[datetime]:
    """Mois complets depuis HISTORY_START absents du stockage Parquet."""
    return [m for m in months_between(HISTORY_START, month_start(now)) if not parquet.has_month(symbol, m)]


def recent_start(table: CandleTable, symbol: str, now: datetime, retention: timedelta) -> datetime:
    """Reprise après la dernière bougie stockée, sans remonter au-delà de la rétention."""
    floor = (now - retention).replace(minute=0, second=0, microsecond=0)
    latest = table.latest_open_time(symbol)
    if latest is None:
        return floor
    return max(floor, latest.astimezone(UTC) + INTERVAL_DELTA)
