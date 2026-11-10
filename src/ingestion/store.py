"""Stockage des bougies normalisées.

- Historique complet : un fichier Parquet par symbole et par mois, issu d'une archive vérifiée,
  `<DATA_DIR>/candles/symbol=<SYMBOL>/year=<YYYY>/<YYYY>-<MM>.parquet`. Un mois présent est complet
  et n'est jamais réécrit (écriture atomique : fichier temporaire puis renommage).
- Bougies récentes : table `market.candles` (PostgreSQL, rôle api_writer), mise à jour par upsert
  sur (symbol, open_time), purgée au-delà de la fenêtre de rétention.

Les deux lectures renvoient des bougies conformes au schéma : même jeu, même empreinte.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
from psycopg import sql

from connectors.config import ROOT, load_env
from connectors.postgres import PostgresConnector
from marketdata.schema import COLUMNS, TIMESTAMP, validate
from marketdata.sources import PRICE_FIELDS, VOLUME_FIELDS
from marketdata.universe import check_symbol

DEFAULT_DATA_DIR = ROOT / 'data'
CANDLES_TABLE = ('market', 'candles')


def data_dir(env: Mapping[str, str] | None = None) -> Path:
    """Racine des données locales : DATA_DIR (relatif à la racine du projet) ou ./data."""
    value = str((load_env() if env is None else env).get('DATA_DIR', '')).strip()
    if not value:
        return DEFAULT_DATA_DIR
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def month_start(moment: datetime | date) -> datetime:
    return datetime(moment.year, moment.month, 1, tzinfo=UTC)


def _empty() -> pd.DataFrame:
    return validate(_typed(pd.DataFrame({name: [] for name in COLUMNS})))


def _typed(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.loc[:, list(COLUMNS)].copy()
    frame['symbol'] = frame['symbol'].astype('str')
    for name in ('open_time', 'close_time'):
        frame[name] = pd.to_datetime(frame[name], utc=True).astype(TIMESTAMP)
    for name in PRICE_FIELDS + VOLUME_FIELDS:
        frame[name] = frame[name].astype('float64')
    frame['trades'] = frame['trades'].astype('int64')
    return frame


def _python(value: object) -> object:
    """Valeur pandas/numpy -> type Python accepté par psycopg."""
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    return value.item() if hasattr(value, 'item') else value


def _sorted(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(['symbol', 'open_time'], ignore_index=True)


class ParquetStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def month_path(self, symbol: str, month: date | datetime) -> Path:
        symbol = check_symbol(symbol)
        return self.root / 'candles' / f'symbol={symbol}' / f'year={month.year}' / f'{month:%Y-%m}.parquet'

    def has_month(self, symbol: str, month: date | datetime) -> bool:
        return self.month_path(symbol, month).is_file()

    def write_month(self, frame: pd.DataFrame, symbol: str, month: date | datetime) -> Path:
        """Écrit le mois (bougies validées, toutes de ce symbole et de ce mois)."""
        validate(frame)
        start = month_start(month)
        end = month_start(start + pd.DateOffset(months=1))
        if frame.empty or set(frame['symbol']) != {symbol}:
            raise ValueError(f'{symbol} {start:%Y-%m} : lot vide ou symboles mélangés')
        if frame['open_time'].min() < start or frame['open_time'].max() >= end:
            raise ValueError(f'{symbol} {start:%Y-%m} : bougies hors du mois')
        path = self.month_path(symbol, start)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.parquet.tmp')
        _sorted(frame).to_parquet(tmp, index=False)
        os.replace(tmp, path)
        return path

    def read(
        self,
        symbols: Iterable[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Bougies dont l'ouverture est dans [start, end), tous les mois stockés par défaut."""
        names = [check_symbol(s) for s in symbols] if symbols is not None else None
        files = sorted((self.root / 'candles').glob('symbol=*/year=*/*.parquet'))
        frames = []
        for path in files:
            symbol = path.parent.parent.name.removeprefix('symbol=')
            if names is not None and symbol not in names:
                continue
            first = month_start(datetime.strptime(path.stem, '%Y-%m'))
            last = month_start(first + pd.DateOffset(months=1))
            if (end is not None and first >= end) or (start is not None and last <= start):
                continue
            frames.append(pd.read_parquet(path))
        if not frames:
            return _empty()
        frame = _typed(pd.concat(frames, ignore_index=True))
        if start is not None:
            frame = frame[frame['open_time'] >= start]
        if end is not None:
            frame = frame[frame['open_time'] < end]
        return validate(_sorted(frame))


class CandleTable:
    """Bougies récentes dans market.candles (colonnes identiques au schéma marketdata)."""

    def __init__(self, db: PostgresConnector) -> None:
        self.db = db
        self._table = sql.Identifier(*CANDLES_TABLE)

    def upsert(self, frame: pd.DataFrame) -> int:
        """Insère ou met à jour les bougies (clé symbol, open_time) ; renvoie le nombre de lignes."""
        validate(frame)
        if frame.empty:
            return 0
        columns = list(COLUMNS)
        updates = [c for c in columns if c not in ('symbol', 'open_time')]
        query = sql.SQL(
            'INSERT INTO {table} ({columns}) VALUES ({values}) '
            'ON CONFLICT (symbol, open_time) DO UPDATE SET {updates}, ingested_at = now()'
        ).format(
            table=self._table,
            columns=sql.SQL(', ').join(map(sql.Identifier, columns)),
            values=sql.SQL(', ').join(sql.Placeholder() * len(columns)),
            updates=sql.SQL(', ').join(
                sql.SQL('{0} = EXCLUDED.{0}').format(sql.Identifier(c)) for c in updates
            ),
        )
        rows = [tuple(_python(v) for v in r) for r in frame.loc[:, columns].itertuples(index=False)]
        with self.db.connection() as conn, conn.cursor() as cur:
            cur.executemany(query, rows)
        return len(rows)

    def latest_open_time(self, symbol: str) -> datetime | None:
        query = sql.SQL('SELECT max(open_time) FROM {} WHERE symbol = %s').format(self._table)
        with self.db.connection() as conn:
            return conn.execute(query, (check_symbol(symbol),)).fetchone()[0]

    def prune(self, symbol: str, before: datetime) -> int:
        """Supprime les bougies du symbole ouvertes avant `before` ; renvoie le nombre supprimé."""
        query = sql.SQL('DELETE FROM {} WHERE symbol = %s AND open_time < %s').format(self._table)
        with self.db.connection() as conn:
            return conn.execute(query, (check_symbol(symbol), before)).rowcount

    def read(
        self,
        symbols: Iterable[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Bougies dont l'ouverture est dans [start, end)."""
        conditions, params = [], []
        if symbols is not None:
            conditions.append(sql.SQL('symbol = ANY(%s)'))
            params.append([check_symbol(s) for s in symbols])
        if start is not None:
            conditions.append(sql.SQL('open_time >= %s'))
            params.append(start)
        if end is not None:
            conditions.append(sql.SQL('open_time < %s'))
            params.append(end)
        where = sql.SQL(' WHERE ') + sql.SQL(' AND ').join(conditions) if conditions else sql.SQL('')
        query = sql.SQL('SELECT {columns} FROM {table}{where} ORDER BY symbol, open_time').format(
            columns=sql.SQL(', ').join(map(sql.Identifier, COLUMNS)), table=self._table, where=where
        )
        with self.db.connection() as conn:
            rows = conn.execute(query, params).fetchall()
        if not rows:
            return _empty()
        return validate(_typed(pd.DataFrame(rows, columns=list(COLUMNS))))
