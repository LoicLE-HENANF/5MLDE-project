"""Stockage : Parquet sur disque temporaire, market.candles avec connexion simulée."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from connectors.config import ROOT
from ingestion import store
from ingestion.store import CandleTable, ParquetStore
from marketdata import identity, sources
from marketdata.schema import COLUMNS
from tests.conftest import HOUR_MS, START_MS, raw_rows

MARKET_SQL = ROOT / 'infra' / 'postgres' / 'init' / '003-market.sql'
JAN = datetime(2025, 1, 1, tzinfo=UTC)


def candles(n=3, symbol='BTCUSDT', start_ms=START_MS):
    return sources.normalize(raw_rows(n, start_ms=start_ms), symbol)


# -- Parquet -----------------------------------------------------------------------------------
def test_month_path_layout(tmp_path):
    path = ParquetStore(tmp_path).month_path('BTCUSDT', JAN)
    assert path == tmp_path / 'candles' / 'symbol=BTCUSDT' / 'year=2025' / '2025-01.parquet'
    with pytest.raises(ValueError):
        ParquetStore(tmp_path).month_path('DOGEUSDT', JAN)


def test_roundtrip_keeps_fingerprint(tmp_path):
    parquet = ParquetStore(tmp_path)
    frame = candles(24)
    parquet.write_month(frame, 'BTCUSDT', JAN)
    assert parquet.has_month('BTCUSDT', JAN)
    assert not list(tmp_path.rglob('*.tmp'))
    back = parquet.read()
    assert identity.fingerprint(back) == identity.fingerprint(frame)
    pd.testing.assert_frame_equal(back, frame)


@pytest.mark.parametrize(
    ('frame', 'match'),
    [
        (candles(2, symbol='ETHUSDT'), 'symboles'),
        (candles(2, start_ms=START_MS - HOUR_MS), 'hors du mois'),
        (candles(0), 'vide'),
    ],
)
def test_write_month_rejects_wrong_batch(tmp_path, frame, match):
    with pytest.raises(ValueError, match=match):
        ParquetStore(tmp_path).write_month(frame, 'BTCUSDT', JAN)


def test_read_filters_symbols_and_period(tmp_path):
    parquet = ParquetStore(tmp_path)
    parquet.write_month(candles(48), 'BTCUSDT', JAN)
    feb_ms = int(datetime(2025, 2, 1, tzinfo=UTC).timestamp() * 1000)
    parquet.write_month(candles(5, start_ms=feb_ms), 'BTCUSDT', datetime(2025, 2, 1, tzinfo=UTC))
    parquet.write_month(candles(3, symbol='ETHUSDT'), 'ETHUSDT', JAN)
    assert len(parquet.read()) == 56
    assert set(parquet.read(['ETHUSDT'])['symbol']) == {'ETHUSDT'}
    window = parquet.read(['BTCUSDT'], start=JAN + timedelta(hours=10), end=JAN + timedelta(hours=20))
    assert len(window) == 10 and window['open_time'].min() == JAN + timedelta(hours=10)
    assert parquet.read(start=datetime(2025, 3, 1, tzinfo=UTC)).empty


def test_read_empty_store_is_valid(tmp_path):
    frame = ParquetStore(tmp_path).read()
    assert frame.empty and list(frame.columns) == list(COLUMNS)


def test_data_dir(tmp_path):
    assert store.data_dir({}) == ROOT / 'data'
    assert store.data_dir({'DATA_DIR': 'autre'}) == ROOT / 'autre'
    assert store.data_dir({'DATA_DIR': str(tmp_path)}) == tmp_path


# -- market.candles ----------------------------------------------------------------------------
class FakeCursor:
    def __init__(self, conn):
        self.conn = conn

    def executemany(self, query, rows):
        self.conn.many.append((query, list(rows)))

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        pass


class FakeConn:
    def __init__(self, result=None):
        self.result = result
        self.executed = []
        self.many = []
        self.rowcount = 4

    def execute(self, query, params=None):
        self.executed.append((query, params))
        return self

    def fetchone(self):
        return self.result

    def fetchall(self):
        return self.result

    def cursor(self):
        return FakeCursor(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        pass


class FakeDb:
    def __init__(self, conn):
        self.conn = conn

    def connection(self):
        return self.conn


def test_upsert_converts_to_python_types():
    conn = FakeConn()
    frame = candles(2)
    assert CandleTable(FakeDb(conn)).upsert(frame) == 2
    query, rows = conn.many[0]
    text = repr(query)
    assert 'ON CONFLICT' in text and "'market'" in text and "'candles'" in text
    first = rows[0]
    assert len(first) == len(COLUMNS)
    assert first[0] == 'BTCUSDT' and isinstance(first[1], datetime) and first[1].tzinfo is not None
    assert type(first[COLUMNS.index('trades')]) is int
    assert type(first[COLUMNS.index('close')]) is float


def test_upsert_empty_does_nothing():
    conn = FakeConn()
    assert CandleTable(FakeDb(conn)).upsert(candles(0)) == 0
    assert conn.many == []


def test_latest_and_prune():
    conn = FakeConn(result=(JAN,))
    table = CandleTable(FakeDb(conn))
    assert table.latest_open_time('BTCUSDT') == JAN
    assert table.prune('BTCUSDT', JAN) == 4
    assert [params for _, params in conn.executed] == [('BTCUSDT',), ('BTCUSDT', JAN)]


def test_read_rebuilds_schema():
    frame = candles(3)
    rows = [tuple(r) for r in frame.itertuples(index=False)]
    conn = FakeConn(result=rows)
    back = CandleTable(FakeDb(conn)).read(['BTCUSDT'], start=JAN)
    assert identity.fingerprint(back) == identity.fingerprint(frame)
    _, params = conn.executed[0]
    assert params == [['BTCUSDT'], JAN]


def test_read_no_rows():
    assert CandleTable(FakeDb(FakeConn(result=[]))).read().empty


# -- cohérence avec le schéma SQL --------------------------------------------------------------
def test_market_sql_matches_schema():
    body = re.search(
        r'CREATE TABLE market\.candles \((.*?)\n\);', MARKET_SQL.read_text(encoding='utf-8'), re.S
    )
    assert body
    columns = [m.group(1) for m in re.finditer(r'^\s{4}([a-z_]+)\s', body.group(1), re.M)]
    assert columns == [*COLUMNS, 'ingested_at']
    assert 'PRIMARY KEY (symbol, open_time)' in body.group(1)
