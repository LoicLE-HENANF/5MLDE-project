"""Ingestion de bout en bout : vraies sources Binance -> Parquet + PostgreSQL -> traces -> Grafana.

Les périodes testées sont en 2022, hors de la fenêtre récente de market.candles : le test
n'écrase aucune donnée réelle. Le stockage Parquet se fait dans un dossier temporaire. Le
nettoyage passe par le compte administrateur (api_writer n'a pas DELETE sur les traces).
"""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime, timedelta

import psycopg
import pytest

from connectors.config import PostgresConfig
from connectors.postgres import PostgresConnector
from ingestion.download import BinanceClient
from ingestion.pipeline import BatchProcessor
from ingestion.store import CandleTable, ParquetStore
from marketdata import identity
from tests.integration.conftest import grafana_sql

pytestmark = pytest.mark.integration

SYMBOL = 'BTCUSDT'
MONTH = datetime(2022, 1, 1, tzinfo=UTC)
NEXT_MONTH = datetime(2022, 2, 1, tzinfo=UTC)
RECENT_START = NEXT_MONTH
RECENT_NOW = datetime(2022, 2, 1, 6, 30, tzinfo=UTC)  # 6 bougies closes : 00h à 05h
FLOW_RUN_ID = str(uuid.uuid4())
SHA256 = re.compile(r'^[0-9a-f]{64}$')


def _admin(env: dict[str, str], config: PostgresConfig) -> psycopg.Connection:
    return psycopg.connect(
        host=config.host,
        port=config.port,
        dbname=config.dbname,
        user='postgres',
        password=env['POSTGRES_PASSWORD'],
        connect_timeout=5,
    )


def _window_rows(conn: psycopg.Connection) -> int:
    return conn.execute(
        'SELECT count(*) FROM market.candles WHERE symbol = %s AND open_time >= %s AND open_time < %s',
        (SYMBOL, RECENT_START, RECENT_NOW),
    ).fetchone()[0]


@pytest.fixture(scope='module')
def db(env):
    config = PostgresConfig.from_env(env)
    with _admin(env, config) as conn:
        if _window_rows(conn):
            pytest.skip('market.candles contient déjà des bougies sur la période de test')
    with PostgresConnector(config) as connector:
        yield connector
    with _admin(env, config) as conn:
        conn.execute(
            'DELETE FROM market.candles WHERE symbol = %s AND open_time >= %s AND open_time < %s',
            (SYMBOL, RECENT_START, RECENT_NOW),
        )
        conn.execute('DELETE FROM tracing.ingestion_batches WHERE prefect_flow_run_id = %s', (FLOW_RUN_ID,))


@pytest.fixture(scope='module')
def parquet(tmp_path_factory) -> ParquetStore:
    return ParquetStore(tmp_path_factory.mktemp('data'))


@pytest.fixture(scope='module')
def processor(db, parquet):
    with BinanceClient() as client:
        yield BatchProcessor(
            client,
            record=db.record_ingestion_batch,
            parquet=parquet,
            table=CandleTable(db),
            flow_run_id=FLOW_RUN_ID,
        )


def _batches(db: PostgresConnector, source: str) -> list[tuple]:
    with db.connection() as conn:
        return conn.execute(
            'SELECT status, rows, source_sha256, data_sha256 FROM tracing.ingestion_batches '
            'WHERE prefect_flow_run_id = %s AND source = %s ORDER BY id',
            (FLOW_RUN_ID, source),
        ).fetchall()


# -- archive mensuelle -> Parquet ----------------------------------------------------------------
def test_archive_month_downloaded_verified_and_stored(processor, parquet, db):
    result = processor.archive_month(SYMBOL, MONTH)
    assert result.accepted, str(result)
    assert 700 < result.rows <= 31 * 24

    frame = parquet.read([SYMBOL], MONTH, NEXT_MONTH)
    assert len(frame) == result.rows
    assert frame['open_time'].min() >= MONTH and frame['open_time'].max() < NEXT_MONTH
    assert identity.fingerprint(frame) == result.data_sha256

    [(status, rows, source_sha256, data_sha256)] = _batches(db, 'archive')
    assert (status, rows, data_sha256) == ('accepted', result.rows, result.data_sha256)
    assert SHA256.match(source_sha256)  # empreinte de l'archive vérifiée contre le .CHECKSUM


# -- API REST -> market.candles (upsert idempotent) ----------------------------------------------
def test_recent_candles_upserted_idempotently(processor, db):
    first = processor.recent(SYMBOL, RECENT_START, RECENT_NOW)
    assert first.accepted, str(first)
    assert first.rows == 6  # la bougie de 06h n'est pas close à 06h30

    table = CandleTable(db)
    frame = table.read([SYMBOL], RECENT_START, RECENT_NOW)
    assert len(frame) == 6
    assert identity.fingerprint(frame) == first.data_sha256

    second = processor.recent(SYMBOL, RECENT_START, RECENT_NOW)
    assert second.data_sha256 == first.data_sha256
    assert len(table.read([SYMBOL], RECENT_START, RECENT_NOW)) == 6

    batches = _batches(db, 'rest')
    assert [(b[0], b[1]) for b in batches] == [('accepted', 6), ('accepted', 6)]
    assert all(SHA256.match(b[2]) for b in batches)


# -- lecture par Grafana (rôle grafana_reader) ---------------------------------------------------
def test_grafana_reads_candles_and_batches(processor, grafana):
    start, end = RECENT_START.isoformat(), (RECENT_START + timedelta(hours=6)).isoformat()
    candles = grafana_sql(
        grafana,
        'projet-traces',
        f"SELECT count(*) FROM market.candles WHERE symbol = '{SYMBOL}' "
        f"AND open_time >= '{start}' AND open_time < '{end}'",
    )
    assert candles[0][0] == 6
    batches = grafana_sql(
        grafana,
        'projet-traces',
        f"SELECT source FROM tracing.ingestion_batches WHERE prefect_flow_run_id = '{FLOW_RUN_ID}'",
    )
    assert sorted(batches[0]) == ['archive', 'rest', 'rest']
