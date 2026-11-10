"""Connecteur PostgreSQL : connexions et pool simulés, cohérence avec le script SQL d'init."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

import psycopg
import pytest
from psycopg.types.json import Jsonb
from psycopg_pool import PoolTimeout

from connectors.config import ROOT, PostgresConfig
from connectors.errors import ConnectorError
from connectors.postgres import (
    BATCH_SOURCES,
    BATCH_STATUSES,
    MODEL_EVENTS,
    RUN_STATUSES,
    TABLES,
    PostgresConnector,
)

INIT_SQL = ROOT / 'infra' / 'postgres' / 'init' / '002-tracing.sql'
T0 = datetime(2026, 1, 1, tzinfo=UTC)


class FakeConnection:
    """Connexion psycopg minimale : enregistre les requêtes, répond avec `rows`."""

    def __init__(self, rows=None):
        self.rows = list(rows or [])
        self.executed: list[tuple[object, object]] = []
        self.closed = False

    def execute(self, query, params=None):
        self.executed.append((query, params))
        return self

    def fetchone(self):
        return self.rows.pop(0) if self.rows else (1,)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.closed = True


class FakeConnect:
    def __init__(self, conn=None, error=None):
        self.conn = conn or FakeConnection()
        self.error = error
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, conninfo, **kwargs):
        self.calls.append((conninfo, kwargs))
        if self.error:
            raise self.error
        return self.conn


@pytest.fixture
def config(env) -> PostgresConfig:
    return PostgresConfig.from_env(env)


def make(config, conn=None, error=None):
    fake = FakeConnect(conn, error)
    return PostgresConnector(config, connect=fake), fake


# -- connexion ---------------------------------------------------------------------------------
def test_conninfo(config):
    info = psycopg.conninfo.conninfo_to_dict(PostgresConnector(config).conninfo)
    assert info == {
        'host': 'db.test',
        'port': '5432',
        'dbname': 'mlops',
        'user': 'api_writer',
        'password': 'secret-api',
        'connect_timeout': '5',
        'application_name': '5mlde-project',
    }


def test_check_ok(config):
    connector, fake = make(
        config, FakeConnection(rows=[('tracing.pipeline_runs',), ('tracing.model_events',)])
    )
    connector.check()
    assert fake.calls[0][1] == {'autocommit': True}
    assert [params for _, params in fake.conn.executed] == [(f'tracing.{t}',) for t in TABLES]


def test_check_missing_tables(config):
    connector, _ = make(config, FakeConnection(rows=[(None,), ('tracing.model_events',)]))
    with pytest.raises(ConnectorError, match=r'tables absentes : pipeline_runs.*down -v'):
        connector.check()


def test_check_server_down(config):
    error = psycopg.OperationalError('connection refused\nIs the server running?')
    connector, _ = make(config, error=error)
    with pytest.raises(ConnectorError) as info:
        connector.check()
    assert 'api_writer@db.test:5432/mlops' in str(info.value)
    assert 'connection refused' in str(info.value) and 'Is the server' not in str(info.value)


def test_password_never_in_error(config):
    connector, _ = make(config, error=psycopg.OperationalError('password authentication failed'))
    with pytest.raises(ConnectorError) as info:
        connector.check()
    assert 'secret-api' not in str(info.value)


def test_constructor_does_no_io(config):
    PostgresConnector(config)  # hôte inexistant : aucune erreur à la construction


# -- pool --------------------------------------------------------------------------------------
class FakePool:
    def __init__(self, conninfo, **kwargs):
        self.kwargs = kwargs
        self.conn = FakeConnection()
        self.opened = self.closed = False
        self.fail_open = False

    def open(self, wait, timeout):
        if self.fail_open:
            raise PoolTimeout('pool initialization incomplete')
        self.opened = True

    def close(self):
        self.closed = True

    def connection(self):
        return self.conn


def test_pool_lifecycle(config):
    pools = []

    def factory(*args, **kwargs):
        pools.append(FakePool(*args, **kwargs))
        return pools[-1]

    connector = PostgresConnector(config, pool_factory=factory, connect=FakeConnect(error=AssertionError()))
    with connector as db:
        db.insert('pipeline_runs', {'pipeline': 'p'})
        assert connector.open() is connector and len(pools) == 1  # open idempotent
    pool = pools[0]
    assert pool.kwargs['min_size'] == 1 and pool.kwargs['max_size'] == 5 and pool.kwargs['open'] is False
    assert pool.opened and pool.closed and len(pool.conn.executed) == 1


def test_pool_unavailable(config):
    def factory(*args, **kwargs):
        pool = FakePool(*args, **kwargs)
        pool.fail_open = True
        return pool

    with pytest.raises(ConnectorError, match='pool non disponible'):
        PostgresConnector(config, pool_factory=factory).open()


# -- écritures ---------------------------------------------------------------------------------
def test_insert_builds_safe_sql(config):
    connector, fake = make(config, FakeConnection(rows=[(12,)]))
    assert connector.insert('pipeline_runs', {'pipeline': 'p', 'details': {'n': 1}}) == 12
    query, params = fake.conn.executed[0]
    text = query.as_string(None)
    assert text == (
        'INSERT INTO "tracing"."pipeline_runs" ("pipeline", "details") VALUES (%s, %s) RETURNING id'
    )
    assert params[0] == 'p' and isinstance(params[1], Jsonb)


@pytest.mark.parametrize(
    'table, row',
    [
        ('users', {'pipeline': 'p'}),
        ('pipeline_runs', {'pipeline': 'p', 'id': 1}),
        ('pipeline_runs', {'pipeline); DROP TABLE x; --': 'p'}),
        ('pipeline_runs', {}),
    ],
)
def test_insert_rejects_unknown_tables_and_columns(config, table, row):
    connector, fake = make(config)
    with pytest.raises(ValueError):
        connector.insert(table, row)
    assert fake.calls == []


def test_record_pipeline_run(config):
    connector, fake = make(config)
    connector.record_pipeline_run(
        pipeline='train',
        status='success',
        started_at=T0,
        ended_at=T0 + timedelta(minutes=2),
        mlflow_run_id='r1',
    )
    _, params = fake.conn.executed[0]
    assert params[:5] == ['train', 'success', T0, T0 + timedelta(minutes=2), None]


@pytest.mark.parametrize(
    'kwargs',
    [
        {'status': 'ok'},
        {'ended_at': T0 - timedelta(seconds=1)},
    ],
)
def test_record_pipeline_run_validation(config, kwargs):
    connector, fake = make(config)
    args = {'pipeline': 'p', 'status': 'success', 'started_at': T0, 'ended_at': T0, **kwargs}
    with pytest.raises(ValueError):
        connector.record_pipeline_run(**args)
    assert fake.calls == []


def test_record_model_event(config):
    connector, fake = make(config)
    connector.record_model_event(
        model_name='m', version='3', event='promoted', alias='champion', model_uri='models:/m/3'
    )
    _, params = fake.conn.executed[0]
    assert params[:4] == ['m', 3, 'promoted', 'champion']
    with pytest.raises(ValueError):
        connector.record_model_event(model_name='m', version=1, event='deleted', model_uri='u')


def test_record_ingestion_batch(config):
    connector, fake = make(config)
    connector.record_ingestion_batch(
        source='archive',
        symbol='BTCUSDT',
        period_start=T0,
        period_end=T0 + timedelta(days=31),
        status='rejected',
        rows=744,
        issues=[{'rule': 'doublons', 'blocking': True, 'count': 2}],
    )
    _, params = fake.conn.executed[0]
    assert params[:6] == ['archive', 'BTCUSDT', T0, T0 + timedelta(days=31), 'rejected', 744]
    issues = params[8]
    assert isinstance(issues, Jsonb) and issues.obj == [{'rule': 'doublons', 'blocking': True, 'count': 2}]


@pytest.mark.parametrize(
    'kwargs',
    [
        {'source': 'ftp'},
        {'status': 'ok'},
        {'period_end': T0},
        {'rows': -1},
    ],
)
def test_record_ingestion_batch_validation(config, kwargs):
    connector, fake = make(config)
    args = {
        'source': 'rest',
        'symbol': 'BTCUSDT',
        'period_start': T0,
        'period_end': T0 + timedelta(hours=1),
        'status': 'accepted',
        'rows': 1,
        **kwargs,
    }
    with pytest.raises(ValueError):
        connector.record_ingestion_batch(**args)
    assert fake.calls == []


def test_connection_lost_during_write(config):
    class Broken(FakeConnection):
        def execute(self, query, params=None):
            raise psycopg.OperationalError('server closed the connection unexpectedly')

    connector, _ = make(config, Broken())
    with pytest.raises(ConnectorError, match='connexion perdue'):
        connector.insert('pipeline_runs', {'pipeline': 'p'})


# -- cohérence avec le schéma SQL --------------------------------------------------------------
def _sql_columns(table: str) -> tuple[set[str], str]:
    body = re.search(
        rf'CREATE TABLE tracing\.{table} \((.*?)\n\);', INIT_SQL.read_text(encoding='utf-8'), re.S
    )
    assert body, f'table {table} absente de {INIT_SQL.name}'
    columns = {m.group(1) for m in re.finditer(r'^\s{4}([a-z][a-z0-9_]*)\s', body.group(1), re.M)}
    return columns - {'id', 'recorded_at'}, body.group(1)


@pytest.mark.parametrize('table', TABLES)
def test_tables_match_init_sql(table):
    columns, _ = _sql_columns(table)
    assert columns == TABLES[table]


def test_allowed_values_match_init_sql():
    _, runs = _sql_columns('pipeline_runs')
    _, events = _sql_columns('model_events')
    assert all(f"'{s}'" in runs for s in RUN_STATUSES)
    assert all(f"'{e}'" in events for e in MODEL_EVENTS)
    _, batches = _sql_columns('ingestion_batches')
    assert all(f"'{s}'" in batches for s in BATCH_SOURCES + BATCH_STATUSES)
