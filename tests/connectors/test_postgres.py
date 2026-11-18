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
    LOAD_SCENARIOS,
    MODEL_EVENTS,
    PREDICTION_ALIASES,
    PREDICTION_MODES,
    RUN_STATUSES,
    TABLES,
    PostgresConnector,
)

INIT_DIR = ROOT / 'infra' / 'postgres' / 'init'
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
    scripts = '\n'.join(path.read_text(encoding='utf-8') for path in sorted(INIT_DIR.glob('*.sql')))
    body = re.search(rf'CREATE TABLE tracing\.{table} \((.*?)\n\);', scripts, re.S)
    assert body, f'table {table} absente de {INIT_DIR}'
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
    _, predictions = _sql_columns('predictions')
    assert all(f"'{v}'" in predictions for v in PREDICTION_ALIASES + PREDICTION_MODES)
    _, health = _sql_columns('model_health')
    assert all(f"'{m}'" in health for m in PREDICTION_MODES)
    _, load = _sql_columns('load_tests')
    assert all(f"'{s}'" in load for s in LOAD_SCENARIOS)


# -- service : prédictions, retours, santé -----------------------------------------------------
NOW = datetime(2026, 11, 18, 12, tzinfo=UTC)


def test_insert_idempotent_ignores_duplicates(config):
    connector, fake = make(config, FakeConnection(rows=[None]))
    row = {'prediction_id': 1, 'realized': 0.5}
    assert connector.insert('feedback', row, idempotent=True) is None
    query, _ = fake.conn.executed[0]
    assert 'ON CONFLICT DO NOTHING RETURNING id' in query.as_string(None)


def test_record_prediction(config):
    connector, fake = make(config, FakeConnection(rows=[(5,)]))
    assert (
        connector.record_prediction(
            model_name='volatilite',
            version='3',
            alias='champion',
            symbol='BTCUSDT',
            open_time=NOW,
            horizon_hours=24,
            prediction=0.01,
            baseline=float('nan'),
            position=0.2,
            mode='live',
        )
        == 5
    )
    query, params = fake.conn.executed[0]
    assert 'ON CONFLICT DO NOTHING' in query.as_string(None)
    assert 3 in params and None in params  # version en entier, NaN -> NULL


@pytest.mark.parametrize(
    'kwargs',
    [
        {'alias': 'precedent'},
        {'mode': 'backtest'},
        {'alias': 'challenger', 'position': 0.1},
        {'horizon_hours': 0},
        {'prediction': float('inf')},
    ],
)
def test_record_prediction_validation(config, kwargs):
    connector, fake = make(config)
    base = {
        'model_name': 'volatilite',
        'version': 1,
        'alias': 'champion',
        'symbol': 'BTCUSDT',
        'open_time': NOW,
        'horizon_hours': 24,
        'prediction': 0.01,
        'mode': 'live',
    }
    with pytest.raises(ValueError):
        connector.record_prediction(**{**base, **kwargs})
    assert fake.calls == []


def test_record_feedback_and_health(config):
    connector, fake = make(config, FakeConnection(rows=[(1,)]))
    connector.record_feedback(prediction_id=7, realized=0.02, loss=0.1)
    connector.record_model_health(
        model_name='volatilite',
        version=1,
        mode='replay',
        day=NOW.date(),
        window_hours=168,
        samples=10,
        metric='qlike',
        value=0.3,
        baseline_value=0.4,
        degraded=False,
    )
    assert len(fake.conn.executed) == 2
    with pytest.raises(ValueError):
        connector.record_feedback(prediction_id=7, realized=float('nan'))
    with pytest.raises(ValueError):
        connector.record_model_health(
            model_name='m',
            version=1,
            mode='x',
            day=NOW.date(),
            window_hours=1,
            samples=0,
            metric='qlike',
            degraded=False,
        )


LOAD_TEST = {
    'campaign': 'c1',
    'scenario': 'pic',
    'workers': 2,
    'cpus': 1.0,
    'users': 222,
    'duration_s': 60.0,
    'requests': 600,
    'failures': 3,
    'slo_ok': True,
}


def test_record_load_test_computes_throughput(config):
    connector, fake = make(config, FakeConnection(rows=[(4,)]))
    assert connector.record_load_test(**LOAD_TEST, p95_ms=float('nan')) == 4
    query, params = fake.conn.executed[0]
    assert 'load_tests' in query.as_string(None)
    assert 10.0 in params  # 600 requêtes / 60 s
    assert None in params  # p95 NaN -> NULL


@pytest.mark.parametrize(
    'override',
    [
        {'scenario': 'autre'},
        {'workers': 0},
        {'cpus': 0},
        {'duration_s': 0},
        {'users': -1},
        {'failures': 601},
    ],
)
def test_record_load_test_validation(config, override):
    connector, fake = make(config)
    with pytest.raises(ValueError):
        connector.record_load_test(**{**LOAD_TEST, **override})
    assert fake.calls == []
